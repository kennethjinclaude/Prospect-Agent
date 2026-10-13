"""Decide whether each filtered company is AI-native, using its own website as evidence."""

import functools
import json
import re
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import edgar

# Load ANTHROPIC_API_KEY from the .env file that sits next to this script.
load_dotenv(Path(__file__).parent / ".env")

MODEL = "claude-haiku-5-5"
RESULTS_FILE = Path(__file__).parent / "data" / "qualified.csv"

# How many new companies to qualify per run. Lower it (e.g. to 10) to test cheaply.
LIMIT = 100

# A "no" below this confidence usually means Claude had too little to go on,
# so it's recorded as insufficient_info instead of ruling the company out.
MIN_CONFIDENCE_FOR_NO = 50

# Max characters of website text sent to Claude (roughly 1,000 tokens).
MAX_WEBSITE_CHARS = 4000

# Pages with less readable text than this are usually built with JavaScript
# and look nearly empty to us, so we look for other sources of text.
MIN_WEBSITE_CHARS = 300

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)

# Published prices, used only to estimate what a run cost.
INPUT_PRICE_PER_TOKEN = 0.10 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 0.50 / 1_000_000
PRICE_PER_SEARCH = 10 / 1_000

# Running totals of what this run used, so we can estimate the cost.
usage_totals = {"input_tokens": 0, "output_tokens": 0, "web_searches": 0}

@functools.cache
def get_client():
    """Create the Claude API client the first time it's needed, then reuse it.

    Waiting until it's needed means code that only reads saved results (like
    the app's demo mode) works without an API key.
    """
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 2}


def _record_usage(response):
    """Add one API response's token and search counts to the running totals."""
    usage_totals["input_tokens"] += response.usage.input_tokens
    usage_totals["output_tokens"] += response.usage.output_tokens
    if response.usage.server_tool_use:
        usage_totals["web_searches"] += response.usage.server_tool_use.web_search_requests


def _executives(related_persons, max_count=2):
    """Pick up to `max_count` executive officer names from edgar's related_persons text.

    The text looks like "JANE DOE (Executive Officer, Director); JOHN ROE (Director)".
    """
    if not isinstance(related_persons, str):  # empty cells come back as NaN
        return []
    names = []
    for person in related_persons.split("; "):
        name, _, roles = person.partition(" (")
        if "Executive Officer" in roles:
            names.append(name)
    return names[:max_count]


def find_website(company):
    """Ask Claude to web-search for the company's official homepage.

    Returns the URL, or None if Claude isn't confident it found the right company.
    """
    executives = _executives(company["related_persons"])
    details = (
        f"Company name: {company['company_name']}\n"
        f"Address on filing: {company['city']}, {company['state']}\n"
        f"Executives: {', '.join(executives) if executives else 'not listed'}"
    )
    response = get_client().messages.create(
        model=MODEL,
        max_tokens=4000,
        output_config={"effort": "low"},  # a simple lookup doesn't need deep thinking
        tools=[WEB_SEARCH_TOOL],
        messages=[{
            "role": "user",
            "content": (
                "Find the official homepage of this US company. It recently filed an "
                "SEC Form D to raise money, so it may be a young startup.\n\n"
                f"{details}\n\n"
                "Many startups list a Delaware (DE) address only because that's where "
                "they're legally registered, often through a registered-agent office. "
                "If the address is in Delaware, don't rely on it; rely on the company "
                "name and executives' names instead (for example, an executive's "
                "LinkedIn profile or a news article naming them). Otherwise, the city "
                "and executives' names can confirm you have the right company.\n\n"
                "Reply with only the homepage URL and nothing else. "
                "If you aren't confident you found this exact company (for example, "
                "only a similarly named company, or just a directory listing), "
                "reply with only the word: unknown"
            ),
        }],
    )
    _record_usage(response)

    reply = " ".join(block.text for block in response.content if block.type == "text")
    match = re.search(r"https?://\S+", reply)
    if "unknown" in reply.lower() or not match:
        return None
    return match.group(0).rstrip(".,)\"'")


def fetch_website_text(url):
    """Download a web page and return its visible text, capped at MAX_WEBSITE_CHARS.

    If the page has little visible text, its title and meta description are
    added, since JavaScript-heavy sites often still describe themselves there.
    """
    response = requests.get(url, timeout=10, headers={"User-Agent": BROWSER_USER_AGENT})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    # Scripts and styles are code, not words a visitor would read.
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)

    if len(text) < MIN_WEBSITE_CHARS:
        title = soup.title.get_text(strip=True) if soup.title else ""
        description = soup.find("meta", attrs={"name": "description"}) or soup.find(
            "meta", attrs={"property": "og:description"}
        )
        description = description.get("content", "") if description else ""
        text = " ".join(part for part in [title, description, text] if part)

    return text[:MAX_WEBSITE_CHARS]


def _location(company):
    """", based in City, ST" when the filing gives a location, else nothing."""
    if isinstance(company.get("city"), str) and isinstance(company.get("state"), str):
        return f", based in {company['city']}, {company['state']}"
    return ""


def search_company_text(company, website):
    """Fallback for thin websites: collect what web search results say about the company.

    Claude searches and writes a short description that cites its sources. We
    keep only the cited snippets, which are exact text from the search results,
    so the evidence quote can be checked against them just like website text.
    """
    response = get_client().messages.create(
        model=MODEL,
        max_tokens=4000,
        output_config={"effort": "low"},
        tools=[WEB_SEARCH_TOOL],
        messages=[{
            "role": "user",
            "content": (
                f"Search the web for what the company {company['company_name']} "
                f"({website}{_location(company)}) sells "
                "or builds. Write 3 to 5 sentences describing its product, citing "
                "your sources. Only use search results about this exact company."
            ),
        }],
    )
    _record_usage(response)

    snippets = []
    for block in response.content:
        if block.type == "text" and block.citations:
            for citation in block.citations:
                if citation.type == "web_search_result_location":
                    snippets.append(citation.cited_text)
    # dict.fromkeys removes duplicate snippets while keeping their order.
    return " ... ".join(dict.fromkeys(snippets))


def judge_ai_native(company_name, source_text, source_name):
    """Ask Claude whether the company is AI-native, based only on the given text."""
    response = get_client().messages.create(
        model=MODEL,
        max_tokens=4000,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "type": "string",
                            "enum": ["yes", "maybe", "no", "insufficient_info"],
                        },
                        "confidence": {"type": "integer"},
                        "evidence": {"type": "string"},
                        "product_summary": {"type": "string"},
                    },
                    "required": ["verdict", "confidence", "evidence", "product_summary"],
                    "additionalProperties": False,
                },
            },
        },
        messages=[{
            "role": "user",
            "content": (
                f"Below is text from {source_name} about {company_name}.\n\n"
                "Question: is this an AI-native product, meaning the company likely "
                "runs machine learning models in production as a core part of what it "
                "sells? A company that only mentions AI in passing, or just uses "
                "off-the-shelf AI tools internally, is not AI-native.\n\n"
                "Use only the text below. Do not use outside knowledge or guess "
                "beyond what the text says.\n\n"
                "If the text is too short or vague to tell what the company sells, "
                "answer \"insufficient_info\" rather than guessing \"no\".\n\n"
                "Return:\n"
                "- verdict: \"yes\", \"maybe\", \"no\", or \"insufficient_info\"\n"
                "- confidence: 0 to 100\n"
                "- evidence: one sentence or phrase copied exactly, word for word, "
                "from the text, that best supports your verdict\n"
                "- product_summary: one sentence describing what the company sells, "
                "based only on the text\n\n"
                f"<source_text>\n{source_text}\n</source_text>"
            ),
        }],
    )
    _record_usage(response)

    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


def _normalize(text):
    """Lowercase, straighten curly quotes, and collapse spaces, for fair comparison."""
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.split())


def qualify_company(company):
    """Run all steps for one company and return a result row. Never raises."""
    result = {
        "filing_url": company["filing_url"],
        "company_name": company["company_name"],
        "state": company["state"],
        "website": None,
        "verdict": "unknown",
        "confidence": None,
        "evidence": None,
        "evidence_source": None,
        "product_summary": None,
        "note": None,
        "qualified_at": datetime.now().isoformat(timespec="seconds"),
    }

    # Companies added by hand come with their website, so there's nothing to search for.
    known_website = company.get("website")
    try:
        website = known_website if isinstance(known_website, str) and known_website else find_website(company)
    except anthropic.APIError as error:
        result["note"] = f"Website search failed: {error}"
        return result
    if not website:
        result["note"] = "Website not found"
        return result
    result["website"] = website

    try:
        source_text = fetch_website_text(website)
    except requests.RequestException as error:
        result["note"] = f"Website could not be fetched: {error}"
        return result
    source_name, evidence_source = "the company's website", "website"

    # Website too thin to judge? Fall back to what search results say.
    if len(source_text) < MIN_WEBSITE_CHARS:
        try:
            source_text = search_company_text(company, website)
        except anthropic.APIError as error:
            result["note"] = f"Website too thin, and search fallback failed: {error}"
            return result
        if not source_text:
            result["note"] = "Website too thin, and search found nothing usable"
            return result
        source_name, evidence_source = "web search results", "search results"

    try:
        judgment = judge_ai_native(company["company_name"], source_text, source_name)
    except (anthropic.APIError, json.JSONDecodeError, StopIteration) as error:
        result["note"] = f"AI judgment failed: {error}"
        return result
    result.update(judgment)
    result["evidence_source"] = evidence_source

    notes = []
    # A low-confidence "no" means "couldn't tell", not "not AI-native".
    if result["verdict"] == "no" and result["confidence"] < MIN_CONFIDENCE_FOR_NO:
        result["verdict"] = "insufficient_info"
        notes.append(f"Low-confidence no ({result['confidence']}), treated as insufficient info")

    # Search snippets are secondhand, so they can't earn a confident "yes".
    if result["evidence_source"] == "search results" and result["verdict"] == "yes":
        result["verdict"] = "maybe"
        notes.append("Judged from search results, so capped at maybe")

    # Check Claude didn't make up the quote: it must appear in the source text.
    if _normalize(judgment["evidence"]) not in _normalize(source_text):
        if result["verdict"] == "yes":
            result["verdict"] = "maybe"
        notes.append(f"Evidence quote could not be verified in the {result['evidence_source']}")

    result["note"] = "; ".join(notes) or None
    return result


def count_to_qualify(companies, limit=LIMIT):
    """How many of these companies aren't in qualified.csv yet (and would cost money)."""
    done = set(pd.read_csv(RESULTS_FILE)["filing_url"]) if RESULTS_FILE.exists() else set()
    return min(int((~companies["filing_url"].isin(done)).sum()), limit)


def qualify_companies(companies, limit=LIMIT, on_progress=None):
    """Qualify up to `limit` companies, skipping any already in qualified.csv.

    Each result is saved right away, so progress isn't lost if the run stops.
    If given, on_progress(done, total, company_name) is called before each
    company, so a caller like the app can show a progress bar.
    """
    saved = pd.read_csv(RESULTS_FILE) if RESULTS_FILE.exists() else pd.DataFrame()
    done = set(saved["filing_url"]) if not saved.empty else set()

    to_do = companies[~companies["filing_url"].isin(done)].head(limit)
    print(f"{len(done)} already qualified, qualifying {len(to_do)} new\n")

    results = []
    for i, (_, company) in enumerate(to_do.iterrows()):
        print(f"Qualifying {company['company_name']}...")
        if on_progress:
            on_progress(i, len(to_do), company["company_name"])
        result = qualify_company(company)
        results.append(result)

        # Rewrite the whole file (rather than appending a line) so it stays
        # correct even when new columns are added to the results.
        saved = pd.concat([saved, pd.DataFrame([result])], ignore_index=True)
        RESULTS_FILE.parent.mkdir(exist_ok=True)
        saved.to_csv(RESULTS_FILE, index=False)

    return pd.DataFrame(results)


def estimated_cost():
    """Estimate this run's cost in dollars from the usage totals."""
    return (
        usage_totals["input_tokens"] * INPUT_PRICE_PER_TOKEN
        + usage_totals["output_tokens"] * OUTPUT_PRICE_PER_TOKEN
        + usage_totals["web_searches"] * PRICE_PER_SEARCH
    )


if __name__ == "__main__":
    companies = edgar.filter_by_icp(edgar.fetch_form_d_filings())
    results = qualify_companies(companies)

    # Blank cells come back from pandas as NaN ("not a number"), so we check
    # pd.notna() before printing each optional field.
    for _, row in results.iterrows():
        website = row["website"] if pd.notna(row["website"]) else "no website"
        print(f"\n{row['company_name']} ({website})")
        if pd.notna(row["confidence"]):
            print(f"  Verdict: {row['verdict']} (confidence {int(row['confidence'])})")
        else:
            print(f"  Verdict: {row['verdict']}")
        if pd.notna(row["product_summary"]):
            print(f"  Product: {row['product_summary']}")
        if pd.notna(row["evidence"]):
            print(f"  Evidence ({row['evidence_source']}): \"{row['evidence']}\"")
        if pd.notna(row["note"]):
            print(f"  Note: {row['note']}")

    print(f"\nUsage: {usage_totals}")
    print(f"Estimated cost of this run: ${estimated_cost():.4f}")
