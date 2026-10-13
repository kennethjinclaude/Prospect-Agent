"""Write an account brief and a cold email sequence for each qualified prospect."""

import json
import re
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd
import requests
from dotenv import load_dotenv

import qualify

# Load ANTHROPIC_API_KEY from the .env file that sits next to this script.
load_dotenv(Path(__file__).parent / ".env")

# Who the emails come from. Change these to reuse the tool for another sender.
SENDER_NAME = "Kenneth Jin"
SENDER_COMPANY = "Baseten"

# The only Baseten claims Claude is allowed to make.
BASETEN_FACTS = (
    "Baseten is an AI inference platform that turns ML models into production APIs. "
    "Offers dedicated deployments with autoscaling GPUs, OpenAI-compatible Model APIs "
    "for open-source models, and multi-model workflows. Supports HIPAA and SOC 2 Type II, "
    "single-tenant and self-hosted options. Usage-based pricing. Public customers include "
    "Cursor, Notion, Abridge, and Clay. "
    "Signed a Strategic Collaboration Agreement with AWS (Dec 2025); enterprises can run "
    "Baseten's inference on their own AWS infrastructure using existing AWS contracts."
)

# Customers Claude may name as social proof, from Baseten's customers page.
# These descriptions must stay exactly as Baseten states them: Claude may use
# them word for word and nothing else about the customer.
SOURCE_URL = "https://www.baseten.co/customers/"
PUBLIC_CUSTOMERS = {
    # name: (Baseten's description, industry or workload type)
    "Notion": ("built AI Meeting Notes on Baseten with 2-3x lower latency", "productivity, AI features"),
    "OpenEvidence": ("gives physicians fast, accurate medical information", "healthcare AI"),
    "Latent Health": ("pharmaceutical search with 99.999% uptime", "healthcare AI"),
    "EliseAI": ("trains specialized models for healthcare and housing operations", "healthcare, operations AI"),
    "Speechify": ("real-time text-to-speech for 60M+ users, cut cost per million characters by 44%", "voice AI"),
    "Wispr Flow": ("voice dictation running Llama on Baseten", "voice AI"),
    "Rime": ("state-of-the-art p99 latencies", "voice AI"),
    "Parallel": ("high-throughput agentic web research, 3x higher throughput and 2x better latency", "AI agents"),
    "Scaled Cognition": ("fast AI agents", "AI agents"),
    "Gamma": ("presentation building, 5x faster image generation", "generative AI, content"),
    "Hebbia": ("AI workflows for major financial institutions", "finance AI"),
    "Writer": ("custom medical and financial LLMs for businesses", "enterprise AI"),
    "OpenCode": ("an AI coding agent for developers", "developer tools"),
    "Posit": ("real-time AI code suggestions", "developer tools"),
}

# Every company named on Baseten's customers pages (https://www.baseten.co/customers/
# and /resources/customers/). Prospects matching one get no outbound email: they
# go to account management. Some names came from logo links, so a few may be inexact.
EXISTING_CUSTOMERS = [
    "Abridge", "Ambience Healthcare", "Amp", "Bria", "Canopy Labs", "Cartesia", "Clay",
    "ClickUp", "Cursor", "Datadog", "Decagon", "Descript", "EliseAI", "Gamma", "Gradium",
    "Harvey", "Hebbia", "Hex", "HeyGen", "HubSpot", "Inception", "Jump Trading", "Krea",
    "Latent Health", "Lovable", "Magicare", "Mercor", "Mirage", "MongoDB", "Musubi",
    "Notion", "NVIDIA", "OpenCode", "OpenEvidence", "Oxen AI", "Parallel", "Patreon",
    "Poolside", "Posit", "Praktika AI", "pyannote", "Ramp", "Retool", "Rime", "Rox",
    "Scaled Cognition", "Sid", "Speechify", "Sourcegraph", "SullyAI", "Subconscious",
    "Superhuman", "Synthefy", "Trajectory", "Turbopuffer", "Vercel", "Wayfair",
    "Wispr Flow", "World Labs", "Writer", "you.com", "Zed Industries",
]
# Other names the same customers go by (legal names on filings, longer names on the site).
CUSTOMER_ALIASES = {
    "Anysphere": "Cursor",
    "Notion Labs": "Notion",
    "Abridge AI": "Abridge",
    "Mercor.io": "Mercor",
    "Superhuman Platform": "Superhuman",
    "Parallel Web Systems": "Parallel",
    "Ambience": "Ambience Healthcare",
    "Praktika": "Praktika AI",
    "Harvey AI": "Harvey",  # Form D name; website harvey.ai
    "Lovable Labs": "Lovable",  # Form D name; website lovable.dev
}

# Legal endings removed before comparing names, so "Ramp Inc." matches "Ramp".
_LEGAL_SUFFIXES = {"inc", "incorporated", "llc", "corp", "corporation", "co", "company", "ltd", "limited", "lp", "pbc", "plc"}


def _core_name(name):
    """Lowercase a company name, drop punctuation and legal endings: "Ramp, Inc." -> "ramp"."""
    words = re.sub(r"[^a-z0-9]+", " ", name.lower().replace("&", " and ")).split()
    while words and words[-1] in _LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)


_CUSTOMER_LOOKUP = {_core_name(n): n for n in EXISTING_CUSTOMERS}
_CUSTOMER_LOOKUP.update({_core_name(alias): n for alias, n in CUSTOMER_ALIASES.items()})


def existing_customer_match(company_name):
    """The Baseten customer this company is, or None.

    The whole name must match after removing legal endings, so "Exa Labs"
    never matches a customer just because a word overlaps.
    """
    return _CUSTOMER_LOOKUP.get(_core_name(company_name))


# Dollar amounts like "$15M", "$24,999,998", or "15 million dollars". Emails mention the
# raise without its amount, so any of these in an email is flagged.
DOLLAR_AMOUNT = re.compile(r"\$\s?\d|\b\d[\d.,]*\s?(million|billion|thousand)\b", re.IGNORECASE)

# Customers named in BASETEN_FACTS but not on the customers page. Emails
# shouldn't name them, since social proof must come from PUBLIC_CUSTOMERS.
OTHER_CUSTOMER_NAMES = ["Cursor", "Abridge", "Clay"]

# Customer used for social proof when none shares the prospect's industry or workload.
DEFAULT_SOCIAL_PROOF = "Notion"

# Companies to leave out by hand, by company name as in the filing. They stay
# in briefs.csv (marked excluded) but are hidden from the final prospect list
# and never researched again.
EXCLUDE = [
    "Eldil Corp.",  # website contradicts search results; sells training data, not an inference buyer
]

# Follow-up 1 must contain this exact text, so a real link is added by a person.
RESOURCE_PLACEHOLDER = "[RESOURCE LINK: add a real case study or article]"

# Which day after the first email each follow-up is sent.
FOLLOWUP_DAYS = [3, 5, 9, 11]

MODEL = "claude-sonnet-5-5"
DATA_DIR = Path(__file__).parent / "data"
QUALIFIED_FILE = DATA_DIR / "qualified.csv"
# Companies added by hand in the app (see manual.py).
MANUAL_FILE = DATA_DIR / "manual_companies.csv"
BRIEFS_FILE = DATA_DIR / "briefs.csv"

# How many new companies to research per run. Lower it (e.g. to 5) to test cheaply.
LIMIT = 100

# Account segments from fit.py, best first. Deprioritized companies get no brief or email.
# existing_customer goes to account management, and needs_review means a person checks
# the stage first. Neither gets an outbound email.
SEGMENT_ORDER = {"tier_1": 0, "tier_2": 1, "tier_3": 2, "existing_customer": 3, "needs_review": 4, "deprioritize": 5}
SEGMENTS_WITH_EMAIL = ["tier_1", "tier_2", "tier_3"]

# Published prices, used only to estimate what a run cost.
INPUT_PRICE_PER_TOKEN = 2.00 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 10.00 / 1_000_000

usage_totals = {"input_tokens": 0, "output_tokens": 0}


# The shape of the JSON Claude must return. The email fields are blanks that
# the code drops into the template; Claude never writes the fixed wording.
BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "relationship": {"type": "string", "enum": ["customer", "competitor", "partner"]},
        "relationship_reason": {"type": "string"},
        "what_they_build": {"type": "string"},
        "likely_models": {"type": "string"},
        "inference_pain": {"type": "string"},
        "why_now": {"type": "string"},
        "ideal_role": {"type": "string"},
        "healthcare_or_patient_data": {"type": "boolean"},
        "company_display_name": {"type": "string"},
        "subject_theme": {"type": "string"},
        "parallel_sentence": {"type": "string"},
        "social_proof_customer": {"type": "string", "enum": list(PUBLIC_CUSTOMERS)},
        "social_proof_sentence": {"type": "string"},
        "how_paragraph": {"type": "string"},
        "followup_day3": {"type": "string"},
        "followup_day5": {"type": "string"},
        "followup_day9": {"type": "string"},
        "followup_day11": {"type": "string"},
        "short_email_body": {"type": "string"},
        "nurture_note": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "relationship", "relationship_reason", "what_they_build", "likely_models",
        "inference_pain", "why_now", "ideal_role", "healthcare_or_patient_data",
        "company_display_name", "subject_theme", "parallel_sentence",
        "social_proof_customer", "social_proof_sentence", "how_paragraph",
        "followup_day3", "followup_day5", "followup_day9", "followup_day11",
        "short_email_body", "nurture_note", "assumptions",
    ],
    "additionalProperties": False,
}

def build_instructions():
    """Build Claude's instructions from the current settings.

    Built at call time (not once when the file loads) so a caller like the
    app can change SENDER_NAME or SENDER_COMPANY before running.
    """
    customer_list = "\n".join(
        f"- {name}: {description} ({category})"
        for name, (description, category) in PUBLIC_CUSTOMERS.items()
    )
    return f"""You are helping {SENDER_NAME}, a sales rep at {SENDER_COMPANY}, research a prospect and write a cold email sequence.

<baseten_facts>
{BASETEN_FACTS}
</baseten_facts>

<public_customers source="{SOURCE_URL}">
{customer_list}
</public_customers>

Rules:
- Every claim about the prospect must come from the Form D filing or the website text below. If you infer something (for example, which models they likely run), list it in "assumptions".
- Every claim about {SENDER_COMPANY} must come from baseten_facts. Do not add any other features, numbers, or results.
- Customers named in emails must come from public_customers, described only with their description from that list.
- Only use {SENDER_COMPANY} facts that are relevant to this company. Mention HIPAA only if the company works in healthcare or handles patient data.
- Any benefit in an email must directly restate a fact in baseten_facts. For example, "turns ML models into production APIs" can become "so you can run models as production APIs". Never promise outcomes the facts don't state, such as data staying protected, costs going down, faster shipping, or not having to manage infrastructure.
- No em dashes or en dashes anywhere. Conversational, plain words. No buzzwords.
- Never open any email, including short emails and follow-ups, with "Congrats", "Congratulations", or any compliment. Mention the raise later in the email, if at all.
- When an email mentions the raise or funding round, never give the dollar amount (write "your recent round", not "$15M round"). This applies to every email in every segment, subjects included.

Brief:
- relationship: "customer" if they build an AI product that would run models in production; "competitor" if they sell AI inference or GPU infrastructure themselves; "partner" if they are an agency or studio that builds AI for clients. relationship_reason: one sentence why.
- what_they_build: what the company sells, from the website text.
- likely_models: the kinds of ML models they likely run (for example, LLMs, speech-to-text, vision models).
- inference_pain: the inference problem they likely face, such as latency, GPU cost, or scaling after the raise.
- why_now: why this is a good moment to reach out, based on the filing.
- ideal_role: the job title to look for if the filing's executives aren't the right technical contact (for example, CTO or Head of ML).
- healthcare_or_patient_data: true if the company works in healthcare or handles patient data.
- assumptions: everything in the brief or emails that is inferred rather than stated in the source data.

Account segment. The message says which segment this company is in. Write for that segment:
- tier_1 (AI-native scale-up): the full first email and follow-ups below, about running their models in production.
- tier_2 (enterprise or regulated industry): the full first email and follow-ups, but lead with security, data control, and running on their own cloud. In parallel_sentence and how_paragraph, use the baseten_facts about SOC 2 Type II, HIPAA (healthcare only), single-tenant and self-hosted options, and the AWS agreement (running on their own AWS infrastructure using existing AWS contracts). Keep that angle in the follow-ups.
- tier_3 (early-stage startup): one short email only. Leave parallel_sentence, social_proof_sentence, how_paragraph, and all followup fields as empty strings. Instead write:
    short_email_body: 2 to 3 short sentences, under 50 words, about their recent raise, connected to the one baseten_fact most useful to an early team, ending with an easy yes/no question. The code adds the greeting and sign-off.
    nurture_note: one sentence for the sales rep on when to reach out again (for example, when their usage spikes or they raise again), plus anything specific from the website.
For tier_1 and tier_2, leave short_email_body and nurture_note as empty strings. subject_theme and company_display_name are used in every segment.

Email blanks. The code builds the first email from this fixed template, and your blanks go in the curly braces:

    Subject: {{first name}} // {{subject_theme}}

    Good morning, {{first name}},

    My name is {SENDER_NAME}, and I'm part of the team at {SENDER_COMPANY}. {{parallel_sentence}} {{social_proof_sentence}}

    {{how_paragraph}}

    Would you have 15 minutes this week or next to see if it's a fit for {{company_display_name}}?

    Thank you in advance,
    {SENDER_NAME.split()[0]}

The whole first email should be 80 to 100 words.

- company_display_name: the name the company uses on its website (for example, a brand name rather than the legal name in the filing).
- subject_theme: 2 to 5 words in Title Case, specific to this company's situation (for example, "Scaling Inference Efficiently").
- parallel_sentence: one sentence, under 25 words, starting with company_display_name. First half: what they do, from their website. Second half, after ", and we": how {SENDER_COMPANY} does the matching thing for the AI behind it. Write it fresh for this company; don't reuse a stock phrase. Examples of the style:
    "PreciseDx is bringing AI to cancer diagnostics, and we make sure the models behind products like that run fast and reliably in production."
    "TRUE is taking manual document work out of mortgage operations, and we take the infrastructure work out of running the AI behind it."
- social_proof_customer: exactly one customer from public_customers, chosen by closest industry or workload type (voice to voice, healthcare to healthcare, agents to agents, finance to finance). If no customer shares the prospect's industry or workload type, use {DEFAULT_SOCIAL_PROOF} (AI features at scale) rather than forcing a mismatched one.
- social_proof_sentence: one sentence, under 20 words, naming only that customer and using only its description from the list (you may rephrase the grammar, but keep its facts and numbers exactly).
- how_paragraph: 1 to 2 sentences, under 30 words, starting with "We do this by", using only the baseten_facts most relevant to this company.

Follow-ups go in the same thread. Write only the body text: the code adds the greeting and the "Thanks," sign-off. Each is 1 to 3 short sentences.
- followup_day3: share a resource by writing exactly {RESOURCE_PLACEHOLDER} where the link goes, with one sentence on why it's relevant to them. Never invent a study, title, or URL. Then ask if Thursday or Friday morning next week works for 15 minutes.
- followup_day5: a short bump checking the note reached them, plus one sentence reminding them what {SENDER_COMPANY} does for companies like theirs (from baseten_facts). Then ask if Tuesday or Wednesday morning next week works for a quick introduction.
- followup_day9: ask if it makes sense to revisit this later, offer to reconnect next quarter, and close the loop either way.
- followup_day11: assume it isn't a priority right now and offer to revisit another quarter.

If relationship is not "customer", fill every email blank with an empty string (and pick any social_proof_customer)."""


def load_form_d_data():
    """Return every Form D filing we've cached, one row per filing link.

    Reads all the form_d_*.csv files in data/, so companies found on an
    earlier day are still covered after the 7-day window moves on. Companies
    added by hand (manual.py) are included too, with source "Manual".
    """
    files = sorted(DATA_DIR.glob("form_d_*.csv"))
    tables = [pd.read_csv(f) for f in files]
    if MANUAL_FILE.exists():
        # The website is stored with the verdict, so leave it out here to avoid a duplicate column.
        tables.append(pd.read_csv(MANUAL_FILE).drop(columns=["website", "added_at"], errors="ignore"))
    filings = pd.concat(tables, ignore_index=True)
    # If a filing appears in several files, keep the copy from the newest file.
    return filings.drop_duplicates("filing_url", keep="last")


def load_prospects():
    """Return companies to write briefs for: tier_1, tier_2, and tier_3 from fit.py.

    Run fit.py first. Deprioritized, unscored, and excluded companies are left out.
    """
    qualified = pd.read_csv(QUALIFIED_FILE)
    qualified = qualified[~qualified["company_name"].isin(EXCLUDE)]
    fit_file = DATA_DIR / "fit.csv"
    if not fit_file.exists():
        print("No fit.csv yet: run fit.py to segment companies first.")
        return qualified.iloc[0:0]
    fit = pd.read_csv(fit_file)[["filing_url", "account_segment", "fit_grade", "workload_type"]]
    prospects = qualified.merge(fit, on="filing_url").merge(
        load_form_d_data().drop(columns=["company_name", "state"]),
        on="filing_url",
        how="left",
    )
    prospects = prospects[prospects["account_segment"].isin(SEGMENTS_WITH_EMAIL)].copy()
    # Best leads first: tier_1, tier_2, tier_3, then grade A, B, C, then confidence.
    prospects["segment_order"] = prospects["account_segment"].map(SEGMENT_ORDER)
    return prospects.sort_values(["segment_order", "fit_grade", "confidence"], ascending=[True, True, False])


def _format_amount(amount):
    return f"${amount:,.0f}" if pd.notna(amount) else "not stated"


def _manual_company_data(prospect):
    """What we know about a company added by hand: no filing, maybe a round size the user entered."""
    if pd.notna(prospect["total_offering_amount"]):
        funding = (f"Round size: {_format_amount(prospect['total_offering_amount'])}, entered manually by the user "
                   "(not from a filing). You may mention their recent round, without the amount.")
    else:
        funding = "Funding: no round is known. Do not mention a raise or funding round."
    return f"""<company_data source="added manually, no Form D filing">
Company: {prospect['company_name']}
{funding}
Related persons: none known
</company_data>"""


def _source_data(prospect, website_text):
    """Lay out everything we know about the prospect, for the prompt."""
    if str(prospect["filing_url"]).startswith("manual:"):
        company_block = _manual_company_data(prospect)
    else:
        company_block = _form_d_data(prospect)
    return f"""<account_segment>{prospect['account_segment']}</account_segment>

{company_block}

<website url="{prospect['website']}">
{website_text or "(website text unavailable)"}
</website>

<qualification_evidence>
{prospect['evidence']}
</qualification_evidence>"""


def _form_d_data(prospect):
    """The Form D facts about a prospect, for the prompt."""
    return f"""<form_d_filing>
Company: {prospect['company_name']}
Address: {prospect['city']}, {prospect['state']}
Filing date: {prospect['filing_date']}
Industry group: {prospect['industry_group']}
Total offering amount: {_format_amount(prospect['total_offering_amount'])}
Amount sold so far: {_format_amount(prospect['amount_sold'])}
Related persons: {prospect['related_persons']}
</form_d_filing>"""


def _normal_case(name):
    """Fix ALL CAPS names from filings: "JIM O'NEILL" -> "Jim O'Neill". Others stay as filed."""
    return name.title() if name.isupper() else name


def pick_contact(related_persons):
    """Return (name, roles, note) for the best contact in the filing.

    Picks the first-listed Executive Officer. If there is none, falls back to
    the first-listed Director and says so in the note. Decided by code, so the
    same company always gets the same contact.
    """
    people = []
    if isinstance(related_persons, str):
        for person in related_persons.split("; "):
            name, _, roles = person.partition(" (")
            people.append((_normal_case(name), roles.rstrip(")")))

    for name, roles in people:
        if "Executive Officer" in roles:
            return name, roles, None
    for name, roles in people:
        if "Director" in roles:
            return name, roles, "Director, no executive listed"
    return None, None, None


def _first_name(full_name):
    """Get a first name from a filing name, e.g. "J. JOHN ROE" -> "John".

    Skips initials, and fixes ALL CAPS names. Without a contact, returns a
    placeholder for a person to fill in, rather than guessing.
    """
    if not full_name:
        return "[First name]"
    for part in full_name.split():
        if not part.endswith("."):
            return part.capitalize() if part.isupper() else part
    return "[First name]"


def build_emails(brief, first_name, segment):
    """Drop Claude's blanks into the email template. Returns a dict of email columns.

    tier_3 gets one short email and no follow-ups; other segments get the full sequence.
    """
    sender_first = SENDER_NAME.split()[0]
    company = brief["company_display_name"]
    subject = f"{first_name} // {brief['subject_theme']}"

    if segment == "tier_3":
        emails = {
            "email_subject": subject,
            "email_body": f"Good morning, {first_name},\n\n{brief['short_email_body']}\n\nThanks,\n{sender_first}",
            "followup_subject": None,
        }
        emails.update({f"followup_day{day}": None for day in FOLLOWUP_DAYS})
        return emails

    initial = (
        f"Good morning, {first_name},\n\n"
        f"My name is {SENDER_NAME}, and I'm part of the team at {SENDER_COMPANY}. "
        f"{brief['parallel_sentence']} {brief['social_proof_sentence']}\n\n"
        f"{brief['how_paragraph']}\n\n"
        f"Would you have 15 minutes this week or next to see if it's a fit for {company}?\n\n"
        f"Thank you in advance,\n{sender_first}"
    )

    emails = {"email_subject": subject, "email_body": initial, "followup_subject": f"Re: {subject}"}
    for day in FOLLOWUP_DAYS:
        text = brief[f"followup_day{day}"]
        emails[f"followup_day{day}"] = f"Hi {first_name},\n\n{text}\n\nThanks,\n{sender_first}"
    return emails


def _email_problems(emails, brief, segment):
    """Check the email rules that can be tested by code, and describe any broken ones."""
    problems = []
    all_text = " ".join(text for text in emails.values() if text)
    customer = brief["social_proof_customer"]
    word_limit = 80 if segment == "tier_3" else 110

    # No email may open with congratulations: check the first line after each greeting.
    for column, text in emails.items():
        if column.endswith("subject") or not text:
            continue
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) > 1 and lines[1].lower().startswith("congrat"):
            problems.append(f"{column} opens with congratulations")

    # No email may state a dollar amount (the raise is mentioned without it).
    for column, text in emails.items():
        if text and DOLLAR_AMOUNT.search(text):
            problems.append(f"{column} states a dollar amount: '{DOLLAR_AMOUNT.search(text).group(0)}...'")

    if "—" in all_text or "–" in all_text:
        problems.append("An email contains a dash that should be edited out")

    word_count = len(emails["email_body"].split())
    if word_count >= word_limit:
        problems.append(f"First email is {word_count} words (limit is under {word_limit})")

    # "Name // Theme" where the theme is 2 to 5 words.
    if not re.fullmatch(r".+ // (\S+ ){1,4}\S+", emails["email_subject"]):
        problems.append(f"Subject '{emails['email_subject']}' doesn't match 'Name // 2-5 word theme'")

    if re.search(r"https?://|www\.", all_text):
        problems.append("An email contains a link, which may be made up")
    if "HIPAA" in all_text and not brief["healthcare_or_patient_data"]:
        problems.append("Mentions HIPAA, but the company isn't in healthcare")

    # tier_3's short email has no social proof or follow-ups to check.
    if segment == "tier_3":
        if any(name in all_text for name in list(PUBLIC_CUSTOMERS) + OTHER_CUSTOMER_NAMES):
            problems.append("Short email names a customer, but it has no checked social proof sentence")
        return problems

    # Social proof: one listed customer, with no facts beyond its description.
    if customer not in PUBLIC_CUSTOMERS:
        problems.append(f"Social proof customer '{customer}' is not on the customers list")
    elif customer not in brief["social_proof_sentence"]:
        problems.append("Social proof sentence doesn't name the chosen customer")
    else:
        description = PUBLIC_CUSTOMERS[customer][0]
        for number in re.findall(r"\d[\d.,%xX+M-]*", brief["social_proof_sentence"]):
            if number.rstrip(".,") not in description:
                problems.append(f"Social proof number '{number}' isn't in {customer}'s description")
    named = [n for n in list(PUBLIC_CUSTOMERS) + OTHER_CUSTOMER_NAMES if n != customer and n in all_text]
    if named:
        problems.append(f"Emails name other customers: {', '.join(named)}")

    if RESOURCE_PLACEHOLDER not in emails["followup_day3"]:
        problems.append("Follow-up 1 is missing the resource link placeholder")

    return problems


def write_brief(prospect):
    """Research one prospect and return a result row. Never raises."""
    result = {
        "filing_url": prospect["filing_url"],
        "company_name": prospect["company_name"],
        "website": prospect["website"],
        "verdict": prospect["verdict"],
        "account_segment": prospect["account_segment"],
        "note": None,
        "sender_name": SENDER_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        website_text = qualify.fetch_website_text(prospect["website"])
    except requests.RequestException:
        website_text = ""

    try:
        response = qualify.get_client().beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": BRIEF_SCHEMA}},
            # If a safety check declines the request, Anthropic retries it on
            # another model where it can, instead of just returning a refusal.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=build_instructions(),
            messages=[{"role": "user", "content": _source_data(prospect, website_text)}],
        )
    except anthropic.APIError as error:
        result["note"] = f"API call failed: {error}"
        return result
    usage_totals["input_tokens"] += response.usage.input_tokens
    usage_totals["output_tokens"] += response.usage.output_tokens

    if response.stop_reason == "refusal":
        result["note"] = "Claude declined to write this brief"
        return result
    text = next(block.text for block in response.content if block.type == "text")
    brief = json.loads(text)

    # Copy the brief fields into the result. The email blanks are only used
    # to build the emails below, so they aren't saved on their own.
    for field in ["relationship", "relationship_reason", "what_they_build", "likely_models",
                  "inference_pain", "why_now", "ideal_role", "company_display_name"]:
        result[field] = brief[field]
    result["assumptions"] = "; ".join(brief["assumptions"])

    # The contact and their title come straight from the filing, never from Claude.
    result["contact_name"], result["contact_title"], contact_note = pick_contact(
        prospect["related_persons"]
    )

    notes = [contact_note] if contact_note else []
    segment = prospect["account_segment"]
    if brief["relationship"] == "customer" and segment in SEGMENTS_WITH_EMAIL:
        emails = build_emails(brief, _first_name(result["contact_name"]), segment)
        result.update(emails)
        if segment == "tier_3":
            result["nurture_note"] = brief["nurture_note"]
        else:
            result["social_proof_customer"] = brief["social_proof_customer"]
        notes.extend(_email_problems(emails, brief, segment))

    result["note"] = "; ".join(notes) or None
    return result


def write_briefs(prospects, limit=LIMIT, on_progress=None):
    """Write briefs for up to `limit` prospects, skipping any already in briefs.csv.

    If given, on_progress(done, total, company_name) is called before each
    company, so a caller like the app can show a progress bar.
    """
    saved = pd.read_csv(BRIEFS_FILE) if BRIEFS_FILE.exists() else pd.DataFrame()
    done = set(saved["filing_url"]) if not saved.empty else set()

    to_do = prospects[~prospects["filing_url"].isin(done)].head(limit)
    print(f"{len(done)} briefs already written, writing {len(to_do)} new\n")

    results = []
    for i, (_, prospect) in enumerate(to_do.iterrows()):
        if on_progress:
            on_progress(i, len(to_do), prospect["company_name"])
        print(f"Researching {prospect['company_name']}...")
        result = write_brief(prospect)
        results.append(result)

        saved = pd.concat([saved, pd.DataFrame([result])], ignore_index=True)
        _save_briefs(saved)

    # Save once more so changes to EXCLUDE apply even when nothing new was written.
    if not saved.empty:
        _save_briefs(saved)
    return pd.DataFrame(results)


def _save_briefs(briefs):
    """Save briefs.csv, marking each row excluded or not from the EXCLUDE list."""
    briefs["excluded"] = briefs["company_name"].isin(EXCLUDE)
    briefs.to_csv(BRIEFS_FILE, index=False)


def final_prospects():
    """Return the briefs ready for outreach: customers that aren't excluded."""
    briefs = pd.read_csv(BRIEFS_FILE)
    return briefs[(briefs["relationship"] == "customer") & ~briefs["excluded"]]


def print_sequence(row):
    """Print a company's first email and follow-ups, in sending order."""
    print(f"--- Day 0 ---\nSubject: {row['email_subject']}\n\n{row['email_body']}\n")
    for day in FOLLOWUP_DAYS:
        print(f"--- Day {day} ---\nSubject: {row['followup_subject']}\n\n{row[f'followup_day{day}']}\n")


def estimated_cost():
    """Estimate this run's cost in dollars from the usage totals."""
    return (
        usage_totals["input_tokens"] * INPUT_PRICE_PER_TOKEN
        + usage_totals["output_tokens"] * OUTPUT_PRICE_PER_TOKEN
    )


if __name__ == "__main__":
    prospects = load_prospects()
    print(f"{len(prospects)} prospects qualify for a brief")
    results = write_briefs(prospects)

    for _, row in results.iterrows():
        print("\n" + "=" * 80)
        print(f"{row['company_name']} ({row['website']})")
        if pd.notna(row.get("note")):
            print(f"Note: {row['note']}")
        if pd.isna(row.get("relationship")):
            continue
        print(f"Relationship: {row['relationship']} ({row['relationship_reason']})")
        print(f"\nWhat they build: {row['what_they_build']}")
        print(f"Likely models:   {row['likely_models']}")
        print(f"Inference pain:  {row['inference_pain']}")
        print(f"Why now:         {row['why_now']}")
        if pd.notna(row["contact_name"]):
            print(f"\nContact: {row['contact_name']} ({row['contact_title']})")
        print(f"If not the right person, look for: {row['ideal_role']}")
        print(f"\nAssumptions: {row['assumptions']}\n")
        if pd.notna(row.get("email_body")):
            print_sequence(row)

    final = final_prospects()
    print(f"\nFinal prospect list: {len(final)} customers ready for outreach")
    print(final[["company_name", "contact_name", "email_subject"]].to_string(index=False))

    print(f"\nUsage: {usage_totals}")
    print(f"Estimated cost of this run: ${estimated_cost():.4f}")
