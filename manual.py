"""Add a company by hand (no Form D needed) and run it through the same pipeline.

A manual company is saved like a filing, under the key "manual:<domain>"
(for example "manual:relaypro.com"). Every step skips companies it has
already done, so adding the same company twice costs nothing the second time.
"""

import io
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd

import agent
import fit
import qualify

MANUAL_FILE = agent.MANUAL_FILE

# Average costs measured in earlier runs, used for the estimate shown before running.
QUALIFY_COST = 0.002  # one Haiku call: the website is given, so no search is needed
THIN_SITE_EXTRA = 0.02  # a web search if the homepage has too little text
FIT_COST = 0.013
BRIEF_COST = 0.02


def normalize_website(website):
    """Add "https://" if it's missing, so "relaypro.com" works too."""
    website = website.strip()
    return website if website.startswith(("http://", "https://")) else f"https://{website}"


def domain_of(website):
    """ "https://www.relaypro.com/about" -> "relaypro.com" """
    return urlparse(normalize_website(website)).netloc.lower().removeprefix("www.")


def manual_key(website):
    return f"manual:{domain_of(website)}"


def find_existing(website):
    """If a company with this website's domain is already known, return (key, name). Else None.

    Checks every qualified company, Form D or manual, so nothing is paid for twice.
    """
    if not qualify.RESULTS_FILE.exists():
        return None
    qualified = pd.read_csv(qualify.RESULTS_FILE)
    domain = domain_of(website)
    for _, row in qualified[qualified["website"].notna()].iterrows():
        if domain_of(row["website"]) == domain:
            return row["filing_url"], row["company_name"]
    return None


def estimated_cost():
    """(typical, highest) cost in dollars for adding one company."""
    typical = QUALIFY_COST + FIT_COST + BRIEF_COST
    return typical, typical + THIN_SITE_EXTRA


def _save_manual_row(name, website, round_millions, demo_sample=False, category=None):
    """Record the company in manual_companies.csv, shaped like a Form D filing."""
    row = {
        "company_name": name.strip(),
        "city": None,
        "state": None,
        "filing_date": None,
        "industry_group": "Manual",
        "total_offering_amount": round_millions * 1e6 if round_millions else None,
        "amount_sold": None,
        "related_persons": None,
        "filing_url": manual_key(website),
        "source": "Manual",
        "round_entered_manually": bool(round_millions),
        "website": normalize_website(website),
        # Sample target accounts for the public demo (company names only).
        "demo_sample": demo_sample,
        # Your own grouping from the import file, e.g. "Voice & support agents".
        "category": category,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    saved = pd.read_csv(MANUAL_FILE) if MANUAL_FILE.exists() else pd.DataFrame()
    if not saved.empty and row["filing_url"] in set(saved["filing_url"]):
        # Already a target account: just update its category if a new one was given.
        if category:
            saved["category"] = saved.get("category", pd.Series(dtype=object)).astype(object)
            saved.loc[saved["filing_url"] == row["filing_url"], "category"] = category
            saved.to_csv(MANUAL_FILE, index=False)
        return row
    saved = pd.concat([saved, pd.DataFrame([row])], ignore_index=True)
    MANUAL_FILE.parent.mkdir(exist_ok=True)
    saved.to_csv(MANUAL_FILE, index=False)
    return row


def add_company(name, website, round_millions=None, on_progress=None, demo_sample=False, category=None):
    """Qualify, segment, and (for tier 1 to 3) write the brief and emails for one company.

    on_progress(fraction, text), if given, is called as each step starts.
    Returns {"key", "verdict", "segment", "cost"}.
    """
    def progress(fraction, text):
        if on_progress:
            on_progress(fraction, text)

    for totals in (qualify.usage_totals, fit.usage_totals, agent.usage_totals):
        totals.update(dict.fromkeys(totals, 0))
    row = _save_manual_row(name, website, round_millions, demo_sample, category)
    key = row["filing_url"]

    progress(0.0, "Checking whether it's AI-native...")
    qualify.qualify_companies(pd.DataFrame([row]), limit=1)

    progress(0.35, "Collecting evidence and deciding the segment...")
    fit.score_companies(fit.companies_to_score(filing_urls=[key]), limit=1)

    progress(0.7, "Writing the brief and emails (tier 1 to 3 only)...")
    prospects = agent.load_prospects()
    agent.write_briefs(prospects[prospects["filing_url"] == key], limit=1)
    progress(1.0, "Done")

    verdict = pd.read_csv(qualify.RESULTS_FILE).set_index("filing_url").loc[key, "verdict"]
    scores = pd.read_csv(fit.FIT_FILE).set_index("filing_url")
    segment = scores.loc[key, "account_segment"] if key in scores.index else None
    cost = qualify.estimated_cost() + fit.estimated_cost() + agent.estimated_cost()
    return {"key": key, "verdict": verdict, "segment": segment, "cost": cost}


# ---------------------------------------------------------------------------
# Bulk import from a CSV file
# ---------------------------------------------------------------------------

# Most companies one file can add, so a mistaken upload can't run up a big bill.
MAX_IMPORT_ROWS = 200

IMPORT_TEMPLATE = "name,website,round_millions,category\nExample AI,example.ai,,Voice & support agents\nAnother Co,another.com,30,Coding & developer tools\n"


def parse_import(file):
    """Read an uploaded CSV of target accounts. Returns (new_rows, known_rows, problems).

    Columns: name and website (required), round_millions (optional, in $M),
    category (optional, your own grouping).
    known_rows are companies already in your results: re-adding them is free.
    """
    try:
        df = pd.read_csv(file, dtype=str).fillna("")
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        return [], [], [f"Couldn't read the file as CSV: {error}"]
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"name", "website"} - set(df.columns)
    if missing:
        return [], [], [f"Missing column(s): {', '.join(sorted(missing))}"]
    if len(df) > MAX_IMPORT_ROWS:
        return [], [], [f"{len(df)} rows: the limit is {MAX_IMPORT_ROWS} per file. Split it into smaller files."]

    new_rows, known_rows, problems, seen = [], [], [], set()
    for line, row in enumerate(df.to_dict("records"), start=2):  # line 1 is the header
        name, website = row["name"].strip(), row["website"].strip()
        if not name or not website:
            problems.append(f"Line {line}: name and website are both required")
            continue
        round_text = row.get("round_millions", "").strip().replace("$", "").replace(",", "")
        try:
            round_millions = float(round_text) if round_text else None
        except ValueError:
            problems.append(f"Line {line}: round_millions '{row['round_millions']}' isn't a number, so it was left blank")
            round_millions = None
        domain = domain_of(website)
        if domain in seen:
            problems.append(f"Line {line}: {domain} appears twice in the file, so it was skipped")
            continue
        seen.add(domain)
        entry = {"name": name, "website": website, "round_millions": round_millions,
                 "category": row.get("category", "").strip() or None}
        existing = find_existing(website)
        if existing:
            entry["known_as"] = existing[1]
            known_rows.append(entry)
        else:
            new_rows.append(entry)
    return new_rows, known_rows, problems


def record_known_targets(rows):
    """Save already-known companies as target accounts too (free: no Claude calls).

    This keeps their category, and a known Form D company from the last 60 days
    then shows in its Raised tab tagged "Just raised".
    """
    for row in rows:
        _save_manual_row(row["name"], row["website"], row["round_millions"], category=row.get("category"))


def import_companies(rows, on_progress=None):
    """Add each new company in turn. Returns the total cost in dollars.

    on_progress(fraction, text), if given, is called before each company.
    """
    total = 0.0
    for i, row in enumerate(rows):
        if on_progress:
            on_progress(i / len(rows), f"Adding {row['name']} ({i + 1} of {len(rows)})")
        total += add_company(row["name"], row["website"], row["round_millions"], category=row.get("category"))["cost"]
    if on_progress:
        on_progress(1.0, "Done")
    return total
