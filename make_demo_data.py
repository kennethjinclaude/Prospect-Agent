"""Copy a live run's results from data/ into demo_data/, with people's names removed.

demo_data/ is committed to git and shown publicly, so:
- every related person's name is replaced with their role from the filing
- email greetings and subjects use "[First name]"
- emails say "I'm writing about Baseten" instead of "I'm part of the team at Baseten",
  and talk about Baseten in the third person ("Baseten does this by..."), not "we"
Company names are kept. Run it with:  python make_demo_data.py
"""

import functools
import json
import re
import shutil
from urllib.parse import urlparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
SOURCE_DIR = ROOT / "data"
DEMO_DIR = ROOT / "demo_data"

# The sender's own name stays in the emails, even if someone in a filing
# shares a first or last name with them (one filing lists a "Kenneth").
DEFAULT_SENDER = "Kenneth Jin"


def _sender_words(row):
    """Lowercased words of the sender's name for this row."""
    sender = row.get("sender_name")
    return set((sender if isinstance(sender, str) else DEFAULT_SENDER).lower().split())


def _is_company(name):
    """Some filings list a firm (e.g. "N/A SCA-South Jersey, LLC") as a related person."""
    return name.upper().startswith("N/A") or bool(re.search(r"\b(LLC|LP|INC|CORP)\b", name.upper()))


def _people(related_persons):
    """Return [(name, roles)] for the individuals in edgar's related_persons text."""
    people = []
    if isinstance(related_persons, str):
        for person in related_persons.split("; "):
            name, _, roles = person.partition(" (")
            if not _is_company(name):
                people.append((name, roles.rstrip(")")))
    return people


def _first_and_last(name):
    """A person's first and last name, skipping initials and suffixes like "Jr."."""
    words = [w for w in re.findall(r"[A-Za-z'\-]+", name) if len(w) >= 2 and w.lower() not in ("jr", "sr", "ii", "iii")]
    return words[0], words[-1]


def _name_pattern(word):
    """Match a name as a capitalized word ("Eric" or "ERIC"), never a lowercase word like "will"."""
    return rf"\b({re.escape(word.title())}|{re.escape(word.upper())})\b"


def scrub(text, people, company_name, protected=frozenset()):
    """Replace each person's full name with their role, then any leftover first or last name.

    Words that are part of the company's own name are left alone (for
    example, "Turner" in "Turner Nanotechnology Corp").
    """
    if not isinstance(text, str):
        return text
    for name, roles in people:
        first, last = _first_and_last(name)
        # The full name as filed, and also just first + last (dropping middle names).
        for variant in sorted({name, f"{first} {last}"}, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(variant)}\b", f"[{roles}]", text, flags=re.IGNORECASE)
        for word, placeholder in [(first, "[First name]"), (last, "[Last name]")]:
            if word.lower() not in company_name.lower() and word.lower() not in protected:
                text = re.sub(_name_pattern(word), placeholder, text)
    return text


# Website quotes (testimonials, leadership pages) can name people who aren't in the filing,
# e.g. "Jane Doe, Vice President of Sales at Acme". When a job title follows the name,
# the name becomes "the": "the Vice President of Sales at Acme".
TITLE_START = (r"(?:CEO|CTO|COO|CFO|CPO|CMO|CIO|CISO|Chief|President|Vice President|VP|SVP|EVP|Head|Director|"
               r"Founder|Co-[Ff]ounder|Founding|Senior|Principal|Managing|General Manager|Partner)")
# "Jane Doe", "Dr. Jane Doe", "John St. Clair", "John Roe, MD"
PERSON = (r"(?:Dr\.\s+)?(?P<first>[A-Z][a-z]+)(?:[-'][A-Z]?[a-z]+)?(?:\s+[A-Z]\.)?\s+(?:St\.\s+)?"
          r"[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?(?:,\s+(?:MD|PhD|DO|RN))?")
NAME_BEFORE_TITLE = re.compile(rf"\b{PERSON},?\s+(?={TITLE_START}\b)")
# A short first name next to a last name the filing scrub already removed ("Jo [Last name]").
FIRST_BEFORE_SCRUBBED_LAST = re.compile(r"\b(?P<first>[A-Z][a-z]+) \[Last name\]")

# Short forms of first names that the Form D list may not include.
EXTRA_FIRST_NAMES = {"aly", "sam", "bob", "bill", "liz", "kate", "matt", "nick", "chris", "alex", "dan", "jen"}

# People named without a title next to their name, and what to show instead: {"Jane Doe": "[name]"}.
# A role is used only when the website itself gives it; otherwise "[name]", so no role is made up.
# The list holds real names, so it lives in data/ (gitignored), not in this public file.
QUOTED_PEOPLE_FILE = SOURCE_DIR / "quoted_people.json"
QUOTED_PEOPLE = json.loads(QUOTED_PEOPLE_FILE.read_text()) if QUOTED_PEOPLE_FILE.exists() else {}

# Capitalized word pairs that start with a common first name but aren't people.
NOT_PEOPLE = {
    "Heidi Health", "Heidi Desktop", "Ludwig Computing", "Liora Life", "Claude Connector", "Claude Generate",
    "Max Mode", "General Cybernetics", "General World", "Morgan Asset", "Hong Kong", "Mac Android",
    "Harvey Mobile", "Devin Desktop", "Israel Lahey", "Alice Outbound", "Management Company",
}

# Every rename made by scrub_quoted_names, printed after a rebuild so a person can check them.
quote_renames = []


@functools.cache
def _known_first_names():
    return _first_names() | EXTRA_FIRST_NAMES


def scrub_quoted_names(text):
    """Replace people named in website quotes with their role (see NAME_BEFORE_TITLE and QUOTED_PEOPLE)."""
    if not isinstance(text, str):
        return text
    for name, role in QUOTED_PEOPLE.items():
        if name in text:
            quote_renames.append((name, role))
            text = text.replace(name, role)

    def replace_if_person(replacement):
        def replace(match):
            # Only a real first name counts, so "Case Studies Creative Partner" is left alone.
            if match.group("first").lower() not in _known_first_names():
                return match.group(0)
            quote_renames.append((match.group(0).strip(" ,"), replacement.strip()))
            return replacement
        return replace
    text = FIRST_BEFORE_SCRUBBED_LAST.sub(replace_if_person("[First name] [Last name]"), text)
    return NAME_BEFORE_TITLE.sub(replace_if_person("the "), text)


EMAIL_COLUMNS = ["email_subject", "email_body", "followup_day3", "followup_day5", "followup_day9", "followup_day11"]

# "we", "our", "us" and their contractions, as whole words.
FIRST_PERSON = re.compile(r"\b(we|we're|we've|we'd|we'll|our|ours|us)\b", re.IGNORECASE)


IRREGULAR_VERBS = {"do": "does", "have": "has", "are": "is", "were": "was", "can": "can", "will": "will"}
# Words that can sit between "we" and its verb ("we also support").
ADVERBS = r"(?:also|now|already|often|just|even|still|actually)"


def _third_person_verb(verb):
    """Turn a "we" verb into a "Baseten" verb: make -> makes, push -> pushes, carry -> carries, do -> does."""
    lower = verb.lower()
    if lower in IRREGULAR_VERBS:
        return IRREGULAR_VERBS[lower]
    if lower.endswith(("s", "sh", "ch", "x", "o")):
        return verb + "es"
    if lower.endswith("y") and lower[-2:-1] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


def third_person(text, company="Baseten"):
    """Rewrite "we", "our", and "us" about the sender's company in the third person.

    "We do this by..."        -> "Baseten does this by..."
    "We also support HIPAA"   -> "Baseten also supports HIPAA"
    "..., and we make sure"   -> "..., and Baseten makes sure"
    "our customers" / "with us" -> "Baseten's customers" / "with Baseten"
    """
    if not isinstance(text, str):
        return text
    text = re.sub(
        rf"\b[Ww]e ((?:{ADVERBS} )?)(\w+)\b",
        lambda m: f"{company} {m.group(1)}{_third_person_verb(m.group(2))}",
        text,
    )
    text = re.sub(r"\b[Oo]ur\b", f"{company}'s", text)
    return re.sub(r"\bus\b", company, text)


def make_demo_data():
    DEMO_DIR.mkdir(exist_ok=True)
    filtered = pd.read_csv(SOURCE_DIR / "filtered.csv")
    people_by_filing = dict(zip(filtered["filing_url"], filtered["related_persons"].map(_people)))

    # Form D data: keep each person's role, drop their name.
    filtered["related_persons"] = filtered["related_persons"].map(
        lambda text: "; ".join(roles for _, roles in _people(text))
    )
    filtered.to_csv(DEMO_DIR / "filtered.csv", index=False)

    # Sample target accounts (company names only) are the one kind of manual company in the demo.
    sample_keys = set()
    manual_file = SOURCE_DIR / "manual_companies.csv"
    if manual_file.exists():
        manual_rows = pd.read_csv(manual_file)
        samples = manual_rows[manual_rows.get("demo_sample", False) == True]  # noqa: E712 (CSV gives True/False)
        sample_keys = set(samples["filing_url"])
        samples.to_csv(DEMO_DIR / "manual_companies.csv", index=False)

    for file_name in ["qualified.csv", "briefs.csv", "fit.csv"]:
        if not (SOURCE_DIR / file_name).exists():
            continue
        df = pd.read_csv(SOURCE_DIR / file_name)
        # Companies added by hand stay out of the public demo, except the sample target accounts.
        is_manual = df["filing_url"].astype(str).str.startswith("manual:")
        df = df[~is_manual | df["filing_url"].isin(sample_keys)].reset_index(drop=True)
        text_columns = df.select_dtypes(include=["object", "string"]).columns.drop(
            ["filing_url", "company_name", "website", "sender_name"], errors="ignore"
        )
        for i, row in df.iterrows():
            people = people_by_filing.get(row["filing_url"], [])
            for column in text_columns:
                df.at[i, column] = scrub_quoted_names(scrub(row[column], people, row["company_name"], _sender_words(row)))
        if "contact_name" in df:
            df["contact_name"] = df["contact_title"]
        if "email_body" in df:
            df["email_body"] = df["email_body"].str.replace(
                "I'm part of the team at Baseten", "I'm writing about Baseten", regex=False
            )
        for column in EMAIL_COLUMNS:
            if column in df:
                df[column] = df[column].map(third_person)
        df.to_csv(DEMO_DIR / file_name, index=False)

    shutil.copy(SOURCE_DIR / "run_stats.json", DEMO_DIR / "run_stats.json")
    copy_research_briefs(people_by_filing, sample_keys)
    copy_saved_sections("contacts", people_by_filing, sample_keys)
    copy_favicons()


def copy_favicons():
    """Copy the saved favicons for demo companies (small embedded images, no names)."""
    source = SOURCE_DIR / "favicons.json"
    if not source.exists():
        return
    icons = json.loads(source.read_text())
    websites = pd.read_csv(DEMO_DIR / "qualified.csv")["website"].dropna().tolist()
    if (DEMO_DIR / "manual_companies.csv").exists():
        websites += pd.read_csv(DEMO_DIR / "manual_companies.csv")["website"].dropna().tolist()
    domains = {urlparse(w if "://" in w else f"https://{w}").netloc.lower().removeprefix("www.") for w in websites}
    (DEMO_DIR / "favicons.json").write_text(json.dumps({d: icons[d] for d in domains if d in icons}))


def copy_research_briefs(people_by_filing, sample_keys):
    """Copy saved research briefs for demo companies, with Form D names scrubbed like everything else."""
    copy_saved_sections("research", people_by_filing, sample_keys)


def copy_saved_sections(folder_name, people_by_filing, sample_keys):
    """Copy one folder of saved JSON sections (research, contacts) for demo companies, names scrubbed."""
    source_dir, demo_dir = SOURCE_DIR / folder_name, DEMO_DIR / folder_name
    shutil.rmtree(demo_dir, ignore_errors=True)
    if not source_dir.exists():
        return
    demo_dir.mkdir()
    for path in source_dir.glob("*.json"):
        saved = json.loads(path.read_text())
        key = saved["filing_url"]
        if key not in people_by_filing and key not in sample_keys:
            continue  # not a demo company
        people = people_by_filing.get(key, [])
        text = scrub_quoted_names(scrub(json.dumps(saved), people, saved["company_name"]))
        (demo_dir / path.name).write_text(json.dumps(json.loads(text), indent=2))


def find_leftover_names():
    """Check every demo row for the first or last name of anyone in its filing.

    Returns a list of (file, name, surrounding text) for each hit. The
    company name and link columns are skipped, since company names are kept.
    """
    source = pd.read_csv(SOURCE_DIR / "filtered.csv")
    people_by_filing = dict(zip(source["filing_url"], source["related_persons"].map(_people)))
    hits = []
    for file_name in ["filtered.csv", "qualified.csv", "briefs.csv", "fit.csv"]:
        if not (DEMO_DIR / file_name).exists():
            continue
        df = pd.read_csv(DEMO_DIR / file_name)
        for _, row in df.iterrows():
            text = " | ".join(
                str(value) for column, value in row.items()
                if column not in ("company_name", "filing_url", "website", "sender_name") and pd.notna(value)
            )
            for name, _ in people_by_filing.get(row["filing_url"], []):
                for word in _first_and_last(name):
                    if word.lower() in row["company_name"].lower() or word.lower() in _sender_words(row):
                        continue
                    for match in re.finditer(_name_pattern(word), text):
                        hits.append((file_name, word, text[max(0, match.start() - 40):match.end() + 40]))
    return hits


def _first_names():
    """Every first name in the downloaded Form D filings (thousands of people), lowercased."""
    names = set()
    for path in SOURCE_DIR.glob("form_d_*.csv"):
        for cell in pd.read_csv(path, usecols=["related_persons"])["related_persons"].dropna():
            for name, _ in _people(cell):
                first = _first_and_last(name)[0]
                if len(first) >= 3:
                    names.add(first.lower())
    return names


def find_quoted_names():
    """Find word pairs in demo_data that look like a person: a known first name, then a capitalized word.

    Catches names that scrub_quoted_names missed (no title next to them). Pairs that
    aren't people go in NOT_PEOPLE; people go in data/quoted_people.json. Returns (file, pair, text).
    """
    first_names = _first_names() - {"kenneth"}  # the sender's own name is shown on purpose
    pair = re.compile(r"\b([A-Z][a-z]+) ([A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)\b")
    hits = []
    files = [p for p in DEMO_DIR.rglob("*") if p.suffix in (".csv", ".json") and p.name != "favicons.json"]
    for path in sorted(files):
        text = path.read_text()
        for match in pair.finditer(text):
            if match.group(1).lower() in first_names and match.group(0) not in NOT_PEOPLE:
                hits.append((path.relative_to(DEMO_DIR), match.group(0), text[max(0, match.start() - 40):match.end() + 40]))
    return hits


def find_first_person():
    """Find any "we", "our", or "us" left in the demo emails. Returns (company, column, text) hits."""
    briefs = pd.read_csv(DEMO_DIR / "briefs.csv")
    hits = []
    for _, row in briefs.iterrows():
        for column in EMAIL_COLUMNS:
            text = row.get(column)
            if isinstance(text, str):
                for match in FIRST_PERSON.finditer(text):
                    hits.append((row["company_name"], column, text[max(0, match.start() - 40):match.end() + 40]))
    return hits


if __name__ == "__main__":
    make_demo_data()
    hits = find_leftover_names()
    print(f"demo_data/ rebuilt. Leftover name matches: {len(hits)}")
    for file_name, name, context in hits:
        print(f"  {file_name}: {name!r} in ...{context!r}...")
    print(f"Names in website quotes replaced with a role: {len(quote_renames)}")
    for name, role in sorted(set(quote_renames)):
        print(f"  {name!r} -> {role!r}")
    hits = find_quoted_names()
    print(f"Possible names left (first name + capitalized word): {len(hits)}")
    for path, name, context in hits:
        print(f"  {path}: {name!r} in ...{context!r}...")
    hits = find_first_person()
    print(f"'we', 'our', or 'us' left in demo emails: {len(hits)}")
    for company, column, context in hits:
        print(f"  {company} ({column}): ...{context!r}...")
