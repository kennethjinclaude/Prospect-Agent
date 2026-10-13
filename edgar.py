"""Fetch recent Form D filings (private fundraising notices) from SEC EDGAR."""

import os
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Baseten ICP (ideal customer profile). Change these to widen or narrow the filter.
SETTINGS = {
    # Industry groups as written in Form D filings.
    "industry_groups": [
        "Computers",
        "Telecommunications",
        "Other Technology",
        "Other Health Care",  # AI-native healthcare companies are Baseten customers too
    ],
    "us_only": True,
    "min_offering_amount": 10_000_000,
    "max_offering_amount": None,  # None means no maximum
    # How many days of filings to search (30 or 60). The app shows the last
    # 30 days and days 31 to 60 in separate tabs.
    "lookback_days": 60,
}

# The 50 states plus DC. EDGAR uses other codes (like A1, X0, XX) for places
# outside the US, so we list the US ones explicitly.
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY",
}

# Load SEC_USER_EMAIL from the .env file that sits next to this script.
load_dotenv(Path(__file__).parent / ".env")

ARCHIVES_URL = "https://www.sec.gov/Archives/edgar"

# Fetched filings are saved here so we only download them once per day.
DATA_DIR = Path(__file__).parent / "data"
# Each past day's filings are also saved on their own: a past day never changes,
# so a longer lookback only downloads the days not already on disk.
DAY_CACHE_DIR = DATA_DIR / "form_d_days"

# The columns each filing row has. If a cached CSV is missing any of these
# (because it was saved by an older version of this code), it's refetched.
COLUMNS = [
    "company_name", "city", "state", "filing_date", "industry_group",
    "total_offering_amount", "amount_sold", "related_persons", "filing_url",
]

# The SEC allows at most 10 requests per second. Waiting 0.12s between
# requests keeps us at about 8 per second, safely under the limit.
SECONDS_BETWEEN_REQUESTS = 0.12

_last_request_time = 0.0


def _make_session():
    """Create a web session that identifies us to the SEC, as they require."""
    email = os.getenv("SEC_USER_EMAIL")
    if not email:
        raise RuntimeError("SEC_USER_EMAIL is missing from .env")
    session = requests.Session()
    session.headers["User-Agent"] = f"Kenneth Jin {email}"
    return session


def _get(session, url):
    """Download a URL, pausing first if needed to respect the SEC rate limit."""
    global _last_request_time
    wait = SECONDS_BETWEEN_REQUESTS - (time.monotonic() - _last_request_time)
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.monotonic()
    return session.get(url, timeout=30)


def _list_form_d_filings(session, day):
    """Return (company name, filing path) for every new Form D filed on one day.

    Uses EDGAR's daily form index. Weekends, holidays, and today (before the
    index is published) have no file, so those days return an empty list.
    """
    quarter = (day.month - 1) // 3 + 1
    url = f"{ARCHIVES_URL}/daily-index/{day.year}/QTR{quarter}/form.{day:%Y%m%d}.idx"
    response = _get(session, url)
    # EDGAR answers a missing file with 403 (Forbidden), not just 404.
    if response.status_code in (403, 404):
        return []
    response.raise_for_status()

    filings = []
    for line in response.text.splitlines():
        parts = line.split()
        # Each row is: form type, company name, CIK, date filed, file path.
        # "D" is a new filing; "D/A" (amendment) is skipped because it updates
        # an older round rather than announcing a new one.
        if len(parts) >= 5 and parts[0] == "D":
            filings.append(parts[-1])
    return filings


def _parse_amount(text):
    """Turn an amount like "1500000" into a number. "Indefinite" becomes None."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_filing(session, file_path, filing_date):
    """Download one filing's primary_doc.xml and pull out the fields we need.

    Returns None for pooled investment funds (VC/PE/hedge funds), since we
    only want operating companies.
    """
    # file_path looks like "edgar/data/2157448/0002157448-26-000001.txt".
    # The filing's folder is named after the accession number without dashes.
    cik = file_path.split("/")[2]
    accession = file_path.split("/")[3].removesuffix(".txt")
    folder = f"{ARCHIVES_URL}/data/{cik}/{accession.replace('-', '')}"

    response = _get(session, f"{folder}/primary_doc.xml")
    if response.status_code != 200:
        return None
    root = ET.fromstring(response.content)

    industry = root.findtext("offeringData/industryGroup/industryGroupType")
    if industry == "Pooled Investment Fund":
        return None

    return {
        "company_name": root.findtext("primaryIssuer/entityName"),
        "city": root.findtext("primaryIssuer/issuerAddress/city"),
        "state": root.findtext("primaryIssuer/issuerAddress/stateOrCountry"),
        "filing_date": filing_date,
        "industry_group": industry,
        "total_offering_amount": _parse_amount(
            root.findtext("offeringData/offeringSalesAmounts/totalOfferingAmount")
        ),
        "amount_sold": _parse_amount(
            root.findtext("offeringData/offeringSalesAmounts/totalAmountSold")
        ),
        "related_persons": _parse_related_persons(root),
        "filing_url": f"{folder}/{accession}-index.htm",
    }


def _parse_related_persons(root):
    """List the filing's related persons as one readable string.

    Example: "JANE DOE (Executive Officer, Director); JOHN ROE (Director)".
    A CSV cell can't hold a list, so we join everyone with "; ".
    """
    people = []
    for person in root.findall("relatedPersonsList/relatedPersonInfo"):
        name_parts = [
            person.findtext("relatedPersonName/firstName"),
            person.findtext("relatedPersonName/middleName"),
            person.findtext("relatedPersonName/lastName"),
        ]
        # Some filers type "n/a" for a missing middle name; leave those out.
        name = " ".join(p for p in name_parts if p and p.lower() not in ("n/a", "na", "none"))
        roles = [r.text for r in person.findall("relatedPersonRelationshipList/relationship")]
        people.append(f"{name} ({', '.join(roles)})")
    return "; ".join(people)


def fetch_form_d_filings(days=None):
    """Return a DataFrame of new Form D filings from the last `days` days.

    If `days` isn't given, uses SETTINGS["lookback_days"]. Each past day is
    cached in data/form_d_days/, so only days not seen before are downloaded.
    Today is never cached, since its filings may still be coming in.
    """
    days = days or SETTINGS["lookback_days"]
    cache_file = DATA_DIR / f"form_d_{date.today()}_{days}d.csv"
    if cache_file.exists():
        cached = pd.read_csv(cache_file)
        if set(COLUMNS).issubset(cached.columns):
            print(f"Loading cached filings from {cache_file.name}")
            return cached
        print("Cached filings are missing newer columns, refetching")

    session = None
    tables = []
    for offset in range(days):
        day = date.today() - timedelta(days=offset)
        day_file = DAY_CACHE_DIR / f"{day}.csv"
        if day_file.exists():
            day_rows = pd.read_csv(day_file)
            if set(COLUMNS).issubset(day_rows.columns):
                tables.append(day_rows)
                continue
        session = session or _make_session()
        day_rows = _download_day(session, day)
        if day < date.today():
            DAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            day_rows.to_csv(day_file, index=False)
        tables.append(day_rows)

    df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=COLUMNS)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(cache_file, index=False)
    return df


def _download_day(session, day):
    """Download and parse every new Form D filed on one day."""
    file_paths = _list_form_d_filings(session, day)
    print(f"{day}: {len(file_paths)} Form D filings")
    rows = []
    for file_path in file_paths:
        try:
            row = _parse_filing(session, file_path, day)
        except (requests.RequestException, ET.ParseError) as error:
            # One broken filing shouldn't stop the whole run.
            print(f"  Skipped {file_path}: {error}")
            continue
        if row:
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def seed_day_cache(full_download_csv, last_day, days):
    """Split an earlier multi-day download into per-day cache files.

    Only for a download known to be complete: a day with no rows really had
    no filings (for example, a weekend).
    """
    df = pd.read_csv(full_download_csv)
    DAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for offset in range(days):
        day = last_day - timedelta(days=offset)
        df[df["filing_date"] == str(day)].to_csv(DAY_CACHE_DIR / f"{day}.csv", index=False)


def filter_by_icp(df, settings=SETTINGS):
    """Keep only the companies that match the ICP settings.

    Companies with no stated offering amount (e.g. "Indefinite") are kept,
    since we can't rule them out, and marked with amount_unknown = True.
    """
    df = df.copy()
    df["amount_unknown"] = df["total_offering_amount"].isna()

    keep = df["industry_group"].isin(settings["industry_groups"])

    if settings["us_only"]:
        keep &= df["state"].isin(US_STATES)

    amount = df["total_offering_amount"]
    amount_ok = pd.Series(True, index=df.index)
    if settings["min_offering_amount"] is not None:
        amount_ok &= amount >= settings["min_offering_amount"]
    if settings["max_offering_amount"] is not None:
        amount_ok &= amount <= settings["max_offering_amount"]
    keep &= amount_ok | df["amount_unknown"]
    matches = df[keep]

    # A company can file more than one Form D. Keep one row per company name:
    # the larger round (an unknown amount counts as smallest). latest_filing_date
    # records the company's most recent filing, which decides its tab in the app.
    latest = matches.groupby("company_name")["filing_date"].transform("max")
    matches = matches.assign(latest_filing_date=latest)
    largest_first = matches.sort_values("total_offering_amount", ascending=False, na_position="last")
    deduped = largest_first.drop_duplicates("company_name", keep="first")
    return matches[matches.index.isin(deduped.index)].reset_index(drop=True)


if __name__ == "__main__":
    df = fetch_form_d_filings()
    print(f"\nFound {len(df)} filings (pooled investment funds excluded)")

    matches = filter_by_icp(df)
    unknown = matches["amount_unknown"].sum()
    print(f"{len(matches)} match the ICP ({unknown} with unknown offering amount)\n")
    print(matches.drop(columns="filing_url").head(10).to_string())
