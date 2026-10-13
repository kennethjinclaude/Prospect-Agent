"""Who to contact: roles (never names) for each prospect, with why, what they care about, and an opening angle.

1. Code picks the roles from saved evidence, using fixed rules (free):
   company size band, whether it's regulated, and whether it's hiring for ML infra.
2. Claude Haiku writes the words for each role from the same saved evidence
   (about 0.15 cents per company). It can't change which roles were picked.
3. Results are saved in data/contacts/, and only rebuilt when the inputs change.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

import agent
import deep_dive
import fit
import qualify
import research

MODEL = "claude-haiku-5-5"
CONTACTS_DIR = Path(__file__).parent / "data" / "contacts"

# Segments that get a "Who to contact" section.
SEGMENTS = ["tier_1", "tier_2", "tier_3", "needs_review"]

# Round size used as a headcount estimate when the website doesn't say.
SMALL_ROUND, LARGE_ROUND = 25_000_000, 100_000_000

TEAM_SIZE = re.compile(r"\bteam of (\d[\d,]*)|(\d[\d,]*)\+?\s*(?:employees|people|engineers|teammates|team members)\b", re.IGNORECASE)
REGULATED_WORDS = re.compile(
    r"\b(health|healthcare|clinical|clinician|patient|medical|hospital|payer|bank|banks|banking|insurance|insurer|"
    r"lending|financial institutions?|legal|law firms?|attorneys?|government|federal)\b",
    re.IGNORECASE,
)
REGULATED_CATEGORIES = {"Healthcare AI", "Finance AI", "Legal AI"}
ML_INFRA_HIRING = re.compile(r"\b(inference|ml infra\w*|machine learning infra\w*|mlops|model serving|serving|gpu|ml platform|rl systems)\b",
                             re.IGNORECASE)
HIRING_WORDS = re.compile(r"\b(hir\w*|job|jobs|careers?|recruit\w*|open roles?|posting)\b", re.IGNORECASE)

# Prices, used to report cost.
HAIKU_INPUT, HAIKU_OUTPUT = 0.10 / 1e6, 0.50 / 1e6
usage_totals = {"input_tokens": 0, "output_tokens": 0}


# ---------------------------------------------------------------------------
# Step 1: the facts the rules use (free)
# ---------------------------------------------------------------------------

def size_band(company, website_text):
    """(band, explanation). Band is "under_50", "50_300", "300_plus", or "unknown"."""
    match = TEAM_SIZE.search(website_text or "")
    if match:
        count = int((match.group(1) or match.group(2)).replace(",", ""))
        band = "under_50" if count < 50 else "50_300" if count < 300 else "300_plus"
        return band, f'from their website: "{match.group(0)}"'
    amount = company.get("total_offering_amount")
    if pd.notna(amount):
        band = "under_50" if amount < SMALL_ROUND else "50_300" if amount < LARGE_ROUND else "300_plus"
        return band, f"estimated from a {agent._format_amount(amount)} round; check the LinkedIn company page for headcount"
    return "unknown", "no round size or team size found; check the LinkedIn company page for headcount"


def regulated(company, evidence):
    """(True/False, reason)."""
    for signal in evidence.get("enterprise_signals", []):
        if signal["type"] in ("regulated_enterprise_building_ai", "vertical_ai_for_finance_or_insurance"):
            return True, f'{signal["type"].replace("_", " ")}: "{signal["quote"]}"'
    if company.get("category") in REGULATED_CATEGORIES:
        return True, f'your category: {company["category"]}'
    if "Health" in str(company.get("industry_group", "")):
        return True, f'Form D industry: {company["industry_group"]}'
    match = REGULATED_WORDS.search(str(company.get("product_summary", "")))
    if match:
        return True, f'sells into a regulated industry ("{match.group(0)}" in what they sell)'
    return False, ""


def ml_infra_hiring(evidence, research_brief):
    """A quote showing they're hiring for inference or ML infra, or None."""
    for trigger in evidence.get("why_now", []):
        text = f'{trigger["trigger"]} {trigger["quote"]}'
        if "source" not in trigger and HIRING_WORDS.search(text) and ML_INFRA_HIRING.search(text):
            return f'"{trigger["quote"]}" (their website)'
    if research_brief:
        for fact in research_brief["brief"]["facts"]:
            if HIRING_WORDS.search(fact["fact"]) and ML_INFRA_HIRING.search(fact["fact"]):
                return f'{fact["fact"]} ({fact["source_url"]})'
    return None


# ---------------------------------------------------------------------------
# Step 2: the role rules (free)
# ---------------------------------------------------------------------------

def pick_roles(band, is_regulated, hiring_quote):
    """Apply the contact rules. Returns (roles, notes).

    Each role is {"slot": champion / economic_buyer / champion_and_buyer / blocker,
    "role": job title, "search": LinkedIn search words}.
    """
    notes = []
    if band in ("under_50", "unknown"):
        roles = [{"slot": "champion_and_buyer", "role": "CTO or technical founder", "search": "CTO"}]
        if band == "unknown":
            notes.append("Thin evidence: company size unknown, so this is the safest default for a startup.")
    elif band == "50_300":
        roles = [
            {"slot": "champion", "role": "Head of ML/AI or ML infrastructure lead", "search": "Head of ML"},
            {"slot": "economic_buyer", "role": "VP Engineering", "search": "VP Engineering"},
        ]
    else:
        roles = [
            {"slot": "champion", "role": "Director of ML Platform", "search": "ML Platform"},
            {"slot": "economic_buyer", "role": "VP Engineering or CTO", "search": "VP Engineering"},
            {"slot": "blocker", "role": "Security lead", "search": "Security"},
            {"slot": "blocker", "role": "Procurement", "search": "Procurement"},
        ]
    if is_regulated:
        roles = [r for r in roles if r["role"] != "Security lead"]
        roles.append({"slot": "blocker", "role": "CISO or security lead", "search": "CISO"})
    if hiring_quote:
        notes.append(f"Best entry point: the team that owns this posting: {hiring_quote}")
    return roles, notes


# ---------------------------------------------------------------------------
# Step 3: the words for each role (Haiku), checks, and cache
# ---------------------------------------------------------------------------

TEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "why_this_role": {"type": "string"},
                    "cares_about": {"type": "string"},
                    "opening_angle": {"type": "string"},
                },
                "required": ["role", "why_this_role", "cares_about", "opening_angle"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["roles"],
    "additionalProperties": False,
}

TEXT_INSTRUCTIONS = f"""You help a Baseten sales rep decide who to contact at a company. The roles are already chosen; for each one, in the same order and with the role written exactly as given, write:
- why_this_role: one sentence on why this role matters here, from the evidence.
- cares_about: one short phrase on what this person cares about.
- opening_angle: one line the rep could open with, tailored to this role and this company.

Rules:
- Use only the evidence given. Mark anything inferred with "likely".
- Never name a person. Roles only.
- Any Baseten claim must come from these facts: {agent.BASETEN_FACTS}
- Any benefit must directly restate one of those facts (for example "turns ML models into production APIs" can become "so you can run models as production APIs"). Never promise outcomes the facts don't state, such as lower costs, faster shipping, or freeing up the team.
- No em dashes. Plain, conversational words."""


def _cache_path(filing_url, folder=CONTACTS_DIR):
    return research._cache_file(filing_url, folder)


def load_contacts(filing_url, folder=CONTACTS_DIR):
    path = _cache_path(filing_url, folder)
    return json.loads(path.read_text()) if path.exists() else None


def _inputs(company):
    """Everything the section depends on, from saved files only."""
    evidence = {c: json.loads(company[c]) if isinstance(company.get(c), str) else [] for c in fit.EVIDENCE_COLUMNS}
    snapshot = fit.website_text_path(company)
    website_text = snapshot.read_text() if snapshot.exists() else ""
    brief = research.load_brief(company["filing_url"])
    band, band_note = size_band(company, website_text)
    is_regulated, regulated_reason = regulated(company, evidence)
    hiring = ml_infra_hiring(evidence, brief)
    roles, notes = pick_roles(band, is_regulated, hiring)
    facts = {
        "company": company["company_name"],
        "segment": company.get("account_segment"),
        "what_they_sell": company.get("product_summary"),
        "fit_reason": company.get("fit_reason"),
        "size": f"{band.replace('_', ' ')} ({band_note})",
        "regulated": regulated_reason or "no",
        "ml_infra_hiring": hiring or "none found",
        "research_hook": brief["brief"]["hook"] if brief else None,
    }
    return {"band": band, "band_note": band_note, "regulated": regulated_reason, "hiring": hiring,
            "roles": roles, "notes": notes, "facts": facts}


def _write_text(inputs):
    """Haiku writes why / cares about / opening angle for each chosen role."""
    response = qualify.get_client().messages.create(
        model=MODEL,
        max_tokens=3000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": TEXT_SCHEMA}},
        system=TEXT_INSTRUCTIONS,
        messages=[{"role": "user", "content": (
            f"<evidence>\n{json.dumps(inputs['facts'], indent=1)}\n</evidence>\n\n"
            f"Roles, in order: {json.dumps([r['role'] for r in inputs['roles']])}"
        )}],
    )
    usage_totals["input_tokens"] += response.usage.input_tokens
    usage_totals["output_tokens"] += response.usage.output_tokens
    return json.loads(next(b.text for b in response.content if b.type == "text"))["roles"]


def build_contacts(company):
    """The "Who to contact" section for one company: from cache if the inputs haven't changed."""
    inputs = _inputs(company)
    # The fingerprint covers the inputs and the instructions, so changing either rebuilds the text.
    fingerprint = hashlib.sha256(json.dumps([inputs, TEXT_INSTRUCTIONS], sort_keys=True, default=str).encode()).hexdigest()[:16]
    saved = load_contacts(company["filing_url"])
    if saved and saved["fingerprint"] == fingerprint:
        return saved

    written = _write_text(inputs)
    warnings = []
    # The roles must be the ones the rules picked, in order.
    if [w["role"] for w in written] != [r["role"] for r in inputs["roles"]]:
        warnings.append("The written roles didn't match the chosen roles, so the chosen ones are shown")
    by_role = {w["role"]: w for w in written}
    cards = []
    for role in inputs["roles"]:
        text = by_role.get(role["role"], {})
        cards.append({
            **role,
            "why_this_role": text.get("why_this_role", ""),
            "cares_about": text.get("cares_about", ""),
            "opening_angle": text.get("opening_angle", ""),
            "linkedin": deep_dive.people_search(company["company_name"], role["search"], company.get("company_display_name")),
        })
    # No personal names: flag anyone from the company's Form D.
    all_text = json.dumps(cards)
    if isinstance(company.get("related_persons"), str):
        for person in company["related_persons"].split("; "):
            name = person.partition(" (")[0]
            if name and name.lower() in all_text.lower():
                warnings.append(f"Mentions a person by name ({name}); this section should give roles only")

    result = {
        "filing_url": company["filing_url"],
        "company_name": company["company_name"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fingerprint": fingerprint,
        "size_band": inputs["band"],
        "size_note": inputs["band_note"],
        "regulated": inputs["regulated"],
        "notes": inputs["notes"],
        "cards": cards,
        "company_page": deep_dive.links(company["company_name"], company.get("website"),
                                        company.get("company_display_name"))["Company"][0][1],
        "warnings": warnings,
    }
    path = _cache_path(company["filing_url"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


def companies():
    """Every Form D and target company in a contact segment, with what the rules need."""
    table = research.companies_table()
    briefs = pd.read_csv(agent.BRIEFS_FILE)[["filing_url", "company_display_name"]]
    fit_rows = pd.read_csv(fit.FIT_FILE).drop(columns=["company_name", "website", "account_segment", "fit_grade"])
    table = table.merge(fit_rows, on="filing_url", how="left").merge(briefs, on="filing_url", how="left")
    return table[table["account_segment"].isin(SEGMENTS)]


def estimated_cost():
    return usage_totals["input_tokens"] * HAIKU_INPUT + usage_totals["output_tokens"] * HAIKU_OUTPUT
