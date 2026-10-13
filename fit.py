"""Segment each AI-native company with Baseten's 3-tier ICP and grade its fit.

How a company is scored, in two parts:
1. Claude reads the website (with icp.md for definitions) and collects evidence:
   labeled facts, each with an exact quote. Code checks every quote is really
   on the website and drops any that aren't.
2. Code makes every decision from that verified evidence, using fixed rules:
   the stage (from the raise size), the account segment (SEGMENT_RULES, in
   order, first match wins), and the A/B/C fit grade. The same evidence
   always gives the same result, and each row records which rule decided it.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import anthropic
import pandas as pd
import requests

import agent
import qualify

MODEL = "claude-sonnet-5-5"
ROOT = Path(__file__).parent
ICP_FILE = ROOT / "icp.md"
DATA_DIR = ROOT / "data"
FIT_FILE = DATA_DIR / "fit.csv"
# Website text used for scoring is saved here, so re-scoring reads the same text.
WEBSITE_TEXT_DIR = DATA_DIR / "website_text"

# Which qualification verdicts get scored.
VERDICTS_TO_SCORE = ["yes", "maybe", "insufficient_info"]

# How many new companies to score per run. Lower it (e.g. to 5) to test cheaply.
LIMIT = 200

# Stage from raise size (total offering amount on the Form D).
# Set a little under $25M, since rounds often close just below a round number
# (for example $24,999,998).
SERIES_B_MIN_RAISE = 24_500_000  # $24.5M+ suggests Series B or later
SERIES_A_MIN_RAISE = 10_000_000  # $10M to $24.5M suggests Series A

WORKLOAD_TYPES = [
    "llm_text", "voice_speech", "computer_vision", "image_video_generation",
    "rag_search", "agents", "multimodal", "other",
]
MODEL_SIGNAL_TYPES = ["custom_or_open", "closed_api_at_scale", "self_managed_gpus", "competitor_user"]
ENTERPRISE_TYPES = ["large_saas_adding_ai", "regulated_enterprise_building_ai", "vertical_ai_for_finance_or_insurance"]
DEPRIORITIZE_REASONS = ["frontier_lab", "experimenting_low_volume", "happy_on_closed_apis"]
# Website evidence of stage, used only when the Form D doesn't state the raise size.
STAGE_EVIDENCE_TYPES = ["scale_up", "early_stage"]
# What a scale_up quote shows. Only a real number or named customers count as scale.
SCALE_KINDS = ["number", "named_customers", "other"]
# Signs a company may compete with Baseten or build its own serving stack.
IN_HOUSE_TYPES = ["sells_raw_model_inference", "runs_own_serving_infrastructure"]
# Evidence saved before the rule was narrowed used these names.
LEGACY_IN_HOUSE_TYPES = {
    "sells_own_model_api": "sells_raw_model_inference",
    "runs_own_model_infra_at_scale": "runs_own_serving_infrastructure",
}
# Your review: these sell a product through an API, not raw model inference,
# so the in-house rule doesn't apply to them.
NOT_IN_HOUSE = {
    "Fireflies.ai": "product API (transcribe your call files), not raw model inference",
    "Photoroom": "product API (photo editing in your app), not raw model inference",
    "Synthesia": "product API (create AI videos), not raw model inference",
}

# A tier 1 or 2 company whose AI-native verdict was one of these needs a human check first.
UNSURE_VERDICTS = ["maybe", "insufficient_info"]

# Model signals that earn grade A. Closed APIs at scale is a cost conversation (grade B).
GRADE_A_MODEL_SIGNALS = {"custom_or_open", "self_managed_gpus", "competitor_user"}

# Manual segment decisions, by company name as in the filing, with the reason shown in the app.
SEGMENT_OVERRIDES = {
    "Concorda, Inc.": ("needs_review", "listing co-founders is not evidence of team size"),
    "Sierra": ("tier_1", 'names enterprise customers on its site, e.g. "How Rocket Mortgage is reimagining '
                         'the journey home with AI." (https://sierra.ai/customers)'),
}

# Words that show a "finance or insurance" vendor signal is really about healthcare
# (for example, selling to health insurers). icp.md: AI vendors are classified by
# their own vertical, so healthcare AI vendors are tier 1 or tier 3 by stage.
HEALTHCARE_WORDS = re.compile(
    r"\b(health|healthcare|medical|clinical|clinician|patient|payer|payers|provider|providers|hospital|claims?)\b",
    re.IGNORECASE,
)

# Who to target in each segment, from icp.md.
PERSONAS = {
    "tier_1": "Champion: Head of ML or ML infra lead. Buyer: CTO / VP Engineering, often a technical co-founder.",
    "tier_2": "VP Engineering or Head of AI Platform, plus security and procurement.",
    "tier_3": "Technical co-founder. Mostly self-serve: reach out again when usage spikes or they raise.",
    "needs_review": "Check the company's stage by hand first.",
    "existing_customer": "Existing Baseten customer: route to account management.",
    "deprioritize": None,
}

# Published prices, used only to estimate what a run cost.
INPUT_PRICE_PER_TOKEN = 2.00 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 10.00 / 1_000_000

usage_totals = {"input_tokens": 0, "output_tokens": 0}


def _quoted_list(label_name, label_schema):
    """Schema for a list of {label, quote} items, e.g. [{"signal": ..., "quote": ...}]."""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {label_name: label_schema, "quote": {"type": "string"}},
            "required": [label_name, "quote"],
            "additionalProperties": False,
        },
    }


EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "workload_type": {"type": "array", "items": {"type": "string", "enum": WORKLOAD_TYPES}},
        "workload_quote": {"type": "string"},
        "ai_core_evidence": _quoted_list("point", {"type": "string"}),
        "model_signals": _quoted_list("type", {"type": "string", "enum": MODEL_SIGNAL_TYPES}),
        "production_signals": _quoted_list("signal", {"type": "string"}),
        "why_now": _quoted_list("trigger", {"type": "string"}),
        "enterprise_signals": _quoted_list("type", {"type": "string", "enum": ENTERPRISE_TYPES}),
        "deprioritize_signals": _quoted_list("reason", {"type": "string", "enum": DEPRIORITIZE_REASONS}),
        "stage_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stage": {"type": "string", "enum": STAGE_EVIDENCE_TYPES},
                    "kind": {"type": "string", "enum": SCALE_KINDS},
                    "quote": {"type": "string"},
                },
                "required": ["stage", "kind", "quote"],
                "additionalProperties": False,
            },
        },
        "in_house_signals": _quoted_list("type", {"type": "string", "enum": IN_HOUSE_TYPES}),
        "fit_reason": {"type": "string"},
    },
    "required": [
        "workload_type", "workload_quote", "ai_core_evidence", "model_signals", "production_signals",
        "why_now", "enterprise_signals", "deprioritize_signals", "stage_evidence", "in_house_signals", "fit_reason",
    ],
    "additionalProperties": False,
}


def build_instructions():
    """Claude's instructions, with the current icp.md pasted in."""
    return f"""You are collecting evidence about a company for Baseten's sales team. The ideal customer profile (ICP) below defines the kinds of evidence that matter. Don't decide the company's tier: code does that from your evidence.

<icp>
{ICP_FILE.read_text()}
</icp>

Use only the website text given. Every quote must be copied exactly, word for word, from the website text. Only include evidence the text positively states. Missing information is not evidence of anything, and neither are demo captions, sample data, or marketing taglines alone.

Return:
- workload_type: one or more of {", ".join(WORKLOAD_TYPES)}. workload_quote: an exact quote showing it.
- ai_core_evidence: quotes showing that AI is the core of what the company sells (users' actions trigger model calls), not a side feature. Empty if AI is minor or absent.
- model_signals, each a "type" and "quote":
    "custom_or_open": they train, fine-tune, or self-host models, or name open models (Llama, Qwen, DeepSeek, Mistral, Whisper, etc.);
    "closed_api_at_scale": they rely on OpenAI, Anthropic, or similar closed APIs at real volume;
    "self_managed_gpus": they run their own GPUs, vLLM, or Kubernetes serving;
    "competitor_user": they use Together AI, Fireworks, Modal, Replicate, or Anyscale.
  "Proprietary AI" or "our AI platform" alone is not a model signal.
- production_signals: evidence of real users or traffic (customer counts, named customers, usage numbers, a live product people use), each a "signal" label and "quote".
- why_now: triggers beyond the recent raise (which is added separately), such as launching voice, real-time, or agent features, or hiring ML infra engineers. Each a "trigger" label and "quote".
- enterprise_signals, each a "type" and "quote":
    "large_saas_adding_ai": a large, established SaaS company adding AI features (like Notion);
    "regulated_enterprise_building_ai": a financial services firm, insurer, healthcare system, or legal firm building its own AI;
    "vertical_ai_for_finance_or_insurance": an AI company whose own product is for finance or insurance (like Rogo selling to banks).
  Classify AI vendors by their own vertical, not their customers' industry: healthcare and legal AI vendors are not enterprise signals, even when they sell to insurers or banks.
- deprioritize_signals, each a "reason" and "quote", only for a positive statement that:
    "frontier_lab": they are a frontier AI lab building their own serving stack;
    "experimenting_low_volume": they say they are only experimenting with AI or at low volume;
    "happy_on_closed_apis": they are built on closed APIs and state no plans for their own models.
- stage_evidence, each a "stage", a "kind", and a "quote", only where the website clearly shows the company's size or stage:
    "scale_up": only with a specific number of customers, users, deployments, or revenue (kind "number", e.g. "1M+ developers", "deployed in 130+ health organizations"), or customers named by name (kind "named_customers", e.g. "used by United Airlines and Cox"). Taglines like "trusted by leading enterprises", security certifications like "SOC 2", and funding amounts are not scale evidence; leave them out.
    "early_stage": a waitlist, beta or early access, pre-launch, a pre-seed or seed round, or a very small team (kind "other").
  A list of founders or leaders is not evidence of team size.
- in_house_signals, each a "type" and "quote", only for a positive statement that:
    "sells_raw_model_inference": they sell raw model inference to developers, priced per call or per token (for example a speech-to-text, text-to-speech, or embeddings API);
    "runs_own_serving_infrastructure": they clearly run their own model-serving infrastructure (their own inference stack or GPU fleet for serving).
  An API that gives access to their product (for example "use our API to transcribe your calls" or "build our editor into your app") is not raw model inference, and training their own models is not evidence on its own. These companies may compete with Baseten or build in-house. Empty if the website doesn't say so.
  Empty if the website doesn't clearly show it.
- fit_reason: one sentence summarizing the evidence."""


def _verified(quote, website_text):
    """True if the quote really appears in the website text (same check as qualify.py)."""
    return bool(quote) and qualify._normalize(quote) in qualify._normalize(website_text)


def _keep_verified(items, website_text, label, notes):
    """Keep only items whose quote is on the website, and note how many were dropped."""
    kept = [item for item in items if _verified(item["quote"], website_text)]
    if len(kept) < len(items):
        notes.append(f"{len(items) - len(kept)} {label} dropped: quote not found on website")
    return kept


def raise_amount(company):
    """The round size from the Form D (total offering amount), or None if not stated.

    The amount sold so far isn't used as a fallback: for an "Indefinite"
    offering it's only a partial number, so it says little about the stage.
    """
    amount = company.get("total_offering_amount")
    return amount if pd.notna(amount) else None


def _raise_trigger(company, amount, filing_url):
    """The "recent raise" why-now trigger. Its source is the Form D, not the website.

    Companies added by hand have no Form D: the trigger says so, and shows the
    round size only if the user entered one.
    """
    if filing_url.startswith("manual:"):
        size = (f"round size entered manually: {agent._format_amount(amount)}" if amount is not None
                else "no Form D, round size unknown")
        return {"trigger": "Added manually", "quote": size, "source": "Manual entry"}
    if amount is None:
        size = f"round size not stated ({agent._format_amount(company.get('amount_sold'))} sold so far)"
    else:
        size = f"{agent._format_amount(amount)} round"
    return {"trigger": "Recent raise", "quote": f"Form D filed {company.get('filing_date')}, {size}", "source": "Form D"}


def estimate_stage(amount):
    """Guess the funding stage from the raise size, using the thresholds above."""
    if amount is None:
        return "unknown"
    if amount >= SERIES_B_MIN_RAISE:
        return "Series B or later"
    if amount >= SERIES_A_MIN_RAISE:
        return "Series A"
    return "Seed or earlier"


def segment_company(company_name, evidence, stage, amount, verdict=None):
    """Decide a company's segment. Returns (segment, rule that decided it).

    Existing customers and manual overrides come first; then SEGMENT_RULES;
    then R10, which sends a tier 1 or 2 company with an unsure verdict to review.
    """
    customer = agent.existing_customer_match(company_name)
    if customer:
        return "existing_customer", f"R0 existing_customer: matches '{customer}' on Baseten's customers page"
    if company_name in SEGMENT_OVERRIDES:
        segment, reason = SEGMENT_OVERRIDES[company_name]
        return segment, f"Override {segment}: {reason}"

    signals = [{**s, "type": LEGACY_IN_HOUSE_TYPES.get(s["type"], s["type"])} for s in evidence.get("in_house_signals", [])]
    if company_name in NOT_IN_HOUSE:
        signals = []
    segment, rule = decide_segment({**evidence, "in_house_signals": signals}, stage, amount)
    if company_name in NOT_IN_HOUSE:
        rule += f" (in-house rule not applied: {NOT_IN_HOUSE[company_name]})"

    if segment in ("tier_1", "tier_2") and verdict in UNSURE_VERDICTS:
        return "needs_review", (f'R10 needs_review: would be {segment}, but the AI-native verdict was '
                                f'"{verdict}", so it needs a human check first')
    return segment, rule


def _vendor_enterprise_signals(evidence):
    """Enterprise signals, minus finance/insurance-vendor ones whose quote is about healthcare.

    Returns (signals kept, whether any healthcare vendor signal was dropped).
    """
    kept, dropped = [], False
    for signal in evidence["enterprise_signals"]:
        if signal["type"] == "vertical_ai_for_finance_or_insurance" and HEALTHCARE_WORDS.search(signal["quote"]):
            dropped = True
        else:
            kept.append(signal)
    return kept, dropped


def decide_segment(evidence, stage, amount):
    """Apply SEGMENT_RULES in order. Returns (segment, rule that decided it).

    Every rule uses only verified evidence and the raise size, so the same
    inputs always give the same answer.
    """
    segment, rule = _decide_segment(evidence, stage, amount)
    if _vendor_enterprise_signals(evidence)[1] and segment != "tier_2":
        rule += " (healthcare AI vendor: classified by its own vertical, not its customers')"
    return segment, rule


def _decide_segment(evidence, stage, amount):
    raise_text = agent._format_amount(amount)
    if evidence["deprioritize_signals"]:
        reasons = ", ".join(sorted({s["reason"].replace("_", " ") for s in evidence["deprioritize_signals"]}))
        return "deprioritize", f"R1 deprioritize: the website says {reasons}"
    if not evidence["ai_core_evidence"]:
        return "deprioritize", "R2 deprioritize: no website evidence that AI is the core product"
    if evidence.get("in_house_signals"):
        first = evidence["in_house_signals"][0]
        return "needs_review", (f'R2b needs_review: possible competitor / builds in-house '
                                f'({first["type"].replace("_", " ")}): "{first["quote"]}"')
    enterprise_signals, _ = _vendor_enterprise_signals(evidence)
    if enterprise_signals:
        types = ", ".join(sorted({s["type"].replace("_", " ") for s in enterprise_signals}))
        return "tier_2", f"R3 tier_2: {types}"
    if stage == "Series B or later" and evidence["production_signals"]:
        return "tier_1", f"R4 tier_1: {raise_text} raise suggests Series B or later, with production evidence"
    if stage == "Series B or later":
        return "tier_3", f"R5 tier_3: {raise_text} raise suggests Series B or later, but no production evidence on the website"
    if amount is None:
        return _segment_from_website_stage(evidence)
    return "tier_3", f"R6 tier_3: {raise_text} raise suggests {stage}"


SCALE_NUMBER = re.compile(r"\d[\d,.]*\s*(?:\+|%|k\b|m\b|b\b|million|billion|thousand)|\b\d{2,}[\d,]*\b", re.IGNORECASE)
NOT_SCALE = re.compile(r"\b(soc\s*\d|iso\s*\d|series\s+[a-z]\b|rais(e|ed|ing)|valuation|funding)", re.IGNORECASE)


def counts_as_scale(item):
    """True if a scale_up quote shows a real count of customers/users/deployments/revenue, or named customers.

    Numbers are checked by code: "SOC 2" or "$70M raised" don't count.
    Named customers are taken from Claude's label (code can't tell a customer name from other words).
    """
    if item.get("stage") != "scale_up":
        return False
    if item.get("kind") == "named_customers":
        return True
    quote = item["quote"]
    return bool(SCALE_NUMBER.search(quote)) and not NOT_SCALE.search(quote)


def _segment_from_website_stage(evidence):
    """For a Form D with no stated raise: use the website's stage evidence, or ask for review."""
    scale = [s for s in evidence["stage_evidence"] if counts_as_scale(s)]
    early = [s for s in evidence["stage_evidence"] if s["stage"] == "early_stage"]
    if scale and not early and evidence["production_signals"]:
        return "tier_1", f'R7 tier_1: no stated raise, but the website shows scale: "{scale[0]["quote"]}"'
    if early and not scale:
        return "tier_3", f'R8 tier_3: no stated raise, and the website shows an early stage: "{early[0]["quote"]}"'
    return "needs_review", "R9 needs_review: raise size not stated, and the website doesn't clearly show the stage"


# The rules above, in order, for display and documentation.
SEGMENT_RULES = [
    "R0 existing_customer: the company is on Baseten's customers page (checked before everything else)",
    "Override: a manual decision in SEGMENT_OVERRIDES",
    "R1 deprioritize: the website positively states a deprioritize reason from icp.md",
    "R2 deprioritize: no website evidence that AI is the core product",
    "R2b needs_review: possible competitor / builds in-house (sells raw model inference to developers, or runs its own serving infrastructure)",
    "R3 tier_2: an enterprise or regulated-industry signal (healthcare AI vendors excluded)",
    f"R4 tier_1: raise of ${SERIES_B_MIN_RAISE / 1e6:g}M+ (Series B or later) plus production evidence",
    f"R5 tier_3: raise of ${SERIES_B_MIN_RAISE / 1e6:g}M+ but no production evidence",
    f"R6 tier_3: raise under ${SERIES_B_MIN_RAISE / 1e6:g}M (Series A or earlier)",
    "R7 tier_1: no stated raise, but the website shows scale (a real count or named customers, quoted) plus production evidence",
    "R8 tier_3: no stated raise, and the website shows an early stage (quoted)",
    "R9 needs_review: no stated raise, and the website doesn't clearly show the stage",
    "R10 needs_review: would be tier 1 or 2, but the AI-native verdict was maybe or insufficient_info",
]


def decide_grade(evidence):
    """A/B/C signal strength within the segment, from verified evidence only."""
    has_a_model_signal = any(s["type"] in GRADE_A_MODEL_SIGNALS for s in evidence["model_signals"])
    if has_a_model_signal and evidence["production_signals"]:
        return "A"
    if evidence["production_signals"]:
        return "B"
    return "C"


def website_text_path(company):
    """Where a company's website snapshot is saved: by filing number, or "manual_<domain>"."""
    accession = re.search(r"/(\d{10}-\d{2}-\d{6})-index\.htm", company["filing_url"])
    name = accession.group(1) if accession else re.sub(r"[^A-Za-z0-9.-]+", "_", company["filing_url"])
    return WEBSITE_TEXT_DIR / f"{name}.txt"


def website_text_for(company):
    """The company's website text, from the saved snapshot if there is one."""
    snapshot = website_text_path(company)
    if snapshot.exists():
        return snapshot.read_text()
    text = qualify.fetch_website_text(company["website"])
    WEBSITE_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(text)
    return text


def score_company(company):
    """Segment and grade one company, and return a result row. Never raises."""
    result = {
        "filing_url": company["filing_url"],
        "company_name": company["company_name"],
        "website": company["website"],
        "account_segment": None,
        "segment_rule": None,
        "stage_estimate": None,
        "fit_grade": None,
        "workload_type": None,
        "model_signal_types": None,
        "model_signals": None,
        "production_signals": None,
        "why_now": None,
        "enterprise_signals": None,
        "deprioritize_signals": None,
        "ai_core_evidence": None,
        "stage_evidence": None,
        "in_house_signals": None,
        "fit_reason": None,
        "persona": None,
        "note": None,
        "scored_at": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        website_text = website_text_for(company)
    except requests.RequestException as error:
        result["note"] = f"Website could not be fetched: {error}"
        return result
    if len(website_text) < qualify.MIN_WEBSITE_CHARS:
        result["note"] = "Website had too little text to score"
        return result

    try:
        response = qualify.get_client().beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": EVIDENCE_SCHEMA}},
            # If a safety check declines the request, Anthropic retries it on
            # another model where it can, instead of just returning a refusal.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=build_instructions(),
            messages=[{
                "role": "user",
                "content": f"Company: {company['company_name']}\n\n<website_text>\n{website_text}\n</website_text>",
            }],
        )
    except anthropic.APIError as error:
        result["note"] = f"API call failed: {error}"
        return result
    usage_totals["input_tokens"] += response.usage.input_tokens
    usage_totals["output_tokens"] += response.usage.output_tokens
    if response.stop_reason == "refusal":
        result["note"] = "Claude declined to score this company"
        return result
    raw = json.loads(next(block.text for block in response.content if block.type == "text"))

    # Step 1: keep only evidence whose quote is really on the website.
    notes = []
    if not _verified(raw["workload_quote"], website_text):
        notes.append("Workload quote could not be verified")
    evidence = {
        column: _keep_verified(raw[column], website_text, column.replace("_", " "), notes)
        for column in ["ai_core_evidence", "model_signals", "production_signals", "why_now",
                       "enterprise_signals", "deprioritize_signals", "stage_evidence", "in_house_signals"]
    }

    # Step 2: code decides stage, segment, and grade from the verified evidence.
    amount = raise_amount(company)
    stage = estimate_stage(amount)
    segment, rule = segment_company(company["company_name"], evidence, stage, amount, company.get("verdict"))

    raise_trigger = _raise_trigger(company, amount, company["filing_url"])

    # CSV cells can't hold lists, so the quoted evidence is stored as JSON text.
    result.update({
        "account_segment": segment,
        "segment_rule": rule,
        "stage_estimate": stage,
        "fit_grade": decide_grade(evidence),
        "workload_type": "; ".join(raw["workload_type"]),
        "model_signal_types": "; ".join(sorted({s["type"] for s in evidence["model_signals"]})) or "unknown",
        "model_signals": json.dumps(evidence["model_signals"]),
        "production_signals": json.dumps(evidence["production_signals"]),
        "why_now": json.dumps([raise_trigger] + evidence["why_now"]),
        "enterprise_signals": json.dumps(evidence["enterprise_signals"]),
        "deprioritize_signals": json.dumps(evidence["deprioritize_signals"]),
        "ai_core_evidence": json.dumps(evidence["ai_core_evidence"]),
        "stage_evidence": json.dumps(evidence["stage_evidence"]),
        "in_house_signals": json.dumps(evidence["in_house_signals"]),
        "fit_reason": raw["fit_reason"],
        "persona": PERSONAS[segment],
        "note": "; ".join(notes) or None,
    })
    return result


EVIDENCE_COLUMNS = [
    "ai_core_evidence", "model_signals", "production_signals", "why_now",
    "enterprise_signals", "deprioritize_signals", "stage_evidence", "in_house_signals",
]


def resegment():
    """Re-apply the segment and grade rules to the evidence saved in fit.csv.

    Free: no Claude calls. Use it after changing the rules or the stage
    thresholds. Returns a table of the companies whose segment changed.
    """
    fit = pd.read_csv(FIT_FILE)
    filings = agent.load_form_d_data().set_index("filing_url")
    verdicts = pd.read_csv(qualify.RESULTS_FILE).set_index("filing_url")["verdict"]
    changes = []
    for i, row in fit.iterrows():
        if pd.isna(row["account_segment"]):
            continue  # never scored (for example, the website was too thin)
        evidence = {c: json.loads(row[c]) if pd.notna(row.get(c)) else [] for c in EVIDENCE_COLUMNS}
        # Drop the saved raise trigger; it's rebuilt below from the Form D (or manual entry).
        evidence["why_now"] = [t for t in evidence["why_now"] if "source" not in t]
        company = filings.loc[row["filing_url"]]
        amount = raise_amount(company)
        stage = estimate_stage(amount)
        segment, rule = segment_company(row["company_name"], evidence, stage, amount, verdicts.get(row["filing_url"]))
        if segment != row["account_segment"]:
            changes.append({"company_name": row["company_name"], "was": row["account_segment"], "now": segment, "rule": rule})
        fit.loc[i, ["account_segment", "segment_rule", "stage_estimate", "fit_grade", "persona", "why_now"]] = [
            segment, rule, stage, decide_grade(evidence), PERSONAS[segment],
            json.dumps([_raise_trigger(company, amount, row["filing_url"])] + evidence["why_now"]),
        ]
    fit.to_csv(FIT_FILE, index=False)
    return pd.DataFrame(changes)


def refresh_company(filing_url):
    """Re-read one company: fresh website text and fresh evidence (about 1.3 cents).

    Nothing else ever re-reads a scored company: every run skips companies
    already in fit.csv, so evidence only changes when you ask for it.
    Returns the new result row.
    """
    company = companies_to_score(filing_urls=[filing_url]).iloc[0]
    scores = pd.read_csv(FIT_FILE)
    scores[scores["filing_url"] != filing_url].to_csv(FIT_FILE, index=False)
    snapshot = website_text_path(company)
    if snapshot.exists():
        snapshot.unlink()
    return score_companies(companies_to_score(filing_urls=[filing_url]), limit=1).iloc[0]


def companies_to_score(filing_urls=None):
    """Qualified companies whose verdict earns a score, with their Form D data.

    If filing_urls is given, only those companies (for example, one run's filtered list).
    """
    qualified = pd.read_csv(qualify.RESULTS_FILE)
    eligible = qualified[qualified["verdict"].isin(VERDICTS_TO_SCORE) & qualified["website"].notna()]
    if filing_urls is not None:
        eligible = eligible[eligible["filing_url"].isin(filing_urls)]
    filings = agent.load_form_d_data().drop(columns=["company_name", "state"])
    return eligible.merge(filings, on="filing_url", how="left")


def score_companies(companies, limit=LIMIT, on_progress=None):
    """Score up to `limit` companies, skipping any already in fit.csv.

    If given, on_progress(done, total, company_name) is called before each
    company, so a caller like the app can show a progress bar.
    """
    saved = pd.read_csv(FIT_FILE) if FIT_FILE.exists() else pd.DataFrame()
    done = set(saved["filing_url"]) if not saved.empty else set()

    to_do = companies[~companies["filing_url"].isin(done)].head(limit)
    print(f"{len(done)} already scored, scoring {len(to_do)} new\n")

    results = []
    for i, (_, company) in enumerate(to_do.iterrows()):
        print(f"Scoring {company['company_name']}...")
        if on_progress:
            on_progress(i, len(to_do), company["company_name"])
        result = score_company(company)
        results.append(result)

        saved = pd.concat([saved, pd.DataFrame([result])], ignore_index=True)
        DATA_DIR.mkdir(exist_ok=True)
        saved.to_csv(FIT_FILE, index=False)

    return pd.DataFrame(results)


def estimated_cost():
    """Estimate this run's cost in dollars from the usage totals."""
    return (
        usage_totals["input_tokens"] * INPUT_PRICE_PER_TOKEN
        + usage_totals["output_tokens"] * OUTPUT_PRICE_PER_TOKEN
    )


if __name__ == "__main__":
    score_companies(companies_to_score())

    fit = pd.read_csv(FIT_FILE)
    fit["segment_order"] = fit["account_segment"].map(agent.SEGMENT_ORDER)
    fit = fit.sort_values(["segment_order", "fit_grade"], na_position="last")
    pd.set_option("display.max_colwidth", 110)
    pd.set_option("display.width", 250)
    print(fit[["company_name", "account_segment", "fit_grade", "stage_estimate", "workload_type", "segment_rule"]].to_string(index=False))
    print(f"\nUsage: {usage_totals}")
    print(f"Estimated cost of this run: ${estimated_cost():.4f}")
