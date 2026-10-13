"""Checks for the segment rules. Free: no Claude calls. Run it with:  python test_rules.py"""

import pandas as pd

import fit
import qualify


def evidence(**overrides):
    """Evidence for an AI-native company with real traffic, changed by `overrides`."""
    base = {
        "ai_core_evidence": [{"point": "AI is the product", "quote": "q"}],
        "model_signals": [], "production_signals": [{"signal": "customers", "quote": "q"}],
        "why_now": [], "enterprise_signals": [], "deprioritize_signals": [],
        "stage_evidence": [], "in_house_signals": [],
    }
    return {**base, **overrides}


def test_unsure_verdict_goes_to_review():
    # A $36M round with production evidence would be tier 1 (Relay's case)...
    assert fit.segment_company("Example AI", evidence(), "Series B or later", 36e6, "yes")[0] == "tier_1"
    # ...but with a "maybe" or "insufficient_info" verdict it needs a human check.
    for verdict in ["maybe", "insufficient_info"]:
        segment, rule = fit.segment_company("Example AI", evidence(), "Series B or later", 36e6, verdict)
        assert segment == "needs_review" and rule.startswith("R10"), (verdict, segment, rule)
    # Tier 2 too.
    tier2 = evidence(enterprise_signals=[{"type": "vertical_ai_for_finance_or_insurance", "quote": "for banks"}])
    assert fit.segment_company("Example AI", tier2, "Series A", 15e6, "maybe")[0] == "needs_review"
    # Tier 3 is not affected.
    assert fit.segment_company("Example AI", evidence(), "Series A", 15e6, "maybe")[0] == "tier_3"


def test_in_house_rule():
    seller = evidence(in_house_signals=[{"type": "sells_own_model_api", "quote": "Speech-to-Text API"}])  # old label
    assert fit.segment_company("Example AI", seller, "Series B or later", 50e6, "yes")[0] == "needs_review"
    # Your review: product APIs, not raw model inference.
    segment, rule = fit.segment_company("Photoroom", seller, "Series B or later", 50e6, "yes")
    assert segment == "tier_1" and "in-house rule not applied" in rule


def test_overrides_and_customers():
    assert fit.segment_company("Sierra", evidence(), "unknown", None, "yes")[0] == "tier_1"
    assert fit.segment_company("Harvey AI Corp", evidence(), "Series B or later", 50e6, "yes")[0] == "existing_customer"


def test_scale_evidence():
    assert fit.counts_as_scale({"stage": "scale_up", "kind": "number", "quote": "Loved by 26k+ Clinicians"})
    assert not fit.counts_as_scale({"stage": "scale_up", "kind": "other", "quote": "Trusted by leading enterprises"})
    assert not fit.counts_as_scale({"stage": "scale_up", "kind": "number", "quote": "SOC 2 Type II"})


def test_contact_roles():
    import contacts

    def roles(band, regulated=False, hiring=None):
        picked, notes = contacts.pick_roles(band, regulated, hiring)
        return [r["role"] for r in picked], notes

    assert roles("under_50")[0] == ["CTO or technical founder"]
    assert roles("50_300")[0] == ["Head of ML/AI or ML infrastructure lead", "VP Engineering"]
    assert roles("300_plus")[0] == ["Director of ML Platform", "VP Engineering or CTO", "Security lead", "Procurement"]
    # Regulated: a CISO is added (and replaces the plain security lead at 300+).
    assert roles("50_300", regulated=True)[0][-1] == "CISO or security lead"
    assert "Security lead" not in roles("300_plus", regulated=True)[0]
    # Thin evidence: the safest default, and it says so.
    picked, notes = roles("unknown")
    assert picked == ["CTO or technical founder"] and notes[0].startswith("Thin evidence")
    # Hiring for ML infra: the posting's team is the entry point.
    assert "entry point" in roles("50_300", hiring='"Hiring inference engineers"')[1][0]
    # Size from round size when the website doesn't say, labeled as an estimate.
    band, note = contacts.size_band({"total_offering_amount": 67e6}, "")
    assert band == "50_300" and "estimated" in note and "LinkedIn" in note
    assert contacts.size_band({"total_offering_amount": float("nan")}, "Our team of 12 builds")[0] == "under_50"


def test_saved_data_follows_the_verdict_rule():
    """No company in the saved results is tier 1 or 2 with an unsure verdict."""
    scores = pd.read_csv(fit.FIT_FILE).merge(pd.read_csv(qualify.RESULTS_FILE)[["filing_url", "verdict"]], on="filing_url")
    overridden = scores["company_name"].isin(fit.SEGMENT_OVERRIDES)
    bad = scores[scores["account_segment"].isin(["tier_1", "tier_2"]) & scores["verdict"].isin(fit.UNSURE_VERDICTS) & ~overridden]
    assert bad.empty, f"tier 1/2 with an unsure verdict: {bad['company_name'].tolist()}"


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"passed  {test.__name__}")
    print(f"All {len(tests)} tests passed.")
