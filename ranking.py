"""Rank prospects best first, and write a one-line "why now" for each.

Only companies you'd actually email are ranked: customers in tier_1, tier_2,
or tier_3 that aren't excluded. Everyone else gets no rank.
"""

import json

import pandas as pd

SEGMENT_RANK = {"tier_1": 0, "tier_2": 1, "tier_3": 2}
GRADE_RANK = {"A": 0, "B": 1, "C": 2}


def is_rankable(results):
    """True for each row that is an emailable prospect."""
    excluded = results["excluded"].eq(True) if "excluded" in results else False
    return (
        results["account_segment"].isin(SEGMENT_RANK)
        & results["relationship"].eq("customer")
        & ~excluded
    )


def add_priority(results):
    """Add a "priority" column: 1 is the best prospect, blank means not ranked.

    The order is strict, each step only breaking ties in the one before:
    segment (tier_1 first), fit grade (A first), confidence (highest first),
    then how recent the raise is (newest Form D filing first).
    """
    df = results.copy()
    ranked = df[is_rankable(df)].assign(
        _segment=lambda d: d["account_segment"].map(SEGMENT_RANK),
        _grade=lambda d: d["fit_grade"].map(GRADE_RANK),
        _filed=lambda d: pd.to_datetime(d["filing_date"], errors="coerce"),
    )
    ranked = ranked.sort_values(
        ["_segment", "_grade", "confidence", "_filed"],
        ascending=[True, True, False, False],
        na_position="last",
    )
    df["priority"] = pd.Series(range(1, len(ranked) + 1), index=ranked.index, dtype="Int64")
    return df


def why_now_line(row):
    """One line from saved evidence only: the Form D date, plus the first website trigger.

    Example: "Form D filed Oct 9 · New agent product launch (Super Agents)".
    """
    parts = []
    filed = pd.to_datetime(row.get("filing_date"), errors="coerce")
    if pd.notna(filed):
        parts.append(f"Form D filed {filed:%b} {filed.day}")
    triggers = json.loads(row["fit_why_now"]) if isinstance(row.get("fit_why_now"), str) else []
    if any(t.get("source") == "Manual entry" for t in triggers):
        parts.append("Added manually")
    website_triggers = [t for t in triggers if "source" not in t]
    if website_triggers:
        parts.append(website_triggers[0]["trigger"])
    return " · ".join(parts)
