"""Which tab each company belongs in, and the order of the Top 10 strip.

Raised: last 30 days   Form D filed 0 to 30 days before the run date
Raised: 31-60 days     Form D filed 31 to 60 days before the run date
Target accounts        added by hand or by CSV import, with no Form D in the last 60 days

Each company appears in exactly one tab. A target account that matches a
Form D from the last 60 days is shown once, in that Raised tab, tagged "Just raised".
"""

import pandas as pd

import agent
import manual
import ranking

RAISED_30, RAISED_60, TARGETS = "raised_30", "raised_60", "targets"
TAB_LABELS = {RAISED_30: "Raised: last 30 days", RAISED_60: "Raised: 31–60 days", TARGETS: "Target accounts"}

# Top 10 order: each segment in turn, and within it the tabs in this order.
TOP10_SEGMENTS = ["tier_1", "tier_2", "tier_3"]
TOP10_TABS = [RAISED_30, TARGETS, RAISED_60]


def _age_in_days(filed, run_date):
    filed = pd.to_datetime(filed, errors="coerce")
    return (pd.Timestamp(run_date) - filed).days if pd.notna(filed) else None


def _domain(website):
    return manual.domain_of(website) if isinstance(website, str) and website else None


def assign_tabs(results, run_date):
    """Add "tab" and "just_raised" columns, and drop target accounts shown in a Raised tab instead.

    A target account matches a Form D company when their website domains are
    the same, or their names are the same after removing endings like "Inc.".
    """
    df = results.copy()
    df["just_raised"] = False
    is_manual = df["source"].eq("Manual")
    filed = df["latest_filing_date"].fillna(df["filing_date"]) if "latest_filing_date" in df else df["filing_date"]
    ages = filed.map(lambda d: _age_in_days(d, run_date))
    df["tab"] = None
    df.loc[~is_manual & ages.between(0, 30), "tab"] = RAISED_30
    df.loc[~is_manual & ages.between(31, 60), "tab"] = RAISED_60

    raised = df[df["tab"].notna()]
    by_domain = {_domain(w): i for i, w in raised["website"].items() if _domain(w)}
    by_name = {agent._core_name(n): i for i, n in raised["company_name"].items()}
    shown_elsewhere = []
    for i, row in df[is_manual].iterrows():
        # Compare with None explicitly: a match at row 0 is valid but counts as False.
        match = by_domain.get(_domain(row["website"]))
        if match is None:
            match = by_name.get(agent._core_name(row["company_name"]))
        if match is not None:
            df.loc[match, "just_raised"] = True
            if "category" in df and isinstance(row.get("category"), str):
                df.loc[match, "category"] = row["category"]
            shown_elsewhere.append(i)
        else:
            df.loc[i, "tab"] = TARGETS
    return df.drop(index=shown_elsewhere)


def top10(results):
    """The Top 10 strip: tier 1 (raised 0-30, targets, raised 31-60), then tier 2, then tier 3.

    Within each group, companies keep their within-tab rank (grade, confidence, recency).
    """
    rankable = results[results["priority"].notna()]
    groups = [
        rankable[(rankable["account_segment"] == segment) & (rankable["tab"] == tab)].sort_values("priority")
        for segment in TOP10_SEGMENTS
        for tab in TOP10_TABS
    ]
    return pd.concat(groups).head(10)


def add_priority_by_tab(results):
    """Rank each tab separately: #1 is the best prospect in that tab."""
    return pd.concat(
        [ranking.add_priority(group) for _, group in results.groupby("tab", dropna=False)]
    ).loc[results.index]
