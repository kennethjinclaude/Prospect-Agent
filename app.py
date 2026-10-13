"""Prospect Agent: a Streamlit app that finds AI-native startups that just raised money.

Run it with:  streamlit run app.py
"""

import io
import json
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st

import agent
import contacts
import deep_dive
import edgar
import fit
import manual
import qualify
import ranking
import research
import tabs
import ui
from ui import GRADE_COLORS, SEGMENT_COLORS, SEGMENT_LABELS, SEGMENT_SHORT, SLOT_LABELS, money_millions, pill

ROOT = Path(__file__).parent
DEMO_DIR = ROOT / "demo_data"  # saved results, committed to git, free to view
LIVE_DIR = ROOT / "data"  # where live runs save their results (not committed)

# The demo emails were written with this sender, so the app can swap in
# whatever name is typed in Settings.
DEMO_SENDER = "Kenneth Jin"

# Average cost per company measured in earlier runs, used for the estimate.
QUALIFY_COST_PER_COMPANY = 0.025
FIT_COST_PER_COMPANY = 0.011
BRIEF_COST_PER_COMPANY = 0.02

LINKEDIN_URL = "https://www.linkedin.com/in/kenneth-jin-699686355"
GITHUB_URL = "https://github.com/kennethjinclaude/prospect-agent"

# Pages, by id, in the order of the navigation bar across the top.
LIST_PAGE_IDS = [tabs.RAISED_30, tabs.RAISED_60, tabs.TARGETS]
RESEARCH, SETTINGS = "research", "settings"
PAGE_LABELS = {
    tabs.RAISED_30: "Raised · 30 days",
    tabs.RAISED_60: "Raised · 31–60 days",
    tabs.TARGETS: "Target accounts",
    RESEARCH: "Research",
    SETTINGS: "Settings",
}

st.set_page_config(page_title="Prospect Agent · Kenneth Jin", layout="wide")
ui.inject_css()


# ---------------------------------------------------------------------------
# Loading results
# ---------------------------------------------------------------------------

def has_results(folder):
    """True if a folder holds a complete set of results to show."""
    return all((folder / name).exists() for name in ["filtered.csv", "qualified.csv", "run_stats.json"])


def load_results(folder):
    """Combine a run's files into one table: one row per qualified company.

    filtered.csv   the filings that passed the ICP filter in that run
    qualified.csv  Claude's AI-native verdict for each company
    briefs.csv     the brief and emails for each prospect (if any yet)
    fit.csv        the segment and fit grade for each AI-native company (if any yet)
    """
    filtered = pd.read_csv(folder / "filtered.csv")
    qualified = pd.read_csv(folder / "qualified.csv")
    stats = json.loads((folder / "run_stats.json").read_text())
    stats["after_filter"] = len(filtered)

    # Companies added by hand join the list, tagged "Manual" (target accounts).
    filtered["source"] = "Form D"
    manual_file = folder / manual.MANUAL_FILE.name
    manual_rows = pd.read_csv(manual_file) if manual_file.exists() else pd.DataFrame()
    stats["manual_count"] = len(manual_rows)
    if not manual_rows.empty:
        # Keep the website you entered as a fallback, for targets that aren't qualified yet.
        filtered = pd.concat([filtered, manual_rows.rename(columns={"website": "entered_website"})], ignore_index=True)

    # Only keep qualified companies that belong to this run's filtered list.
    columns = ["filing_url", "company_name", "state", "filing_date", "latest_filing_date",
               "total_offering_amount", "amount_sold", "related_persons", "source", "category", "entered_website"]
    # A left merge keeps target accounts that aren't qualified yet (e.g. a known company recorded
    # as a target), so they can still be matched to a Form D and tagged "Just raised".
    results = filtered[[c for c in columns if c in filtered]].merge(
        qualified.drop(columns=["company_name", "state"]), on="filing_url", how="left"
    )

    briefs_file = folder / "briefs.csv"
    if briefs_file.exists():
        briefs = pd.read_csv(briefs_file)
        # fit.csv is the source of truth for segments, so drop the copy saved with the brief.
        briefs = briefs.drop(columns=["company_name", "website", "verdict", "account_segment"], errors="ignore").rename(
            columns={"note": "brief_note", "created_at": "brief_created_at"}
        )
        results = results.merge(briefs, on="filing_url", how="left")

    fit_file = folder / "fit.csv"
    if fit_file.exists():
        fit_scores = pd.read_csv(fit_file).drop(columns=["company_name", "website"])
        # The brief also has a why_now column, so the fit one gets a prefix.
        fit_scores = fit_scores.rename(columns={"note": "fit_note", "scored_at": "fit_scored_at", "why_now": "fit_why_now"})
        results = results.merge(fit_scores, on="filing_url", how="left")
    if "entered_website" in results:
        results["website"] = results["website"].fillna(results["entered_website"])
    # Older result files may be missing these columns, so add them empty.
    for column in ["account_segment", "fit_grade", "workload_type", "relationship", "contact_name",
                   "company_display_name", "category"]:
        if column not in results:
            results[column] = None

    # Anyone on Baseten's customers page is an existing customer, even if never segmented.
    is_customer = results["company_name"].map(lambda n: bool(agent.existing_customer_match(n)))
    results.loc[is_customer, "account_segment"] = "existing_customer"

    # Each company goes in one tab (by filing age, or target accounts), ranked within it.
    results = tabs.assign_tabs(results, stats["run_date"])
    results = results[results["tab"].notna()]
    results = tabs.add_priority_by_tab(results)
    results["why_now_line"] = results.apply(ranking.why_now_line, axis=1)
    return results, stats


def personalize(text, written_by, sender_name):
    """Swap the sender's name into an email that was written for someone else."""
    if not isinstance(text, str) or not sender_name or sender_name == written_by:
        return text
    text = text.replace(written_by, sender_name)
    # The sign-off is the sender's first name on its own line.
    old_first, new_first = written_by.split()[0], sender_name.split()[0]
    return "\n".join(new_first if line == old_first else line for line in text.split("\n"))


# ---------------------------------------------------------------------------
# Detail panel sections (shown when a company is opened)
# ---------------------------------------------------------------------------

def show_fit(row):
    """Show a company's account segment, fit grade, and the website quotes behind them."""
    if pd.isna(row.get("account_segment")):
        if pd.notna(row.get("fit_note")):
            st.caption(f"Not segmented: {row['fit_note']}")
        return
    st.markdown(
        f"**{SEGMENT_LABELS[row['account_segment']]}**, fit grade {row['fit_grade']} "
        f"(stage from raise: {row['stage_estimate']})"
    )
    st.caption(f"Decided by rule {row['segment_rule']}")
    st.markdown(row["fit_reason"])

    lines = [f"- **Workload:** {ui.pretty_workload(row['workload_type'])}"]
    # Each group is a JSON list of {label, quote} items saved by fit.py.
    groups = [
        ("AI is the core product", "ai_core_evidence", "point", "no evidence on the website"),
        ("Model signals", "model_signals", "type", "unknown (no evidence on the website)"),
        ("Production signals", "production_signals", "signal", "none found"),
        ("Why now", "fit_why_now", "trigger", "none found"),
        ("Enterprise or regulated", "enterprise_signals", "type", "none found"),
        ("Stage on the website", "stage_evidence", "stage", "none found"),
        ("Deprioritize reasons", "deprioritize_signals", "reason", "none found"),
    ]
    for title, column, label, empty_text in groups:
        items = ui.md_all(json.loads(row[column])) if pd.notna(row.get(column)) else []
        if not items:
            lines.append(f"- **{title}:** {empty_text}")
            continue
        lines.append(f"- **{title}:**")
        for item in items:
            source = f" ({item['source']})" if "source" in item else ""
            lines.append(f'    - {item[label].replace("_", " ")}{source}: "{item["quote"]}"')
    if pd.notna(row.get("persona")):
        lines.append(f"- **Who to target:** {row['persona']}")
    st.markdown("\n".join(lines))

    st.caption("Every quote above was checked by code to appear on the company's website, except the raise, which comes from the Form D.")
    if pd.notna(row.get("fit_note")):
        st.caption(f"Fit note: {row['fit_note']}")


def refresh_evidence(filing_url):
    """Re-read one company (fresh website text and evidence), then bring its emails in line."""
    new = fit.refresh_company(filing_url)
    briefs = pd.read_csv(agent.BRIEFS_FILE)
    old = briefs[briefs["filing_url"] == filing_url]
    if not old.empty and old["account_segment"].iloc[0] != new["account_segment"]:
        briefs[briefs["filing_url"] != filing_url].to_csv(agent.BRIEFS_FILE, index=False)
    agent.SENDER_NAME, agent.SENDER_COMPANY = sender_name, sender_company
    prospects = agent.load_prospects()
    agent.write_briefs(prospects[prospects["filing_url"] == filing_url], limit=1)
    return new


def show_overview(row, folder):
    """Qualification evidence, segment evidence, brief, and contact."""
    if mode == "Live" and pd.notna(row.get("account_segment")) and row.get("account_segment") != "existing_customer":
        if st.button("Refresh evidence (about 1.3¢)", key=f"refresh_{row['filing_url']}",
                     help="Re-reads the website and evidence. Nothing is re-read unless you click this."):
            with st.spinner("Re-reading the website..."):
                new = refresh_evidence(row["filing_url"])
            st.success(f"Refreshed: now {SEGMENT_SHORT.get(new['account_segment'], 'not segmented')}. "
                       "Reopen the company to see the update.")
    if row.get("source") == "Manual":
        entered = " (entered by you, not from a filing)" if pd.notna(row["total_offering_amount"]) else ""
        st.markdown(ui.md(f"**Round:** {money_millions(row['total_offering_amount'])}{entered} · added manually, no Form D"))
    else:
        st.markdown(ui.md(
            f"**Round:** {money_millions(row['total_offering_amount'])} "
            f"(sold so far: {money_millions(row['amount_sold'])}) · Form D filed {row['filing_date']}"
        ))
    if pd.notna(row["evidence"]):
        st.markdown(f"**AI-native evidence** (verified in the {row['evidence_source']}):")
        st.markdown(f"> {row['evidence']}")
    if pd.notna(row["note"]):
        st.caption(f"Qualification note: {row['note']}")
    show_fit(row)

    if pd.isna(row.get("relationship")):
        return
    st.markdown(f"**Relationship:** {row['relationship']}. {row['relationship_reason']}")
    st.markdown(
        f"**Brief**\n\n"
        f"- **What they build:** {row['what_they_build']}\n"
        f"- **Likely models:** {row['likely_models']}\n"
        f"- **Inference pain:** {row['inference_pain']}\n"
        f"- **Why now:** {row['why_now']}"
    )
    if pd.notna(row["assumptions"]):
        st.markdown("**Assumptions** (inferred, not stated in the sources)")
        st.markdown("\n".join(f"- {a}" for a in row["assumptions"].split("; ")))

    if folder == DEMO_DIR and pd.notna(row["contact_title"]):
        # Demo data has names removed (see make_demo_data.py), so show the role only.
        st.markdown(f"**Contact:** {row['contact_title']}, from the Form D (name hidden in demo)")
    elif pd.notna(row["contact_name"]):
        st.markdown(f"**Contact:** {row['contact_name']} ({row['contact_title']}, from the Form D)")
    st.markdown(f"**If not the right person, look for:** {row['ideal_role']}")
    if pd.notna(row.get("brief_note")):
        st.caption(f"Brief note: {row['brief_note']}")


def show_deep_dive(row):
    """One-click research links as buttons, grouped. They only open URLs."""
    groups = deep_dive.links(row["company_name"], row.get("website"), row.get("company_display_name"))
    for group, items in groups.items():
        st.caption(group)
        columns = st.columns(2)
        for i, (label, url) in enumerate(items):
            columns[i % 2].link_button(label, url, width="stretch")


def show_contacts(row, folder):
    """Roles to contact (never names), each with why, what they care about, an opening angle, and a LinkedIn search."""
    saved = ui.md_all(contacts.load_contacts(row["filing_url"], folder / "contacts"))
    if not saved:
        st.caption("No contact roles for this company (only tier 1 to 3 and needs review get them).")
        return
    size = {"under_50": "under 50 people", "50_300": "50 to 300 people", "300_plus": "300+ people", "unknown": "unknown"}
    st.markdown(f"**Company size:** {size[saved['size_band']]} · {saved['size_note']} "
                f"· [LinkedIn company page]({saved['company_page']})")
    if saved["regulated"]:
        st.caption(f"Regulated: {saved['regulated']}")
    for note in saved["notes"]:
        (st.warning if note.startswith("Thin evidence") else st.info)(note)
    for warning in saved["warnings"]:
        st.warning(warning)
    for card in saved["cards"]:
        with st.container(border=True):
            st.markdown(f"**{SLOT_LABELS[card['slot']]}: {card['role']}**")
            st.markdown(f"- **Why this role:** {card['why_this_role']}\n"
                        f"- **Cares about:** {card['cares_about']}\n"
                        f"- **Opening angle:** {card['opening_angle']}")
            st.markdown(ui.external_link("Find them on LinkedIn", card["linkedin"]), unsafe_allow_html=True)


def _source_link(url, verified=True):
    if not url:
        return "no source"
    mark = "" if verified else f' <span style="color:#B45309">{ui.ICON_WARNING} unverified source</span>'
    return f"[source]({url}){mark}"


def show_research(row, folder):
    """The research brief: run it (live mode, paid), or show the saved one with source links."""
    saved = research.load_brief(row["filing_url"], folder / "research")
    allowed, reason = research.can_research(row)
    if not allowed:
        st.info(reason)
        return

    if mode == "Live":
        label = (f"Refresh research (paid, about {research.ESTIMATE_TYPICAL * 100:.0f}¢)" if saved
                 else f"Run research brief (about {research.ESTIMATE_TYPICAL * 100:.0f}¢, at most {research.ESTIMATE_MAX * 100:.0f}¢)")
        if st.button(label, key=f"research_{row['filing_url']}"):
            with st.spinner("Researching: up to 8 searches and 4 page reads, usually 1 to 3 minutes..."):
                try:
                    saved, _ = research.get_or_run(row, refresh=True)
                except (anthropic.APIError, RuntimeError, ValueError) as error:
                    st.error(f"Research failed: {error}")
                    return
    if not saved:
        st.caption("No research brief yet." + ("" if mode == "Live" else " Research briefs are run in live mode."))
        return

    age = research.age_in_days(saved)
    usage = saved["usage"]
    st.caption(
        f"Researched {'today' if age == 0 else f'{age} days ago'} · cost {saved['cost'] * 100:.0f}¢ · "
        f"{usage['searches']} searches · {usage['pages_read']} pages read"
    )
    if age > research.STALE_AFTER_DAYS:
        st.warning(f"May be outdated: this brief is {age} days old. Refresh it before using it.")
    for warning in saved["warnings"]:
        st.warning(warning)

    brief = ui.md_all(saved["brief"])
    st.markdown("**3 facts that matter for Baseten**")
    for fact in brief["facts"]:
        outdated = " *(possibly outdated)*" if fact["possibly_outdated"] else ""
        st.markdown(f"- {fact['fact']}{outdated} · {_source_link(fact['source_url'], fact['source_verified'])}",
                    unsafe_allow_html=True)
    st.markdown(f"**Best hook:** {brief['hook']} · {_source_link(brief['hook_source_url'], brief['hook_source_verified'])}",
                unsafe_allow_html=True)
    st.caption(f"Why: {brief['hook_why']}")
    st.markdown(f"**Discovery question:** {brief['discovery_question']}")
    if brief["what_not_to_say"]:
        st.markdown("**What not to say**\n" + "\n".join(f"- {item}" for item in brief["what_not_to_say"]))
    st.markdown(f"**Confidence:** {brief['confidence']}. {brief['confidence_reason']}")
    if brief["contact_roles"]:
        st.markdown(f"**Roles to contact:** {', '.join(brief['contact_roles'])}")
    if brief["competitors_that_are_customers"]:
        st.markdown(f"**Competitors that are Baseten customers:** {', '.join(brief['competitors_that_are_customers'])}")
    if brief["not_found"]:
        st.caption(f"Not found: {'; '.join(brief['not_found'])}")
    with st.expander(f"All {len(saved['sources'])} sources the research read"):
        st.markdown("\n".join(f"- {url}" for url in saved["sources"]))


def show_emails(row, sender_name):
    """The email sequence (or tier 3's single email), each in a copyable box."""
    if pd.isna(row.get("email_body")):
        st.caption("No email for this company (only tier 1 to 3 customers get outreach).")
        return
    written_by = row["sender_name"] if pd.notna(row.get("sender_name")) else DEMO_SENDER
    if row.get("account_segment") == "tier_3" and pd.notna(row.get("nurture_note")):
        st.markdown(f"**Nurture note:** {ui.md(row['nurture_note'])}")
    st.caption("Day 0 · Initial email")
    st.code(
        personalize(f"Subject: {row['email_subject']}\n\n{row['email_body']}", written_by, sender_name),
        language=None, wrap_lines=True,
    )
    for number, day in enumerate(agent.FOLLOWUP_DAYS, start=1):
        if pd.isna(row.get(f"followup_day{day}")):
            continue  # tier_3 has no follow-ups
        st.caption(f"Day {day} · Follow-up {number}")
        st.code(
            personalize(f"Subject: {row['followup_subject']}\n\n{row[f'followup_day{day}']}", written_by, sender_name),
            language=None, wrap_lines=True,
        )


def company_tags(row):
    """Small pills shown next to a company's name: Manual, Just raised."""
    tags = []
    if row.get("source") == "Manual":
        tags.append(pill("Manual", ui.GRAY))
    if row.get("just_raised"):
        tags.append(pill("Just raised", ui.GRAY))
    return " ".join(tags)


def show_detail(row, folder, sender_name):
    """The right-side panel for one company: short sections open, long ones collapsed."""
    text = ui.md_all(row)  # for display in st.markdown; emails and research runs use the raw row
    top, close = st.columns([5, 1], vertical_alignment="center")
    icon = ui.favicon_for(row.get("website"), favicons)
    top.markdown(f'<img src="{icon}" width="18" style="vertical-align:middle;margin-right:8px;border-radius:4px">'
                 f'<span class="detail-name">{ui.esc(row["company_name"])}</span>', unsafe_allow_html=True)
    close.button("Close", on_click=close_panel, key="close_panel")

    tags = [company_tags(row), pill(tabs.TAB_LABELS[row["tab"]], ui.GRAY)]
    if pd.notna(row.get("priority")):
        tags.append(pill(f"#{row['priority']}", ui.GRAY))
    if pd.notna(row.get("account_segment")):
        tags.append(pill(SEGMENT_SHORT[row["account_segment"]], SEGMENT_COLORS[row["account_segment"]]))
    if pd.notna(row.get("fit_grade")):
        tags.append(pill(f"Grade {row['fit_grade']}", GRADE_COLORS[row["fit_grade"]]))
    st.markdown(" ".join(t for t in tags if t), unsafe_allow_html=True)

    ui.label("Why now")
    st.markdown(text["why_now_line"] or "No specific trigger found beyond being on the list.")
    if pd.notna(row.get("product_summary")):
        st.markdown(f"**Product:** {text['product_summary']}")
    if pd.notna(row.get("website")):
        st.markdown(ui.external_link(row["website"], row["website"]), unsafe_allow_html=True)

    st.divider()
    ui.label("Who to contact")
    show_contacts(row, folder)

    with st.expander("Research brief"):
        show_research(row, folder)
    with st.expander("Emails"):
        show_emails(row, sender_name)
    with st.expander("Deep-dive links"):
        show_deep_dive(row)
    with st.expander("Evidence and fit details"):
        show_overview(text, folder)


# ---------------------------------------------------------------------------
# Add a company by hand, or import a CSV (live mode only, since it costs money)
# ---------------------------------------------------------------------------

@st.dialog("Add a company")
def add_company_dialog():
    """Form, then a cost check, then the run. Adding a known company costs nothing."""
    with st.form("add_company"):
        name = st.text_input("Company name")
        website = st.text_input("Website (required)", placeholder="example.com")
        round_m = st.number_input(
            "Round size ($M), optional", value=None, min_value=0.0, step=1.0,
            help="Shown as entered by you, not from a filing. Used to estimate the stage.",
        )
        if st.form_submit_button("Check", type="primary"):
            if not name.strip() or not website.strip():
                st.error("Company name and website are both required.")
            else:
                st.session_state["pending_add"] = {"name": name, "website": website, "round": round_m}

    pending = st.session_state.get("pending_add")
    if not pending:
        return
    existing = manual.find_existing(pending["website"])
    if existing:
        key, known_name = existing
        st.info(f"{manual.domain_of(pending['website'])} is already in your results as **{known_name}**. No cost.")
        if st.button("Open it"):
            st.session_state["selected"] = key
            del st.session_state["pending_add"]
            st.rerun()
        return

    typical, highest = manual.estimated_cost()
    st.write(
        f"Runs the AI-native check, evidence and segment, and (for tier 1 to 3) the brief and emails "
        f"for **{pending['name']}**. Estimated cost: **about {typical * 100:.1f}¢**, at most {highest * 100:.1f}¢."
    )
    if st.button(f"Run (about {typical * 100:.1f}¢)", type="primary"):
        progress = st.progress(0.0, text="Starting...")
        agent.SENDER_NAME, agent.SENDER_COMPANY = sender_name, sender_company
        result = manual.add_company(
            pending["name"], pending["website"], pending["round"],
            on_progress=lambda fraction, text: progress.progress(fraction, text=text),
        )
        st.session_state["selected"] = result["key"]
        st.session_state["last_add"] = result
        del st.session_state["pending_add"]
        st.rerun()


@st.dialog("Import target accounts", width="large")
def import_dialog():
    """Upload a CSV, see what's new and what it costs, then run. Known companies are free."""
    st.download_button("Download the template CSV", manual.IMPORT_TEMPLATE,
                       file_name="target_accounts_template.csv", mime="text/csv")
    upload = st.file_uploader("CSV with columns name, website, optional round_millions ($M) and category", type="csv")
    if upload is None:
        return
    new_rows, known_rows, problems = manual.parse_import(io.BytesIO(upload.getvalue()))
    for problem in problems:
        st.warning(problem)
    if known_rows:
        names = ", ".join(row["known_as"] for row in known_rows[:8]) + ("…" if len(known_rows) > 8 else "")
        st.caption(f"{len(known_rows)} already in your results, so they're free: {names}")
    if not new_rows:
        st.info("Nothing new to add.")
        return
    typical, highest = manual.estimated_cost()
    st.write(
        f"**{len(new_rows)} new companies.** Each gets the AI-native check, evidence and segment, and (for tier 1 "
        f"to 3) a brief and emails. Estimated cost: **about ${len(new_rows) * typical:.2f}**, "
        f"at most ${len(new_rows) * highest:.2f}."
    )
    if st.button(f"Import {len(new_rows)} (about ${len(new_rows) * typical:.2f})", type="primary"):
        manual.record_known_targets(known_rows)  # free: keeps their categories
        progress = st.progress(0.0, text="Starting...")
        agent.SENDER_NAME, agent.SENDER_COMPANY = sender_name, sender_company
        cost = manual.import_companies(new_rows, on_progress=lambda f, text: progress.progress(f, text=text))
        st.session_state["last_import"] = {"count": len(new_rows), "cost": cost}
        st.rerun()


# ---------------------------------------------------------------------------
# Opening and closing a company; saving and clearing a view
# ---------------------------------------------------------------------------

st.session_state.setdefault("selected", None)


def open_company(filing_url, page_id=None):
    """Open a company in the right column (and switch to its page, from the Top 10)."""
    st.session_state["selected"] = filing_url
    if page_id:
        st.session_state["nav"] = page_id


def close_panel():
    st.session_state["selected"] = None


# Filters each list page keeps, with their default values. Widget keys end in the page id.
FILTER_DEFAULTS = {"segments": ["tier_1"], "categories": [], "search": "", "verdict": "all", "round": 0.0, "hidden": False}
CLEARED = {"segments": [], "categories": [], "search": "", "verdict": "all", "round": 0.0, "hidden": False}


def save_view(page_id):
    """Put the page and its filters in the link, so the view can be bookmarked or shared."""
    st.session_state["_restored_view"] = True  # don't re-read the link on the next run
    params = {"page": page_id}
    for name in FILTER_DEFAULTS:
        value = st.session_state.get(f"{name}_{page_id}")
        if value in (None, "", [], 0.0, False, "all"):
            continue
        params[name] = ",".join(value) if isinstance(value, list) else str(value)
    st.query_params.from_dict(params)
    st.session_state["view_saved"] = True


def clear_view(page_id):
    """Reset this page's filters (showing everyone) and remove any saved view from the link."""
    for name, value in CLEARED.items():
        st.session_state[f"{name}_{page_id}"] = value
    st.session_state[f"pg_{page_id}"] = 1
    st.query_params.clear()


def restore_view_from_link():
    """On the first load, apply a view saved in the link (page and filters)."""
    if st.session_state.get("_restored_view"):
        return
    st.session_state["_restored_view"] = True
    params = st.query_params
    page_id = params.get("page")
    if page_id not in LIST_PAGE_IDS:
        return
    st.session_state["nav"] = page_id
    for name, default in FILTER_DEFAULTS.items():
        if name not in params:
            continue
        raw = params[name]
        if isinstance(default, list):
            value = [v for v in raw.split(",") if v]
        elif isinstance(default, bool):
            value = raw == "True"
        elif isinstance(default, float):
            value = float(raw)
        else:
            value = raw
        st.session_state[f"{name}_{page_id}"] = value


restore_view_from_link()


# ---------------------------------------------------------------------------
# Settings that last across pages (their inputs live on the Settings page)
# ---------------------------------------------------------------------------

st.session_state.setdefault("mode", "Demo")
st.session_state.setdefault("live_ok", False)
st.session_state.setdefault("sender_name", agent.SENDER_NAME)
st.session_state.setdefault("sender_company", agent.SENDER_COMPANY)
mode = st.session_state["mode"]
sender_name = st.session_state["sender_name"]
sender_company = st.session_state["sender_company"]


def keep(setting):
    """Copy a Settings input into the setting that lasts across pages."""
    st.session_state[setting] = st.session_state[f"{setting}_input"]


# ---------------------------------------------------------------------------
# Live mode: password, then the settings page with the two-step run
# ---------------------------------------------------------------------------

def ask_live_password():
    """Ask for the password from Streamlit secrets. Remembers it for the session once it matches."""
    try:
        password = st.secrets["APP_PASSWORD"]
    # OSError covers a missing or unreadable secrets file.
    except (KeyError, OSError, st.errors.StreamlitSecretNotFoundError):
        st.warning("Live mode is off: set APP_PASSWORD in .streamlit/secrets.toml to enable it.")
        return
    entered = st.text_input("Password", type="password")
    if entered == password:
        st.session_state["live_ok"] = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")


def run_new_search(days, min_raise):
    """Step 1 finds and filters filings (free). Step 2 qualifies and writes briefs (paid)."""
    st.markdown("##### Run a new search")

    # Step 1: free. Only downloads public SEC data.
    if st.button(f"1. Find and filter the last {days} days of filings (free)"):
        with st.spinner(f"Downloading {days} days of Form D filings from SEC EDGAR (about {days * 8 // 30} minutes)..."):
            filings = edgar.fetch_form_d_filings(days)
        settings = {**edgar.SETTINGS, "min_offering_amount": min_raise}
        filtered = edgar.filter_by_icp(filings, settings)
        LIVE_DIR.mkdir(exist_ok=True)
        filtered.to_csv(LIVE_DIR / "filtered.csv", index=False)
        stats = {
            "run_date": str(pd.Timestamp.today().date()),
            "days": days,
            "min_offering_amount": min_raise,
            "filings_found": len(filings),
        }
        (LIVE_DIR / "run_stats.json").write_text(json.dumps(stats, indent=2))

    if not (LIVE_DIR / "filtered.csv").exists():
        return
    filtered = pd.read_csv(LIVE_DIR / "filtered.csv")

    # Step 2: paid. Show the estimate first.
    new_count = qualify.count_to_qualify(filtered, limit=len(filtered))
    estimate = new_count * (QUALIFY_COST_PER_COMPANY + FIT_COST_PER_COMPANY + BRIEF_COST_PER_COMPANY)
    st.write(
        f"{len(filtered)} companies passed the filter; **{new_count}** haven't been qualified yet. "
        f"Estimated cost: **about ${estimate:.2f}** (at most, if every company becomes a prospect)."
    )
    if new_count == 0 or not st.button(f"2. Qualify, score fit, and write briefs (about ${estimate:.2f})"):
        return

    for totals in (qualify.usage_totals, fit.usage_totals, agent.usage_totals):
        totals.update(dict.fromkeys(totals, 0))
    progress = st.progress(0.0, text="Starting...")

    # Qualifying fills the first 60% of the bar, fit scoring the next 20%, briefs the rest.
    def qualify_progress(done, total, name):
        progress.progress(0.6 * done / total, text=f"Qualifying {name} ({done + 1} of {total})")

    def fit_progress(done, total, name):
        progress.progress(0.6 + 0.2 * done / total, text=f"Scoring Baseten fit for {name} ({done + 1} of {total})")

    def brief_progress(done, total, name):
        progress.progress(0.8 + 0.2 * done / total, text=f"Writing brief for {name} ({done + 1} of {total})")

    qualify.qualify_companies(filtered, limit=len(filtered), on_progress=qualify_progress)

    to_score = fit.companies_to_score(filing_urls=filtered["filing_url"])
    fit.score_companies(to_score, limit=len(to_score), on_progress=fit_progress)

    agent.SENDER_NAME, agent.SENDER_COMPANY = sender_name, sender_company
    prospects = agent.load_prospects()
    prospects = prospects[prospects["filing_url"].isin(filtered["filing_url"])]
    agent.write_briefs(prospects, limit=len(prospects), on_progress=brief_progress)

    progress.progress(1.0, text="Done")
    cost = qualify.estimated_cost() + fit.estimated_cost() + agent.estimated_cost()
    st.success(f"Run complete. Actual cost: about ${cost:.2f}.")


def show_settings_page():
    """Data mode and sender for everyone; in live mode, new searches and adding target accounts."""
    with st.container(key="card_settings_mode"):
        st.radio("Data", ["Demo", "Live"], horizontal=True, key="mode_input", index=["Demo", "Live"].index(mode),
                 on_change=keep, args=("mode",),
                 help="Demo shows a real past run for free. Live runs new searches, which use API credit.")
        name_col, company_col = st.columns(2)
        name_col.text_input("Sender name", value=sender_name, key="sender_name_input", on_change=keep, args=("sender_name",))
        company_col.text_input("Sender company", value=sender_company, key="sender_company_input",
                               on_change=keep, args=("sender_company",),
                               help="Used when writing new emails. The fact sheet in agent.py is Baseten's, "
                                    "so update BASETEN_FACTS there before using another company.")
    if mode != "Live":
        return
    if not st.session_state["live_ok"]:
        with st.container(key="card_settings_password"):
            ask_live_password()
        return
    st.divider()
    with st.container(key="card_settings_search"):
        days = st.segmented_control(
            "Days back", [30, 60], default=edgar.SETTINGS["lookback_days"], format_func=lambda d: f"{d} days",
            help="How many days of Form D filings to search. Downloading takes about 8 minutes per 30 days.",
        ) or edgar.SETTINGS["lookback_days"]
        min_raise = st.number_input("Minimum round ($)", min_value=0, value=edgar.SETTINGS["min_offering_amount"], step=250_000)
        run_new_search(days, min_raise)
    st.divider()
    with st.container(key="card_targets"):
        st.markdown("##### Target accounts")
        add_col, import_col, _ = st.columns([1, 1, 3])
        if add_col.button("Add company", width="stretch"):
            st.session_state.pop("pending_add", None)
            add_company_dialog()
        if import_col.button("Import CSV", width="stretch"):
            import_dialog()


# ---------------------------------------------------------------------------
# Main page: header
# ---------------------------------------------------------------------------

folder = LIVE_DIR if mode == "Live" else DEMO_DIR
can_show = (mode != "Live" or st.session_state["live_ok"]) and has_results(folder)
results, stats = load_results(folder) if can_show else (None, None)
header_stats = stats or (json.loads((folder / "run_stats.json").read_text()) if (folder / "run_stats.json").exists() else None)

data_tag = "Live"
if mode != "Live" and header_stats:
    run_day = pd.Timestamp(header_stats["run_date"])
    data_tag = f"Demo · {run_day:%b} {run_day.day}"

# One row: the wordmark on the left, the navigation (with a count on each page) to its right.
page_counts = {}
if results is not None:
    for page_id in LIST_PAGE_IDS:
        page_counts[page_id] = int((results["tab"] == page_id).sum())
    research_dir = folder / "research"
    page_counts[RESEARCH] = len(list(research_dir.glob("*.json"))) if research_dir.exists() else 0
st.session_state.setdefault("nav", tabs.RAISED_30)
brand_col, nav_col = st.columns([1.5, 5], vertical_alignment="center")
brand_col.markdown(
    f'<div class="wordmark">Prospect Agent<span class="meta">by Kenneth Jin</span>'
    f'<span class="tag">{ui.esc(data_tag)}</span></div>',
    unsafe_allow_html=True,
)
with nav_col:
    page = st.segmented_control(
        "Pages", list(PAGE_LABELS), key="nav", required=True, label_visibility="collapsed",
        format_func=lambda p: f"{PAGE_LABELS[p]} :gray-background[{page_counts[p]}]" if p in page_counts else PAGE_LABELS[p],
    )
st.markdown(
    '<div class="header-rule"></div><div class="strip">SEC filings<span class="arrow">→</span>'
    'AI qualification<span class="arrow">→</span>Ranked outreach</div>',
    unsafe_allow_html=True,
)

if page == SETTINGS:
    show_settings_page()
    st.stop()
if mode == "Live" and not st.session_state["live_ok"]:
    st.info("Live mode needs the password: enter it in Settings.")
    st.stop()
if results is None:
    st.info("No results yet. Run a new search in Settings.")
    st.stop()

favicons = ui.load_favicons(folder)
excluded = results["excluded"].eq(True) if "excluded" in results else pd.Series(False, index=results.index)

if "last_add" in st.session_state:
    added = st.session_state.pop("last_add")
    segment = SEGMENT_SHORT.get(added["segment"], "not segmented")
    st.success(f"Added: verdict {added['verdict']}, {segment}. Actual cost: about {added['cost'] * 100:.1f}¢.")
if "last_import" in st.session_state:
    imported = st.session_state.pop("last_import")
    st.success(f"Imported {imported['count']} companies. Actual cost: about ${imported['cost']:.2f}.")
if st.session_state.pop("view_saved", False):
    st.toast("View saved in the link: bookmark or share this page's address.")


# ---------------------------------------------------------------------------
# Main page: list pages (filters, rows, pager) and the research page
# ---------------------------------------------------------------------------

def filtered_view(tab_rows, page_id):
    """Apply this page's filters (read from its widgets) and sort best first."""
    get = lambda name: st.session_state.get(f"{name}_{page_id}", FILTER_DEFAULTS[name])  # noqa: E731
    view = tab_rows
    hidden_mask = view["relationship"].isin(["competitor", "partner"]) | (view["account_segment"] == "deprioritize")
    if not get("hidden"):
        view = view[~hidden_mask]
    if get("search"):
        # Match the legal filing name, the brand name from the brief, or the website domain.
        searchable = (
            view["company_name"].fillna("") + " " + view["company_display_name"].fillna("")
            + " " + view["website"].fillna("").map(lambda w: manual.domain_of(w) if w else "")
        )
        view = view[searchable.str.contains(get("search"), case=False, regex=False)]
    if get("segments"):
        view = view[view["account_segment"].isin(get("segments"))]
    if get("verdict") != "all":
        view = view[view["verdict"] == get("verdict")]
    if get("categories"):
        view = view[view["category"].isin(get("categories"))]
    if get("round"):
        view = view[(view["total_offering_amount"] >= get("round") * 1e6) | view["total_offering_amount"].isna()]
    view = view.assign(segment_order=view["account_segment"].map(agent.SEGMENT_ORDER)).sort_values(
        ["priority", "segment_order", "fit_grade", "confidence"],
        ascending=[True, True, True, False], na_position="last",
    )
    return view, int(hidden_mask.sum())


def count_headline(count, segments):
    """ "6 tier 1 prospects", or "12 companies" when several or no segments are chosen."""
    if len(segments) == 1:
        word = "prospect" if count == 1 else "prospects"
        return f"{count} {SEGMENT_SHORT[segments[0]].lower()} {word}"
    return f"{count} {'company' if count == 1 else 'companies'}"


ROW_COLUMNS = [0.35, 3.2, 1.3, 0.85, 0.55, 3.4, 0.55]
ROW_HEADERS = ["#", "Company", "Category", "Segment", "Grade", "Why now", ""]


def show_rows(view, page_id):
    """A plain list, one row per company: rank, favicon + name, category, pills, why now, Open."""
    head = st.columns(ROW_COLUMNS)
    for column, title in zip(head, ROW_HEADERS):
        column.markdown(f'<div class="row-head">{title}</div>', unsafe_allow_html=True)
    start, end = ui.page_slice(f"pg_{page_id}", len(view))
    with st.container(key=f"list_{page_id}"):
        for _, row in view.iloc[start:end].iterrows():
            row_id = ui.key_id(row["filing_url"])
            selected = "_sel" if st.session_state["selected"] == row["filing_url"] else ""
            with st.container(key=f"row_{page_id}_{row_id}{selected}"):
                cells = st.columns(ROW_COLUMNS, vertical_alignment="center")
                rank = row["priority"] if pd.notna(row.get("priority")) else "·"
                cells[0].markdown(f'<span class="muted rank-cell">{rank}</span>', unsafe_allow_html=True)
                parts = []
                if pd.notna(row["total_offering_amount"]):
                    parts.append(f"{money_millions(row['total_offering_amount'])} round")
                elif row.get("source") != "Manual":
                    parts.append("round not stated")
                if pd.notna(row.get("confidence")):
                    parts.append(f"confidence {int(row['confidence'])}")
                company_page = deep_dive.links(row["company_name"], row.get("website"),
                                               row.get("company_display_name"))["Company"][0][1]
                tags = f"{ui.icon_link(company_page, 'LinkedIn company page')} {company_tags(row)}"
                cells[1].markdown(ui.company_cell(row["company_name"], ui.favicon_for(row.get("website"), favicons),
                                                  tags, ui.esc(" · ".join(parts))), unsafe_allow_html=True)
                # Only target accounts have a category (from your import file); others show a dash.
                category = row["category"] if isinstance(row.get("category"), str) else "–"
                cells[2].markdown(f'<span class="muted">{ui.esc(category)}</span>', unsafe_allow_html=True)
                if pd.notna(row.get("account_segment")):
                    cells[3].markdown(pill(SEGMENT_SHORT[row["account_segment"]], SEGMENT_COLORS[row["account_segment"]]),
                                      unsafe_allow_html=True)
                if pd.notna(row.get("fit_grade")):
                    cells[4].markdown(pill(row["fit_grade"], GRADE_COLORS[row["fit_grade"]]), unsafe_allow_html=True)
                cells[5].markdown(ui.why_now_cell(row["why_now_line"]), unsafe_allow_html=True)
                cells[6].button("Open", key=f"open_{page_id}_{row_id}", on_click=open_company, args=(row["filing_url"],))
    ui.pager_controls(f"pg_{page_id}", len(view))


def show_list_page(page_id):
    """Count line, filter chips, and actions, then the rows."""
    tab_rows = results[(results["tab"] == page_id) & ~excluded]
    with st.container(key=f"card_header_{page_id}"):
        count_slot = st.empty()
        if tab_rows.empty:
            count_slot.markdown('<div class="count">0 companies</div>', unsafe_allow_html=True)
            if page_id == tabs.TARGETS and mode == "Live":
                st.info("No target accounts yet. Add or import companies in Settings.")
            elif page_id == tabs.TARGETS:
                st.info("No target accounts in this demo.")
            else:
                st.info("No companies raised in this window yet.")
            return
        for name, default in FILTER_DEFAULTS.items():
            st.session_state.setdefault(f"{name}_{page_id}", default)
        st.pills("Segment", options=list(SEGMENT_SHORT)[:5], format_func=SEGMENT_SHORT.get,
                 selection_mode="multi", label_visibility="collapsed", key=f"segments_{page_id}")
        if page_id == tabs.TARGETS:
            st.pills("Category", options=sorted(tab_rows["category"].dropna().unique()), selection_mode="multi",
                     label_visibility="collapsed", key=f"categories_{page_id}")
        # One row: the search box takes the free space, the buttons are only as wide as their text.
        with st.container(horizontal=True, vertical_alignment="center", gap="small", key=f"toolbar_{page_id}"):
            st.text_input("Search", placeholder="Search companies", label_visibility="collapsed",
                          key=f"search_{page_id}", width="stretch")
            with st.popover("Filters", width="content"):
                st.radio("Verdict", ["all", "yes", "maybe"], horizontal=True, key=f"verdict_{page_id}")
                st.number_input("Minimum round ($M)", min_value=0.0, step=5.0, key=f"round_{page_id}")
                st.checkbox("Show competitors, partners, and deprioritized", key=f"hidden_{page_id}",
                            help="Hidden by default: they get no outreach.")
            st.button("Save view", width="content", on_click=save_view, args=(page_id,),
                      help="Puts this page and its filters in the link, to bookmark or share")
            st.button("Clear", width="content", on_click=clear_view, args=(page_id,), help="Show everyone on this page")
            view, hidden_count = filtered_view(tab_rows, page_id)
            st.download_button("Export", view.to_csv(index=False).encode("utf-8"), key=f"download_{page_id}",
                               file_name=f"prospects_{page_id}.csv", mime="text/csv", width="content",
                               help="Download this view as a CSV")
        hidden_note = "" if st.session_state[f"hidden_{page_id}"] else f" · {hidden_count} hidden"
        count_slot.markdown(
            f'<div class="count">{count_headline(len(view), st.session_state[f"segments_{page_id}"])}'
            f'<span>{len(view)} of {len(tab_rows)}{hidden_note}</span></div>',
            unsafe_allow_html=True,
        )
    if view.empty:
        chosen = st.session_state[f"segments_{page_id}"]
        chips = ", ".join(SEGMENT_SHORT[x] for x in chosen) if chosen else "matching"
        st.info(f"No {chips} companies here. Try other segments, or press Clear.")
        return
    show_rows(view, page_id)


def show_research_page():
    """Companies with a saved research brief, newest first."""
    research_dir = folder / "research"
    saved = [json.loads(p.read_text()) for p in research_dir.glob("*.json")] if research_dir.exists() else []
    known = results.set_index("filing_url")
    saved = sorted((s for s in saved if s["filing_url"] in known.index), key=lambda s: s["created_at"], reverse=True)
    st.markdown(f'<div class="count">{len(saved)} research briefs</div>', unsafe_allow_html=True)
    if not saved:
        st.info("No research briefs yet. Run one from a company's panel in live mode.")
        return
    columns = [3, 1.2, 3.6, 0.6]
    for column, title in zip(st.columns(columns), ["Company", "Confidence", "Best hook", ""]):
        column.markdown(f'<div class="row-head">{title}</div>', unsafe_allow_html=True)
    with st.container(key="list_research"):
        for brief in saved:
            row = known.loc[brief["filing_url"]]
            row_id = ui.key_id(brief["filing_url"])
            selected = "_sel" if st.session_state["selected"] == brief["filing_url"] else ""
            with st.container(key=f"row_research_{row_id}{selected}"):
                cells = st.columns(columns, vertical_alignment="center")
                age = research.age_in_days(brief)
                subtitle = f"researched {'today' if age == 0 else f'{age} days ago'} · cost {brief['cost'] * 100:.0f}¢"
                cells[0].markdown(ui.company_cell(brief["company_name"], ui.favicon_for(row.get("website"), favicons), "",
                                                  ui.esc(subtitle)), unsafe_allow_html=True)
                cells[1].markdown(f'<span class="muted">{ui.esc(brief["brief"]["confidence"])}</span>', unsafe_allow_html=True)
                cells[2].markdown(ui.why_now_cell(brief["brief"]["hook"]), unsafe_allow_html=True)
                cells[3].button("Open", key=f"open_research_{row_id}", on_click=open_company, args=(brief["filing_url"],))


def show_right_column():
    """The opened company, or the pipeline overview and Top 10."""
    selected = st.session_state["selected"]
    if selected in set(results["filing_url"]):
        with st.container(key="card_detail"):
            show_detail(results.set_index("filing_url").loc[selected].to_dict() | {"filing_url": selected}, folder, sender_name)
        return

    form_d = results[results["source"] == "Form D"]
    form_d_ranked = form_d[form_d["priority"].notna()]
    targets = results[results["tab"] == tabs.TARGETS]
    with st.container(key="card_pipeline"):
        ui.label("Pipeline")
        steps = [
            ("Form D filings", stats["filings_found"]),
            ("Passed the ICP filter", stats["after_filter"]),
            ("AI-native (yes or maybe)", int(form_d["verdict"].isin(["yes", "maybe"]).sum())),
            ("Prospects (tier 1 to 3)", len(form_d_ranked)),
            ("Tier 1", int((form_d_ranked["account_segment"] == "tier_1").sum())),
        ]
        footnote = f"Plus {len(targets)} target accounts ({int((targets['account_segment'] == 'tier_1').sum())} tier 1)."
        st.markdown(ui.pipeline_html(steps, footnote), unsafe_allow_html=True)
    st.write("")
    with st.container(key="card_top10"):
        ui.label("Top 10")
        # Rank and favicon, then the name as a text link that opens the company, then its segment.
        for i, (_, row) in enumerate(tabs.top10(results).iterrows(), start=1):
            name = deep_dive.search_name(row["company_name"], row["company_display_name"])
            segment = row["account_segment"]
            with st.container(key=f"t10row_{i}"):
                rank_col, name_col, pill_col = st.columns([0.55, 3, 1.1], vertical_alignment="center", gap="small")
                rank_col.markdown(f'<div class="rank">{i}<img src="{ui.favicon_for(row.get("website"), favicons)}"></div>',
                                  unsafe_allow_html=True)
                name_col.button(name, key=f"top10_{ui.key_id(row['filing_url'])}",
                                on_click=open_company, args=(row["filing_url"], row["tab"]))
                pill_col.markdown(pill(SEGMENT_SHORT[segment], SEGMENT_COLORS[segment]), unsafe_allow_html=True)


main_col, side_col = st.columns([2.6, 1], gap="medium")
with main_col:
    if page == RESEARCH:
        show_research_page()
    else:
        show_list_page(page)
with side_col, st.container(key="sticky_side"):
    show_right_column()

st.divider()
st.markdown(
    '<div class="footer">Independent portfolio project, not affiliated with Baseten. No emails were sent. · '
    f'<a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a> · <a href="{GITHUB_URL}" target="_blank">GitHub</a></div>',
    unsafe_allow_html=True,
)
with st.expander("How the checks work"):
    st.caption(
        "Each evidence quote is checked by code to appear word for word in the company's website "
        "(or in cited search snippets); if it doesn't, the verdict is downgraded. Verdicts based only "
        "on search results are capped at \"maybe\". Segments and contact roles are chosen by fixed rules "
        "from that verified evidence; Claude only writes the wording. Contacts and their titles are copied "
        "from the Form D filing by code. Baseten claims must come from a fixed fact sheet, and customer "
        "names are checked against Baseten's customers page. Research sources are checked against the "
        "pages actually read. Anything inferred is listed under Assumptions."
    )
