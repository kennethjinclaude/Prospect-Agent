"""Visual building blocks for the app: colors, pills, icons, favicons, styles, the pipeline list, and the pager.

The look is quiet and dense: neutral grays, one accent color (deep green), and
1px dividers instead of boxes. The styles use Streamlit's container classes
(st-key-<key>), which are built in but could change in a future Streamlit
version. If they do, the page still works; it just falls back to plain styling.
"""

import hashlib
import html
import json

import pandas as pd
import streamlit as st

import manual

ACCENT = "#0F766E"  # deep green: only the selected tab, primary buttons, and the Tier 1 pill
ACCENT_SOFT, ACCENT_EDGE, ACCENT_DARK = "#E6F4F1", "#B7DFD7", "#115E59"  # the selected-bubble fill, border, and text
TEXT, MUTED, FAINT, LINE = "#111827", "#6B7280", "#9CA3AF", "#E5E7EB"

# Pill colors. Only Tier 1 gets the accent; everything else is a shade of gray.
SEGMENT_COLORS = {
    "tier_1": ACCENT,
    "tier_2": "#374151",
    "tier_3": "#4B5563",
    "existing_customer": "#4B5563",
    "needs_review": MUTED,
    "deprioritize": FAINT,
}
SEGMENT_SHORT = {
    "tier_1": "Tier 1", "tier_2": "Tier 2", "tier_3": "Tier 3",
    "existing_customer": "Customer", "needs_review": "Review", "deprioritize": "Deprioritized",
}
SEGMENT_LABELS = {
    "tier_1": "Tier 1: AI-native scale-up",
    "tier_2": "Tier 2: enterprise or regulated",
    "tier_3": "Tier 3: early-stage, nurture",
    "existing_customer": "Existing Baseten customer: route to account management",
    "needs_review": "Needs review: check the stage by hand",
    "deprioritize": "Deprioritized",
}
GRADE_COLORS = {"A": "#374151", "B": MUTED, "C": FAINT}
GRAY = MUTED  # for pills that carry no ranking meaning (Manual, Just raised, confidence)
SLOT_LABELS = {
    "champion_and_buyer": "Champion and economic buyer",
    "champion": "Champion",
    "economic_buyer": "Economic buyer",
    "blocker": "Possible blocker",
}

# Icons from Lucide (lucide.dev, ISC license), drawn in the text color around them.
_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'style="vertical-align:-1px">{paths}</svg>')
ICON_EXTERNAL = _SVG.format(size=11, paths='<path d="M15 3h6v6"/><path d="M10 14 21 3"/>'
                            '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>')
ICON_WARNING = _SVG.format(size=13, paths='<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
                           '<path d="M12 9v4"/><path d="M12 17h.01"/>')
# The search icon sits inside the search box as a background image, so it's written as a data URL.
_SEARCH_ICON_URL = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' "
                    "viewBox='0 0 24 24' fill='none' stroke='%239CA3AF' stroke-width='2' stroke-linecap='round' "
                    "stroke-linejoin='round'><circle cx='11' cy='11' r='8'/><path d='m21 21-4.3-4.3'/></svg>")

# A light gray square, shown when a company has no favicon.
BLANK_ICON = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16'>"
              "<rect width='16' height='16' rx='3' fill='%23E5E7EB'/></svg>")

ROWS_PER_PAGE = 25

# Buttons inside containers with these key prefixes are drawn as plain text links.
TEXT_BUTTON_KEYS = ["open_", "top10_", "close_panel", "pg_"]
_text_buttons = ", ".join(f'[class*="st-key-{k}"] button' for k in TEXT_BUTTON_KEYS)

# Option groups drawn as bubbles: the top nav, and the segment and category chips on each list page.
BUBBLE_KEYS = ["nav", "segments_", "categories_"]
_bubble_groups = ", ".join(f'[class*="st-key-{k}"] [data-testid="stButtonGroup"]' for k in BUBBLE_KEYS)
_bubbles = ", ".join(f'[class*="st-key-{k}"] button' for k in BUBBLE_KEYS)
_bubbles_on = ", ".join(f'[class*="st-key-{k}"] button[data-selected="true"]' for k in BUBBLE_KEYS)
_bubbles_on_text = ", ".join(f'{selector} p' for selector in _bubbles_on.split(", "))

CSS = f"""<style>
/* Base: tabular numbers so columns of figures line up, and a slightly tighter line height. */
.stApp {{font-variant-numeric: tabular-nums;}}
.stApp p, .stApp li {{line-height: 1.45;}}
[data-testid="stMainBlockContainer"] {{padding-top: 2.75rem; max-width: 1480px;}}
h1, h2, h3, h4, h5 {{letter-spacing: -0.01em;}}

/* Header: the wordmark, then the nav. */
.wordmark {{font-size: 16px; font-weight: 650; color: {TEXT}; white-space: nowrap;}}
.wordmark .meta {{font-size: 12px; font-weight: 400; color: {MUTED}; margin-left: 8px;}}
.wordmark .tag {{font-size: 11px; font-weight: 500; color: {MUTED}; background: #F3F4F6; border-radius: 4px;
                 padding: 1px 6px; margin-left: 6px;}}

/* Bubbles: the nav and the segment/category chips. Each option is a soft rounded pill: plain gray text,
   a light gray bubble on hover, and a filled light-green bubble when selected (Streamlit marks it data-selected). */
{_bubble_groups} {{gap: 4px !important;}}
{_bubbles} {{
    background: transparent !important; border: 1px solid transparent !important; box-shadow: none !important;
    border-radius: 999px !important; color: {MUTED} !important; padding: 3px 12px !important; min-height: 30px !important;
}}
{_bubbles.replace(" button", " button:hover")} {{background: #F3F4F6 !important; color: {TEXT} !important;}}
{_bubbles_on} {{background: {ACCENT_SOFT} !important; border-color: {ACCENT_EDGE} !important; color: {ACCENT_DARK} !important;}}
{_bubbles_on_text} {{font-weight: 600; color: {ACCENT_DARK};}}
/* The count inside each nav pill, as a small gray badge. */
.st-key-nav button p span {{background: #F3F4F6 !important; color: {MUTED} !important; font-size: 11px; font-weight: 500;
                            border-radius: 999px; padding: 0 6px; margin-left: 6px;}}
.st-key-nav button[data-selected="true"] p span {{background: #FFFFFF !important;
                            color: {ACCENT_DARK} !important;}}
.header-rule {{border-bottom: 1px solid {LINE}; margin: -6px 0 6px;}}
.strip {{font-size: 12px; color: {MUTED};}}
.strip .arrow {{color: {FAINT}; margin: 0 6px;}}

/* Small gray labels: column headers and section names. */
.label, .row-head {{font-size: 11px; font-weight: 500; color: {MUTED}; text-transform: uppercase; letter-spacing: .04em;
                    white-space: nowrap;}}
.row-head {{padding: 0 4px;}}

/* Company rows: a plain list, 1px divider under each row, faint hover. */
[class*="st-key-list_"] {{gap: 0 !important; border-top: 1px solid {LINE};}}
[class*="st-key-row_"] {{border-bottom: 1px solid {LINE}; padding: 4px 4px; min-height: 44px; gap: 0 !important;}}
[class*="st-key-row_"]:hover {{background: #F9FAFB;}}
[class*="st-key-row_"][class*="_sel"] {{background: #F3F4F6;}}
[class*="st-key-row_"] p {{margin: 0;}}
.co {{display: flex; align-items: center; gap: 10px; min-width: 0;}}
.co img {{width: 18px; height: 18px; border-radius: 4px; flex: none;}}
.co-name {{font-weight: 500; color: {TEXT};}}
.co-sub, .muted {{font-size: 12px; color: {MUTED};}}
.co > div {{min-width: 0;}}
.co-sub {{white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
.co-sub a, .ext {{color: {MUTED} !important; text-decoration: none; white-space: nowrap;}}
.rank-cell {{white-space: nowrap; overflow-wrap: normal; word-break: normal;}}
.co-sub a:hover, .ext:hover {{color: {TEXT} !important; text-decoration: underline;}}
.why {{font-size: 13px; color: #374151; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;}}
.pill {{display: inline-block; padding: 0 8px; border-radius: 999px; border: 1px solid {LINE}; background: #F9FAFB;
        font-size: 11.5px; font-weight: 500; line-height: 18px; margin-right: 4px; white-space: nowrap;}}
.pill.on {{background: {ACCENT_SOFT}; border-color: {ACCENT_EDGE}; color: {ACCENT_DARK};}}
.count {{font-size: 15px; font-weight: 600; color: {TEXT};}}
.count span {{font-size: 12px; font-weight: 400; color: {MUTED}; margin-left: 8px;}}

/* Text-link buttons (Open, Top 10 names, Close, pager). */
{_text_buttons} {{
    background: transparent !important; border: none !important; box-shadow: none !important;
    padding: 0 !important; min-height: 0 !important; color: #374151 !important; justify-content: flex-start;
}}
{_text_buttons.replace(" button", " button:hover")} {{color: {TEXT} !important; text-decoration: underline;}}
[class*="st-key-open_"] button p {{font-size: 13px; text-decoration: underline; text-decoration-color: #D1D5DB;
                                   text-underline-offset: 3px;}}
[class*="st-key-top10_"] button p {{font-weight: 500; color: {TEXT};}}
[class*="st-key-pg_"] button p {{font-size: 13px;}}

/* Search box with a search icon inside it. */
[class*="st-key-search_"] input {{background: url("{_SEARCH_ICON_URL}") no-repeat 9px center; padding-left: 30px;}}

/* The one box on the page: the opened company's panel. Expanders inside it become plain dividers. */
.st-key-card_detail {{border: 1px solid {LINE}; border-radius: 8px; padding: 16px; background: #FFFFFF;}}
.st-key-card_detail [data-testid="stExpander"] details {{border: none; border-top: 1px solid {LINE}; border-radius: 0;}}
.detail-name {{font-size: 18px; font-weight: 600; color: {TEXT}; vertical-align: middle;}}

/* The right column stays in view while the list scrolls, separated by a thin line. */
.st-key-sticky_side {{position: sticky; top: 3.5rem;}}

/* Pipeline: label left, number and share right, a thin bar under each. */
.stage {{margin: 0 0 9px;}}
/* Every row stays on one line: the label shortens with "…" if it must, the numbers never wrap. */
.stage-top {{display: flex; justify-content: space-between; gap: 8px; font-size: 13px; color: #374151; margin-bottom: 3px;
             white-space: nowrap;}}
.stage-top > span:first-child {{min-width: 0; overflow: hidden; text-overflow: ellipsis;}}
.stage-top > span:last-child {{flex: none;}}
/* Streamlit lets text break anywhere; numbers must never break. */
.stage-top, .stage-top * {{white-space: nowrap !important; overflow-wrap: normal !important; word-break: normal !important;}}
.stage-top b {{color: {TEXT}; font-weight: 600;}}
.stage-top .share {{color: {FAINT}; display: inline-block; width: 54px; text-align: right;}}
.bar {{background: #F3F4F6; height: 3px; border-radius: 2px;}}
.bar > div {{background: {FAINT}; height: 3px; border-radius: 2px;}}

/* Top 10: rank and favicon before the name. */
.rank {{display: flex; align-items: center; gap: 8px; color: {FAINT}; font-size: 12px;}}
.rank img {{width: 16px; height: 16px; border-radius: 3px;}}
/* Top 10 rows: tight, with the same 1px divider and hover as the main list. */
.st-key-card_top10 {{gap: 0 !important;}}
.st-key-card_top10 > div:first-child {{margin-bottom: 6px;}}
[class*="st-key-t10row_"] {{border-bottom: 1px solid {LINE}; padding: 3px 2px; min-height: 0; gap: 0 !important;}}
[class*="st-key-t10row_"]:hover {{background: #F9FAFB;}}
[class*="st-key-t10row_"] [data-testid="stHorizontalBlock"] {{min-height: 0; gap: 8px;}}
[class*="st-key-t10row_"] p, [class*="st-key-t10row_"] [data-testid="stMarkdownContainer"] {{margin: 0;}}
[class*="st-key-t10row_"] [data-testid="stElementContainer"] {{margin: 0;}}

.footer {{font-size: 12px; color: {FAINT};}}
.footer a {{color: {MUTED} !important;}}
</style>"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text):
    """Make text safe to put inside HTML."""
    return html.escape("" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text))


def md(text):
    """Escape dollar signs for st.markdown, which reads "$950M at a $15B" as a math formula."""
    return text.replace("$", "\\$") if isinstance(text, str) else text


def md_all(data):
    """md() on every string in a dict or list (from a saved row or brief), leaving links alone."""
    if isinstance(data, dict):
        return {k: v if any(word in k for word in ("url", "website", "linkedin", "page")) else md_all(v)
                for k, v in data.items()}
    if isinstance(data, list):
        return [md_all(item) for item in data]
    if isinstance(data, str) and data[:1] in "[{":
        return data  # saved JSON (evidence lists): escaped after it's read, by the code that reads it
    return md(data)


def key_id(text):
    """A short, stable id for widget keys (filing URLs contain characters CSS class names can't)."""
    return hashlib.md5(str(text).encode()).hexdigest()[:10]


def pill(text, color):
    """A small rounded tag, as HTML, matching the chip bubbles. The accent color (Tier 1) gets the green bubble;
    any other color is used for the text on a light gray bubble."""
    if color == ACCENT:
        return f'<span class="pill on">{esc(text)}</span>'
    return f'<span class="pill" style="color:{color}">{esc(text)}</span>'


def label(text):
    """A small gray uppercase section name."""
    st.markdown(f'<div class="label">{esc(text)}</div>', unsafe_allow_html=True)


def icon_link(url, title):
    """Just the external-link arrow, with a tooltip, opening in a new tab."""
    return f'<a class="ext" href="{esc(url)}" target="_blank" title="{esc(title)}">{ICON_EXTERNAL}</a>'


def external_link(text, url):
    """A small gray link with an arrow icon, opening in a new tab."""
    return f'<a class="ext" href="{esc(url)}" target="_blank">{esc(text)} {ICON_EXTERNAL}</a>'


def pretty_workload(text):
    """Workload label for display: "llm_text; other" -> "llm text; General AI"."""
    if not isinstance(text, str):
        return ""
    return "; ".join("General AI" if part.strip() == "other" else part.strip().replace("_", " ") for part in text.split(";"))


def money_millions(amount):
    return f"${amount / 1e6:,.1f}M" if pd.notna(amount) else "not stated"


def load_favicons(folder):
    """{domain: small embedded image} saved by favicons.py, or {} if not built yet."""
    path = folder / "favicons.json"
    return json.loads(path.read_text()) if path.exists() else {}


def favicon_for(website, icons):
    """The company's saved favicon, or a blank square if there isn't one."""
    icon = icons.get(manual.domain_of(website), "") if isinstance(website, str) and website else ""
    return icon or BLANK_ICON


def company_cell(name, icon, tags_html, subtitle_html):
    """Favicon + name, small tags, and a muted line underneath (subtitle_html is already escaped)."""
    return (f'<div class="co"><img src="{icon}"><div><span class="co-name">{esc(name)}</span> {tags_html}'
            f'<div class="co-sub">{subtitle_html}</div></div></div>')


def why_now_cell(text):
    """Why-now text cut to two lines, with the full text on hover."""
    return f'<div class="why" title="{esc(text)}">{esc(text) or "&nbsp;"}</div>'


def pipeline_html(steps, footnote):
    """The funnel as a compact list. steps: [(label, count)], each bar sized against the first."""
    top = steps[0][1] or 1
    rows = []
    for name, count in steps:
        share = count / top * 100
        rows.append(
            f'<div class="stage"><div class="stage-top"><span>{esc(name)}</span>'
            f'<span><b>{count:,}</b><span class="share">{share:.1f}%</span></span></div>'
            f'<div class="bar"><div style="width:{max(share, 1):.1f}%"></div></div></div>'
        )
    return "".join(rows) + f'<div class="muted">{esc(footnote)}</div>'


def page_slice(key, total, per_page=ROWS_PER_PAGE):
    """(start, end) of the rows on the current page. Call before drawing the rows."""
    pages = max(1, -(-total // per_page))
    page = min(max(st.session_state.get(key, 1), 1), pages)
    st.session_state[key] = page
    start = (page - 1) * per_page
    return start, min(start + per_page, total)


def _turn_page(key, step):
    st.session_state[key] = st.session_state.get(key, 1) + step


def pager_controls(key, total, per_page=ROWS_PER_PAGE):
    """Previous / Page x of y / Next, under the rows, when there's more than one page."""
    pages = max(1, -(-total // per_page))
    if pages == 1:
        return
    page = st.session_state.get(key, 1)
    prev_col, info_col, next_col = st.columns([1, 6, 1], vertical_alignment="center")
    prev_col.button("Previous", key=f"{key}_prev", disabled=page <= 1, on_click=_turn_page, args=(key, -1))
    next_col.button("Next", key=f"{key}_next", disabled=page >= pages, on_click=_turn_page, args=(key, 1))
    info_col.markdown(f'<div style="text-align:center" class="muted">Page {page} of {pages}</div>', unsafe_allow_html=True)
