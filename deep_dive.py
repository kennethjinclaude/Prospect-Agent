"""One-click research links for a company. These are only URLs: nothing is fetched."""

import re
from urllib.parse import quote_plus, urlparse

# Legal endings removed so searches use the name people actually know.
_LEGAL_ENDING = re.compile(
    r"[,\s]+(inc|incorporated|llc|corp|corporation|co|company|ltd|limited|lp|pbc|plc)\.?$",
    re.IGNORECASE,
)

PEOPLE_ROLES = ["Head of ML", "ML infrastructure", "CTO", "VP Engineering"]


def search_name(company_name, display_name=None):
    """The name to search for: the brand from the brief if known, else the filing name
    without endings like ", Inc." (e.g. "Brightline Interactive, Inc./NV" -> "Brightline Interactive")."""
    if isinstance(display_name, str) and display_name.strip():
        return display_name.strip()
    name = re.sub(r"/[A-Z]{2}$", "", company_name.strip())  # state suffix like "/NV"
    while _LEGAL_ENDING.search(name):
        name = _LEGAL_ENDING.sub("", name)
    return name.strip()


def _google(query):
    return f"https://www.google.com/search?q={quote_plus(query)}"


def _linkedin(kind, keywords):
    return f"https://www.linkedin.com/search/results/{kind}/?keywords={quote_plus(keywords)}"


def people_search(company_name, role_words, display_name=None):
    """LinkedIn people search for a role at this company, e.g. "Exa CTO"."""
    return _linkedin("people", f"{search_name(company_name, display_name)} {role_words}")


def links(company_name, website=None, display_name=None):
    """Research links grouped for display: {group: [(label, url), ...]}.

    The company name is put in quotes for Google, so a common-word name like
    "Relay" doesn't match unrelated pages.
    """
    name = search_name(company_name, display_name)
    quoted = f'"{name}"'
    groups = {
        "People": [(role, _linkedin("people", f"{name} {role}")) for role in PEOPLE_ROLES],
        "Hiring signals": [
            ("ML infra jobs", _google(f'{quoted} inference engineer OR "ML infrastructure" jobs')),
        ],
        "Model work": [
            ("Fine-tuning and open models", _google(f"{quoted} blog post-training OR fine-tuning OR open-source models")),
        ],
        "Vendors": [
            ("Inference vendors", _google(f'{quoted} Baseten OR "Together AI" OR Fireworks OR Modal')),
        ],
        # Google's news filter has "past 3 months", the closest option to 90 days.
        "News": [("News, past 3 months", f"https://www.google.com/search?q={quote_plus(quoted)}&tbm=nws&tbs=qdr:m3")],
        "Company": [("LinkedIn company page", _linkedin("companies", name))],
    }
    if isinstance(website, str) and website.startswith("http"):
        domain = urlparse(website).netloc.removeprefix("www.")
        groups["Company"] += [
            ("Website", website),
            # We don't know each careers page address, so search their site for it.
            ("Careers page", _google(f"site:{domain} careers OR jobs")),
        ]
    return groups
