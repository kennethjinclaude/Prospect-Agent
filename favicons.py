"""Download each company's favicon once, from its own website, and save it as a small embedded image.

Saved in data/favicons.json as {domain: "data:image/...;base64,..."} (or "" if none
was found), so the app shows icons without contacting any outside service.
Run it with:  python favicons.py
"""

import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image

import manual
import qualify

FAVICONS_FILE = Path(__file__).parent / "data" / "favicons.json"
MAX_BYTES = 60_000  # skip huge images; a favicon is usually 1-10 KB
IMAGE_TYPES = ("image/png", "image/x-icon", "image/vnd.microsoft.icon", "image/svg+xml", "image/jpeg", "image/gif", "image/webp")
HEADERS = {"User-Agent": qualify.BROWSER_USER_AGENT}


def _icon_candidates(website):
    """Icon URLs to try: those the homepage declares, then /favicon.ico."""
    candidates = []
    try:
        page = requests.get(website, timeout=6, headers=HEADERS)
        soup = BeautifulSoup(page.text, "html.parser")
        for link in soup.find_all("link", href=True):
            rel = " ".join(link.get("rel", [])).lower()
            if "icon" in rel:
                candidates.append(urljoin(page.url, link["href"]))
    except requests.RequestException:
        pass
    candidates.append(urljoin(website, "/favicon.ico"))
    return candidates


ICON_SIZE = 32  # pixels; icons are shrunk to this so the table stays fast
MAX_SVG_BYTES = 8_000


def shrink(data_url):
    """Resize an icon to 32x32 PNG so it stays small (about 1-2 KB). SVGs are kept only if small."""
    header, _, encoded = data_url.partition(",")
    raw = base64.b64decode(encoded)
    if "svg" in header:
        return data_url if len(raw) <= MAX_SVG_BYTES else ""
    try:
        image = Image.open(io.BytesIO(raw))
        image = image.convert("RGBA")
        image.thumbnail((ICON_SIZE, ICON_SIZE))
        out = io.BytesIO()
        image.save(out, format="PNG", optimize=True)
        return f"data:image/png;base64,{base64.b64encode(out.getvalue()).decode()}"
    except (OSError, ValueError):
        return ""  # not an image Pillow can read: use the blank fallback


def fetch_favicon(website):
    """The favicon as a data URL, or "" if none could be downloaded."""
    for url in _icon_candidates(website):
        if url.startswith("data:image/"):
            return shrink(url)
        try:
            response = requests.get(url, timeout=6, headers=HEADERS)
        except requests.RequestException:
            continue
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if response.ok and content_type in IMAGE_TYPES and 0 < len(response.content) <= MAX_BYTES:
            return shrink(f"data:{content_type};base64,{base64.b64encode(response.content).decode()}")
    return ""


def build_favicons():
    """Fetch icons for every company with a website that isn't saved yet. Returns {domain: data URL}."""
    saved = json.loads(FAVICONS_FILE.read_text()) if FAVICONS_FILE.exists() else {}
    websites = pd.read_csv(qualify.RESULTS_FILE)["website"].dropna().tolist()
    if manual.MANUAL_FILE.exists():
        websites += pd.read_csv(manual.MANUAL_FILE)["website"].dropna().tolist()
    todo = {manual.domain_of(w): manual.normalize_website(w) for w in websites}
    todo = {domain: site for domain, site in todo.items() if domain not in saved}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for domain, icon in zip(todo, pool.map(fetch_favicon, todo.values())):
            saved[domain] = icon
    FAVICONS_FILE.parent.mkdir(exist_ok=True)
    FAVICONS_FILE.write_text(json.dumps(saved))
    return saved


if __name__ == "__main__":
    icons = build_favicons()
    found = sum(1 for icon in icons.values() if icon)
    print(f"{found} of {len(icons)} companies have a favicon ({FAVICONS_FILE.stat().st_size / 1024:.0f} KB saved)")
