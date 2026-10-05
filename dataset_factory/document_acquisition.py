"""Bounded, serial, cache-first acquisition for the explicitly admitted text pilot."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/document_pilot/2026-10-02"
RUN = ROOT / "runs/document-pilot-2026-10-02"
AUDIT = ROOT / "runs/source-access-audit-2026-10-02"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BoundedClient:
    """A hard byte cap and host spacing remain effective across cached responses."""

    def __init__(self, cache: Path, maximum: int = 100_000_000):
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.maximum = maximum
        self.bytes = 0
        # Cached/imported bytes conservatively consume the same pilot budget,
        # so retrying the bounded acquisition cannot reset its download allowance.
        self.initial_bytes = sum(path.stat().st_size for path in cache.iterdir() if path.is_file())
        if self.initial_bytes > maximum:
            raise ValueError("Existing cache exceeds the pilot byte budget")
        self.last: dict[str, float] = {}
        self.records: list[dict] = []
        contact = subprocess.check_output(
            ["git", "config", "--global", "--get", "user.email"], text=True
        ).strip()
        if not contact or "@" not in contact:
            raise ValueError("A real contact is required for Wikimedia acquisition")
        self.user_agent = f"HorschtDocumentPilot/0.1 (bounded research; mailto:{contact})"

    def get(self, url: str, name: str, *, delay: float = 1) -> bytes:
        target = self.cache / name
        if target.exists():
            data = target.read_bytes()
            self.records.append({"url": url, "bytes": len(data), "sha256": sha(data), "cached": True})
            return data
        host = urlparse(url).netloc
        time.sleep(max(0, delay - (time.monotonic() - self.last.get(host, 0))))
        with requests.get(url, headers={"User-Agent": self.user_agent}, timeout=45, stream=True) as response:
            self.last[host] = time.monotonic()
            response.raise_for_status()
            blocks = []
            for block in response.iter_content(65536):
                self.bytes += len(block)
                if self.initial_bytes + self.bytes > self.maximum:
                    raise ValueError("Network byte budget exceeded")
                blocks.append(block)
            data = b"".join(blocks)
        target.write_bytes(data)
        self.records.append({"url": url, "bytes": len(data), "sha256": sha(data), "cached": False})
        return data


def clean_article(html: str) -> tuple[str, dict]:
    soup = BeautifulSoup(html, "html.parser")
    removed = {"media": 0, "quoted_blocks": 0}
    for tag in list(soup.select("figure, .wp-caption, iframe, script, style, img, video, audio")):
        if tag.parent is not None:
            tag.decompose()
            removed["media"] += 1
    # Third-party quotation blocks have separate rights; the pilot uses the reporter's prose.
    for tag in soup.select("blockquote"):
        tag.decompose()
        removed["quoted_blocks"] += 1
    paragraphs = [tag.get_text(" ", strip=True) for tag in soup.find_all(["p", "h2", "h3", "li"])]
    return "\n\n".join(p for p in paragraphs if p), removed


def acquire() -> list[dict]:
    raw = OUTPUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    client = BoundedClient(raw)
    policy_url = "https://globalvoices.org/about/global-voices-attribution-policy/"
    policy = client.get(policy_url, "gv-policy.html", delay=10)
    policy_text = BeautifulSoup(policy, "html.parser").get_text(" ", strip=True)
    if "Creative Commons" not in policy_text or "Attribution" not in policy_text:
        raise ValueError("Global Voices rights policy not recognized")
    now = datetime.now(timezone.utc).isoformat()
    wiki_policy_url = "https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_API_Usage_Guidelines"
    wiki_policy = client.get(wiki_policy_url, "wikimedia-api-policy.html")
    documents = []
    inventory = {"Global Voices": [], "Wikipedia": [], "scope": "all other editions deferred: item/host preflight not performed in this bounded six-language pilot"}
    for language, index in (("es", 1), ("ar", 2), ("fr", 2)):
        feed_bytes = (AUDIT / f"gv_{language}_feed.txt").read_bytes()
        (raw / f"gv_{language}_feed.xml").write_bytes(feed_bytes)
        item = ET.fromstring(feed_bytes).findall("./channel/item")[index]
        url = item.findtext("link")
        robots_bytes = (AUDIT / f"gv_{language}_robots.txt").read_bytes()
        robots = RobotFileParser()
        robots.parse(robots_bytes.decode().splitlines())
        if not robots.can_fetch(client.user_agent, url):
            raise ValueError("Robots disallows item")
        page = client.get(url, f"gv_{language}_item.html", delay=10)
        soup = BeautifulSoup(page, "html.parser")
        licenses = sorted({a["href"] for a in soup.select("a[href]") if "creativecommons.org/licenses/by/3.0" in a["href"]})
        if not licenses:
            raise ValueError("No admitted item license")
        credits = [t.get_text(" ", strip=True) for t in soup.select(".avatar-credits-container .contributor-name")]
        if not credits:
            raise ValueError("Missing contributor attribution")
        text, extraction = clean_article(item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded"))
        translation_links = sorted({a["href"] for a in soup.select(".post-translations a[href]")})
        original = next((u for u in translation_links if urlparse(u).netloc == "globalvoices.org"), url)
        inventory["Global Voices"] = sorted({a["href"] for a in soup.select(".gv-header-site-translations a[href]")})
        documents.append({"id": f"gv-{language}", "source_family": "reporting", "publisher": "Global Voices", "language": language,
            "title": item.findtext("title"), "url": url, "revision": f"feed-sha256:{sha(feed_bytes)};item-sha256:{sha(page)}",
            "retrieved_at": now, "published_at": parsedate_to_datetime(item.findtext("pubDate")).isoformat(), "content_sha256": sha(text.encode()), "text": text,
            "attribution": credits, "license": {"id": "CC-BY-3.0", "url": licenses[0], "share_alike": False},
            "rights_evidence": {"checked_at": now, "policy_url": policy_url, "policy_sha256": sha(policy), "item_license_url": licenses[0], "item_sha256": sha(page), "route": "cached official full-text RSS + live item rights", "robots_sha256": sha(robots_bytes)},
            "relation_keys": [f"original:{original}", "event:kazakhstan-water" if language in {"es", "ar"} else "event:folklore-indigenous-recording"],
            "original_translation": "human_translation", "original_url": original, "translations": translation_links, "extraction": extraction,
            "changes": "HTML/media/captions/third-party blockquotes removed; paragraph separators normalized; original-language prose retained"})
    for language, title in (("de", "Solarenergie"), ("ja", "自転車"), ("pt", "Manguezal")):
        host = f"https://{language}.wikipedia.org"
        robots_data = client.get(host + "/robots.txt", f"wiki-{language}-robots.txt")
        # The published official API policy is the route authority; website robots are retained as evidence.
        params = {"action": "query", "meta": "siteinfo", "siprop": "general|rightsinfo", "titles": title, "redirects": 1, "prop": "extracts|revisions|contributors|info", "explaintext": 1, "rvprop": "ids|timestamp", "inprop": "url", "pclimit": "max", "format": "json"}
        url = requests.Request("GET", host + "/w/api.php", params=params).prepare().url
        data = client.get(url, f"wiki-{language}-resolved-query.json")
        query = json.loads(data)
        rights = query["query"]["rightsinfo"]
        if "creativecommons.org/licenses/by-sa/4.0" not in rights.get("url", ""):
            raise ValueError("Unexpected Wikimedia edition rights")
        page = next(iter(query["query"]["pages"].values()))
        text = page["extract"]
        if len(text) < 3000 or len(text) > 250000:
            raise ValueError("Document outside bounded text range")
        revision = page["revisions"][0]
        documents.append({"id": f"wiki-{language}", "source_family": "encyclopedia", "publisher": "Wikipedia", "language": language, "title": page["title"], "url": page["fullurl"],
            "revision": str(revision["revid"]), "source_updated_at": revision["timestamp"], "retrieved_at": now, "content_sha256": sha(text.encode()), "text": text,
            "attribution": [f"Wikipedia contributors; revision https://{language}.wikipedia.org/w/index.php?oldid={revision['revid']}", f"Full contributors/history: {host}/w/index.php?title={page['title']}&action=history"],
            "license": {"id": "CC-BY-SA-4.0", "url": rights["url"], "share_alike": True},
            "rights_evidence": {"checked_at": now, "policy_url": wiki_policy_url, "policy_sha256": sha(wiki_policy), "item_license_url": rights["url"], "route": "official MediaWiki query API; edition rightsinfo", "robots_sha256": sha(robots_data), "response_sha256": sha(data)},
            "relation_keys": [f"wikipedia-page:{language}:{page['pageid']}"], "original_translation": "source_native_edition", "extraction": {"method": "MediaWiki plaintext extract", "sections_preserved": True}, "changes": "MediaWiki plaintext extraction; media omitted; full returned extract retained"})
    matrix_url = "https://en.wikipedia.org/w/api.php?action=sitematrix&smtype=language&format=json"
    matrix_data = client.get(matrix_url, "wikipedia-edition-inventory.json")
    matrix = json.loads(matrix_data).get("sitematrix", {})
    inventory["Wikipedia"] = sorted({x["code"] for x in matrix.values() if isinstance(x, dict) and any(y.get("code") == "wiki" for y in x.get("site", []))})
    (OUTPUT / "source_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    (OUTPUT / "documents.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in documents))
    (RUN / "acquisition.json").write_text(json.dumps({"network_bytes": client.bytes, "requests": client.records, "documents": len(documents), "max_network_bytes": client.maximum}, indent=2) + "\n")
    return documents


if __name__ == "__main__":
    docs = acquire()
    print(json.dumps({"documents": len(docs), "languages": [d["language"] for d in docs], "text_chars": [len(d["text"]) for d in docs]}))
