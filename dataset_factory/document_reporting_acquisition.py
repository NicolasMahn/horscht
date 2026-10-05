"""Bounded acquisition of Global Voices reporting originals and human translations.

This is an isolated expansion batch. It reads the saved publisher inventory and
the three eligible pilot feeds, checks robots and each item's live CC BY footer,
then stores source pages and admitted prose separately with provenance.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup

from dataset_factory.document_acquisition import BoundedClient, clean_article, sha

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "data/document_expansion/2026-10-02"
RAW = BATCH / "raw/reporting"
DOCS = BATCH / "documents"
JSONL = BATCH / "reporting-documents.jsonl"
RUN = ROOT / "runs/document-expansion-2026-10-02"
STATUS = RUN / "acquisition-reporting-status.json"
MANIFEST = BATCH / "reporting-acquisition.json"
PILOT = ROOT / "data/document_pilot/2026-10-02"
AUDIT = ROOT / "runs/source-access-audit-2026-10-02"
INVENTORY = PILOT / "source_inventory.json"
POLICY_URL = "https://globalvoices.org/about/global-voices-attribution-policy/"
MAX_NETWORK_BYTES = 100_000_000
TARGET_LANGUAGES = ("es", "ar", "fr", "pt", "ja", "de", "it", "ru", "zh", "hi", "bn", "tr")
HOST_CODES = {"ja": "jp", "zh": "zhs"}
LANGUAGE_NAMES = {
    "en": ("english",), "es": ("spanish", "español", "espanol"),
    "ar": ("arabic", "عربي", "العربية"), "fr": ("french", "français", "francais"),
    "pt": ("portuguese", "português", "portugues"), "ja": ("japanese", "日本語", "日本"),
    "de": ("german", "deutsch"), "it": ("italian", "italiano"),
    "ru": ("russian", "русский", "русском"),
    "zh": ("chinese", "中文", "简体中文", "中文（简体）", "中文(简体)"),
    "hi": ("hindi", "हिन्दी", "हिंदी"), "bn": ("bengali", "বাংলা"),
    "tr": ("turkish", "türkçe", "turkce"),
}
TOPIC_TERMS = {
    "es": ("ambiente", "economía", "economia", "ciencia", "cultura", "gobierno", "gobernanza", "activismo", "derechos", "salud", "educación", "educacion", "tecnología", "tecnologia", "energía", "energia", "agricultura", "sociedad", "agua", "bosque", "pesca", "comunidad", "protesta", "elecciones", "ley", "periodismo", "indígena", "indigena"),
    "ar": ("بيئة", "البيئة", "الاقتصاد", "الأعمال", "العلوم", "الثقافة", "الحكومة", "الحكم", "النشاط", "الحقوق", "حقوق", "الصحة", "التعليم", "التكنولوجيا", "الطاقة", "الزراعة", "المجتمع", "المياه", "الماء", "المناخ", "الأسماك", "الاستزراع", "السكان", "الأصليون", "الاحتجاج", "الانتخابات", "القانون", "الصحافة", "الإنترنت"),
    "fr": ("environnement", "économie", "economie", "science", "culture", "gouvernance", "gouvernement", "activisme", "droits", "santé", "education", "éducation", "technologie", "énergie", "agriculture", "société"),
    "pt": ("ambiente", "economia", "ciência", "ciencia", "cultura", "governo", "governança", "ativismo", "direitos", "saúde", "educação", "tecnologia", "energia", "agricultura", "sociedade", "água", "agua", "floresta", "pesca", "comunidade", "protesto", "eleições", "lei", "jornalismo", "indígena", "indigena"),
    "ja": ("環境", "経済", "科学", "文化", "政治", "社会", "教育", "技術", "エネルギー", "農業", "権利", "水", "森林", "漁業", "地域", "抗議", "選挙", "法律", "報道", "先住民"),
    "de": ("umwelt", "wirtschaft", "wissenschaft", "kultur", "regierung", "politik", "rechte", "gesundheit", "bildung", "technologie", "energie", "landwirtschaft", "landwirt", "gesellschaft", "wasser", "wald", "fischerei", "gemeinschaft", "protest", "unterdrückung", "bewegung", "wahl", "gesetz", "journalismus", "indigen", "klima", "hitzewelle", "humanit", "hilfe"),
    "it": ("ambiente", "economia", "scienza", "cultura", "governo", "governance", "attivismo", "diritti", "salute", "istruzione", "tecnologia", "energia", "agricoltura", "società", "acqua", "foresta", "pesca", "comunità", "protesta", "elezioni", "legge", "giornalismo", "indigen"),
    "ru": ("экология", "окружающей среды", "экономика", "наука", "культура", "правительство", "общество", "права", "здоровье", "образование", "технологии", "энергетика", "сельское хозяйство", "вода", "лес", "рыболовство", "сообщество", "протест", "выборы", "закон", "журналистика", "коренные народы"),
    "zh": ("环境", "經濟", "经济", "科学", "文化", "治理", "政府", "社会", "教育", "技术", "能源", "农业", "权利", "水", "森林", "渔业", "社群", "抗议", "选举", "法律", "新闻", "原住民"),
    "hi": ("पर्यावरण", "अर्थव्यवस्था", "विज्ञान", "संस्कृति", "शासन", "सरकार", "समाज", "अधिकार", "स्वास्थ्य", "शिक्षा", "प्रौद्योगिकी", "ऊर्जा", "कृषि", "पानी", "वन", "मछली", "समुदाय", "विरोध", "चुनाव", "कानून", "पत्रकारिता", "आदिवासी"),
    "bn": ("পরিবেশ", "অর্থনীতি", "বিজ্ঞান", "সংস্কৃতি", "সরকার", "শাসন", "সমাজ", "অধিকার", "স্বাস্থ্য", "শিক্ষা", "প্রযুক্তি", "শক্তি", "কৃষি", "পানি", "বন", "মাছ", "সম্প্রদায়", "প্রতিবাদ", "নির্বাচন", "আইন", "সাংবাদিকতা", "আদিবাসী"),
    "tr": ("çevre", "ekonomi", "bilim", "kültür", "hükümet", "yönetişim", "aktivizm", "haklar", "sağlık", "eğitim", "teknoloji", "enerji", "tarım", "toplum", "su", "orman", "balıkçılık", "topluluk", "protesto", "seçim", "yasa", "gazetecilik", "yerli"),
}
PERSONAL_TERMS = (
    "biography", "biographies", "obituary", "obituaries", "profile", "personal life",
    "my life", "my story", "life story", "in memoriam", "remembering ", "autobiography",
    "spricht über das gefühl fehlender sicherheit", "speaks about the feeling of missing safety",
    "about the feeling of missing security", "opening up about her identity", "opening up about his identity",
    "سيرة ذاتية", "الحياة الشخصية", "حياتي", "biographie", "nécrologie", "vie personnelle",
    "biografía", "vida personal", "mi historia", "nachruf", "biografie", "私生活", "个人生活",
)
OUT_OF_SCOPE_TERMS = (
    "donation campaign", "fundraising campaign", "fundraiser", "support global voices",
    "spendenkampagne", "spendenaktion", "fundraising", "campagne de dons", "collecte de fonds",
    "campaña de donaciones", "campanha de doações", "campagna di raccolta fondi",
    "募金キャンペーン", "募捐", "প্রতিবাদী অনুদান", "দান সংগ্রহ",
)
REPRINT_TERMS = (
    "republished with permission", "reprinted with permission", "originally published by",
    "originally published on", "reprinted from", "republished from", "content provided by",
    "reproduced with permission", "content originally appeared on",
)
MIN_TEXT_CHARS = 1_200
MAX_TEXT_CHARS = 250_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def persisted_preflight_evidence(user_agent: str) -> tuple[int, list[dict]]:
    inventory = json.loads(INVENTORY.read_text())
    inventory_hosts = {urlparse(x).netloc for x in inventory["Global Voices"]}
    evidence = []
    for language in TARGET_LANGUAGES:
        code = HOST_CODES.get(language, language)
        host = f"{code}.globalvoices.org"
        robots_path = RAW / f"robots/{code}-robots.txt"
        feed_path = RAW / f"feeds/{code}-feed.xml"
        if not robots_path.exists() or not feed_path.exists():
            continue
        robots_bytes = robots_path.read_bytes()
        feed_bytes = feed_path.read_bytes()
        parser = RobotFileParser()
        parser.parse(robots_bytes.decode("utf-8", "replace").splitlines())
        feed_url = f"https://{host}/feed/"
        allowed = parser.can_fetch(user_agent, feed_url)
        evidence.append({
            "language": language,
            "publisher_host": host,
            "feed_url": feed_url,
            "robots_sha256": sha(robots_bytes),
            "feed_sha256": sha(feed_bytes),
            "robots_bytes": len(robots_bytes),
            "feed_bytes": len(feed_bytes),
            "feed_allowed_by_robots": allowed,
            "host_in_saved_inventory": host in inventory_hosts,
            "cached_pilot_feed": language in {"es", "ar", "fr"},
        })
    return len(inventory_hosts), evidence


def norm(value: str) -> str:
    return " ".join(value.casefold().split())


def record_status(*, phase: str, docs: list[dict], checked_hosts: list[str], rejected: dict[str, int], client: BoundedClient, target_count: int, network_bytes_prior_runs: int = 0) -> None:
    language_counts: dict[str, int] = {}
    for item in docs:
        language_counts[item["language"]] = language_counts.get(item["language"], 0) + 1
    inventory_count, preflight = persisted_preflight_evidence(client.user_agent)
    write_json(STATUS, {
        "state": "acquiring" if len(docs) < target_count else "acquired",
        "updated_at": utc_now(),
        "phase": phase,
        "publisher": "Global Voices",
        "target_documents": target_count,
        "admitted_documents": len(docs),
        "language_counts": language_counts,
        "target_languages": list(TARGET_LANGUAGES),
        "source_inventory_hosts": inventory_count,
        "checked_host_count": len(preflight),
        "checked_hosts": [row["publisher_host"] for row in preflight],
        "preflight_evidence": preflight,
        "rejected_counts_by_reason": rejected,
        "network_bytes_prior_runs": network_bytes_prior_runs,
        "network_bytes_this_run": client.bytes,
        "network_bytes_total": network_bytes_prior_runs + client.bytes,
        "network_bytes_cap": MAX_NETWORK_BYTES,
        "network_bytes_aim": 20_000_000,
        "raw_bytes_current_batch": sum(p.stat().st_size for p in RAW.rglob("*") if p.is_file()),
        "raw_directory": "data/document_expansion/2026-10-02/raw/reporting",
        "documents_jsonl": "data/document_expansion/2026-10-02/reporting-documents.jsonl",
        "reporting_budget_manifest": "data/document_expansion/2026-10-02/reporting-acquisition.json",
        "freeze_safe_initial_cohort": len(docs) >= 12,
        "external_benchmark_access": False,
    })


def append_document(doc: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    JSONL.parent.mkdir(parents=True, exist_ok=True)
    target = DOCS / f"{doc['id']}.json"
    if not JSONL.exists():
        JSONL.touch()
    if not target.exists():
        write_json(target, doc)
    with JSONL.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(doc, ensure_ascii=False) + "\n")


def old_relations() -> tuple[set[str], set[str], set[str]]:
    excluded_urls: set[str] = set()
    excluded_originals: set[str] = set()
    excluded_events: set[str] = set()
    for path in sorted((PILOT / "documents").glob("gv-*.json")):
        row = json.loads(path.read_text())
        excluded_urls.add(row["url"].rstrip("/"))
        excluded_originals.add(row.get("original_url", row["url"]).rstrip("/"))
        excluded_urls.update(u.rstrip("/") for u in row.get("translations", []))
        for relation in row.get("relation_keys", []):
            if relation.startswith("event:"):
                excluded_events.add(relation)
    return excluded_urls, excluded_originals, excluded_events


def feed_rows(data: bytes) -> list[dict]:
    root = ET.fromstring(data)
    rows = []
    for index, item in enumerate(root.findall("./channel/item")):
        encoded = item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded") or ""
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        if not link:
            continue
        categories = [tag.text or "" for tag in item.findall("category")]
        author = item.findtext("{http://purl.org/dc/elements/1.1/}creator") or item.findtext("author") or ""
        published = item.findtext("pubDate") or ""
        content = BeautifulSoup(encoded, "html.parser").get_text(" ", strip=True)
        rows.append({"index": index, "title": title, "url": link.strip(), "categories": categories,
                     "author": author, "published": published, "feed_text_chars": len(content), "encoded": encoded,
                     "_topic_search": " ".join((title, *categories, content))})
    return rows


def broad_topic(item: dict, language: str) -> bool:
    # Feed prose often mentions unrelated context. Scope is established from
    # the headline and publisher categories before downloading a page.
    metadata = norm(" ".join([item["title"], *item["categories"]]))
    terms = TOPIC_TERMS[language]
    return any(term.casefold() in metadata for term in terms)


def personal_topic(item: dict) -> bool:
    metadata = norm(" ".join([item["title"], *item["categories"]]))
    return (any(term.casefold() in metadata for term in PERSONAL_TERMS)
            or any(term.casefold() in metadata for term in OUT_OF_SCOPE_TERMS))


def item_id(language: str, url: str) -> str:
    slug = Path(urlparse(url).path.rstrip("/")).name
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug).strip("-").lower()[:48]
    suffix = hashlib.sha256(url.encode()).hexdigest()[:8]
    return f"gv-{language}-{slug or 'story'}-{suffix}"


def parse_page(page: bytes, language: str) -> tuple[str, dict, list[str], list[str], str, list[str], list[str]]:
    soup = BeautifulSoup(page, "html.parser")
    credit_nodes = soup.select(".avatar-credits-container .contributor")
    credit_lines = []
    for node in credit_nodes:
        label = node.select_one(".credit-label")
        name = node.select_one(".contributor-name")
        if label and name:
            contributor = name.get_text(" ", strip=True)
            label_text = label.get_text(" ", strip=True)
            if contributor:
                credit_lines.append({"label": label_text, "name": contributor})
    if not credit_lines:
        raise ValueError("missing contributor attribution")

    # Extract only the article body. Sidebars, credits, related stories, and site
    # navigation are outside .entry and therefore cannot leak into the document.
    entry = soup.select_one(".entry")
    if entry is None:
        raise ValueError("missing article body")
    text, extraction = clean_article(str(entry))
    if len(text) < MIN_TEXT_CHARS or len(text) > MAX_TEXT_CHARS:
        raise ValueError("article text outside the admitted size range")

    # Article-level credit labels are enough to establish a named writer or
    # translator. The complete labels and names remain in the admitted record.
    translation_markers = ("translat", "traduit", "traduc", "ترجم", "ترجمة", "翻訳", "翻译", "अनुवाद", "অনুবাদ", "traduz", "перев", "çeviri")
    primary = next((c for c in credit_lines if not any(m in norm(c["label"]) for m in translation_markers)), credit_lines[0])

    translations = []
    for anchor in soup.select(".post-translations a[href]"):
        href = anchor.get("href", "").strip()
        label = norm(anchor.get_text(" ", strip=True))
        if href.startswith("https://") and urlparse(href).netloc.endswith("globalvoices.org"):
            translations.append({"language_label": anchor.get_text(" ", strip=True), "url": href})

    current_url = soup.select_one('link[rel="canonical"]')
    canonical_url = (current_url.get("href") if current_url else None) or ""
    translations.append({"language_label": language, "url": canonical_url}) if canonical_url else None
    translations = list({entry["url"].rstrip("/"): entry for entry in translations}.values())

    # Credit labels name the written language in parentheses on Global Voices.
    label = norm(primary["label"])
    source_code = None
    for code, names in LANGUAGE_NAMES.items():
        if any(name.casefold() in label for name in names):
            source_code = code
            break
    if source_code:
        matching = [entry["url"] for entry in translations
                    if any(name.casefold() == norm(entry["language_label"]) for name in LANGUAGE_NAMES.get(source_code, ()))
                    or (source_code == "en" and norm(entry["language_label"]) == "english")]
        original_url = matching[0] if matching else canonical_url
    else:
        original_url = canonical_url
    if not original_url:
        raise ValueError("missing canonical source URL")

    license_nodes = []
    for a in soup.select(".license a[href], a[href*='creativecommons.org/licenses/by/3.0']"):
        href = a.get("href", "").strip()
        if "creativecommons.org/licenses/by/3.0/" in href:
            license_nodes.append(href)
    if not license_nodes:
        raise ValueError("item has no live CC BY 3.0 footer")

    event = "gv-story:" + hashlib.sha256(original_url.rstrip("/").encode()).hexdigest()[:20]
    relation_keys = ["original:" + original_url.rstrip("/"), "translation-event:" + event]
    return text, extraction, [c["label"] + " — " + c["name"] for c in credit_lines], [u["url"] for u in translations], original_url, sorted(set(license_nodes)), relation_keys


def item_is_reprint(page_soup: BeautifulSoup, item: dict) -> bool:
    text = norm(" ".join([item["title"], item["author"], page_soup.select_one(".entry").get_text(" ", strip=True) if page_soup.select_one(".entry") else ""]))
    return any(term in text for term in REPRINT_TERMS)


def check_host(client: BoundedClient, language: str, cached: bool) -> tuple[str, bytes, bytes]:
    host_code = HOST_CODES.get(language, language)
    host = f"https://{host_code}.globalvoices.org"
    robots_name = f"robots/{host_code}-robots.txt"
    feed_name = f"feeds/{host_code}-feed.xml"
    if cached:
        old_robots = AUDIT / f"gv_{host_code}_robots.txt"
        old_feed = AUDIT / f"gv_{host_code}_feed.txt"
        if not old_robots.exists() or not old_feed.exists():
            raise FileNotFoundError("Eligible cached host feed or robots file is missing")
        (RAW / robots_name).parent.mkdir(parents=True, exist_ok=True)
        (RAW / feed_name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old_robots, RAW / robots_name)
        shutil.copyfile(old_feed, RAW / feed_name)
        robots_bytes = (RAW / robots_name).read_bytes()
        feed_bytes = (RAW / feed_name).read_bytes()
        client.records.extend([
            {"url": host + "/robots.txt", "bytes": len(robots_bytes), "sha256": sha(robots_bytes), "cached": True},
            {"url": host + "/feed/", "bytes": len(feed_bytes), "sha256": sha(feed_bytes), "cached": True},
        ])
    else:
        robots_bytes = client.get(host + "/robots.txt", robots_name, delay=10)
        feed_url = host + "/feed/"
        robots = RobotFileParser()
        robots.parse(robots_bytes.decode("utf-8", "replace").splitlines())
        if not robots.can_fetch(client.user_agent, feed_url):
            raise PermissionError("robots disallows the official RSS feed")
        feed_bytes = client.get(feed_url, feed_name, delay=10)
    robots = RobotFileParser()
    robots.parse(robots_bytes.decode("utf-8", "replace").splitlines())
    feed_url = host + "/feed/"
    if not robots.can_fetch(client.user_agent, feed_url):
        raise PermissionError("robots disallows the official RSS feed")
    ET.fromstring(feed_bytes)
    return host, robots_bytes, feed_bytes


def acquire(*, per_language: int = 2, max_documents: int = 36, candidates_per_host: int = 7) -> list[dict]:
    if max_documents > 36 or max_documents < 1 or per_language < 1:
        raise ValueError("requested batch exceeds its fixed acquisition bounds")
    RAW.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)

    policy_source = PILOT / "raw/gv-policy.html"
    if not policy_source.exists():
        raise FileNotFoundError("cached Global Voices attribution policy is unavailable")
    policy_data = policy_source.read_bytes()
    policy_soup = BeautifulSoup(policy_data, "html.parser")
    policy_text = policy_soup.get_text(" ", strip=True)
    if "Creative Commons" not in policy_text or "Attribution" not in policy_text:
        raise ValueError("cached official publisher policy not recognized")
    (RAW / "policy").mkdir(exist_ok=True)
    shutil.copyfile(policy_source, RAW / "policy/global-voices-attribution-policy.html")

    inventory = json.loads(INVENTORY.read_text())
    inventory_hosts = {urlparse(x).netloc for x in inventory["Global Voices"]}
    saved_count = len(inventory_hosts)
    if saved_count < 53:
        raise ValueError("saved publisher inventory is incomplete")
    for language in TARGET_LANGUAGES:
        host_code = HOST_CODES.get(language, language)
        if f"{host_code}.globalvoices.org" not in inventory_hosts and language not in {"es", "ar", "fr"}:
            raise ValueError("target language host is absent from the saved inventory")

    # Keep cached bytes and each run's network bytes inside this expansion area.
    previous_network_bytes = 0
    previous_rejections: dict[str, int] = {}
    if STATUS.exists():
        previous = json.loads(STATUS.read_text())
        previous_network_bytes = previous.get("network_bytes_total", previous.get("network_bytes_this_run", 0))
        previous_rejections = previous.get("rejected_counts_by_reason", {})
    if previous_network_bytes >= MAX_NETWORK_BYTES:
        raise ValueError("cumulative reporting network byte budget is exhausted")
    client = BoundedClient(RAW, maximum=MAX_NETWORK_BYTES - previous_network_bytes)
    old_urls, old_originals, old_events = old_relations()
    docs: list[dict] = []
    checked_hosts: list[str] = []
    rejected: dict[str, int] = dict(previous_rejections)
    admitted_originals: set[str] = set()
    page_checks: dict[str, int] = {}
    existing = []
    if JSONL.exists():
        existing = [json.loads(line) for line in JSONL.read_text().splitlines() if line.strip()]
        docs.extend(existing)
        admitted_originals.update(d["original_url"].rstrip("/") for d in docs)
    else:
        JSONL.touch()

    target_count = min(max_documents, per_language * len(TARGET_LANGUAGES))
    record_status(phase="preflight", docs=docs, checked_hosts=checked_hosts, rejected=rejected, client=client, target_count=target_count, network_bytes_prior_runs=previous_network_bytes)
    for language in TARGET_LANGUAGES:
        if len(docs) >= max_documents:
            break
        have = sum(d["language"] == language for d in docs)
        if have >= per_language:
            continue
        cached = language in {"es", "ar", "fr"}
        host_code = HOST_CODES.get(language, language)
        host = f"https://{host_code}.globalvoices.org"
        try:
            host, robots_bytes, feed_bytes = check_host(client, language, cached)
        except Exception as exc:
            reason = "host_preflight_failed:" + type(exc).__name__
            rejected[reason] = rejected.get(reason, 0) + 1
            if host_code not in checked_hosts:
                checked_hosts.append(host_code)
            record_status(phase="host_preflight", docs=docs, checked_hosts=checked_hosts, rejected=rejected, client=client, target_count=target_count, network_bytes_prior_runs=previous_network_bytes)
            continue
        if host_code not in checked_hosts:
            checked_hosts.append(host_code)
        robots = RobotFileParser()
        robots.parse(robots_bytes.decode("utf-8", "replace").splitlines())
        rows = [item for item in feed_rows(feed_bytes) if broad_topic(item, language) and not personal_topic(item)]
        checked = 0
        for item in rows:
            if have >= per_language or len(docs) >= max_documents or checked >= candidates_per_host:
                break
            norm_url = item["url"].rstrip("/")
            if norm_url in old_urls:
                rejected["pilot_item_or_known_translation"] = rejected.get("pilot_item_or_known_translation", 0) + 1
                continue
            if item["feed_text_chars"] == 0:
                rejected["feed_missing_full_text"] = rejected.get("feed_missing_full_text", 0) + 1
                continue
            if not robots.can_fetch(client.user_agent, item["url"]):
                rejected["robots_disallows_item"] = rejected.get("robots_disallows_item", 0) + 1
                continue
            checked += 1
            page_name = f"items/{host_code}-{item['index']:02d}-{hashlib.sha256(item['url'].encode()).hexdigest()[:8]}.html"
            try:
                (RAW / page_name).parent.mkdir(parents=True, exist_ok=True)
                page = client.get(item["url"], page_name, delay=10)
                soup = BeautifulSoup(page, "html.parser")
                if item_is_reprint(soup, item):
                    raise ValueError("third-party reprint indicator")
                text, extraction, attribution, translation_urls, original_url, licenses, relation_keys = parse_page(page, language)
                original_norm = original_url.rstrip("/")
                if original_norm in old_originals or any(u.rstrip("/") in old_urls for u in translation_urls):
                    raise ValueError("item belongs to an excluded pilot event or translation group")
                if original_norm in admitted_originals:
                    raise ValueError("duplicate translation event already admitted")
                if not item.get("published"):
                    raise ValueError("missing publication date")
                published = parsedate_to_datetime(item["published"]).isoformat()
                page_hash = sha(page)
                canonical = soup.select_one('link[rel="canonical"]')
                canonical_url = canonical.get("href") if canonical else item["url"]
                doc = {
                    "id": item_id(language, canonical_url),
                    "source_family": "reporting",
                    "publisher": "Global Voices",
                    "language": language,
                    "publisher_language_host": host_code,
                    "title": item["title"],
                    "url": item["url"],
                    "revision": f"feed-sha256:{sha(feed_bytes)};item-sha256:{page_hash}",
                    "retrieved_at": utc_now(),
                    "published_at": published,
                    "content_sha256": sha(text.encode("utf-8")),
                    "text": text,
                    "attribution": attribution,
                    "license": {"id": "CC-BY-3.0", "url": licenses[0], "share_alike": False},
                    "rights_evidence": {
                        "checked_at": utc_now(),
                        "policy_url": POLICY_URL,
                        "policy_sha256": sha(policy_data),
                        "item_license_url": licenses[0],
                        "item_sha256": page_hash,
                        "route": "official localized Global Voices full-text RSS + live localized item footer",
                        "robots_sha256": sha(robots_bytes),
                        "feed_sha256": sha(feed_bytes),
                        "host_inventory_count": saved_count,
                    },
                    "relation_keys": relation_keys,
                    "original_translation": "human_translation" if original_url.rstrip("/") != canonical_url.rstrip("/") else "source_native_or_canonical",
                    "original_url": original_url,
                    "translations": sorted(set(translation_urls)),
                    "extraction": extraction,
                    "changes": "HTML/media/captions/third-party blockquotes removed; article-body prose and headings retained; paragraph separators normalized",
                    "source_metadata": {
                        "feed_index": item["index"],
                        "feed_categories": item["categories"],
                        "feed_author": item["author"],
                        "feed_published_at": item["published"],
                        "feed_item_text_chars": item["feed_text_chars"],
                        "page_canonical_url": canonical_url,
                    },
                }
            except PermissionError:
                rejected["rights_or_access_denied"] = rejected.get("rights_or_access_denied", 0) + 1
                continue
            except Exception as exc:
                # Record only a reason class in agent-visible metadata.
                key = "item_rejected:" + str(exc) if str(exc) in {
                    "third-party reprint indicator", "item belongs to an excluded pilot event or translation group",
                    "duplicate translation event already admitted", "missing publication date",
                    "missing contributor attribution", "missing article body", "article text outside the admitted size range",
                    "missing canonical source URL", "item has no live CC BY 3.0 footer",
                } else "item_rejected:" + type(exc).__name__
                rejected[key] = rejected.get(key, 0) + 1
                continue
            if doc["id"] in {d["id"] for d in docs}:
                rejected["duplicate_document_id"] = rejected.get("duplicate_document_id", 0) + 1
                continue
            docs.append(doc)
            have += 1
            admitted_originals.add(doc["original_url"].rstrip("/"))
            append_document(doc)
            record_status(phase="item_admitted", docs=docs, checked_hosts=checked_hosts, rejected=rejected, client=client, target_count=target_count, network_bytes_prior_runs=previous_network_bytes)
            print(json.dumps({"event": "admitted", "id": doc["id"], "language": language, "language_total": have,
                              "documents": len(docs), "languages": sorted({d['language'] for d in docs}),
                              "network_bytes": client.bytes}, ensure_ascii=False), flush=True)
        page_checks[host_code] = checked
        record_status(phase="language_complete", docs=docs, checked_hosts=checked_hosts, rejected=rejected, client=client, target_count=target_count, network_bytes_prior_runs=previous_network_bytes)
        print(json.dumps({"event": "language_complete", "language": language, "admitted": have,
                          "candidate_pages_checked": checked, "documents": len(docs),
                          "network_bytes": client.bytes}, ensure_ascii=False), flush=True)

    inventory_count, preflight = persisted_preflight_evidence(client.user_agent)
    budget = {
        "updated_at": utc_now(),
        "publisher": "Global Voices",
        "source_inventory_hosts": inventory_count,
        "checked_hosts": [row["publisher_host"] for row in preflight],
        "checked_host_count": len(preflight),
        "preflight_evidence": preflight,
        "target_languages": list(TARGET_LANGUAGES),
        "admitted_documents": len(docs),
        "target_documents": target_count,
        "language_counts": {lang: sum(d["language"] == lang for d in docs) for lang in TARGET_LANGUAGES},
        "max_documents": max_documents,
        "per_language_target": per_language,
        "candidate_page_checks_by_host": page_checks,
        "network_bytes_prior_runs": previous_network_bytes,
        "network_bytes_this_run": client.bytes,
        "network_bytes_total": previous_network_bytes + client.bytes,
        "network_bytes_cap": MAX_NETWORK_BYTES,
        "network_bytes_aim": 20_000_000,
        "raw_bytes_current_batch": sum(p.stat().st_size for p in RAW.rglob("*") if p.is_file()),
        "requests": client.records,
        "rejected_counts_by_reason": rejected,
        "external_benchmark_access": False,
        "raw_directory": "data/document_expansion/2026-10-02/raw/reporting",
        "documents_jsonl": "data/document_expansion/2026-10-02/reporting-documents.jsonl",
    }
    write_json(MANIFEST, budget)
    record_status(phase="complete", docs=docs, checked_hosts=checked_hosts, rejected=rejected, client=client, target_count=target_count, network_bytes_prior_runs=previous_network_bytes)
    status = json.loads(STATUS.read_text())
    status["network_bytes_prior_runs"] = previous_network_bytes
    status["network_bytes_total"] = previous_network_bytes + client.bytes
    status["state"] = "acquired" if len(docs) >= target_count else "coverage_limited"
    status["updated_at"] = utc_now()
    write_json(STATUS, status)
    return docs


if __name__ == "__main__":
    try:
        docs = acquire()
        print(json.dumps({"event": "finished", "documents": len(docs),
                          "languages": sorted({d["language"] for d in docs}),
                          "network_bytes": json.loads(MANIFEST.read_text())["network_bytes_this_run"]}, ensure_ascii=False), flush=True)
    except Exception as exc:
        # Do not expose article-bearing parser errors to the caller.
        print(json.dumps({"event": "failed", "error_type": type(exc).__name__}, ensure_ascii=False), file=sys.stderr, flush=True)
        raise
