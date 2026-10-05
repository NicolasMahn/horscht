"""Bounded, serial Wikipedia acquisition for the multilingual document expansion.

This module writes immutable source records and metadata-only progress. It does
not create questions, assign splits, or export training data.
"""
from __future__ import annotations

import bz2
import hashlib
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlencode, urlsplit
from urllib.robotparser import RobotFileParser
import xml.etree.ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-10-02"
DATA = ROOT / "data/document_expansion" / DATE
RAW = DATA / "raw/wiki"
DOCS = DATA / "documents"
STATUS = ROOT / "runs/document-expansion-2026-10-02/acquisition-wikipedia-status.json"
PILOT = ROOT / "data/document_pilot/2026-10-02"
POLICY_CACHE = PILOT / "raw/wikimedia-api-policy.html"
POLICY_URL = "https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_API_Usage_Guidelines"
DUMP_BASE = "https://dumps.wikimedia.org"
DUMP_LEGAL_URL = "https://dumps.wikimedia.org/legal.html"
DUMP_ROBOT_POLICY_URL = "https://wikitech.wikimedia.org/wiki/Robot_policy"
USER_AGENT_PRODUCT = "HorschtNicolasSampler/0.1"
LANGUAGES = (
    "ar", "es", "fr", "de", "ja", "pt", "it", "ru", "uk", "pl", "nl", "sv",
    "fi", "cs", "tr", "id", "vi", "zh", "ko", "hi", "bn", "ta", "sw", "he",
)
TARGET_PER_LANGUAGE = 5
MAX_BATCHES_PER_LANGUAGE = 12
MAX_NETWORK_BYTES = 900_000_000
INITIAL_RANGE_BYTES = 8_388_608
MAX_RANGE_BYTES_PER_LANGUAGE = 16_777_216
MIN_CHARS = 3_500
MAX_CHARS = 70_000
GLOBAL_DELAY_SECONDS = 1.0
SLOW_REQUEST_DELAY_SECONDS = 5.0
MAX_RETRY_AFTER_SECONDS = 60
WIKIDATA_BATCH_SIZE = 10

LIST_TITLE_MARKERS = {
    "ar": ("قائمة ",), "es": ("lista de ",), "fr": ("liste de ",),
    "de": ("liste der ", "liste von "), "ja": ("一覧", "リスト"),
    "pt": ("lista de ",), "it": ("lista di ", "elenco di "),
    "ru": ("список ",), "uk": ("список ",), "pl": ("lista ",),
    "nl": ("lijst van ",), "sv": ("lista över "), "fi": ("luettelo ", "lista "),
    "cs": ("seznam ",), "tr": ("listesi",), "id": ("daftar ",),
    "vi": ("danh sách ",), "zh": ("列表", "名單", "名单"), "ko": ("목록",),
    "hi": ("सूची ",), "bn": ("তালিকা ",), "ta": ("பட்டியல் ",),
    "sw": ("orodha ya ",), "he": ("רשימת ",),
}
POSITIVE_TOPIC_TERMS = {
    "ar": ("مدينة", "بلدية", "دولة", "نهر", "جبل", "جزيرة", "بحيرة", "نوع", "نظام بيئي", "عنصر كيميائي", "جامعة", "مؤسسة", "علم", "تقنية", "كوكب", "لغة", "مبنى"),
    "es": ("ciudad", "municipio", "país", "río", "montaña", "isla", "lago", "especie", "ecosistema", "elemento químico", "universidad", "institución", "ciencia", "tecnología", "planeta", "idioma", "edificio"),
    "fr": ("ville", "commune", "pays", "fleuve", "rivière", "montagne", "île", "lac", "espèce", "écosystème", "élément chimique", "université", "institution", "science", "technologie", "planète", "langue", "bâtiment"),
    "de": ("stadt", "gemeinde", "staat", "fluss", "gebirge", "insel", "see", "insekt", "tierart", "pflanzenart", "spezies", "ökosystem", "chemisches element", "universität", "institution", "wissenschaft", "technik", "planet", "sprache", "gebäude"),
    "ja": ("都市", "市町村", "河川", "山", "島", "国", "湖", "種", "生態系", "化学元素", "大学", "施設", "科学", "技術", "惑星", "言語", "建築物"),
    "pt": ("cidade", "município", "país", "rio", "montanha", "ilha", "lago", "espécie", "ecossistema", "elemento químico", "universidade", "instituição", "ciência", "tecnologia", "planeta", "língua", "edifício"),
    "it": ("città", "comune", "paese", "fiume", "montagna", "isola", "lago", "specie", "ecosistema", "elemento chimico", "università", "istituzione", "scienza", "tecnologia", "pianeta", "lingua", "edificio"),
    "ru": ("город", "страна", "река", "гора", "остров", "озеро", "вид", "экосистема", "химический элемент", "университет", "учреждение", "наука", "технология", "планета", "язык", "здание"),
    "uk": ("місто", "країна", "річка", "гора", "острів", "озеро", "вид", "екосистема", "хімічний елемент", "університет", "установа", "наука", "технологія", "планета", "мова", "будівля"),
    "pl": ("miasto", "państwo", "rzeka", "góra", "wyspa", "jezioro", "gatunek", "ekosystem", "pierwiastek chemiczny", "uniwersytet", "instytucja", "nauka", "technologia", "planeta", "język", "budynek"),
    "nl": ("stad", "gemeente", "land", "rivier", "berg", "eiland", "meer", "soort", "ecosysteem", "chemisch element", "universiteit", "instelling", "wetenschap", "technologie", "planeet", "taal", "gebouw"),
    "sv": ("stad", "kommun", "land", "flod", "berg", "ö", "sjö", "art", "ekosystem", "grundämne", "universitet", "institution", "vetenskap", "teknik", "planet", "språk", "byggnad"),
    "fi": ("kaupunki", "kunta", "valtio", "joki", "vuori", "saari", "järvi", "laji", "ekosysteemi", "alkuaine", "yliopisto", "instituutio", "tiede", "teknologia", "planeetta", "kieli", "rakennus"),
    "cs": ("město", "obec", "stát", "řeka", "hora", "ostrov", "jezero", "druh", "ekosystém", "chemický prvek", "univerzita", "instituce", "věda", "technologie", "planeta", "jazyk", "budova"),
    "tr": ("şehir", "kent", "ülke", "nehir", "dağ", "ada", "göl", "tür", "ekosistem", "kimyasal element", "üniversite", "kurum", "bilim", "teknoloji", "gezegen", "dil", "yapı"),
    "id": ("kota", "kabupaten", "negara", "sungai", "gunung", "pulau", "danau", "spesies", "ekosistem", "unsur kimia", "universitas", "institusi", "sains", "teknologi", "planet", "bahasa", "bangunan"),
    "vi": ("thành phố", "quốc gia", "sông", "núi", "đảo", "hồ", "loài", "hệ sinh thái", "nguyên tố hóa học", "đại học", "tổ chức", "khoa học", "công nghệ", "hành tinh", "ngôn ngữ", "công trình"),
    "zh": ("城市", "国家", "河流", "山", "岛", "湖", "物种", "生态系统", "化学元素", "大学", "机构", "科学", "技术", "行星", "语言", "建筑"),
    "ko": ("도시", "국가", "강", "산", "섬", "호수", "종", "생태계", "화학 원소", "대학", "기관", "과학", "기술", "행성", "언어", "건축물"),
    "hi": ("नगर", "शहर", "देश", "नदी", "पर्वत", "द्वीप", "झील", "प्रजाति", "पारिस्थितिकी तंत्र", "रासायनिक तत्व", "विश्वविद्यालय", "संस्था", "विज्ञान", "प्रौद्योगिकी", "ग्रह", "भाषा", "भवन"),
    "bn": ("শহর", "দেশ", "নদী", "পর্বত", "দ্বীপ", "হ্রদ", "প্রজাতি", "বাস্তুতন্ত্র", "রাসায়নিক মৌল", "বিশ্ববিদ্যালয়", "প্রতিষ্ঠান", "বিজ্ঞান", "প্রযুক্তি", "গ্রহ", "ভাষা", "ভবন"),
    "ta": ("நகரம்", "நாடு", "ஆறு", "மலை", "தீவு", "ஏரி", "உயிரினம்", "சூழலமைப்பு", "வேதியியல் தனிமம்", "பல்கலைக்கழகம்", "நிறுவனம்", "அறிவியல்", "தொழில்நுட்பம்", "கோள்", "மொழி", "கட்டிடம்"),
    "sw": ("mji", "nchi", "mto", "mlima", "kisiwa", "ziwa", "spishi", "mfumo ikolojia", "elementi ya kemikali", "chuo kikuu", "taasisi", "sayansi", "teknolojia", "sayari", "lugha", "jengo"),
    "he": ("עיר", "מדינה", "נהר", "הר", "אי", "אגם", "מין", "מערכת אקולוגית", "יסוד כימי", "אוניברסיטה", "מוסד", "מדע", "טכנולוגיה", "כוכב לכת", "שפה", "מבנה"),
}
PERSON_SIGNAL_PROPERTIES = {"P569", "P570", "P19", "P20", "P22", "P25", "P26"}
WIKIMEDIA_LIST_ITEM_QID = "Q13406463"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def robots_allows_url(robots_text: str, user_agent: str, url: str) -> bool:
    """Evaluate rules against path plus query, as Wikimedia's robots allows require."""
    parser = RobotFileParser()
    parser.parse(robots_text.splitlines())
    ua = user_agent.split("/", 1)[0].casefold()
    entries = [entry for entry in parser.entries if any(token.casefold() in ua for token in entry.useragents if token != "*")]
    if entries:
        longest = max(
            len(token) for entry in entries for token in entry.useragents
            if token != "*" and token.casefold() in ua
        )
        entries = [entry for entry in entries if any(
            len(token) == longest and token.casefold() in ua
            for token in entry.useragents if token != "*"
        )]
    elif parser.default_entry:
        entries = [parser.default_entry]
    else:
        return True

    parts = urlsplit(url)
    target = parts.path + (f"?{parts.query}" if parts.query else "")
    matched: list[tuple[int, bool]] = []
    for entry in entries:
        for rule in entry.rulelines:
            if rule.path and rule.applies_to(target):
                matched.append((len(rule.path.rstrip("$")), rule.allowance))
    if not matched:
        return True
    specificity = max(length for length, _ in matched)
    return any(allow for length, allow in matched if length == specificity)


def refresh_route_checks(status: dict[str, Any], user_agent: str) -> None:
    for language, edition in status.get("editions", {}).items():
        robots_file = RAW / f"preflight-{language}-robots.txt"
        if not robots_file.exists():
            continue
        text = robots_file.read_text(errors="replace")
        host = edition.get("host", f"{language}.wikipedia.org")
        routes = {
            "action_query_random": f"https://{host}/w/api.php?action=query&generator=random&grnnamespace=0&grnlimit=10",
            "rest_random_summary": f"https://{host}/api/rest_v1/page/random/summary",
            "rest_random_title": f"https://{host}/api/rest_v1/page/random/title",
            "action_mobileview_random": f"https://{host}/w/api.php?action=mobileview&page=Special%3ARandom&prop=sections%7Ctext&sections=all",
        }
        edition["alternate_routes"] = {
            name: {"url": url, "robots_allows": robots_allows_url(text, user_agent, url)}
            for name, url in routes.items()
        }


def load_prior_documents() -> list[dict[str, Any]]:
    docs_by_id: dict[str, dict[str, Any]] = {}
    path = DATA / "wiki-documents.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                doc = json.loads(line)
                docs_by_id[doc["id"]] = doc
    if DOCS.exists():
        for snapshot in DOCS.glob("wiki-*.json"):
            doc = json.loads(snapshot.read_text())
            docs_by_id.setdefault(doc["id"], doc)
    return list(docs_by_id.values())


class WikimediaClient:
    """One request at a time, globally spaced, with a hard decoded-byte cap."""

    def __init__(self, status: dict[str, Any], persist_status):
        contact = subprocess.check_output(
            ["git", "config", "--global", "--get", "user.email"], text=True
        ).strip()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", contact):
            raise ValueError("A configured real contact is required for Wikimedia requests")
        self._contact = contact
        self.user_agent = f"{USER_AGENT_PRODUCT} (bounded research; mailto:{contact})"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self.status = status
        self.persist_status = persist_status
        self.last_completed = 0.0
        self.last_duration = 0.0
        self.cached_network_bytes = int(status.get("network_bytes", 0))

    def get(self, url: str, raw_name: str, *, endpoint_kind: str,
            request_headers: dict[str, str] | None = None,
            expected_status: int | None = None,
            expected_range: tuple[int, int] | None = None,
            allow_404: bool = False) -> tuple[bytes | None, dict[str, Any]]:
        RAW.mkdir(parents=True, exist_ok=True)
        target = RAW / raw_name
        if target.exists():
            body = target.read_bytes()
            prior = next((row for row in reversed(self.status.get("requests", []))
                          if row.get("raw_file") == str(target.relative_to(ROOT))), {})
            record = {
                "url": url,
                "raw_file": str(target.relative_to(ROOT)),
                "bytes": len(body),
                "sha256": sha(body),
                "cached": True,
                "endpoint_kind": endpoint_kind,
                "http_status": prior.get("http_status", expected_status or 200),
                "content_range": prior.get("content_range"),
                "etag": prior.get("etag"),
            }
            if expected_range:
                record["requested_range"] = {"start": expected_range[0], "end": expected_range[1]}
            self.status["requests"].append(record)
            self.persist_status()
            return body, record

        delay_after_previous = SLOW_REQUEST_DELAY_SECONDS if self.last_duration > 1 else GLOBAL_DELAY_SECONDS
        wait = delay_after_previous - (time.monotonic() - self.last_completed)
        if wait > 0:
            time.sleep(wait)
        attempts = 0
        while True:
            attempts += 1
            request_started = time.monotonic()
            try:
                response = self.session.get(url, headers=request_headers, timeout=(15, 45), stream=True)
                self.last_completed = time.monotonic()
                self.last_duration = self.last_completed - request_started
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
                if response.status_code in (429, 503):
                    response.close()
                    if retry_after is not None and retry_after <= MAX_RETRY_AFTER_SECONDS and attempts == 1:
                        time.sleep(retry_after)
                        spacing = GLOBAL_DELAY_SECONDS - (time.monotonic() - self.last_completed)
                        if spacing > 0:
                            time.sleep(spacing)
                        continue
                    record = {
                        "url": url,
                        "endpoint_kind": endpoint_kind,
                        "http_status": response.status_code,
                        "error": "rate_limited; stopped after bounded Retry-After handling",
                        "retry_after_seconds": retry_after,
                        "attempts": attempts,
                        "cached": False,
                    }
                    self.status["requests"].append(record)
                    self.persist_status()
                    return None, record
                if response.status_code in (401, 403, 410) or (response.status_code == 404 and not allow_404):
                    record = {
                        "url": url,
                        "endpoint_kind": endpoint_kind,
                        "http_status": response.status_code,
                        "error": "access_or_route_denied; stopped without retry",
                        "cached": False,
                    }
                    response.close()
                    self.status["requests"].append(record)
                    self.persist_status()
                    return None, record
                if expected_status is not None and response.status_code != expected_status:
                    record = {
                        "url": url,
                        "endpoint_kind": endpoint_kind,
                        "http_status": response.status_code,
                        "error": "expected_http_status_not_returned",
                        "expected_http_status": expected_status,
                        "cached": False,
                    }
                    response.close()
                    self.status["requests"].append(record)
                    self.persist_status()
                    return None, record
                if response.status_code != 404:
                    response.raise_for_status()
                content_range = response.headers.get("Content-Range")
                if expected_range is not None:
                    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range or "", flags=re.IGNORECASE)
                    if not match or int(match.group(1)) != expected_range[0] or int(match.group(2)) > expected_range[1]:
                        record = {
                            "url": url,
                            "endpoint_kind": endpoint_kind,
                            "http_status": response.status_code,
                            "content_range": content_range,
                            "error": "invalid_content_range",
                            "cached": False,
                        }
                        response.close()
                        self.status["requests"].append(record)
                        self.persist_status()
                        return None, record
                declared_size = response.headers.get("Content-Length")
                if declared_size and int(self.status.get("network_bytes", 0)) + int(declared_size) > MAX_NETWORK_BYTES:
                    response.close()
                    record = {
                        "url": url,
                        "endpoint_kind": endpoint_kind,
                        "http_status": response.status_code,
                        "error": "hard_network_byte_budget_reached_from_content_length",
                        "cached": False,
                    }
                    self.status["budget_exhausted"] = True
                    self.status["requests"].append(record)
                    self.persist_status()
                    return None, record
                chunks = []
                total = 0
                for chunk in response.iter_content(65536):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if int(self.status.get("network_bytes", 0)) + total > MAX_NETWORK_BYTES:
                        response.close()
                        record = {
                            "url": url,
                            "endpoint_kind": endpoint_kind,
                            "http_status": response.status_code,
                            "error": "hard_network_byte_budget_reached",
                            "cached": False,
                        }
                        self.status["budget_exhausted"] = True
                        self.status["requests"].append(record)
                        self.persist_status()
                        return None, record
                    chunks.append(chunk)
                response.close()
                self.last_completed = time.monotonic()
                self.last_duration = self.last_completed - request_started
                body = b"".join(chunks)
                target.write_bytes(body)
                self.status["network_bytes"] = int(self.status.get("network_bytes", 0)) + len(body)
                record = {
                    "url": url,
                    "raw_file": str(target.relative_to(ROOT)),
                    "bytes": len(body),
                    "sha256": sha(body),
                    "cached": False,
                    "endpoint_kind": endpoint_kind,
                    "http_status": response.status_code,
                    "attempts": attempts,
                    "content_range": content_range,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }
                if expected_range:
                    record["requested_range"] = {"start": expected_range[0], "end": expected_range[1]}
                self.status["requests"].append(record)
                self.persist_status()
                return body, record
            except requests.RequestException as exc:
                self.last_completed = time.monotonic()
                self.last_duration = self.last_completed - request_started
                record = {
                    "url": url,
                    "endpoint_kind": endpoint_kind,
                    "error": type(exc).__name__,
                    "cached": False,
                    "attempts": attempts,
                }
                self.status["requests"].append(record)
                self.persist_status()
                return None, record


def api_url(host: str, params: dict[str, Any]) -> str:
    return f"https://{host}/w/api.php?{urlencode(params, doseq=True)}"


def classify_text(language: str, title: str, text: str, pageprops: dict[str, Any]) -> tuple[bool, str | None, int]:
    normalized_title = " ".join(title.casefold().split())
    if "disambiguation" in pageprops:
        return False, "disambiguation", 0
    if any(marker.casefold() in normalized_title for marker in LIST_TITLE_MARKERS.get(language, ())):
        return False, "list_title", 0
    if re.search(r"\([^)]*(?:disambiguation|disambig)\)$", normalized_title):
        return False, "disambiguation_title", 0
    chars = len(text)
    if chars < MIN_CHARS:
        return False, "below_minimum_length", 0
    if chars > MAX_CHARS:
        return False, "above_maximum_length", 0
    headings = len(re.findall(r"(?m)^\s*={2,}\s*[^=\n].*?\s*={2,}\s*$", text))
    if headings < 3:
        return False, "too_few_sections", headings
    return True, None, headings


def strip_balanced_templates(text: str) -> str:
    out: list[str] = []
    depth = 0
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth += 1
            i += 1
            continue
        if text[i] == "}" and depth:
            depth -= 1
            i += 1
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)


def strip_balanced_tables(text: str) -> str:
    out: list[str] = []
    depth = 0
    i = 0
    while i < len(text):
        if text.startswith("{|", i):
            depth += 1
            i += 2
            continue
        if text.startswith("|}", i) and depth:
            depth -= 1
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)


FILE_NAMESPACE_NAMES = {
    "ar": ("ملف", "صورة"), "bn": ("চিত্র", "মিডিয়া", "মিডিয়া"),
    "cs": ("Soubor",), "de": ("Datei", "Bild"), "es": ("Archivo", "Imagen"),
    "fi": ("Tiedosto", "Kuva"), "fr": ("Fichier", "Image"), "he": ("קובץ", "תמונה"),
    "hi": ("चित्र", "मीडिया"), "id": ("Berkas", "Gambar"), "it": ("File", "Immagine"),
    "ja": ("ファイル", "画像"), "ko": ("파일", "그림"), "nl": ("Bestand", "Afbeelding"),
    "pl": ("Plik",), "pt": ("Ficheiro", "Arquivo", "Imagem"), "ru": ("Файл",),
    "sv": ("Fil", "Bild"), "sw": ("Faili", "Picha"), "ta": ("படிமம்", "ஊடகம்"),
    "tr": ("Dosya",), "uk": ("Файл",), "vi": ("Tập tin", "Hình"), "zh": ("文件", "檔案", "圖像", "图像"),
}


def wikitext_to_plain(language: str, wikitext: str) -> tuple[str, dict[str, int]]:
    """Keep prose and headings while dropping markup-only and separately hosted media."""
    removed = {"templates": 0, "tables": 0, "references": 0, "media_links": 0, "html_tags": 0}
    text = re.sub(r"<!--.*?-->", " ", wikitext, flags=re.DOTALL)
    text, removed_refs = re.subn(r"<ref\b[^>]*>.*?</ref\s*>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text, self_closing_refs = re.subn(r"<ref\b[^>]*/\s*>", " ", text, flags=re.IGNORECASE)
    removed["references"] = removed_refs + self_closing_refs
    removed["tables"] = text.count("{|")
    text = strip_balanced_tables(text)
    removed["templates"] = text.count("{{")
    text = strip_balanced_templates(text)
    file_names = "|".join(re.escape(name) for name in FILE_NAMESPACE_NAMES.get(language, ()))
    if file_names:
        text, removed["media_links"] = re.subn(rf"\[\[(?:{file_names}):[^\]]*\]\]", " ", text, flags=re.IGNORECASE)
    category_names = "Category|Категорія|Категория|Kategorie|Categoría|Catégorie|Categoria|Kategori|分類|분류|श्रेणी|শ্রেণী|வகை|קטגוריה|Jamii|Luokka|Kategoria|Klasifikasi"
    text = re.sub(rf"\[\[(?:{category_names}):[^\]]*\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[(?:[a-z][a-z0-9-]*):[^\]|]+(?:\|[^\]]*)?\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[(?:https?://)[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[(?:https?://)[^\]]+\]", " ", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"</?(?:ref|references|gallery|imagemap|timeline|score|syntaxhighlight|source|nowiki)\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text, removed["html_tags"] = re.subn(r"</?[A-Za-z][^>]*>", " ", text)
    text = re.sub(r"(?m)^\s*={2,}\s*(.*?)\s*={2,}\s*$", r"\1", text)
    text = re.sub(r"(?m)^\s*[*#:;]+\s*", "", text)
    text = re.sub(r"\[\d+\]", " ", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    kept = []
    for line in lines:
        if not line or re.fullmatch(r"[|=*_{}\-–—,.;:()\[\]]+", line):
            continue
        kept.append(line)
    plain = "\n\n".join(kept)
    return plain, removed


def native_script_ok(language: str, text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    counts = {
        "ar": sum("\u0600" <= c <= "\u06ff" for c in letters),
        "bn": sum("\u0980" <= c <= "\u09ff" for c in letters),
        "hi": sum("\u0900" <= c <= "\u097f" for c in letters),
        "he": sum("\u0590" <= c <= "\u05ff" for c in letters),
        "ja": sum("\u3040" <= c <= "\u30ff" for c in letters),
        "ko": sum("\uac00" <= c <= "\ud7af" for c in letters),
        "ru": sum("\u0400" <= c <= "\u052f" for c in letters),
        "uk": sum("\u0400" <= c <= "\u052f" for c in letters),
        "ta": sum("\u0b80" <= c <= "\u0bff" for c in letters),
        "zh": sum("\u3400" <= c <= "\u9fff" for c in letters),
    }
    if language == "uk":
        if counts[language] / len(letters) < 0.5:
            return False
        return any(c in text.casefold() for c in "іїєґ")
    if language in counts:
        ratio = counts[language] / len(letters)
        return ratio >= (0.18 if language == "ja" else 0.35)
    stopwords = {
        "cs": {"a", "i", "se", "že", "v", "na", "do", "z", "je", "byl", "pro", "jako", "který"},
        "de": {"der", "die", "das", "und", "von", "zu", "den", "mit", "ist", "für", "ein", "eine"},
        "es": {"el", "la", "de", "en", "y", "que", "los", "las", "un", "una", "por", "con"},
        "fi": {"ja", "on", "oli", "että", "sekä", "kun", "myös", "tai", "ei", "vuonna", "sen"},
        "fr": {"le", "la", "les", "de", "des", "du", "et", "en", "que", "un", "une", "dans"},
        "id": {"yang", "dan", "di", "ke", "dari", "untuk", "dalam", "pada", "ini", "itu", "oleh"},
        "it": {"il", "la", "di", "che", "e", "in", "un", "una", "del", "della", "con", "per"},
        "nl": {"de", "het", "van", "een", "en", "in", "op", "dat", "met", "voor", "is", "zijn"},
        "pl": {"i", "w", "z", "na", "że", "nie", "do", "jest", "się", "oraz", "przez", "jako"},
        "pt": {"de", "do", "da", "em", "e", "que", "o", "a", "os", "as", "uma", "para"},
        "sv": {"och", "att", "i", "en", "av", "för", "det", "som", "på", "med", "är", "till"},
        "tr": {"ve", "bir", "bu", "için", "ile", "olarak", "da", "de", "olan", "gibi", "daha"},
        "vi": {"và", "của", "là", "các", "trong", "cho", "một", "với", "được", "này", "đã"},
        "sw": {"na", "ya", "wa", "kwa", "katika", "hii", "kuwa", "ni", "za", "vya", "kutoka"},
    }
    if language not in stopwords:
        return len(letters) > 500
    words = re.findall(r"[\w’'-]+", text.casefold())
    hits = [word for word in words if word in stopwords[language]]
    return len(words) >= 180 and len(hits) >= 12 and len(set(hits)) >= 4


BIOGRAPHY_SIGNALS = re.compile(
    r"(?:personendaten|infobox\s+person|infobox\s+biograph|living people|births?\s+in\s+\d{3,4}|"
    r"nacidos?\s+en\s+\d{3,4}|fallecidos?\s+en\s+\d{3,4}|n[ée]e?\s+le\s+\d{1,2}|"
    r"geboren\s+am|gestorben\s+am|rodil[ao]s?\s+em|nasceu\s+em|nato\s+il|"
    r"родивш(?:ийся|аяся)|умерш(?:ий|ая)|народивш(?:ийся|ася)|померл(?:ий|а)|"
    r"출생|사망|生年|没年|出生日期|死亡日期|مواليد|وفيات|জন্ম\s*[:=]|মৃত্যু\s*[:=]|"
    r"பிறந்த\s*[:=]|இறந்த\s*[:=]|ولد\s+في|توفي\s+في|născut|născută|"
    r"born\s+\d{1,2}\s+[a-z]+\s+\d{3,4}|died\s+\d{1,2}\s+[a-z]+\s+\d{3,4})",
    re.IGNORECASE,
)
BIOGRAPHY_TEMPLATE_SIGNALS = re.compile(
    r"(?:^|[\s_/-])(?:bio|biografia|biographie|biography|personendaten)(?:$|[\s_/-])|"
    r"infobox\s+(?:person|royalty|philosopher|writer|poet|scientist|politician)|"
    r"birth\s+date|death\s+date|存命人物|人物の|manga-artist-stub|"
    r"भक्ति\s*काल\s*के\s*कवि|जन्म\s*तिथि|मृत्यु\s*तिथि|"
    r"infobox\s+filosofo|infobox\s+filósofo|infobox\s+escritor|infobox\s+musician|"
    r"nascita|morte|data\s+di\s+nascita|data\s+di\s+morte",
    re.IGNORECASE,
)
BIOGRAPHY_SIGNALS = re.compile(
    BIOGRAPHY_SIGNALS.pattern + r"|" +
    r"urodzon[yae]|zmarł[ay]?|narozen[ýá]?|narozena|zemřel[ay]?|"
    r"született|meghalt|geboren\s+(?:am|in)|gestorben\s+(?:am|in)|"
    r"föd[d]?|synty(?:nyt|nyt|mä|nyt)?|syntynyt|kuollut|"
    r"nato\s+(?:il|a|nel)|morto\s+(?:il|a|nel)|nasce?u?\s+(?:em|a)|"
    r"né\s+(?:le|en)|décédé[e]?\s+(?:le|en)|"
    r"sinh\s+(?:ngày|năm)|mất\s+(?:ngày|năm)|alizaliwa|alifariki|"
    r"जन्म\s*(?:तिथि|दिनांक|[:=])|मृत्यु\s*(?:तिथि|दिनांक|[:=])|"
    r"জন্ম\s*(?:তারিখ|সাল|[:=])|মৃত্যু\s*(?:তারিখ|সাল|[:=])|"
    r"生年月日|生誕|死去|출생\s*[:=]|사망\s*[:=]|"
    r"תאריך\s+לידה|תאריך\s+פטירה|נולד[ה]?\s+ב|נפטר[ה]?\s+ב",
    re.IGNORECASE,
)


def dump_page_record(element: ET.Element) -> dict[str, Any] | None:
    def find_text(parent: ET.Element, name: str) -> str | None:
        return next((child.text for child in parent.iter() if child.tag.rsplit("}", 1)[-1] == name), None)
    if element.tag.rsplit("}", 1)[-1] != "page":
        return None
    title = find_text(element, "title")
    namespace = find_text(element, "ns")
    pageid = find_text(element, "id")
    if not title or pageid is None or namespace is None:
        return None
    revision = next((child for child in element if child.tag.rsplit("}", 1)[-1] == "revision"), None)
    if revision is None:
        return None
    rev_text = find_text(revision, "text") or ""
    contributor = next((child for child in revision if child.tag.rsplit("}", 1)[-1] == "contributor"), None)
    user_name = find_text(contributor, "username") if contributor is not None else None
    user_id = find_text(contributor, "id") if contributor is not None else None
    ip = find_text(contributor, "ip") if contributor is not None else None
    redirect = any(child.tag.rsplit("}", 1)[-1] == "redirect" for child in element)
    return {
        "title": title,
        "namespace": int(namespace),
        "page_id": int(pageid),
        "redirect": redirect,
        "revision_id": find_text(revision, "id"),
        "timestamp": find_text(revision, "timestamp"),
        "revision_sha1": find_text(revision, "sha1"),
        "contributor": {"name": user_name, "user_id": user_id, "anonymous": bool(ip)} if contributor is not None else {},
        "wikitext": rev_text,
    }


def parse_multistream_prefix(data: bytes) -> tuple[list[dict[str, Any]], int, int]:
    parser = ET.XMLPullParser(events=("end",))
    pages: list[dict[str, Any]] = []
    complete_streams = 0
    expanded_bytes = 0
    remaining = data
    while remaining:
        decoder = bz2.BZ2Decompressor()
        try:
            expanded = decoder.decompress(remaining)
        except (EOFError, OSError):
            break
        if not decoder.eof:
            # The final range may cut through a member. Keep only whole members.
            break
        complete_streams += 1
        expanded_bytes += len(expanded)
        try:
            parser.feed(expanded)
            for _event, element in parser.read_events():
                if element.tag.rsplit("}", 1)[-1] == "page":
                    record = dump_page_record(element)
                    if record:
                        pages.append(record)
                    element.clear()
        except ET.ParseError:
            break
        remaining = decoder.unused_data
        if not remaining:
            break
    return pages, complete_streams, expanded_bytes


def pageprops_qids_from_gzip_prefix(data: bytes) -> dict[int, str]:
    import zlib
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        sql = decompressor.decompress(data, 128_000_000).decode("utf-8", errors="replace")
    except zlib.error:
        return {}
    return {int(page_id): qid for page_id, qid in re.findall(
        r"\((\d+),'wikibase_item','(Q\d+)'", sql
    )}


def latest_snapshot_dates(project_index: bytes) -> list[str]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(project_index, "html.parser")
    dates = {
        value.strip("/") for anchor in soup.select("a[href]")
        if re.fullmatch(r"\d{8}/?", value := anchor.get("href", ""))
    }
    return sorted(dates, reverse=True)


def manifest_file(manifest: dict[str, Any], suffix: str, *, job_hint: str) -> tuple[str, dict[str, Any], str] | None:
    jobs = manifest.get("jobs", {})
    ordered = sorted(
        jobs.items(),
        key=lambda item: (job_hint not in item[0], item[0]),
    )
    for job_name, job in ordered:
        if job.get("status") != "done":
            continue
        for filename, info in job.get("files", {}).items():
            if filename.endswith(suffix):
                return job_name, info, filename
    return None


def parse_main_page_license(html: bytes) -> tuple[str | None, str | None]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    footer = soup.select_one("footer#footer, #footer-info-copyright, footer")
    if footer is None:
        return None, None
    for anchor in footer.select("a[href]"):
        href = anchor.get("href", "")
        if "creativecommons.org/licenses/by-sa/4.0" in href:
            label = anchor.get_text(" ", strip=True)
            return href, label
        if re.search(r"(?:CC[_ -]?BY-SA|license|licen[cs]ia|licen[cs]e|licenza|lizenz|licence|giấy[_ -]?phép|رخصة|协议|許諾|ライセンス).{0,180}4[._ -]?0|4[._ -]?0.{0,180}(?:CC[_ -]?BY-SA|license|licen[cs]ia|licen[cs]e|licenza|lizenz|licence|giấy[_ -]?phép|رخصة|协议|許諾|ライセンス)", unquote(href), re.IGNORECASE):
            return href, anchor.get_text(" ", strip=True)
    return None, None


def read_dump_ranges(range_records: list[dict[str, Any]]) -> bytes:
    chunks = []
    for record in sorted(range_records, key=lambda row: row["start"]):
        path = ROOT / record["raw_file"]
        if path.exists():
            chunks.append(path.read_bytes())
    return b"".join(chunks)


def fetch_dump_range(client: WikimediaClient, url: str, language: str, snapshot: str,
                     start: int, end: int, kind: str) -> tuple[bytes | None, dict[str, Any]]:
    name = f"{kind}-{language}-{snapshot}-{start}-{end}.bin"
    body, record = client.get(
        url, name, endpoint_kind=f"{kind}_range",
        request_headers={"Range": f"bytes={start}-{end}"},
        expected_status=206,
        expected_range=(start, end),
    )
    if body is None:
        return None, record
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", record.get("content_range", ""), flags=re.IGNORECASE)
    if not match or len(body) != int(match.group(2)) - int(match.group(1)) + 1:
        record["error"] = "range_body_length_does_not_match_content_range"
        record["status"] = "invalid_range"
        client.status["requests"].append({
            "url": url, "endpoint_kind": f"{kind}_range_validation",
            "error": record["error"], "raw_file": record.get("raw_file"),
        })
        client.persist_status()
        return None, record
    record["start"] = int(match.group(1))
    record["end"] = int(match.group(2))
    record["total_size"] = int(match.group(3)) if match.group(3) != "*" else None
    return body, record


def make_dump_document(language: str, page: dict[str, Any], text: str,
                       qid: str | None, edition: dict[str, Any],
                       range_records: list[dict[str, Any]], removed: dict[str, int],
                       section_count: int, topic_signal: str) -> dict[str, Any]:
    host = f"{language}.wikipedia.org"
    project = f"{language}wiki"
    title = page["title"]
    revision = page["revision_id"]
    title_url = requests.utils.quote(title.replace(" ", "_"), safe="/:()")
    page_url = f"https://{host}/wiki/{title_url}"
    revision_url = f"https://{host}/w/index.php?oldid={revision}"
    history_url = f"https://{host}/w/index.php?title={title_url}&action=history"
    source_bytes = page["wikitext"].encode("utf-8")
    relation_key = f"wikidata:{qid}" if qid else f"wikipedia-page:{language}:{page['page_id']}"
    return {
        "id": f"wiki-{language}-{page['page_id']}-r{revision}",
        "source_family": "encyclopedia",
        "publisher": "Wikipedia",
        "language": language,
        "title": title,
        "url": page_url,
        "source_work_id": qid or f"{project}:{page['page_id']}",
        "wikidata_qid": qid,
        "page_id": page["page_id"],
        "revision": str(revision),
        "revision_sha1": page.get("revision_sha1"),
        "source_updated_at": page.get("timestamp"),
        "retrieved_at": utc_now(),
        "content_sha256": sha(text.encode("utf-8")),
        "source_wikitext_sha256": sha(source_bytes),
        "text": text,
        "attribution": [
            f"Wikipedia contributors; revision {revision_url}",
            f"Contributor history: {history_url}",
        ],
        "contributors": [page.get("contributor", {})] if page.get("contributor") else [],
        "license": {
            "id": "CC-BY-SA-4.0",
            "url": edition["license_url"],
            "share_alike": True,
            "additional_published_license": "GFDL may also apply per Wikimedia dump licensing information",
        },
        "rights_evidence": {
            "checked_at": edition["checked_at"],
            "item_license_url": edition["license_url"],
            "edition_footer_license_reference_url": edition.get("license_reference_url"),
            "license_source_url": edition["license_source_url"],
            "license_source_sha256": edition["license_source_sha256"],
            "dump_legal_url": DUMP_LEGAL_URL,
            "dump_legal_sha256": edition["dump_legal_sha256"],
            "dump_robot_policy_url": DUMP_ROBOT_POLICY_URL,
            "dump_host_robots_status": edition["dump_host_robots_status"],
            "edition_robots_url": edition["edition_robots_url"],
            "edition_robots_sha256": edition["edition_robots_sha256"],
            "route": "version-pinned Wikimedia multistream XML dump; item host footer and dump licensing page checked",
        },
        "dump_source": {
            "project": project,
            "snapshot": edition["snapshot"],
            "job": edition["dump_job"],
            "filename": edition["dump_filename"],
            "url": edition["dump_url"],
            "compressed_file_size": edition["dump_size"],
            "dumpstatus_url": edition["dumpstatus_url"],
            "dumpstatus_sha256": edition["dumpstatus_sha256"],
            "dump_file_md5": edition.get("dump_md5"),
            "range_manifest": [
                {k: row.get(k) for k in ("start", "end", "bytes", "sha256", "content_range", "etag", "raw_file")}
                for row in range_records
            ],
        },
        "relation_keys": [relation_key, f"wikipedia-page:{language}:{page['page_id']}"],
        "original_translation": "source_native_edition",
        "entity_screening": {
            "method": "positive native-language topic cue plus biography/list pattern exclusion",
            "positive_topic_signal": topic_signal,
            "wikidata_q5_checked": False,
            "wikidata_claims_route_robots_allows": False,
            "limitation": "Wikidata claims route is disallowed by robots.txt; no P31 or Q5 assertion was made",
        },
        "extraction": {
            "method": "bounded deterministic wikitext prose extraction",
            "sections_preserved": True,
            "section_count": section_count,
            "removed": removed,
        },
        "changes": "Templates, tables, references, file links and HTML markup omitted; headings and article prose retained.",
    }


def _record_dump_host_setup(status: dict[str, Any], client: WikimediaClient, persist_status) -> bool:
    """Check the dump host once; its current robots endpoint returns 404."""
    robots_path = RAW / "dumps-robots.txt"
    setup = status.setdefault("dump_host_preflight", {})
    if robots_path.exists():
        body = robots_path.read_bytes()
        http_status = 404  # the initial bounded preflight received HTTP 404
        robots_sha = sha(body)
        network_bytes = len(body)
        if not setup.get("manual_probes_accounted"):
            status["network_bytes"] = int(status.get("network_bytes", 0)) + network_bytes
            status.setdefault("manual_setup_probes", []).append({
                "url": f"{DUMP_BASE}/robots.txt", "http_status": http_status,
                "bytes": network_bytes, "sha256": robots_sha, "cached": False,
            })
            # This small directory listing was probed before the collector had its cache accounting.
            status["network_bytes"] += 1_287
            status["manual_setup_probes"].append({
                "url": f"{DUMP_BASE}/dewiki/", "http_status": 200,
                "bytes": 1_287, "sha256": None, "cached": False,
            })
            setup["manual_probes_accounted"] = True
        setup.update({
            "robots_url": f"{DUMP_BASE}/robots.txt",
            "robots_http_status": http_status,
            "robots_sha256": robots_sha,
            "robots_bytes": network_bytes,
            "robots_directives_available": False,
            "robots_reason": "HTTP 404; the dump host published no robots.txt directives",
        })
        robots_text = ""
    else:
        body, meta = client.get(f"{DUMP_BASE}/robots.txt", "dumps-robots.txt", endpoint_kind="dumps_robots", allow_404=True)
        if body is None:
            setup.update({"robots_url": f"{DUMP_BASE}/robots.txt", "robots_http_status": meta.get("http_status"), "robots_reason": meta.get("error")})
            persist_status()
            return False
        setup.update({
            "robots_url": f"{DUMP_BASE}/robots.txt",
            "robots_http_status": meta.get("http_status"),
            "robots_sha256": meta.get("sha256"),
            "robots_bytes": len(body),
            "robots_directives_available": meta.get("http_status") == 200,
            "robots_reason": "HTTP 404; no published directives" if meta.get("http_status") == 404 else None,
        })
        robots_text = body.decode("utf-8", errors="replace") if meta.get("http_status") == 200 else ""
    latest_status_url = f"{DUMP_BASE}/dewiki/20261001/dumpstatus.json"
    setup["dumpstatus_route_allowed"] = robots_allows_url(robots_text, client.user_agent, latest_status_url)
    setup["file_route_allowed"] = robots_allows_url(robots_text, client.user_agent, f"{DUMP_BASE}/dewiki/20261001/dewiki-20261001-pages-articles-multistream.xml.bz2")
    setup["network_byte_budget"] = MAX_NETWORK_BYTES
    persist_status()
    return bool(setup["dumpstatus_route_allowed"] and setup["file_route_allowed"])


def _get_snapshot_manifest(language: str, client: WikimediaClient, status: dict[str, Any], persist_status) -> dict[str, Any] | None:
    project = f"{language}wiki"
    edition = status.setdefault("dump_editions", {}).setdefault(language, {"language": language, "project": project})
    hint = DATE.replace("-", "")[:6] + "01"
    dates = [hint]
    attempted: list[str] = []
    manifest_body: bytes | None = None
    chosen_date: str | None = None
    for snapshot in dates:
        url = f"{DUMP_BASE}/{project}/{snapshot}/dumpstatus.json"
        body, meta = client.get(url, f"dumpstatus-{project}-{snapshot}.json", endpoint_kind="dumpstatus")
        attempted.append(snapshot)
        if body is None:
            if meta.get("http_status") not in (404, None):
                edition.update({"status": "blocked", "reason": f"dumpstatus request returned {meta.get('http_status')}"})
                persist_status()
                return None
            break
        try:
            parsed = json.loads(body)
        except ValueError:
            edition.update({"status": "blocked", "reason": "dumpstatus JSON invalid"})
            persist_status()
            return None
        if manifest_file(parsed, "pages-articles-multistream.xml.bz2", job_hint="articlesmultistream"):
            chosen_date, manifest_body = snapshot, body
            manifest = parsed
            break

    if chosen_date is None:
        listing_url = f"{DUMP_BASE}/{project}/"
        listing, list_meta = client.get(listing_url, f"dump-index-{project}.html", endpoint_kind="dump_directory_index")
        if listing is None:
            edition.update({"status": "blocked", "reason": f"dump directory unavailable ({list_meta.get('http_status')})"})
            persist_status()
            return None
        for snapshot in latest_snapshot_dates(listing):
            if snapshot in attempted or snapshot > hint:
                continue
            url = f"{DUMP_BASE}/{project}/{snapshot}/dumpstatus.json"
            body, meta = client.get(url, f"dumpstatus-{project}-{snapshot}.json", endpoint_kind="dumpstatus")
            attempted.append(snapshot)
            if body is None:
                if meta.get("http_status") not in (404, None):
                    edition.update({"status": "blocked", "reason": f"dumpstatus request returned {meta.get('http_status')}"})
                    persist_status()
                    return None
                continue
            try:
                parsed = json.loads(body)
            except ValueError:
                continue
            if manifest_file(parsed, "pages-articles-multistream.xml.bz2", job_hint="articlesmultistream"):
                chosen_date, manifest_body, manifest = snapshot, body, parsed
                break
    if chosen_date is None or manifest_body is None:
        edition.update({"status": "blocked", "reason": "no completed version-pinned multistream dump in available snapshots"})
        persist_status()
        return None

    match = manifest_file(manifest, "pages-articles-multistream.xml.bz2", job_hint="articlesmultistream")
    assert match is not None
    job_name, dump_meta, filename = match
    if "url" not in dump_meta or not isinstance(dump_meta.get("size"), int):
        edition.update({"status": "blocked", "reason": "dump manifest lacks file URL or size"})
        persist_status()
        return None
    dump_url = DUMP_BASE + dump_meta["url"]
    if not status["dump_host_preflight"].get("file_route_allowed"):
        edition.update({"status": "blocked", "reason": "dump robots preflight disallows the multistream file"})
        persist_status()
        return None
    edition.update({
        "status": "manifest_ready",
        "snapshot": chosen_date,
        "dumpstatus_url": f"{DUMP_BASE}/{project}/{chosen_date}/dumpstatus.json",
        "dumpstatus_sha256": sha(manifest_body),
        "dump_job": job_name,
        "dump_filename": filename,
        "dump_url": dump_url,
        "dump_size": dump_meta["size"],
        "dump_md5": dump_meta.get("md5"),
        "candidate_manifest_checked_at": utc_now(),
        "manifest_attempted_snapshots": attempted,
    })
    pageprops = manifest_file(manifest, "page_props.sql.gz", job_hint="pagepropstable")
    if pageprops:
        _, pageprops_info, pageprops_name = pageprops
        edition.update({
            "pageprops_filename": pageprops_name,
            "pageprops_url": DUMP_BASE + pageprops_info["url"],
            "pageprops_size": pageprops_info.get("size"),
            "pageprops_md5": pageprops_info.get("md5"),
            "pageprops_job": "pagepropstable",
        })
    edition["manifest_raw_file"] = f"data/document_expansion/{DATE}/raw/wiki/dumpstatus-{project}-{chosen_date}.json"
    persist_status()
    return edition


def _preflight_wikipedia_license(language: str, client: WikimediaClient,
                                 status: dict[str, Any], persist_status) -> dict[str, Any] | None:
    edition = status["dump_editions"][language]
    host = f"{language}.wikipedia.org"
    robots_file = RAW / f"preflight-{language}-robots.txt"
    if not robots_file.exists():
        edition.update({"status": "blocked", "reason": "edition robots evidence is missing"})
        persist_status()
        return None
    robots = robots_file.read_text(errors="replace")
    main_url = f"https://{host}/wiki/Main_Page"
    edition["edition_robots_url"] = f"https://{host}/robots.txt"
    edition["edition_robots_sha256"] = sha(robots_file.read_bytes())
    if not robots_allows_url(robots, client.user_agent, main_url):
        edition.update({"status": "blocked", "reason": "edition robots.txt disallows the canonical metadata page route"})
        persist_status()
        return None
    body, meta = client.get(main_url, f"license-{language}-main-page.html", endpoint_kind="edition_license_metadata")
    if body is None:
        edition.update({"status": "blocked", "reason": f"main-page license metadata unavailable ({meta.get('http_status')})"})
        persist_status()
        return None
    license_url, label = parse_main_page_license(body)
    edition["license_source_url"] = main_url
    edition["license_source_sha256"] = meta.get("sha256")
    edition["license_label"] = label
    if not license_url:
        edition.update({"status": "blocked", "reason": "main-page footer did not expose CC BY-SA 4.0"})
        persist_status()
        return None
    if "creativecommons.org/licenses/by-sa/4.0" in license_url:
        edition["license_url"] = license_url
    else:
        edition["license_reference_url"] = requests.compat.urljoin(main_url, license_url)
        edition["license_url"] = "https://creativecommons.org/licenses/by-sa/4.0/"
        edition["license_reference_version_in_url"] = True
    edition["checked_at"] = utc_now()
    persist_status()
    return edition


def _preflight_wikidata_claims(client: WikimediaClient, status: dict[str, Any], persist_status) -> bool:
    url = "https://www.wikidata.org/robots.txt"
    body, meta = client.get(url, "wikidata-robots.txt", endpoint_kind="wikidata_robots", allow_404=True)
    if body is None:
        status["wikidata_claims_preflight"] = {"eligible": False, "reason": "Wikidata robots policy could not be checked", "http_status": meta.get("http_status")}
        persist_status()
        return False
    if meta.get("http_status") == 404:
        robots_text = ""
        robots_sha = meta.get("sha256")
        directives = False
    else:
        robots_text = body.decode("utf-8", errors="replace")
        robots_sha = meta.get("sha256")
        directives = True
    probe_url = api_url("www.wikidata.org", {
        "action": "wbgetentities", "ids": "Q5", "props": "claims", "format": "json",
    })
    allowed = robots_allows_url(robots_text, client.user_agent, probe_url)
    status["wikidata_claims_preflight"] = {
        "eligible": allowed,
        "robots_url": url,
        "robots_http_status": meta.get("http_status"),
        "robots_sha256": robots_sha,
        "directives_available": directives,
        "claims_endpoint": "https://www.wikidata.org/w/api.php?action=wbgetentities",
        "robots_allows_claims_endpoint": allowed,
        "reason": None if allowed else "Wikidata robots.txt disallows the claims endpoint",
        "checked_at": utc_now(),
    }
    persist_status()
    return allowed


def _page_quality(language: str, page: dict[str, Any]) -> tuple[str | None, dict[str, int], int, str | None]:
    if page["namespace"] != 0:
        return None, {}, 0, "non_article_namespace"
    if page["redirect"]:
        return None, {}, 0, "redirect"
    personal, reason = is_list_or_biography(language, page["title"], page["wikitext"])
    if personal:
        return None, {}, 0, reason
    section_count = len(re.findall(r"(?m)^\s*={2,}\s*[^=\n].*?\s*={2,}\s*$", page["wikitext"]))
    if section_count < 3:
        return None, {}, section_count, "too_few_sections"
    text, removed = wikitext_to_plain(language, page["wikitext"])
    chars = len(text)
    if chars < MIN_CHARS:
        return None, removed, section_count, "below_minimum_length"
    if chars > MAX_CHARS:
        return None, removed, section_count, "above_maximum_length"
    if not native_script_ok(language, text):
        return None, removed, section_count, "native_language_check_failed"
    if "{{" in text or "}}" in text or "[[" in text or "]]" in text:
        return None, removed, section_count, "markup_remains_after_extraction"
    return text, removed, section_count, None


def positive_nonpersonal_topic(language: str, page: dict[str, Any], text: str) -> str | None:
    """Require a native-language place, science, institution, or infrastructure cue."""
    template_names = " ".join(re.findall(r"\{\{\s*([^|}\n]{1,120})", page["wikitext"][:6000]))
    evidence = " ".join((page["title"], text[:900], template_names)).casefold()
    for term in POSITIVE_TOPIC_TERMS.get(language, ()):
        term = term.casefold()
        if re.fullmatch(r"[a-zà-öø-ÿ0-9 -]+", term):
            matched = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", evidence)
        else:
            matched = term in evidence
        if matched:
            return term
    return None


def _load_pilot_qids(status: dict[str, Any]) -> set[str]:
    qids = {
        row["qid"] for row in status.get("pilot_qid_lookup", [])
        if isinstance(row, dict) and isinstance(row.get("qid"), str) and re.fullmatch(r"Q\d+", row["qid"])
    }
    manifest = PILOT / "documents.jsonl"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except ValueError:
                continue
            if document.get("publisher") != "Wikipedia":
                continue
            qid = document.get("wikidata_qid")
            if not qid and isinstance(document.get("pageprops"), dict):
                qid = document["pageprops"].get("wikibase_item")
            if not qid and isinstance(document.get("source_work_id"), str):
                qid = document["source_work_id"].removeprefix("wikidata:")
            if isinstance(qid, str) and re.fullmatch(r"Q\d+", qid):
                qids.add(qid)
    return qids


def acquire_wikipedia_dumps(languages: tuple[str, ...] = LANGUAGES,
                            target_per_language: int = TARGET_PER_LANGUAGE) -> dict[str, Any]:
    """Collect current native-edition text from small, version-pinned dump ranges."""
    RAW.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    if not POLICY_CACHE.exists():
        raise FileNotFoundError(f"Cached Wikimedia API policy evidence missing: {POLICY_CACHE}")
    prior_status = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    status = prior_status
    previous_fatal_error = status.pop("fatal_error", None)
    if previous_fatal_error:
        status.setdefault("prior_blockers", []).append(previous_fatal_error)
    status.update({
        "date": DATE,
        "source": "Wikipedia",
        "phase": "dump_preflight",
        "updated_at": utc_now(),
        "requested_languages": list(languages),
        "target_per_language": target_per_language,
        "network_byte_budget": MAX_NETWORK_BYTES,
        "range_bytes_initial_per_language": INITIAL_RANGE_BYTES,
        "range_bytes_maximum_per_language": MAX_RANGE_BYTES_PER_LANGUAGE,
        "dump_policy_evidence": {
            "api_policy_url": POLICY_URL,
            "api_policy_sha256": sha(POLICY_CACHE.read_bytes()),
            "dump_legal_url": DUMP_LEGAL_URL,
            "robot_policy_url": DUMP_ROBOT_POLICY_URL,
        },
    })
    status.setdefault("started_at", utc_now())
    status.setdefault("network_bytes", 0)
    status.setdefault("requests", [])
    status.setdefault("dump_editions", {})
    status.setdefault("wikidata_batches", [])
    status.setdefault("accepted_documents", [])
    status.setdefault("rejection_counts", {})
    status.setdefault("dump_host_preflight", {})
    all_existing_docs = load_prior_documents()
    quarantined_ids = set(status.get("quarantined_document_ids", []))
    existing_docs = [document for document in all_existing_docs if document.get("id") not in quarantined_ids]
    accepted_qids = {
        d.get("wikidata_qid") or d.get("source_work_id")
        for d in all_existing_docs if d.get("publisher") == "Wikipedia"
    }
    accepted_qids.update(_load_pilot_qids(status))
    if quarantined_ids:
        status["accepted_documents"] = [
            row for row in status.get("accepted_documents", [])
            if row.get("id") not in quarantined_ids
        ]

    def persist_status() -> None:
        status["updated_at"] = utc_now()
        status["accepted_total"] = len(existing_docs)
        status["actual_languages"] = sorted({
            d.get("language") for d in existing_docs if d.get("publisher") == "Wikipedia"
        })
        write_json(STATUS, status)

    client = WikimediaClient(status, persist_status)
    status["user_agent"] = {
        "product": USER_AGENT_PRODUCT,
        "contact": "configured mailto contact; value not persisted",
        "sha256": sha(client.user_agent.encode()),
    }
    refresh_route_checks(status, client.user_agent)
    persist_status()

    if not _record_dump_host_setup(status, client, persist_status):
        status["phase"] = "blocked"
        status["fatal_error"] = {"type": "robots", "message": "dumpservice file routes were not allowed"}
        persist_status()
        return status

    legal_file = RAW / "wikimedia-dumps-legal.html"
    if legal_file.exists():
        legal_body = legal_file.read_bytes()
        legal_meta = {"sha256": sha(legal_body), "cached": True}
    else:
        legal_body, legal_meta = client.get(DUMP_LEGAL_URL, legal_file.name, endpoint_kind="dump_legal_policy")
    if legal_body is None:
        status["phase"] = "blocked"
        status["fatal_error"] = {"type": "dump_licensing_policy", "message": "official dump licensing page unavailable"}
        persist_status()
        return status
    from bs4 import BeautifulSoup
    legal_text = " ".join(BeautifulSoup(legal_body, "html.parser").get_text(" ", strip=True).split())
    if "Creative Commons Attribution-Share-Alike 4.0" not in legal_text or "Exceptions" not in legal_text:
        status["phase"] = "blocked"
        status["fatal_error"] = {"type": "dump_licensing_policy", "message": "official dump licensing terms did not match expected text"}
        persist_status()
        return status
    status["dump_legal_evidence"] = {
        "url": DUMP_LEGAL_URL,
        "sha256": legal_meta.get("sha256"),
        "cached": legal_meta.get("cached", False),
        "cc_by_sa_4_present": True,
        "exceptions_section_present": True,
    }
    persist_status()

    status["phase"] = "edition_dump_preflight"
    persist_status()
    for language in languages:
        prior = status["dump_editions"].get(language, {})
        if prior.get("status") == "eligible" and prior.get("accepted_documents", 0) >= target_per_language:
            continue
        edition = _get_snapshot_manifest(language, client, status, persist_status)
        if edition is None:
            continue
        if edition.get("status") == "eligible" and edition.get("license_url"):
            continue
        edition = _preflight_wikipedia_license(language, client, status, persist_status)
        if edition is None:
            continue
        edition["dump_legal_sha256"] = status["dump_legal_evidence"]["sha256"]
        edition["dump_host_robots_status"] = status["dump_host_preflight"].get("robots_http_status")
        edition["status"] = "eligible"
        persist_status()

    claims_allowed = _preflight_wikidata_claims(client, status, persist_status)
    status["entity_screening_policy"] = {
        "method": "positive native-language topic cues and biography/list pattern exclusions",
        "wikidata_q5_claims_available": claims_allowed,
        "wikidata_q5_claims_used": False,
        "robots_sha256": status.get("wikidata_claims_preflight", {}).get("robots_sha256"),
        "limitation": None if claims_allowed else "Wikidata robots.txt disallows wbgetentities; no P31 or Q5 assertion will be made",
    }
    persist_status()

    status["phase"] = "multistream_acquisition"
    persist_status()
    for language in languages:
        edition = status["dump_editions"].get(language, {})
        if edition.get("status") != "eligible":
            continue
        have_count = sum(
            1 for document in existing_docs
            if document.get("publisher") == "Wikipedia" and document.get("language") == language
        )
        edition["accepted_documents"] = have_count
        if have_count >= target_per_language:
            continue
        if not edition.get("pageprops_url"):
            edition["status"] = "blocked"
            edition["reason"] = "dumpstatus lacks page_props QID mapping required for global entity screening"
            persist_status()
            continue

        # The first eight compressed megabytes normally include page IDs aligned with the dump prefix.
        pageprop_records = edition.setdefault("pageprops_ranges", [])
        if not pageprop_records:
            prop_size = int(edition.get("pageprops_size") or 0)
            wanted_end = min(INITIAL_RANGE_BYTES, prop_size) - 1
            if wanted_end >= 0:
                body, meta = fetch_dump_range(
                    client, edition["pageprops_url"], language, edition["snapshot"],
                    0, wanted_end, "pageprops",
                )
                if body is not None:
                    pageprop_records.append({
                        "start": 0, "end": meta["end"], "bytes": len(body),
                        "sha256": meta["sha256"], "content_range": meta.get("content_range"),
                        "raw_file": meta["raw_file"],
                    })
                else:
                    edition["status"] = "blocked"
                    edition["reason"] = f"page_props Range request failed ({meta.get('error', meta.get('http_status'))})"
                    persist_status()
                    continue
        pageprops_sql = read_dump_ranges(pageprop_records)
        qids_by_page = pageprops_qids_from_gzip_prefix(pageprops_sql)
        edition["pageprops_qids_found"] = len(qids_by_page)
        edition["pageprops_mapping_sha256"] = sha(pageprops_sql)
        persist_status()

        range_records = edition.setdefault("ranges", [])
        parsed_pages: list[dict[str, Any]] = []
        seen_page_ids: set[int] = set()
        current_limit = INITIAL_RANGE_BYTES
        while have_count < target_per_language and current_limit <= MAX_RANGE_BYTES_PER_LANGUAGE:
            merged_range_bytes = sum(int(record.get("bytes", 0)) for record in range_records)
            if merged_range_bytes < current_limit:
                current_start = max((int(record["end"]) + 1 for record in range_records), default=0)
                while current_start < min(current_limit, int(edition["dump_size"])):
                    request_end = min(current_limit, int(edition["dump_size"])) - 1
                    body, meta = fetch_dump_range(
                        client, edition["dump_url"], language, edition["snapshot"],
                        current_start, request_end, "multistream",
                    )
                    if body is None:
                        edition["status"] = "blocked"
                        edition["reason"] = f"multistream Range request failed ({meta.get('error', meta.get('http_status'))})"
                        persist_status()
                        break
                    row = {
                        "start": current_start,
                        "end": meta["end"],
                        "bytes": len(body),
                        "sha256": meta["sha256"],
                        "content_range": meta.get("content_range"),
                        "etag": meta.get("etag"),
                        "raw_file": meta["raw_file"],
                    }
                    range_records.append(row)
                    current_start = meta["end"] + 1
                    persist_status()
                if edition.get("status") == "blocked":
                    break
            compressed_prefix = read_dump_ranges(range_records)
            parsed_pages, complete_streams, expanded_bytes = parse_multistream_prefix(compressed_prefix)
            edition["complete_bzip2_streams"] = complete_streams
            edition["complete_xml_pages_seen"] = len(parsed_pages)
            edition["expanded_xml_bytes"] = expanded_bytes
            edition["parsed_at"] = utc_now()
            persist_status()

            candidates: list[dict[str, Any]] = []
            page_ids = {row["page_id"] for row in parsed_pages}
            missing_qid_page_ids = page_ids.difference(qids_by_page)
            mapping_stops_before_parsed_pages = (
                bool(qids_by_page) and max(qids_by_page) < max(page_ids, default=0)
            )
            if missing_qid_page_ids and mapping_stops_before_parsed_pages and pageprop_records and pageprop_records[-1]["end"] + 1 < min(MAX_RANGE_BYTES_PER_LANGUAGE, int(edition["pageprops_size"])):
                prop_start = pageprop_records[-1]["end"] + 1
                prop_end = min(MAX_RANGE_BYTES_PER_LANGUAGE, int(edition["pageprops_size"])) - 1
                prop_body, prop_meta = fetch_dump_range(
                    client, edition["pageprops_url"], language, edition["snapshot"],
                    prop_start, prop_end, "pageprops",
                )
                if prop_body is not None:
                    pageprop_records.append({
                        "start": prop_start, "end": prop_meta["end"], "bytes": len(prop_body),
                        "sha256": prop_meta["sha256"], "content_range": prop_meta.get("content_range"),
                        "raw_file": prop_meta["raw_file"],
                    })
                    qids_by_page = pageprops_qids_from_gzip_prefix(read_dump_ranges(pageprop_records))
                    edition["pageprops_qids_found"] = len(qids_by_page)
                    persist_status()

            for page in parsed_pages:
                if page["page_id"] in seen_page_ids:
                    continue
                seen_page_ids.add(page["page_id"])
                qid = qids_by_page.get(page["page_id"])
                if not qid:
                    record_rejection(status, language, "pageprops_qid_missing")
                    continue
                if qid in accepted_qids:
                    record_rejection(status, language, "duplicate_wikidata_entity_or_pilot_concept")
                    continue
                text, removed, sections, reject = _page_quality(language, page)
                if reject:
                    record_rejection(status, language, reject)
                    continue
                topic_signal = positive_nonpersonal_topic(language, page, text)
                if not topic_signal:
                    record_rejection(status, language, "no_positive_nonpersonal_topic_signal")
                    continue
                candidates.append({
                    "page": page, "qid": qid, "text": text,
                    "removed": removed, "sections": sections,
                    "topic_signal": topic_signal,
                })
            candidates.sort(key=lambda row: hashlib.sha256(
                f"{edition['snapshot']}:{language}:{row['qid']}".encode()
            ).hexdigest())
            candidates = candidates[:40]
            if claims_allowed:
                wikidata_result = wikidata_classes([row["qid"] for row in candidates], client, status, persist_status)
                if wikidata_result is None:
                    edition["status"] = "blocked"
                    edition["reason"] = "Wikidata claims batch failed; edition stopped"
                    persist_status()
                    break
                classes_by_qid, personal_signal_qids = wikidata_result
            else:
                classes_by_qid, personal_signal_qids = {}, set()
            for candidate in candidates:
                if have_count >= target_per_language:
                    break
                qid = candidate["qid"]
                classes = classes_by_qid.get(qid, set())
                personal, personal_reason = is_person_or_list(qid, classes, personal_signal_qids)
                if personal:
                    record_rejection(status, language, personal_reason or "personal_or_list_entity")
                    continue
                if qid in accepted_qids:
                    record_rejection(status, language, "duplicate_wikidata_entity_after_claims")
                    continue
                document = make_dump_document(
                    language, candidate["page"], candidate["text"], qid,
                    edition, range_records, candidate["removed"], candidate["sections"],
                    candidate["topic_signal"],
                )
                append_document(document)
                existing_docs.append(document)
                accepted_qids.add(qid)
                have_count += 1
                status["accepted_documents"].append({
                    "id": document["id"],
                    "language": language,
                    "page_id": document["page_id"],
                    "wikidata_qid": qid,
                    "revision": document["revision"],
                    "source_updated_at": document["source_updated_at"],
                    "characters": len(candidate["text"]),
                    "content_sha256": document["content_sha256"],
                    "snapshot": edition["snapshot"],
                })
                edition["accepted_documents"] = have_count
                persist_status()

            if have_count >= target_per_language:
                edition["status"] = "eligible"
                edition["reason"] = None
                break
            if current_limit >= MAX_RANGE_BYTES_PER_LANGUAGE:
                break
            current_limit = MAX_RANGE_BYTES_PER_LANGUAGE

        if have_count < target_per_language:
            edition["status"] = "incomplete"
            edition["reason"] = f"only {have_count} of {target_per_language} documents after the {MAX_RANGE_BYTES_PER_LANGUAGE}-byte per-language range cap"
        edition["accepted_documents"] = have_count
        persist_status()

    totals = {
        language: sum(
            1 for document in existing_docs
            if document.get("publisher") == "Wikipedia" and document.get("language") == language
        )
        for language in languages
    }
    status["language_totals"] = totals
    status["accepted_total"] = sum(totals.values())
    status["actual_languages"] = sorted(language for language, count in totals.items() if count)
    status["unmet_targets"] = {language: target_per_language - count for language, count in totals.items() if count < target_per_language}
    status["phase"] = "complete" if not status["unmet_targets"] else "incomplete"
    status["finished_at"] = utc_now()
    persist_status()
    return status


def is_list_or_biography(language: str, title: str, wikitext: str) -> tuple[bool, str | None]:
    lower_title = " ".join(title.casefold().split())
    markers = LIST_TITLE_MARKERS.get(language, ())
    if any(marker.casefold() in lower_title for marker in markers):
        return True, "list_title"
    if BIOGRAPHY_SIGNALS.search(wikitext[:8000]):
        return True, "biography_signal"
    lower = wikitext.casefold()
    template_names = " ".join(re.findall(r"\{\{\s*([^|}\n]{1,120})", lower[:12000]))
    if BIOGRAPHY_TEMPLATE_SIGNALS.search(template_names):
        return True, "biography_template"
    category_markers = (
        "living people", "births", "deaths", "nacidos", "nacidas", "fallecidos", "nacimiento",
        "geburt", "geboren", "gestorben", "naissance", "décès", "deces", "nati", "nato", "morti", "morte",
        "födelse", "födda", "syntynyt", "kuollut", "urodzeni", "zmarli", "narozen", "narození",
        "родивш", "умерш", "народивш", "померл", "doğumlular", "ölenler", "kelahiran", "kematian",
        "sinh năm", "mất năm", "출생", "사망", "生年", "没年", "出生", "死亡", "مواليد", "وفيات",
        "जन्म", "मृत्यु", "জন্ম", "মৃত্যু", "பிறந்த", "இறந்த", "ילידי", "נפטרים",
        "waliozaliwa", "alizaliwa", "yaliyofariki", "alifariki",
        "นักปรัชญา", "นักวิทยาศาสตร์", "นักเขียน", "राजा", "कवि", "शासक", "व्यक्ति",
        "मुग़ल उच्चाधिकारी", "भक्तिकाल के कवि", "nati per", "membri dell'accademia",
        "accademici", "professori", "scienziati", "biologi", "biochimici", "filosofi",
        "scrittori", "poeti", "compositori", "artisti", "miembros de la academia",
        "cientistas", "biólogos", "bioquímicos", "professores", "nascidos", "sepultados",
        "comunistas", "heróis do trabalho", "漫画家", "俳優", "政治家", "小説家", "作家",
        "詩人", "学者", "哲学者", "科学者", "人物", "年生", "年没", "年逝世",
        "哲学家", "物理学家", "科学家", "诗人", "作家", "人物", "逝世者", "出生于",
        "manga-artist-stub", "philosopher", "philosophers", "biologists", "scientists",
        "professors", "poets", "writers", "politicians", "born in", "deaths in",
    )
    category_text = " ".join(re.findall(r"\[\[[^:\]]+:[^\]]+\]\]", lower))
    if any(marker in category_text for marker in category_markers):
        return True, "biography_category"
    if re.search(r"\b(?:18|19|20)\d{2}\s*(?:年生|年没|年逝世|年出生)\b", category_text):
        return True, "birth_year_category"
    return False, None


def qid_of(page: dict[str, Any]) -> str | None:
    value = page.get("pageprops", {}).get("wikibase_item")
    return value if isinstance(value, str) and re.fullmatch(r"Q\d+", value) else None


def get_pilot_qids(client: WikimediaClient, status: dict[str, Any], persist_status) -> set[str]:
    qids: set[str] = set()
    pilot_manifest = PILOT / "documents.jsonl"
    existing = []
    if pilot_manifest.exists():
        existing = [json.loads(line) for line in pilot_manifest.read_text().splitlines() if line.strip()]
    for doc in existing:
        if doc.get("publisher") != "Wikipedia":
            continue
        language = doc.get("language")
        title = doc.get("title")
        if language not in {"de", "ja", "pt"} or not title:
            continue
        edition = status.get("editions", {}).get(language, {})
        if not edition.get("eligible") or not edition.get("robots", {}).get("allows_api_query"):
            status.setdefault("pilot_qid_lookup", []).append({
                "language": language,
                "status": "skipped",
                "reason": "edition robots.txt disallows the Action API route",
            })
            persist_status()
            continue
        params = {
            "action": "query", "prop": "pageprops", "titles": title,
            "format": "json", "formatversion": 2,
        }
        url = api_url(f"{language}.wikipedia.org", params)
        name = f"pilot-qid-{language}.json"
        body, request_meta = client.get(url, name, endpoint_kind="pilot_page_qid")
        if body is None:
            status.setdefault("pilot_qid_lookup", []).append({"language": language, "status": "unavailable"})
            persist_status()
            continue
        try:
            page = json.loads(body).get("query", {}).get("pages", [{}])[0]
            qid = qid_of(page)
            if qid:
                qids.add(qid)
            status.setdefault("pilot_qid_lookup", []).append({
                "language": language, "pageid": page.get("pageid"), "qid": qid,
                "response_sha256": request_meta.get("sha256"),
            })
        except (ValueError, IndexError, TypeError):
            status.setdefault("pilot_qid_lookup", []).append({"language": language, "status": "parse_error"})
        persist_status()
    return qids


def preflight_edition(language: str, client: WikimediaClient, status: dict[str, Any], persist_status) -> dict[str, Any]:
    host = f"{language}.wikipedia.org"
    result: dict[str, Any] = {
        "language": language,
        "host": host,
        "checked_at": utc_now(),
        "eligible": False,
        "accepted_documents": 0,
    }
    robots_url = f"https://{host}/robots.txt"
    robots_body, robots_meta = client.get(robots_url, f"preflight-{language}-robots.txt", endpoint_kind="robots")
    result["robots"] = {
        "url": robots_url,
        "sha256": robots_meta.get("sha256"),
        "http_status": robots_meta.get("http_status"),
    }
    if robots_body is None:
        result["reason"] = "robots.txt unavailable; edition preflight incomplete"
        status["editions"][language] = result
        persist_status()
        return result

    robots_text = robots_body.decode("utf-8", errors="replace")
    api_path = f"https://{host}/w/api.php?action=query"
    result["robots"]["allows_api_query"] = robots_allows_url(robots_text, client.user_agent, api_path)
    if not result["robots"]["allows_api_query"]:
        result["reason"] = "robots.txt disallows the API route"
        status["editions"][language] = result
        persist_status()
        return result

    params = {
        "action": "query", "meta": "siteinfo", "siprop": "general|rightsinfo",
        "format": "json", "formatversion": 2,
    }
    body, request_meta = client.get(api_url(host, params), f"preflight-{language}-siteinfo.json", endpoint_kind="edition_siteinfo")
    if body is None:
        result["reason"] = request_meta.get("error", "siteinfo request failed")
        result["siteinfo_http_status"] = request_meta.get("http_status")
        status["editions"][language] = result
        persist_status()
        return result
    try:
        info = json.loads(body).get("query", {})
        general = info.get("general", {})
        rights = info.get("rightsinfo", {})
    except (ValueError, TypeError):
        result["reason"] = "invalid siteinfo JSON"
        status["editions"][language] = result
        persist_status()
        return result
    rights_url = rights.get("url", "")
    result["siteinfo"] = {
        "sitename": general.get("sitename"),
        "lang": general.get("lang"),
        "rights": general.get("rights"),
        "rights_url": rights_url,
        "rights_text": rights.get("text"),
        "response_sha256": request_meta.get("sha256"),
    }
    if "creativecommons.org/licenses/by-sa/4.0" not in rights_url:
        result["reason"] = "edition rightsinfo is not CC BY-SA 4.0"
    elif not general.get("lang"):
        result["reason"] = "edition did not report a language"
    else:
        result["eligible"] = True
        result["reason"] = None
    status["editions"][language] = result
    persist_status()
    return result


def wikidata_classes(qids: list[str], client: WikimediaClient, status: dict[str, Any], persist_status) -> tuple[dict[str, set[str]], set[str]] | None:
    if not qids:
        return {}, set()
    result: dict[str, set[str]] = {}
    personal_signals: set[str] = set()
    for offset in range(0, len(qids), WIKIDATA_BATCH_SIZE):
        batch = qids[offset:offset + WIKIDATA_BATCH_SIZE]
        batch_no = len(status.setdefault("wikidata_batches", [])) + 1
        params = {
            "action": "wbgetentities", "ids": "|".join(batch), "props": "claims",
            "format": "json",
        }
        url = api_url("www.wikidata.org", params)
        body, request_meta = client.get(url, f"wikidata-claims-{batch_no:03d}.json", endpoint_kind="wikidata_claims")
        if body is None:
            status["wikidata_batches"].append({"batch": batch_no, "qids": len(batch), "status": "unavailable", "http_status": request_meta.get("http_status")})
            persist_status()
            return None
        try:
            entities = json.loads(body).get("entities", {})
        except (ValueError, TypeError):
            status["wikidata_batches"].append({"batch": batch_no, "qids": len(batch), "status": "invalid_json"})
            persist_status()
            return None
        for qid, entity in entities.items():
            claims = entity.get("claims", {})
            if PERSON_SIGNAL_PROPERTIES.intersection(claims):
                personal_signals.add(qid)
            classes = set()
            for claim in claims.get("P31", []):
                try:
                    value = claim["mainsnak"]["datavalue"]["value"]["id"]
                    if isinstance(value, str):
                        classes.add(value)
                except (KeyError, TypeError):
                    continue
            result[qid] = classes
        status["wikidata_batches"].append({
            "batch": batch_no, "qids": len(batch), "status": "ok",
            "response_sha256": request_meta.get("sha256"),
        })
        persist_status()
    return result, personal_signals


def is_person_or_list(qid: str, classes: set[str], personal_signals: set[str]) -> tuple[bool, str | None]:
    if "Q5" in classes:
        return True, "wikidata_instance_of_human"
    if WIKIMEDIA_LIST_ITEM_QID in classes:
        return True, "wikidata_list_article"
    if qid in personal_signals:
        return True, "wikidata_personal_life_claims"
    return False, None


def append_document(document: dict[str, Any]) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    jsonl = DATA / "wiki-documents.jsonl"
    path = DOCS / f"{document['id']}.json"
    encoded = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        prior = json.loads(path.read_text())
        if (prior.get("id"), prior.get("wikidata_qid"), prior.get("content_sha256")) != (
            document.get("id"), document.get("wikidata_qid"), document.get("content_sha256")
        ):
            raise RuntimeError(f"Immutable source record collision: {path.name}")
        document = prior
    else:
        with path.open("x") as f:
            f.write(encoded)
            f.flush()
    if (DATA / "wiki-documents.jsonl").exists() and any(
        line and json.loads(line).get("id") == document["id"]
        for line in (DATA / "wiki-documents.jsonl").read_text().splitlines()
    ):
        return
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def record_rejection(status: dict[str, Any], language: str, reason: str) -> None:
    counts = status.setdefault("rejection_counts", {}).setdefault(language, {})
    counts[reason] = counts.get(reason, 0) + 1


def make_document(language: str, page: dict[str, Any], qid: str, text: str,
                  classes: set[str], contributors: list[dict[str, Any]],
                  response_sha256: str, edition: dict[str, Any], retrieved_at: str) -> dict[str, Any]:
    revision = (page.get("revisions") or [{}])[0]
    pageid = int(page["pageid"])
    revid = int(revision["revid"])
    title = page["title"]
    host = edition["host"]
    fullurl = page.get("fullurl") or f"https://{host}/wiki/{requests.utils.quote(title.replace(' ', '_'))}"
    revision_url = f"https://{host}/w/index.php?oldid={revid}"
    history_url = f"https://{host}/w/index.php?title={requests.utils.quote(title.replace(' ', '_'))}&action=history"
    license_url = edition["siteinfo"]["rights_url"]
    return {
        "id": f"wiki-{language}-{pageid}-r{revid}",
        "source_family": "encyclopedia",
        "publisher": "Wikipedia",
        "language": language,
        "title": title,
        "url": fullurl,
        "source_work_id": qid,
        "wikidata_qid": qid,
        "page_id": pageid,
        "revision": str(revid),
        "source_updated_at": revision.get("timestamp"),
        "retrieved_at": retrieved_at,
        "content_sha256": sha(text.encode("utf-8")),
        "text": text,
        "attribution": [
            f"Wikipedia contributors; revision {revision_url}",
            f"Contributor history: {history_url}",
        ],
        "contributors": contributors,
        "license": {"id": "CC-BY-SA-4.0", "url": license_url, "share_alike": True},
        "rights_evidence": {
            "checked_at": edition["checked_at"],
            "policy_url": POLICY_URL,
            "policy_sha256": edition["policy_sha256"],
            "item_license_url": license_url,
            "edition_host": host,
            "route": "official MediaWiki generator=random query API; edition rightsinfo and robots preflight",
            "robots_url": edition["robots"]["url"],
            "robots_sha256": edition["robots"]["sha256"],
            "preflight_response_sha256": edition["siteinfo"]["response_sha256"],
            "source_response_sha256": response_sha256,
            "wikidata_instance_of": sorted(classes),
        },
        "relation_keys": [f"wikidata:{qid}", f"wikipedia-page:{language}:{pageid}"],
        "original_translation": "source_native_edition",
        "extraction": {
            "method": "MediaWiki plaintext extracts",
            "sections_preserved": True,
            "section_count": len(re.findall(r"(?m)^\s*={2,}\s*[^=\n].*?\s*={2,}\s*$", text)),
        },
        "changes": "MediaWiki plaintext extraction; native-language article text retained; non-text media omitted by API",
    }


def acquire_wikipedia() -> dict[str, Any]:
    RAW.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    if not POLICY_CACHE.exists():
        raise FileNotFoundError(f"Cached Wikimedia API policy evidence missing: {POLICY_CACHE}")
    policy_bytes = POLICY_CACHE.read_bytes()
    policy_copy = RAW / "wikimedia-api-policy.html"
    if not policy_copy.exists():
        shutil.copyfile(POLICY_CACHE, policy_copy)
    now = utc_now()
    status: dict[str, Any] = {
        "date": DATE,
        "source": "Wikipedia",
        "phase": "edition_preflight",
        "started_at": now,
        "updated_at": now,
        "target_per_language": TARGET_PER_LANGUAGE,
        "max_batches_per_language": MAX_BATCHES_PER_LANGUAGE,
        "requested_languages": list(LANGUAGES),
        "policy_evidence": {"url": POLICY_URL, "sha256": sha(policy_bytes), "cached_from": str(POLICY_CACHE.relative_to(ROOT))},
        "user_agent": {"product": "HorschtDocumentExpansion/0.1", "contact": "configured mailto contact; value not persisted"},
        "network_byte_budget": MAX_NETWORK_BYTES,
        "network_bytes": 0,
        "requests": [],
        "editions": {},
        "wikidata_batches": [],
        "pilot_qid_lookup": [],
        "accepted_documents": [],
        "rejection_counts": {},
        "fatal_error": None,
        "budget_exhausted": False,
    }
    if STATUS.exists():
        old = json.loads(STATUS.read_text())
        if old.get("requested_languages") == list(LANGUAGES):
            status["network_bytes"] = int(old.get("network_bytes", 0))
            status["requests"] = list(old.get("requests", []))
            status["editions"] = dict(old.get("editions", {}))
            status["wikidata_batches"] = list(old.get("wikidata_batches", []))
            status["pilot_qid_lookup"] = list(old.get("pilot_qid_lookup", []))
            status["accepted_documents"] = list(old.get("accepted_documents", []))
            status["rejection_counts"] = dict(old.get("rejection_counts", {}))
            status["budget_exhausted"] = bool(old.get("budget_exhausted", False))

    existing_docs = load_prior_documents()
    accepted_qids = {
        d.get("wikidata_qid") or d.get("source_work_id")
        for d in existing_docs if d.get("publisher") == "Wikipedia"
    }
    accepted_by_language = {
        language: sum(1 for d in existing_docs if d.get("publisher") == "Wikipedia" and d.get("language") == language)
        for language in LANGUAGES
    }

    def persist_status() -> None:
        status["updated_at"] = utc_now()
        status["accepted_total"] = len(existing_docs) + sum(
            1 for row in status["accepted_documents"] if row.get("id") not in {d.get("id") for d in existing_docs}
        )
        write_json(STATUS, status)

    client = WikimediaClient(status, persist_status)
    status["user_agent"]["sha256"] = sha(client.user_agent.encode())
    refresh_route_checks(status, client.user_agent)
    persist_status()

    try:
        for language in LANGUAGES:
            if status.get("budget_exhausted"):
                break
            if language in status["editions"]:
                cached = status["editions"][language]
                if cached.get("eligible") or cached.get("reason"):
                    continue
            preflight_edition(language, client, status, persist_status)
            edition = status["editions"][language]
            edition["policy_sha256"] = sha(policy_bytes)
            persist_status()

        # Map old pilot source concepts to Wikidata when their exact page titles resolve.
        pilot_qids = get_pilot_qids(client, status, persist_status)
        accepted_qids.update(pilot_qids)
        status["pilot_qids_excluded"] = len(pilot_qids)
        persist_status()

        status["phase"] = "random_acquisition"
        persist_status()
        for language in LANGUAGES:
            if status.get("budget_exhausted"):
                break
            edition = status["editions"].get(language, {})
            if not edition.get("eligible"):
                continue
            accepted_by_language[language] = sum(
                1 for d in existing_docs
                if d.get("publisher") == "Wikipedia" and d.get("language") == language
            )
            if accepted_by_language[language] >= TARGET_PER_LANGUAGE:
                continue
            seen_pageids: set[int] = set()
            for batch_no in range(1, MAX_BATCHES_PER_LANGUAGE + 1):
                if status.get("budget_exhausted"):
                    break
                if accepted_by_language[language] >= TARGET_PER_LANGUAGE:
                    break
                params = {
                    "action": "query",
                    "generator": "random",
                    "grnnamespace": 0,
                    "grnlimit": 10,
                    "prop": "extracts|revisions|info|pageprops|contributors",
                    "explaintext": 1,
                    "rvprop": "ids|timestamp|user|userid|comment",
                    "rvlimit": 1,
                    "inprop": "url",
                    "pclimit": 10,
                    "meta": "siteinfo",
                    "siprop": "rightsinfo",
                    "format": "json",
                    "formatversion": 2,
                }
                url = api_url(edition["host"], params)
                batch_file = f"sample-{language}-{batch_no:02d}.json"
                body, request_meta = client.get(url, batch_file, endpoint_kind="random_article_batch")
                edition["random_batches_attempted"] = batch_no
                if body is None:
                    edition["reason"] = request_meta.get("error", "random API batch failed")
                    edition["eligible"] = False
                    persist_status()
                    break
                try:
                    response = json.loads(body)
                    query = response.get("query", {})
                    pages = query.get("pages", [])
                    rights = query.get("rightsinfo", {})
                    rights_url = rights.get("url", edition.get("siteinfo", {}).get("rights_url", ""))
                except (ValueError, TypeError):
                    record_rejection(status, language, "invalid_random_api_json")
                    persist_status()
                    continue
                if "creativecommons.org/licenses/by-sa/4.0" not in rights_url:
                    edition["reason"] = "random response rightsinfo changed or is incompatible"
                    edition["eligible"] = False
                    persist_status()
                    break

                candidates = []
                for page in pages:
                    try:
                        pageid = int(page["pageid"])
                    except (KeyError, TypeError, ValueError):
                        record_rejection(status, language, "missing_page_id")
                        continue
                    if pageid in seen_pageids:
                        record_rejection(status, language, "duplicate_page_in_edition")
                        continue
                    seen_pageids.add(pageid)
                    text = page.get("extract", "")
                    pageprops = page.get("pageprops", {})
                    accepted, reason, heading_count = classify_text(language, page.get("title", ""), text, pageprops)
                    if not accepted:
                        record_rejection(status, language, reason or "ineligible_text")
                        continue
                    qid = qid_of(page)
                    if not qid:
                        record_rejection(status, language, "missing_wikidata_item")
                        continue
                    if qid in accepted_qids:
                        record_rejection(status, language, "duplicate_wikidata_entity_or_pilot_concept")
                        continue
                    if page.get("ns") not in (None, 0):
                        record_rejection(status, language, "non_article_namespace")
                        continue
                    if not page.get("revisions"):
                        record_rejection(status, language, "missing_revision_metadata")
                        continue
                    candidates.append({
                        "page": page,
                        "qid": qid,
                        "text": text,
                        "heading_count": heading_count,
                        "response_sha256": request_meta.get("sha256"),
                        "batch": batch_no,
                    })
                wikidata_result = wikidata_classes([c["qid"] for c in candidates], client, status, persist_status)
                if wikidata_result is None:
                    edition["reason"] = "Wikidata claims lookup failed; edition stopped"
                    edition["eligible"] = False
                    persist_status()
                    break
                classes_by_qid, personal_signal_qids = wikidata_result

                for candidate in candidates:
                    qid = candidate["qid"]
                    classes = classes_by_qid.get(qid, set())
                    personal, personal_reason = is_person_or_list(qid, classes, personal_signal_qids)
                    if personal:
                        record_rejection(status, language, personal_reason or "personal_or_list_entity")
                        continue
                    page = candidate["page"]
                    contributors = []
                    for contributor in page.get("contributors", [])[:10]:
                        entry: dict[str, Any] = {}
                        if contributor.get("name"):
                            entry["name"] = contributor["name"]
                        if contributor.get("userid") is not None:
                            entry["user_id"] = contributor["userid"]
                        if contributor.get("anon"):
                            entry["anonymous"] = True
                        contributors.append(entry)
                    document = make_document(
                        language, page, qid, candidate["text"], classes, contributors,
                        candidate["response_sha256"], edition, utc_now(),
                    )
                    append_document(document)
                    existing_docs.append(document)
                    accepted_qids.add(qid)
                    accepted_by_language[language] += 1
                    accepted_row = {
                        "id": document["id"],
                        "language": language,
                        "page_id": document["page_id"],
                        "wikidata_qid": qid,
                        "revision": document["revision"],
                        "source_updated_at": document["source_updated_at"],
                        "characters": len(candidate["text"]),
                        "content_sha256": document["content_sha256"],
                        "random_batch": batch_no,
                    }
                    status["accepted_documents"].append(accepted_row)
                    edition["accepted_documents"] = accepted_by_language[language]
                    persist_status()

                edition["last_batch_candidates"] = len(candidates)
                edition["accepted_documents"] = accepted_by_language[language]
                persist_status()
            if accepted_by_language[language] < TARGET_PER_LANGUAGE and edition.get("eligible"):
                edition["reason"] = "target_not_reached_within_bounded_random_sampling"
                edition["eligible"] = False
                persist_status()

    except Exception as exc:
        # Persist only an exception type and safe static message; never dump response bodies.
        status["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)[:240]}
        status["phase"] = "failed"
        persist_status()
        raise

    totals = {
        language: sum(
            1 for doc in existing_docs
            if doc.get("publisher") == "Wikipedia" and doc.get("language") == language
        )
        for language in LANGUAGES
    }
    status["language_totals"] = totals
    status["accepted_total"] = sum(totals.values())
    status["actual_languages"] = sorted(language for language, count in totals.items() if count)
    status["unmet_targets"] = {language: TARGET_PER_LANGUAGE - count for language, count in totals.items() if count < TARGET_PER_LANGUAGE}
    status["phase"] = "complete" if not status["unmet_targets"] else "incomplete"
    status["finished_at"] = utc_now()
    persist_status()
    return status


if __name__ == "__main__":
    result = acquire_wikipedia_dumps()
    print(json.dumps({
        "phase": result["phase"],
        "accepted_total": result.get("accepted_total", 0),
        "language_totals": result.get("language_totals", {}),
        "blocked_or_unmet": result.get("unmet_targets", {}),
        "network_bytes": result.get("network_bytes", 0),
        "request_count": len(result.get("requests", [])),
    }, ensure_ascii=False))
