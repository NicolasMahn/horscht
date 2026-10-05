"""Sample generator-only seed words from the attributed wordfreq pool."""

from __future__ import annotations

import gzip
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import TypedDict


class SeedWord(TypedDict):
    word: str
    source_language: str
    source_version: str
    pool_id: str


_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "generated_train"
_POOL_PATH = _DATA_ROOT / "seed-word-pool.jsonl.gz"
_MANIFEST_PATH = _DATA_ROOT / "seed-word-pool-manifest.json"


@lru_cache(maxsize=1)
def _load_pool() -> tuple[tuple[str, str], ...]:
    with gzip.open(_POOL_PATH, "rt", encoding="utf-8") as stream:
        rows = tuple(
            (entry["word"], entry["source_language"])
            for entry in map(json.loads, stream)
        )
    if not rows:
        raise RuntimeError(f"Seed-word pool is empty: {_POOL_PATH}")
    return rows


@lru_cache(maxsize=1)
def _load_pool_metadata() -> tuple[str, str]:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest["source"]["source_version"], manifest["pool_id"]


def sample_seed_words(rng: random.Random) -> list[SeedWord]:
    """Return 3–8 unique pool entries with metadata for a generation sidecar.

    The function has no output-language input: all source-language lists share
    one pool, so seed selection cannot follow the language of a generated item.
    """
    pool = _load_pool()
    count = rng.randint(3, 8)
    source_version, pool_id = _load_pool_metadata()
    return [
        {
            "word": word,
            "source_language": source_language,
            "source_version": source_version,
            "pool_id": pool_id,
        }
        for word, source_language in rng.sample(pool, count)
    ]
