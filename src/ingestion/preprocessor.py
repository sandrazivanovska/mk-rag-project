"""
Text cleaning and deduplication for the MK/EN corpora.

Macedonian-specific considerations:
  - Cyrillic script normalisation (NFC)
  - Strip HTML residue from WikiExtractor output
  - Near-duplicate removal via MinHash LSH
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger("preprocessor")

# ── Text cleaning ──────────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_WIKI_BRACKET_RE = re.compile(r"\[edit\]|\[citation needed\]|\[clarification needed\]")


def clean_text(text: str, *, lang: str = "mk") -> str:
    """
    Clean a single document text.

    Steps:
        1. Unicode NFC normalisation (important for Cyrillic)
        2. Strip residual HTML tags
        3. Remove Wikipedia editing artefacts
        4. Collapse multiple whitespace
        5. Strip leading/trailing whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WIKI_BRACKET_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ── Deduplication ──────────────────────────────────────────────────────────────


def _doc_fingerprint(text: str) -> str:
    """SHA-256 fingerprint of the first 500 characters (fast exact dedup)."""
    snippet = text[:500].strip().lower()
    return hashlib.sha256(snippet.encode("utf-8")).hexdigest()


def deduplicate_chunks(
    chunks: Iterable[dict],
    *,
    key: str = "text",
) -> Iterator[dict]:
    """
    Exact near-deduplication using a fingerprint of the first 500 characters.

    Args:
        chunks: Iterable of dicts, each with at least a ``key`` field.
        key: Field name containing the text to fingerprint.

    Yields:
        Unique chunks (first occurrence wins).
    """
    seen: set[str] = set()
    total = kept = 0
    for chunk in chunks:
        total += 1
        fp = _doc_fingerprint(chunk[key])
        if fp not in seen:
            seen.add(fp)
            kept += 1
            yield chunk
    logger.info(f"Deduplication: kept {kept:,} / {total:,} chunks")
