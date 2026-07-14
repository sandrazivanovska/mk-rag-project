"""
Step 08 — Build English documents from the MK↔EN alignment.

Two origins (project plan §4.2):

  Path A  en_status="linked"   → fetch the full English Wikipedia article
                                  (action=query&prop=extracts&explaintext).
                                  source = "en_wikipedia"
  Path B  en_status="needs_mt" → machine-translate the MK document text MK→EN
                                  with deep-translator (free, no API key), capped
                                  at --max-mt-docs. source = "mt_from_mk"

Every output row carries alignment keys (mk_doc_id, wikidata_qid) and a source
tag so real-EN vs MT-EN can be analysed separately downstream.

The run is resumable: documents already present in the output file are skipped,
so an interrupted run (throttling, network) can simply be re-run.

At scale, Path A via the API is impractical: Wikimedia's per-IP throttle drops
to ~1 request/minute once triggered (4+ days for 7k docs). Use ``--hf-wikipedia``
to stream a full EN Wikipedia snapshot from the HuggingFace CDN instead (no rate
limits, ~20 min for any number of titles). Titles missing from the snapshot
(renamed/new pages) can be topped up afterwards by re-running without the flag —
resumability skips everything already written.

Usage:
    python scripts/mk/08_build_en_documents.py
    python scripts/mk/08_build_en_documents.py --max-docs 20 --max-mt-docs 5   # smoke run
    python scripts/mk/08_build_en_documents.py --hf-wikipedia 20231101.en --no-mt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_ALIGNMENT_PATH = Path("data/processed/mk_en_alignment.jsonl")
DEFAULT_MK_DOCS_PATH = Path("data/processed/mk_documents.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/en_documents.jsonl")

EN_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "mk-rag-research/0.1 (parallel-corpus-builder)"

# Quality gate for fetched / translated English text.
MIN_TEXT_LENGTH = 200
MIN_LATIN_RATIO = 0.60

# Google translate endpoints (and deep-translator) reject very long inputs.
# The limit applies to the URL-encoded request, and Cyrillic encodes at ~3 bytes
# per character — parts over ~2000 chars of Macedonian text fail with a
# RequestError even though the same length in Latin text would pass.
TRANSLATE_MAX_CHARS = 1500

WHITESPACE_PATTERN = re.compile(r"\s+")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
LETTER_PATTERN = re.compile(r"[A-Za-zА-Яа-яЃѓЌќЅѕЉљЊњЈјЏџШшЧчЖж]")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?…])\s+")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ── Pure helpers (unit-tested) ───────────────────────────────────────────────


def clean_text(text: str) -> str:
    """NFC-normalize and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def latin_ratio(text: str) -> float:
    """Ratio of Latin letters to all letters (Latin + Cyrillic)."""
    letters = LETTER_PATTERN.findall(text)
    if not letters:
        return 0.0
    latin = LATIN_PATTERN.findall(text)
    return len(latin) / len(letters)


def is_good_english_text(
    text: str,
    min_length: int = MIN_TEXT_LENGTH,
    min_latin_ratio: float = MIN_LATIN_RATIO,
) -> bool:
    """A fetched/translated doc is kept only if long enough and mostly Latin script."""
    cleaned = clean_text(text)
    if len(cleaned) < min_length:
        return False
    if latin_ratio(cleaned) < min_latin_ratio:
        return False
    return True


def split_text_for_translation(text: str, max_chars: int = TRANSLATE_MAX_CHARS) -> List[str]:
    """
    Split text into chunks no longer than ``max_chars``, preferring sentence
    boundaries. A single sentence longer than ``max_chars`` is hard-split on
    whitespace so no piece exceeds the limit.
    """
    text = clean_text(text)
    if not text:
        return []

    sentences = [s for s in SENTENCE_SPLIT_PATTERN.split(text) if s]

    parts: List[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            parts.append(current.strip())
        current = ""

    for sentence in sentences:
        # Hard-split a sentence that is itself too long.
        while len(sentence) > max_chars:
            flush()
            head = sentence[:max_chars]
            # Break at the last space within the limit if possible.
            cut = head.rfind(" ")
            if cut <= 0:
                cut = max_chars
            parts.append(sentence[:cut].strip())
            sentence = sentence[cut:].lstrip()

        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            flush()
            current = sentence

    flush()
    return parts


def translate_long_text(
    text: str,
    translate_fn: Callable[[str], str],
    max_chars: int = TRANSLATE_MAX_CHARS,
) -> str:
    """Translate arbitrarily long text by splitting under the API char limit."""
    parts = split_text_for_translation(text, max_chars=max_chars)
    translated = [translate_fn(part) for part in parts]
    return clean_text(" ".join(t for t in translated if t))


def parse_extract_response(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse an EN Wikipedia extracts API (formatversion=2) response into
    {pageid, title, text, url}. Returns None when the page is missing/empty.
    """
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None

    page = pages[0]
    if page.get("missing"):
        return None

    extract = (page.get("extract") or "").strip()
    if not extract:
        return None

    title = page.get("title", "")
    return {
        "pageid": page.get("pageid"),
        "title": title,
        "text": extract,
        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
    }


def build_en_doc_wiki(alignment: Dict[str, Any], extract: Dict[str, Any]) -> Dict[str, Any]:
    """Build an EN document record from a fetched Wikipedia extract."""
    return {
        "doc_id": f"en_wiki_{extract['pageid']}",
        "source": "en_wikipedia",
        "title": extract["title"],
        "language": "en",
        "text": clean_text(extract["text"]),
        "url": extract["url"],
        "wikidata_qid": alignment.get("wikidata_qid"),
        "mk_doc_id": alignment["mk_doc_id"],
    }


def build_en_doc_mt(alignment: Dict[str, Any], translated_text: str) -> Dict[str, Any]:
    """Build an EN document record from machine-translated MK text."""
    mk_doc_id = alignment["mk_doc_id"]
    return {
        "doc_id": f"en_mt_{mk_doc_id}",
        "source": "mt_from_mk",
        "title": alignment.get("mk_title", ""),
        "language": "en",
        "text": clean_text(translated_text),
        "url": None,
        "wikidata_qid": alignment.get("wikidata_qid"),
        "mk_doc_id": mk_doc_id,
    }


# ── I/O ──────────────────────────────────────────────────────────────────────


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_mk_ids(output_path: Path) -> set[str]:
    """Read already-written EN docs (for resumability) keyed by mk_doc_id."""
    done: set[str] = set()
    if output_path.exists():
        for record in iter_jsonl(output_path):
            mk_id = record.get("mk_doc_id")
            if mk_id:
                done.add(mk_id)
    return done


# ── Network ──────────────────────────────────────────────────────────────────


def request_with_retry(
    session: Any,
    url: str,
    params: Dict[str, Any],
    *,
    max_retries: int = 5,
    base_sleep: float = 1.0,
) -> Any:
    """
    GET with retry + exponential backoff, honoring Retry-After on HTTP 429.

    Raises the last error if all retries are exhausted.
    """
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else base_sleep * (2 ** attempt)
                logger.warning("HTTP 429; backing off %.1fs (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except Exception as error:  # noqa: BLE001 - retry on any transient error
            last_error = error
            time.sleep(base_sleep * (2 ** attempt))
    raise last_error if last_error else RuntimeError("request_with_retry exhausted")


def fetch_en_extract(title: str, session: Any) -> Optional[Dict[str, Any]]:
    """Fetch the full plaintext extract for an EN Wikipedia title."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
        "format": "json",
        "formatversion": "2",
    }
    resp = request_with_retry(session, EN_API_URL, params)
    return parse_extract_response(resp.json())


# ── HF snapshot streaming (Path A alternative, no rate limits) ────────────────


def normalize_link_title(title: str) -> str:
    """Normalize a langlinks EN title for matching: drop #fragment, _ → space."""
    return title.split("#", 1)[0].replace("_", " ").strip()


def stream_hf_extracts(
    hf_config: str,
    wanted_titles: Dict[str, List[Dict[str, Any]]],
    log_every: int = 500_000,
) -> Iterator[tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Stream a ``wikimedia/wikipedia`` snapshot and yield ``(alignment, extract)``
    for every row whose title matches a wanted alignment. ``wanted_titles`` maps
    normalized EN title → list of alignment records (deduped MK docs can share
    an EN article).
    """
    from datasets import load_dataset

    dataset = load_dataset("wikimedia/wikipedia", hf_config, split="train", streaming=True)

    seen_rows = 0
    for row in dataset:
        seen_rows += 1
        if seen_rows % log_every == 0:
            logger.info("...streamed %d snapshot rows", seen_rows)

        alignments = wanted_titles.get(row["title"])
        if not alignments:
            continue

        extract = {
            "pageid": row["id"],
            "title": row["title"],
            "text": row["text"],
            "url": row.get("url") or f"https://en.wikipedia.org/wiki/{row['title'].replace(' ', '_')}",
        }
        for alignment in alignments:
            yield alignment, extract


# ── Main ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EN documents (Wikipedia + MT) from alignment.")
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT_PATH)
    parser.add_argument("--mk-docs", type=Path, default=DEFAULT_MK_DOCS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-docs", type=int, default=None, help="Cap linked EN Wikipedia fetches.")
    parser.add_argument("--max-mt-docs", type=int, default=2000, help="Cap MK→EN translations.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between network/MT calls.")
    parser.add_argument("--no-mt", action="store_true", help="Skip the MT path entirely.")
    parser.add_argument("--hf-wikipedia", type=str, default=None, metavar="CONFIG",
                        help="Stream Path A from a wikimedia/wikipedia HF snapshot "
                             "(e.g. '20231101.en') instead of the rate-limited API.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if not args.alignment.exists():
        raise FileNotFoundError(
            f"Alignment file not found: {args.alignment}. Run scripts/mk/07 first."
        )

    import requests

    alignments = list(iter_jsonl(args.alignment))
    mk_text_by_id = {d["doc_id"]: d.get("text", "") for d in iter_jsonl(args.mk_docs)} \
        if args.mk_docs.exists() else {}

    done_mk_ids = load_done_mk_ids(args.output)
    if done_mk_ids:
        logger.info("Resuming: %d EN docs already present, will skip them", len(done_mk_ids))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Lazily construct the translator only if the MT path runs.
    translate_fn: Optional[Callable[[str], str]] = None

    def get_translate_fn() -> Callable[[str], str]:
        nonlocal translate_fn
        if translate_fn is None:
            repo_root = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(repo_root))
            from src.generator.translator import Translator

            translator = Translator(source_lang="mk", target_lang="en")
            translate_fn = translator.translate
        return translate_fn

    linked = [a for a in alignments if a["en_status"] == "linked"]
    needs_mt = [a for a in alignments if a["en_status"] == "needs_mt"]

    if args.max_docs is not None:
        linked = linked[: args.max_docs]
    if args.no_mt:
        needs_mt = []
    else:
        needs_mt = needs_mt[: args.max_mt_docs]

    logger.info("Path A (EN Wikipedia): %d | Path B (MT): %d", len(linked), len(needs_mt))

    written = wiki_count = mt_count = skipped = 0

    with args.output.open("a", encoding="utf-8") as out:

        # ── Path A: EN Wikipedia ──────────────────────────────────────────────
        if args.hf_wikipedia:
            # Stream the snapshot once; write every wanted title as it flies by.
            wanted_titles: Dict[str, List[Dict[str, Any]]] = {}
            for alignment in linked:
                if alignment["mk_doc_id"] in done_mk_ids:
                    continue
                title = normalize_link_title(alignment.get("en_title") or "")
                if title:
                    wanted_titles.setdefault(title, []).append(alignment)

            logger.info("Streaming HF snapshot %s for %d wanted titles",
                        args.hf_wikipedia, len(wanted_titles))

            for alignment, extract in stream_hf_extracts(args.hf_wikipedia, wanted_titles):
                if not is_good_english_text(extract["text"]):
                    skipped += 1
                    continue
                doc = build_en_doc_wiki(alignment, extract)
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                written += 1
                wiki_count += 1
                if wiki_count % 500 == 0:
                    logger.info("...%d EN wiki docs written", wiki_count)
        else:
            for alignment in linked:
                if alignment["mk_doc_id"] in done_mk_ids:
                    continue
                try:
                    extract = fetch_en_extract(alignment["en_title"], session)
                except Exception as error:
                    logger.warning("EN fetch failed for '%s': %s", alignment["en_title"], error)
                    extract = None

                if not extract or not is_good_english_text(extract["text"]):
                    skipped += 1
                else:
                    doc = build_en_doc_wiki(alignment, extract)
                    out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    written += 1
                    wiki_count += 1

                if args.sleep:
                    time.sleep(args.sleep)

        # ── Path B: machine translation ───────────────────────────────────────
        for alignment in needs_mt:
            mk_id = alignment["mk_doc_id"]
            if mk_id in done_mk_ids:
                continue
            mk_text = mk_text_by_id.get(mk_id, "")
            if not mk_text.strip():
                skipped += 1
                continue
            try:
                translated = translate_long_text(mk_text, get_translate_fn())
            except Exception as error:
                logger.warning("MT failed for %s: %s", mk_id, error)
                skipped += 1
                continue

            if not is_good_english_text(translated):
                skipped += 1
            else:
                doc = build_en_doc_mt(alignment, translated)
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                written += 1
                mt_count += 1

            if args.sleep:
                time.sleep(args.sleep)

    logger.info("EN documents written this run: %d (wiki=%d, mt=%d, skipped=%d)",
                written, wiki_count, mt_count, skipped)
    logger.info("Output → %s", args.output)


if __name__ == "__main__":
    main()
