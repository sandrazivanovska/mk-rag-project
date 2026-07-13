"""
Step 07 — Link Macedonian documents to their English Wikipedia equivalents.

For each MK Wikipedia article in ``mk_documents.jsonl`` we query the Macedonian
Wikipedia API for:
  - the English interlanguage link (``langlinks``, lllang=en)
  - the Wikidata entity id (``pageprops.wikibase_item``)

The result is an alignment file that splits the corpus into two buckets:
  - ``en_status = "linked"``   → has an EN Wikipedia article  (Path A in step 08)
  - ``en_status = "needs_mt"`` → no EN article, translate later (Path B in step 08)

Only Wikipedia-sourced MK documents are linkable; manual / non-wiki docs are
emitted directly as ``needs_mt`` (no title to look up).

Usage:
    python scripts/mk/07_link_mk_to_en.py
    python scripts/mk/07_link_mk_to_en.py --input data/processed/mk_documents.jsonl --max-docs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path("data/processed/mk_documents.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/mk_en_alignment.jsonl")

MK_API_URL = "https://mk.wikipedia.org/w/api.php"
BATCH_SIZE = 50  # MediaWiki allows up to 50 titles per query for non-bot clients
USER_AGENT = "mk-rag-research/0.1 (parallel-corpus-builder)"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ── Pure helpers (unit-tested) ───────────────────────────────────────────────


def chunked(items: Iterable[Any], size: int) -> Iterator[List[Any]]:
    """Yield successive ``size``-length batches from ``items``."""
    batch: List[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_langlinks_response(data: Dict[str, Any]) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Parse a formatversion=2 MK Wikipedia API response into a mapping:

        {mk_title: {"en_title": <str|None>, "wikidata_qid": <str|None>}}
    """
    result: Dict[str, Dict[str, Optional[str]]] = {}

    pages = data.get("query", {}).get("pages", [])
    for page in pages:
        title = page.get("title")
        if not title:
            continue

        en_title: Optional[str] = None
        for langlink in page.get("langlinks", []):
            if langlink.get("lang") == "en":
                en_title = langlink.get("title")
                break

        wikidata_qid = page.get("pageprops", {}).get("wikibase_item")

        result[title] = {"en_title": en_title, "wikidata_qid": wikidata_qid}

    return result


def build_alignment_record(
    mk_doc: Dict[str, Any],
    link_info: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """Build one alignment record from an MK document and its resolved link info."""
    en_title = link_info.get("en_title")
    return {
        "mk_doc_id": mk_doc["doc_id"],
        "mk_title": mk_doc.get("title", ""),
        "en_title": en_title,
        "wikidata_qid": link_info.get("wikidata_qid"),
        "en_status": "linked" if en_title else "needs_mt",
    }


# ── I/O ──────────────────────────────────────────────────────────────────────


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} on line {line_number}") from error
            yield record


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


# ── Network ──────────────────────────────────────────────────────────────────


def fetch_langlinks_batch(titles: List[str], session: Any) -> Dict[str, Dict[str, Optional[str]]]:
    """Query the MK Wikipedia API for a batch of titles. Returns parsed mapping."""
    params = {
        "action": "query",
        "prop": "langlinks|pageprops",
        "lllang": "en",
        "lllimit": "max",
        "ppprop": "wikibase_item",
        "titles": "|".join(titles),
        "redirects": "1",
        "format": "json",
        "formatversion": "2",
    }
    resp = session.get(MK_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return parse_langlinks_response(resp.json())


# ── Main ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Link MK documents to EN Wikipedia via langlinks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-docs", type=int, default=None, help="Limit for debugging.")
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds between API batches.")
    return parser.parse_args()


def iter_alignment_records(
    documents: List[Dict[str, Any]],
    session: Any,
    sleep_seconds: float,
) -> Iterator[Dict[str, Any]]:
    # Wikipedia-sourced docs are linkable by title; everything else is needs_mt.
    wiki_docs = [d for d in documents if d.get("source") == "wikipedia" and d.get("title")]
    other_docs = [d for d in documents if d not in wiki_docs]

    title_to_doc = {d["title"]: d for d in wiki_docs}

    for batch in chunked(list(title_to_doc.keys()), BATCH_SIZE):
        try:
            mapping = fetch_langlinks_batch(batch, session)
        except Exception as error:  # network / API failure → treat batch as needs_mt
            logger.warning("langlinks batch failed (%s); marking %d docs needs_mt", error, len(batch))
            mapping = {}

        for title in batch:
            link_info = mapping.get(title, {"en_title": None, "wikidata_qid": None})
            yield build_alignment_record(title_to_doc[title], link_info)

        if sleep_seconds:
            time.sleep(sleep_seconds)

    for doc in other_docs:
        yield build_alignment_record(doc, {"en_title": None, "wikidata_qid": None})


def main() -> None:
    setup_logging()
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {args.input}. "
            "Run scripts/mk/01-02 first to produce mk_documents.jsonl."
        )

    import requests

    documents = list(iter_jsonl(args.input))
    if args.max_docs is not None:
        documents = documents[: args.max_docs]

    logger.info("Linking %d MK documents to EN Wikipedia", len(documents))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    records = list(iter_alignment_records(documents, session, args.sleep))
    written = write_jsonl(records, args.output)

    linked = sum(1 for r in records if r["en_status"] == "linked")
    needs_mt = written - linked
    logger.info("Alignment written to %s", args.output)
    logger.info("Total: %d | linked (EN Wiki): %d | needs_mt: %d", written, linked, needs_mt)


if __name__ == "__main__":
    main()
