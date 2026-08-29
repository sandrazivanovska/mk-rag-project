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

Two link sources are supported:
  - **API mode** (default): batched queries against the MK Wikipedia API.
    Fine for small corpora, but per-IP rate limits make it slow at scale.
  - **Offline mode** (``--langlinks-sql`` + ``--pageprops-sql``): parse the
    ``langlinks`` and ``page_props`` SQL table dumps published alongside the
    Wikipedia dump. No network, no rate limits — use this for 10k+ docs.
    The tables are keyed by page id, which our ``doc_id``s already embed
    (``mk_wiki_<page_id>``), so no title matching is involved.

Usage:
    python scripts/mk/07_link_mk_to_en.py
    python scripts/mk/07_link_mk_to_en.py --input data/processed/mk_documents.jsonl --max-docs 100
    python scripts/mk/07_link_mk_to_en.py \
        --langlinks-sql data/raw/mk_wikipedia/mkwiki-latest-langlinks.sql.gz \
        --pageprops-sql data/raw/mk_wikipedia/mkwiki-latest-page_props.sql.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
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


# ── Offline mode: parse SQL table dumps ──────────────────────────────────────

# Matches (page_id, 'propname', 'value' tuples inside INSERT statements, with
# MySQL backslash escaping inside the quoted strings.
SQL_PAIR_PATTERN = re.compile(r"\((\d+),'((?:[^'\\]|\\.)*)','((?:[^'\\]|\\.)*)'")

MYSQL_UNESCAPE = {
    "\\'": "'",
    '\\"': '"',
    "\\\\": "\\",
    "\\n": "\n",
    "\\t": "\t",
    "\\r": "\r",
    "\\0": "\0",
}


def unescape_mysql(value: str) -> str:
    return re.sub(
        r"\\.",
        lambda match: MYSQL_UNESCAPE.get(match.group(0), match.group(0)[1]),
        value,
    )


def load_sql_pairs(path: Path, key_filter: str) -> Dict[str, str]:
    """
    Stream a ``.sql.gz`` table dump and return ``{page_id: value}`` for tuples
    whose middle column equals ``key_filter``.

    Works for both tables we need:
      - ``langlinks``  (ll_from, ll_lang, ll_title)      with key_filter="en"
      - ``page_props`` (pp_page, pp_propname, pp_value)  with key_filter="wikibase_item"
    """
    result: Dict[str, str] = {}

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not line.startswith("INSERT INTO"):
                continue
            for page_id, middle, value in SQL_PAIR_PATTERN.findall(line):
                if middle == key_filter:
                    result[page_id] = unescape_mysql(value)

    return result


def iter_alignment_records_offline(
    documents: List[Dict[str, Any]],
    langlinks_sql: Path,
    pageprops_sql: Path,
) -> Iterator[Dict[str, Any]]:
    logger.info("Parsing langlinks dump: %s", langlinks_sql)
    en_titles = load_sql_pairs(langlinks_sql, "en")
    logger.info("Found %d MK pages with an EN interlanguage link", len(en_titles))

    logger.info("Parsing page_props dump: %s", pageprops_sql)
    qids = load_sql_pairs(pageprops_sql, "wikibase_item")
    logger.info("Found %d MK pages with a Wikidata QID", len(qids))

    for doc in documents:
        page_id = str(doc.get("metadata", {}).get("original_id", "")).strip()
        if not page_id and str(doc.get("doc_id", "")).startswith("mk_wiki_"):
            page_id = str(doc["doc_id"])[len("mk_wiki_"):]

        link_info = {
            "en_title": en_titles.get(page_id),
            "wikidata_qid": qids.get(page_id),
        }
        yield build_alignment_record(doc, link_info)


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


def request_with_retry(
    session: Any,
    url: str,
    params: Dict[str, Any],
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
    resp = request_with_retry(session, MK_API_URL, params)
    return parse_langlinks_response(resp.json())


# ── Main ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Link MK documents to EN Wikipedia via langlinks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-docs", type=int, default=None, help="Limit for debugging.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between API batches.")
    parser.add_argument("--langlinks-sql", type=Path, default=None,
                        help="Path to <wiki>-latest-langlinks.sql.gz for offline linking.")
    parser.add_argument("--pageprops-sql", type=Path, default=None,
                        help="Path to <wiki>-latest-page_props.sql.gz for offline linking.")
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

    documents = list(iter_jsonl(args.input))
    if args.max_docs is not None:
        documents = documents[: args.max_docs]

    logger.info("Linking %d MK documents to EN Wikipedia", len(documents))

    if bool(args.langlinks_sql) != bool(args.pageprops_sql):
        raise SystemExit("--langlinks-sql and --pageprops-sql must be given together.")

    if args.langlinks_sql:
        records = list(
            iter_alignment_records_offline(documents, args.langlinks_sql, args.pageprops_sql)
        )
    else:
        import requests

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
