"""
Step 08b — Fetch EN Wikipedia articles from the multistream dump (Path A at scale).

Alternative to step 08's Path A for large corpora. The MediaWiki API throttles
per-IP to ~1 request/minute once triggered, and anonymous access to the HF
snapshot mirrors is blocked — but Wikipedia's *multistream* dump is served as a
static file with HTTP Range support. Its companion index maps every article to
the byte offset of the ~100-page bz2 block that contains it, so we can download
only the blocks we need (~150 KB each) instead of the 26 GB dump.

Pipeline:
  1. Parse the index (offset:pageid:title) for the linked EN titles in the
     alignment file.
  2. Group titles by block offset; Range-fetch each block; decompress bz2;
     parse the <page> XML inside.
  3. Convert wikitext → plaintext with mwparserfromhell, apply step 08's
     quality gate, and append records in step 08's exact output format
     (resumable the same way — already-written mk_doc_ids are skipped).
  4. Pages that turn out to be redirects are resolved with one extra pass.

Usage:
    python scripts/mk/08b_fetch_en_multistream.py \
        --index data/raw/enwiki-multistream-index.txt.bz2
    # then run step 08 (optionally with --no-mt) to top up any stragglers / MT.
"""

from __future__ import annotations

import argparse
import bz2
import importlib.util
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent

DEFAULT_ALIGNMENT_PATH = Path("data/processed/mk_en_alignment.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/en_documents.jsonl")
DEFAULT_INDEX_PATH = Path("data/raw/enwiki-multistream-index.txt.bz2")
DEFAULT_DUMP_URL = (
    "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2"
)
USER_AGENT = "mk-rag-research/0.1 (parallel-corpus-builder)"

REDIRECT_PATTERN = re.compile(r"#REDIRECT\s*\[\[([^\]|#]+)", re.IGNORECASE)

# Wikitext block elements that should not survive into plaintext.
HEADING_PATTERN = re.compile(r"^=+\s*.*?\s*=+\s*$", re.MULTILINE)


def _load_step08():
    """Load step 08 by path so we reuse its cleaning/quality/record helpers."""
    path = SCRIPTS_DIR / "08_build_en_documents.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


step08 = _load_step08()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ── Wikitext → plaintext ─────────────────────────────────────────────────────


def wikitext_to_text(wikitext: str) -> str:
    """Convert raw wikitext to plaintext via mwparserfromhell."""
    import mwparserfromhell

    parsed = mwparserfromhell.parse(wikitext)
    text = parsed.strip_code(normalize=True, collapse=True)
    text = HEADING_PATTERN.sub(" ", text)
    return text


# ── Index parsing ────────────────────────────────────────────────────────────


def parse_index(
    index_path: Path,
    wanted_titles: set[str],
) -> Tuple[Dict[str, Tuple[int, int]], List[int]]:
    """
    Stream the multistream index and return:
      - {title: (block_offset, pageid)} for every wanted title
      - the sorted list of ALL block offsets (needed to compute block ends)
    """
    found: Dict[str, Tuple[int, int]] = {}
    offsets: set[int] = set()

    with bz2.open(index_path, "rt", encoding="utf-8", errors="replace") as file:
        for line in file:
            offset_str, _, rest = line.partition(":")
            pageid_str, _, title = rest.partition(":")
            title = title.rstrip("\n")
            try:
                offset = int(offset_str)
            except ValueError:
                continue
            offsets.add(offset)
            if title in wanted_titles and title not in found:
                found[title] = (offset, int(pageid_str))

    return found, sorted(offsets)


def block_ranges(
    found: Dict[str, Tuple[int, int]],
    all_offsets: List[int],
    dump_size: int,
) -> Dict[int, int]:
    """Map each needed block offset → its end offset (start of the next block)."""
    import bisect

    ranges: Dict[int, int] = {}
    for offset, _pageid in found.values():
        if offset in ranges:
            continue
        i = bisect.bisect_right(all_offsets, offset)
        end = all_offsets[i] if i < len(all_offsets) else dump_size
        ranges[offset] = end
    return ranges


# ── Block fetching / parsing ─────────────────────────────────────────────────


def fetch_block(session: Any, dump_url: str, start: int, end: int,
                max_retries: int = 5, base_sleep: float = 1.0) -> bytes:
    """Range-fetch one bz2 block [start, end) with retry."""
    headers = {"Range": f"bytes={start}-{end - 1}"}
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = session.get(dump_url, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.content
        except Exception as error:  # noqa: BLE001 - retry on any transient error
            last_error = error
            time.sleep(base_sleep * (2 ** attempt))
    raise last_error if last_error else RuntimeError("fetch_block exhausted")


def parse_block_pages(block: bytes) -> Iterator[Dict[str, Any]]:
    """Yield {pageid, title, wikitext} for every <page> in a decompressed block."""
    xml = bz2.decompress(block).decode("utf-8", errors="replace")
    root = ET.fromstring(f"<root>{xml}</root>")
    for page in root.iter("page"):
        ns = page.findtext("ns")
        if ns != "0":
            continue
        title = page.findtext("title") or ""
        pageid = page.findtext("id") or ""
        text = page.findtext("revision/text") or ""
        yield {"pageid": pageid, "title": title, "wikitext": text}


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch EN Wikipedia articles from the multistream dump."
    )
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--dump-url", type=str, default=DEFAULT_DUMP_URL)
    parser.add_argument("--max-docs", type=int, default=None, help="Limit for debugging.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Seconds between block fetches.")
    return parser.parse_args()


def run_fetch_round(
    wanted: Dict[str, List[Dict[str, Any]]],
    args: argparse.Namespace,
    session: Any,
    dump_size: int,
    out_file: Any,
) -> Tuple[int, int, Dict[str, List[Dict[str, Any]]]]:
    """
    One round: parse index for wanted titles, fetch their blocks, write docs.
    Returns (written, skipped, redirects) where redirects maps the redirect
    *target* title → the original alignment records.
    """
    logger.info("Parsing index for %d titles: %s", len(wanted), args.index)
    found, all_offsets = parse_index(args.index, set(wanted))
    logger.info("Index: found %d/%d titles across %d total blocks",
                len(found), len(wanted), len(all_offsets))

    ranges = block_ranges(found, all_offsets, dump_size)

    # Group wanted titles by their block offset so each block is fetched once.
    titles_by_offset: Dict[int, List[str]] = {}
    for title, (offset, _pageid) in found.items():
        titles_by_offset.setdefault(offset, []).append(title)

    written = skipped = 0
    redirects: Dict[str, List[Dict[str, Any]]] = {}

    for block_number, (offset, titles) in enumerate(sorted(titles_by_offset.items()), start=1):
        try:
            block = fetch_block(session, args.dump_url, offset, ranges[offset])
            pages_by_title = {p["title"]: p for p in parse_block_pages(block)}
        except Exception as error:
            logger.warning("Block at offset %d failed: %s (skipping %d titles)",
                           offset, error, len(titles))
            skipped += len(titles)
            continue

        for title in titles:
            page = pages_by_title.get(title)
            if page is None:
                skipped += 1
                continue

            redirect_match = REDIRECT_PATTERN.match(page["wikitext"].strip())
            if redirect_match:
                target = redirect_match.group(1).strip()
                redirects.setdefault(target, []).extend(wanted[title])
                continue

            text = wikitext_to_text(page["wikitext"])
            if not step08.is_good_english_text(text):
                skipped += 1
                continue

            extract = {
                "pageid": page["pageid"],
                "title": page["title"],
                "text": text,
                "url": f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}",
            }
            for alignment in wanted[title]:
                doc = step08.build_en_doc_wiki(alignment, extract)
                out_file.write(json.dumps(doc, ensure_ascii=False) + "\n")
                written += 1

        if block_number % 200 == 0:
            logger.info("...%d/%d blocks fetched, %d docs written",
                        block_number, len(titles_by_offset), written)
        if args.sleep:
            time.sleep(args.sleep)

    return written, skipped, redirects


def main() -> None:
    setup_logging()
    args = parse_args()

    if not args.index.exists():
        raise FileNotFoundError(
            f"Index not found: {args.index}. Download "
            "enwiki-latest-pages-articles-multistream-index.txt.bz2 first."
        )

    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    dump_size = int(
        session.head(args.dump_url, timeout=30).headers["Content-Length"]
    )

    alignments = list(step08.iter_jsonl(args.alignment))
    done_mk_ids = step08.load_done_mk_ids(args.output)
    if done_mk_ids:
        logger.info("Resuming: %d EN docs already present, will skip them", len(done_mk_ids))

    wanted: Dict[str, List[Dict[str, Any]]] = {}
    count = 0
    for alignment in alignments:
        if alignment["en_status"] != "linked" or alignment["mk_doc_id"] in done_mk_ids:
            continue
        if args.max_docs is not None and count >= args.max_docs:
            break
        title = step08.normalize_link_title(alignment.get("en_title") or "")
        if title:
            wanted.setdefault(title, []).append(alignment)
            count += 1

    logger.info("Fetching %d EN titles from the multistream dump", count)

    total_written = total_skipped = 0
    with args.output.open("a", encoding="utf-8") as out_file:
        written, skipped, redirects = run_fetch_round(wanted, args, session, dump_size, out_file)
        total_written += written
        total_skipped += skipped

        if redirects:
            logger.info("Resolving %d redirect targets", len(redirects))
            written, skipped, leftover = run_fetch_round(
                redirects, args, session, dump_size, out_file
            )
            total_written += written
            total_skipped += skipped + sum(len(v) for v in leftover.values())

    logger.info("EN documents written this run: %d (skipped/missing: %d)",
                total_written, total_skipped)
    logger.info("Output → %s", args.output)


if __name__ == "__main__":
    main()
