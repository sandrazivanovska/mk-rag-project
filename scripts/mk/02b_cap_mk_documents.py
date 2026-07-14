"""Cap ``mk_documents.jsonl`` to a fixed-size subset that pins the gold articles.

The full MK Wikipedia corpus (~160k articles) is too large to embed on CPU, so
experiments run on a capped subset. The subset MUST contain every article the
gold QA questions are sourced from (``source_doc_id`` in the QA CSV) — otherwise
the questions have no answers in the corpus — plus deterministic, seeded-random
distractor articles up to the target size.

The full corpus is preserved next to the output as ``mk_documents_full.jsonl``.

Usage:
    python scripts/mk/02b_cap_mk_documents.py --target-size 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_INPUT_PATH = Path("data/processed/mk_documents.jsonl")
DEFAULT_QA_CSV_PATH = Path("data/evaluation/mk_qa_template_300.csv")
DEFAULT_FULL_BACKUP_PATH = Path("data/processed/mk_documents_full.jsonl")

logger = logging.getLogger(__name__)


def load_gold_doc_refs(qa_csv_path: Path) -> Dict[str, str]:
    """Return ``{source_doc_id: source_title}`` for every row in the QA CSV."""

    refs: Dict[str, str] = {}

    with qa_csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            doc_id = (row.get("source_doc_id") or "").strip()
            title = (row.get("source_title") or "").strip()

            if doc_id:
                refs[doc_id] = title

    return refs


def load_documents(input_path: Path) -> List[Dict[str, Any]]:
    documents = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                documents.append(json.loads(line))

    return documents


def select_subset(
    documents: List[Dict[str, Any]],
    gold_refs: Dict[str, str],
    target_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    by_doc_id = {str(doc.get("doc_id", "")): doc for doc in documents}
    by_title = {str(doc.get("title", "")).strip(): doc for doc in documents}

    gold_docs: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []

    for doc_id, title in gold_refs.items():
        doc = by_doc_id.get(doc_id) or by_title.get(title)

        if doc is None:
            missing.append(f"{doc_id} ({title})")
        else:
            gold_docs[str(doc["doc_id"])] = doc

    if missing:
        logger.warning(
            "%d gold articles not found in the corpus: %s",
            len(missing),
            "; ".join(missing[:10]) + ("..." if len(missing) > 10 else ""),
        )

    logger.info("Pinned %d/%d gold articles", len(gold_docs), len(gold_refs))

    distractor_pool = [
        doc for doc in documents if str(doc.get("doc_id", "")) not in gold_docs
    ]
    distractor_count = max(0, target_size - len(gold_docs))

    rng = random.Random(seed)
    distractors = (
        rng.sample(distractor_pool, distractor_count)
        if distractor_count < len(distractor_pool)
        else distractor_pool
    )

    logger.info(
        "Selected %d distractors (seed=%d) from a pool of %d",
        len(distractors),
        seed,
        len(distractor_pool),
    )

    subset = list(gold_docs.values()) + distractors
    subset.sort(key=lambda doc: str(doc.get("doc_id", "")))
    return subset


def write_jsonl(documents: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for doc in documents:
            file.write(json.dumps(doc, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cap MK corpus, pinning gold articles.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--qa-csv", type=Path, default=DEFAULT_QA_CSV_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT_PATH,
                        help="Defaults to overwriting --input (full corpus is backed up first).")
    parser.add_argument("--full-backup", type=Path, default=DEFAULT_FULL_BACKUP_PATH)
    parser.add_argument("--target-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()

    gold_refs = load_gold_doc_refs(args.qa_csv)
    logger.info("Gold QA CSV references %d unique articles", len(gold_refs))

    documents = load_documents(args.input)
    logger.info("Loaded %d documents from %s", len(documents), args.input)

    subset = select_subset(documents, gold_refs, args.target_size, args.seed)

    if args.output == args.input and not args.full_backup.exists():
        args.input.replace(args.full_backup)
        logger.info("Full corpus preserved at %s", args.full_backup)

    write_jsonl(subset, args.output)
    logger.info("Wrote %d documents to %s", len(subset), args.output)


if __name__ == "__main__":
    main()
