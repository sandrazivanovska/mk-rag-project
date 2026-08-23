"""
Step 12 — Cap the full Wikipedia chunk file to a fixed number of ARTICLES.

We already have the full `mk_chunks.jsonl` (~201k Wikipedia chunks). Embedding all
of it is slow and it dwarfs the ~51k-chunk English side. This selects a subset of
articles — pinning every gold-QA source article so all questions stay answerable —
plus seeded-random distractor articles up to --target-articles, then keeps only the
chunks belonging to those articles. Deterministic (seed) for reproducibility.

Usage:
    python scripts/mk/12_cap_mk_chunks.py \
        --input ~/Downloads/mk_rag_artifacts/mk_chunks.jsonl \
        --gold data/evaluation/gold_dataset.jsonl \
        --output data/processed/mk_chunks_10k.jsonl \
        --target-articles 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Set

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_gold_doc_ids(gold_path: Path) -> Set[str]:
    ids: Set[str] = set()
    with gold_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            did = str(rec.get("source_doc_id", "")).strip()
            if did:
                ids.add(did)
    return ids


def collect_doc_ids(chunks_path: Path) -> Set[str]:
    ids: Set[str] = set()
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(str(json.loads(line).get("doc_id", "")))
    ids.discard("")
    return ids


def select_articles(all_ids: Set[str], gold_ids: Set[str], target: int, seed: int) -> Set[str]:
    present_gold = gold_ids & all_ids
    missing = gold_ids - all_ids
    if missing:
        logger.warning("%d gold articles not in corpus (unanswerable): %s",
                       len(missing), sorted(missing))
    selected = set(present_gold)
    pool = sorted(all_ids - selected)
    need = max(0, target - len(selected))
    rng = random.Random(seed)
    if need < len(pool):
        selected.update(rng.sample(pool, need))
    else:
        selected.update(pool)
    logger.info("Pinned %d gold + %d distractors = %d articles",
                len(present_gold), len(selected) - len(present_gold), len(selected))
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cap Wikipedia chunks to N articles (gold pinned).")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--gold", type=Path, default=Path("data/evaluation/gold_dataset.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/processed/mk_chunks_10k.jsonl"))
    p.add_argument("--target-articles", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    gold_ids = load_gold_doc_ids(args.gold)
    logger.info("Gold references %d unique articles", len(gold_ids))

    logger.info("Scanning corpus for article ids: %s", args.input)
    all_ids = collect_doc_ids(args.input)
    logger.info("Corpus has %d unique articles", len(all_ids))

    keep = select_articles(all_ids, gold_ids, args.target_articles, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = total = 0
    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            if str(json.loads(line).get("doc_id", "")) in keep:
                fout.write(line + "\n")
                kept += 1

    logger.info("Kept %d / %d chunks (%d articles) → %s", kept, total, len(keep), args.output)


if __name__ == "__main__":
    main()
