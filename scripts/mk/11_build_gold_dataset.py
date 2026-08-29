"""
Step 11 — Convert the generated QA dataset into the evaluation-harness schema.

Dimitar's `mk_qa_dataset_generated.jsonl` uses:
    {question_id, question_mk, gold_answer_mk, gold_chunk_ids, source_doc_id, domain}
with `gold_chunk_ids` stored as a *stringified* Python list (e.g. "['mk_wiki_1_chunk_0']").

`main.py` / the evaluator expect:
    {query, answer, relevant_doc_ids, ...}

This script renames the fields and parses `gold_chunk_ids` into a real list.
Pure logic, no LLM, no cost.

Usage:
    python scripts/mk/11_build_gold_dataset.py
    python scripts/mk/11_build_gold_dataset.py --input <in.jsonl> --output <out.jsonl>
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/evaluation/mk_qa_dataset_generated.jsonl")
DEFAULT_OUTPUT = Path("data/evaluation/gold_dataset.jsonl")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_chunk_ids(raw: Any) -> List[str]:
    """
    Normalize gold_chunk_ids into a list of strings. Handles:
      - a real JSON list                     ["a", "b"]
      - a stringified Python/JSON list       "['a', 'b']"
      - a plain / semicolon-separated string "a" or "a;b"
    """
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]

    text = str(raw or "").strip()
    if not text:
        return []

    # Try JSON, then Python-literal (handles single quotes).
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(text)
            if isinstance(value, (list, tuple)):
                return [str(x).strip() for x in value if str(x).strip()]
        except (ValueError, SyntaxError):
            pass

    # Fallback: split on ';' or ','.
    for sep in (";", ","):
        if sep in text:
            return [p.strip() for p in text.split(sep) if p.strip()]
    return [text]


def convert_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Map one generated-QA record into the harness schema."""
    return {
        "query": str(record.get("question_mk", "")).strip(),
        "answer": str(record.get("gold_answer_mk", "")).strip(),
        "relevant_doc_ids": parse_chunk_ids(record.get("gold_chunk_ids")),
        # carried through for traceability
        "question_id": record.get("question_id"),
        "source_doc_id": record.get("source_doc_id"),
        "domain": record.get("domain", "general"),
    }


def is_complete(converted: Dict[str, Any]) -> bool:
    return bool(converted["query"] and converted["answer"] and converted["relevant_doc_ids"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert generated QA to harness gold schema.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    total = kept = skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            converted = convert_record(json.loads(line))
            if not is_complete(converted):
                skipped += 1
                continue
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            kept += 1

    logger.info("Converted %d/%d rows → %s (skipped %d incomplete)", kept, total, args.output, skipped)


if __name__ == "__main__":
    main()
