from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_PATH = Path("data/processed/mk_chunks.jsonl")
DEFAULT_OUTPUT_CSV = Path("data/evaluation/mk_qa_template.csv")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path} on line {line_number}"
                )

            yield record


def load_chunks(path: Path, min_tokens: int) -> List[Dict[str, Any]]:
    chunks = []

    for chunk in iter_jsonl(path):
        token_count = int(chunk.get("token_count", 0))

        if token_count < min_tokens:
            continue

        chunks.append(chunk)

    return chunks


def sample_chunks(
    chunks: List[Dict[str, Any]],
    number_of_rows: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    if number_of_rows >= len(chunks):
        return chunks

    return rng.sample(chunks, number_of_rows)


def get_domain(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})

    if isinstance(metadata, dict):
        domain = metadata.get("domain")

        if domain:
            return str(domain)

    return "general"


def write_qa_template(
    chunks: List[Dict[str, Any]],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question_id",
        "question_mk",
        "gold_answer_mk",
        "gold_chunk_ids",
        "source_doc_id",
        "source_title",
        "domain",
        "chunk_text",
        "valid",
        "notes",
    ]

    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, chunk in enumerate(chunks, start=1):
            writer.writerow(
                {
                    "question_id": f"q_{index:04d}",
                    "question_mk": "",
                    "gold_answer_mk": "",
                    "gold_chunk_ids": chunk["chunk_id"],
                    "source_doc_id": chunk["doc_id"],
                    "source_title": chunk["title"],
                    "domain": get_domain(chunk),
                    "chunk_text": chunk["text"],
                    "valid": "",
                    "notes": "",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a CSV annotation template for Macedonian QA examples."
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Input chunks JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV annotation template.",
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=300,
        help="Number of chunks to sample for QA annotation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )

    parser.add_argument(
        "--min-tokens",
        type=int,
        default=120,
        help="Only sample chunks with at least this many tokens.",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if not args.chunks.exists():
        raise FileNotFoundError(f"Chunks file does not exist: {args.chunks}")

    logger.info("Loading chunks from: %s", args.chunks)

    chunks = load_chunks(
        path=args.chunks,
        min_tokens=args.min_tokens,
    )

    if not chunks:
        raise RuntimeError("No chunks available for QA template creation.")

    logger.info("Eligible chunks: %s", len(chunks))

    selected_chunks = sample_chunks(
        chunks=chunks,
        number_of_rows=args.rows,
        seed=args.seed,
    )

    write_qa_template(
        chunks=selected_chunks,
        output_csv=args.output,
    )

    logger.info("QA template rows written: %s", len(selected_chunks))
    logger.info("Output: %s", args.output)


if __name__ == "__main__":
    main()