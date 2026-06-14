from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set

import pandas as pd


logger = logging.getLogger(__name__)

DEFAULT_INPUT_CSV = Path("data/evaluation/mk_qa_template.csv")
DEFAULT_CHUNKS_PATH = Path("data/processed/mk_chunks.jsonl")
DEFAULT_OUTPUT_JSONL = Path("data/evaluation/mk_qa_dataset.jsonl")


REQUIRED_COLUMNS = [
    "question_id",
    "question_mk",
    "gold_answer_mk",
    "gold_chunk_ids",
    "source_doc_id",
    "domain",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def normalize_cell(value: Any) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


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


def load_valid_chunk_ids(chunks_path: Path) -> Set[str]:
    chunk_ids = set()

    for chunk in iter_jsonl(chunks_path):
        chunk_id = chunk.get("chunk_id")

        if isinstance(chunk_id, str) and chunk_id.strip():
            chunk_ids.add(chunk_id.strip())

    return chunk_ids


def validate_columns(df: pd.DataFrame) -> None:
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def parse_gold_chunk_ids(raw_value: str) -> List[str]:
    return [
        chunk_id.strip()
        for chunk_id in raw_value.split(";")
        if chunk_id.strip()
    ]


def is_row_marked_invalid(row: pd.Series) -> bool:
    if "valid" not in row:
        return False

    value = normalize_cell(row["valid"]).lower()

    return value in {"no", "false", "0", "invalid", "не", "невалидно"}


def build_qa_record(
    row: pd.Series,
    valid_chunk_ids: Set[str],
) -> Dict[str, Any] | None:
    if is_row_marked_invalid(row):
        return None

    question_id = normalize_cell(row["question_id"])
    question_mk = normalize_cell(row["question_mk"])
    gold_answer_mk = normalize_cell(row["gold_answer_mk"])
    raw_gold_chunk_ids = normalize_cell(row["gold_chunk_ids"])
    source_doc_id = normalize_cell(row["source_doc_id"])
    domain = normalize_cell(row["domain"]) or "general"

    if not question_id or not question_mk or not gold_answer_mk or not raw_gold_chunk_ids:
        return None

    gold_chunk_ids = parse_gold_chunk_ids(raw_gold_chunk_ids)

    if not gold_chunk_ids:
        return None

    unknown_chunk_ids = [
        chunk_id for chunk_id in gold_chunk_ids if chunk_id not in valid_chunk_ids
    ]

    if unknown_chunk_ids:
        raise ValueError(
            f"Question {question_id} references unknown chunk IDs: "
            f"{unknown_chunk_ids}"
        )

    return {
        "question_id": question_id,
        "question_mk": question_mk,
        "gold_answer_mk": gold_answer_mk,
        "gold_chunk_ids": gold_chunk_ids,
        "source_doc_id": source_doc_id,
        "domain": domain,
    }


def write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert filled Macedonian QA CSV template into JSONL dataset."
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Filled QA template CSV.",
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Chunks JSONL file used to validate gold_chunk_ids.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Output QA dataset JSONL file.",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {args.input_csv}")

    if not args.chunks.exists():
        raise FileNotFoundError(f"Chunks file does not exist: {args.chunks}")

    logger.info("Reading QA template: %s", args.input_csv)
    df = pd.read_csv(args.input_csv)

    validate_columns(df)

    logger.info("Loading valid chunk IDs from: %s", args.chunks)
    valid_chunk_ids = load_valid_chunk_ids(args.chunks)

    if not valid_chunk_ids:
        raise RuntimeError("No valid chunk IDs found.")

    records: List[Dict[str, Any]] = []
    skipped_rows = 0

    for _, row in df.iterrows():
        record = build_qa_record(
            row=row,
            valid_chunk_ids=valid_chunk_ids,
        )

        if record is None:
            skipped_rows += 1
            continue

        records.append(record)

    if not records:
        raise RuntimeError("No QA records were created. Did you fill the CSV?")

    write_jsonl(records, args.output)

    logger.info("QA examples written: %s", len(records))
    logger.info("Rows skipped: %s", skipped_rows)
    logger.info("Output: %s", args.output)


if __name__ == "__main__":
    main()