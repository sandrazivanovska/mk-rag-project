from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List


logger = logging.getLogger(__name__)

DEFAULT_DOCUMENTS_PATH = Path("data/processed/mk_documents.jsonl")
DEFAULT_CHUNKS_PATH = Path("data/processed/mk_chunks.jsonl")
DEFAULT_QA_PATH = Path("data/evaluation/mk_qa_dataset.jsonl")


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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def require_fields(
    record: Dict[str, Any],
    required_fields: List[str],
    record_name: str,
    errors: List[str],
) -> None:
    for field in required_fields:
        if field not in record:
            errors.append(f"{record_name}: missing field '{field}'")
        elif record[field] is None or str(record[field]).strip() == "":
            errors.append(f"{record_name}: empty field '{field}'")


def validate_documents(documents: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []

    seen_doc_ids = set()

    for document in documents:
        doc_id = str(document.get("doc_id", ""))

        require_fields(
            record=document,
            required_fields=["doc_id", "source", "title", "language", "text"],
            record_name=f"document {doc_id or '<unknown>'}",
            errors=errors,
        )

        if doc_id in seen_doc_ids:
            errors.append(f"Duplicate document doc_id: {doc_id}")

        seen_doc_ids.add(doc_id)

        if document.get("language") != "mk":
            errors.append(f"Document {doc_id} has non-Macedonian language value")

    return errors


def validate_chunks(
    chunks: List[Dict[str, Any]],
    valid_doc_ids: set[str],
) -> List[str]:
    errors: List[str] = []

    seen_chunk_ids = set()

    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        doc_id = str(chunk.get("doc_id", ""))

        require_fields(
            record=chunk,
            required_fields=[
                "chunk_id",
                "doc_id",
                "title",
                "language",
                "source",
                "text",
                "chunk_index",
                "token_count",
            ],
            record_name=f"chunk {chunk_id or '<unknown>'}",
            errors=errors,
        )

        if chunk_id in seen_chunk_ids:
            errors.append(f"Duplicate chunk_id: {chunk_id}")

        seen_chunk_ids.add(chunk_id)

        if doc_id not in valid_doc_ids:
            errors.append(f"Chunk {chunk_id} references unknown doc_id: {doc_id}")

        if chunk.get("language") != "mk":
            errors.append(f"Chunk {chunk_id} has non-Macedonian language value")

        try:
            token_count = int(chunk.get("token_count", 0))

            if token_count <= 0:
                errors.append(f"Chunk {chunk_id} has invalid token_count")

        except ValueError:
            errors.append(f"Chunk {chunk_id} has non-integer token_count")

    return errors


def validate_qa_examples(
    qa_examples: List[Dict[str, Any]],
    valid_doc_ids: set[str],
    valid_chunk_ids: set[str],
) -> List[str]:
    errors: List[str] = []

    seen_question_ids = set()

    for qa in qa_examples:
        question_id = str(qa.get("question_id", ""))

        require_fields(
            record=qa,
            required_fields=[
                "question_id",
                "question_mk",
                "gold_answer_mk",
                "gold_chunk_ids",
                "source_doc_id",
                "domain",
            ],
            record_name=f"QA example {question_id or '<unknown>'}",
            errors=errors,
        )

        if question_id in seen_question_ids:
            errors.append(f"Duplicate question_id: {question_id}")

        seen_question_ids.add(question_id)

        source_doc_id = str(qa.get("source_doc_id", ""))

        if source_doc_id and source_doc_id not in valid_doc_ids:
            errors.append(
                f"QA example {question_id} references unknown source_doc_id: "
                f"{source_doc_id}"
            )

        gold_chunk_ids = qa.get("gold_chunk_ids")

        if not isinstance(gold_chunk_ids, list) or not gold_chunk_ids:
            errors.append(f"QA example {question_id} has invalid gold_chunk_ids")
            continue

        for chunk_id in gold_chunk_ids:
            if chunk_id not in valid_chunk_ids:
                errors.append(
                    f"QA example {question_id} references unknown chunk_id: "
                    f"{chunk_id}"
                )

    return errors


def print_summary(
    documents: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    qa_examples: List[Dict[str, Any]],
) -> None:
    source_counts = Counter(str(doc.get("source", "unknown")) for doc in documents)
    domain_counts = Counter(str(qa.get("domain", "unknown")) for qa in qa_examples)

    logger.info("Validation summary")
    logger.info("------------------")
    logger.info("Documents: %s", len(documents))
    logger.info("Chunks: %s", len(chunks))
    logger.info("QA examples: %s", len(qa_examples))
    logger.info("Document sources: %s", dict(source_counts))
    logger.info("QA domains: %s", dict(domain_counts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Macedonian RAG data package."
    )

    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
        help="Documents JSONL file.",
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Chunks JSONL file.",
    )

    parser.add_argument(
        "--qa",
        type=Path,
        default=DEFAULT_QA_PATH,
        help="QA dataset JSONL file.",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    for path in [args.documents, args.chunks, args.qa]:
        if not path.exists():
            raise FileNotFoundError(f"Required file does not exist: {path}")

    documents = load_jsonl(args.documents)
    chunks = load_jsonl(args.chunks)
    qa_examples = load_jsonl(args.qa)

    valid_doc_ids = {str(document["doc_id"]) for document in documents}
    valid_chunk_ids = {str(chunk["chunk_id"]) for chunk in chunks}

    errors: List[str] = []

    errors.extend(validate_documents(documents))
    errors.extend(validate_chunks(chunks, valid_doc_ids))
    errors.extend(validate_qa_examples(qa_examples, valid_doc_ids, valid_chunk_ids))

    print_summary(documents, chunks, qa_examples)

    if errors:
        logger.error("Validation failed with %s error(s).", len(errors))

        for error in errors[:50]:
            logger.error("- %s", error)

        if len(errors) > 50:
            logger.error("Only showing first 50 errors.")

        raise SystemExit(1)

    logger.info("Validation passed.")


if __name__ == "__main__":
    main()