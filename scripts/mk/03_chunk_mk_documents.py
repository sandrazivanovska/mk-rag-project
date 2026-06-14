from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path("data/processed/mk_documents.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/mk_chunks.jsonl")


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = 400
    overlap_tokens: int = 64
    min_chunk_tokens: int = 80
    max_sentence_tokens: int = 120
    strategy_name: str = "sentence_window_with_overlap"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def count_tokens(text: str) -> int:
    if not text:
        return 0

    return len(text.split())


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
                    f"Expected JSON object in {path} on line {line_number}, "
                    f"got {type(record).__name__}"
                )

            yield record


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)

    written = 0

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return written


def split_into_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)

    if not text:
        return []

    raw_sentences = re.split(r"(?<=[.!?…])\s+", text)

    return [sentence.strip() for sentence in raw_sentences if sentence.strip()]


def split_long_sentence(sentence: str, max_tokens: int) -> List[str]:
    words = sentence.split()

    if len(words) <= max_tokens:
        return [sentence]

    parts = []

    for start in range(0, len(words), max_tokens):
        part = " ".join(words[start : start + max_tokens]).strip()

        if part:
            parts.append(part)

    return parts


def prepare_sentences(sentences: List[str], config: ChunkingConfig) -> List[str]:
    prepared_sentences: List[str] = []

    for sentence in sentences:
        prepared_sentences.extend(
            split_long_sentence(
                sentence=sentence,
                max_tokens=config.max_sentence_tokens,
            )
        )

    return prepared_sentences


def build_overlap_sentences(
    current_sentences: List[str],
    overlap_tokens: int,
) -> tuple[List[str], int]:
    overlap_sentences: List[str] = []
    overlap_token_count = 0

    for previous_sentence in reversed(current_sentences):
        previous_token_count = count_tokens(previous_sentence)

        if overlap_token_count + previous_token_count > overlap_tokens:
            break

        overlap_sentences.insert(0, previous_sentence)
        overlap_token_count += previous_token_count

    return overlap_sentences, overlap_token_count


def chunk_sentences(
    sentences: List[str],
    config: ChunkingConfig,
) -> List[str]:
    chunks: List[str] = []

    current_sentences: List[str] = []
    current_token_count = 0

    for sentence in sentences:
        sentence_token_count = count_tokens(sentence)

        sentence_would_exceed_target = (
            current_sentences
            and current_token_count + sentence_token_count > config.target_tokens
        )

        if sentence_would_exceed_target:
            chunk_text = normalize_whitespace(" ".join(current_sentences))
            chunks.append(chunk_text)

            overlap_sentences, overlap_token_count = build_overlap_sentences(
                current_sentences=current_sentences,
                overlap_tokens=config.overlap_tokens,
            )

            current_sentences = overlap_sentences
            current_token_count = overlap_token_count

        current_sentences.append(sentence)
        current_token_count += sentence_token_count

    if current_sentences:
        chunk_text = normalize_whitespace(" ".join(current_sentences))
        chunks.append(chunk_text)

    return chunks


def validate_config(config: ChunkingConfig) -> None:
    if config.target_tokens <= 0:
        raise ValueError("target_tokens must be greater than 0")

    if config.overlap_tokens < 0:
        raise ValueError("overlap_tokens must be greater than or equal to 0")

    if config.min_chunk_tokens <= 0:
        raise ValueError("min_chunk_tokens must be greater than 0")

    if config.max_sentence_tokens <= 0:
        raise ValueError("max_sentence_tokens must be greater than 0")

    if config.overlap_tokens >= config.target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens")

    if config.min_chunk_tokens > config.target_tokens:
        logger.warning(
            "min_chunk_tokens is larger than target_tokens. "
            "This may skip many chunks."
        )


def validate_document(document: Dict[str, Any]) -> None:
    required_fields = ["doc_id", "source", "title", "language", "text"]

    for field in required_fields:
        if field not in document:
            raise ValueError(f"Document is missing required field: {field}")

    if not isinstance(document["doc_id"], str) or not document["doc_id"].strip():
        raise ValueError("Document field 'doc_id' must be a non-empty string")

    if not isinstance(document["text"], str) or not document["text"].strip():
        raise ValueError(f"Document {document.get('doc_id')} has empty text")

    if document["language"] != "mk":
        raise ValueError(
            f"Expected Macedonian document language='mk', "
            f"got language={document['language']} for doc_id={document['doc_id']}"
        )


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}_chunk_{chunk_index:04d}"


def make_chunks_for_document(
    document: Dict[str, Any],
    config: ChunkingConfig,
) -> List[Dict[str, Any]]:
    validate_document(document)

    document_id = document["doc_id"]
    document_text = normalize_whitespace(document["text"])

    sentences = split_into_sentences(document_text)
    prepared_sentences = prepare_sentences(sentences, config)
    raw_chunk_texts = chunk_sentences(prepared_sentences, config)

    chunks: List[Dict[str, Any]] = []
    kept_chunk_index = 0

    for raw_chunk_index, chunk_text in enumerate(raw_chunk_texts):
        chunk_text = normalize_whitespace(chunk_text)
        token_count = count_tokens(chunk_text)

        if token_count < config.min_chunk_tokens:
            continue

        chunk_id = make_chunk_id(document_id, kept_chunk_index)

        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": document_id,
                "title": document["title"],
                "language": document["language"],
                "source": document["source"],
                "text": chunk_text,
                "chunk_index": kept_chunk_index,
                "raw_chunk_index": raw_chunk_index,
                "token_count": token_count,
                "url": document.get("url"),
                "metadata": {
                    **document.get("metadata", {}),
                    "chunking_strategy": config.strategy_name,
                    "target_tokens": config.target_tokens,
                    "overlap_tokens": config.overlap_tokens,
                    "min_chunk_tokens": config.min_chunk_tokens,
                    "max_sentence_tokens": config.max_sentence_tokens,
                },
            }
        )

        kept_chunk_index += 1

    return chunks


def iter_chunks(
    input_path: Path,
    config: ChunkingConfig,
    max_documents: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    documents_seen = 0
    documents_with_chunks = 0
    chunks_created = 0
    skipped_documents = 0

    for document in iter_jsonl(input_path):
        if max_documents is not None and documents_seen >= max_documents:
            break

        documents_seen += 1

        try:
            chunks = make_chunks_for_document(document, config)
        except ValueError as error:
            skipped_documents += 1
            logger.warning("Skipping invalid document: %s", error)
            continue

        if chunks:
            documents_with_chunks += 1

        for chunk in chunks:
            chunks_created += 1
            yield chunk

        if documents_seen % 10_000 == 0:
            logger.info(
                "Progress: documents=%s, documents_with_chunks=%s, chunks=%s",
                documents_seen,
                documents_with_chunks,
                chunks_created,
            )

    logger.info("Documents read: %s", documents_seen)
    logger.info("Documents with chunks: %s", documents_with_chunks)
    logger.info("Documents skipped: %s", skipped_documents)
    logger.info("Chunks created: %s", chunks_created)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create sentence-aware Macedonian RAG chunks from JSONL documents."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSONL documents file. Default: {DEFAULT_INPUT_PATH}",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSONL chunks file. Default: {DEFAULT_OUTPUT_PATH}",
    )

    parser.add_argument(
        "--target-tokens",
        type=int,
        default=400,
        help="Approximate target number of whitespace tokens per chunk.",
    )

    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=64,
        help="Approximate number of whitespace tokens to overlap between chunks.",
    )

    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=80,
        help="Minimum number of whitespace tokens required to keep a chunk.",
    )

    parser.add_argument(
        "--max-sentence-tokens",
        type=int,
        default=120,
        help="Split sentences longer than this many whitespace tokens.",
    )

    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional limit for debugging. Example: --max-documents 100",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    config = ChunkingConfig(
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
        max_sentence_tokens=args.max_sentence_tokens,
    )

    validate_config(config)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    logger.info("Starting Macedonian document chunking")
    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output)
    logger.info("Config: %s", config)

    total_written = write_jsonl(
        records=iter_chunks(
            input_path=args.input,
            config=config,
            max_documents=args.max_documents,
        ),
        path=args.output,
    )

    logger.info("Chunks written to disk: %s", total_written)
    logger.info("Finished successfully")


if __name__ == "__main__":
    main()