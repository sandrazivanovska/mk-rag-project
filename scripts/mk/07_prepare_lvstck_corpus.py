from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


DATASET_NAME = "LVSTCK/macedonian-corpus-cleaned-dedup"

DEFAULT_RAW_OUTPUT_PATH = Path("data/raw/mk_lvstck/mk_random_subset_5000.jsonl")
DEFAULT_DOCUMENTS_OUTPUT_PATH = Path("data/processed/mk_lvstck_documents.jsonl")
DEFAULT_CHUNKS_OUTPUT_PATH = Path("data/processed/mk_lvstck_chunks.jsonl")

DEFAULT_SAMPLE_SIZE = 5_000
DEFAULT_SAMPLE_PROBABILITY = 0.001
DEFAULT_SEED = 42

logger = logging.getLogger(__name__)


Document = Dict[str, Any]


def load_script_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


build_mk_documents = load_script_module(
    Path(__file__).with_name("02_build_mk_documents.py"),
    "mk_build_documents",
)
chunk_mk_documents = load_script_module(
    Path(__file__).with_name("03_chunk_mk_documents.py"),
    "mk_chunk_documents",
)


clean_text = build_mk_documents.clean_text
is_good_macedonian_text = build_mk_documents.is_good_macedonian_text


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "The 'datasets' package is required to stream the LVSTCK corpus. "
            "Install project dependencies first: python -m pip install -r requirements.txt"
        ) from error

    return load_dataset


def get_text(record: Dict[str, Any]) -> str:
    for field in ["text", "content", "document", "sentence"]:
        value = record.get(field)

        if isinstance(value, str) and value.strip():
            return value

    return ""


def get_optional_string(record: Dict[str, Any], field: str) -> Optional[str]:
    value = record.get(field)

    if value is None:
        return None

    value = str(value).strip()
    return value or None


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


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0

    return sum(1 for _ in path.open("r", encoding="utf-8"))


def load_seen_texts(path: Path) -> set[str]:
    if not path.exists():
        return set()

    seen_texts = set()

    for record in iter_jsonl(path):
        text = get_text(record)

        if text:
            seen_texts.add(text)

    return seen_texts


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def stream_random_subset(
    *,
    dataset_name: str,
    output_path: Path,
    sample_size: int,
    sample_probability: float,
    seed: int,
    append: bool = False,
) -> int:
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")

    if not 0 < sample_probability <= 1:
        raise ValueError("sample_probability must be in the interval (0, 1]")

    load_dataset = require_datasets()
    rng = random.Random(seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_count = count_jsonl(output_path) if append else 0
    seen_texts = load_seen_texts(output_path) if append else set()

    if existing_count >= sample_size:
        logger.info(
            "Raw subset already has at least the requested number of records: %s",
            existing_count,
        )
        return existing_count

    logger.info("Streaming dataset: %s", dataset_name)
    logger.info("Sample probability: %s", sample_probability)
    logger.info("Target sample size: %s", sample_size)
    logger.info("Raw subset output: %s", output_path)
    logger.info("Existing raw records: %s", existing_count)

    dataset = load_dataset(dataset_name, split="train", streaming=True)

    scanned = 0
    saved = existing_count
    duplicates = 0

    mode = "a" if append else "w"

    with output_path.open(mode, encoding="utf-8") as output_file:
        for item in dataset:
            scanned += 1

            if rng.random() >= sample_probability:
                continue

            text = get_text(item)

            if append and text and text in seen_texts:
                duplicates += 1
                continue

            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            output_file.flush()

            if text:
                seen_texts.add(text)

            saved += 1

            if saved % 500 == 0:
                logger.info("Sampled %s records after scanning %s", saved, scanned)

            if saved >= sample_size:
                break

    logger.info(
        "Finished sampling. Scanned=%s, saved=%s, duplicate_samples_skipped=%s",
        scanned,
        saved,
        duplicates,
    )

    if saved < sample_size:
        logger.warning(
            "Saved fewer records than requested. Requested=%s, saved=%s",
            sample_size,
            saved,
        )

    return saved


def make_document(
    *,
    raw_record: Dict[str, Any],
    raw_index: int,
    document_index: int,
    raw_path: Path,
    dataset_name: str,
) -> Optional[Document]:
    text = clean_text(get_text(raw_record))

    if not is_good_macedonian_text(text):
        return None

    title = get_optional_string(raw_record, "title")

    if title is None:
        title = f"LVSTCK document {document_index:05d}"

    original_id = (
        get_optional_string(raw_record, "id")
        or get_optional_string(raw_record, "_id")
        or str(raw_index)
    )

    return {
        "doc_id": f"mk_lvstck_{document_index:07d}",
        "source": "lvstck",
        "title": title,
        "language": "mk",
        "text": text,
        "url": get_optional_string(raw_record, "url"),
        "metadata": {
            "original_id": original_id,
            "dataset": dataset_name,
            "domain": "general",
            "source_file": str(raw_path),
            "raw_record_index": raw_index,
        },
    }


def iter_lvstck_documents(
    *,
    raw_path: Path,
    dataset_name: str,
) -> Iterator[Document]:
    seen_texts: set[str] = set()
    kept = 0
    skipped = 0
    duplicates = 0

    for raw_index, raw_record in enumerate(iter_jsonl(raw_path), start=1):
        document = make_document(
            raw_record=raw_record,
            raw_index=raw_index,
            document_index=kept,
            raw_path=raw_path,
            dataset_name=dataset_name,
        )

        if document is None:
            skipped += 1
            continue

        text = document["text"]

        if text in seen_texts:
            duplicates += 1
            continue

        seen_texts.add(text)
        kept += 1
        yield document

    logger.info("LVSTCK documents kept: %s", kept)
    logger.info("LVSTCK records skipped by quality filter: %s", skipped)
    logger.info("LVSTCK duplicate texts skipped: %s", duplicates)


def build_documents(
    *,
    raw_path: Path,
    output_path: Path,
    dataset_name: str,
) -> int:
    logger.info("Building LVSTCK documents from %s", raw_path)
    logger.info("Documents output: %s", output_path)

    return write_jsonl(
        iter_lvstck_documents(raw_path=raw_path, dataset_name=dataset_name),
        output_path,
    )


def build_chunks(
    *,
    documents_path: Path,
    chunks_path: Path,
    target_tokens: int,
    overlap_tokens: int,
    min_chunk_tokens: int,
    max_sentence_tokens: int,
) -> int:
    config = chunk_mk_documents.ChunkingConfig(
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        min_chunk_tokens=min_chunk_tokens,
        max_sentence_tokens=max_sentence_tokens,
    )

    chunk_mk_documents.validate_config(config)

    logger.info("Chunking LVSTCK documents")
    logger.info("Chunks output: %s", chunks_path)
    logger.info("Chunking config: %s", config)

    return chunk_mk_documents.write_jsonl(
        records=chunk_mk_documents.iter_chunks(
            input_path=documents_path,
            config=config,
        ),
        path=chunks_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare LVSTCK Macedonian corpus data using the same document and "
            "chunk format as the Macedonian Wikipedia pipeline."
        )
    )

    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help=f"HuggingFace dataset name. Default: {DATASET_NAME}",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=DEFAULT_RAW_OUTPUT_PATH,
        help=f"Raw sampled JSONL output. Default: {DEFAULT_RAW_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--documents-output",
        type=Path,
        default=DEFAULT_DOCUMENTS_OUTPUT_PATH,
        help=f"Processed documents JSONL output. Default: {DEFAULT_DOCUMENTS_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--chunks-output",
        type=Path,
        default=DEFAULT_CHUNKS_OUTPUT_PATH,
        help=f"Processed chunks JSONL output. Default: {DEFAULT_CHUNKS_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of sampled raw records to keep. Default: {DEFAULT_SAMPLE_SIZE}",
    )
    parser.add_argument(
        "--sample-probability",
        type=float,
        default=DEFAULT_SAMPLE_PROBABILITY,
        help=(
            "Probability for keeping each streamed record before the sample-size "
            f"cap is reached. Default: {DEFAULT_SAMPLE_PROBABILITY}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible sampling. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Skip HuggingFace streaming when --raw-output already exists.",
    )
    parser.add_argument(
        "--append-raw",
        action="store_true",
        help=(
            "Append sampled records to an existing --raw-output until "
            "--sample-size is reached. Use a different --seed for an independent "
            "fill pass."
        ),
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

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.reuse_raw and args.raw_output.exists():
        logger.info("Reusing existing raw subset: %s", args.raw_output)
    else:
        stream_random_subset(
            dataset_name=args.dataset_name,
            output_path=args.raw_output,
            sample_size=args.sample_size,
            sample_probability=args.sample_probability,
            seed=args.seed,
            append=args.append_raw,
        )

    documents_written = build_documents(
        raw_path=args.raw_output,
        output_path=args.documents_output,
        dataset_name=args.dataset_name,
    )
    logger.info("Documents written: %s", documents_written)

    chunks_written = build_chunks(
        documents_path=args.documents_output,
        chunks_path=args.chunks_output,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
        max_sentence_tokens=args.max_sentence_tokens,
    )
    logger.info("Chunks written: %s", chunks_written)
    logger.info("LVSTCK preparation finished successfully")


if __name__ == "__main__":
    main()
