"""
Step 09 — Chunk the English documents.

Reuses the exact sentence-window chunking primitives from
``03_chunk_mk_documents.py`` so MK and EN are chunked identically (same target
size, overlap, min length, sentence splitting). Each EN chunk:

  - is tagged ``lang="en"`` (and ``language="en"``) for the dense index builder,
  - carries the alignment keys ``wikidata_qid`` and ``mk_doc_id`` plus ``source``,
    so the bilingual-fusion pipeline can join MK↔EN chunks on the same topic.

Usage:
    python scripts/mk/09_chunk_en_documents.py
    python scripts/mk/09_chunk_en_documents.py --input data/processed/en_documents.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path("data/processed/en_documents.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/en_chunks.jsonl")


def _load_mk_chunker():
    """Import the numeric-prefixed MK chunker module by path."""
    path = Path(__file__).resolve().parent / "03_chunk_mk_documents.py"
    spec = importlib.util.spec_from_file_location("mk_chunker", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # required so @dataclass can resolve __module__
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_mk = _load_mk_chunker()
ChunkingConfig = _mk.ChunkingConfig


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def make_en_chunks_for_document(
    document: Dict[str, Any],
    config: "ChunkingConfig",
) -> List[Dict[str, Any]]:
    """
    Chunk one EN document, reusing the MK sentence-window pipeline and carrying
    the EN language tag + alignment keys into every chunk.
    """
    document_id = document["doc_id"]
    document_text = _mk.normalize_whitespace(document.get("text", ""))

    sentences = _mk.split_into_sentences(document_text)
    prepared = _mk.prepare_sentences(sentences, config)
    raw_chunk_texts = _mk.chunk_sentences(prepared, config)

    chunks: List[Dict[str, Any]] = []
    kept_index = 0
    for raw_index, chunk_text in enumerate(raw_chunk_texts):
        chunk_text = _mk.normalize_whitespace(chunk_text)
        token_count = _mk.count_tokens(chunk_text)
        if token_count < config.min_chunk_tokens:
            continue

        chunks.append(
            {
                "chunk_id": _mk.make_chunk_id(document_id, kept_index),
                "doc_id": document_id,
                "title": document.get("title", ""),
                # Both keys: `lang` for src/ DenseRetriever, `language` for MK-pipeline parity.
                "lang": "en",
                "language": "en",
                "source": document.get("source", ""),
                "text": chunk_text,
                "chunk_index": kept_index,
                "raw_chunk_index": raw_index,
                "token_count": token_count,
                "url": document.get("url"),
                # Alignment keys → enable MK↔EN joins for bilingual fusion.
                "wikidata_qid": document.get("wikidata_qid"),
                "mk_doc_id": document.get("mk_doc_id"),
                "metadata": {
                    "chunking_strategy": config.strategy_name,
                    "target_tokens": config.target_tokens,
                    "overlap_tokens": config.overlap_tokens,
                    "min_chunk_tokens": config.min_chunk_tokens,
                    "max_sentence_tokens": config.max_sentence_tokens,
                },
            }
        )
        kept_index += 1

    return chunks


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def iter_chunks(input_path: Path, config: "ChunkingConfig", max_documents: Optional[int]) -> Iterator[Dict[str, Any]]:
    docs_seen = chunks_created = docs_with_chunks = 0
    for document in iter_jsonl(input_path):
        if max_documents is not None and docs_seen >= max_documents:
            break
        docs_seen += 1
        chunks = make_en_chunks_for_document(document, config)
        if chunks:
            docs_with_chunks += 1
        for chunk in chunks:
            chunks_created += 1
            yield chunk
    logger.info("Documents read: %d | with chunks: %d | chunks: %d", docs_seen, docs_with_chunks, chunks_created)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk EN documents (reuses MK sentence chunker).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target-tokens", type=int, default=400)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--min-chunk-tokens", type=int, default=80)
    parser.add_argument("--max-sentence-tokens", type=int, default=120)
    parser.add_argument("--max-documents", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}. Run scripts/mk/08 first.")

    config = ChunkingConfig(
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
        max_sentence_tokens=args.max_sentence_tokens,
    )
    _mk.validate_config(config)

    logger.info("Chunking EN documents: %s → %s", args.input, args.output)
    total = write_jsonl(iter_chunks(args.input, config, args.max_documents), args.output)
    logger.info("EN chunks written: %d", total)


if __name__ == "__main__":
    main()
