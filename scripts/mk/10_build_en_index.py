"""
Step 10 — Build the English BGE-M3 FAISS index from en_chunks.jsonl.

Thin wrapper around src.retrieval.index_builder.build_faiss_index. The heavy
dependencies (faiss, FlagEmbedding/torch) are imported only when this script
runs, so steps 07-09 stay lightweight.

Usage:
    python scripts/mk/10_build_en_index.py
    python scripts/mk/10_build_en_index.py --chunks data/processed/en_chunks.jsonl \
        --index data/indices/faiss/en_bge_m3.index --model BAAI/bge-m3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_PATH = Path("data/processed/en_chunks.jsonl")
DEFAULT_INDEX_PATH = Path("data/indices/faiss/en_bge_m3.index")
DEFAULT_MODEL = "BAAI/bge-m3"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EN FAISS index (BGE-M3) from en_chunks.jsonl.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if not args.chunks.exists():
        raise FileNotFoundError(f"Chunks file does not exist: {args.chunks}. Run scripts/mk/09 first.")

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    # Force CPU + fp32: BGE-M3 fp16 on Apple-Silicon MPS crashes ('mps.add' type
    # mismatch). Disable MPS before the model is constructed.
    import torch

    torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]
    torch.backends.mps.is_built = lambda: False  # type: ignore[assignment]

    from src.retrieval.dense_retriever import DenseRetriever

    args.index.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building EN FAISS index: %s → %s (model=%s, cpu/fp32)", args.chunks, args.index, args.model)
    DenseRetriever.from_jsonl(
        args.chunks,
        model_name=args.model,
        index_path=args.index,
        batch_size=args.batch_size,
        use_fp16=False,
    )
    logger.info("Done. EN index → %s", args.index)


if __name__ == "__main__":
    main()
