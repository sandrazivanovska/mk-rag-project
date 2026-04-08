"""
Convenience functions for building FAISS and BM25 indices from JSONL files.
Used by the setup scripts.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.logging import get_logger
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever

logger = get_logger("index_builder")


def build_faiss_index(
    jsonl_path: str | Path,
    index_path: str | Path,
    *,
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 32,
) -> DenseRetriever:
    """Build and persist a FAISS index from a chunks JSONL."""
    logger.info(f"build_faiss_index: {jsonl_path} → {index_path}")
    return DenseRetriever.from_jsonl(
        jsonl_path,
        model_name=model_name,
        index_path=index_path,
        batch_size=batch_size,
    )


def build_bm25_index(
    jsonl_path: str | Path,
    cache_path: str | Path,
) -> BM25Retriever:
    """Build and persist a BM25 index from a chunks JSONL."""
    logger.info(f"build_bm25_index: {jsonl_path} → {cache_path}")
    return BM25Retriever.from_jsonl(jsonl_path, cache_path=cache_path)
