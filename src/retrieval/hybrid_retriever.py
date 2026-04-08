"""
Hybrid retriever: BGE-M3 dense + BM25, fused via Reciprocal Rank Fusion (RRF).

alpha controls the weight: alpha=1 → dense only, alpha=0 → BM25 only.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseRetriever, RetrievedDoc
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from src.utils.logging import get_logger

logger = get_logger("hybrid_retriever")

_RRF_K = 60  # Standard RRF constant


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining BM25 and dense retrieval via RRF.

    Example
    -------
    >>> hybrid = HybridRetriever(bm25=bm25_retriever, dense=dense_retriever, alpha=0.5)
    >>> docs = hybrid.retrieve("Скопје е главниот град на Македонија")
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        top_k: int = 50,
        alpha: float = 0.5,
        rrf_k: int = _RRF_K,
    ):
        super().__init__(top_k=top_k)
        self._bm25 = bm25
        self._dense = dense
        self.alpha = alpha
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedDoc]:
        k = top_k or self.top_k

        # Retrieve from both
        bm25_results = self._bm25.retrieve(query, top_k=k * 2)
        dense_results = self._dense.retrieve(query, top_k=k * 2)

        # RRF fusion
        scores: dict[str, float] = {}
        doc_map: dict[str, RetrievedDoc] = {}

        for rank, doc in enumerate(bm25_results):
            cid = doc.chunk_id
            scores[cid] = scores.get(cid, 0.0) + (1 - self.alpha) / (self.rrf_k + rank + 1)
            doc_map[cid] = doc

        for rank, doc in enumerate(dense_results):
            cid = doc.chunk_id
            scores[cid] = scores.get(cid, 0.0) + self.alpha / (self.rrf_k + rank + 1)
            doc_map[cid] = doc

        # Sort by fused score
        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:k]

        results: list[RetrievedDoc] = []
        for rank, cid in enumerate(sorted_ids):
            doc = doc_map[cid]
            results.append(
                RetrievedDoc(
                    chunk_id=doc.chunk_id,
                    doc_id=doc.doc_id,
                    text=doc.text,
                    lang=doc.lang,
                    score=scores[cid],
                    rank=rank,
                    metadata=doc.metadata,
                )
            )
        return results
