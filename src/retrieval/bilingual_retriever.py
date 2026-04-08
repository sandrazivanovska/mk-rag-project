"""
Bilingual Fusion retriever.

Retrieves top-k from BOTH the MK and EN corpora, then merges and
optionally reranks the combined result.

Pipeline 5: Cross-lingual Embed  — MK query → BGE-M3 directly on EN corpus
Pipeline 6: Bilingual Fusion     — MK query → retrieve from MK + EN → merge + rerank
"""

from __future__ import annotations

from typing import Optional

from .base import BaseRetriever, RetrievedDoc
from .dense_retriever import DenseRetriever
from src.utils.logging import get_logger

logger = get_logger("bilingual_retriever")


class BilingualFusionRetriever(BaseRetriever):
    """
    Retrieves from MK and EN corpora independently and merges results.

    Merge strategy: score normalisation → weighted combination.
    MK results get weight ``mk_weight``, EN results get ``(1 - mk_weight)``.

    The fused list is then passed to the reranker (if configured).
    """

    def __init__(
        self,
        mk_retriever: DenseRetriever,
        en_retriever: DenseRetriever,
        top_k: int = 50,
        mk_weight: float = 0.5,
    ):
        super().__init__(top_k=top_k)
        self._mk = mk_retriever
        self._en = en_retriever
        self.mk_weight = mk_weight

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedDoc]:
        """
        Retrieve top-k from MK and EN, normalise scores, merge and re-sort.
        """
        k = top_k or self.top_k

        mk_results = self._mk.retrieve(query, top_k=k)
        en_results = self._en.retrieve(query, top_k=k)

        def _normalise(docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
            if not docs:
                return docs
            min_s = min(d.score for d in docs)
            max_s = max(d.score for d in docs)
            span = max_s - min_s if max_s > min_s else 1.0
            for d in docs:
                d.score = (d.score - min_s) / span
            return docs

        mk_results = _normalise(mk_results)
        en_results = _normalise(en_results)

        # Apply language weights
        for d in mk_results:
            d.score *= self.mk_weight
        for d in en_results:
            d.score *= 1 - self.mk_weight

        # Merge and deduplicate by chunk_id
        seen: set[str] = set()
        merged: list[RetrievedDoc] = []
        for doc in mk_results + en_results:
            if doc.chunk_id not in seen:
                seen.add(doc.chunk_id)
                merged.append(doc)

        # Sort by fused score
        merged.sort(key=lambda d: d.score, reverse=True)
        merged = merged[:k]

        # Update ranks
        for rank, doc in enumerate(merged):
            doc.rank = rank

        return merged


class CrossLingualRetriever(BaseRetriever):
    """
    Pipeline 5: Cross-lingual Embed.

    Uses BGE-M3 to encode the Macedonian query and retrieve directly
    from the English FAISS index — no translation required.
    """

    def __init__(self, en_retriever: DenseRetriever, top_k: int = 50):
        super().__init__(top_k=top_k)
        self._en = en_retriever

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedDoc]:
        """Encode MK query with multilingual model → search EN index."""
        return self._en.retrieve(query, top_k=top_k or self.top_k)
