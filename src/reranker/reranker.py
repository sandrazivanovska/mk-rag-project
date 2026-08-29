"""
Cross-encoder reranker using bge-reranker-v2-m3.

Takes the first-stage results (top-50) and reranks them to top-5.
bge-reranker-v2-m3 is a multilingual cross-encoder that works well
with Macedonian text.
"""

from __future__ import annotations

from typing import Optional

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None  # type: ignore

from src.retrieval.base import RetrievedDoc
from src.utils.logging import get_logger

logger = get_logger("reranker")


class Reranker:
    """
    Multilingual cross-encoder reranker.

    Example
    -------
    >>> reranker = Reranker()
    >>> top5 = reranker.rerank(query, first_stage_docs, top_k=5)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        max_length: int = 384,
    ):
        self._model_name = model_name
        self._use_fp16 = use_fp16
        # The tokenizer's model_max_length is 8192, and leaving it unset makes
        # the cross-encoder pad far beyond anything we actually feed it. Chunks
        # are built at chunk_size=384 tokens, so capping here truncates nothing
        # and cuts reranking time by ~27% on a small GPU.
        self._max_length = max_length
        self._model: Optional[FlagReranker] = None

    @property
    def model(self) -> FlagReranker:
        if self._model is None:
            logger.info(f"Loading reranker model: {self._model_name}")
            self._model = FlagReranker(self._model_name, use_fp16=self._use_fp16)
        return self._model

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDoc],
        top_k: int = 5,
    ) -> list[RetrievedDoc]:
        """
        Rerank ``docs`` using the cross-encoder and return top ``top_k``.

        Args:
            query: The original query string (in Macedonian).
            docs: First-stage retrieved documents.
            top_k: How many to return after reranking.

        Returns:
            Top-k documents sorted by reranker score (descending).
        """
        if not docs:
            return []

        pairs = [[query, doc.text] for doc in docs]
        scores = self.model.compute_score(
            pairs, normalize=True, max_length=self._max_length
        )

        # Attach reranker scores
        for doc, score in zip(docs, scores):
            doc.score = float(score)
            doc.metadata["reranker_score"] = float(score)

        # Sort and take top-k
        reranked = sorted(docs, key=lambda d: d.score, reverse=True)[:top_k]

        # Update ranks
        for rank, doc in enumerate(reranked):
            doc.rank = rank

        logger.debug(
            f"Reranked {len(docs)} → {len(reranked)} docs. "
            f"Top score: {reranked[0].score:.4f}" if reranked else "No docs."
        )
        return reranked
