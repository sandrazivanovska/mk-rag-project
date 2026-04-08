"""Abstract base class for all retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievedDoc:
    """A single retrieved document chunk."""

    chunk_id: str
    doc_id: str
    text: str
    lang: str
    score: float
    rank: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "lang": self.lang,
            "score": self.score,
            "rank": self.rank,
            **self.metadata,
        }


class BaseRetriever(ABC):
    """All retriever implementations inherit from this."""

    def __init__(self, top_k: int = 50):
        self.top_k = top_k

    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedDoc]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query string.
            top_k: Override the default top_k.

        Returns:
            List of RetrievedDoc sorted by descending relevance score.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(top_k={self.top_k})"
