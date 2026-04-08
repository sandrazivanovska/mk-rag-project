"""
BM25 retriever using rank_bm25.

Supports Macedonian (Cyrillic) out of the box since BM25 is purely
token-frequency based and language-agnostic.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.utils.logging import get_logger
from .base import BaseRetriever, RetrievedDoc

logger = get_logger("bm25_retriever")


def _tokenise_mk(text: str) -> list[str]:
    """Simple whitespace tokeniser that handles Cyrillic well."""
    return text.lower().split()


class BM25Retriever(BaseRetriever):
    """
    BM25 retriever backed by rank_bm25 (BM25Okapi).

    Example
    -------
    >>> retriever = BM25Retriever.from_jsonl("data/processed/mk/chunks.jsonl")
    >>> docs = retriever.retrieve("Скопје е главниот град на Македонија", top_k=10)
    """

    def __init__(
        self,
        bm25: BM25Okapi,
        corpus: list[dict],
        top_k: int = 50,
    ):
        super().__init__(top_k=top_k)
        self._bm25 = bm25
        self._corpus = corpus  # parallel list of chunk dicts

    # ── Factory methods ────────────────────────────────────────────────────────

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: str | Path,
        *,
        top_k: int = 50,
        cache_path: Optional[str | Path] = None,
    ) -> "BM25Retriever":
        """Build a BM25 index from a JSONL file of chunks."""
        jsonl_path = Path(jsonl_path)

        # Try to load from cache
        if cache_path and Path(cache_path).exists():
            return cls._load(cache_path, top_k=top_k)

        logger.info(f"Building BM25 index from {jsonl_path}")
        corpus: list[dict] = []
        tokenised: list[list[str]] = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading corpus"):
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                corpus.append(doc)
                tokenised.append(_tokenise_mk(doc["text"]))

        bm25 = BM25Okapi(tokenised)
        retriever = cls(bm25, corpus, top_k=top_k)

        if cache_path:
            retriever._save(cache_path)

        logger.info(f"BM25 index built: {len(corpus):,} chunks")
        return retriever

    # ── Core retrieval ─────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedDoc]:
        k = top_k or self.top_k
        query_tokens = _tokenise_mk(query)
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[RetrievedDoc] = []
        for rank, idx in enumerate(top_indices):
            doc = self._corpus[idx]
            results.append(
                RetrievedDoc(
                    chunk_id=doc.get("chunk_id", str(idx)),
                    doc_id=doc.get("doc_id", str(idx)),
                    text=doc["text"],
                    lang=doc.get("lang", "mk"),
                    score=float(scores[idx]),
                    rank=rank,
                    metadata={k: v for k, v in doc.items() if k not in ("text", "chunk_id", "doc_id", "lang")},
                )
            )
        return results

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "corpus": self._corpus}, f)
        logger.info(f"BM25 index saved → {path}")

    @classmethod
    def _load(cls, path: str | Path, top_k: int = 50) -> "BM25Retriever":
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"BM25 index loaded from {path}")
        return cls(data["bm25"], data["corpus"], top_k=top_k)
