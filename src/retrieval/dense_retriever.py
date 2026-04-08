"""
Dense retriever using FAISS + BGE-M3 (or multilingual-e5-large-instruct).

BGE-M3 supports 100+ languages including Macedonian and produces
dense vectors of dimension 1024.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

from src.utils.logging import get_logger
from .base import BaseRetriever, RetrievedDoc

logger = get_logger("dense_retriever")


class DenseRetriever(BaseRetriever):
    """
    FAISS-backed dense retriever.

    Supports BGE-M3 (primary) and multilingual-e5-large-instruct (baseline).

    Example
    -------
    >>> retriever = DenseRetriever.from_jsonl(
    ...     "data/processed/mk/chunks.jsonl",
    ...     model_name="BAAI/bge-m3",
    ...     index_path="data/indices/faiss/mk_bge_m3.index",
    ... )
    >>> docs = retriever.retrieve("Скопје е главниот град на Македонија")
    """

    def __init__(
        self,
        index: faiss.Index,
        corpus: list[dict],
        model_name: str,
        top_k: int = 50,
        use_fp16: bool = True,
    ):
        super().__init__(top_k=top_k)
        self._index = index
        self._corpus = corpus
        self._model_name = model_name
        self._model: Optional[BGEM3FlagModel] = None  # lazy-loaded
        self._use_fp16 = use_fp16

    @property
    def model(self) -> BGEM3FlagModel:
        if self._model is None:
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = BGEM3FlagModel(
                self._model_name, use_fp16=self._use_fp16
            )
        return self._model

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: str | Path,
        *,
        model_name: str = "BAAI/bge-m3",
        index_path: Optional[str | Path] = None,
        corpus_cache_path: Optional[str | Path] = None,
        batch_size: int = 32,
        top_k: int = 50,
        use_fp16: bool = True,
    ) -> "DenseRetriever":
        """Build a FAISS index from a JSONL file of chunks."""
        jsonl_path = Path(jsonl_path)

        # Load corpus
        corpus: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    corpus.append(json.loads(line))

        # Try to load existing FAISS index
        if index_path and Path(index_path).exists():
            logger.info(f"Loading FAISS index from {index_path}")
            index = faiss.read_index(str(index_path))
            return cls(index, corpus, model_name, top_k=top_k, use_fp16=use_fp16)

        # Build index
        logger.info(f"Building FAISS index ({len(corpus):,} chunks, model={model_name})")
        embed_model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

        texts = [doc["text"] for doc in corpus]
        all_embeddings: list[np.ndarray] = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i : i + batch_size]
            output = embed_model.encode(batch, batch_size=batch_size, max_length=512)
            all_embeddings.append(output["dense_vecs"])

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        faiss.normalize_L2(embeddings)  # For cosine similarity via inner product

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product = cosine after L2-norm
        index.add(embeddings)

        if index_path:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(index_path))
            logger.info(f"FAISS index saved → {index_path}")

        return cls(index, corpus, model_name, top_k=top_k, use_fp16=use_fp16)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedDoc]:
        k = top_k or self.top_k
        output = self.model.encode([query], batch_size=1, max_length=512)
        query_vec = output["dense_vecs"].astype(np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self._index.search(query_vec, k)
        scores = scores[0]
        indices = indices[0]

        results: list[RetrievedDoc] = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < 0:
                break
            doc = self._corpus[idx]
            results.append(
                RetrievedDoc(
                    chunk_id=doc.get("chunk_id", str(idx)),
                    doc_id=doc.get("doc_id", str(idx)),
                    text=doc["text"],
                    lang=doc.get("lang", "mk"),
                    score=float(score),
                    rank=rank,
                    metadata={
                        k: v
                        for k, v in doc.items()
                        if k not in ("text", "chunk_id", "doc_id", "lang")
                    },
                )
            )
        return results

    def encode_query(self, query: str) -> np.ndarray:
        """Return a normalised query embedding (for hybrid retrieval)."""
        output = self.model.encode([query], batch_size=1, max_length=512)
        vec = output["dense_vecs"].astype(np.float32)
        faiss.normalize_L2(vec)
        return vec[0]
