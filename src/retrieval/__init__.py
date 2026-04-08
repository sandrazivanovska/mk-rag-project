from .base import BaseRetriever, RetrievedDoc
from .bm25_retriever import BM25Retriever

# Heavy ML retrievers — imported lazily to avoid errors when FlagEmbedding is not installed
try:
    from .dense_retriever import DenseRetriever
    from .hybrid_retriever import HybridRetriever
    from .bilingual_retriever import BilingualFusionRetriever, CrossLingualRetriever
    from .index_builder import build_faiss_index, build_bm25_index
except ImportError:
    pass  # FlagEmbedding / faiss not installed yet — run: pip install FlagEmbedding faiss-cpu

__all__ = [
    "BaseRetriever",
    "RetrievedDoc",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "BilingualFusionRetriever",
    "build_faiss_index",
    "build_bm25_index",
]
