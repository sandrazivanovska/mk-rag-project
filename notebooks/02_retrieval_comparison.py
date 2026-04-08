"""
Notebook 02: Retrieval Comparison

Compare BM25 vs Dense vs Hybrid retrieval on a set of sample queries.
Requires data and indices to be built first.
"""

# %%
import sys
sys.path.insert(0, ".")

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.config import get_settings

settings = get_settings()

# %%
SAMPLE_QUERIES = [
    "Кој е главниот град на Македонија?",
    "Историјата на македонскиот јазик",
    "Охридско Езеро природа",
]

# %%
# Load BM25
print("Loading BM25...")
bm25 = BM25Retriever.from_jsonl(
    "data/processed/mk/chunks.jsonl",
    cache_path="data/indices/bm25/mk/mk_bm25.pkl",
    top_k=10,
)

# %%
# Query BM25
for query in SAMPLE_QUERIES:
    print(f"\nQuery: {query}")
    print("─" * 60)
    results = bm25.retrieve(query, top_k=3)
    for r in results:
        print(f"[{r.score:.3f}] {r.text[:150]}...")

# %%
# Note: Dense and Hybrid require the FAISS index to be built.
# Run: python main.py build-indices
print("\n[Dense and Hybrid retrieval require FAISS index — run main.py build-indices first]")
