# Retrieval Quality vs Generation Quality in Low-Resource RAG
## A Bilingual Study on Macedonian

---

## Research Questions

1. How do multilingual embedding models (BGE-M3, multilingual-e5-large) compare to BM25 for retrieving Macedonian documents?
2. Is it better to translate a Macedonian query to English and search English documents, or search Macedonian documents directly?
3. What happens when we mix Macedonian and English contexts — does it help or confuse the model?
4. How much does retrieval quality affect the final answer vs the choice of generator model?

---

## Project Structure

```
mk-rag-project/
├── configs/
│   └── default.yaml              # Experiment configuration
├── data/
│   ├── raw/                      # Raw corpora (not committed)
│   ├── processed/                # Cleaned & chunked JSONL files
│   └── indices/                  # FAISS + BM25 indices
├── notebooks/
│   ├── 01_data_exploration.py
│   └── 02_retrieval_comparison.py
├── scripts/
│   ├── download_mk_wiki.sh       # Download MK Wikipedia dump
│   └── create_gold_dataset.py    # Create evaluation gold dataset
├── src/
│   ├── ingestion/                # Data loading, cleaning, chunking
│   ├── retrieval/                # BM25, Dense, Hybrid, Bilingual retrievers
│   ├── reranker/                 # bge-reranker-v2-m3 cross-encoder
│   ├── generator/                # GPT-4o + Claude Sonnet 4.5 generators
│   ├── evaluation/               # RAGAS + custom MK metrics
│   ├── pipelines/                # Pipeline orchestration + factory
│   └── utils/                   # Config, logging
├── tests/
│   ├── test_retrieval/
│   ├── test_generation/
│   └── test_evaluation/
├── main.py                       # CLI entry point
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## The 12 Pipeline Variants

| # | Retrieval | Generator | ID |
|---|-----------|-----------|-----|
| 1 | MK-BM25 | GPT-4o | `mk_bm25_gpt4o` |
| 2 | MK-BM25 | Claude Sonnet | `mk_bm25_claude_sonnet` |
| 3 | MK-Dense (BGE-M3) | GPT-4o | `mk_dense_gpt4o` |
| 4 | MK-Dense (BGE-M3) | Claude Sonnet | `mk_dense_claude_sonnet` |
| 5 | MK-Hybrid (BGE-M3 + BM25) | GPT-4o | `mk_hybrid_gpt4o` |
| 6 | MK-Hybrid (BGE-M3 + BM25) | Claude Sonnet | `mk_hybrid_claude_sonnet` |
| 7 | Translate-Retrieve (MK→EN→EN corpus) | GPT-4o | `translate_retrieve_gpt4o` |
| 8 | Translate-Retrieve | Claude Sonnet | `translate_retrieve_claude_sonnet` |
| 9 | Cross-lingual Embed (MK query on EN corpus) | GPT-4o | `cross_lingual_embed_gpt4o` |
| 10 | Cross-lingual Embed | Claude Sonnet | `cross_lingual_embed_claude_sonnet` |
| 11 | Bilingual Fusion (MK + EN → rerank) | GPT-4o | `bilingual_fusion_gpt4o` |
| 12 | Bilingual Fusion | Claude Sonnet | `bilingual_fusion_claude_sonnet` |

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your OpenAI + Anthropic API keys
```

### 3. Download MK Wikipedia

```bash
bash scripts/download_mk_wiki.sh
```

### 4. Set up data (extract, clean, chunk)

```bash
python main.py setup-data \
    --mk-dump data/raw/mk/mkwiki-latest-pages-articles.xml.bz2
```

### 5. Build indices (FAISS + BM25)

```bash
python main.py build-indices --lang mk
```

### 6. Create gold evaluation dataset

```bash
python scripts/create_gold_dataset.py
```

### 7. Run a single pipeline

```bash
python main.py run-experiment \
    --pipeline mk_dense \
    --generator claude_sonnet \
    --gold-path data/gold_dataset.jsonl
```

### 8. Run all 12 pipelines

```bash
python main.py run-all --gold-path data/gold_dataset.jsonl
```

### 9. View results

```bash
python main.py evaluate --results-dir results/
```

---

## Evaluation Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| Faithfulness | RAGAS | Is the answer grounded in the context? |
| Answer Relevancy | RAGAS | Does the answer address the question? |
| Context Precision | RAGAS | How relevant are the retrieved chunks? |
| Context Recall | RAGAS | Are the relevant chunks retrieved? |
| Answer Correctness | RAGAS | How correct is the answer vs gold? |
| Token F1 | Custom | Token-level F1 vs gold answer (MK-normalised) |
| Exact Match | Custom | Normalised exact match (Cyrillic-aware) |
| Context Coverage | Custom | Fraction of answer tokens found in context |

---

## Technology Stack

| Component | Tool |
|-----------|------|
| Embeddings (primary) | BGE-M3 (`BAAI/bge-m3`) |
| Embeddings (baseline) | multilingual-e5-large-instruct |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Vector database | FAISS |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Translation | Google Translate API / deep-translator |
| Generators | GPT-4o + Claude Sonnet 4.5 |
| Evaluation | RAGAS + custom metrics |
| Orchestration | LangChain + LlamaIndex |
| Language | Python 3.11+ |

---

## References

From Prof. Sonja:
- [ACL 2024 Long Paper](https://aclanthology.org/2024.acl-long.192.pdf)
- [arXiv 2410.14815](https://arxiv.org/pdf/2410.14815)
- [EACL 2024 Tutorial](https://aclanthology.org/2024.eacl-tutorials.5.pdf)
- [NAACL 2024 Long Paper](https://aclanthology.org/2024.naacl-long.24v2.pdf)
