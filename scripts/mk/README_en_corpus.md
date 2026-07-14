# English Parallel Corpus — Runbook (steps 07–10)

Builds the English knowledge base for the cross-lingual pipelines (Translate-Retrieve,
Cross-lingual Embed, Bilingual Fusion). Two origins, per project plan §4.2:

- **EN Wikipedia** for MK articles that have an English equivalent (Wikidata interlanguage links)
- **Machine translation (MK→EN)** for MK articles with no English equivalent — free `deep-translator`, no API key

Every EN document and chunk carries `mk_doc_id`, `wikidata_qid`, and a `source` tag
(`en_wikipedia` vs `mt_from_mk`) so MK↔EN chunks can be joined and real-EN vs MT-EN
analysed separately.

## Prerequisites

1. **MK side must exist first** — these steps read `data/processed/mk_documents.jsonl`.
   Run Dimitar's pipeline if you haven't:
   ```bash
   python scripts/mk/01_download_mk_wikipedia.py
   # extract the dump with WikiExtractor into data/raw/mk_wikipedia/extracted/
   python scripts/mk/02_build_mk_documents.py
   ```
2. **Install deps** not present by default:
   ```bash
   pip install deep-translator          # MT path (step 08)
   pip install faiss-cpu FlagEmbedding  # index (step 10)
   ```

## Pipeline

```bash
# 07 — link MK docs to EN Wikipedia titles + Wikidata QIDs; split linked / needs_mt
python scripts/mk/07_link_mk_to_en.py
#   → data/processed/mk_en_alignment.jsonl
#
# At scale (1k+ docs) use OFFLINE mode instead — the per-IP API rate limit
# (~1 batch/minute once throttled) makes the API path impractical. Download the
# langlinks + page_props SQL dumps once and parse them locally (no network):
#   curl -A "mk-rag-research/1.0 (<your email>)" -o data/raw/mk_wikipedia/mkwiki-latest-langlinks.sql.gz \
#     https://dumps.wikimedia.org/mkwiki/latest/mkwiki-latest-langlinks.sql.gz
#   curl -A "mk-rag-research/1.0 (<your email>)" -o data/raw/mk_wikipedia/mkwiki-latest-page_props.sql.gz \
#     https://dumps.wikimedia.org/mkwiki/latest/mkwiki-latest-page_props.sql.gz
python scripts/mk/07_link_mk_to_en.py \
    --langlinks-sql data/raw/mk_wikipedia/mkwiki-latest-langlinks.sql.gz \
    --pageprops-sql data/raw/mk_wikipedia/mkwiki-latest-page_props.sql.gz

# 08 — fetch EN Wikipedia full text (Path A) + machine-translate the rest (Path B)
#   Resumable: re-run after a throttle/interrupt and it skips what's done.
python scripts/mk/08_build_en_documents.py --max-mt-docs 2000
#   → data/processed/en_documents.jsonl
#
# At scale, Path A via the API hits the same per-IP throttle as 07 (~1 req/min
# once triggered → days for thousands of docs). Use 08b instead: it Range-fetches
# only the needed ~100-page bz2 blocks from the enwiki *multistream* dump
# (static file server, no rate limits; ~1-2 h for 7k docs). Same output file and
# resume semantics as 08, so run 08b first, then 08 to top up stragglers + MT:
#   curl -A "mk-rag-research/1.0 (<your email>)" -o data/raw/enwiki-multistream-index.txt.bz2 \
#     https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream-index.txt.bz2
python scripts/mk/08b_fetch_en_multistream.py
python scripts/mk/08_build_en_documents.py --max-mt-docs 3000   # stragglers + MT

# 09 — chunk EN docs (reuses the MK sentence chunker; identical strategy)
python scripts/mk/09_chunk_en_documents.py
#   → data/processed/en_chunks.jsonl

# 10 — build the EN BGE-M3 FAISS index
python scripts/mk/10_build_en_index.py
#   → data/indices/faiss/en_bge_m3.index
```

## Smoke run first (recommended)

```bash
python scripts/mk/07_link_mk_to_en.py --max-docs 50
python scripts/mk/08_build_en_documents.py --max-docs 20 --max-mt-docs 5
python scripts/mk/09_chunk_en_documents.py
```

## Corpus capping (before 07)

For CPU-only machines, cap the MK corpus before building the EN side — embedding
the full corpus takes days. The capping script pins every gold-QA source article
(from `data/evaluation/mk_qa_template_300.csv`) and fills the rest with seeded
random distractors, so retrieval metrics stay valid:

```bash
python scripts/mk/02b_cap_mk_documents.py --target-size 10000
# full corpus is preserved at data/processed/mk_documents_full.jsonl
```

## Notes

- **Cost:** the MT engine is the free Google endpoint via `deep-translator` — **$0**. The
  `--max-mt-docs` cap (default 2000) only guards against rate-limiting/time on the first run;
  raise it later and run in batches (`08` is resumable).
- **`--no-mt`** on step 08 builds an EN-Wikipedia-only corpus (Path A only).
- **Throttling:** `--sleep` (default 0.1s) spaces out API/MT calls. Increase if you hit limits.
- **Field compatibility:** chunks emit both `lang` and `language` plus `chunk_id`, so
  `src/retrieval/index_builder.build_faiss_index` consumes `en_chunks.jsonl` unchanged.

## Tests

```bash
python -m pytest tests/test_ingestion/test_en_pipeline.py -q
```
