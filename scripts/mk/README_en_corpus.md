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

# 08 — fetch EN Wikipedia full text (Path A) + machine-translate the rest (Path B)
#   Resumable: re-run after a throttle/interrupt and it skips what's done.
python scripts/mk/08_build_en_documents.py --max-mt-docs 2000
#   → data/processed/en_documents.jsonl

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
