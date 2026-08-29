# Setup and Run Guide

End-to-end instructions for reproducing the experiment, including the
non-obvious failures encountered along the way and how each was resolved.

---

## 1. Dependencies

Python 3.12. Install from `requirements.txt`, then apply these pins — each one
matters:

```bash
pip install -r requirements.txt
pip install "ragas==0.1.21" "transformers==4.44.2" "protobuf>=5.27" "google-genai"
```

**`ragas` must be pinned to 0.1.21.** `requirements.txt` says `>=0.1.9`, which
resolves to 0.4.x. That version hard-imports
`langchain_community.chat_models.vertexai` (removed upstream) and its API
changed. Worse, `RAGEvaluator._run_ragas` wraps everything in a bare `except`
that only logs a warning — so a broken RAGAS produces **null metrics for every
pipeline while the run appears to succeed**. Verify with a direct import:

```bash
python -c "from ragas import evaluate; from ragas.metrics import faithfulness; print('ok')"
```

**Never install `langchain-google-genai` or `google-cloud-translate`.** Both
pin `protobuf<5`, while `transformers` needs `>=5.27`. Installing either
downgrades protobuf and breaks `FlagEmbedding`, killing all dense retrieval,
with an error that points at transformers rather than at the real cause. The
Gemini judge uses Google's OpenAI-compatible endpoint instead, and translation
uses the REST API directly — neither needs those packages.

Check the stack is intact at any time:

```bash
python -c "import FlagEmbedding, transformers; print('embedding stack ok')"
```

## 2. Credentials

The Gemini Developer API (`generativelanguage.googleapis.com`) may reject your
API key with `401 ACCESS_TOKEN_TYPE_UNSUPPORTED` — this affects newer `AQ.`
prefixed keys and organisations that disable API-key auth. **Use Vertex AI**,
which authenticates with Google Cloud credentials and draws on Cloud credit:

```bash
gcloud auth application-default login          # tick ALL permission boxes
gcloud services enable aiplatform.googleapis.com translate.googleapis.com
```

Then in `.env`:

```ini
USE_VERTEX=true
VERTEX_PROJECT=your-project-id
VERTEX_LOCATION=us-central1
```

Note: `gcloud auth application-default login` sets **no environment variable**,
so code must detect ADC via `google.auth.default()` rather than checking
`GOOGLE_APPLICATION_CREDENTIALS`. User ADC also requires an
`x-goog-user-project` header on REST calls or the API returns 403.

Verify everything before spending money:

```bash
python scripts/mk/13_verify_gemini.py
```

This checks the key, lists reachable models, validates the configured model
names, performs a real Macedonian generation round-trip, and constructs the
judge.

## 3. Data

Requires `data/processed/{mk,en}/chunks.jsonl` and the FAISS indices. The BM25
index does **not** need transferring between machines — it rebuilds from
`chunks.jsonl` in about 8 seconds and is deterministic.

Build the cross-lingual gold set (adds English counterpart IDs, needed so
English-index pipelines can be scored at all):

```bash
python scripts/mk/14_build_crosslingual_gold.py --strict-doc-level
```

## 4. Run

```bash
python main.py run-all \
  --gold-path data/evaluation/gold_dataset_crosslingual.jsonl \
  --generators gemini_flash \
  --output-dir results/full
```

Useful flags: `--limit N` (first N questions), `--judge-sample N` (RAGAS on N
questions per pipeline, `0` = all), `--resume/--no-resume` (skip pipelines whose
results already exist — on by default).

**On Windows, launch detached** or the run dies when the parent shell exits:

```powershell
Start-Process cmd.exe -ArgumentList "/c","run_full.cmd" -WindowStyle Hidden
```

Also disable sleep (`powercfg /change standby-timeout-ac 0`) and, if the C:
drive is short on space, move the HuggingFace cache (`setx HF_HOME D:\hf-cache`).

## 5. Analysis

```bash
python scripts/mk/15_analyse_results.py       # comparison table + validity warnings
python scripts/mk/16_significance.py --metric hit_at_5_doc
python scripts/mk/17_rescore_ragas.py --target 150   # raise RAGAS n without re-running
python scripts/mk/18_second_generator.py --model gemini-2.5-flash-lite
python scripts/mk/19_generator_comparison.py  # retrieval × generator interaction
```

Scripts 17 and 18 both exploit the fact that saved rows keep `query`, `answer`,
`context`, and `reference_answer` — so RAGAS can be re-scored, and a different
generator replayed over identical contexts, **without re-running retrieval**.

## Cost

Measured on Vertex, August 2026:

| Item | Cost |
|---|---|
| Generation, 1,800 calls (`gemini-2.5-flash`) | ~$3 |
| RAGAS judging, 900 samples | ~$75 |
| Second generator replay, 1,800 calls (`flash-lite`) | ~$1 |

**RAGAS dominates**: ~270,000 prompt tokens per evaluated sample, of which
roughly 58,000 characters per call is the library's own few-shot boilerplate,
not your data. Control it with `--judge-sample` and by using `flash` rather than
`pro` as the judge (`pro` costs ~4× and would exceed a $100 budget by itself).

## Gotchas worth knowing

- **Thinking tokens count against `max_tokens`.** At the default 512,
  `gemini-2.5-pro` spent ~490 on reasoning and returned answers truncated
  mid-word with `finish_reason=MAX_TOKENS` — silently corrupting every
  generation metric. `GEN_MAX_TOKENS=4096`, and the generator now warns on
  truncation.
- **`gemini-2.5-pro` rejects `thinking_budget=0`.** The generator caches this
  per model so it does not retry on every call.
- **Retrieval metrics must use the full first-stage ranking**, not the reranked
  top-5. Scoring `final_docs` gave `mk_bm25` 50 candidates and every reranking
  pipeline only 5, making `recall@50` arithmetically impossible for them.
- **Document-level vs chunk-level.** Retrievers often return the right article
  but a neighbouring chunk. Chunk-level Hit@5 was 0.253 where document-level was
  0.523 for the same run — quote document-level as the headline.
- **`pydantic-settings` never touches `os.environ`**, so `.env` keys were
  invisible to every SDK until `get_settings()` began exporting them.
