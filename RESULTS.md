# Results — Retrieval Quality and Generation Quality in Macedonian RAG

Experiment run: 29 August 2026. 6 retrieval pipelines × 300 Macedonian
questions, generated with `gemini-2.5-flash` on Vertex AI, judged with
`gemini-2.5-flash` (RAGAS on 150 questions per pipeline).

Raw per-question results: [`results_public/`](results_public/) — same as the
full outputs minus the `context` field (dropped for size; reconstructable from
`retrieved_ids`).

---

## Headline numbers

| Pipeline | Hit@5 (doc) | MRR (doc) | Faithfulness | Answer rel. | Token F1 |
|---|---|---|---|---|---|
| **mk_dense** | 0.983 | 0.970 | **0.967** | **0.629** | **0.326** |
| bilingual_fusion | **0.987** | **0.970** | 0.956 | 0.620 | 0.313 |
| mk_hybrid | 0.903 | 0.746 | 0.932 | 0.608 | 0.325 |
| translate_retrieve | 0.969 | 0.932 | 0.905 | 0.510 | 0.240 |
| cross_lingual_embed | 0.864 | 0.822 | 0.892 | 0.491 | 0.229 |
| mk_bm25 | 0.523 | 0.458 | 0.552 | 0.298 | 0.162 |

Retrieval metrics: n = 300 (294 for English-index pipelines, see below).
RAGAS metrics: n ≈ 142–150.

## Findings

### 1. Retrieval quality drives answer quality — then saturates

Holding the generator constant and splitting by whether the gold article was
retrieved:

| | Retrieval hit | Retrieval miss |
|---|---|---|
| Faithfulness | 0.921 | 0.400 |
| Answer relevancy | 0.571 | 0.083 |
| Token F1 | 0.288 | 0.086 |

Same model, same prompt — only the retrieved context differs.

But **among the five non-BM25 pipelines, no generation-quality difference is
statistically significant** (paired Wilcoxon, Holm-Bonferroni over 15
comparisons, n ≈ 150), despite Hit@5 ranging 0.864–0.987. Only the gap to BM25
is real: +0.34 to +0.42 faithfulness, p < 10⁻⁸.

**The relationship saturates.** Going from bad retrieval to good transforms
answers; going from good to marginally better does not.

### 2. Lexical retrieval fails on Macedonian

BM25 reaches Hit@5 of 0.523 against dense retrieval's 0.983 — it misses the
correct *article* nearly half the time. Consistent with Macedonian's rich
morphology defeating exact term matching.

### 3. Pivoting through English buys nothing

`mk_dense` (0.983) vs `translate_retrieve` (0.969) is **not significant**
(p = 0.57). `cross_lingual_embed` (0.864) is significantly worse. Bilingual
fusion tops retrieval at 0.987 but ties with plain dense (p = 0.66).

A multilingual embedding model handles Macedonian well enough that translation
infrastructure is not worth its cost. **The simplest architecture wins.**

### 4. The model fails safe

When retrieval failed, the generator answered "Не знам" (I don't know) **75%**
of the time; when retrieval succeeded, only **3%**. It refuses rather than
fabricating — and this is why faithfulness on retrieval misses is 0.40 rather
than near zero: an honest refusal is faithful to a context lacking the answer.

## Limitations

1. **Single generator.** `gemini-2.5-flash` throughout, so the retrieval→
   generation relationship is not verified across models. (A second generator
   comparison is in `scripts/mk/18_second_generator.py`.)
2. **Gold set.** 65% of questions are templated (28% *"…во дадениот извадок?"*,
   37% *"Кој важен податок…"*) and 93% of answers are copied verbatim from the
   source chunk. This limits how far any retriever can be distinguished and is
   the likely cause of the saturation in finding 1.
3. **Candidate-pool asymmetry.** MK searches 20,408 chunks, EN searches 51,109,
   for matched 10k-article corpora. Part of any MK-vs-EN gap is pool size.
4. **English relevance is article-level only.** The gold marks a specific MK
   *chunk*; for English we know only the counterpart *article*. Document-level
   metrics are therefore the fair basis for MK-vs-EN; chunk-level flatters EN.
5. **6 of 300 questions have no English counterpart** and are excluded from
   English-pipeline retrieval scoring (n = 294), not counted as misses.
6. **Reranking pool.** The cross-encoder scores the top 20 first-stage hits,
   not all 50, for tractability on the available GPU. Applied identically to
   every pipeline.
7. **Answer brevity.** The system prompt asks for one or two sentences. Without
   it models produced multi-hundred-word summaries against ~29-word gold
   answers, driving token-F1 toward zero for reasons unrelated to retrieval.
   Applied identically to every pipeline.

## Reproducing the analysis

```bash
python scripts/mk/15_analyse_results.py                       # comparison + warnings
python scripts/mk/16_significance.py --metric hit_at_5_doc    # paired tests
python scripts/mk/16_significance.py --metric faithfulness
```
