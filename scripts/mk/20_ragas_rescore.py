"""
Step 20 — Add / redo RAGAS scores for an already-generated results directory.

Pure scoring: no retrieval, no re-generation. Each input row must already carry
the fields RAGAS needs — `query`, `answer`, `context`, `reference_answer` — which
the pipeline runs and script 18 both save. Gemini-only (no OpenAI/Anthropic).

Two uses
--------
1. Complete the SECOND generator. Script 18 saved gen2 answers + context but
   deliberately skipped RAGAS to save judge cost. Score them with the SAME
   Gemini-Flash judge used for the first generator, so the two are comparable:

       python scripts/mk/20_ragas_rescore.py --results-dir results/gen2 \
           --judge-model gemini-2.5-flash

2. Circularity cross-check. Re-score the FIRST-generator (Flash) results with a
   DIFFERENT Gemini model (Pro) as judge, on a seeded subset, to show the
   ranking is not a Flash-judging-Flash artifact:

       python scripts/mk/20_ragas_rescore.py --results-dir results/full \
           --judge-model gemini-2.5-pro --limit 50 --out-suffix _projudge

Requires GOOGLE_API_KEY (or GEMINI_API_KEY) in the environment.
Run against the FULL results dir (rows keep `context`), NOT results_public/
(context is stripped there).
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Dir of *.jsonl rows with query/answer/context/reference_answer")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write scored files (default: --results-dir)")
    ap.add_argument("--judge-model", default="gemini-2.5-flash",
                    help="Gemini model used as the RAGAS judge")
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only N rows per file (seeded random subset)")
    ap.add_argument("--out-suffix", default="",
                    help="Suffix for output filenames, e.g. _projudge")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from src.utils.config import get_settings
    from src.evaluation.evaluator import RAGEvaluator, EvaluationResult

    settings = get_settings()
    out_dir = args.out_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(args.results_dir / "*.jsonl")))
    if not files:
        print(f"No *.jsonl in {args.results_dir}")
        return 1

    evaluator = RAGEvaluator(
        ragas_llm_model=args.judge_model,        # judge model id
        use_ragas=True,
        judge_provider="gemini",
        judge_embed_model=settings.judge_embed_model,
        gemini_base_url=settings.gemini_openai_base_url,
        use_vertex=settings.use_vertex,
        vertex_project=settings.vertex_project,
        vertex_location=settings.vertex_location,
    )

    summaries = []
    for f in files:
        stem = Path(f).stem
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]

        # Guard: RAGAS needs the context that was fed to the generator.
        missing_ctx = sum(1 for r in rows if not (r.get("context") or "").strip())
        if missing_ctx == len(rows):
            print(f"[skip] {stem}: no 'context' in any row — run against the FULL "
                  f"results dir, not results_public/.")
            continue

        if args.limit and args.limit < len(rows):
            rng = random.Random(args.seed)
            rows = rng.sample(rows, args.limit)

        results = [
            EvaluationResult(
                pipeline_id=r.get("pipeline_id", stem),
                generator_id=r.get("generator_id", args.judge_model),
                query=r.get("query", ""),
                answer=r.get("answer", ""),
                reference_answer=r.get("reference_answer", "") or "",
                context=r.get("context", "") or "",
                retrieved_ids=r.get("retrieved_ids", []) or [],
            )
            for r in rows
        ]

        print(f"\n{stem}: scoring {len(results)} rows with {args.judge_model} ...")
        evaluator._run_ragas(results)   # fills faithfulness/relevancy/precision/recall in place

        out_path = out_dir / f"{stem}{args.out_suffix}.jsonl"
        with open(out_path, "w", encoding="utf-8") as out:
            for orig, res in zip(rows, results):
                orig.update({
                    "faithfulness": res.faithfulness,
                    "answer_relevancy": res.answer_relevancy,
                    "context_precision": res.context_precision,
                    "context_recall": res.context_recall,
                    "answer_correctness": res.answer_correctness,
                    "judge_model": args.judge_model,
                })
                out.write(json.dumps(orig, ensure_ascii=False) + "\n")

        summ = evaluator.aggregate(results)
        summ["source_file"] = stem
        summaries.append(summ)
        print(f"  faith={summ.get('faithfulness')}  ans_rel={summ.get('answer_relevancy')}  "
              f"ctx_prec={summ.get('context_precision')}  ctx_rec={summ.get('context_recall')}")
        print(f"  → {out_path}")

    summary_path = out_dir / f"summary_rescore{args.out_suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\nSummary → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
