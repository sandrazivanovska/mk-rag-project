"""
Step 17 — Raise RAGAS coverage on results that already exist.

Generation and retrieval are the slow, expensive stages and they are already
done: every saved row keeps query / answer / context / reference_answer, which
is everything RAGAS needs. So statistical power on the GENERATION metrics can be
increased by re-scoring alone — no pipeline re-run, no new generation spend.

Design notes
------------
* The scored subset is IDENTICAL across pipelines (same seed, same indices), so
  comparisons stay paired and the significance tests remain valid.
* The existing 40 scored rows are a subset of the new target set and are NOT
  re-scored — you only pay for the increment.
* Files are written after every batch, so an interruption keeps finished work.

    python scripts/mk/17_rescore_ragas.py --target 150
    python scripts/mk/17_rescore_ragas.py --target 150 --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")

METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision",
               "context_recall", "answer_correctness"]


def target_indices(n_rows: int, target: int, seed: int) -> list[int]:
    """Existing seed-42 sample of 40, extended deterministically to `target`."""
    base = sorted(random.Random(seed).sample(range(n_rows), min(40, n_rows)))
    if target <= len(base):
        return base
    rest = [i for i in range(n_rows) if i not in set(base)]
    extra = random.Random(seed + 1).sample(rest, min(target - len(base), len(rest)))
    return sorted(base + extra)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/full"))
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from src.utils.config import get_settings
    from src.evaluation.evaluator import RAGEvaluator

    settings = get_settings()
    files = sorted(glob.glob(str(args.results_dir / "*.jsonl")))
    if not files:
        print(f"No result files in {args.results_dir}")
        return 1

    # Plan first, so cost is visible before anything is spent.
    plan, total_new = [], 0
    for f in files:
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        idx = target_indices(len(rows), args.target, args.seed)
        todo = [i for i in idx if rows[i].get("faithfulness") is None]
        plan.append((f, rows, idx, todo))
        total_new += len(todo)
        print(f"  {Path(f).stem:34} rows={len(rows)} target={len(idx)} "
              f"already={len(idx)-len(todo)} to_score={len(todo)}")

    per_sample = 269937 * 0.30 / 1e6 + 1085 * 2.50 / 1e6
    print(f"\n  new samples to score : {total_new}")
    print(f"  est. cost            : ${total_new * per_sample:.2f}"
          f"   (~{per_sample:.4f}/sample, measured)")
    if args.dry_run:
        print("\n  dry run — nothing scored.")
        return 0

    ev = RAGEvaluator(
        ragas_llm_model=settings.judge_model,
        judge_provider=settings.judge_provider,
        judge_embed_model=settings.judge_embed_model,
        gemini_base_url=settings.gemini_openai_base_url,
        use_vertex=settings.use_vertex,
        vertex_project=settings.vertex_project,
        vertex_location=settings.vertex_location,
        ragas_sample=None,          # score exactly what we hand it
    )

    from src.evaluation.evaluator import EvaluationResult

    for f, rows, idx, todo in plan:
        if not todo:
            print(f"\n{Path(f).stem}: nothing to do")
            continue
        print(f"\n{Path(f).stem}: scoring {len(todo)} rows in batches of {args.batch}")
        for start in range(0, len(todo), args.batch):
            chunk = todo[start:start + args.batch]
            stubs = [
                EvaluationResult(
                    pipeline_id=rows[i].get("pipeline_id", ""),
                    generator_id=rows[i].get("generator_id", ""),
                    query=rows[i]["query"],
                    answer=rows[i]["answer"] or "",
                    reference_answer=rows[i]["reference_answer"] or "",
                    context=rows[i]["context"] or "",
                    retrieved_ids=rows[i].get("retrieved_ids") or [],
                )
                for i in chunk
            ]
            ev._run_ragas(stubs)
            got = 0
            for i, s in zip(chunk, stubs):
                for k in METRIC_KEYS:
                    v = getattr(s, k)
                    if v is not None and v == v:
                        rows[i][k] = float(v)
                        got += 1
            # persist after every batch
            with open(f, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            done = start + len(chunk)
            print(f"    {done}/{len(todo)} rows  (+{got} metric values written)")

    print("\nDone. Re-run scripts/mk/15_analyse_results.py and 16_significance.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
