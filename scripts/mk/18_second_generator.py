"""
Step 18 — Add a second generator by replaying the SAVED contexts.

Why this design
---------------
The thesis contrasts retrieval quality with GENERATION quality, but generation
was never manipulated: every result so far uses one model. Re-running whole
pipelines to add a second generator would repeat retrieval and reranking, which
is the slow part (~9s/query) and would also let retrieval vary between the two
arms.

Every saved row already stores the exact `context` handed to the generator, so
we can replay those contexts through a different model. Retrieval is then
IDENTICAL by construction, not merely similar, which isolates the generator as
the only changing variable — a cleaner design than re-running, and far cheaper.

Free local metrics (token_f1, exact_match, context_coverage) are computed for
every one of the 300 questions per pipeline, giving n=300 for the generator
comparison — more statistical power than the RAGAS subset, at no LLM-judge cost.

    python scripts/mk/18_second_generator.py --model gemini-2.5-flash-lite
    python scripts/mk/18_second_generator.py --model gemini-2.5-flash-lite --limit 5
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")


class _Doc:
    """Minimal stand-in so Generator.generate() can join saved context back."""
    def __init__(self, text: str):
        self.text = text


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/full"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/gen2"))
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--limit", type=int, default=None, help="only N rows per pipeline")
    args = ap.parse_args()

    from src.utils.config import get_settings
    from src.generator.generator import Generator, GeneratorConfig, ProviderType
    from src.evaluation.metrics import mk_exact_match, mk_token_f1, context_coverage

    settings = get_settings()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gen = Generator(GeneratorConfig(
        provider=ProviderType.GEMINI,
        model_id=args.model,
        max_tokens=settings.gen_max_tokens,
        use_vertex=settings.use_vertex,
        vertex_project=settings.vertex_project,
        vertex_location=settings.vertex_location,
    ))

    files = sorted(glob.glob(str(args.results_dir / "*.jsonl")))
    if not files:
        print(f"No results in {args.results_dir}")
        return 1

    for f in files:
        stem = Path(f).stem
        base = stem.replace("_gemini_flash", "").replace("_gemini_pro", "")
        out_path = args.out_dir / f"{base}_{args.model.replace('.', '')}.jsonl"

        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        if args.limit:
            rows = rows[: args.limit]

        done = 0
        if out_path.exists():                      # resume
            done = sum(1 for _ in open(out_path, encoding="utf-8"))
        if done >= len(rows):
            print(f"{base}: already complete ({done} rows)")
            continue

        print(f"\n{base}: replaying {len(rows)-done} contexts through {args.model}")
        t0 = time.time()
        with open(out_path, "a", encoding="utf-8") as out:
            for i, r in enumerate(rows):
                if i < done:
                    continue
                ctx = r.get("context") or ""
                res = gen.generate(query=r["query"], context_docs=[_Doc(ctx)])
                ans = res.answer
                ref = r.get("reference_answer") or ""
                out.write(json.dumps({
                    "pipeline_id": base,
                    "generator_id": args.model,
                    "query": r["query"],
                    "answer": ans,
                    "reference_answer": ref,
                    "context": ctx,
                    # retrieval is identical by construction — carried over so the
                    # hit/miss split can be reused without recomputation
                    "hit_at_5_doc": r.get("hit_at_5_doc"),
                    "mrr_doc": r.get("mrr_doc"),
                    "hit_at_5": r.get("hit_at_5"),
                    "token_f1": mk_token_f1(ans, ref) if ref else None,
                    "exact_match": mk_exact_match(ans, ref) if ref else None,
                    "context_coverage": context_coverage(ans, ctx),
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                }, ensure_ascii=False) + "\n")
                out.flush()
                if (i + 1) % 25 == 0:
                    el = time.time() - t0
                    print(f"  {i+1}/{len(rows)}  ({el/(i+1-done):.1f}s/row)")
        print(f"  → {out_path}")

    print("\nDone. Compare with scripts/mk/19_generator_comparison.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
