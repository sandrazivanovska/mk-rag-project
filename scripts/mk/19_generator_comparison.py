"""
Step 19 — Retrieval quality x Generator strength interaction.

This is the analysis the thesis title actually promises. Two factors:

  * RETRIEVAL quality — manipulated across 6 pipelines, and per question by
    whether the gold article was retrieved (hit_at_5_doc).
  * GENERATOR strength — gemini-2.5-flash vs the weaker gemini-2.5-flash-lite,
    replayed over IDENTICAL contexts, so retrieval is held exactly constant.

The question that matters: does a stronger generator RESCUE bad retrieval? If
the gap between generators is small when retrieval succeeds but large when it
fails, generator strength compensates. If the gap is flat, retrieval dominates
regardless of model — which is the stronger claim for a low-resource setting.

Uses token_f1 / exact_match / context_coverage, which are computed locally for
all 300 questions per pipeline (n is ~1800 per generator) rather than the
smaller LLM-judged RAGAS subset.

    python scripts/mk/19_generator_comparison.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def load(d: Path, key_fix=lambda s: s):
    out = {}
    for f in sorted(glob.glob(str(d / "*.jsonl"))):
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        for r in rows:
            pid = key_fix(r.get("pipeline_id", Path(f).stem))
            out.setdefault(pid, {})[r["query"]] = r
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", type=Path, default=Path("results/full"))
    ap.add_argument("--gen2-dir", type=Path, default=Path("results/gen2"))
    ap.add_argument("--metric", default="token_f1")
    args = ap.parse_args()

    strip = lambda s: s.replace("_gemini_flash", "").replace("_gemini_pro", "")
    a = load(args.base_dir, strip)
    b = load(args.gen2_dir, strip)
    if not b:
        print(f"No second-generator results in {args.gen2_dir} yet.")
        return 1

    pipes = sorted(set(a) & set(b))
    print(f"metric: {args.metric}   pipelines: {len(pipes)}\n")
    print(f"{'pipeline':24} {'flash':>8} {'lite':>8} {'diff':>8} {'p':>10} {'n':>6}")
    print("-" * 68)

    all_d, all_hit, all_miss = [], [], []
    for p in pipes:
        qs = [q for q in a[p] if q in b[p]]
        va = np.array([a[p][q].get(args.metric) for q in qs], dtype=float)
        vb = np.array([b[p][q].get(args.metric) for q in qs], dtype=float)
        ok = ~(np.isnan(va) | np.isnan(vb))
        va, vb = va[ok], vb[ok]
        d = va - vb
        pv = stats.wilcoxon(va, vb).pvalue if not np.allclose(d, 0) else 1.0
        all_d.append(d)
        print(f"{p:24} {va.mean():8.4f} {vb.mean():8.4f} {d.mean():+8.4f} {pv:10.2e} {len(va):6}")

        hits = np.array([a[p][q].get("hit_at_5_doc") for q in qs], dtype=float)[ok]
        all_hit.append(d[hits == 1.0])
        all_miss.append(d[hits == 0.0])

    d_all = np.concatenate(all_d)
    hit = np.concatenate([x for x in all_hit if len(x)])
    miss = np.concatenate([x for x in all_miss if len(x)])

    print("\n" + "=" * 68)
    print("DOES A STRONGER GENERATOR RESCUE BAD RETRIEVAL?")
    print("=" * 68)
    print(f"  generator gap when retrieval SUCCEEDED : {hit.mean():+.4f}  (n={len(hit)})")
    print(f"  generator gap when retrieval FAILED    : {miss.mean():+.4f}  (n={len(miss)})")
    if len(hit) > 5 and len(miss) > 5:
        u = stats.mannwhitneyu(hit, miss).pvalue
        print(f"  difference between those two gaps      : p={u:.2e}")
        print()
        if u < 0.05 and abs(miss.mean()) > abs(hit.mean()):
            print("  => The generator matters MORE when retrieval fails: model")
            print("     strength partially compensates for weak retrieval.")
        elif u < 0.05:
            print("  => The generator matters MORE when retrieval succeeds: a better")
            print("     model can only exploit context that retrieval actually found.")
        else:
            print("  => No significant interaction. Generator strength does NOT")
            print("     compensate for retrieval failure — retrieval quality")
            print("     dominates regardless of which model generates.")
    print(f"\n  overall generator effect: {d_all.mean():+.4f} "
          f"(flash - flash-lite, n={len(d_all)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
