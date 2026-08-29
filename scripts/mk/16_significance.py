"""
Step 16 — Statistical significance for the pipeline comparison.

Every pipeline answered the SAME 300 questions, so comparisons are PAIRED.
That is much more powerful than treating the pipelines as independent samples,
and it is what lets you claim a difference is real rather than noise.

Reports, for each pair of pipelines:
  - the mean difference,
  - a 95% bootstrap confidence interval on that difference,
  - a Wilcoxon signed-rank p-value (paired, non-parametric — appropriate here
    because per-question metrics are 0/1 or heavily skewed, not normal),
  - Holm-Bonferroni correction across the 15 pairwise tests.

    python scripts/mk/16_significance.py --metric hit_at_5_doc
    python scripts/mk/16_significance.py --metric faithfulness
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats


def load(results_dir: Path) -> dict[str, dict[str, float]]:
    """pipeline -> {query: metric_value}"""
    out = {}
    for f in sorted(glob.glob(str(results_dir / "*.jsonl"))):
        name = Path(f).stem.replace("_gemini_flash", "").replace("_gemini_pro", "")
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        out[name] = rows
    return out


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/full"))
    ap.add_argument("--metric", default="hit_at_5_doc")
    args = ap.parse_args()

    data = load(args.results_dir)
    if not data:
        print(f"No results in {args.results_dir}")
        return 1

    # Align on query text so pipelines are compared question-by-question.
    per = {}
    for name, rows in data.items():
        per[name] = {r["query"]: r.get(args.metric) for r in rows}
    common = set.intersection(*(set(v) for v in per.values()))
    common = sorted(q for q in common
                    if all(per[n][q] is not None and per[n][q] == per[n][q] for n in per))

    print(f"metric: {args.metric}")
    print(f"pipelines: {len(per)}   paired questions usable: {len(common)}\n")

    means = {n: float(np.mean([per[n][q] for q in common])) for n in per}
    for n, v in sorted(means.items(), key=lambda kv: -kv[1]):
        print(f"  {n:26} {v:.4f}")

    print(f"\n{'comparison':46} {'diff':>8} {'95% CI':>18} {'p':>10} {'sig':>5}")
    print("-" * 92)

    results = []
    for a, b in combinations(sorted(per, key=lambda n: -means[n]), 2):
        va = np.array([per[a][q] for q in common], dtype=float)
        vb = np.array([per[b][q] for q in common], dtype=float)
        d = va - vb
        if np.allclose(d, 0):
            p = 1.0
        else:
            p = stats.wilcoxon(va, vb, zero_method="wilcox").pvalue
        lo, hi = bootstrap_ci(d)
        results.append((a, b, float(d.mean()), lo, hi, float(p)))

    # Holm-Bonferroni across all pairwise tests
    order = sorted(range(len(results)), key=lambda i: results[i][5])
    m = len(results)
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * results[i][5])
        prev = max(prev, val)
        adj[i] = prev

    for i, (a, b, diff, lo, hi, p) in enumerate(results):
        star = "***" if adj[i] < 0.001 else "**" if adj[i] < 0.01 else "*" if adj[i] < 0.05 else "ns"
        print(f"  {a[:21]:21} vs {b[:21]:21} {diff:+8.4f} [{lo:+.3f},{hi:+.3f}] {adj[i]:10.2e} {star:>5}")

    print("\n  p-values are Holm-Bonferroni corrected across all "
          f"{m} pairwise tests.")
    print("  *** p<0.001   ** p<0.01   * p<0.05   ns = not significant")
    print("  A CI that excludes 0 means the difference is real at the 95% level.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
