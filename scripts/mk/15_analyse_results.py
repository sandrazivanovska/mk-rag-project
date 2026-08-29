"""
Step 15 — Summarise and sanity-check the experiment results.

    python scripts/mk/15_analyse_results.py
    python scripts/mk/15_analyse_results.py --results-dir results/full

Prints three things:
  1. A pipeline comparison table (retrieval + generation quality).
  2. The MK-vs-EN headline comparison, on DOCUMENT-level retrieval.
  3. Validity warnings — things that would make a number misleading if quoted
     without qualification.

Read the warnings before writing any of these numbers into the thesis.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, ".")

# Which language each retrieval setup actually searches in.
MK_PIPELINES = {"mk_bm25", "mk_dense", "mk_hybrid"}
EN_PIPELINES = {"translate_retrieve", "cross_lingual_embed"}
MIXED_PIPELINES = {"bilingual_fusion"}


def base_pipeline(pid: str) -> str:
    for sep in ("_gemini", "_gpt", "_claude"):
        if sep in pid:
            return pid.split(sep)[0]
    return pid


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "  -  "


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/full"))
    args = ap.parse_args()

    files = sorted(p for p in args.results_dir.glob("*.jsonl"))
    if not files:
        print(f"No result files in {args.results_dir}. Has the run finished?")
        return 1

    rows_by_pipeline = {}
    for f in files:
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        if rows:
            rows_by_pipeline[f.stem] = rows

    print(f"Loaded {len(rows_by_pipeline)} pipeline result files "
          f"({sum(len(v) for v in rows_by_pipeline.values())} rows)\n")

    # ── 1. Comparison table ────────────────────────────────────────────────────
    hdr = (f"{'pipeline':34} {'Hit@5':>7} {'MRR':>7} {'Hit@5':>7} {'MRR':>7} "
           f"{'faith':>7} {'ansRel':>7} {'tokF1':>7} {'n':>5} {'nRAG':>5}")
    print(hdr)
    print(f"{'':34} {'chunk':>7} {'chunk':>7} {'DOC':>7} {'DOC':>7}")
    print("-" * len(hdr))

    table = {}
    for pid, rows in sorted(rows_by_pipeline.items()):
        agg = {
            "hit5": mean(r.get("hit_at_5") for r in rows),
            "mrr": mean(r.get("mrr") for r in rows),
            "hit5d": mean(r.get("hit_at_5_doc") for r in rows),
            "mrrd": mean(r.get("mrr_doc") for r in rows),
            "faith": mean(r.get("faithfulness") for r in rows),
            "arel": mean(r.get("answer_relevancy") for r in rows),
            "tf1": mean(r.get("token_f1") for r in rows),
            "n": len(rows),
            "nrag": sum(1 for r in rows if r.get("faithfulness") is not None),
            "nret": sum(1 for r in rows if r.get("mrr_doc") is not None),
        }
        table[pid] = agg
        print(f"{pid:34} {fmt(agg['hit5']):>7} {fmt(agg['mrr']):>7} "
              f"{fmt(agg['hit5d']):>7} {fmt(agg['mrrd']):>7} "
              f"{fmt(agg['faith']):>7} {fmt(agg['arel']):>7} {fmt(agg['tf1']):>7} "
              f"{agg['n']:>5} {agg['nrag']:>5}")

    # ── 2. MK vs EN headline ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("MK vs EN retrieval  (DOCUMENT level — the fair basis; see warnings)")
    print("=" * 72)
    for label, group in (("Macedonian index", MK_PIPELINES),
                         ("English index", EN_PIPELINES),
                         ("Bilingual fusion", MIXED_PIPELINES)):
        vals_h, vals_m = [], []
        for pid, rows in rows_by_pipeline.items():
            if base_pipeline(pid) in group:
                vals_h += [r.get("hit_at_5_doc") for r in rows]
                vals_m += [r.get("mrr_doc") for r in rows]
        print(f"  {label:20} Hit@5={fmt(mean(vals_h))}  MRR={fmt(mean(vals_m))}")

    # ── 3. Validity warnings ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("VALIDITY WARNINGS")
    print("=" * 72)
    warn = []

    for pid, agg in table.items():
        if agg["nret"] == 0:
            warn.append(f"{pid}: NO rows scored for retrieval — gold ids are in the "
                        f"wrong namespace for this pipeline.")
        elif agg["nret"] < agg["n"] * 0.9:
            warn.append(f"{pid}: only {agg['nret']}/{agg['n']} rows scoreable for "
                        f"retrieval (questions lacking an EN counterpart are excluded, "
                        f"not counted as misses).")
        if agg["nrag"] == 0:
            warn.append(f"{pid}: RAGAS produced NOTHING — the judge failed silently. "
                        f"Do not report generation metrics for this pipeline.")

    spread = [a["mrrd"] for a in table.values() if a["mrrd"] is not None]
    if spread and (max(spread) - min(spread)) < 0.05:
        warn.append(
            f"All pipelines fall within {max(spread) - min(spread):.3f} MRR of each "
            f"other. Suspect the GOLD SET before concluding the systems are "
            f"equivalent: 65% of questions are templated and 93% of answers are "
            f"copied verbatim from their source chunk, which limits how much any "
            f"retriever can be distinguished.")

    chunk_vs_doc = [(a["mrrd"] or 0) - (a["mrr"] or 0) for a in table.values()]
    if chunk_vs_doc and mean(chunk_vs_doc) > 0.2:
        warn.append(
            "Document-level scores are far above chunk-level. That is expected: "
            "retrievers often return the right ARTICLE but a neighbouring chunk. "
            "Quote document-level as the headline and chunk-level as a stricter "
            "secondary measure — and say which is which.")

    warn.append("EN relevance is known only at ARTICLE level, so EN pipelines are "
                "scored against every chunk of the counterpart article. At chunk "
                "level this flatters EN; document level avoids it.")
    warn.append("MK searches 20,408 chunks, EN searches 51,109. Part of any gap is "
                "candidate-pool size, not language. State this as a limitation.")

    for i, w in enumerate(warn, 1):
        print(f"\n  {i}. {w}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
