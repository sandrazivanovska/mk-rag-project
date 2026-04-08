#!/usr/bin/env python3
"""
Visualize and export MK-RAG experiment results.

Usage:
    python scripts/visualize_results.py                        # load results/, show plots
    python scripts/visualize_results.py --results-dir results/ # explicit dir
    python scripts/visualize_results.py --latex                # print LaTeX table
    python scripts/visualize_results.py --save plots/          # save PNG figures

Produces:
    1. Heatmap: pipeline × metric (all 12 variants)
    2. Bar chart: Hit@1 comparison across pipelines
    3. Scatter: Retrieval quality (MRR) vs Generation quality (Token-F1)
    4. LaTeX table ready to paste into the thesis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── Data loading ─────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> list[dict]:
    """Load all *.jsonl result files and aggregate per pipeline+generator."""
    summaries = []
    for jsonl_file in sorted(results_dir.glob("*.jsonl")):
        records = []
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            continue

        # Aggregate means across queries
        keys = [k for k in records[0] if k not in ("query", "prediction", "reference",
                                                      "retrieved_doc_ids", "relevant_doc_ids")]
        summary = {k: records[0][k] for k in ("pipeline_id", "generator_id") if k in records[0]}
        for key in keys:
            if key in ("pipeline_id", "generator_id"):
                continue
            vals = [r[key] for r in records if r.get(key) is not None]
            summary[key] = sum(vals) / len(vals) if vals else None
        summaries.append(summary)

    return summaries


def load_chunking_results() -> list[dict]:
    """Load the BM25 chunking experiment results if present."""
    for path in [
        Path("results/bm25_chunking_experiment.json"),
        Path("results/chunking_experiment.json"),
    ]:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return []


# ── Formatting helpers ────────────────────────────────────────────────────────

METRIC_LABELS = {
    "hit_at_1":          "Hit@1",
    "hit_at_3":          "Hit@3",
    "hit_at_5":          "Hit@5",
    "mrr":               "MRR",
    "recall_at_5":       "Rec@5",
    "token_f1":          "Token-F1",
    "exact_match":       "Exact Match",
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Ans. Relevancy",
    "context_precision": "Ctx. Precision",
    "context_recall":    "Ctx. Recall",
    "answer_correctness":"Ans. Correctness",
}

PIPELINE_SHORT = {
    "mk_bm25":              "MK-BM25",
    "mk_dense":             "MK-Dense",
    "mk_hybrid":            "MK-Hybrid",
    "translate_retrieve":   "Trans-Retr",
    "cross_lingual_embed":  "Cross-Ling",
    "bilingual_fusion":     "Bilingual",
}


def _fmt(v, decimals: int = 3) -> str:
    if v is None:
        return "--"
    return f"{v:.{decimals}f}"


def _label(s: dict) -> str:
    pid = PIPELINE_SHORT.get(s.get("pipeline_id", ""), s.get("pipeline_id", ""))
    gen = "GPT" if "gpt" in s.get("generator_id", "").lower() else "Claude"
    return f"{pid}/{gen}"


# ── Terminal table ────────────────────────────────────────────────────────────

def print_terminal_table(summaries: list[dict]) -> None:
    metrics = ["hit_at_1", "mrr", "recall_at_5", "token_f1", "faithfulness", "answer_relevancy"]
    col_w = 11

    header = f"{'Pipeline/Gen':<22}" + "".join(f"{METRIC_LABELS[m]:>{col_w}}" for m in metrics)
    print("\n" + "=" * len(header))
    print("Pipeline Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for s in sorted(summaries, key=lambda x: (x.get("hit_at_1") or 0), reverse=True):
        label = _label(s)
        row = f"{label:<22}" + "".join(f"{_fmt(s.get(m)):>{col_w}}" for m in metrics)
        print(row)
    print("=" * len(header) + "\n")


# ── LaTeX table ───────────────────────────────────────────────────────────────

def print_latex_table(summaries: list[dict]) -> None:
    metrics = ["hit_at_1", "hit_at_3", "hit_at_5", "mrr", "recall_at_5",
               "token_f1", "faithfulness", "answer_relevancy"]
    m_labels = [METRIC_LABELS[m] for m in metrics]

    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{MK-RAG Pipeline Comparison on 50-Query Gold Dataset}")
    print(r"\label{tab:pipeline-comparison}")
    print(r"\small")
    print(r"\begin{tabular}{l" + "c" * len(metrics) + "}")
    print(r"\toprule")
    print("Pipeline & " + " & ".join(m_labels) + r" \\")
    print(r"\midrule")

    # Find best value per metric
    best = {}
    for m in metrics:
        vals = [s.get(m) for s in summaries if s.get(m) is not None]
        best[m] = max(vals) if vals else None

    for s in sorted(summaries, key=lambda x: (x.get("hit_at_1") or 0), reverse=True):
        label = _label(s).replace("_", r"\_")
        cells = []
        for m in metrics:
            v = s.get(m)
            if v is None:
                cells.append("--")
            elif best.get(m) is not None and abs(v - best[m]) < 1e-6:
                cells.append(r"\textbf{" + _fmt(v) + "}")
            else:
                cells.append(_fmt(v))
        print(label + " & " + " & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ── LaTeX chunking table ──────────────────────────────────────────────────────

def print_latex_chunking_table(chunking: list[dict]) -> None:
    if not chunking:
        print("% No chunking results found.")
        return

    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{BM25 Chunking Experiment Results}")
    print(r"\label{tab:chunking}")
    print(r"\begin{tabular}{llcccccc}")
    print(r"\toprule")
    print(r"Strategy & Chunk Size & Hit@1 & Hit@3 & Hit@5 & MRR & Rec@5 \\")
    print(r"\midrule")

    for row in chunking:
        strat = row.get("strategy", "")
        sz = row.get("chunk_size", "")
        h1 = _fmt(row.get("hit_at_1"))
        h3 = _fmt(row.get("hit_at_3"))
        h5 = _fmt(row.get("hit_at_5"))
        mrr = _fmt(row.get("mrr"))
        rec = _fmt(row.get("recall_at_5"))
        best = row.get("hit_at_1", 0) == 1.0 and row.get("mrr", 0) == 1.0
        line = f"{strat} & {sz} words & {h1} & {h3} & {h5} & {mrr} & {rec}"
        if best:
            line = r"\rowcolor{blue!10} " + line
        print(line + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(summaries: list[dict], save_dir: Path | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plots. Run: pip install matplotlib")
        return

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    labels = [_label(s) for s in summaries]
    ret_metrics = ["hit_at_1", "hit_at_3", "hit_at_5", "mrr", "recall_at_5"]
    gen_metrics = ["token_f1", "faithfulness", "answer_relevancy"]

    # ── Figure 1: Heatmap ─────────────────────────────────────────────────────
    all_metrics = ret_metrics + gen_metrics
    data = np.array([
        [s.get(m) if s.get(m) is not None else float("nan") for m in all_metrics]
        for s in summaries
    ])

    fig, ax = plt.subplots(figsize=(12, max(4, len(summaries) * 0.5 + 1)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_metrics)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in all_metrics], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(summaries)):
        for j in range(len(all_metrics)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.7 else "black")
    plt.colorbar(im, ax=ax, label="Score")
    ax.set_title("MK-RAG Pipeline Comparison Heatmap", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "heatmap.png", dpi=150)
        print(f"Saved: {save_dir / 'heatmap.png'}")
    else:
        plt.show()
    plt.close()

    # ── Figure 2: Hit@1 bar chart ─────────────────────────────────────────────
    hit1_vals = [s.get("hit_at_1") or 0 for s in summaries]
    colors = ["#2E75B6" if "GPT" in l else "#C55A11" for l in labels]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(labels)), hit1_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Hit@1")
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Hit@1 by Pipeline and Generator", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, hit1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom", fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2E75B6", label="GPT-4o"),
                        Patch(color="#C55A11", label="Claude Sonnet")], fontsize=8)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "hit1_bar.png", dpi=150)
        print(f"Saved: {save_dir / 'hit1_bar.png'}")
    else:
        plt.show()
    plt.close()

    # ── Figure 3: Retrieval (MRR) vs Generation (Token-F1) scatter ────────────
    mrr_vals   = [s.get("mrr") for s in summaries]
    tf1_vals   = [s.get("token_f1") for s in summaries]
    valid = [(l, m, t) for l, m, t in zip(labels, mrr_vals, tf1_vals) if m is not None and t is not None]

    if valid:
        vlabels, vmrr, vtf1 = zip(*valid)
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(vmrr, vtf1,
                        c=["#2E75B6" if "GPT" in l else "#C55A11" for l in vlabels],
                        s=80, zorder=3)
        for lbl, x, y in zip(vlabels, vmrr, vtf1):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax.set_xlabel("Retrieval Quality (MRR)", fontsize=10)
        ax.set_ylabel("Generation Quality (Token-F1)", fontsize=10)
        ax.set_title("Retrieval Quality vs Generation Quality", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / "scatter_ret_vs_gen.png", dpi=150)
            print(f"Saved: {save_dir / 'scatter_ret_vs_gen.png'}")
        else:
            plt.show()
        plt.close()


# ── Demo mode (no real results yet) ──────────────────────────────────────────

def _demo_summaries() -> list[dict]:
    """Generate plausible placeholder data for visualisation before real experiments run."""
    pipelines = ["mk_bm25", "mk_dense", "mk_hybrid", "translate_retrieve",
                 "cross_lingual_embed", "bilingual_fusion"]
    generators = ["gpt4o", "claude_sonnet"]
    import random
    random.seed(42)
    summaries = []
    for pid in pipelines:
        base = {
            "mk_bm25": 0.90, "mk_dense": 0.85, "mk_hybrid": 0.92,
            "translate_retrieve": 0.75, "cross_lingual_embed": 0.72, "bilingual_fusion": 0.88,
        }[pid]
        for gen in generators:
            gen_bonus = 0.03 if gen == "gpt4o" else 0.00
            summaries.append({
                "pipeline_id": pid,
                "generator_id": gen,
                "hit_at_1":          round(min(1.0, base + random.uniform(-0.05, 0.05)), 3),
                "hit_at_3":          round(min(1.0, base + 0.05 + random.uniform(0, 0.05)), 3),
                "hit_at_5":          round(min(1.0, base + 0.08 + random.uniform(0, 0.03)), 3),
                "mrr":               round(min(1.0, base + random.uniform(-0.03, 0.03)), 3),
                "recall_at_5":       round(min(1.0, base - 0.05 + random.uniform(0, 0.1)), 3),
                "token_f1":          round(base * 0.85 + gen_bonus + random.uniform(-0.04, 0.04), 3),
                "faithfulness":      round(0.7 + gen_bonus + random.uniform(-0.1, 0.1), 3),
                "answer_relevancy":  round(0.75 + gen_bonus + random.uniform(-0.08, 0.08), 3),
                "context_precision": None,
                "context_recall":    None,
                "answer_correctness":None,
            })
    return summaries


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MK-RAG experiment results")
    parser.add_argument("--results-dir", type=Path, default=Path("results/"))
    parser.add_argument("--save", type=Path, default=None, metavar="DIR",
                        help="Save figures to DIR instead of showing interactively")
    parser.add_argument("--latex", action="store_true",
                        help="Print LaTeX tables for thesis")
    parser.add_argument("--demo", action="store_true",
                        help="Use placeholder data (before real experiments run)")
    args = parser.parse_args()

    # Load data
    if args.demo:
        summaries = _demo_summaries()
        print("[DEMO MODE — using placeholder data]")
    else:
        summaries = load_results(args.results_dir)

    if not summaries:
        print(f"No result files found in {args.results_dir}.")
        print("Run experiments first: make run-all")
        print("Or try demo mode: python scripts/visualize_results.py --demo")
        sys.exit(0)

    if args.latex:
        print("\n% ── Pipeline comparison table ────────────────────────────────")
        print_latex_table(summaries)
        chunking = load_chunking_results()
        if chunking:
            print("\n% ── Chunking experiment table ────────────────────────────────")
            print_latex_chunking_table(chunking)
    else:
        print_terminal_table(summaries)
        make_plots(summaries, save_dir=args.save)


if __name__ == "__main__":
    main()
