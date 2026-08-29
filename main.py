#!/usr/bin/env python3
"""
mk-rag: Retrieval Quality vs Generation Quality in Low-Resource RAG
A Bilingual Study on Macedonian

CLI entry point.

Usage:
    python main.py setup-data
    python main.py build-indices
    python main.py run-experiment --pipeline mk_dense --generator claude_sonnet
    python main.py run-all
    python main.py evaluate --results-dir results/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="MK-RAG research pipeline CLI")


# ── Setup commands ─────────────────────────────────────────────────────────────


@app.command()
def setup_data(
    mk_dump: Optional[Path] = typer.Option(None, help="Path to MK Wikipedia dump (.xml.bz2)"),
    hf_corpus: bool = typer.Option(True, help="Download LVSTCK HuggingFace corpus subset"),
    max_hf_docs: int = typer.Option(200_000, help="Max LVSTCK documents to load"),
):
    """
    Step 1: Download and preprocess corpora.

    Extracts MK Wikipedia, optionally downloads the LVSTCK corpus,
    cleans, deduplicates, and chunks all documents.
    """
    from src.ingestion.wiki_extractor import extract_mk_wikipedia, extract_en_wikipedia, get_mk_en_wikidata_links
    from src.ingestion.corpus_loader import load_hf_corpus, merge_jsonl
    from src.ingestion.preprocessor import clean_text, deduplicate_chunks
    from src.ingestion.chunker import chunk_documents, ChunkingStrategy
    from src.utils.config import get_settings

    settings = get_settings()
    console.print("[bold green]Setting up data...[/bold green]")

    # MK Wikipedia
    if mk_dump and mk_dump.exists():
        console.print(f"Extracting MK Wikipedia from {mk_dump}")
        mk_wiki_jsonl = extract_mk_wikipedia(mk_dump, settings.processed_mk_path)
    else:
        console.print("[yellow]No MK dump provided – skipping Wikipedia extraction[/yellow]")
        mk_wiki_jsonl = None

    # LVSTCK corpus
    if hf_corpus:
        console.print("Loading LVSTCK Macedonian corpus (streaming)...")
        lvstck_jsonl = load_hf_corpus(settings.processed_mk_path, max_docs=max_hf_docs)

    # Merge MK sources
    sources = []
    if mk_wiki_jsonl:
        sources.append(mk_wiki_jsonl)
    if hf_corpus:
        sources.append(Path(settings.processed_mk_path) / "lvstck_mk.jsonl")

    if sources:
        merged_mk = merge_jsonl(sources, Path(settings.processed_mk_path) / "mk_merged.jsonl")

        # Clean + chunk
        import jsonlines
        with jsonlines.open(merged_mk) as reader:
            docs = [
                {**doc, "text": clean_text(doc["text"])}
                for doc in reader
                if doc.get("text", "").strip()
            ]

        docs = list(deduplicate_chunks(docs, key="text"))
        chunks = chunk_documents(
            docs,
            strategy=ChunkingStrategy(settings.chunking_strategy),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        out_path = Path(settings.processed_mk_path) / "chunks.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        console.print(f"[green]MK chunks saved → {out_path}[/green]")

    console.print("[bold green]✓ Data setup complete.[/bold green]")


@app.command()
def build_indices(
    lang: str = typer.Option("mk", help="Language to index: mk | en | both"),
):
    """
    Step 2: Build FAISS and BM25 indices.

    Must be run after setup-data.
    """
    from src.retrieval.index_builder import build_faiss_index, build_bm25_index
    from src.utils.config import get_settings

    settings = get_settings()

    if lang in ("mk", "both"):
        console.print("[bold]Building MK FAISS index (BGE-M3)...[/bold]")
        build_faiss_index(
            Path(settings.processed_mk_path) / "chunks.jsonl",
            settings.faiss_index_mk,
            model_name=settings.embed_model_primary,
        )
        console.print("[bold]Building MK BM25 index...[/bold]")
        build_bm25_index(
            Path(settings.processed_mk_path) / "chunks.jsonl",
            Path(settings.bm25_index_mk) / "mk_bm25.pkl",
        )

    if lang in ("en", "both"):
        console.print("[bold]Building EN FAISS index (BGE-M3)...[/bold]")
        build_faiss_index(
            Path(settings.processed_en_path) / "chunks.jsonl",
            settings.faiss_index_en,
            model_name=settings.embed_model_primary,
        )

    console.print("[bold green]✓ Indices built.[/bold green]")


# ── Experiment commands ────────────────────────────────────────────────────────


@app.command()
def run_experiment(
    pipeline: str = typer.Option(..., help="Pipeline ID: mk_bm25|mk_dense|mk_hybrid|translate_retrieve|cross_lingual_embed|bilingual_fusion"),
    generator: str = typer.Option(..., help="Generator ID: gpt4o|claude_sonnet"),
    gold_path: Optional[Path] = typer.Option(None, help="Path to gold JSONL dataset"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory for results"),
    judge_sample: Optional[int] = typer.Option(
        None, help="Score RAGAS on only N questions (0 = all)."
    ),
):
    """
    Step 3: Run a single pipeline experiment.
    """
    from src.pipelines.factory import build_pipeline
    from src.evaluation.evaluator import RAGEvaluator
    from src.utils.config import get_settings

    settings = get_settings()
    out_dir = output_dir or Path(settings.experiment_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if judge_sample is None:
        judge_sample = settings.ragas_sample
    elif judge_sample == 0:
        judge_sample = None

    console.print(f"[bold]Running: {pipeline} + {generator}[/bold]")

    # Load queries from gold file
    queries, gold_data = [], []
    if gold_path and gold_path.exists():
        with open(gold_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                queries.append(item["query"])
                gold_data.append(item)
    else:
        console.print("[yellow]No gold dataset – using sample queries[/yellow]")
        queries = [
            "Кој е главниот град на Македонија?",
            "Кога е прогласена независноста на Македонија?",
            "Кој го напишал делото Македонска крвава свадба?",
        ]

    # Build and run pipeline
    rag_pipeline = build_pipeline(pipeline, generator)
    results_raw = rag_pipeline.run_batch(queries)

    # Evaluate
    evaluator = RAGEvaluator(
            ragas_llm_model=settings.judge_model,
            judge_provider=settings.judge_provider,
            judge_embed_model=settings.judge_embed_model,
            gemini_base_url=settings.gemini_openai_base_url,
            use_vertex=settings.use_vertex,
            vertex_project=settings.vertex_project,
            vertex_location=settings.vertex_location,
            ragas_sample=judge_sample,
            ragas_seed=settings.ragas_seed,
        )
    predictions = [r.generation for r in results_raw]
    retrieved_docs_list = [r.retrieved_docs for r in results_raw]

    eval_results = evaluator.evaluate(
        pipeline_id=f"{pipeline}_{generator}",
        generator_id=generator,
        predictions=predictions,
        gold_data=gold_data if gold_data else None,
        retrieved_docs_list=retrieved_docs_list,
    )

    # Save
    out_file = out_dir / f"{pipeline}_{generator}.jsonl"
    evaluator.save_results(eval_results, out_file)

    # Print summary
    summary = evaluator.aggregate(eval_results)
    _print_summary_table([summary])
    console.print(f"[green]Results → {out_file}[/green]")


@app.command()
def run_all(
    gold_path: Optional[Path] = typer.Option(None, help="Path to gold JSONL dataset"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory for results"),
    limit: Optional[int] = typer.Option(None, help="Only run the first N gold questions"),
    generators: Optional[str] = typer.Option(
        None, help="Comma-separated generator IDs, e.g. 'gemini_flash'"
    ),
    judge_sample: Optional[int] = typer.Option(
        None, help="Score RAGAS on only N questions per pipeline (0 = all). "
                   "RAGAS dominates cost; free local metrics still cover all."
    ),
    resume: bool = typer.Option(
        True, help="Skip pipelines whose result file already exists."
    ),
):
    """Run every retrieval setup × generator and save results."""
    from src.pipelines.factory import build_all_pipelines
    from src.evaluation.evaluator import RAGEvaluator
    from src.utils.config import get_settings

    settings = get_settings()
    out_dir = output_dir or Path(settings.experiment_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if judge_sample is None:
        judge_sample = settings.ragas_sample
    elif judge_sample == 0:
        judge_sample = None

    gids = [g.strip() for g in generators.split(",")] if generators else None

    queries, gold_data = [], []
    if gold_path and gold_path.exists():
        with open(gold_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                queries.append(item["query"])
                gold_data.append(item)
        if limit:
            queries, gold_data = queries[:limit], gold_data[:limit]
            console.print(f"[yellow]Limited to first {len(queries)} questions[/yellow]")
    else:
        console.print("[yellow]No gold dataset – using sample queries[/yellow]")
        queries = [
            "Кој е главниот град на Македонија?",
            "Кога е прогласена независноста на Македонија?",
        ]

    all_summaries = []
    for pipeline in build_all_pipelines(settings=settings, generator_ids=gids):
        pid = pipeline.config.pipeline_id
        out_file = out_dir / f"{pid}.jsonl"
        if resume and out_file.exists():
            # Each pipeline costs real money and hours; never redo one that
            # already produced results just because a later pipeline crashed.
            console.print(f"[dim]Skipping {pid} (results exist)[/dim]")
            continue
        console.print(f"[bold]Running {pid}...[/bold]")
        try:
            raw = pipeline.run_batch(queries)
        except Exception as exc:
            # One broken pipeline must not discard the other eleven.
            console.print(f"[red]FAILED {pid}: {exc}[/red]")
            logging.getLogger("main").exception("pipeline %s failed", pid)
            continue
        evaluator = RAGEvaluator(
            ragas_llm_model=settings.judge_model,
            judge_provider=settings.judge_provider,
            judge_embed_model=settings.judge_embed_model,
            gemini_base_url=settings.gemini_openai_base_url,
            use_vertex=settings.use_vertex,
            vertex_project=settings.vertex_project,
            vertex_location=settings.vertex_location,
            ragas_sample=judge_sample,
            ragas_seed=settings.ragas_seed,
        )
        eval_results = evaluator.evaluate(
            pipeline_id=pid,
            generator_id=pipeline.config.generator_id,
            predictions=[r.generation for r in raw],
            gold_data=gold_data or None,
            retrieved_docs_list=[r.retrieved_docs for r in raw],
        )
        evaluator.save_results(eval_results, out_dir / f"{pid}.jsonl")
        all_summaries.append(evaluator.aggregate(eval_results))

    _print_summary_table(all_summaries)

    # Save overall summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    console.print(f"[green]Summary → {summary_path}[/green]")


@app.command()
def evaluate(
    results_dir: Path = typer.Option(Path("results/"), help="Directory with JSONL results"),
):
    """Re-run evaluation on existing results and print a comparison table."""
    import glob

    summaries = []
    for jsonl_file in sorted(glob.glob(str(results_dir / "*.jsonl"))):
        from src.evaluation.evaluator import RAGEvaluator, EvaluationResult

        results = []
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                results.append(EvaluationResult(**d))

        if results:
            evaluator = RAGEvaluator(use_ragas=False)
            summaries.append(evaluator.aggregate(results))

    _print_summary_table(summaries)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_summary_table(summaries: list[dict]) -> None:
    table = Table(title="Pipeline Comparison", show_lines=True)
    table.add_column("Pipeline", style="cyan")
    table.add_column("Generator", style="magenta")
    # Retrieval quality
    table.add_column("Hit@5", style="green")
    table.add_column("MRR", style="green")
    table.add_column("Hit@5\ndoc", style="bold green")
    table.add_column("MRR\ndoc", style="bold green")
    table.add_column("n_ret", style="dim")
    # Generation quality
    table.add_column("Faithfulness")
    table.add_column("Ans. Relevancy")
    table.add_column("Ctx. Precision")
    table.add_column("Ctx. Recall")
    table.add_column("Token F1")
    table.add_column("EM")

    for s in summaries:
        def fmt(v):
            return f"{v:.3f}" if v is not None else "-"

        table.add_row(
            s.get("pipeline_id", "-"),
            s.get("generator_id", "-"),
            fmt(s.get("hit_at_5")),
            fmt(s.get("mrr")),
            fmt(s.get("hit_at_5_doc")),
            fmt(s.get("mrr_doc")),
            str(s.get("n_retrieval_scored", "-")),
            fmt(s.get("faithfulness")),
            fmt(s.get("answer_relevancy")),
            fmt(s.get("context_precision")),
            fmt(s.get("context_recall")),
            fmt(s.get("token_f1")),
            fmt(s.get("exact_match")),
        )

    console.print(table)


if __name__ == "__main__":
    app()
