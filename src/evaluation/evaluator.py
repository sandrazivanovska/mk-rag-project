"""
End-to-end RAG evaluator.

Runs RAGAS metrics (faithfulness, answer_relevancy, context_precision,
context_recall, answer_correctness) plus the custom MK-specific metrics
defined in metrics.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.utils.logging import get_logger
from .metrics import mk_exact_match, mk_token_f1, context_coverage

logger = get_logger("evaluator")


@dataclass
class EvaluationResult:
    pipeline_id: str
    generator_id: str
    query: str
    answer: str
    reference_answer: str
    context: str
    retrieved_ids: list[str]

    # RAGAS metrics
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None

    # Custom MK metrics
    exact_match: Optional[float] = None
    token_f1: Optional[float] = None
    ctx_coverage: Optional[float] = None

    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pipeline_id": self.pipeline_id,
            "generator_id": self.generator_id,
            "query": self.query,
            "answer": self.answer,
            "reference_answer": self.reference_answer,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_correctness": self.answer_correctness,
            "exact_match": self.exact_match,
            "token_f1": self.token_f1,
            "context_coverage": self.ctx_coverage,
        }


class RAGEvaluator:
    """
    Evaluates one or more pipeline results against a gold dataset.

    Gold dataset format (JSONL):
        {"query": "...", "answer": "...", "relevant_doc_ids": ["id1", "id2"]}

    Example
    -------
    >>> evaluator = RAGEvaluator(ragas_llm_model="gpt-4o")
    >>> results = evaluator.evaluate(
    ...     pipeline_id="mk_dense",
    ...     generator_id="claude_sonnet",
    ...     predictions=generation_results,
    ...     gold_path="data/gold_dataset.jsonl",
    ... )
    >>> evaluator.save_results(results, "results/mk_dense_claude.jsonl")
    """

    def __init__(self, ragas_llm_model: str = "gpt-4o", use_ragas: bool = True):
        self.ragas_llm_model = ragas_llm_model
        self.use_ragas = use_ragas
        self._ragas_evaluator = None

    def evaluate(
        self,
        pipeline_id: str,
        generator_id: str,
        predictions: list,  # list[GenerationResult]
        gold_path: Optional[str | Path] = None,
        gold_data: Optional[list[dict]] = None,
        retrieved_docs_list: Optional[list[list]] = None,
    ) -> list[EvaluationResult]:
        """
        Evaluate a list of generation results.

        Args:
            pipeline_id: Identifier for the retrieval pipeline.
            generator_id: Identifier for the generator.
            predictions: List of GenerationResult objects.
            gold_path: Path to gold JSONL file.
            gold_data: Pre-loaded gold data list (alternative to gold_path).
            retrieved_docs_list: List of retrieved doc lists per query.

        Returns:
            List of EvaluationResult objects.
        """
        # Load gold data
        if gold_data is None and gold_path:
            gold_data = self._load_gold(gold_path)

        results: list[EvaluationResult] = []

        for i, pred in enumerate(tqdm(predictions, desc="Evaluating")):
            gold = gold_data[i] if gold_data else {}
            reference = gold.get("answer", "")
            relevant_ids = set(gold.get("relevant_doc_ids", []))
            retrieved_ids = (
                [d.chunk_id for d in retrieved_docs_list[i]]
                if retrieved_docs_list
                else []
            )

            result = EvaluationResult(
                pipeline_id=pipeline_id,
                generator_id=generator_id,
                query=pred.query,
                answer=pred.answer,
                reference_answer=reference,
                context=pred.context,
                retrieved_ids=retrieved_ids,
                exact_match=mk_exact_match(pred.answer, reference) if reference else None,
                token_f1=mk_token_f1(pred.answer, reference) if reference else None,
                ctx_coverage=context_coverage(pred.answer, pred.context),
            )
            results.append(result)

        if self.use_ragas and results:
            self._run_ragas(results)

        return results

    def _run_ragas(self, results: list[EvaluationResult]) -> None:
        """Compute RAGAS metrics and attach them to results in place."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            )
            from datasets import Dataset
            from langchain_openai import ChatOpenAI

            ragas_data = {
                "question": [r.query for r in results],
                "answer": [r.answer for r in results],
                "contexts": [[r.context] for r in results],
                "ground_truth": [r.reference_answer for r in results],
            }
            dataset = Dataset.from_dict(ragas_data)

            llm = ChatOpenAI(model=self.ragas_llm_model, temperature=0)

            scores = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    answer_correctness,
                ],
                llm=llm,
            )
            score_df = scores.to_pandas()

            for i, result in enumerate(results):
                row = score_df.iloc[i]
                result.faithfulness = row.get("faithfulness")
                result.answer_relevancy = row.get("answer_relevancy")
                result.context_precision = row.get("context_precision")
                result.context_recall = row.get("context_recall")
                result.answer_correctness = row.get("answer_correctness")

        except Exception as exc:
            logger.warning(f"RAGAS evaluation failed: {exc}")

    def aggregate(self, results: list[EvaluationResult]) -> dict:
        """Compute mean scores across all results."""
        import numpy as np

        def _mean(key: str) -> Optional[float]:
            vals = [getattr(r, key) for r in results if getattr(r, key) is not None]
            return float(np.mean(vals)) if vals else None

        return {
            "pipeline_id": results[0].pipeline_id if results else "",
            "generator_id": results[0].generator_id if results else "",
            "n": len(results),
            "faithfulness": _mean("faithfulness"),
            "answer_relevancy": _mean("answer_relevancy"),
            "context_precision": _mean("context_precision"),
            "context_recall": _mean("context_recall"),
            "answer_correctness": _mean("answer_correctness"),
            "exact_match": _mean("exact_match"),
            "token_f1": _mean("token_f1"),
            "context_coverage": _mean("ctx_coverage"),
        }

    def save_results(
        self,
        results: list[EvaluationResult],
        output_path: str | Path,
    ) -> None:
        """Save results to JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(results)} results → {output_path}")

    def _load_gold(self, path: str | Path) -> list[dict]:
        gold = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    gold.append(json.loads(line))
        return gold
