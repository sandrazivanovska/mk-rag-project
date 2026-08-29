"""
End-to-end RAG evaluator.

Runs RAGAS metrics (faithfulness, answer_relevancy, context_precision,
context_recall, answer_correctness) plus the custom MK-specific metrics
defined in metrics.py.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.utils.logging import get_logger
from .metrics import (
    mk_exact_match,
    mk_token_f1,
    context_coverage,
    retrieval_hit_at_k,
    retrieval_mrr,
    retrieval_recall_at_k,
)

# Pipelines that retrieve from the ENGLISH index. Their retrieved ids live in the
# en_* namespace, so they must be scored against relevant_doc_ids_en, not the
# Macedonian ids. Scoring them against MK ids reports a false 0.0.
EN_RETRIEVAL_PIPELINES = {
    "translate_retrieve",
    "cross_lingual_embed",
}
# bilingual_fusion returns docs from BOTH indices, so either namespace can hit.
MIXED_RETRIEVAL_PIPELINES = {"bilingual_fusion"}

_CHUNK_SUFFIX = re.compile(r"_chunk_\d+$")


def _doc_of(chunk_id: str) -> str:
    """'mk_wiki_886_chunk_0003' -> 'mk_wiki_886'."""
    return _CHUNK_SUFFIX.sub("", chunk_id)


def _dedupe(items: list[str]) -> list[str]:
    """Drop repeats while preserving rank order."""
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

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

    # Retrieval metrics. None means "not scoreable for this row" (e.g. an EN
    # pipeline on a question whose article has no English counterpart) — which
    # aggregate() skips, rather than counting it as a miss.
    hit_at_5: Optional[float] = None
    hit_at_10: Optional[float] = None
    recall_at_50: Optional[float] = None
    mrr: Optional[float] = None

    # Document-level retrieval: does the ranking surface the right ARTICLE,
    # ignoring which chunk of it? Chunk-level scoring alone is misleading here —
    # a retriever that returns chunk_0008 of the correct article when the gold
    # is chunk_0006 scores 0, which understates real retrieval quality and
    # compresses the differences between pipelines. It is also the only fair
    # basis for MK-vs-EN, since EN relevance is known only at article level.
    hit_at_5_doc: Optional[float] = None
    mrr_doc: Optional[float] = None
    recall_at_50_doc: Optional[float] = None

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
            # Retrieval quality. Omitted here originally, which silently dropped
            # these from every saved result file even though they were computed.
            "hit_at_5": self.hit_at_5,
            "hit_at_10": self.hit_at_10,
            "recall_at_50": self.recall_at_50,
            "mrr": self.mrr,
            "hit_at_5_doc": self.hit_at_5_doc,
            "mrr_doc": self.mrr_doc,
            "recall_at_50_doc": self.recall_at_50_doc,
            # Kept so retrieval can be re-scored offline without a paid re-run.
            "retrieved_ids": self.retrieved_ids,
            "context": self.context,
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

    def __init__(
        self,
        ragas_llm_model: str = "gpt-4o",
        use_ragas: bool = True,
        judge_provider: str = "openai",
        judge_embed_model: Optional[str] = None,
        gemini_base_url: Optional[str] = None,
        use_vertex: bool = False,
        vertex_project: str = "",
        vertex_location: str = "us-central1",
        ragas_sample: Optional[int] = None,
        ragas_seed: int = 42,
    ):
        self.ragas_llm_model = ragas_llm_model
        self.use_ragas = use_ragas
        self.judge_provider = judge_provider
        self.judge_embed_model = judge_embed_model
        self.gemini_base_url = gemini_base_url
        self.use_vertex = use_vertex
        self.vertex_project = vertex_project
        self.vertex_location = vertex_location
        # RAGAS is ~270k prompt tokens per sample and dominates run cost, so it
        # can be scored on a seeded random subset while the free local metrics
        # (exact match, token F1, context coverage) still cover every question.
        self.ragas_sample = ragas_sample
        self.ragas_seed = ragas_seed
        self._ragas_evaluator = None

    def _gemini_openai_kwargs(self) -> dict:
        """
        Credentials for Gemini's OpenAI-compatible endpoint.

        Used instead of langchain-google-genai, which drags in the legacy
        google-generativeai SDK and its protobuf<5 pin — that conflicts with
        transformers>=protobuf 5.27 and breaks FlagEmbedding.
        """
        import os

        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("judge_provider='gemini' but no GOOGLE_API_KEY is set.")
        return {"api_key": key, "base_url": self.gemini_base_url}

    def _build_judge_llm(self):
        """Build the LangChain chat model RAGAS scores with."""
        from langchain_openai import ChatOpenAI

        if self.judge_provider != "gemini":
            return ChatOpenAI(model=self.ragas_llm_model, temperature=0)

        if self.use_vertex:
            from .gemini_judge import vertex_openai_credentials
            creds = vertex_openai_credentials(self.vertex_project, self.vertex_location)
            model = self.ragas_llm_model
            # Vertex's OpenAI layer namespaces Google models as "google/<model>".
            if not model.startswith("google/"):
                model = f"google/{model}"
            return ChatOpenAI(model=model, temperature=0, **creds)

        return ChatOpenAI(
            model=self.ragas_llm_model, temperature=0, **self._gemini_openai_kwargs()
        )

    def _build_judge_embeddings(self):
        """
        Build the embedding model RAGAS uses for answer_relevancy /
        answer_correctness. RAGAS falls back to OpenAI embeddings when this is
        not supplied, which fails on a Gemini-only key.
        """
        if self.judge_provider != "gemini":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings()

        from .gemini_judge import GeminiEmbeddings

        return GeminiEmbeddings(
            model=self.judge_embed_model or "text-embedding-004",
            use_vertex=self.use_vertex,
            project=self.vertex_project,
            location=self.vertex_location,
        )

    @staticmethod
    def _relevant_ids_for(pipeline_id: str, gold: dict) -> set[str]:
        """
        Pick the gold id set matching the namespace this pipeline retrieves in.

        MK pipelines score against Macedonian chunk ids; EN pipelines against
        the English counterpart ids added by
        ``scripts/mk/14_build_crosslingual_gold.py``; bilingual_fusion may hit
        either, so it gets the union.
        """
        mk = set(gold.get("relevant_doc_ids", []))
        en = set(gold.get("relevant_doc_ids_en", []))
        base = pipeline_id.split("_gemini")[0].split("_gpt")[0].split("_claude")[0]
        if base in EN_RETRIEVAL_PIPELINES:
            return en
        if base in MIXED_RETRIEVAL_PIPELINES:
            return mk | en
        return mk

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
            retrieved_ids = (
                [d.chunk_id for d in retrieved_docs_list[i]]
                if retrieved_docs_list
                else []
            )
            relevant_ids = self._relevant_ids_for(pipeline_id, gold)

            # Score retrieval only when we have both a ranking and a gold set.
            # An empty relevant set means this row is not scoreable for this
            # pipeline's language, which is different from a miss.
            if retrieved_ids and relevant_ids:
                hit5 = retrieval_hit_at_k(retrieved_ids, relevant_ids, 5)
                hit10 = retrieval_hit_at_k(retrieved_ids, relevant_ids, 10)
                rec50 = retrieval_recall_at_k(retrieved_ids, relevant_ids, 50)
                mrr_ = retrieval_mrr(retrieved_ids, relevant_ids)
                # Same ranking, collapsed to articles.
                ret_docs = _dedupe([_doc_of(c) for c in retrieved_ids])
                rel_docs = {_doc_of(c) for c in relevant_ids}
                hit5d = retrieval_hit_at_k(ret_docs, rel_docs, 5)
                mrrd = retrieval_mrr(ret_docs, rel_docs)
                rec50d = retrieval_recall_at_k(ret_docs, rel_docs, 50)
            else:
                hit5 = hit10 = rec50 = mrr_ = None
                hit5d = mrrd = rec50d = None

            result = EvaluationResult(
                hit_at_5=hit5,
                hit_at_10=hit10,
                recall_at_50=rec50,
                mrr=mrr_,
                hit_at_5_doc=hit5d,
                mrr_doc=mrrd,
                recall_at_50_doc=rec50d,
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
            scored = results
            if self.ragas_sample and self.ragas_sample < len(results):
                rng = random.Random(self.ragas_seed)
                # Same seed across pipelines, so every pipeline is judged on the
                # identical question subset and the comparison stays fair.
                idx = sorted(rng.sample(range(len(results)), self.ragas_sample))
                scored = [results[i] for i in idx]
                logger.info(
                    f"RAGAS on a {len(scored)}/{len(results)} sample "
                    f"(seed={self.ragas_seed}); other rows keep null RAGAS scores."
                )
            self._run_ragas(scored)

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

            ragas_data = {
                "question": [r.query for r in results],
                "answer": [r.answer for r in results],
                "contexts": [[r.context] for r in results],
                "ground_truth": [r.reference_answer for r in results],
            }
            dataset = Dataset.from_dict(ragas_data)

            llm = self._build_judge_llm()

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
                embeddings=self._build_judge_embeddings(),
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
            # Retrieval quality
            "hit_at_5": _mean("hit_at_5"),
            "hit_at_10": _mean("hit_at_10"),
            "recall_at_50": _mean("recall_at_50"),
            "mrr": _mean("mrr"),
            "hit_at_5_doc": _mean("hit_at_5_doc"),
            "mrr_doc": _mean("mrr_doc"),
            "recall_at_50_doc": _mean("recall_at_50_doc"),
            # How many rows were actually scoreable for retrieval — a low value
            # means the gold set lacked ids in this pipeline's namespace.
            "n_retrieval_scored": sum(1 for r in results if r.mrr is not None),
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
