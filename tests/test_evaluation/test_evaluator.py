"""
Tests for the RAGEvaluator and EvaluationResult aggregation.
"""
from __future__ import annotations

import pytest

from src.evaluation.evaluator import RAGEvaluator, EvaluationResult
from src.retrieval.base import RetrievedDoc


def _make_doc(doc_id: str, text: str = "text") -> RetrievedDoc:
    return RetrievedDoc(chunk_id=doc_id, doc_id=doc_id, text=text, score=1.0, lang="mk", rank=0)


def _make_result(
    pipeline_id: str = "mk_bm25",
    generator_id: str = "gpt4o",
    answer: str = "Скопје е главниот град.",
    reference: str = "Скопје е главниот град на Македонија.",
    token_f1: float = 0.8,
    exact_match: float = 0.0,
) -> EvaluationResult:
    return EvaluationResult(
        pipeline_id=pipeline_id,
        generator_id=generator_id,
        query="Кој е главниот град?",
        answer=answer,
        reference_answer=reference,
        context="Скопје е главниот град на Македонија.",
        retrieved_ids=["mk_001"],
        token_f1=token_f1,
        exact_match=exact_match,
    )


# ── Aggregation ───────────────────────────────────────────────────────────────

def test_aggregate_means():
    evaluator = RAGEvaluator(use_ragas=False)
    results = [
        _make_result(token_f1=0.8),
        _make_result(token_f1=0.6),
        _make_result(token_f1=1.0),
    ]
    summary = evaluator.aggregate(results)
    expected = (0.8 + 0.6 + 1.0) / 3
    assert abs(summary["token_f1"] - expected) < 1e-6


def test_aggregate_empty_list():
    evaluator = RAGEvaluator(use_ragas=False)
    summary = evaluator.aggregate([])
    assert summary == {} or summary.get("token_f1") is None


def test_aggregate_preserves_pipeline_id():
    evaluator = RAGEvaluator(use_ragas=False)
    results = [_make_result(pipeline_id="mk_dense")]
    summary = evaluator.aggregate(results)
    assert summary.get("pipeline_id") == "mk_dense"


def test_aggregate_skips_none_metrics():
    """None values should be excluded from the mean, not treated as 0."""
    evaluator = RAGEvaluator(use_ragas=False)
    r1 = _make_result(token_f1=0.8)
    r2 = _make_result(token_f1=0.6)
    r2.faithfulness = None  # no RAGAS on r2
    r1.faithfulness = 0.9
    results = [r1, r2]
    summary = evaluator.aggregate(results)
    # faithfulness mean should only be over r1
    assert abs(summary["faithfulness"] - 0.9) < 1e-6


# ── EvaluationResult ──────────────────────────────────────────────────────────

def test_evaluation_result_fields():
    r = _make_result()
    assert r.pipeline_id == "mk_bm25"
    assert r.token_f1 == 0.8
    assert r.faithfulness is None  # RAGAS not computed


def test_evaluation_result_dict_serialisable():
    import json
    r = _make_result()
    d = r.to_dict()
    json.dumps(d)  # should not raise


# ── evaluate() without RAGAS ─────────────────────────────────────────────────

def test_evaluate_hit_on_correct_doc():
    """When retrieved doc matches relevant_doc_ids, hit@1 and MRR should be 1."""
    from unittest.mock import MagicMock
    from src.generator.generator import GenerationResult

    evaluator = RAGEvaluator(use_ragas=False)

    gold_data = [{
        "query": "Кој е главниот град?",
        "answer": "Скопје е главниот град на Македонија.",
        "relevant_doc_ids": ["mk_001"],
    }]

    # Wrap plain string in a mock GenerationResult
    mock_pred = MagicMock(spec=GenerationResult)
    mock_pred.answer = "Скопје е главниот град."
    mock_pred.query = "Кој е главниот град?"
    mock_pred.context = "Скопје е главниот град на Македонија."
    mock_pred.latency_ms = 100.0

    retrieved = [[_make_doc("mk_001", "Скопје е главниот град на Македонија.")]]

    results = evaluator.evaluate(
        pipeline_id="mk_bm25",
        generator_id="gpt4o",
        predictions=[mock_pred],
        gold_data=gold_data,
        retrieved_docs_list=retrieved,
    )

    assert len(results) == 1
    r = results[0]
    assert r.token_f1 > 0
    assert r.pipeline_id == "mk_bm25"


def test_evaluate_miss_on_wrong_doc():
    """When retrieved doc does not match relevant_doc_ids, custom metrics reflect the miss."""
    from unittest.mock import MagicMock
    from src.generator.generator import GenerationResult

    evaluator = RAGEvaluator(use_ragas=False)

    gold_data = [{
        "query": "Кој е главниот град?",
        "answer": "Скопје.",
        "relevant_doc_ids": ["mk_001"],
    }]

    mock_pred = MagicMock(spec=GenerationResult)
    mock_pred.answer = "Не знам."
    mock_pred.query = "Кој е главниот град?"
    mock_pred.context = "Нерелевантен текст."
    mock_pred.latency_ms = 100.0

    retrieved = [[_make_doc("mk_999", "Нерелевантен текст.")]]  # wrong doc

    results = evaluator.evaluate(
        pipeline_id="mk_bm25",
        generator_id="gpt4o",
        predictions=[mock_pred],
        gold_data=gold_data,
        retrieved_docs_list=retrieved,
    )

    r = results[0]
    # Token F1 should be low since answer is wrong
    assert r.token_f1 < 0.5


# ── Save results ──────────────────────────────────────────────────────────────

def test_save_and_reload_results(tmp_path):
    evaluator = RAGEvaluator(use_ragas=False)
    results = [_make_result(), _make_result(pipeline_id="mk_dense")]

    out_file = tmp_path / "results.jsonl"
    evaluator.save_results(results, out_file)

    assert out_file.exists()
    import json
    lines = [json.loads(l) for l in out_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]["pipeline_id"] == "mk_bm25"
    assert lines[1]["pipeline_id"] == "mk_dense"
