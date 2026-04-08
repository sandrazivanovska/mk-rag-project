"""
Integration tests for the RAGPipeline (mocked retriever + generator).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.pipelines.pipeline import RAGPipeline, PipelineConfig, PipelineResult
from src.retrieval.base import RetrievedDoc
from src.generator.generator import GenerationResult


def _mock_doc(doc_id: str, text: str, score: float = 0.9) -> RetrievedDoc:
    return RetrievedDoc(chunk_id=doc_id, doc_id=doc_id, text=text, score=score, lang="mk", rank=0)


def _mock_retriever(docs: list[RetrievedDoc]):
    retriever = MagicMock()
    retriever.retrieve.return_value = docs
    return retriever


def _mock_gen_result(answer: str = "Тест одговор.") -> GenerationResult:
    r = MagicMock(spec=GenerationResult)
    r.answer = answer
    r.latency_ms = 100.0
    r.prompt_tokens = 50
    r.completion_tokens = 10
    return r


def _mock_generator(answer: str = "Тест одговор."):
    gen = MagicMock()
    gen.generate.return_value = _mock_gen_result(answer)
    return gen


def _make_config(**kwargs) -> PipelineConfig:
    defaults = dict(
        pipeline_id="mk_bm25",
        name="MK-BM25",
        retrieval_type="bm25",
        corpus_lang="mk",
        generator_id="gpt4o",
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


# ── Basic run ─────────────────────────────────────────────────────────────────

def test_pipeline_run_returns_result():
    docs = [_mock_doc("mk_001", "Скопје е главниот град на Македонија.")]
    pipeline = RAGPipeline(
        config=_make_config(),
        retriever=_mock_retriever(docs),
        generator=_mock_generator(),
    )

    result = pipeline.run("Кој е главниот град?")
    assert isinstance(result, PipelineResult)
    assert result.generation.answer == "Тест одговор."
    assert len(result.retrieved_docs) == 1
    assert result.retrieved_docs[0].doc_id == "mk_001"


def test_pipeline_run_batch():
    docs = [_mock_doc("mk_001", "Скопје е главниот град.")]
    pipeline = RAGPipeline(
        config=_make_config(),
        retriever=_mock_retriever(docs),
        generator=_mock_generator("Одговор."),
    )

    queries = ["Прашање 1?", "Прашање 2?", "Прашање 3?"]
    results = pipeline.run_batch(queries)
    assert len(results) == 3
    assert all(r.generation.answer == "Одговор." for r in results)


# ── Reranker integration ──────────────────────────────────────────────────────

def test_pipeline_with_reranker():
    docs = [
        _mock_doc("mk_001", "Скопје е главниот град.", 0.9),
        _mock_doc("mk_002", "Македонија прогласи независност.", 0.7),
        _mock_doc("mk_003", "Охрид е на езерото.", 0.5),
    ]
    reranker = MagicMock()
    reranker.rerank.return_value = [docs[2], docs[0]]

    pipeline = RAGPipeline(
        config=_make_config(pipeline_id="mk_hybrid", retrieval_type="hybrid", use_reranker=True),
        retriever=_mock_retriever(docs),
        generator=_mock_generator("Реранкиран одговор."),
        reranker=reranker,
    )

    result = pipeline.run("Кој е главниот град?")
    assert reranker.rerank.called
    assert result.generation.answer == "Реранкиран одговор."


# ── Translator integration ────────────────────────────────────────────────────

def test_pipeline_with_translation():
    docs = [_mock_doc("en_001", "Skopje is the capital of Macedonia.")]
    translator = MagicMock()
    translator.translate.side_effect = lambda text, **_: f"[EN]{text}"

    pipeline = RAGPipeline(
        config=_make_config(
            pipeline_id="translate_retrieve",
            name="Translate-Retrieve",
            retrieval_type="dense",
            corpus_lang="en",
            translate_query=True,
        ),
        retriever=_mock_retriever(docs),
        generator=_mock_generator("Скопје."),
        translator=translator,
    )

    result = pipeline.run("Кој е главниот град?")
    assert translator.translate.called
    assert result.generation.answer == "Скопје."


# ── PipelineResult helpers ────────────────────────────────────────────────────

def test_pipeline_result_final_docs():
    docs = [_mock_doc("mk_001", "Текст А."), _mock_doc("mk_002", "Текст Б.")]
    result = PipelineResult(
        query="Q",
        generation=_mock_gen_result(),
        retrieved_docs=docs,
        reranked_docs=None,
        pipeline_id="mk_bm25",
        generator_id="gpt4o",
        translated_query=None,
    )
    assert result.final_docs == docs


def test_pipeline_result_final_docs_uses_reranked():
    orig = [_mock_doc("mk_001", "A"), _mock_doc("mk_002", "B")]
    reranked = [_mock_doc("mk_002", "B")]
    result = PipelineResult(
        query="Q",
        generation=_mock_gen_result(),
        retrieved_docs=orig,
        reranked_docs=reranked,
        pipeline_id="mk_bm25",
        generator_id="gpt4o",
        translated_query=None,
    )
    assert result.final_docs == reranked


def test_pipeline_result_to_dict():
    docs = [_mock_doc("mk_001", "Text")]
    result = PipelineResult(
        query="Q",
        generation=_mock_gen_result(),
        retrieved_docs=docs,
        reranked_docs=None,
        pipeline_id="mk_bm25",
        generator_id="gpt4o",
        translated_query=None,
    )
    d = result.to_dict()
    assert d["query"] == "Q"
    assert d["pipeline_id"] == "mk_bm25"
    assert isinstance(d["retrieved_count"], int)
