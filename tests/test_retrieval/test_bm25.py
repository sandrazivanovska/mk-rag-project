"""Unit tests for BM25Retriever."""

import json
import tempfile
from pathlib import Path

import pytest

from src.retrieval.bm25_retriever import BM25Retriever


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    docs = [
        {"id": "d1", "chunk_id": "d1_0", "doc_id": "d1", "text": "Скопје е главниот град на Македонија.", "lang": "mk"},
        {"id": "d2", "chunk_id": "d2_0", "doc_id": "d2", "text": "Македонија ја прогласи независноста во 1991 година.", "lang": "mk"},
        {"id": "d3", "chunk_id": "d3_0", "doc_id": "d3", "text": "Охридското Езеро е најголемото езеро во Македонија.", "lang": "mk"},
    ]
    jsonl_path = tmp_path / "chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return jsonl_path


def test_bm25_returns_results(sample_jsonl: Path):
    retriever = BM25Retriever.from_jsonl(sample_jsonl, top_k=3)
    results = retriever.retrieve("главниот град Скопје")
    assert len(results) > 0
    assert results[0].text != ""


def test_bm25_top_k(sample_jsonl: Path):
    retriever = BM25Retriever.from_jsonl(sample_jsonl, top_k=10)
    results = retriever.retrieve("Македонија", top_k=2)
    assert len(results) <= 2


def test_bm25_rank_order(sample_jsonl: Path):
    retriever = BM25Retriever.from_jsonl(sample_jsonl, top_k=3)
    results = retriever.retrieve("главниот град Скопје")
    # First result should contain "Скопје"
    assert "Скопје" in results[0].text


def test_bm25_persistence(sample_jsonl: Path, tmp_path: Path):
    cache = tmp_path / "bm25_cache.pkl"
    r1 = BM25Retriever.from_jsonl(sample_jsonl, cache_path=cache, top_k=3)
    assert cache.exists()
    r2 = BM25Retriever.from_jsonl(sample_jsonl, cache_path=cache, top_k=3)
    res1 = r1.retrieve("Скопје")
    res2 = r2.retrieve("Скопје")
    assert [d.chunk_id for d in res1] == [d.chunk_id for d in res2]
