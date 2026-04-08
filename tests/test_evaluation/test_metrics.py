"""Unit tests for custom MK evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    mk_exact_match,
    mk_token_f1,
    context_coverage,
    retrieval_hit_at_k,
    retrieval_mrr,
    retrieval_recall_at_k,
)


class TestMkExactMatch:
    def test_identical(self):
        assert mk_exact_match("Скопје", "Скопје") == 1.0

    def test_case_insensitive(self):
        assert mk_exact_match("скопје", "Скопје") == 1.0

    def test_different(self):
        assert mk_exact_match("Скопје", "Охрид") == 0.0

    def test_whitespace_normalised(self):
        assert mk_exact_match("  Скопје  ", "Скопје") == 1.0


class TestMkTokenF1:
    def test_perfect(self):
        score = mk_token_f1("Скопје е главниот град", "Скопје е главниот град")
        assert score == pytest.approx(1.0)

    def test_partial(self):
        score = mk_token_f1("Скопје е главниот град на Македонија", "Скопје е главниот")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert mk_token_f1("Охрид", "Битола") == 0.0


class TestContextCoverage:
    def test_full_coverage(self):
        answer = "Скопје е главниот град"
        context = "Скопје е главниот град на Македонија"
        assert context_coverage(answer, context) == pytest.approx(1.0)

    def test_partial_coverage(self):
        answer = "Скопје е главниот"
        context = "Охрид е убав град"
        score = context_coverage(answer, context)
        assert 0.0 <= score <= 1.0

    def test_empty_answer(self):
        assert context_coverage("", "some context") == 0.0


class TestRetrievalMetrics:
    def test_hit_at_k_hit(self):
        assert retrieval_hit_at_k(["d1", "d2", "d3"], {"d1"}, k=3) == 1.0

    def test_hit_at_k_miss(self):
        assert retrieval_hit_at_k(["d1", "d2", "d3"], {"d4"}, k=3) == 0.0

    def test_mrr_first(self):
        assert retrieval_mrr(["d1", "d2"], {"d1"}) == pytest.approx(1.0)

    def test_mrr_second(self):
        assert retrieval_mrr(["d1", "d2"], {"d2"}) == pytest.approx(0.5)

    def test_recall_at_k(self):
        score = retrieval_recall_at_k(["d1", "d2", "d3"], {"d1", "d2"}, k=2)
        assert score == pytest.approx(1.0)
