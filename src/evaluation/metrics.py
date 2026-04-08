"""
Custom evaluation metrics for Macedonian RAG.

These supplement the standard RAGAS metrics with MK-specific measures:
  - mk_exact_match    : normalised exact string match (Cyrillic-aware)
  - mk_token_f1       : token-level F1 (commonly used in QA benchmarks)
  - context_coverage  : proportion of answer tokens found in context
  - retrieval_mrr     : Mean Reciprocal Rank for retrieval quality
  - retrieval_hit_k   : Hit@k for retrieval quality
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Optional


# ── Text normalisation ─────────────────────────────────────────────────────────

def _normalise_mk(text: str) -> str:
    """NFC normalise, lowercase, strip punctuation for Cyrillic text."""
    text = unicodedata.normalize("NFC", text).lower()
    # Remove Cyrillic and Latin punctuation
    import re
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _tokenise(text: str) -> list[str]:
    return _normalise_mk(text).split()


# ── QA Metrics ─────────────────────────────────────────────────────────────────

def mk_exact_match(prediction: str, reference: str) -> float:
    """
    Normalised Exact Match (0 or 1).

    Args:
        prediction: Model-generated answer.
        reference: Gold-standard answer.

    Returns:
        1.0 if normalised strings match, else 0.0.
    """
    return float(_normalise_mk(prediction) == _normalise_mk(reference))


def mk_token_f1(prediction: str, reference: str) -> float:
    """
    Token-level F1 score (SQuAD-style).

    Args:
        prediction: Model-generated answer.
        reference: Gold-standard answer.

    Returns:
        F1 in [0, 1].
    """
    pred_tokens = Counter(_tokenise(prediction))
    ref_tokens = Counter(_tokenise(reference))

    common = sum((pred_tokens & ref_tokens).values())
    if common == 0:
        return 0.0

    precision = common / sum(pred_tokens.values())
    recall = common / sum(ref_tokens.values())
    return 2 * precision * recall / (precision + recall)


def context_coverage(answer: str, context: str) -> float:
    """
    What proportion of answer tokens appear in the context?

    Higher is better (ideally ~1.0 for faithful answers).
    """
    answer_tokens = set(_tokenise(answer))
    context_tokens = set(_tokenise(context))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


# ── Retrieval Metrics ──────────────────────────────────────────────────────────

def retrieval_hit_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Hit@k: 1 if any relevant doc appears in top-k, else 0."""
    return float(bool(set(retrieved_ids[:k]) & relevant_ids))


def retrieval_mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Mean Reciprocal Rank for a single query."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def retrieval_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Recall@k: fraction of relevant docs found in top-k."""
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def retrieval_precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Precision@k: fraction of top-k that are relevant."""
    if k == 0:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / k
