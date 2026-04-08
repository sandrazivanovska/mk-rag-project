"""
Document chunking strategies.

Supports three strategies as described in the project spec:
  - sentence  : sentence-boundary-aware splitting (default, recommended for MK)
  - fixed     : fixed token-count windows
  - paragraph : paragraph-level splitting

All strategies produce dicts with:
  chunk_id, doc_id, text, lang, chunk_index, total_chunks
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Iterator

from src.utils.logging import get_logger

logger = get_logger("chunker")

# Try tiktoken for accurate token counting; fall back to whitespace-split when
# the network is unavailable (sandbox / offline environments).
try:
    import tiktoken
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    _USE_TIKTOKEN = True
except Exception:
    _TOKENIZER = None
    _USE_TIKTOKEN = False


class ChunkingStrategy(str, enum.Enum):
    SENTENCE = "sentence"
    FIXED = "fixed"
    PARAGRAPH = "paragraph"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    lang: str
    chunk_index: int
    total_chunks: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "lang": self.lang,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            **self.metadata,
        }


def chunk_documents(
    documents: list[dict],
    *,
    strategy: ChunkingStrategy | str = ChunkingStrategy.SENTENCE,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> list[dict]:
    """
    Chunk a list of document dicts into smaller pieces.

    Args:
        documents: List of dicts with at least 'id', 'text', 'lang' fields.
        strategy: Chunking strategy.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap in tokens between consecutive chunks.

    Returns:
        Flat list of chunk dicts.
    """
    strategy = ChunkingStrategy(strategy)
    all_chunks: list[dict] = []

    for doc in documents:
        doc_id = doc.get("id", "unknown")
        text = doc.get("text", "")
        lang = doc.get("lang", "mk")

        if strategy == ChunkingStrategy.SENTENCE:
            raw_chunks = _sentence_chunks(text, chunk_size, chunk_overlap)
        elif strategy == ChunkingStrategy.FIXED:
            raw_chunks = _fixed_chunks(text, chunk_size, chunk_overlap)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            raw_chunks = _paragraph_chunks(text, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        for idx, chunk_text in enumerate(raw_chunks):
            chunk = Chunk(
                chunk_id=f"{doc_id}_{idx}",
                doc_id=doc_id,
                text=chunk_text,
                lang=lang,
                chunk_index=idx,
                metadata={k: v for k, v in doc.items() if k not in ("id", "text", "lang")},
            )
            all_chunks.append(chunk.to_dict())

    # Backfill total_chunks
    from collections import Counter
    counts: Counter = Counter(c["doc_id"] for c in all_chunks)
    for c in all_chunks:
        c["total_chunks"] = counts[c["doc_id"]]

    logger.info(
        f"Chunked {len(documents):,} documents → {len(all_chunks):,} chunks "
        f"(strategy={strategy.value}, size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks


# ── Strategy implementations ───────────────────────────────────────────────────

# Macedonian sentence endings (Cyrillic + Latin punctuation)
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> list:
    """Return a list of tokens (ints if tiktoken available, else words)."""
    if _USE_TIKTOKEN:
        return _TOKENIZER.encode(text)
    # Fallback: treat whitespace-separated words as tokens
    return text.split()


def _decode(tokens: list) -> str:
    """Decode a token list back to a string."""
    if _USE_TIKTOKEN:
        return _TOKENIZER.decode(tokens)
    return " ".join(tokens)


def _sentence_chunks(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Split on sentence boundaries, then merge into target-size windows."""
    sentences = _SENT_BOUNDARY_RE.split(text)
    chunks: list[str] = []
    current_tokens: list[int] = []

    for sentence in sentences:
        sentence_tokens = _tokenize(sentence)
        if len(current_tokens) + len(sentence_tokens) <= chunk_size:
            current_tokens.extend(sentence_tokens)
        else:
            if current_tokens:
                chunks.append(_decode(current_tokens))
            # Start new chunk with overlap
            overlap = current_tokens[-chunk_overlap:] if chunk_overlap > 0 else []
            current_tokens = overlap + sentence_tokens

    if current_tokens:
        chunks.append(_decode(current_tokens))

    return [c.strip() for c in chunks if c.strip()]


def _fixed_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Fixed-size token windows."""
    tokens = _tokenize(text)
    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(_decode(tokens[start:end]))
        start += step

    return [c.strip() for c in chunks if c.strip()]


def _paragraph_chunks(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Split on double newlines (paragraphs), merge small ones."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current_tokens: list[int] = []

    for para in paragraphs:
        para_tokens = _tokenize(para)
        if len(current_tokens) + len(para_tokens) <= chunk_size:
            current_tokens.extend(para_tokens)
        else:
            if current_tokens:
                chunks.append(_decode(current_tokens))
            overlap = current_tokens[-chunk_overlap:] if chunk_overlap > 0 else []
            current_tokens = overlap + para_tokens

    if current_tokens:
        chunks.append(_decode(current_tokens))

    return [c.strip() for c in chunks if c.strip()]
