"""Unit tests for the chunking module."""

import pytest
from src.ingestion.chunker import chunk_documents, ChunkingStrategy


SAMPLE_DOCS = [
    {
        "id": "doc1",
        "text": (
            "Македонија е земја во Југоисточна Европа. "
            "Таа е членка на Обединетите нации. "
            "Скопје е нејзиниот главен град. "
            "Државата има население од околу два милиони жители. "
            "Официјалниот јазик е македонскиот јазик."
        ),
        "lang": "mk",
    }
]


class TestChunkDocuments:
    def test_sentence_strategy_returns_chunks(self):
        chunks = chunk_documents(SAMPLE_DOCS, strategy=ChunkingStrategy.SENTENCE)
        assert len(chunks) >= 1
        for c in chunks:
            assert c["text"].strip() != ""

    def test_fixed_strategy(self):
        chunks = chunk_documents(
            SAMPLE_DOCS,
            strategy=ChunkingStrategy.FIXED,
            chunk_size=50,
            chunk_overlap=10,
        )
        assert len(chunks) >= 1

    def test_paragraph_strategy(self):
        docs = [
            {
                "id": "doc2",
                "text": "Прв параграф.\n\nВтор параграф.\n\nТрет параграф.",
                "lang": "mk",
            }
        ]
        chunks = chunk_documents(docs, strategy=ChunkingStrategy.PARAGRAPH)
        assert len(chunks) >= 1

    def test_chunk_id_format(self):
        chunks = chunk_documents(SAMPLE_DOCS)
        for c in chunks:
            assert c["chunk_id"].startswith("doc1_")

    def test_total_chunks_backfilled(self):
        chunks = chunk_documents(SAMPLE_DOCS)
        total = chunks[0]["total_chunks"]
        assert total == len(chunks)

    def test_lang_preserved(self):
        chunks = chunk_documents(SAMPLE_DOCS)
        for c in chunks:
            assert c["lang"] == "mk"
