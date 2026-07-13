"""
Unit tests for the EN parallel-corpus pipeline (scripts/mk/07-09).

The scripts have numeric-prefixed filenames, so they are loaded by path via
importlib. Only the network-free pure functions are exercised here; HTTP and
translation calls are represented by parsed-response fixtures and fake callables.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "mk"


def _load(module_filename: str):
    path = SCRIPTS_DIR / module_filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # required so @dataclass can resolve __module__
    spec.loader.exec_module(module)
    return module


link = _load("07_link_mk_to_en.py")
build = _load("08_build_en_documents.py")
chunk_en = _load("09_chunk_en_documents.py")


# ── 07: langlinks parsing + alignment split ──────────────────────────────────


def test_parse_langlinks_response_extracts_title_and_qid():
    data = {
        "query": {
            "pages": [
                {
                    "title": "Скопје",
                    "langlinks": [{"lang": "en", "title": "Skopje"}],
                    "pageprops": {"wikibase_item": "Q3919"},
                },
                {
                    "title": "Некоја статија",
                    "pageprops": {"wikibase_item": "Q999"},
                },
            ]
        }
    }
    result = link.parse_langlinks_response(data)
    assert result["Скопје"]["en_title"] == "Skopje"
    assert result["Скопје"]["wikidata_qid"] == "Q3919"
    # No EN langlink → en_title is None but QID still captured.
    assert result["Некоја статија"]["en_title"] is None
    assert result["Некоја статија"]["wikidata_qid"] == "Q999"


def test_build_alignment_record_linked_and_needs_mt():
    mk_doc = {"doc_id": "mk_wiki_42", "title": "Скопје"}

    linked = link.build_alignment_record(
        mk_doc, {"en_title": "Skopje", "wikidata_qid": "Q3919"}
    )
    assert linked["mk_doc_id"] == "mk_wiki_42"
    assert linked["en_title"] == "Skopje"
    assert linked["wikidata_qid"] == "Q3919"
    assert linked["en_status"] == "linked"

    needs_mt = link.build_alignment_record(
        mk_doc, {"en_title": None, "wikidata_qid": "Q3919"}
    )
    assert needs_mt["en_status"] == "needs_mt"
    assert needs_mt["en_title"] is None


def test_chunked_batches():
    assert list(link.chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


# ── 08: EN quality filter ────────────────────────────────────────────────────


def test_latin_ratio():
    assert build.latin_ratio("Hello world") == 1.0
    assert build.latin_ratio("Скопје") == 0.0
    assert build.latin_ratio("") == 0.0


def test_is_good_english_text():
    good = "Skopje is the capital and largest city of North Macedonia. " * 5
    assert build.is_good_english_text(good) is True
    # Too short
    assert build.is_good_english_text("Skopje.") is False
    # Mostly Cyrillic (e.g. fetch returned the wrong language)
    assert build.is_good_english_text("Скопје е главен град на Македонија. " * 10) is False


# ── 08: translation text-splitting + reassembly ──────────────────────────────


def test_split_text_for_translation_respects_max_chars():
    text = "Едно. Две. Три. Четири. Пет."
    parts = build.split_text_for_translation(text, max_chars=10)
    assert all(len(p) <= 10 for p in parts)
    # Nothing dropped: every sentence survives somewhere.
    joined = " ".join(parts)
    for token in ["Едно", "Две", "Три", "Четири", "Пет"]:
        assert token in joined


def test_translate_long_text_uses_fake_translator():
    text = "Скопје. " * 100  # long enough to require multiple parts
    # Fake translator: uppercases, proving each part is translated and rejoined.
    out = build.translate_long_text(text, translate_fn=str.upper, max_chars=50)
    assert "СКОПЈЕ" in out
    assert "скопје" not in out


def test_parse_extract_response():
    data = {
        "query": {
            "pages": [
                {
                    "pageid": 534366,
                    "title": "Skopje",
                    "extract": "Skopje is the capital of North Macedonia.",
                }
            ]
        }
    }
    parsed = build.parse_extract_response(data)
    assert parsed["pageid"] == 534366
    assert parsed["title"] == "Skopje"
    assert "capital" in parsed["text"]


def test_parse_extract_response_missing_or_empty_returns_none():
    assert build.parse_extract_response({"query": {"pages": []}}) is None
    empty = {"query": {"pages": [{"pageid": 1, "title": "X", "extract": ""}]}}
    assert build.parse_extract_response(empty) is None


# ── 09: chunking carries alignment keys and EN language tag ───────────────────


def test_make_en_chunks_propagates_alignment_keys():
    document = {
        "doc_id": "en_wiki_534366",
        "source": "en_wikipedia",
        "title": "Skopje",
        "language": "en",
        "text": "Skopje is the capital of North Macedonia. " * 40,
        "url": "https://en.wikipedia.org/wiki/Skopje",
        "wikidata_qid": "Q3919",
        "mk_doc_id": "mk_wiki_42",
    }
    config = chunk_en.ChunkingConfig()
    chunks = chunk_en.make_en_chunks_for_document(document, config)

    assert len(chunks) >= 1
    for c in chunks:
        # Index-builder compatibility: needs text, chunk_id, doc_id, lang.
        assert c["text"]
        assert c["chunk_id"].startswith("en_wiki_534366_chunk_")
        assert c["doc_id"] == "en_wiki_534366"
        assert c["lang"] == "en"
        # Alignment keys round-trip into every chunk.
        assert c["wikidata_qid"] == "Q3919"
        assert c["mk_doc_id"] == "mk_wiki_42"
        assert c["source"] == "en_wikipedia"
