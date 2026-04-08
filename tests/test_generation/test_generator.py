"""
Tests for the Generator module (mocked — no real API calls).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.generator.generator import Generator, GeneratorConfig, GenerationResult, ProviderType
from src.retrieval.base import RetrievedDoc


def _make_doc(text: str, doc_id: str = "d1") -> RetrievedDoc:
    return RetrievedDoc(chunk_id=doc_id, doc_id=doc_id, text=text, score=1.0, lang="mk", rank=0)


# ── Config ────────────────────────────────────────────────────────────────────

def test_generator_config_defaults():
    cfg = GeneratorConfig(provider=ProviderType.OPENAI, model_id="gpt-4o")
    assert cfg.max_tokens == 512
    assert cfg.temperature == 0.0
    assert "македонски" in cfg.system_prompt.lower()


def test_build_user_message():
    cfg = GeneratorConfig(provider=ProviderType.OPENAI, model_id="gpt-4o")
    gen = Generator(cfg)
    msg = gen._build_user_message("Кој е главниот град?", "Скопје е главниот град.")
    assert "Скопје" in msg
    assert "Кој е главниот град?" in msg
    assert "Контекст" in msg


# ── OpenAI (mocked) ───────────────────────────────────────────────────────────

def test_generate_openai_mocked():
    cfg = GeneratorConfig(provider=ProviderType.OPENAI, model_id="gpt-4o")
    gen = Generator(cfg)

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Скопје е главниот град."
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 10

    # Inject a fake openai_client so no real import needed
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    gen._openai_client = mock_client

    result = gen.generate(
        query="Кој е главниот град на Македонија?",
        context_docs=[_make_doc("Скопје е главниот град на Македонија.")],
    )

    assert isinstance(result, GenerationResult)
    assert result.answer == "Скопје е главниот град."
    assert result.provider == "openai"
    assert result.prompt_tokens == 50
    assert result.completion_tokens == 10
    assert result.latency_ms >= 0


# ── Anthropic (mocked) ────────────────────────────────────────────────────────

def test_generate_anthropic_mocked():
    cfg = GeneratorConfig(provider=ProviderType.ANTHROPIC, model_id="claude-sonnet-4-5")
    gen = Generator(cfg)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Скопје.")]
    mock_response.usage.input_tokens = 40
    mock_response.usage.output_tokens = 5

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    gen._anthropic_client = mock_client

    result = gen.generate(
        query="Кој е главниот град?",
        context_docs=[_make_doc("Скопје е главниот град.")],
    )

    assert result.answer == "Скопје."
    assert result.provider == "anthropic"
    assert result.prompt_tokens == 40


# ── Multi-doc context ─────────────────────────────────────────────────────────

def test_context_concatenation():
    cfg = GeneratorConfig(provider=ProviderType.OPENAI, model_id="gpt-4o")
    gen = Generator(cfg)

    docs = [_make_doc("Факт А.", "d1"), _make_doc("Факт Б.", "d2")]
    context = "\n\n---\n\n".join(d.text for d in docs)
    msg = gen._build_user_message("Прашање?", context)
    assert "Факт А." in msg
    assert "Факт Б." in msg
    assert "---" in msg


# ── Unknown provider ──────────────────────────────────────────────────────────

def test_unknown_provider_raises():
    cfg = GeneratorConfig(provider="unknown_provider", model_id="x")  # type: ignore
    gen = Generator(cfg)
    with pytest.raises((ValueError, AttributeError)):
        gen.generate("test", [_make_doc("ctx", "d1")])


# ── Retry logic ───────────────────────────────────────────────────────────────

def test_retry_on_rate_limit():
    """Generator should retry up to _MAX_RETRIES on rate-limit errors."""
    from src.generator.generator import _retry_with_backoff

    call_count = 0

    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("rate limit exceeded (429)")
        return "ok"

    with patch("src.generator.generator.time.sleep"):  # don't actually sleep
        result = _retry_with_backoff(flaky, label="test")

    assert result == "ok"
    assert call_count == 3


def test_retry_raises_after_max():
    """After max retries, the exception should propagate."""
    from src.generator.generator import _retry_with_backoff

    def always_fail():
        raise Exception("too many requests")

    with patch("src.generator.generator.time.sleep"):
        with pytest.raises(Exception, match="too many requests"):
            _retry_with_backoff(always_fail, label="test")
