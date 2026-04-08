"""
LLM generators: GPT-4o (OpenAI) and Claude Sonnet 4.5 (Anthropic).

Both accept a list of retrieved context chunks and a Macedonian query
and produce an answer in Macedonian.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger("generator")

# ── Retry / rate-limit config ──────────────────────────────────────────────────
_MAX_RETRIES = 5
_BASE_DELAY  = 2.0   # seconds — doubles each retry (exponential back-off)
_JITTER      = 0.5   # seconds — random jitter to avoid thundering herd

def _retry_with_backoff(func, *args, label: str = "", **kwargs):
    """Call func with exponential back-off on rate-limit / transient errors."""
    delay = _BASE_DELAY
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            err_str = str(exc).lower()
            is_rate_limit = any(k in err_str for k in ("rate limit", "rate_limit", "429", "too many requests"))
            is_transient  = any(k in err_str for k in ("timeout", "connection", "503", "502", "overloaded"))
            if (is_rate_limit or is_transient) and attempt < _MAX_RETRIES:
                wait = delay + random.uniform(0, _JITTER)
                logger.warning(f"{label} attempt {attempt}/{_MAX_RETRIES} failed ({exc}). Retrying in {wait:.1f}s...")
                time.sleep(wait)
                delay *= 2
            else:
                raise

# ── Macedonian system prompt ───────────────────────────────────────────────────

_MK_SYSTEM_PROMPT = (
    "Одговори на прашањето само врз основа на дадениот контекст. "
    "Одговорот напиши на македонски јазик. "
    "Ако контекстот не содржи доволно информации, напиши дека не знаеш."
)


# ── Data classes ───────────────────────────────────────────────────────────────


class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class GeneratorConfig:
    provider: ProviderType
    model_id: str
    max_tokens: int = 512
    temperature: float = 0.0
    system_prompt: str = _MK_SYSTEM_PROMPT


@dataclass
class GenerationResult:
    answer: str
    query: str
    context: str
    model_id: str
    provider: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict = field(default_factory=dict)


# ── Generator class ────────────────────────────────────────────────────────────


class Generator:
    """
    Unified generator interface for OpenAI and Anthropic models.

    Example
    -------
    >>> gen = Generator(GeneratorConfig(provider=ProviderType.ANTHROPIC,
    ...                                 model_id="claude-sonnet-4-5"))
    >>> result = gen.generate(query="Кој е главниот град на Македонија?",
    ...                       context_docs=retrieved_docs)
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self._openai_client = None
        self._anthropic_client = None

    @property
    def openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI()
        return self._openai_client

    @property
    def anthropic_client(self):
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic()
        return self._anthropic_client

    def generate(
        self,
        query: str,
        context_docs: list,  # list[RetrievedDoc]
        *,
        context_separator: str = "\n\n---\n\n",
    ) -> GenerationResult:
        """
        Generate an answer given a query and retrieved context documents.

        Args:
            query: The user query in Macedonian.
            context_docs: List of RetrievedDoc (from retrieval stage).
            context_separator: Separator between context chunks.

        Returns:
            GenerationResult with the answer and metadata.
        """
        context = context_separator.join(doc.text for doc in context_docs)
        user_message = self._build_user_message(query, context)

        t0 = time.perf_counter()

        if self.config.provider == ProviderType.OPENAI:
            result = self._generate_openai(user_message)
        elif self.config.provider == ProviderType.ANTHROPIC:
            result = self._generate_anthropic(user_message)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        latency_ms = (time.perf_counter() - t0) * 1000

        return GenerationResult(
            answer=result["answer"],
            query=query,
            context=context,
            model_id=self.config.model_id,
            provider=self.config.provider.value,
            latency_ms=latency_ms,
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_user_message(self, query: str, context: str) -> str:
        return (
            f"Контекст:\n{context}\n\n"
            f"Прашање: {query}"
        )

    def _generate_openai(self, user_message: str) -> dict:
        def _call():
            return self.openai_client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        response = _retry_with_backoff(_call, label=f"OpenAI/{self.config.model_id}")
        return {
            "answer": response.choices[0].message.content or "",
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    def _generate_anthropic(self, user_message: str) -> dict:
        def _call():
            return self.anthropic_client.messages.create(
                model=self.config.model_id,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self.config.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        response = _retry_with_backoff(_call, label=f"Anthropic/{self.config.model_id}")
        return {
            "answer": response.content[0].text if response.content else "",
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }
