"""
LLM generators: GPT-4o (OpenAI), Claude Sonnet 4.5 (Anthropic) and Gemini (Google).

All accept a list of retrieved context chunks and a Macedonian query
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

# Model ids that reject an explicit thinking_config (mandatory minimum budget).
# Populated on first rejection so we stop re-attempting it on every call.
_NO_THINKING_CONTROL: set[str] = set()

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
            is_rate_limit = any(k in err_str for k in (
                "rate limit", "rate_limit", "429", "too many requests",
                "resource_exhausted", "quota",          # Gemini
            ))
            is_transient  = any(k in err_str for k in (
                "timeout", "connection", "503", "502", "overloaded",
                "unavailable", "internal error", "500",  # Gemini
            ))
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
    "Одговори кратко и директно, во една или две реченици. "
    "Не користи наслови, списоци или воведни фрази. "
    "Ако контекстот не содржи доволно информации, напиши дека не знаеш."
)
# The brevity instruction is deliberate and applies to every provider equally,
# so it does not bias the comparison. Without it the models answer with
# multi-hundred-word bulleted summaries while the gold answers are ~29 words —
# which drives token_f1 and exact_match toward zero for reasons that have
# nothing to do with retrieval or answer quality. It also keeps generations
# inside the output-token budget.


# ── Data classes ───────────────────────────────────────────────────────────────


class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


@dataclass
class GeneratorConfig:
    provider: ProviderType
    model_id: str
    max_tokens: int = 512
    temperature: float = 0.0
    system_prompt: str = _MK_SYSTEM_PROMPT
    # Gemini only. Thinking tokens are billed as output and count against
    # max_output_tokens, so leaving thinking on with a 512-token cap can consume
    # the whole budget and return an empty answer. 0 disables it where the model
    # allows; models with a mandatory minimum budget ignore this.
    thinking_budget: Optional[int] = 0
    # Gemini only. Authenticate via Google Cloud instead of an API key.
    use_vertex: bool = False
    vertex_project: str = ""
    vertex_location: str = "us-central1"


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
        self._gemini_client = None

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

    @property
    def gemini_client(self):
        if self._gemini_client is None:
            import os
            from google import genai

            if self.config.use_vertex:
                project = self.config.vertex_project or os.getenv("VERTEX_PROJECT", "")
                if not project:
                    raise RuntimeError(
                        "USE_VERTEX is set but VERTEX_PROJECT is empty. Put your "
                        "Google Cloud project ID in .env as VERTEX_PROJECT=..."
                    )
                # Credentials come from GOOGLE_APPLICATION_CREDENTIALS (service
                # account JSON) or from `gcloud auth application-default login`.
                self._gemini_client = genai.Client(
                    vertexai=True,
                    project=project,
                    location=self.config.vertex_location,
                )
            else:
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "No Gemini API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) "
                        "in your .env — get one at https://aistudio.google.com/apikey"
                    )
                self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

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
        elif self.config.provider == ProviderType.GEMINI:
            result = self._generate_gemini(user_message)
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

    def _generate_gemini(self, user_message: str) -> dict:
        from google.genai import types

        def _build_config(with_thinking_control: bool):
            kwargs = dict(
                system_instruction=self.config.system_prompt,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )
            if with_thinking_control and self.config.thinking_budget is not None:
                kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.config.thinking_budget
                )
            return types.GenerateContentConfig(**kwargs)

        def _call():
            # Once a model has rejected thinking_config, stop sending it. Models
            # with a mandatory thinking budget reject it on EVERY call, and
            # retrying each time doubles the request count and wall-clock time
            # across a full run.
            if self.config.model_id in _NO_THINKING_CONTROL:
                return self.gemini_client.models.generate_content(
                    model=self.config.model_id,
                    contents=user_message,
                    config=_build_config(with_thinking_control=False),
                )
            try:
                return self.gemini_client.models.generate_content(
                    model=self.config.model_id,
                    contents=user_message,
                    config=_build_config(with_thinking_control=True),
                )
            except Exception as exc:
                if "thinking" not in str(exc).lower():
                    raise
                _NO_THINKING_CONTROL.add(self.config.model_id)
                logger.warning(
                    f"{self.config.model_id} rejects thinking_config ({exc}); "
                    "disabling it for this model for the rest of the run."
                )
                return self.gemini_client.models.generate_content(
                    model=self.config.model_id,
                    contents=user_message,
                    config=_build_config(with_thinking_control=False),
                )

        response = _retry_with_backoff(_call, label=f"Gemini/{self.config.model_id}")

        # .text is None when the candidate was blocked by a safety filter or the
        # output budget was exhausted before any answer tokens were emitted.
        answer = response.text or ""
        finish = None
        if getattr(response, "candidates", None):
            finish = getattr(response.candidates[0], "finish_reason", None)

        if not answer:
            logger.warning(
                f"Gemini/{self.config.model_id} returned no text (finish_reason={finish})."
            )
        elif finish is not None and "MAX_TOKENS" in str(finish):
            # Silent-corruption guard: a truncated answer still looks like a
            # valid answer downstream, but every generation metric computed from
            # it is wrong. Thinking tokens count against max_tokens, so raise
            # GEN_MAX_TOKENS rather than assuming the model was merely verbose.
            logger.warning(
                f"Gemini/{self.config.model_id} TRUNCATED at max_tokens="
                f"{self.config.max_tokens} (thinking consumes this budget too). "
                f"Answer is incomplete — raise GEN_MAX_TOKENS."
            )

        usage = getattr(response, "usage_metadata", None)
        return {
            "answer": answer,
            "prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "completion_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        }
