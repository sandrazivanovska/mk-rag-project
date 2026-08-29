"""
Configuration management.

Reads from environment variables (via .env) and a YAML config file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-level settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-5"

    # Gemini. GOOGLE_API_KEY is the google-genai SDK's own env var; GEMINI_API_KEY
    # is accepted as an alias so either name in .env works.
    google_api_key: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"
    gemini_model_fast: str = "gemini-2.5-flash"

    # Output budget per generation. Gemini 2.5 counts THINKING tokens against
    # this, and gemini-2.5-pro spends 500-1000 of them before writing anything.
    # At the old 512 default, Pro's answers came back truncated mid-word with
    # finish_reason=MAX_TOKENS, which silently corrupted every generation metric.
    # 2048 leaves room for reasoning plus a complete Macedonian answer.
    gen_max_tokens: int = 4096

    # Vertex AI mode. Set USE_VERTEX=true to authenticate with Google Cloud
    # credentials (service account or ADC) instead of an API key. Needed when
    # AI Studio only issues "AQ." keys, which the Gemini API rejects with
    # 401 ACCESS_TOKEN_TYPE_UNSUPPORTED. Vertex also draws on Cloud trial credit.
    use_vertex: bool = False
    vertex_project: str = ""
    vertex_location: str = "us-central1"

    # Which generators build_all_pipelines() runs. Trim this to cut cost.
    generator_ids: list[str] = ["gemini_flash", "gemini_pro"]

    # Provider used as the RAGAS judge: "gemini" or "openai".
    #
    # The Gemini judge goes through Google's OpenAI-compatible endpoint rather
    # than langchain-google-genai. That package pulls in the legacy
    # google-generativeai SDK, which pins protobuf<5, while transformers needs
    # protobuf>=5.27 — installing it silently breaks FlagEmbedding. The compat
    # endpoint gives the same models with no dependency conflict.
    judge_provider: str = "gemini"
    # Flash, not pro: RAGAS sends ~270k prompt tokens per sample (its own
    # few-shot boilerplate, not our context), so the judge dominates run cost.
    # Pro as judge costs ~4x and would put the full study well over budget.
    judge_model: str = "gemini-2.5-flash"
    judge_embed_model: str = "text-embedding-004"
    # Score RAGAS on this many questions per pipeline (None = all). The free
    # local metrics still cover every question.
    ragas_sample: Optional[int] = 50
    ragas_seed: int = 42
    gemini_openai_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Google Translate
    google_application_credentials: Optional[str] = None
    google_translate_api_key: Optional[str] = None

    # Model IDs
    embed_model_primary: str = "BAAI/bge-m3"
    embed_model_baseline: str = "intfloat/multilingual-e5-large-instruct"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Retrieval
    top_k_retrieval: int = 50
    top_k_rerank: int = 5

    # Chunking
    chunk_size: int = 384
    chunk_overlap: int = 64
    chunking_strategy: str = "sentence"

    # Paths
    mk_wiki_dump_path: str = "data/raw/mk/mkwiki-latest-pages-articles.xml.bz2"
    processed_mk_path: str = "data/processed/mk/"
    processed_en_path: str = "data/processed/en/"
    faiss_index_mk: str = "data/indices/faiss/mk_bge_m3.index"
    faiss_index_en: str = "data/indices/faiss/en_bge_m3.index"
    bm25_index_mk: str = "data/indices/bm25/mk/"

    # Experiment
    experiment_output_dir: str = "results/"
    log_level: str = "INFO"


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """Load a YAML config file and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _export_api_keys(_settings)
    return _settings


def _export_api_keys(settings: Settings) -> None:
    """
    Publish keys from .env into os.environ.

    pydantic-settings reads .env into this object only — it never touches
    os.environ, which is where the OpenAI / Anthropic / google-genai clients
    look for their credentials. Without this, keys set in .env are invisible
    to every SDK. Real environment variables always win over .env.
    """
    gemini_key = settings.google_api_key or settings.gemini_api_key
    for var, value in (
        ("OPENAI_API_KEY", settings.openai_api_key),
        ("ANTHROPIC_API_KEY", settings.anthropic_api_key),
        ("GOOGLE_API_KEY", gemini_key),
        ("GEMINI_API_KEY", gemini_key),
    ):
        if value and not os.environ.get(var):
            os.environ[var] = value
