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
    return _settings
