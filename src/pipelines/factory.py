"""
Factory functions to build all 12 pipeline variants from config.

6 retrieval setups × 2 generators = 12 pipelines total.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.bilingual_retriever import BilingualFusionRetriever, CrossLingualRetriever
from src.reranker.reranker import Reranker
from src.generator.generator import Generator, GeneratorConfig, ProviderType
from src.generator.translator import Translator
from src.utils.config import Settings, get_settings
from src.utils.logging import get_logger
from .pipeline import RAGPipeline, PipelineConfig

logger = get_logger("factory")


def build_pipeline(
    pipeline_id: str,
    generator_id: str,
    settings: Optional[Settings] = None,
    cfg: Optional[dict] = None,
) -> RAGPipeline:
    """
    Build a single pipeline by ID.

    Pipeline IDs (retrieval): mk_bm25, mk_dense, mk_hybrid,
                               translate_retrieve, cross_lingual_embed, bilingual_fusion
    Generator IDs: gpt4o, claude_sonnet
    """
    settings = settings or get_settings()
    return _PIPELINE_BUILDERS[pipeline_id](pipeline_id, generator_id, settings, cfg or {})


def build_all_pipelines(
    settings: Optional[Settings] = None,
    cfg: Optional[dict] = None,
) -> list[RAGPipeline]:
    """Build all 12 pipeline variants."""
    settings = settings or get_settings()
    cfg = cfg or {}
    pipelines = []
    for pid in _PIPELINE_BUILDERS:
        for gid in ["gpt4o", "claude_sonnet"]:
            logger.info(f"Building pipeline: {pid} + {gid}")
            pipelines.append(build_pipeline(pid, gid, settings, cfg))
    return pipelines


# ── Generator factory ──────────────────────────────────────────────────────────

def _make_generator(generator_id: str, settings: Settings) -> Generator:
    if generator_id == "gpt4o":
        return Generator(
            GeneratorConfig(
                provider=ProviderType.OPENAI,
                model_id=settings.openai_model,
            )
        )
    elif generator_id == "claude_sonnet":
        return Generator(
            GeneratorConfig(
                provider=ProviderType.ANTHROPIC,
                model_id=settings.anthropic_model,
            )
        )
    else:
        raise ValueError(f"Unknown generator_id: {generator_id}")


def _make_reranker(settings: Settings) -> Reranker:
    return Reranker(model_name=settings.reranker_model)


def _make_mk_bm25(settings: Settings) -> BM25Retriever:
    cache = Path(settings.bm25_index_mk) / "mk_bm25.pkl"
    jsonl = Path(settings.processed_mk_path) / "chunks.jsonl"
    return BM25Retriever.from_jsonl(jsonl, cache_path=cache, top_k=settings.top_k_retrieval)


def _make_mk_dense(settings: Settings, model_name: Optional[str] = None) -> DenseRetriever:
    model = model_name or settings.embed_model_primary
    return DenseRetriever.from_jsonl(
        Path(settings.processed_mk_path) / "chunks.jsonl",
        model_name=model,
        index_path=settings.faiss_index_mk,
        top_k=settings.top_k_retrieval,
    )


def _make_en_dense(settings: Settings) -> DenseRetriever:
    return DenseRetriever.from_jsonl(
        Path(settings.processed_en_path) / "chunks.jsonl",
        model_name=settings.embed_model_primary,
        index_path=settings.faiss_index_en,
        top_k=settings.top_k_retrieval,
    )


# ── Individual pipeline builders ───────────────────────────────────────────────

def _build_mk_bm25(pid: str, gid: str, settings: Settings, cfg: dict) -> RAGPipeline:
    config = PipelineConfig(
        pipeline_id=f"{pid}_{gid}",
        name=f"MK-BM25 + {gid}",
        retrieval_type="bm25",
        corpus_lang="mk",
        generator_id=gid,
        translate_query=False,
        use_reranker=False,
        top_k_retrieve=settings.top_k_retrieval,
        top_k_rerank=settings.top_k_rerank,
    )
    return RAGPipeline(
        config=config,
        retriever=_make_mk_bm25(settings),
        generator=_make_generator(gid, settings),
    )


def _build_mk_dense(pid: str, gid: str, settings: Settings, cfg: dict) -> RAGPipeline:
    config = PipelineConfig(
        pipeline_id=f"{pid}_{gid}",
        name=f"MK-Dense (BGE-M3) + {gid}",
        retrieval_type="dense",
        corpus_lang="mk",
        generator_id=gid,
        use_reranker=True,
        top_k_retrieve=settings.top_k_retrieval,
        top_k_rerank=settings.top_k_rerank,
    )
    return RAGPipeline(
        config=config,
        retriever=_make_mk_dense(settings),
        generator=_make_generator(gid, settings),
        reranker=_make_reranker(settings),
    )


def _build_mk_hybrid(pid: str, gid: str, settings: Settings, cfg: dict) -> RAGPipeline:
    config = PipelineConfig(
        pipeline_id=f"{pid}_{gid}",
        name=f"MK-Hybrid (BGE-M3 + BM25) + {gid}",
        retrieval_type="hybrid",
        corpus_lang="mk",
        generator_id=gid,
        use_reranker=True,
        top_k_retrieve=settings.top_k_retrieval,
        top_k_rerank=settings.top_k_rerank,
    )
    hybrid = HybridRetriever(
        bm25=_make_mk_bm25(settings),
        dense=_make_mk_dense(settings),
        alpha=cfg.get("hybrid_alpha", 0.5),
        top_k=settings.top_k_retrieval,
    )
    return RAGPipeline(
        config=config,
        retriever=hybrid,
        generator=_make_generator(gid, settings),
        reranker=_make_reranker(settings),
    )


def _build_translate_retrieve(pid: str, gid: str, settings: Settings, cfg: dict) -> RAGPipeline:
    config = PipelineConfig(
        pipeline_id=f"{pid}_{gid}",
        name=f"Translate-Retrieve + {gid}",
        retrieval_type="dense",
        corpus_lang="en",
        generator_id=gid,
        translate_query=True,
        use_reranker=True,
        top_k_retrieve=settings.top_k_retrieval,
        top_k_rerank=settings.top_k_rerank,
    )
    return RAGPipeline(
        config=config,
        retriever=_make_en_dense(settings),
        generator=_make_generator(gid, settings),
        reranker=_make_reranker(settings),
        translator=Translator(),
    )


def _build_cross_lingual_embed(pid: str, gid: str, settings: Settings, cfg: dict) -> RAGPipeline:
    config = PipelineConfig(
        pipeline_id=f"{pid}_{gid}",
        name=f"Cross-lingual Embed + {gid}",
        retrieval_type="cross_lingual",
        corpus_lang="en",
        generator_id=gid,
        translate_query=False,
        use_reranker=True,
        top_k_retrieve=settings.top_k_retrieval,
        top_k_rerank=settings.top_k_rerank,
    )
    return RAGPipeline(
        config=config,
        retriever=CrossLingualRetriever(_make_en_dense(settings), top_k=settings.top_k_retrieval),
        generator=_make_generator(gid, settings),
        reranker=_make_reranker(settings),
    )


def _build_bilingual_fusion(pid: str, gid: str, settings: Settings, cfg: dict) -> RAGPipeline:
    config = PipelineConfig(
        pipeline_id=f"{pid}_{gid}",
        name=f"Bilingual Fusion + {gid}",
        retrieval_type="bilingual_fusion",
        corpus_lang="both",
        generator_id=gid,
        use_reranker=True,
        top_k_retrieve=settings.top_k_retrieval,
        top_k_rerank=settings.top_k_rerank,
    )
    retriever = BilingualFusionRetriever(
        mk_retriever=_make_mk_dense(settings),
        en_retriever=_make_en_dense(settings),
        top_k=settings.top_k_retrieval,
    )
    return RAGPipeline(
        config=config,
        retriever=retriever,
        generator=_make_generator(gid, settings),
        reranker=_make_reranker(settings),
    )


_PIPELINE_BUILDERS = {
    "mk_bm25": _build_mk_bm25,
    "mk_dense": _build_mk_dense,
    "mk_hybrid": _build_mk_hybrid,
    "translate_retrieve": _build_translate_retrieve,
    "cross_lingual_embed": _build_cross_lingual_embed,
    "bilingual_fusion": _build_bilingual_fusion,
}
