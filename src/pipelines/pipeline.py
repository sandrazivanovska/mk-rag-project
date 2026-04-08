"""
RAGPipeline: wires together retrieval → (optional) reranking → generation.

Each of the 6 retrieval setups × 2 generators = 12 pipeline variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.retrieval.base import BaseRetriever, RetrievedDoc
from src.reranker.reranker import Reranker
from src.generator.generator import Generator, GenerationResult
from src.generator.translator import Translator
from src.utils.logging import get_logger

logger = get_logger("pipeline")


@dataclass
class PipelineConfig:
    """Configuration for a single pipeline variant."""

    pipeline_id: str
    name: str
    retrieval_type: str        # bm25 | dense | hybrid | translate_retrieve | cross_lingual | bilingual_fusion
    corpus_lang: str           # mk | en | both
    generator_id: str          # gpt4o | claude_sonnet
    translate_query: bool = False
    use_reranker: bool = True
    top_k_retrieve: int = 50
    top_k_rerank: int = 5
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete result for a single query through a pipeline."""

    pipeline_id: str
    generator_id: str
    query: str
    translated_query: Optional[str]
    retrieved_docs: list[RetrievedDoc]
    reranked_docs: Optional[list[RetrievedDoc]]
    generation: GenerationResult

    @property
    def final_docs(self) -> list[RetrievedDoc]:
        return self.reranked_docs if self.reranked_docs else self.retrieved_docs

    def to_dict(self) -> dict:
        return {
            "pipeline_id": self.pipeline_id,
            "generator_id": self.generator_id,
            "query": self.query,
            "translated_query": self.translated_query,
            "answer": self.generation.answer,
            "retrieved_count": len(self.retrieved_docs),
            "final_context_count": len(self.final_docs),
            "latency_ms": self.generation.latency_ms,
            "prompt_tokens": self.generation.prompt_tokens,
            "completion_tokens": self.generation.completion_tokens,
        }


class RAGPipeline:
    """
    Full RAG pipeline: query → retrieve → rerank (optional) → generate.

    Example
    -------
    >>> pipeline = RAGPipeline(
    ...     config=PipelineConfig(pipeline_id="mk_dense", ...),
    ...     retriever=dense_retriever,
    ...     generator=claude_generator,
    ...     reranker=reranker,
    ... )
    >>> result = pipeline.run("Кој е главниот град на Македонија?")
    >>> print(result.generation.answer)
    """

    def __init__(
        self,
        config: PipelineConfig,
        retriever: BaseRetriever,
        generator: Generator,
        reranker: Optional[Reranker] = None,
        translator: Optional[Translator] = None,
    ):
        self.config = config
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.translator = translator

    def run(self, query: str) -> PipelineResult:
        """
        Execute the full pipeline for a single query.

        Args:
            query: Input query in Macedonian.

        Returns:
            PipelineResult with retrieved docs, reranked docs, and generation.
        """
        translated_query: Optional[str] = None

        # Step 1: Optionally translate the query
        retrieval_query = query
        if self.config.translate_query and self.translator:
            translated_query = self.translator.translate(query)
            retrieval_query = translated_query
            logger.debug(f"Translated query: '{query}' → '{translated_query}'")

        # Step 2: Retrieve
        retrieved = self.retriever.retrieve(
            retrieval_query, top_k=self.config.top_k_retrieve
        )
        logger.debug(
            f"[{self.config.pipeline_id}] Retrieved {len(retrieved)} docs "
            f"for query: '{query[:60]}...'"
        )

        # Step 3: Optional reranking
        reranked: Optional[list[RetrievedDoc]] = None
        if self.config.use_reranker and self.reranker and retrieved:
            # Rerank using the original MK query (not translated)
            reranked = self.reranker.rerank(query, retrieved, top_k=self.config.top_k_rerank)
            logger.debug(f"Reranked to {len(reranked)} docs")

        # Step 4: Generate
        context_docs = reranked if reranked else retrieved
        context_docs = context_docs[: self.config.top_k_rerank]
        generation = self.generator.generate(query, context_docs)

        return PipelineResult(
            pipeline_id=self.config.pipeline_id,
            generator_id=self.config.generator_id,
            query=query,
            translated_query=translated_query,
            retrieved_docs=retrieved,
            reranked_docs=reranked,
            generation=generation,
        )

    def run_batch(self, queries: list[str]) -> list[PipelineResult]:
        """Run the pipeline on multiple queries."""
        results = []
        for i, query in enumerate(queries):
            logger.info(
                f"[{self.config.pipeline_id}] Query {i+1}/{len(queries)}: {query[:60]}"
            )
            results.append(self.run(query))
        return results
