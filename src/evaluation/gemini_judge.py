"""
Gemini-backed judge components for RAGAS.

RAGAS needs a LangChain chat model and a LangChain embeddings object. Both are
built here directly on the `google-genai` SDK so they work identically whether
we authenticate with an API key (Gemini Developer API) or with Google Cloud
credentials (Vertex AI).

We deliberately do NOT use langchain-google-genai or langchain-google-vertexai:
the first pins protobuf<5 and breaks FlagEmbedding, the second downgrades
google-genai. See docs and the project memory for the full story.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.embeddings import Embeddings

from src.utils.logging import get_logger

logger = get_logger("gemini_judge")


def build_gemini_client(
    *,
    use_vertex: bool = False,
    project: str = "",
    location: str = "us-central1",
):
    """Construct a google-genai client in either API-key or Vertex mode."""
    from google import genai

    if use_vertex:
        if not project:
            raise RuntimeError("use_vertex=True requires a project id (VERTEX_PROJECT).")
        return genai.Client(vertexai=True, project=project, location=location)

    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No GOOGLE_API_KEY set and USE_VERTEX is false.")
    return genai.Client(api_key=key)


class GeminiEmbeddings(Embeddings):
    """
    Minimal LangChain Embeddings adapter over google-genai.

    RAGAS uses this for answer_relevancy and answer_correctness. Implementing it
    against the native SDK avoids depending on Vertex's OpenAI-compatibility
    layer, whose embeddings support is not guaranteed.
    """

    def __init__(
        self,
        model: str = "text-embedding-004",
        *,
        use_vertex: bool = False,
        project: str = "",
        location: str = "us-central1",
        batch_size: int = 32,
    ):
        self.model = model
        self.batch_size = batch_size
        self._client = build_gemini_client(
            use_vertex=use_vertex, project=project, location=location
        )

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self._client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            out.extend(e.values for e in resp.embeddings)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(list(texts), "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def vertex_openai_credentials(project: str, location: str) -> dict:
    """
    Base URL + bearer token for Vertex's OpenAI-compatible chat endpoint.

    The token is a short-lived OAuth credential (about an hour). It is minted
    fresh each time the judge is constructed, which happens once per pipeline
    evaluation, so a long overall run is fine — but a single evaluation lasting
    over an hour could see it expire.
    """
    import google.auth
    import google.auth.transport.requests

    creds, detected_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(google.auth.transport.requests.Request())
    project = project or detected_project or ""
    if not project:
        raise RuntimeError("Could not determine a Google Cloud project for Vertex.")
    return {
        "api_key": creds.token,
        "base_url": (
            f"https://{location}-aiplatform.googleapis.com/v1/"
            f"projects/{project}/locations/{location}/endpoints/openapi"
        ),
    }
