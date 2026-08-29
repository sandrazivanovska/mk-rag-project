"""
Step 13 — Verify the Gemini API key and show which models it can reach.

Run this before any paid experiment run:

    python scripts/mk/13_verify_gemini.py

It checks, in order:
  1. a key is visible to the SDK (via .env → os.environ),
  2. the key authenticates and can list models,
  3. the configured GEMINI_MODEL / GEMINI_MODEL_FAST actually exist,
  4. a real generation round-trip returns non-empty Macedonian text,
  5. the RAGAS judge LLM + embeddings can be constructed.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from src.utils.config import get_settings  # noqa: E402


def main() -> int:
    # Answers are Macedonian; the default Windows console codepage cannot encode
    # Cyrillic and would raise UnicodeEncodeError mid-report.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    settings = get_settings()  # also exports keys from .env into os.environ

    import os

    from src.evaluation.gemini_judge import build_gemini_client

    if settings.use_vertex:
        print(f"MODE Vertex AI (project={settings.vertex_project or '<unset>'}, "
              f"location={settings.vertex_location})")
        cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        print(f"     GOOGLE_APPLICATION_CREDENTIALS = {cred or '<unset, will use ADC>'}")
        if cred and not os.path.exists(cred):
            print(f"FAIL: that credentials file does not exist: {cred}")
            return 1
        if not settings.vertex_project:
            print("FAIL: USE_VERTEX=true but VERTEX_PROJECT is empty in .env.")
            return 1
    else:
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            print("FAIL: no GOOGLE_API_KEY. Put it in .env, or set USE_VERTEX=true.")
            print("      Get a key at https://aistudio.google.com/apikey")
            return 1
        # Only the placeholder is worth catching locally. Do NOT gate on key
        # prefix: AI Studio issues both the legacy "AIza..." and newer "AQ."
        # formats, and guessing which is valid rejects real keys.
        if key.endswith("...") or len(key) < 20:
            print(f"FAIL: GOOGLE_API_KEY is still the placeholder ({len(key)} chars).")
            print("      Edit .env with the real key — and save the file (Ctrl+S).")
            return 1
        print(f"MODE API key ({key[:6]}...{key[-4:]}, {len(key)} chars)")

    try:
        client = build_gemini_client(
            use_vertex=settings.use_vertex,
            project=settings.vertex_project,
            location=settings.vertex_location,
        )
        models = [m.name for m in client.models.list()]
    except Exception as exc:
        msg = str(exc)
        print(f"FAIL: could not authenticate: {msg[:300]}")
        if "ACCESS_TOKEN_TYPE_UNSUPPORTED" in msg:
            print()
            print("      This is the known 'AQ.' API-key problem: AI Studio issues")
            print("      the key, but the Gemini API rejects it. It is a Google-side")
            print("      issue, not a mistake in your key.")
            print("      Workaround: switch to Vertex AI — set USE_VERTEX=true and")
            print("      VERTEX_PROJECT=<your project id> in .env, then authenticate")
            print("      with a service-account JSON via GOOGLE_APPLICATION_CREDENTIALS.")
        return 1
    print(f"OK   authenticated — {len(models)} models visible")

    # API-key mode returns "models/<id>"; Vertex returns "publishers/google/<id>".
    def _short(name: str) -> str:
        return name.rsplit("/", 1)[-1]

    generative = sorted(
        _short(m) for m in models if "embedding" not in m and "aqa" not in m
    )
    gemini_only = [m for m in generative if m.startswith("gemini")]
    print(f"\nGemini models available ({len(gemini_only)} of {len(generative)} total):")
    for m in gemini_only:
        print(f"  - {m}")

    print("\nConfigured models:")
    ok = True
    for label, model_id in (
        ("GEMINI_MODEL", settings.gemini_model),
        ("GEMINI_MODEL_FAST", settings.gemini_model_fast),
        ("JUDGE_MODEL", settings.judge_model),
    ):
        mark = "OK  " if model_id in generative else "FAIL"
        if model_id not in generative:
            ok = False
        print(f"  {mark} {label} = {model_id}")
    if not ok:
        print("\n  ^ Pick one of the available names above and set it in .env.")
        return 1

    # Real round-trip through the project's own generator.
    print("\nGeneration round-trip:")
    from src.generator.generator import Generator, GeneratorConfig, ProviderType

    class _Doc:
        text = (
            "Скопје е главен и најголем град на Северна Македонија. "
            "Во градот живеат околу 526.000 жители."
        )

    for label, model_id in (
        ("gemini_pro", settings.gemini_model),
        ("gemini_flash", settings.gemini_model_fast),
    ):
        gen = Generator(GeneratorConfig(
            provider=ProviderType.GEMINI,
            model_id=model_id,
            use_vertex=settings.use_vertex,
            vertex_project=settings.vertex_project,
            vertex_location=settings.vertex_location,
        ))
        try:
            res = gen.generate(query="Кој е главниот град на Северна Македонија?",
                               context_docs=[_Doc()])
        except Exception as exc:
            print(f"  FAIL {label} ({model_id}): {exc}")
            return 1
        if not res.answer.strip():
            print(f"  FAIL {label} ({model_id}): empty answer — see warning above")
            return 1
        print(f"  OK   {label} ({model_id}) → {res.answer.strip()[:70]!r}")
        print(f"       {res.prompt_tokens} prompt / {res.completion_tokens} completion "
              f"tokens, {res.latency_ms:.0f} ms")

    # Judge construction (not a paid call — just proves the wiring imports).
    print("\nRAGAS judge:")
    from src.evaluation.evaluator import RAGEvaluator

    ev = RAGEvaluator(ragas_llm_model=settings.judge_model,
                      judge_provider=settings.judge_provider,
                      judge_embed_model=settings.judge_embed_model,
                      gemini_base_url=settings.gemini_openai_base_url,
                      use_vertex=settings.use_vertex,
                      vertex_project=settings.vertex_project,
                      vertex_location=settings.vertex_location)
    try:
        ev._build_judge_llm()
        ev._build_judge_embeddings()
    except Exception as exc:
        print(f"  FAIL judge_provider={settings.judge_provider}: {exc}")
        return 1
    print(f"  OK   judge_provider={settings.judge_provider}, model={settings.judge_model}")

    print("\nAll checks passed — safe to run the pilot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
