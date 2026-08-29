"""
Macedonian → English query translator.

Primary: Google Cloud Translation API
Fallback: deep-translator (no API key, rate-limited)
"""

from __future__ import annotations

import os
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger("translator")


class Translator:
    """
    Translates Macedonian queries to English for Pipeline 4
    (Translate-Retrieve).

    Example
    -------
    >>> t = Translator()
    >>> en = t.translate("Кој е главниот град на Македонија?")
    >>> print(en)
    "What is the capital of Macedonia?"
    """

    def __init__(
        self,
        source_lang: str = "mk",
        target_lang: str = "en",
        use_google_api: bool = True,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.use_google_api = use_google_api
        self._google_client = None
        self._creds = None  # cached ADC credentials, refreshed on expiry
        self._quota_project = None

    @property
    def google_client(self):
        if self._google_client is None:
            from google.cloud import translate_v2 as translate
            self._google_client = translate.Client()
        return self._google_client

    def translate(self, text: str) -> str:
        """
        Translate ``text`` from Macedonian to English.

        Falls back to deep-translator if Google API is unavailable.
        """
        if self.use_google_api and self._google_available():
            return self._translate_google(text)
        return self._translate_deep(text)

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Batch translate a list of strings."""
        return [self.translate(t) for t in texts]

    def _google_available(self) -> bool:
        """
        Is the paid Cloud Translation API usable?

        Application Default Credentials (from `gcloud auth application-default
        login`) are a valid way to authenticate but set NO environment variable,
        so checking only for GOOGLE_APPLICATION_CREDENTIALS silently fell
        through to the free scraper — which then failed and killed the run.
        """
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_TRANSLATE_API_KEY"):
            return True
        try:
            import google.auth

            creds, _ = google.auth.default()
            return creds is not None
        except Exception:
            return False

    def _translate_google(self, text: str) -> str:
        try:
            return self._translate_rest(text)
        except Exception as exc:
            logger.warning(f"Google Translate failed ({exc}), falling back to deep-translator")
            return self._translate_deep(text)

    def _translate_rest(self, text: str) -> str:
        """
        Cloud Translation v2 over plain REST, authenticated with ADC.

        Deliberately avoids the google-cloud-translate client library: it pins
        protobuf<7, while transformers needs >=5.27 and the installed stack runs
        protobuf 7.x. Installing it downgrades protobuf and breaks FlagEmbedding
        (and therefore all dense retrieval). google-auth is already a
        dependency, so REST costs nothing extra.
        """
        import json as _json
        import urllib.request

        import google.auth
        import google.auth.transport.requests

        if self._creds is None:
            self._creds, detected = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self._quota_project = (
                os.getenv("VERTEX_PROJECT")
                or getattr(self._creds, "quota_project_id", None)
                or detected
            )
        if not self._creds.valid:
            self._creds.refresh(google.auth.transport.requests.Request())

        body = _json.dumps({
            "q": text,
            "source": self.source_lang,
            "target": self.target_lang,
            "format": "text",
        }).encode()
        req = urllib.request.Request(
            "https://translation.googleapis.com/language/translate/v2",
            data=body,
            headers={
                "Authorization": f"Bearer {self._creds.token}",
                "Content-Type": "application/json",
                # Required when authenticating with USER Application Default
                # Credentials: without it the API returns 403 PERMISSION_DENIED
                # because there is no project to bill the quota against.
                "x-goog-user-project": self._quota_project or "",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        return data["data"]["translations"][0]["translatedText"]

    def _translate_deep(self, text: str) -> str:
        """
        Free-tier fallback. This scrapes Google Translate and is rate-limited
        and flaky, so it retries and then degrades to returning the source text
        rather than raising. A single failed translation must never abort a
        multi-hour experiment: one untranslated query costs one data point,
        an exception costs the whole run.
        """
        import random
        import time as _time

        from deep_translator import GoogleTranslator

        last_exc = None
        for attempt in range(1, 4):
            try:
                out = GoogleTranslator(
                    source=self.source_lang, target=self.target_lang
                ).translate(text)
                if out:
                    return out
                last_exc = RuntimeError("empty translation")
            except Exception as exc:
                last_exc = exc
            if attempt < 3:
                _time.sleep(2 * attempt + random.uniform(0, 0.5))

        logger.warning(
            f"Translation failed after 3 attempts ({last_exc}); "
            f"passing the source text through untranslated."
        )
        return text
