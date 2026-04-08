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
        return bool(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            or os.getenv("GOOGLE_TRANSLATE_API_KEY")
        )

    def _translate_google(self, text: str) -> str:
        try:
            result = self.google_client.translate(
                text,
                source_language=self.source_lang,
                target_language=self.target_lang,
            )
            return result["translatedText"]
        except Exception as exc:
            logger.warning(f"Google Translate failed ({exc}), falling back to deep-translator")
            return self._translate_deep(text)

    def _translate_deep(self, text: str) -> str:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(
            source=self.source_lang, target=self.target_lang
        ).translate(text)
