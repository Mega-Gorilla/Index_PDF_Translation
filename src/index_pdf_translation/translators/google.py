# SPDX-License-Identifier: AGPL-3.0-only
"""Google Translate backend using deep-translator."""

import asyncio

from deep_translator import GoogleTranslator as DTGoogleTranslator
from deep_translator.exceptions import TranslationNotFound

from .base import TranslationError


class GoogleTranslator:
    """
    Google Translate backend.

    No API key required. Uses deep-translator library.
    Language codes use internal format ("en", "ja") as-is.
    """

    @property
    def name(self) -> str:
        return "google"

    async def translate(self, text: str, target_lang: str) -> str:
        """
        Translate text.

        Handles text with newlines. Single API call.

        Args:
            text: Text to translate
            target_lang: Target language ("en", "ja") - used as-is
        """
        if not text.strip():
            return text

        def _translate() -> str:
            try:
                # deep-translator accepts "en", "ja" as-is
                translator = DTGoogleTranslator(
                    source="auto",
                    target=target_lang
                )
                return translator.translate(text)
            except TranslationNotFound as e:
                raise TranslationError(f"Translation failed: {e}")
            except Exception as e:
                raise TranslationError(f"Google Translate error: {e}")

        return await asyncio.to_thread(_translate)
