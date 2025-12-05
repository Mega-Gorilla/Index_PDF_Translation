# SPDX-License-Identifier: AGPL-3.0-only
"""Base class for translation backends."""

from typing import Protocol, runtime_checkable


class TranslationError(Exception):
    """Error raised during translation."""
    pass


@runtime_checkable
class TranslatorBackend(Protocol):
    """
    Translation backend protocol.

    Each backend implements only the translate() method.
    Batch processing (newline joining) is handled by translate_blocks().
    """

    @property
    def name(self) -> str:
        """Backend name ("google", "deepl")."""
        ...

    async def translate(self, text: str, target_lang: str) -> str:
        """
        Translate text.

        Accepts text with newlines and preserves them during translation.

        Args:
            text: Text to translate (may contain newlines)
            target_lang: Target language code ("en", "ja")

        Returns:
            Translated text

        Raises:
            TranslationError: When translation fails
        """
        ...
