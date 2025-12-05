# SPDX-License-Identifier: AGPL-3.0-only
"""Translation backend module."""

from .base import TranslatorBackend, TranslationError
from .google import GoogleTranslator

__all__ = [
    "TranslatorBackend",
    "TranslationError",
    "GoogleTranslator",
]


def get_deepl_translator():
    """Get DeepLTranslator class (requires aiohttp)."""
    from .deepl import DeepLTranslator
    return DeepLTranslator
