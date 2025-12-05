# SPDX-License-Identifier: AGPL-3.0-only
"""DeepL translation backend."""

try:
    import aiohttp
except ImportError:
    raise ImportError(
        "aiohttp is required for DeepL backend. "
        "Install with: pip install index-pdf-translation[deepl]"
    )

from .base import TranslationError


class DeepLTranslator:
    """
    DeepL API translation backend.

    High quality translation but requires API key.
    Language codes are converted with .upper() ("en" -> "EN").
    """

    DEFAULT_API_URL = "https://api-free.deepl.com/v2/translate"

    def __init__(self, api_key: str, api_url: str | None = None):
        """
        Args:
            api_key: DeepL API key
            api_url: DeepL API URL (None for Free API)
        """
        if not api_key:
            raise ValueError("DeepL API key is required")
        self._api_key = api_key
        self._api_url = api_url or self.DEFAULT_API_URL

    @property
    def name(self) -> str:
        return "deepl"

    async def translate(self, text: str, target_lang: str) -> str:
        """
        Translate text.

        Handles text with newlines. Single API call.

        Args:
            text: Text to translate
            target_lang: Target language ("en", "ja") - converted with .upper()
        """
        if not text.strip():
            return text

        params = {
            "auth_key": self._api_key,
            "text": text,
            "target_lang": target_lang.upper(),  # "en" -> "EN", "ja" -> "JA"
            "tag_handling": "xml",
            "formality": "more",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self._api_url, data=params) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["translations"][0]["text"]
                else:
                    error_text = await response.text()
                    raise TranslationError(
                        f"DeepL API error (status {response.status}): {error_text}"
                    )
