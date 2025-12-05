# SPDX-License-Identifier: AGPL-3.0-only
"""
Index PDF Translation - Configuration

Manages translation settings and language configuration.
"""

import os
from dataclasses import dataclass, field
from typing import Literal, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from index_pdf_translation.translators import TranslatorBackend


class LanguageConfig(TypedDict):
    """Language configuration type definition."""

    spacy: str  # spaCy model name


# Language configuration
# - Language code conversion is handled by each TranslatorBackend
# - Google: uses "en", "ja" as-is
# - DeepL: converts with .upper() to "EN", "JA"
SUPPORTED_LANGUAGES: dict[str, LanguageConfig] = {
    "en": {"spacy": "en_core_web_sm"},
    "ja": {"spacy": "ja_core_news_sm"},
}

# Default output directory
DEFAULT_OUTPUT_DIR: str = "./output/"

# Translation backend type
TranslatorBackendType = Literal["google", "deepl", "openai"]


@dataclass
class TranslationConfig:
    """
    Translation configuration dataclass.

    Default is Google Translate (no API key required).
    Use DeepL for higher quality translation.
    Use OpenAI for customizable GPT-based translation.

    Attributes:
        backend: Translation backend ("google", "deepl", or "openai")
        api_key: DeepL API key (required only for backend="deepl")
        api_url: DeepL API URL (used only for backend="deepl")
        openai_api_key: OpenAI API key (required only for backend="openai")
        openai_model: OpenAI model (default: "gpt-4o-mini")
        openai_system_prompt: Custom system prompt for OpenAI (optional)
        source_lang: Source language code (default: "en")
        target_lang: Target language code (default: "ja")
        add_logo: Add logo watermark (default: True)
        debug: Debug mode (default: False)

    Examples:
        >>> # Google Translate (default, no API key required)
        >>> config = TranslationConfig()

        >>> # DeepL Translate (high quality)
        >>> config = TranslationConfig(
        ...     backend="deepl",
        ...     api_key="your-deepl-key"
        ... )

        >>> # OpenAI GPT Translate (customizable)
        >>> config = TranslationConfig(
        ...     backend="openai",
        ...     openai_api_key="your-openai-key"
        ... )
    """

    backend: TranslatorBackendType = "google"
    api_key: str = field(
        default_factory=lambda: os.environ.get("DEEPL_API_KEY", "")
    )
    api_url: str = field(
        default_factory=lambda: os.environ.get(
            "DEEPL_API_URL",
            "https://api-free.deepl.com/v2/translate"
        )
    )
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    openai_model: str = "gpt-4o-mini"
    openai_system_prompt: str | None = None
    source_lang: str = "en"
    target_lang: str = "ja"
    add_logo: bool = True
    debug: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # DeepL backend requires API key
        if self.backend == "deepl" and not self.api_key:
            raise ValueError(
                "DeepL API key required when using 'deepl' backend. "
                "Set DEEPL_API_KEY environment variable or pass api_key parameter. "
                "Or use backend='google' for API-key-free translation."
            )

        # OpenAI backend requires API key
        if self.backend == "openai" and not self.openai_api_key:
            raise ValueError(
                "OpenAI API key required when using 'openai' backend. "
                "Set OPENAI_API_KEY environment variable or pass openai_api_key parameter. "
                "Or use backend='google' for API-key-free translation."
            )

        # Validate language codes
        if self.source_lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported source language: {self.source_lang}. "
                f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        if self.target_lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported target language: {self.target_lang}. "
                f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )

    def create_translator(self) -> "TranslatorBackend":
        """
        Create translator backend based on configuration.

        Returns:
            TranslatorBackend instance

        Raises:
            ValueError: When unknown backend is specified
            ImportError: When DeepL/OpenAI backend is used without dependencies
        """
        from index_pdf_translation.translators import GoogleTranslator

        if self.backend == "google":
            return GoogleTranslator()
        elif self.backend == "deepl":
            from index_pdf_translation.translators import get_deepl_translator
            DeepLTranslator = get_deepl_translator()
            return DeepLTranslator(self.api_key, self.api_url)
        elif self.backend == "openai":
            from index_pdf_translation.translators import get_openai_translator
            OpenAITranslator = get_openai_translator()
            return OpenAITranslator(
                api_key=self.openai_api_key,
                model=self.openai_model,
                source_lang=self.source_lang,
                system_prompt=self.openai_system_prompt,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
