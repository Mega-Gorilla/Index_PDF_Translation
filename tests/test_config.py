# SPDX-License-Identifier: AGPL-3.0-only
"""
Tests for index_pdf_translation.config module.
"""

import os

import pytest

from index_pdf_translation.config import (
    SUPPORTED_LANGUAGES,
    TranslationConfig,
)


class TestSupportedLanguages:
    """Tests for SUPPORTED_LANGUAGES constant."""

    def test_english_supported(self) -> None:
        """English should be a supported language."""
        assert "en" in SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES["en"]["spacy"] == "en_core_web_sm"

    def test_japanese_supported(self) -> None:
        """Japanese should be a supported language."""
        assert "ja" in SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES["ja"]["spacy"] == "ja_core_news_sm"


class TestTranslationConfig:
    """Tests for TranslationConfig dataclass."""

    def test_config_default_backend_is_google(self) -> None:
        """Default backend should be Google."""
        config = TranslationConfig()
        assert config.backend == "google"

    def test_config_google_no_api_key_required(self) -> None:
        """Google backend should not require API key."""
        config = TranslationConfig(backend="google")
        assert config.backend == "google"
        # api_key can be empty for Google

    def test_config_deepl_requires_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DeepL backend should require API key."""
        monkeypatch.delenv("DEEPL_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DeepL API key required"):
            TranslationConfig(backend="deepl", api_key="")

    def test_config_deepl_with_api_key(self) -> None:
        """DeepL backend should work with API key."""
        config = TranslationConfig(backend="deepl", api_key="test-key")
        assert config.backend == "deepl"
        assert config.api_key == "test-key"

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config should read api_key from environment variable."""
        monkeypatch.setenv("DEEPL_API_KEY", "env-test-key")
        config = TranslationConfig(backend="deepl")
        assert config.api_key == "env-test-key"

    def test_config_default_values(self) -> None:
        """Config should have correct default values."""
        config = TranslationConfig()

        assert config.backend == "google"
        assert config.source_lang == "en"
        assert config.target_lang == "ja"
        assert config.add_logo is True
        assert config.debug is False

    def test_config_custom_values(self) -> None:
        """Config should accept custom values."""
        config = TranslationConfig(
            backend="deepl",
            api_key="test-key",
            api_url="https://api.deepl.com/v2/translate",
            source_lang="ja",
            target_lang="en",
            add_logo=False,
            debug=True,
        )

        assert config.backend == "deepl"
        assert config.source_lang == "ja"
        assert config.target_lang == "en"
        assert config.add_logo is False
        assert config.debug is True
        assert config.api_url == "https://api.deepl.com/v2/translate"

    def test_config_api_url_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config should read api_url from environment variable."""
        monkeypatch.setenv("DEEPL_API_URL", "https://custom.api.url")
        config = TranslationConfig()
        assert config.api_url == "https://custom.api.url"

    def test_config_invalid_source_lang(self) -> None:
        """Config should raise ValueError for unsupported source language."""
        with pytest.raises(ValueError, match="Unsupported source language"):
            TranslationConfig(source_lang="xx")

    def test_config_invalid_target_lang(self) -> None:
        """Config should raise ValueError for unsupported target language."""
        with pytest.raises(ValueError, match="Unsupported target language"):
            TranslationConfig(target_lang="xx")

    def test_config_create_translator_google(self) -> None:
        """create_translator() should create Google backend."""
        config = TranslationConfig(backend="google")
        translator = config.create_translator()
        assert translator.name == "google"

    def test_config_create_translator_deepl(self) -> None:
        """create_translator() should create DeepL backend."""
        config = TranslationConfig(backend="deepl", api_key="test-key")
        translator = config.create_translator()
        assert translator.name == "deepl"
