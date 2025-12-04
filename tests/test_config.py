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
        assert SUPPORTED_LANGUAGES["en"]["deepl"] == "EN"
        assert SUPPORTED_LANGUAGES["en"]["spacy"] == "en_core_web_sm"

    def test_japanese_supported(self) -> None:
        """Japanese should be a supported language."""
        assert "ja" in SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES["ja"]["deepl"] == "JA"
        assert SUPPORTED_LANGUAGES["ja"]["spacy"] == "ja_core_news_sm"


class TestTranslationConfig:
    """Tests for TranslationConfig dataclass."""

    def test_config_with_api_key(self, mock_api_key: str) -> None:
        """Config should be created with explicit api_key."""
        config = TranslationConfig(api_key=mock_api_key)
        assert config.api_key == mock_api_key

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config should read api_key from environment variable."""
        monkeypatch.setenv("DEEPL_API_KEY", "env-test-key")
        config = TranslationConfig()
        assert config.api_key == "env-test-key"

    def test_config_missing_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config should raise ValueError when api_key is missing."""
        monkeypatch.delenv("DEEPL_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            TranslationConfig(api_key="")

    def test_config_default_values(self, mock_api_key: str) -> None:
        """Config should have correct default values."""
        config = TranslationConfig(api_key=mock_api_key)

        assert config.source_lang == "en"
        assert config.target_lang == "ja"
        assert config.add_logo is True
        assert config.debug is False
        assert "api-free.deepl.com" in config.api_url

    def test_config_custom_values(self, mock_api_key: str) -> None:
        """Config should accept custom values."""
        config = TranslationConfig(
            api_key=mock_api_key,
            api_url="https://api.deepl.com/v2/translate",
            source_lang="ja",
            target_lang="en",
            add_logo=False,
            debug=True,
        )

        assert config.source_lang == "ja"
        assert config.target_lang == "en"
        assert config.add_logo is False
        assert config.debug is True
        assert config.api_url == "https://api.deepl.com/v2/translate"

    def test_config_api_url_from_env(
        self, mock_api_key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config should read api_url from environment variable."""
        monkeypatch.setenv("DEEPL_API_URL", "https://custom.api.url")
        config = TranslationConfig(api_key=mock_api_key)
        assert config.api_url == "https://custom.api.url"

    def test_config_invalid_source_lang(self, mock_api_key: str) -> None:
        """Config should raise ValueError for unsupported source language."""
        with pytest.raises(ValueError, match="Unsupported source language"):
            TranslationConfig(api_key=mock_api_key, source_lang="xx")

    def test_config_invalid_target_lang(self, mock_api_key: str) -> None:
        """Config should raise ValueError for unsupported target language."""
        with pytest.raises(ValueError, match="Unsupported target language"):
            TranslationConfig(api_key=mock_api_key, target_lang="xx")

    def test_deepl_target_lang_property(self, mock_api_key: str) -> None:
        """deepl_target_lang property should return correct DeepL code."""
        config = TranslationConfig(api_key=mock_api_key, target_lang="ja")
        assert config.deepl_target_lang == "JA"

        config = TranslationConfig(api_key=mock_api_key, target_lang="en")
        assert config.deepl_target_lang == "EN"

    def test_deepl_source_lang_property(self, mock_api_key: str) -> None:
        """deepl_source_lang property should return correct DeepL code."""
        config = TranslationConfig(api_key=mock_api_key, source_lang="en")
        assert config.deepl_source_lang == "EN"

        config = TranslationConfig(api_key=mock_api_key, source_lang="ja")
        assert config.deepl_source_lang == "JA"
