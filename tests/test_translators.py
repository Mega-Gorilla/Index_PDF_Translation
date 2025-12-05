# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for translation backends (using mocks)."""

import os

import pytest
from unittest.mock import patch, MagicMock

from index_pdf_translation.translators import GoogleTranslator, TranslationError


class TestGoogleTranslator:
    """Tests for Google Translate backend."""

    def test_name(self):
        """Backend name should be 'google'."""
        translator = GoogleTranslator()
        assert translator.name == "google"

    @pytest.mark.asyncio
    async def test_translate_empty_string(self):
        """Empty string should not call API."""
        translator = GoogleTranslator()
        result = await translator.translate("", "ja")
        assert result == ""

    @pytest.mark.asyncio
    async def test_translate_whitespace_only(self):
        """Whitespace-only string should not call API."""
        translator = GoogleTranslator()
        result = await translator.translate("   ", "ja")
        assert result == "   "

    @pytest.mark.asyncio
    async def test_translate_simple_mocked(self):
        """Basic translation test with mock."""
        translator = GoogleTranslator()

        with patch(
            "index_pdf_translation.translators.google.DTGoogleTranslator"
        ) as mock_dt:
            mock_instance = MagicMock()
            mock_instance.translate.return_value = "Hello"
            mock_dt.return_value = mock_instance

            result = await translator.translate("Hello", "ja")

            assert result == "Hello"
            mock_dt.assert_called_once_with(source="auto", target="ja")
            mock_instance.translate.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_translate_with_separator_mocked(self):
        """Test that separator is preserved in translation."""
        translator = GoogleTranslator()

        with patch(
            "index_pdf_translation.translators.google.DTGoogleTranslator"
        ) as mock_dt:
            mock_instance = MagicMock()
            # Simulate separator preservation
            mock_instance.translate.return_value = "Hello[[[BR]]]World[[[BR]]]Good"
            mock_dt.return_value = mock_instance

            result = await translator.translate(
                "Hello[[[BR]]]World[[[BR]]]Good morning", "ja"
            )

            assert "[[[BR]]]" in result
            parts = result.split("[[[BR]]]")
            assert len(parts) == 3

    @pytest.mark.asyncio
    async def test_translate_error_handling(self):
        """Test error handling."""
        translator = GoogleTranslator()

        with patch(
            "index_pdf_translation.translators.google.DTGoogleTranslator"
        ) as mock_dt:
            mock_instance = MagicMock()
            mock_instance.translate.side_effect = Exception("API Error")
            mock_dt.return_value = mock_instance

            with pytest.raises(TranslationError, match="Google Translate error"):
                await translator.translate("Hello", "ja")


class TestDeepLTranslator:
    """Tests for DeepL translation backend (limited, requires API key)."""

    def test_requires_api_key(self):
        """DeepL should require API key."""
        from index_pdf_translation.translators import get_deepl_translator
        DeepLTranslator = get_deepl_translator()

        with pytest.raises(ValueError, match="API key is required"):
            DeepLTranslator(api_key="")

    def test_name(self):
        """Backend name should be 'deepl'."""
        from index_pdf_translation.translators import get_deepl_translator
        DeepLTranslator = get_deepl_translator()

        translator = DeepLTranslator(api_key="dummy-key")
        assert translator.name == "deepl"

    def test_custom_api_url(self):
        """DeepL should accept custom API URL."""
        from index_pdf_translation.translators import get_deepl_translator
        DeepLTranslator = get_deepl_translator()

        translator = DeepLTranslator(
            api_key="dummy-key",
            api_url="https://api.deepl.com/v2/translate"
        )
        assert translator._api_url == "https://api.deepl.com/v2/translate"


# Integration tests (use real API, skip in CI)
@pytest.mark.integration
@pytest.mark.skipif(
    "CI" in os.environ,
    reason="Skip integration tests in CI"
)
class TestGoogleTranslatorIntegration:
    """Google Translate integration tests (uses real API)."""

    @pytest.mark.asyncio
    async def test_real_translation(self):
        """Test real Google Translate API call."""
        translator = GoogleTranslator()
        result = await translator.translate("Hello", "ja")
        assert result
        assert result != "Hello"

    @pytest.mark.asyncio
    async def test_separator_preserved(self):
        """Test that separator is preserved in real translation."""
        translator = GoogleTranslator()
        text = "Hello[[[BR]]]World[[[BR]]]Good morning"
        result = await translator.translate(text, "ja")

        parts = result.split("[[[BR]]]")
        assert len(parts) == 3, f"Expected 3 parts, got {len(parts)}: {result}"


@pytest.mark.integration
@pytest.mark.skipif(
    "DEEPL_API_KEY" not in os.environ,
    reason="DEEPL_API_KEY not set"
)
class TestDeepLTranslatorIntegration:
    """DeepL integration tests (uses real API, requires DEEPL_API_KEY)."""

    @pytest.mark.asyncio
    async def test_real_translation(self):
        """Test real DeepL API call."""
        from index_pdf_translation.translators import get_deepl_translator
        DeepLTranslator = get_deepl_translator()

        translator = DeepLTranslator(api_key=os.environ["DEEPL_API_KEY"])
        result = await translator.translate("Hello", "ja")
        assert result
        assert result != "Hello"

    @pytest.mark.asyncio
    async def test_separator_preserved(self):
        """Test that separator is preserved in DeepL translation."""
        from index_pdf_translation.translators import get_deepl_translator
        DeepLTranslator = get_deepl_translator()

        translator = DeepLTranslator(api_key=os.environ["DEEPL_API_KEY"])
        text = "Hello[[[BR]]]World[[[BR]]]Good morning"
        result = await translator.translate(text, "ja")

        parts = result.split("[[[BR]]]")
        assert len(parts) == 3, f"Expected 3 parts, got {len(parts)}: {result}"
