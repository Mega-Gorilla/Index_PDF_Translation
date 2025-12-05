# SPDX-License-Identifier: AGPL-3.0-only
"""
Tests for index_pdf_translation.core.translate module.
"""

import pytest

from index_pdf_translation import pdf_translate, TranslationConfig, TranslationResult


class TestTranslationResult:
    """Tests for TranslationResult dataclass."""

    def test_translation_result_has_pdf(self) -> None:
        """TranslationResult should have pdf field."""
        result = TranslationResult(pdf=b"test")
        assert result.pdf == b"test"

    def test_translation_result_debug_pdf_optional(self) -> None:
        """debug_pdf field should be optional."""
        result = TranslationResult(pdf=b"test")
        assert result.debug_pdf is None

    def test_translation_result_with_debug_pdf(self) -> None:
        """TranslationResult should accept debug_pdf."""
        result = TranslationResult(pdf=b"test", debug_pdf=b"debug")
        assert result.pdf == b"test"
        assert result.debug_pdf == b"debug"


class TestPdfTranslateDebugMode:
    """Tests for pdf_translate with debug mode."""

    @pytest.mark.asyncio
    async def test_debug_mode_returns_debug_pdf(
        self, sample_en_pdf_bytes: bytes
    ) -> None:
        """Debug mode should return debug_pdf with histograms + block frames."""
        config = TranslationConfig(debug=True)
        result = await pdf_translate(
            sample_en_pdf_bytes, config=config, disable_translate=True
        )
        assert result is not None
        assert result.debug_pdf is not None

        # Verify debug PDF structure (histograms + original pages)
        import fitz

        doc = fitz.open(stream=result.debug_pdf, filetype="pdf")
        # At least 4 pages: 3 histograms + 1 original page
        assert len(doc) >= 4
        doc.close()

    @pytest.mark.asyncio
    async def test_non_debug_mode_no_debug_pdf(
        self, sample_en_pdf_bytes: bytes
    ) -> None:
        """Non-debug mode should not return debug_pdf."""
        config = TranslationConfig(debug=False)
        result = await pdf_translate(
            sample_en_pdf_bytes, config=config, disable_translate=True
        )
        assert result is not None
        assert result.debug_pdf is None

    @pytest.mark.asyncio
    async def test_result_has_translated_pdf(
        self, sample_en_pdf_bytes: bytes
    ) -> None:
        """Result should always have translated PDF."""
        config = TranslationConfig()
        result = await pdf_translate(
            sample_en_pdf_bytes, config=config, disable_translate=True
        )
        assert result is not None
        assert result.pdf is not None
        assert len(result.pdf) > 0

        # Verify it's a valid PDF
        import fitz

        doc = fitz.open(stream=result.pdf, filetype="pdf")
        assert len(doc) >= 1
        doc.close()


class TestPdfTranslateBasic:
    """Basic tests for pdf_translate function."""

    @pytest.mark.asyncio
    async def test_disable_translate_skips_translation(
        self, sample_en_pdf_bytes: bytes
    ) -> None:
        """disable_translate=True should skip translation."""
        config = TranslationConfig()
        result = await pdf_translate(
            sample_en_pdf_bytes, config=config, disable_translate=True
        )
        assert result is not None
        assert result.pdf is not None
