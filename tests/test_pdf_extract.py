# SPDX-License-Identifier: AGPL-3.0-only
"""
Tests for PDF text extraction functionality.

Uses CC BY 4.0 licensed academic papers as test fixtures.
"""

from pathlib import Path

import pytest

from index_pdf_translation.core.pdf_edit import (
    extract_text_coordinates_blocks,
    extract_text_coordinates_dict,
)


class TestExtractTextCoordinatesBlocks:
    """Tests for extract_text_coordinates_blocks function."""

    @pytest.mark.asyncio
    async def test_extracts_blocks_from_llama_pdf(
        self, sample_llama_pdf_bytes: bytes
    ) -> None:
        """Should extract text blocks from LLaMA paper."""
        blocks = await extract_text_coordinates_blocks(sample_llama_pdf_bytes)

        # Should have exactly 1 page (first page only)
        assert len(blocks) == 1

        # Should have some text blocks
        page_blocks = blocks[0]
        assert len(page_blocks) > 0

        # Each block should have required fields
        for block in page_blocks:
            assert "block_no" in block
            assert "text" in block
            assert "size" in block
            assert "coordinates" in block

    @pytest.mark.asyncio
    async def test_extracts_blocks_from_cot_pdf(
        self, sample_cot_pdf_bytes: bytes
    ) -> None:
        """Should extract text blocks from Chain-of-Thought paper."""
        blocks = await extract_text_coordinates_blocks(sample_cot_pdf_bytes)

        assert len(blocks) == 1
        page_blocks = blocks[0]
        assert len(page_blocks) > 0

    @pytest.mark.asyncio
    async def test_block_coordinates_are_valid(
        self, sample_llama_pdf_bytes: bytes
    ) -> None:
        """Block coordinates should be valid (x0 < x1, y0 < y1)."""
        blocks = await extract_text_coordinates_blocks(sample_llama_pdf_bytes)

        for page_blocks in blocks:
            for block in page_blocks:
                x0, y0, x1, y1 = block["coordinates"]
                assert x0 < x1, f"Invalid x coordinates: {x0} >= {x1}"
                assert y0 < y1, f"Invalid y coordinates: {y0} >= {y1}"

    @pytest.mark.asyncio
    async def test_block_font_size_positive(
        self, sample_llama_pdf_bytes: bytes
    ) -> None:
        """Block font sizes should be positive numbers."""
        blocks = await extract_text_coordinates_blocks(sample_llama_pdf_bytes)

        for page_blocks in blocks:
            for block in page_blocks:
                assert block["size"] > 0, "Font size should be positive"


class TestExtractTextCoordinatesDict:
    """Tests for extract_text_coordinates_dict function."""

    @pytest.mark.asyncio
    async def test_extracts_dict_from_llama_pdf(
        self, sample_llama_pdf_bytes: bytes
    ) -> None:
        """Should extract text with dict format from LLaMA paper."""
        blocks = await extract_text_coordinates_dict(sample_llama_pdf_bytes)

        assert len(blocks) == 1
        page_blocks = blocks[0]
        assert len(page_blocks) > 0

    @pytest.mark.asyncio
    async def test_dict_has_page_no(
        self, sample_llama_pdf_bytes: bytes
    ) -> None:
        """Dict format should include page_no field."""
        blocks = await extract_text_coordinates_dict(sample_llama_pdf_bytes)

        for page_blocks in blocks:
            for block in page_blocks:
                assert "page_no" in block
                assert block["page_no"] == 0  # First page


class TestTextContent:
    """Tests for extracted text content."""

    @pytest.mark.asyncio
    async def test_llama_paper_contains_title(
        self, sample_llama_pdf_bytes: bytes
    ) -> None:
        """LLaMA paper should contain 'LLaMA' in extracted text."""
        blocks = await extract_text_coordinates_blocks(sample_llama_pdf_bytes)

        all_text = ""
        for page_blocks in blocks:
            for block in page_blocks:
                all_text += block["text"]

        assert "LLaMA" in all_text, "Should contain paper title"

    @pytest.mark.asyncio
    async def test_cot_paper_contains_title(
        self, sample_cot_pdf_bytes: bytes
    ) -> None:
        """Chain-of-Thought paper should contain relevant keywords."""
        blocks = await extract_text_coordinates_blocks(sample_cot_pdf_bytes)

        all_text = ""
        for page_blocks in blocks:
            for block in page_blocks:
                all_text += block["text"]

        # Check for key terms from the paper
        assert "Chain" in all_text or "Thought" in all_text or "Reasoning" in all_text
