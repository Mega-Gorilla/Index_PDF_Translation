# SPDX-License-Identifier: AGPL-3.0-only
"""
Tests for index_pdf_translation.resources module.
"""

from pathlib import Path

import pytest

from index_pdf_translation.resources import (
    get_font_path,
    get_font_path_for_lang,
    get_logo_path,
)


class TestGetFontPath:
    """Tests for get_font_path function."""

    def test_liberation_serif_regular(self) -> None:
        """LiberationSerif-Regular.ttf should be found."""
        path = get_font_path("LiberationSerif-Regular.ttf")
        assert path.exists()
        assert path.name == "LiberationSerif-Regular.ttf"

    def test_liberation_serif_bold(self) -> None:
        """LiberationSerif-Bold.ttf should be found."""
        path = get_font_path("LiberationSerif-Bold.ttf")
        assert path.exists()
        assert path.name == "LiberationSerif-Bold.ttf"

    def test_ipam_font(self) -> None:
        """ipam.ttf (Japanese font) should be found."""
        path = get_font_path("ipam.ttf")
        assert path.exists()
        assert path.name == "ipam.ttf"

    def test_nonexistent_font_raises(self) -> None:
        """Non-existent font should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Font file not found"):
            get_font_path("nonexistent-font.ttf")

    def test_returns_path_object(self) -> None:
        """get_font_path should return a Path object."""
        path = get_font_path("LiberationSerif-Regular.ttf")
        assert isinstance(path, Path)

    def test_caching(self) -> None:
        """Font paths should be cached for performance."""
        path1 = get_font_path("LiberationSerif-Regular.ttf")
        path2 = get_font_path("LiberationSerif-Regular.ttf")
        # Should return the exact same object due to caching
        assert path1 is path2


class TestGetLogoPath:
    """Tests for get_logo_path function."""

    def test_logo_exists(self) -> None:
        """Logo file should be found."""
        path = get_logo_path()
        assert path.exists()
        assert path.name == "indqx_qr.png"

    def test_returns_path_object(self) -> None:
        """get_logo_path should return a Path object."""
        path = get_logo_path()
        assert isinstance(path, Path)

    def test_caching(self) -> None:
        """Logo path should be cached for performance."""
        path1 = get_logo_path()
        path2 = get_logo_path()
        # Should return the exact same object due to caching
        assert path1 is path2


class TestGetFontPathForLang:
    """Tests for get_font_path_for_lang function."""

    def test_english_font(self) -> None:
        """English should use Liberation Serif."""
        path = get_font_path_for_lang("en")
        assert path.exists()
        assert "LiberationSerif" in path.name

    def test_japanese_font(self) -> None:
        """Japanese should use IPA Mincho."""
        path = get_font_path_for_lang("ja")
        assert path.exists()
        assert path.name == "ipam.ttf"

    def test_unknown_lang_defaults_to_english(self) -> None:
        """Unknown language should default to English font."""
        path = get_font_path_for_lang("xx")
        assert path.exists()
        assert "LiberationSerif" in path.name
