# SPDX-License-Identifier: AGPL-3.0-only
"""
Package resource access utilities.

Provides functions to access bundled resources (fonts, logos) using importlib.resources,
ensuring compatibility with installed packages and editable installs.

This subpackage contains static resources:
- fonts/: Font files for PDF text rendering
- data/: Logo and other assets
"""

from importlib import resources
from pathlib import Path
from typing import Optional

# Cache for extracted resource paths
_font_cache: dict[str, Path] = {}
_logo_path_cache: Optional[Path] = None


def get_font_path(font_name: str) -> Path:
    """
    Get the path to a bundled font file.

    Uses importlib.resources to locate font files within the package.
    Paths are cached for performance.

    Args:
        font_name: Font filename (e.g., "LiberationSerif-Regular.ttf", "ipam.ttf")

    Returns:
        Path to the font file

    Raises:
        FileNotFoundError: If the font file doesn't exist in the package
    """
    if font_name in _font_cache:
        return _font_cache[font_name]

    font_files = resources.files("index_pdf_translation.resources.fonts")
    font_resource = font_files.joinpath(font_name)

    # For traversable resources, get the actual path
    if hasattr(font_resource, "__fspath__"):
        path = Path(font_resource)
    else:
        # Fallback for older Python or edge cases
        with resources.as_file(font_resource) as p:
            path = Path(p)

    if not path.exists():
        raise FileNotFoundError(f"Font file not found: {font_name}")

    _font_cache[font_name] = path
    return path


def get_logo_path() -> Path:
    """
    Get the path to the logo image file.

    Uses importlib.resources to locate the logo within the package.
    The path is cached for performance.

    Returns:
        Path to the logo file (indqx_qr.png)

    Raises:
        FileNotFoundError: If the logo file doesn't exist in the package
    """
    global _logo_path_cache

    if _logo_path_cache is not None:
        return _logo_path_cache

    data_files = resources.files("index_pdf_translation.resources.data")
    logo_resource = data_files.joinpath("indqx_qr.png")

    # For traversable resources, get the actual path
    if hasattr(logo_resource, "__fspath__"):
        path = Path(logo_resource)
    else:
        # Fallback for older Python or edge cases
        with resources.as_file(logo_resource) as p:
            path = Path(p)

    if not path.exists():
        raise FileNotFoundError("Logo file not found: indqx_qr.png")

    _logo_path_cache = path
    return path


def get_font_path_for_lang(lang: str) -> Path:
    """
    Get the appropriate font path for a given language.

    Args:
        lang: Language code ("en" for English, "ja" for Japanese)

    Returns:
        Path to the appropriate font file
    """
    if lang == "ja":
        return get_font_path("ipam.ttf")
    else:
        return get_font_path("LiberationSerif-Regular.ttf")


__all__ = [
    "get_font_path",
    "get_logo_path",
    "get_font_path_for_lang",
]
