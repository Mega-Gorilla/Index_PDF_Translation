# SPDX-License-Identifier: AGPL-3.0-only
"""
Index PDF Translation - PDF Translation Library

Translates academic paper PDFs and generates side-by-side PDF (original + translated).

Basic usage (Google Translate, no API key required):
    >>> from index_pdf_translation import pdf_translate, TranslationConfig
    >>> config = TranslationConfig()
    >>> result = await pdf_translate(pdf_data, config=config)

DeepL usage (high quality, requires API key):
    >>> config = TranslationConfig(backend="deepl", api_key="your-key")
    >>> result = await pdf_translate(pdf_data, config=config)

CLI usage:
    $ translate-pdf paper.pdf                    # Google Translate (default)
    $ translate-pdf paper.pdf --backend deepl   # DeepL
"""

from index_pdf_translation._version import __version__
from index_pdf_translation.config import TranslationConfig
from index_pdf_translation.core.translate import pdf_translate, TranslationResult

__all__ = [
    "__version__",
    "pdf_translate",
    "TranslationConfig",
    "TranslationResult",
]
