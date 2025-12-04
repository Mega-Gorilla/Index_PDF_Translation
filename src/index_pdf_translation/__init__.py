# SPDX-License-Identifier: AGPL-3.0-only
"""
Index PDF Translation - PDF翻訳ライブラリ

学術論文PDFを翻訳し、見開きPDF（オリジナル + 翻訳）を生成します。

Basic usage:
    >>> from index_pdf_translation import pdf_translate, TranslationConfig
    >>> config = TranslationConfig(api_key="your-key", target_lang="ja")
    >>> result = await pdf_translate(pdf_data, config=config)

CLI usage:
    $ translate-pdf paper.pdf -o output.pdf
"""

from index_pdf_translation._version import __version__
from index_pdf_translation.core.translate import pdf_translate

# TranslationConfig will be added in Phase 4
# from index_pdf_translation.config import TranslationConfig

__all__ = [
    "__version__",
    "pdf_translate",
    # "TranslationConfig",  # Available after Phase 4
]
