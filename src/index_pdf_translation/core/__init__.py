# SPDX-License-Identifier: AGPL-3.0-only
"""
Core translation and PDF processing modules.

This subpackage contains:
- translate.py: Translation orchestration with DeepL API
- pdf_edit.py: PDF processing engine using PyMuPDF
"""

from index_pdf_translation.core.translate import pdf_translate
from index_pdf_translation.core.pdf_edit import (
    DocumentBlocks,
    extract_text_coordinates_dict,
    extract_text_coordinates_blocks,
    remove_blocks,
    remove_textbox_for_pdf,
    preprocess_write_blocks,
    write_pdf_text,
    write_logo_data,
    create_viewing_pdf,
)

__all__ = [
    "pdf_translate",
    "DocumentBlocks",
    "extract_text_coordinates_dict",
    "extract_text_coordinates_blocks",
    "remove_blocks",
    "remove_textbox_for_pdf",
    "preprocess_write_blocks",
    "write_pdf_text",
    "write_logo_data",
    "create_viewing_pdf",
]
