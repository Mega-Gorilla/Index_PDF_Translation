# SPDX-License-Identifier: AGPL-3.0-only
"""
Natural Language Processing modules.

This subpackage contains:
- tokenizer.py: spaCy-based text tokenization
"""

from index_pdf_translation.nlp.tokenizer import tokenize_text, load_model

__all__ = [
    "tokenize_text",
    "load_model",
]
