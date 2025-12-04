# SPDX-License-Identifier: AGPL-3.0-only
"""
Pytest configuration and fixtures for Index PDF Translation tests.
"""

from pathlib import Path
from typing import Optional

import pytest


# Fixture paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
PDF_SAMPLE_DIR = Path(__file__).parent / "pdf_sample"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_llama_pdf() -> Path:
    """
    Return path to LLaMA paper sample PDF (first page).

    Paper: "LLaMA: Open and Efficient Foundation Language Models"
    arXiv: 2302.13971
    License: CC BY 4.0
    """
    return FIXTURES_DIR / "sample_llama.pdf"


@pytest.fixture
def sample_cot_pdf() -> Path:
    """
    Return path to Chain-of-Thought paper sample PDF (first page).

    Paper: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    arXiv: 2201.11903
    License: CC BY 4.0
    """
    return FIXTURES_DIR / "sample_cot.pdf"


@pytest.fixture
def sample_en_pdf(sample_llama_pdf: Path) -> Path:
    """
    Return path to a sample English PDF for testing.

    Alias for sample_llama_pdf for backward compatibility.
    """
    return sample_llama_pdf


@pytest.fixture
def sample_en_pdf_bytes(sample_en_pdf: Path) -> bytes:
    """Return sample English PDF as bytes."""
    return sample_en_pdf.read_bytes()


@pytest.fixture
def sample_llama_pdf_bytes(sample_llama_pdf: Path) -> bytes:
    """Return LLaMA sample PDF as bytes."""
    return sample_llama_pdf.read_bytes()


@pytest.fixture
def sample_cot_pdf_bytes(sample_cot_pdf: Path) -> bytes:
    """Return Chain-of-Thought sample PDF as bytes."""
    return sample_cot_pdf.read_bytes()


@pytest.fixture
def large_sample_pdf() -> Optional[Path]:
    """
    Return path to large PDF sample if available.

    This fixture returns None if pdf_sample/ directory doesn't exist
    or is empty. Tests using this fixture should skip if None.

    Note: pdf_sample/ is gitignored and contains full papers for
    local integration testing only.
    """
    if not PDF_SAMPLE_DIR.exists():
        return None

    pdf_files = list(PDF_SAMPLE_DIR.glob("*.pdf"))
    if not pdf_files:
        return None

    return pdf_files[0]


@pytest.fixture
def mock_api_key() -> str:
    """Return a mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def mock_api_url() -> str:
    """Return a mock API URL for testing."""
    return "https://api-mock.deepl.com/v2/translate"
