#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Extract the first page from PDF files in pdf_sample/ directory.

This script creates lightweight test fixtures from academic papers.
Only the first page is extracted to minimize file size while
preserving realistic document structure for testing.

Usage:
    python tests/fixtures/extract_first_page.py
"""

import fitz  # PyMuPDF
from pathlib import Path

# Mapping of source files to output names and metadata
PDF_MAPPINGS = {
    "2302.13971": {
        "output": "sample_llama.pdf",
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "authors": "Hugo Touvron et al.",
        "arxiv": "arXiv:2302.13971",
        "license": "CC BY 4.0",
    },
    "2201.11903": {
        "output": "sample_cot.pdf",  # Chain of Thought
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": "Jason Wei et al.",
        "arxiv": "arXiv:2201.11903",
        "license": "CC BY 4.0",
    },
}


def extract_first_page(input_path: Path, output_path: Path) -> None:
    """
    Extract the first page from a PDF file.

    Args:
        input_path: Path to the source PDF
        output_path: Path to save the single-page PDF
    """
    doc = fitz.open(input_path)

    # Create new document with only first page
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=0, to_page=0)

    new_doc.save(output_path)
    new_doc.close()
    doc.close()

    print(f"Extracted: {input_path.name} -> {output_path.name}")
    print(f"  Original: {input_path.stat().st_size / 1024:.1f} KB")
    print(f"  Extracted: {output_path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    """Extract first pages from all PDFs in pdf_sample/."""
    fixtures_dir = Path(__file__).parent
    pdf_sample_dir = fixtures_dir.parent / "pdf_sample"

    if not pdf_sample_dir.exists():
        print(f"Error: {pdf_sample_dir} does not exist")
        return

    pdf_files = list(pdf_sample_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_sample_dir}")
        return

    extracted_count = 0
    for pdf_file in pdf_files:
        # Find matching mapping by arxiv ID in filename
        arxiv_id = None
        for key in PDF_MAPPINGS:
            if key in pdf_file.name:
                arxiv_id = key
                break

        if arxiv_id is None:
            print(f"Skipping {pdf_file.name}: no mapping found")
            continue

        mapping = PDF_MAPPINGS[arxiv_id]
        output_path = fixtures_dir / mapping["output"]
        extract_first_page(pdf_file, output_path)
        extracted_count += 1

    print(f"\nExtracted {extracted_count} PDF(s) to fixtures/")


if __name__ == "__main__":
    main()
