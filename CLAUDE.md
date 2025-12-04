# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Index PDF Translation is a PDF translation tool designed for academic papers. It intelligently preserves PDF formatting while translating content, detecting main body text and ignoring formulas, titles, and metadata. Uses DeepL API for translation and produces side-by-side original/translated PDFs.

**Languages**: Python 3.11+
**Supported translations**: English ↔ Japanese

## Licensing

- The project is licensed under **AGPL-3.0** (corrected to comply with PyMuPDF's AGPL-3.0).
- Keep SPDX identifiers as `AGPL-3.0-only` on source files you touch.
- When adding dependencies, verify AGPL compatibility.

## Commands

### Installation
```bash
pip install -r requirements.txt
```

### Run Local Translation (CLI)
```bash
python manual_translate_pdf.py
```
Opens file dialog, translates selected PDF, saves to `./output/result_*.pdf`

### Debug Mode
In `manual_translate_pdf.py`, call `translate_test()` instead of `translate_local()` to generate visualization PDFs showing block classification (blue=body, green=figures, red=removed) in `./debug/`.

## Configuration

**config.py** - Set your DeepL API key:
```python
DeepL_API_Key = "your-api-key"
DeepL_URL = "https://api-free.deepl.com/v2/translate"  # or Pro URL
```

## Architecture

### Core Modules (`modules/`)

- **pdf_edit.py** - PDF processing engine using PyMuPDF (fitz)
  - Text extraction with spatial coordinates
  - Histogram-based block classification (Sturges/Freedman-Diaconis binning)
  - Text removal and reinsertion with auto font sizing
  - Side-by-side PDF layout generation

- **translate.py** - Translation orchestration
  - DeepL API integration
  - Cross-block sentence merging (handles text spanning pages/blocks)
  - Main workflow: extract → filter → remove → translate → insert → layout

- **spacy_api.py** - Language tokenization (en_core_web_sm, ja_core_news_sm)

### Entry Points

- **manual_translate_pdf.py** - CLI tool for local PDF translation

### Translation Algorithm

1. Extract text blocks with coordinates and font metrics
2. Score blocks using token count, width (IQR outlier detection), and font size deviation
3. Classify: body text | figure/table captions (keyword detection) | removed
4. Merge consecutive blocks without terminal punctuation (preserves sentence context)
5. Translate via DeepL API
6. Calculate optimal font size for target language, insert translated text
7. Generate side-by-side layout PDF

### Key Technical Details

- All I/O operations use `asyncio.to_thread()` for async execution
- Font files in `fonts/` for Japanese (ipam.ttf) and English (Liberation Serif)
