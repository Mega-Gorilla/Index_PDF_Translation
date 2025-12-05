# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Index PDF Translation is a PDF translation tool designed for academic papers. It intelligently preserves PDF formatting while translating content, detecting main body text and ignoring formulas, titles, and metadata. Supports Google Translate (default, no API key) and DeepL (high quality) backends, and produces side-by-side original/translated PDFs.

**Languages**: Python 3.11+
**Supported translations**: English ↔ Japanese

## Licensing

- The project is licensed under **AGPL-3.0** (corrected to comply with PyMuPDF's AGPL-3.0).
- Keep SPDX identifiers as `AGPL-3.0-only` on source files you touch.
- When adding dependencies, verify AGPL compatibility.

## Commands

### Installation (uv - recommended)
```bash
uv sync
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download ja_core_news_sm
```

### Installation (pip)
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download ja_core_news_sm
```

### Run Local Translation (CLI)
```bash
# Basic usage (Google Translate, no API key required)
uv run translate-pdf paper.pdf

# DeepL (high quality, requires API key)
uv run translate-pdf paper.pdf --backend deepl

# With options
uv run translate-pdf paper.pdf -o ./result.pdf
uv run translate-pdf paper.pdf --source en --target ja
uv run translate-pdf paper.pdf --no-logo --debug
```
Translates PDF and saves side-by-side PDF to `./output/translated_*.pdf`

### CLI Options
- `-o, --output`: Output file path
- `-b, --backend`: Translation backend (google/deepl, default: google)
- `-s, --source`: Source language (en/ja, default: en)
- `-t, --target`: Target language (en/ja, default: ja)
- `--api-key`: DeepL API key (required for --backend deepl)
- `--api-url`: DeepL API URL (for Pro users)
- `--no-logo`: Disable logo watermark
- `--debug`: Enable debug mode (generate visualization PDFs)

## Configuration

For DeepL backend, set your API key via environment variable:
```bash
export DEEPL_API_KEY="your-api-key"
# For Pro API users:
export DEEPL_API_URL="https://api.deepl.com/v2/translate"
```

Google Translate (default) requires no API key.

## Architecture

### Core Modules (`src/index_pdf_translation/`)

- **core/pdf_edit.py** - PDF processing engine using PyMuPDF (fitz)
  - Text extraction with spatial coordinates
  - Histogram-based block classification (Sturges/Freedman-Diaconis binning)
  - Text removal and reinsertion with auto font sizing
  - Side-by-side PDF layout generation

- **core/translate.py** - Translation orchestration
  - Google Translate / DeepL integration via Strategy pattern
  - Separator token method (`[[[BR]]]`) for reliable block translation
  - Chunking for 5,000 character limit, retry mechanism
  - Cross-block sentence merging (handles text spanning pages/blocks)
  - Main workflow: extract → filter → remove → translate → insert → layout

- **translators/** - Translation backend module
  - `base.py`: TranslatorBackend protocol
  - `google.py`: Google Translate (default, no API key)
  - `deepl.py`: DeepL (high quality, requires API key)

- **nlp/tokenizer.py** - Language tokenization (en_core_web_sm, ja_core_news_sm)

### Entry Points

- **cli.py** - CLI tool (`translate-pdf` command) via entry_points

### Translation Algorithm

1. Extract text blocks with coordinates and font metrics
2. Score blocks using token count, width (IQR outlier detection), and font size deviation
3. Classify: body text | figure/table captions (keyword detection) | removed
4. Merge consecutive blocks without terminal punctuation (preserves sentence context)
5. Translate via Google Translate / DeepL API
6. Calculate optimal font size for target language, insert translated text
7. Generate side-by-side layout PDF

### Key Technical Details

- All I/O operations use `asyncio.to_thread()` for async execution
- Font files bundled in `resources/fonts/` for Japanese (ipam.ttf) and English (Liberation Serif)
- Resources accessed via `importlib.resources` for proper package installation support
