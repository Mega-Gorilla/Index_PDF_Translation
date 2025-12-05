# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.1.0] - 2025-12-05

### Breaking Changes

- **`pdf_translate()` return type changed**: `Optional[bytes]` -> `Optional[TranslationResult]`
  - Before: `result = await pdf_translate(pdf_data, config=config)` (result is bytes)
  - After: `result = await pdf_translate(pdf_data, config=config)` (result is `TranslationResult`)
  - Access translated PDF via `result.pdf`
  - Access debug PDF via `result.debug_pdf` (when `config.debug=True`)

### Added

- `TranslationResult` dataclass for structured return values
- Debug PDF generation with `--debug` CLI option
  - Histogram pages (tokens, font sizes, scores distribution)
  - Block visualization with color coding (green=text, yellow=figures, red=excluded)
- `create_debug_pdf()` function in `pdf_edit.py`

### Migration Guide (3.0.0 -> 3.1.0)

```python
# Before (v3.0.0)
result = await pdf_translate(pdf_data, config=config)
with open("output.pdf", "wb") as f:
    f.write(result)  # result was bytes

# After (v3.1.0)
result = await pdf_translate(pdf_data, config=config)
with open("output.pdf", "wb") as f:
    f.write(result.pdf)  # access via .pdf attribute

# Debug mode (v3.1.0)
config = TranslationConfig(debug=True)
result = await pdf_translate(pdf_data, config=config)
if result.debug_pdf:
    with open("debug.pdf", "wb") as f:
        f.write(result.debug_pdf)
```

## [3.0.0] - 2025-12-05

### Breaking Changes

- **Default translator changed**: DeepL -> Google Translate
  - API key is no longer required for basic usage
  - Use `--backend deepl` or `backend="deepl"` for DeepL
- **`pdf_translate()` signature changed**: Individual parameters replaced with `config` parameter
  - Before: `pdf_translate(pdf_data, api_key="xxx", target_lang="ja")`
  - After: `pdf_translate(pdf_data, config=TranslationConfig(...))`
- **`TranslationConfig` changes**:
  - `api_key` only required when `backend="deepl"`
  - Removed `deepl_target_lang` and `deepl_source_lang` properties
- **`SUPPORTED_LANGUAGES` structure changed**:
  - Removed `deepl` key (now only contains `spacy`)
- **`aiohttp` dependency moved to optional**:
  - Install with `pip install index-pdf-translation[deepl]` for DeepL support

### Added

- Google Translate backend (default, no API key required)
- OpenAI GPT backend (customizable prompts, Structured Outputs)
- `--backend` CLI option to select translator (`google`, `deepl`, or `openai`)
- OpenAI CLI options (`--openai-model`, `--openai-prompt`, `--openai-prompt-file`)
- Separator token method (`[[[BR]]]`) for reliable block translation
- Character limit chunking for large documents (4500 chars per chunk)
- Retry mechanism for translation errors (3 retries, 1s delay)
- `.env.example` template for API key configuration

### Changed

- Translation backend abstraction using Strategy pattern
- Improved error messages with migration guidance
- Log messages changed to English

### Migration Guide

```python
# Before (v2.x)
from index_pdf_translation import pdf_translate
result = await pdf_translate(
    pdf_data,
    api_key="xxx",
    target_lang="ja"
)

# After (v3.0.0) - Google Translate (default)
from index_pdf_translation import pdf_translate, TranslationConfig
config = TranslationConfig()
result = await pdf_translate(pdf_data, config=config)

# After (v3.0.0) - DeepL
config = TranslationConfig(backend="deepl", api_key="xxx")
result = await pdf_translate(pdf_data, config=config)

# After (v3.0.0) - OpenAI GPT
config = TranslationConfig(backend="openai", openai_api_key="xxx")
result = await pdf_translate(pdf_data, config=config)

# OpenAI with custom model and prompt
config = TranslationConfig(
    backend="openai",
    openai_api_key="xxx",
    openai_model="gpt-4o",
    openai_system_prompt="You are a medical translator..."
)
result = await pdf_translate(pdf_data, config=config)
```

## [2.0.0] - Previous Release

- Initial release with DeepL-only support
