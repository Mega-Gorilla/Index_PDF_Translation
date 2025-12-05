# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.0.0] - YYYY-MM-DD

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
- `--backend` CLI option to select translator (`google` or `deepl`)
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
```

## [2.0.0] - Previous Release

- Initial release with DeepL-only support
