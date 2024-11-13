# Index PDF Translation

This is a fork of [Index_PDF_Translation](https://github.com/Mega-Gorilla/Index_PDF_Translation) that enables the use of LLM APIs such as the OpenAI API.

## Usage

set the environment variables in the .env file.

```.env
OPENAI_API_KEY=sk-proj-...
```

```bash
uv sync
uv run python manual_translate_pdf.py
```

Select a file and wait for translation to complete. The result will be output to the output folder.
Currently, only English to Japanese translation is supported.