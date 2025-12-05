# SPDX-License-Identifier: AGPL-3.0-only
"""OpenAI GPT translation backend with Structured Outputs."""

import json
from typing import TYPE_CHECKING

from .base import TranslationError

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from pydantic import BaseModel


def _get_openai_imports():
    """Lazy import openai and pydantic."""
    try:
        from openai import AsyncOpenAI
        from pydantic import BaseModel
        return AsyncOpenAI, BaseModel
    except ImportError:
        raise ImportError(
            "openai and pydantic are required for OpenAI backend. "
            "Install with: uv pip install index-pdf-translation[openai]"
        )


# Block separator token (same as core/translate.py)
BLOCK_SEPARATOR = "[[[BR]]]"


class OpenAITranslator:
    """
    OpenAI GPT translation backend.

    Uses Structured Outputs for guaranteed array structure, avoiding newline issues.
    Prompts can be customized at 3 levels:
    - Default: Academic paper translation prompt
    - Constructor: Project-specific fixed settings
    - Method argument: Dynamic override
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    DEFAULT_SYSTEM_PROMPT = """You are a professional translator specializing in academic papers.
Translate the following texts from {source_lang} to {target_lang}.

Requirements:
- Preserve technical terminology accurately
- Maintain the exact number of input texts in the output array
- Each input text corresponds to one output text at the same index
- Do not merge or split texts
- Preserve any special characters or formatting within each text

Return a JSON object with a "translations" array containing the translated texts."""

    # Language code -> Language name
    LANG_NAMES = {"en": "English", "ja": "Japanese"}

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        source_lang: str = "en",
        system_prompt: str | None = None,
    ):
        """
        Initialize OpenAI translator.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
            source_lang: Source language code
            system_prompt: Custom system prompt ({source_lang}, {target_lang} placeholders)
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")

        AsyncOpenAI, _ = _get_openai_imports()
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model or self.DEFAULT_MODEL
        self._source_lang = source_lang
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    @property
    def name(self) -> str:
        return "openai"

    def _format_prompt(self, prompt_template: str, target_lang: str) -> str:
        """Format prompt template with language names."""
        return prompt_template.format(
            source_lang=self.LANG_NAMES.get(self._source_lang, self._source_lang),
            target_lang=self.LANG_NAMES.get(target_lang, target_lang),
        )

    def _create_translation_result_class(self):
        """Create Pydantic model for Structured Outputs."""
        _, BaseModel = _get_openai_imports()

        class TranslationResult(BaseModel):
            """Schema for Structured Outputs."""
            translations: list[str]

        return TranslationResult

    async def translate(
        self,
        text: str,
        target_lang: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """
        Translate text (handles separator token blocks).

        Args:
            text: Text to translate (may contain [[[BR]]] separator tokens)
            target_lang: Target language code
            system_prompt: Custom prompt (overrides constructor setting)

        Returns:
            Translated text with structure preserved
        """
        if not text.strip():
            return text

        # Split by separator token and translate as array
        texts = text.split(BLOCK_SEPARATOR)
        translated_texts = await self.translate_texts(
            texts, target_lang, system_prompt=system_prompt
        )
        return BLOCK_SEPARATOR.join(translated_texts)

    async def translate_texts(
        self,
        texts: list[str],
        target_lang: str,
        *,
        system_prompt: str | None = None,
    ) -> list[str]:
        """
        Translate multiple texts using Structured Outputs.

        Args:
            texts: List of texts to translate
            target_lang: Target language code
            system_prompt: Custom prompt (overrides constructor setting)

        Returns:
            List of translated texts (same length as input)
        """
        if not texts:
            return []

        # Track empty string indices
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return texts

        # Resolve prompt: method arg > constructor > default
        effective_prompt = system_prompt or self._system_prompt
        formatted_prompt = self._format_prompt(effective_prompt, target_lang)

        try:
            TranslationResult = self._create_translation_result_class()

            response = await self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": formatted_prompt},
                    {"role": "user", "content": json.dumps(non_empty_texts, ensure_ascii=False)},
                ],
                response_format=TranslationResult,
                temperature=0.2,
                top_p=0.6,
            )

            result = response.choices[0].message.parsed
            if result is None:
                raise TranslationError("OpenAI returned no parsed result")

            translated_parts = result.translations

            # Validate result length
            if len(translated_parts) != len(non_empty_texts):
                raise TranslationError(
                    f"OpenAI returned {len(translated_parts)} translations "
                    f"for {len(non_empty_texts)} texts"
                )

            # Restore to original positions
            results = list(texts)
            for idx, translated in zip(non_empty_indices, translated_parts):
                if idx < len(results):
                    results[idx] = translated

            return results

        except TranslationError:
            raise
        except Exception as e:
            raise TranslationError(f"OpenAI API error: {e}")
