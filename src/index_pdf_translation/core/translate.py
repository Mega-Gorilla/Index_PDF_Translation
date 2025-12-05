# SPDX-License-Identifier: AGPL-3.0-only
"""
Index PDF Translation - Translation Orchestration

Provides translation workflow for PDF documents.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from index_pdf_translation.config import TranslationConfig
from index_pdf_translation.logger import get_logger
from index_pdf_translation.core.pdf_edit import (
    DocumentBlocks,
    create_debug_pdf,
    create_viewing_pdf,
    extract_text_coordinates_dict,
    pdf_draw_blocks,
    preprocess_write_blocks,
    remove_blocks,
    remove_textbox_for_pdf,
    write_logo_data,
    write_pdf_text,
)


@dataclass
class TranslationResult:
    """翻訳結果を格納するデータクラス。

    Attributes:
        pdf: 翻訳済みPDF（見開き）
        debug_pdf: デバッグ可視化PDF（統合版）
    """

    pdf: bytes
    debug_pdf: Optional[bytes] = None

if TYPE_CHECKING:
    from index_pdf_translation.translators import TranslatorBackend

logger = get_logger("translate")

# Separator token (verified 100% success rate with Google Translate)
# Uses symbol-only token that won't be translated
BLOCK_SEPARATOR = "[[[BR]]]"

# Character limit (deep-translator has 5,000 char limit for Google Translate)
MAX_CHUNK_SIZE = 4500  # With margin for separator


def chunk_texts_for_translation(
    texts: list[str],
    separator: str = BLOCK_SEPARATOR,
    max_size: int = MAX_CHUNK_SIZE,
) -> list[list[str]]:
    """
    Split text list into chunks within character limit.

    Args:
        texts: List of texts to translate
        separator: Separator token
        max_size: Maximum characters per chunk

    Returns:
        List of text lists (chunks)

    Note:
        If a single block exceeds max_size, it's treated as a standalone chunk.
        May fail at translation API, but rare in academic papers.
    """
    chunks: list[list[str]] = []
    current_chunk: list[str] = []
    current_size = 0
    separator_len = len(separator)

    for text in texts:
        # Single block exceeds limit - warn and add as standalone
        if len(text) > max_size:
            # Save current chunk first
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            # Add oversized block as standalone chunk
            chunks.append([text])
            logger.warning(
                f"Single block exceeds MAX_CHUNK_SIZE ({len(text)} > {max_size}). "
                f"May fail at translation API."
            )
            continue

        # Calculate size including separator
        item_size = len(text)
        if current_chunk:
            item_size += separator_len

        # Would exceed limit if added to current chunk
        if current_size + item_size > max_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
            item_size = len(text)  # No separator for new chunk

        current_chunk.append(text)
        current_size += item_size

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def translate_chunk_with_retry(
    translator: "TranslatorBackend",
    text: str,
    target_lang: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """
    Translate with fixed-delay retry (Phase 1 implementation).

    Args:
        translator: Translation backend
        text: Text to translate
        target_lang: Target language code
        max_retries: Maximum retry count
        retry_delay: Retry interval in seconds (fixed)

    Returns:
        Translated text

    Raises:
        TranslationError: When all retries fail
    """
    from index_pdf_translation.translators.base import TranslationError

    last_error: Optional[TranslationError] = None
    for attempt in range(max_retries + 1):
        try:
            return await translator.translate(text, target_lang)
        except TranslationError as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    f"Translation failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {retry_delay}s: {e}"
                )
                await asyncio.sleep(retry_delay)
            else:
                raise

    raise last_error  # For type checker (unreachable)


async def translate_blocks(
    blocks: DocumentBlocks,
    translator: "TranslatorBackend",
    target_lang: str,
) -> DocumentBlocks:
    """
    Translate multiple text blocks in batch.

    Uses separator token method with chunking:
    - Handles 5,000 character limit
    - Minimizes API calls (rate limit avoidance)
    - Maintains context for quality translation
    - Accurately preserves empty lines/whitespace

    Args:
        blocks: Block information list
        translator: Translation backend
        target_lang: Target language code

    Returns:
        Translated block information
    """
    # Extract all texts
    texts: list[str] = []
    for page in blocks:
        for block in page:
            texts.append(block["text"])

    if not texts:
        return blocks

    # Split into chunks
    chunks = chunk_texts_for_translation(texts, BLOCK_SEPARATOR, MAX_CHUNK_SIZE)
    logger.info(f"Split {len(texts)} blocks into {len(chunks)} chunks for translation")

    # Translate each chunk
    translated_texts: list[str] = []
    for i, chunk in enumerate(chunks):
        combined_text = BLOCK_SEPARATOR.join(chunk)
        logger.debug(f"Translating chunk {i + 1}/{len(chunks)} ({len(combined_text)} chars)")

        translated_combined = await translate_chunk_with_retry(
            translator, combined_text, target_lang
        )

        # Split translated result
        chunk_results = translated_combined.split(BLOCK_SEPARATOR)

        # Verify chunk line count
        if len(chunk_results) != len(chunk):
            logger.warning(
                f"Chunk {i + 1} block count mismatch: "
                f"expected {len(chunk)}, got {len(chunk_results)}"
            )

        translated_texts.extend(chunk_results)

    # Verify total line count
    if len(translated_texts) != len(texts):
        logger.warning(
            f"Total block count mismatch after translation: "
            f"expected {len(texts)}, got {len(translated_texts)}"
        )

    # Assign translated texts to blocks
    idx = 0
    for page in blocks:
        for block in page:
            if idx < len(translated_texts):
                block["text"] = translated_texts[idx]
            else:
                block["text"] = ""
            idx += 1

    return blocks


async def preprocess_translation_blocks(
    blocks: DocumentBlocks,
    end_markers: tuple[str, ...] = (".", ":", ";"),
    end_marker_enable: bool = True,
) -> DocumentBlocks:
    """
    Preprocess blocks before translation.

    Merges multiple blocks without end markers to improve translation quality.

    Args:
        blocks: Block information list
        end_markers: Tuple of sentence-ending markers
        end_marker_enable: Enable end-marker based splitting

    Returns:
        Preprocessed block information
    """
    results: DocumentBlocks = []

    text = ""
    coordinates: list[Any] = []
    block_no: list[int] = []
    page_no: list[int] = []
    font_size: list[float] = []

    for page in blocks:
        page_results: list[dict[str, Any]] = []
        temp_block_no = 0

        for block in page:
            text += " " + block["text"]
            page_no.append(block["page_no"])
            coordinates.append(block["coordinates"])
            block_no.append(block["block_no"])
            font_size.append(block["size"])

            should_save = (
                text.endswith(end_markers)
                or block["block_no"] - temp_block_no <= 1
                or not end_marker_enable
            )

            if should_save:
                page_results.append({
                    "page_no": page_no,
                    "block_no": block_no,
                    "coordinates": coordinates,
                    "text": text,
                    "size": font_size,
                })
                # Reset
                text = ""
                coordinates = []
                block_no = []
                page_no = []
                font_size = []

            temp_block_no = block["block_no"]

        results.append(page_results)

    return results


async def pdf_translate(
    pdf_data: bytes,
    *,
    config: TranslationConfig,
    disable_translate: bool = False,
) -> Optional[TranslationResult]:
    """
    Translate PDF and generate side-by-side PDF (original + translated).

    Translation workflow:
    1. Extract text blocks
    2. Classify blocks (body/figures/excluded)
    3. Remove text
    4. Translate
    5. Insert translated text
    6. Add logo (optional)
    7. Generate side-by-side PDF
    8. Generate debug PDF (if debug mode)

    Args:
        pdf_data: Input PDF binary data
        config: Translation configuration
        disable_translate: Disable translation (for testing)

    Returns:
        TranslationResult containing PDF data and optional debug PDF

    Examples:
        >>> # Google Translate (default)
        >>> config = TranslationConfig()
        >>> result = await pdf_translate(pdf_data, config=config)
        >>> translated_pdf = result.pdf

        >>> # DeepL Translate
        >>> config = TranslationConfig(backend="deepl", api_key="xxx")
        >>> result = await pdf_translate(pdf_data, config=config)

        >>> # Debug mode
        >>> config = TranslationConfig(debug=True)
        >>> result = await pdf_translate(pdf_data, config=config)
        >>> debug_pdf = result.debug_pdf  # Contains histograms + block frames
    """
    # Create translation backend
    translator = config.create_translator()
    logger.info(f"Using translator: {translator.name}")

    # 1. Extract text blocks
    block_info = await extract_text_coordinates_dict(pdf_data)

    # 2. Classify blocks
    plot_images: Optional[list[bytes]] = None
    remove_info: Optional[DocumentBlocks] = None
    if config.debug:
        text_blocks, fig_blocks, remove_info, plot_images = await remove_blocks(
            block_info, 10, lang=config.source_lang, debug=True
        )
    else:
        text_blocks, fig_blocks, _, _ = await remove_blocks(
            block_info, 10, lang=config.source_lang
        )

    # 3. Remove text
    removed_textbox_pdf_data = await remove_textbox_for_pdf(pdf_data, text_blocks)
    removed_textbox_pdf_data = await remove_textbox_for_pdf(
        removed_textbox_pdf_data, fig_blocks
    )
    logger.info("1. Text box removal complete")

    # Prepare blocks for translation
    preprocess_text_blocks = await preprocess_translation_blocks(
        text_blocks, (".", ":", ";"), True
    )
    preprocess_fig_blocks = await preprocess_translation_blocks(
        fig_blocks, (".", ":", ";"), False
    )
    logger.info("2. Block preprocessing complete")

    # 4. Translate
    if not disable_translate:
        translate_text_blocks = await translate_blocks(
            preprocess_text_blocks,
            translator,
            config.target_lang,
        )
        translate_fig_blocks = await translate_blocks(
            preprocess_fig_blocks,
            translator,
            config.target_lang,
        )
        logger.info("3. Translation complete")

        # 5. Create PDF write data
        write_text_blocks = await preprocess_write_blocks(
            translate_text_blocks, config.target_lang
        )
        write_fig_blocks = await preprocess_write_blocks(
            translate_fig_blocks, config.target_lang
        )
        logger.info("4. Write block generation complete")

        # Create PDF
        translated_pdf_data = removed_textbox_pdf_data
        if write_text_blocks:
            translated_pdf_data = await write_pdf_text(
                translated_pdf_data, write_text_blocks, config.target_lang
            )
        if write_fig_blocks:
            translated_pdf_data = await write_pdf_text(
                translated_pdf_data, write_fig_blocks, config.target_lang
            )

        # 6. Add logo (optional)
        if config.add_logo:
            translated_pdf_data = await write_logo_data(translated_pdf_data)
    else:
        logger.info("Translation skipped (disable_translate=True)")
        translated_pdf_data = removed_textbox_pdf_data

    # 7. Create side-by-side layout
    merged_pdf_data = await create_viewing_pdf(pdf_data, translated_pdf_data)
    logger.info("5. Side-by-side PDF generation complete")

    # 8. Generate debug PDF (if debug mode)
    debug_pdf: Optional[bytes] = None
    if config.debug and remove_info is not None:
        # Draw block frames on original PDF
        blocks_pdf = pdf_data

        # text_blocks: 緑 (green)
        blocks_pdf = await pdf_draw_blocks(
            blocks_pdf,
            text_blocks,
            line_color_rgb=[0, 0.7, 0],
            fill_color_rgb=[0, 0.7, 0],
            fill_opacity=0.2,
            width=1,
        )

        # fig_blocks: 黄 (yellow)
        blocks_pdf = await pdf_draw_blocks(
            blocks_pdf,
            fig_blocks,
            line_color_rgb=[1, 0.8, 0],
            fill_color_rgb=[1, 0.8, 0],
            fill_opacity=0.2,
            width=1,
        )

        # removed_blocks: 赤 (red)
        blocks_pdf = await pdf_draw_blocks(
            blocks_pdf,
            remove_info,
            line_color_rgb=[1, 0, 0],
            fill_color_rgb=[1, 0, 0],
            fill_opacity=0.2,
            width=1,
        )

        # Create debug PDF with histograms as leading pages
        debug_pdf = await create_debug_pdf(blocks_pdf, plot_images)
        logger.info("6. Debug PDF generation complete")

    return TranslationResult(pdf=merged_pdf_data, debug_pdf=debug_pdf)
