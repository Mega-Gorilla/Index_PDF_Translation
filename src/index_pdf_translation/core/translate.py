# SPDX-License-Identifier: AGPL-3.0-only
"""
Index PDF Translation - Translation Orchestration

DeepL APIを使用した翻訳機能と、PDF翻訳ワークフローを提供します。
"""

from typing import Any, Optional

import aiohttp

from index_pdf_translation.config import TranslationConfig
from index_pdf_translation.logger import get_logger
from index_pdf_translation.core.pdf_edit import (
    DocumentBlocks,
    create_viewing_pdf,
    extract_text_coordinates_dict,
    preprocess_write_blocks,
    remove_blocks,
    remove_textbox_for_pdf,
    write_logo_data,
    write_pdf_text,
)

logger = get_logger("translate")

# 型エイリアス
TranslationResult = dict[str, Any]


async def translate_str_data(
    key: str, text: str, target_lang: str, api_url: str
) -> TranslationResult:
    """
    DeepL APIを使用してテキストを翻訳します。

    Args:
        key: DeepL APIキー
        text: 翻訳するテキスト
        target_lang: 翻訳先の言語コード（例: "EN", "JA"）
        api_url: DeepL API URL

    Returns:
        翻訳結果を含む辞書:
        - 成功時: {"ok": True, "data": "翻訳テキスト"}
        - 失敗時: {"ok": False, "message": "エラーメッセージ"}
    """
    params = {
        "auth_key": key,
        "text": text,
        "target_lang": target_lang,
        "tag_handling": "xml",
        "formality": "more",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, data=params) as response:
            if response.status == 200:
                result = await response.json()
                translated_text = result["translations"][0]["text"]
                return {"ok": True, "data": translated_text}
            else:
                error_msg = f"DeepL API request failed with status code {response.status}"
                logger.error(error_msg)
                return {"ok": False, "message": error_msg}


async def translate_blocks(
    blocks: DocumentBlocks, key: str, target_lang: str, api_url: str
) -> DocumentBlocks:
    """
    複数のテキストブロックを一括翻訳します。

    Args:
        blocks: 翻訳するブロック情報のリスト
        key: DeepL APIキー
        target_lang: 翻訳先の言語コード
        api_url: DeepL API URL

    Returns:
        翻訳後のブロック情報（元のブロックのtextが翻訳テキストに置換）

    Raises:
        Exception: 翻訳APIが失敗した場合
    """
    # テキストを改行で連結
    translate_text = ""
    for page in blocks:
        for block in page:
            translate_text += block["text"] + "\n"

    # 翻訳実行
    translated_result = await translate_str_data(key, translate_text, target_lang, api_url)

    if translated_result["ok"]:
        translated_text = translated_result["data"]
    else:
        raise Exception(translated_result["message"])

    # 翻訳テキストを分割して各ブロックに割り当て
    translated_lines = translated_text.split("\n")

    for page in blocks:
        for block in page:
            if translated_lines:
                block["text"] = translated_lines.pop(0)
            else:
                block["text"] = ""

    return blocks


async def preprocess_translation_blocks(
    blocks: DocumentBlocks,
    end_markers: tuple[str, ...] = (".", ":", ";"),
    end_marker_enable: bool = True,
) -> DocumentBlocks:
    """
    翻訳前のブロック前処理を行います。

    終端記号がない場合、複数のブロックを1つにまとめて翻訳品質を向上させます。

    Args:
        blocks: ブロック情報のリスト
        end_markers: 文末を示す記号のタプル
        end_marker_enable: 終端記号による分割を有効化するフラグ

    Returns:
        前処理後のブロック情報
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
                # リセット
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
    config: Optional[TranslationConfig] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    source_lang: str = "en",
    target_lang: str = "ja",
    debug: bool = False,
    add_logo: bool = True,
    disable_translate: bool = False,
) -> Optional[bytes]:
    """
    PDFを翻訳し、見開きPDF（オリジナル + 翻訳）を生成します。

    翻訳ワークフロー:
    1. テキストブロック抽出
    2. ブロック分類（本文/図表/除外）
    3. テキスト削除
    4. DeepL翻訳
    5. 翻訳テキスト挿入
    6. ロゴ追加（オプション）
    7. 見開きPDF生成

    Args:
        pdf_data: 入力PDFのバイナリデータ
        config: TranslationConfig オブジェクト。指定時は他のパラメータより優先。
        api_key: DeepL APIキー (config未指定時に使用)
        api_url: DeepL API URL (config未指定時に使用)
        source_lang: 翻訳元言語コード (config未指定時に使用, default: 'en')
        target_lang: 翻訳先言語コード (config未指定時に使用, default: 'ja')
        debug: デバッグモード (config未指定時に使用, default: False)
        add_logo: ロゴウォーターマークを追加 (config未指定時に使用, default: True)
        disable_translate: 翻訳を無効化（テスト用）

    Returns:
        見開きPDFのバイナリデータ、または失敗時はNone

    Examples:
        >>> # TranslationConfig を使用
        >>> config = TranslationConfig(api_key="your-key")
        >>> result = await pdf_translate(pdf_data, config=config)

        >>> # 個別パラメータを使用
        >>> result = await pdf_translate(
        ...     pdf_data,
        ...     api_key="your-key",
        ...     source_lang="en",
        ...     target_lang="ja",
        ... )
    """
    # 設定の解決
    if config is not None:
        # TranslationConfig が指定された場合、その値を使用
        effective_api_key = config.api_key
        effective_api_url = config.api_url
        effective_source_lang = config.source_lang
        effective_target_lang = config.target_lang
        effective_debug = config.debug
        effective_add_logo = config.add_logo
    else:
        # 個別パラメータを使用
        if api_key is None:
            raise ValueError(
                "api_key is required. Pass api_key parameter or use TranslationConfig."
            )
        effective_api_key = api_key
        effective_api_url = api_url or "https://api-free.deepl.com/v2/translate"
        effective_source_lang = source_lang
        effective_target_lang = target_lang
        effective_debug = debug
        effective_add_logo = add_logo

    # 1. テキストブロック抽出
    block_info = await extract_text_coordinates_dict(pdf_data)

    # 2. ブロック分類
    if effective_debug:
        text_blocks, fig_blocks, remove_info, plot_images = await remove_blocks(
            block_info, 10, lang=effective_source_lang, debug=True
        )
    else:
        text_blocks, fig_blocks, _, _ = await remove_blocks(
            block_info, 10, lang=effective_source_lang
        )

    # 3. テキスト削除
    removed_textbox_pdf_data = await remove_textbox_for_pdf(pdf_data, text_blocks)
    removed_textbox_pdf_data = await remove_textbox_for_pdf(
        removed_textbox_pdf_data, fig_blocks
    )
    logger.info("1. テキストボックス削除完了")

    # 翻訳前のブロック準備
    preprocess_text_blocks = await preprocess_translation_blocks(
        text_blocks, (".", ":", ";"), True
    )
    preprocess_fig_blocks = await preprocess_translation_blocks(
        fig_blocks, (".", ":", ";"), False
    )
    logger.info("2. ブロック前処理完了")

    # 4. 翻訳実施
    if not disable_translate:
        translate_text_blocks = await translate_blocks(
            preprocess_text_blocks,
            effective_api_key,
            effective_target_lang,
            effective_api_url,
        )
        translate_fig_blocks = await translate_blocks(
            preprocess_fig_blocks,
            effective_api_key,
            effective_target_lang,
            effective_api_url,
        )
        logger.info("3. 翻訳完了")

        # 5. PDF書き込みデータ作成
        write_text_blocks = await preprocess_write_blocks(
            translate_text_blocks, effective_target_lang
        )
        write_fig_blocks = await preprocess_write_blocks(
            translate_fig_blocks, effective_target_lang
        )
        logger.info("4. 書き込みブロック生成完了")

        # PDFの作成
        translated_pdf_data = removed_textbox_pdf_data
        if write_text_blocks:
            translated_pdf_data = await write_pdf_text(
                translated_pdf_data, write_text_blocks, effective_target_lang
            )
        if write_fig_blocks:
            translated_pdf_data = await write_pdf_text(
                translated_pdf_data, write_fig_blocks, effective_target_lang
            )

        # 6. ロゴ追加（オプション）
        if effective_add_logo:
            translated_pdf_data = await write_logo_data(translated_pdf_data)
    else:
        logger.info("翻訳スキップ（disable_translate=True）")
        translated_pdf_data = removed_textbox_pdf_data

    # 7. 見開き結合
    merged_pdf_data = await create_viewing_pdf(pdf_data, translated_pdf_data)
    logger.info("5. 見開きPDF生成完了")

    return merged_pdf_data
