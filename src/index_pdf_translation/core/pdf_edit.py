# SPDX-License-Identifier: AGPL-3.0-only
"""
Index PDF Translation - PDF Processing Engine

PyMuPDFを使用したPDF処理機能を提供します。
- テキスト抽出と座標取得
- ブロック分類（本文/図表/除外）
- テキスト削除と挿入
- 見開きPDF生成
"""

import asyncio
import copy
import math
import os
import string
from collections import defaultdict
from io import BytesIO
from statistics import median
from typing import Any, Optional

import fitz  # PyMuPDF
import numpy as np

from index_pdf_translation.logger import get_logger
from index_pdf_translation.nlp.tokenizer import tokenize_text

logger = get_logger("pdf_edit")

# 定数
LINE_HEIGHT_FACTOR = 1.5  # 行の高さの係数
LH_CALC_FACTOR = 1.3  # 行高さ計算係数
FONT_SIZE_DECREMENT = 0.1  # フォントサイズ調整時の減少量

# フォントパス
FONT_PATH_EN = "fonts/LiberationSerif-Regular.ttf"
FONT_PATH_JA = "fonts/ipam.ttf"
FALLBACK_FONT = "helv"

# ロゴ設定
LOGO_PATH = "./data/indqx_qr.png"
LOGO_RECT = (5, 5, 35, 35)

# 図表キーワード
FIG_KEYWORDS_EN = ["fig", "table"]
FIG_KEYWORDS_JA = ["表", "グラフ"]

# 型エイリアス
BlockInfo = dict[str, Any]
PageBlocks = list[BlockInfo]
DocumentBlocks = list[PageBlocks]
Coordinates = tuple[float, float, float, float]


async def extract_text_coordinates_blocks(pdf_data: bytes) -> DocumentBlocks:
    """
    PDFからテキストブロックの座標を取得します（ブロック形式）。

    Args:
        pdf_data: PDFのバイナリデータ

    Returns:
        ページごとのテキストブロック情報のリスト
    """
    document = await asyncio.to_thread(fitz.open, stream=pdf_data, filetype="pdf")
    content: DocumentBlocks = []

    for page_num in range(len(document)):
        page_content: PageBlocks = []
        page = await asyncio.to_thread(document.load_page, page_num)
        blocks = await asyncio.to_thread(page.get_text, "blocks")

        for b in blocks:
            x0, y0, x1, y1, content_text, block_no, block_type = b[:7]

            # フォントサイズ逆算
            count_lines = content_text.count("\n")
            if count_lines != 0:
                calc_fs = (y1 - y0) / count_lines * 0.98
            else:
                calc_fs = y1 - y0
            calc_fs = math.floor(calc_fs * 100) / 100

            if block_type == 0:  # テキストブロック
                block_info: BlockInfo = {
                    "block_no": block_no,
                    "text": content_text,
                    "size": calc_fs,
                    "coordinates": (x0, y0, x1, y1),
                }
                page_content.append(block_info)
            else:
                logger.debug(f"非テキストブロック検出: {b}")

        content.append(page_content)

    await asyncio.to_thread(document.close)
    return content


async def extract_text_coordinates_dict(pdf_data: bytes) -> DocumentBlocks:
    """
    PDFからテキストブロックの座標を詳細形式で取得します。

    Args:
        pdf_data: PDFのバイナリデータ

    Returns:
        ページごとのテキストブロック情報のリスト
    """
    document = await asyncio.to_thread(fitz.open, stream=pdf_data, filetype="pdf")
    content: DocumentBlocks = []

    for page_num in range(len(document)):
        page = await asyncio.to_thread(document.load_page, page_num)
        text_instances_dict = await asyncio.to_thread(page.get_text, "dict")
        text_instances = text_instances_dict["blocks"]
        page_content: PageBlocks = []

        for lines in text_instances:
            if lines["type"] != 0:
                # テキストブロック以外はスキップ
                continue

            block: BlockInfo = {
                "page_no": page_num,
                "block_no": lines["number"],
                "coordinates": lines["bbox"],
                "text": "",
            }

            sizes: list[float] = []
            for line in lines["lines"]:
                for span in line["spans"]:
                    if block["text"] == "":
                        block["text"] += span["text"]
                    else:
                        block["text"] += " " + span["text"]
                    sizes.append(span["size"])
                    block["font"] = span["font"]

            block["size"] = np.mean(sizes) if sizes else 0.0
            page_content.append(block)

        content.append(page_content)

    await asyncio.to_thread(document.close)
    return content


def _check_first_num_tokens(
    input_list: list[str], keywords: list[str], num: int = 2
) -> bool:
    """
    トークンリストの先頭N個にキーワードが含まれるかチェックします。

    Args:
        input_list: トークンのリスト
        keywords: 検索するキーワードのリスト
        num: チェックするトークン数

    Returns:
        キーワードが見つかった場合True
    """
    for item in input_list[:num]:
        for keyword in keywords:
            if keyword.lower() in item.lower():
                return True
    return False


async def remove_blocks(
    block_info: DocumentBlocks,
    token_threshold: int = 10,
    debug: bool = False,
    lang: str = "en",
) -> tuple[DocumentBlocks, DocumentBlocks, DocumentBlocks, Optional[list[bytes]]]:
    """
    ブロックを本文/図表/除外に分類します。

    トークン数、ブロック幅、フォントサイズに基づいてスコアを計算し、
    ヒストグラム分析により本文ブロックを識別します。

    Args:
        block_info: ブロック情報のリスト
        token_threshold: トークン数しきい値（これ以下は除外候補）
        debug: デバッグモード（可視化画像を生成）
        lang: 言語コード

    Returns:
        (本文ブロック, 図表ブロック, 除外ブロック, デバッグ画像リスト)
    """
    filtered_blocks: DocumentBlocks = []
    fig_table_blocks: DocumentBlocks = []
    removed_blocks: DocumentBlocks = []

    # データの抽出
    bboxs = [item["coordinates"] for sublist in block_info for item in sublist]
    widths = [x1 - x0 for x0, _, x1, _ in bboxs]
    sizes = [item["size"] for sublist in block_info for item in sublist]

    # テキストのトークン化
    text_list = [item["text"] for sublist in block_info for item in sublist]
    for i in range(len(text_list)):
        text_list[i] = text_list[i].replace("\n", "")
        text_list[i] = "".join(
            char
            for char in text_list[i]
            if char not in string.punctuation and char not in string.digits
        )
    texts = [tokenize_text(lang, text) for text in text_list]
    texts = [len(text) for text in texts]

    # スコア計算
    scores: list[list[float]] = []
    for text in texts:
        if token_threshold <= text:
            scores.append([0.0])
        else:
            scores.append([1.0])

    # IQR（ロバストスケーリング）によるスコア計算
    for item in [widths, sizes]:
        item_median = median(item)
        item_75_percentile = float(np.percentile(item, 75))
        item_25_percentile = float(np.percentile(item, 25))

        for value, score_list in zip(item, scores):
            iqr = item_75_percentile - item_25_percentile
            if iqr > 0:
                score = abs((value - item_median) / iqr)
            else:
                score = 0.0
            score_list.append(score)

    # スコアの合計
    marge_score = [sum(list_score) for list_score in scores]

    # ヒストグラムから基準値を算出
    n = len(marge_score)
    num_bins_sturges = math.ceil(math.log2(n) + 1)

    q75, q25 = np.percentile(marge_score, [75, 25])
    iqr = q75 - q25

    bin_width_fd = 2 * iqr / n ** (1 / 3) if n > 0 else 1
    bin_range = max(marge_score) - min(marge_score) if marge_score else 1
    num_bins_fd = math.ceil(bin_range / bin_width_fd) if bin_width_fd > 0 else 1

    num_bins = min(num_bins_sturges, num_bins_fd)
    num_bins = max(num_bins, 1)

    histogram, bin_edges = np.histogram(marge_score, bins=num_bins)
    max_index = int(np.argmax(histogram))
    most_frequent_range = (bin_edges[max_index], bin_edges[max_index + 1])

    # ブロック分類
    i = 0
    for pages in block_info:
        page_filtered_blocks: PageBlocks = []
        page_fig_table_blocks: PageBlocks = []
        page_removed_blocks: PageBlocks = []

        for block in pages:
            block_text = block["text"]
            tokens_list = tokenize_text(lang, block_text)

            score = marge_score[i]
            size = math.floor((sizes[i]) * 100) / 100
            result = bool(
                most_frequent_range[0] <= score <= most_frequent_range[1]
                and scores[i][0] == 0
            )

            # 図表キーワードチェック
            keywords = FIG_KEYWORDS_JA if lang == "ja" else FIG_KEYWORDS_EN
            is_figure = _check_first_num_tokens(tokens_list, keywords)

            if is_figure:
                page_fig_table_blocks.append(block)
            elif most_frequent_range[0] <= score <= most_frequent_range[1] and scores[i][0] == 0:
                page_filtered_blocks.append(block)
            else:
                add_block = copy.copy(block)
                printscore = (
                    f"[{math.floor(score * 100) / 100}/{result}] "
                    f"/T:{math.floor((scores[i][0])*100)/100}({texts[i]})"
                    f"/W:{math.floor((scores[i][1])*100)/100}"
                    f"/S:{math.floor((scores[i][2])*100)/100}({size})"
                )
                add_block["text"] = printscore
                page_removed_blocks.append(add_block)

            i += 1

        fig_table_blocks.append(page_fig_table_blocks)
        filtered_blocks.append(page_filtered_blocks)
        removed_blocks.append(page_removed_blocks)

    if debug:
        # デバッグ用可視化データ生成
        size_median = median(sizes)
        size_mean = float(np.mean(sizes))

        texts_raw = [item["text"] for sublist in block_info for item in sublist]
        tokens: list[int] = []
        for text in texts_raw:
            text = text.replace("\n", "")
            text = "".join(
                char
                for char in text
                if char not in string.punctuation and char not in string.digits
            )
            token = tokenize_text("en", text)
            tokens.append(len(token))

        token_median = median(tokens)
        token_mean = float(np.mean(tokens))

        token_mean_img = plot_area_distribution(
            areas=tokens,
            labels_values=[{"Median": token_median}, {"Mean": token_mean}],
            title="token Mean",
            xlabel="Token",
            ylabel="Frequency",
        )
        size_mean_img = plot_area_distribution(
            areas=sizes,
            labels_values=[{"Median": size_median}, {"Mean": size_mean}],
            title="Size Mean",
            xlabel="font size",
            ylabel="Frequency",
        )
        score_mean_img = plot_area_distribution(
            areas=marge_score,
            labels_values=[
                {"Histogram Low": most_frequent_range[0]},
                {"Histogram High": most_frequent_range[1]},
            ],
            title="score Mean",
            xlabel="score",
            ylabel="Frequency",
        )

        return (
            filtered_blocks,
            fig_table_blocks,
            removed_blocks,
            [token_mean_img, size_mean_img, score_mean_img],
        )

    return filtered_blocks, fig_table_blocks, removed_blocks, None


async def remove_textbox_for_pdf(
    pdf_data: bytes, remove_list: DocumentBlocks
) -> bytes:
    """
    PDFからテキストブロックを削除します。

    Args:
        pdf_data: PDFのバイナリデータ
        remove_list: 削除するブロックのリスト

    Returns:
        処理後のPDFバイナリデータ
    """
    doc = await asyncio.to_thread(fitz.open, stream=pdf_data, filetype="pdf")

    for remove_data, page in zip(remove_list, doc):
        for remove_item in remove_data:
            rect = fitz.Rect(remove_item["coordinates"])
            await asyncio.to_thread(page.add_redact_annot, rect)
        await asyncio.to_thread(page.apply_redactions)

    output_buffer = BytesIO()
    await asyncio.to_thread(doc.save, output_buffer, garbage=4, deflate=True, clean=True)
    await asyncio.to_thread(doc.close)

    return output_buffer.getvalue()


async def pdf_draw_blocks(
    input_pdf_data: bytes,
    block_info: DocumentBlocks,
    line_color_rgb: list[float] = None,
    width: float = 5,
    fill_color_rgb: list[float] = None,
    fill_opacity: float = 1,
) -> bytes:
    """
    PDFにデバッグ用の枠を描画します。

    Args:
        input_pdf_data: PDFのバイナリデータ
        block_info: ブロック情報のリスト
        line_color_rgb: 線の色 [R, G, B]
        width: 線の幅
        fill_color_rgb: 塗りつぶしの色 [R, G, B]
        fill_opacity: 塗りつぶしの透明度

    Returns:
        処理後のPDFバイナリデータ
    """
    if line_color_rgb is None:
        line_color_rgb = [0, 0, 1]
    if fill_color_rgb is None:
        fill_color_rgb = [0, 0, 1]

    doc = await asyncio.to_thread(fitz.open, stream=input_pdf_data, filetype="pdf")

    for i, pages in enumerate(block_info):
        page = doc[i]
        for block in pages:
            x0, y0, x1, y1 = block["coordinates"]
            text_rect = fitz.Rect(x0, y0, x1, y1)
            await asyncio.to_thread(
                page.draw_rect,
                text_rect,
                color=line_color_rgb,
                width=width,
                fill=fill_color_rgb,
                fill_opacity=fill_opacity,
            )

    output_buffer = BytesIO()
    await asyncio.to_thread(doc.save, output_buffer, garbage=4, deflate=True, clean=True)
    await asyncio.to_thread(doc.close)

    return output_buffer.getvalue()


def _get_font_config(to_lang: str) -> tuple[str, str, str]:
    """
    言語に応じたフォント設定を取得します。

    Args:
        to_lang: 対象言語コード

    Returns:
        (フォントパス, サンプル文字, フォールバックフォント名)
    """
    if to_lang == "en":
        return FONT_PATH_EN, "a", FALLBACK_FONT
    elif to_lang == "ja":
        return FONT_PATH_JA, "あ", FALLBACK_FONT
    else:
        return FONT_PATH_EN, "a", FALLBACK_FONT


def _check_font_availability(font_path: str, to_lang: str) -> tuple[bool, Optional[str]]:
    """
    フォントファイルの存在を確認します。

    Args:
        font_path: フォントファイルのパス
        to_lang: 対象言語コード

    Returns:
        (組み込みフォント使用フラグ, 使用可能なフォントパス)
    """
    if not os.path.exists(font_path):
        logger.warning(f"フォントファイルが見つかりません: {font_path}")
        logger.warning(f"PyMuPDF組み込みフォント '{FALLBACK_FONT}' にフォールバックします。")
        if to_lang == "ja":
            logger.warning("注意: 日本語テキストは正しく表示されない可能性があります。")
        return True, None
    return False, font_path


async def preprocess_write_blocks(
    block_info: DocumentBlocks, to_lang: str = "ja"
) -> DocumentBlocks:
    """
    翻訳テキストをPDFに書き込むための前処理を行います。

    テキストがブロック領域に収まるようにフォントサイズを調整し、
    ブロックごとにテキストを分割します。

    Args:
        block_info: ブロック情報のリスト
        to_lang: 対象言語コード

    Returns:
        処理後のブロック情報
    """
    font_path, a_text, fallback_fontname = _get_font_config(to_lang)
    use_builtin_font, actual_font_path = _check_font_availability(font_path, to_lang)

    any_blocks: list[BlockInfo] = []

    for page in block_info:
        for box in page:
            font_size = box["size"][0]

            while True:
                # フォントサイズ計算
                if use_builtin_font:
                    font = fitz.Font(fallback_fontname)
                else:
                    font = fitz.Font("F0", actual_font_path)
                a_width = font.text_length(a_text, font_size)

                # BOXに収まるテキスト数を計算
                max_chars_per_boxes: list[list[int]] = []
                for coordinates in box["coordinates"]:
                    x1, y1, x2, y2 = coordinates
                    height = y2 - y1
                    width = x2 - x1

                    num_columns = int(height / (font_size * LH_CALC_FACTOR))
                    num_raw = int(width / a_width)
                    max_chars_per_boxes.append([num_raw] * num_columns)

                # テキストを処理
                text_all = box["text"].replace(" ", "\u00A0")
                text_list = text_all.split("\n")

                text = text_list.pop(0) if text_list else ""
                text_num = len(text)
                box_texts: list[str] = []
                exit_flag = False

                for chars_per_box in max_chars_per_boxes:
                    if exit_flag:
                        break
                    box_text = ""

                    for chars_per_line in chars_per_box:
                        if exit_flag:
                            break
                        text_num = text_num - chars_per_line

                        if text_num <= 0:
                            box_text += text + "\n"
                            if not text_list:
                                exit_flag = True
                                text = ""
                                break
                            text = text_list.pop(0)
                            text_num = len(text)

                    if len(text) != text_num:
                        cut_length = len(text) - text_num
                        box_text += text[:cut_length]
                        text = text[cut_length:]
                    box_texts.append(box_text)

                if not text_list and text == "":
                    break
                else:
                    font_size -= FONT_SIZE_DECREMENT

            box_texts = [t.lstrip().rstrip("\n") for t in box_texts]

            for page_no, block_no, coordinates, text in zip(
                box["page_no"], box["block_no"], box["coordinates"], box_texts
            ):
                result_block: BlockInfo = {
                    "page_no": page_no,
                    "block_no": block_no,
                    "coordinates": coordinates,
                    "text": text,
                    "size": font_size,
                }
                any_blocks.append(result_block)

    page_groups: dict[int, list[BlockInfo]] = defaultdict(list)
    for block in any_blocks:
        page_groups[block["page_no"]].append(block)

    return list(page_groups.values())


async def write_pdf_text(
    input_pdf_data: bytes,
    block_info: DocumentBlocks,
    to_lang: str = "en",
    text_color: list[float] = None,
    font_path: Optional[str] = None,
) -> bytes:
    """
    PDFにテキストを書き込みます。

    Args:
        input_pdf_data: PDFのバイナリデータ
        block_info: ブロック情報のリスト
        to_lang: 対象言語コード
        text_color: テキスト色 [R, G, B]
        font_path: フォントファイルのパス

    Returns:
        処理後のPDFバイナリデータ
    """
    if text_color is None:
        text_color = [0, 0, 0]

    # フォント選択
    if font_path is None:
        if to_lang == "en":
            font_path = FONT_PATH_EN
        elif to_lang == "ja":
            font_path = FONT_PATH_JA

    use_builtin_font, actual_font_path = _check_font_availability(font_path, to_lang)

    doc = await asyncio.to_thread(fitz.open, stream=input_pdf_data, filetype="pdf")

    for page_block in block_info:
        for block in page_block:
            page_num = block["page_no"]
            page = doc[page_num]

            if use_builtin_font:
                page.insert_font(fontname=FALLBACK_FONT)
            else:
                page.insert_font(fontname="F0", fontfile=actual_font_path)

            coordinates = list(block["coordinates"])
            text = block["text"]
            font_size = block["size"]
            active_fontname = FALLBACK_FONT if use_builtin_font else "F0"

            while True:
                rect = fitz.Rect(coordinates)
                result = page.insert_textbox(
                    rect,
                    text,
                    fontsize=font_size,
                    fontname=active_fontname,
                    align=3,
                    lineheight=LINE_HEIGHT_FACTOR,
                    color=text_color,
                )
                if result >= 0:
                    break
                else:
                    coordinates[3] += 1

    output_buffer = BytesIO()
    await asyncio.to_thread(doc.save, output_buffer, garbage=4, deflate=True, clean=True)
    await asyncio.to_thread(doc.close)

    return output_buffer.getvalue()


async def write_logo_data(input_pdf_data: bytes) -> bytes:
    """
    PDFにサービスロゴを描画します。

    Args:
        input_pdf_data: PDFのバイナリデータ

    Returns:
        処理後のPDFバイナリデータ
    """
    doc = await asyncio.to_thread(fitz.open, stream=input_pdf_data, filetype="pdf")

    use_builtin_font, actual_font_path = _check_font_availability(FONT_PATH_EN, "en")
    active_fontname = FALLBACK_FONT if use_builtin_font else "F0"

    for page in doc:
        if use_builtin_font:
            page.insert_font(fontname=FALLBACK_FONT)
        else:
            page.insert_font(fontname="F0", fontfile=actual_font_path)

        page.insert_image(LOGO_RECT, filename=LOGO_PATH)
        page.insert_textbox(
            (37, 5, 100, 35), "Translated by.", fontsize=5, fontname=active_fontname
        )
        page.insert_textbox(
            (37, 12, 100, 35), "IndQx", fontsize=10, fontname=active_fontname
        )
        page.insert_textbox(
            (37, 25, 100, 35), "Translation.", fontsize=5, fontname=active_fontname
        )

    output_buffer = BytesIO()
    await asyncio.to_thread(doc.save, output_buffer, garbage=4, deflate=True, clean=True)
    await asyncio.to_thread(doc.close)

    return output_buffer.getvalue()


async def create_viewing_pdf(
    base_pdf_data: bytes, translated_pdf_data: bytes
) -> bytes:
    """
    オリジナルPDFと翻訳PDFを見開き形式で結合します。

    Args:
        base_pdf_data: オリジナルPDFのバイナリデータ
        translated_pdf_data: 翻訳PDFのバイナリデータ

    Returns:
        見開きPDFのバイナリデータ
    """
    doc_base = await asyncio.to_thread(
        fitz.open, stream=base_pdf_data, filetype="pdf"
    )
    doc_translate = await asyncio.to_thread(
        fitz.open, stream=translated_pdf_data, filetype="pdf"
    )

    new_doc = fitz.open()

    for page_num in range(len(doc_base)):
        page_base = doc_base.load_page(page_num)
        page_translate = doc_translate.load_page(page_num)

        rect_base = page_base.rect
        rect_translate = page_translate.rect

        max_height = max(rect_base.height, rect_translate.height)

        # オリジナルページを左に追加
        new_page = new_doc.new_page(width=rect_base.width, height=max_height)
        new_page.show_pdf_page(new_page.rect, doc_base, page_num)

        # 翻訳ページを右に追加
        new_page = new_doc.new_page(width=rect_translate.width, height=max_height)
        new_page.show_pdf_page(new_page.rect, doc_translate, page_num)

    new_doc.set_pagelayout("TwoPageLeft")

    output_buffer = BytesIO()
    await asyncio.to_thread(
        new_doc.save, output_buffer, garbage=4, deflate=True, clean=True
    )
    await asyncio.to_thread(new_doc.close)
    await asyncio.to_thread(doc_base.close)
    await asyncio.to_thread(doc_translate.close)

    return output_buffer.getvalue()


def plot_area_distribution(
    areas: list[float],
    labels_values: list[dict[str, float]],
    title: str = "Distribution of Areas",
    xlabel: str = "Area",
    ylabel: str = "Frequency",
) -> bytes:
    """
    デバッグ用のヒストグラムを生成します。

    Args:
        areas: データ値のリスト
        labels_values: ラベルと値の辞書のリスト
        title: グラフタイトル
        xlabel: X軸ラベル
        ylabel: Y軸ラベル

    Returns:
        PNG画像のバイナリデータ
    """
    import matplotlib.pyplot as plt

    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=100, color="skyblue", edgecolor="black", alpha=0.7)

    for i, label_value in enumerate(labels_values):
        for label, value in label_value.items():
            color = colors[i % len(colors)]
            plt.axvline(
                value,
                color=color,
                linestyle="dashed",
                linewidth=1.5,
                label=f"{label}: {value:.2f}",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return buf.read()
