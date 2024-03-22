import fitz  # PyMuPDF
import asyncio
from io import BytesIO
from statistics import median
from spacy_api import *

async def extract_text_coordinates(pdf_data):
    """
    pdf バイトデータのテキストファイル座標を取得します
    """
    # PDFファイルを開く
    document = await asyncio.to_thread(fitz.open, stream=pdf_data, filetype="pdf")

    # 全ページのテキスト、画像と座標を格納するためのリスト
    content = []

    for page_num in range(len(document)):
        page_content = []

        # ページを取得
        page = await asyncio.to_thread(document.load_page, page_num)
        # ページサイズを取得
        page_size = await asyncio.to_thread(lambda: page.rect)
        width, height = page_size.width, page_size.height

        # ページからテキストブロックを取得
        blocks = await asyncio.to_thread(page.get_text, "blocks")

        # 各テキストブロックからテキスト、画像と座標を抽出
        for b in blocks:
            x0, y0, x1, y1, content_text, block_no, block_type = b[:7]

            if block_type == 0:  # テキストブロック
                block_info = {
                    "page_num": page_num,
                    "text": content_text,
                    "coordinates": (x0, y0, x1, y1),
                    "block_no": block_no,
                    "block_type": block_type,
                    "page_width": width,
                    "page_height": height
                }
                page_content.append(block_info)
            else:
                print("Block:")
                print(b)

        content.append(page_content)

    await asyncio.to_thread(document.close)

    return content

async def remove_blocks_with_few_words(block_info, token_threshold=10,debug=False):
    import string
    import numpy as np
    """
    英単語数が指定された閾値以下のブロックをリストから削除します。更にブロックの面積中央値を求め中央値以下のブロックもリストから消去します。
    削除されたブロックも返します。

    :param block_info: ブロック情報のリスト
    :param token_threshold: 単語トークンしきい値。この値を下回る場合は無視される
    :return: 更新されたブロック情報のリストと削除されたブロック情報のリスト
    """
    #フィルターに基づいて分離する。
    filtered_blocks = []
    removed_blocks = []

    # boxデータの分割
    bboxs = [item['coordinates'] for sublist in block_info for item in sublist]
    # ブロック幅のしきい値を求める
    widths = [x1 - x0 for x0, _, x1, _ in bboxs]
    for i in range(100,-1,-25):
        percentile = np.percentile(widths,i)
        if percentile < 300:
            break
    width_threshold_low = 0.9 * percentile
    width_threshold_high = 1.1 * percentile

    save_data = []
    for pages in block_info:
        page_filtered_blocks = []
        page_removed_blocks = []
        for block in pages:
            #boxを変数化
            block_text = block["text"]
            block_coordinates = block['coordinates']
            #widthを計算
            width = (block_coordinates[2] - block_coordinates[0])
            #tokenを計算
            token = len(tokenize_text('en', block_text))

            #記号と数字が50%を超える場合は、リストから消去
            no_many_symbol = True
            symbol_and_digit_count = sum(1 for char in block_text if char in string.punctuation or char in string.digits)
            if len(block_text)!=0:
                no_many_symbol = symbol_and_digit_count / len(block_text) < 0.5

            width_bool = bool(width_threshold_high > width > width_threshold_low)
            no_symbol_bool = bool(no_many_symbol)
            token_bool = bool(token_threshold<token)
            
            if debug:
                save_data.append({"Text": block_text,
                                    "result bool": no_symbol_bool and width_bool,
                                    "width bool": width_bool,
                                    "token bool":token_bool,
                                    "no symbol Bool": no_symbol_bool,
                                    "width_threshold_low": float(width_threshold_low),
                                    "this with": float(width),
                                    "width_threshold_high": float(width_threshold_high),
                                    "this token": token,
                                    "token threshold": token_threshold
                                    })
            
            if no_symbol_bool and width_bool and token_bool:
                page_filtered_blocks.append(block)
            else:
                page_removed_blocks.append(block)
        filtered_blocks.append(page_filtered_blocks)
        removed_blocks.append(page_removed_blocks)
    
    if debug:
        import json
        with open('save_string.json', 'w', encoding='utf-8') as json_file:
            json.dump(save_data, json_file, ensure_ascii=False, indent=2)

        # 解析用にデータを保存する
        width_median = median(widths)
        width_percentile_90 = np.percentile(widths, 90)
        width_percentile_75 = np.percentile(widths, 75)
        width_percentile_80 = np.percentile(widths, 80)
        mean_width = np.mean(widths)

        #tokenのしきい値を求める
        texts = [item['text'] for sublist in block_info for item in sublist]
        tokens = []
        for text in texts:
            # 記号と数字を除外し、それ以外の文字だけを含む文字列を生成
            text = text.replace("\n","")
            text = ''.join(char for char in text if char not in string.punctuation and char not in string.digits)
            token = tokenize_text('en', text)
            tokens.append(len(token))
        token_median = median(tokens)
        token_percentile_75 = np.percentile(tokens, 75)
        token_percentile_25 = np.percentile(tokens, 25)
        token_mean = np.mean(tokens)
        plot_area_distribution(areas=widths,labels_values=[{"Median":width_median},
                                                        {"threshold_low":width_threshold_low},
                                                        {"threshold_high":width_threshold_high},
                                                        {"Mean":mean_width},
                                                        {"percentile_75":width_percentile_75},
                                                        {"percentile_80":width_percentile_80},
                                                        {"percentile_90":width_percentile_90}],title="Awidth Mean",xlabel='width size',ylabel='Frequency',save_path='grah_With.png')
        plot_area_distribution(areas=tokens,labels_values=[{"Median":token_median},
                                                        {"Mean":token_mean},
                                                        {"percentile_25":token_percentile_25},
                                                        {"percentile_75":token_percentile_75}],title="token Mean",xlabel='Token',ylabel='Frequency',save_path='grah_token.png')
    return filtered_blocks, removed_blocks

async def remove_textbox_for_pdf(pdf_data, leave_text_list):
    """
    読み込んだPDFより、すべてのテキストデータを消去します。
    leave_text_listが設定されている場合、該当リストに含まれる文字列(部分一致)は保持します。
    """
    doc = await asyncio.to_thread(fitz.open, stream=pdf_data, filetype="pdf")

    for page in doc:  # ドキュメント内の各ページに対して
        text_instances_dict = await asyncio.to_thread(page.get_text, "dict")
        text_instances = text_instances_dict["blocks"]  # ページ内のテキストブロックを取得

        for inst in text_instances:
            if inst["type"] == 0:  # テキストブロックの場合
                for line in inst['lines']:
                    for span in line['spans']:
                        if any(span['text'] in data for data in leave_text_list):
                            pass
                        else:
                            rect = fitz.Rect(inst["bbox"])  # テキストブロックの領域を取得
                            await asyncio.to_thread(page.add_redact_annot, rect)  # レダクションアノテーションを追加

        await asyncio.to_thread(page.apply_redactions)  # レダクションを適用してテキストを削除

    output_buffer = BytesIO()
    await asyncio.to_thread(doc.save, output_buffer)  # 変更をバッファに保存
    await asyncio.to_thread(doc.close)

    output_data = output_buffer.getvalue()
    return output_data

async def pdf_draw_blocks(input_pdf_data, block_info, line_colorRGB=[0, 0, 1], width=5, fill_colorRGB=[0, 0, 1], fill_opacity=1):
    """
    PDFデータに、デバッグ用、四角い枠を作画します
    """
    doc = await asyncio.to_thread(fitz.open, stream=input_pdf_data, filetype="pdf")

    for i, pages in enumerate(block_info):
        page = doc[i]
        for block in pages:
            x0, y0, x1, y1 = block["coordinates"]
            text_rect = fitz.Rect(x0, y0, x1, y1)
            await asyncio.to_thread(page.draw_rect, text_rect, color=line_colorRGB, width=width, fill=fill_colorRGB, fill_opacity=fill_opacity)

    output_buffer = BytesIO()
    await asyncio.to_thread(doc.save, output_buffer)
    await asyncio.to_thread(doc.close)

    output_data = output_buffer.getvalue()
    return output_data

async def write_pdf_text(input_pdf_data, block_info, lang='en', debug=False):
    """
    指定されたフォントで、文字を作画します。
    """
    lh_factor = 1.5  # 行の高さの係数
    # フォント選択
    if lang == 'en':
        font_path = 'fonts/TIMES.TTF'
        lang_a = 'a'
    elif lang == 'ja':
        font_path = 'fonts/MSMINCHO.TTC'
        lang_a ="あ"

    doc = await asyncio.to_thread(fitz.open, stream=input_pdf_data, filetype="pdf")
    font = await asyncio.to_thread(fitz.Font, fontfile=font_path)

    for i, pages in enumerate(block_info):
        #ページごとのループ
        page = doc[i]
        await asyncio.to_thread(page.insert_font, fontfile=font_path, fontname="F0")

        for block in pages:
            #ブロックごとのループ
            # 文字列挿入用ブロックの定義
            text = block["text"]
            x0, y0, x1, y1 = block["coordinates"]
            text_rect = fitz.Rect(x0, y0, x1, y1)

            # フォントサイズと行の高さの計算
            fs_min = 1  # 最小フォントサイズ
            fs_max = int(y1-y0)  # 最大フォントサイズ

            while fs_min <= fs_max:
                fs = (fs_min + fs_max) // 2  # 中間値のフォントサイズを計算

                # text_boxに入力可能な幅文字数を計算
                lang_a_width = await asyncio.to_thread(font.text_length, lang_a, fontsize=fs)
                width_tc = int((x1-x0)/lang_a_width)

                # text_boxに入力可能な高さ文字数を計算
                lh = (font.ascender - font.descender) * fs * lh_factor  # 自然行の高さの計算
                hight_tc = int((y1-y0)/lh)

                text_no_newlines = text.replace('\n', '')
                text_length = len(text_no_newlines)
                n_text_count = text.count('\n')

                if text_length <= width_tc*hight_tc:
                    if debug:
                        print(f"文字数:{text_length} <= 入力可能文字数:{width_tc*hight_tc}")
                        print(f"横文字数:{width_tc} 縦文字数:{hight_tc}")
                        print(f"改行数:{n_text_count} <= 改行可能数:{hight_tc}")
                    fs_min = fs + 1
                else:
                    fs_max = fs - 1

            fs = fs_max  # 適切なフォントサイズを設定

            if fs > 0:  # 適切なフォントサイズが見つかった場合
                # テキストをページに追加
                result = await asyncio.to_thread(page.insert_textbox, text_rect, text_no_newlines, fontsize=fs, fontname="F0", lineheight=lh_factor)
                print(result)

    output_buffer = BytesIO()
    await asyncio.to_thread(doc.save, output_buffer)
    await asyncio.to_thread(doc.close)
    output_data = output_buffer.getvalue()

    return output_data

def plot_area_distribution(areas, labels_values, title='Distribution of Areas', xlabel='Area', ylabel='Frequency', save_path=None):
    """
    デバッグ用、グラフを作画する
    """
    import numpy as np
    import matplotlib.pyplot as plt
    # 主要な色のリスト
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=100, color='skyblue', edgecolor='black', alpha=0.7)

    for i, label_value in enumerate(labels_values):
        for label, value in label_value.items():
            color = colors[i % len(colors)]  # 色のリストから順番に色を選択
            plt.axvline(value, color=color, linestyle='dashed', linewidth=1.5, label=f'{label}: {value:.2f}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel,)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


async def text_draw_test():
    import json
    to_lang = 'ja'

    with open("input.pdf", "rb") as f:
        input_pdf_data = f.read()

    block_info = await extract_text_coordinates(input_pdf_data)
    blocked_pdf_data = await pdf_draw_blocks(input_pdf_data,block_info,width=0,fill_opacity=0.3)

    block_info,removed_blocks = await remove_blocks_with_few_words(block_info,10)

    # removed_blockをリストに分解
    leave_str_list = [item['text'] for sublist in removed_blocks for item in sublist]
    removed_textbox_pdf_data = await remove_textbox_for_pdf(blocked_pdf_data,leave_str_list)
    
    with open('block_info.json', 'w', encoding='utf-8') as json_file:
        json.dump(block_info, json_file, ensure_ascii=False, indent=2)

    
    with open('block_info_translated.json', 'r', encoding='utf-8') as json_file:
        block_info = json.load(json_file)
    
    translated_pdf_data = await write_pdf_text(removed_textbox_pdf_data,block_info,to_lang)

    with open("output.pdf", "wb") as f:
        f.write(translated_pdf_data)

if __name__ == "__main__":
    asyncio.run(text_draw_test())