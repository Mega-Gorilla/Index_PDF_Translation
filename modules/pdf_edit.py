import fitz  # PyMuPDF
import asyncio
import math,re,html
from io import BytesIO
from statistics import median
from modules.spacy_api import *

async def extract_text_coordinates_blocks(pdf_data):
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
            
            # フォントサイズ　逆算
            count_lines = content_text.count('\n')
            if count_lines != 0:
                calc_fs=(y1-y0)/count_lines*0.98
            else:
                calc_fs = y1-y0
            calc_fs = math.floor(calc_fs * 100) / 100

            if block_type == 0:  # テキストブロック
                block_info = {
                    "block_no": block_no,
                    "text": content_text,
                    "size": calc_fs,
                    "coordinates": (x0, y0, x1, y1)
                }
                page_content.append(block_info)
            else:
                print("Block:")
                print(b)

        content.append(page_content)

    await asyncio.to_thread(document.close)

    return content

async def extract_text_coordinates_dict(pdf_data):
    # PDFファイルを開く
    document = await asyncio.to_thread(fitz.open, stream=pdf_data, filetype="pdf")

    content = []
    for page_num in range(len(document)):
        # ページを取得
        page = await asyncio.to_thread(document.load_page, page_num)
        # ページからテキストブロックを取得
        text_instances_dict = await asyncio.to_thread(page.get_text, "dict")
        text_instances = text_instances_dict["blocks"]
        page_content = []
        
        for lines in text_instances:
            block = {}
            if lines["type"] != 0:
                # テキストブロック以外はスキップ
                continue
            block["block_no"] = lines["number"]
            block["coordinates"] = lines["bbox"]
            block["text"] = ""
            for line in lines['lines']:
                for span in line['spans']:
                    if block["text"] == "":
                        block["text"]+=span["text"]
                    else:
                        block["text"]+="\n" + span["text"]
                    block["size"]=span['size']
                    block["font"]=span['font']
            page_content.append(block)
        
        content.append(page_content)
    await asyncio.to_thread(document.close)
    return content

def check_first_num_tokens(input_list, keywords, num=2):
    for item in input_list[:num]:
        for keyword in keywords:
            if keyword.lower() in item.lower():
                return True
    return False

async def remove_blocks(block_info, token_threshold=10,debug=False,lang='en'):
    import string
    import numpy as np
    """
    トークン数が指定された閾値以下のブロックをリストから削除します。更にブロックの幅のパーセンタイルを求め、幅300以上のパーセンタイルブロックをリストから消去します。
    削除されたブロックも返します。

    :param block_info: ブロック情報のリスト
    :param token_threshold: 単語トークンしきい値。この値を下回る場合は無視される
    :return: 更新されたブロック情報のリストと削除されたブロック情報のリスト
    """
    #フィルターに基づいて分離する。
    filtered_blocks = []
    fig_table_blocks = []
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
        page_fig_table_blocks = []
        page_removed_blocks = []
        for block in pages:
            #boxを変数化
            block_text = block["text"]
            block_coordinates = block['coordinates']
            #widthを計算
            width = (block_coordinates[2] - block_coordinates[0])
            #tokenを計算
            tokens_list = tokenize_text(lang,block_text)
            token = len(tokens_list)

            #記号と数字が50%を超える場合は、リストから消去
            no_many_symbol = True
            symbol_and_digit_count = sum(1 for char in block_text if char in string.punctuation or char in string.digits)
            if len(block_text)!=0:
                no_many_symbol = symbol_and_digit_count / len(block_text) < 0.5

            # tokenリスト３番目までに特定ワードが入ってる場合はグラフ表として認識する
            if lang == 'ja':
                keyword = ['表','グラフ']
            else:
                keyword = ['fig','table']
            table_bool = check_first_num_tokens(tokens_list,keyword)
            
            #文字列として認識するBool関数
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
            if table_bool:
                page_fig_table_blocks.append(block)
            elif no_symbol_bool and width_bool and token_bool:
                page_filtered_blocks.append(block)
            else:
                page_removed_blocks.append(block)
        fig_table_blocks.append(page_fig_table_blocks)
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
    return filtered_blocks,fig_table_blocks, removed_blocks

async def remove_textbox_for_pdf(pdf_data, remove_list):
    """
    読み込んだPDFより、すべてのテキストデータを消去します。
    leave_text_listが設定されている場合、該当リストに含まれる文字列(部分一致)は保持します。
    """
    doc = await asyncio.to_thread(fitz.open, stream=pdf_data, filetype="pdf")
    for remove_data,page in zip(remove_list,doc):
        for remove_item in remove_data:
            rect = fitz.Rect(remove_item["coordinates"])  # テキストブロックの領域を取得
            await asyncio.to_thread(page.add_redact_annot, rect)
        await asyncio.to_thread(page.apply_redactions)  # レダクションを適用してテキストを削除

    output_buffer = BytesIO()
    await asyncio.to_thread(doc.save, output_buffer, garbage=4, deflate=True, clean=True)
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
    await asyncio.to_thread(doc.save, output_buffer, garbage=4, deflate=True, clean=True)
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
    elif lang == 'ja':
        font_path = 'fonts/MSMINCHO.TTC'

    doc = await asyncio.to_thread(fitz.open, stream=input_pdf_data, filetype="pdf")
     # 一時的なPDFの作成
    temp_doc = fitz.open()

    # 元のPDFと同じページ構成で一時的なPDFを設定
    for _ in range(len(doc)):
        temp_doc.new_page()

    for i, pages in enumerate(block_info):
        #ページごとのループ
        page = doc[i]
        temp_page = temp_doc[i]
        
        page.insert_font(fontname="F0", fontfile=font_path)
        temp_page.insert_font(fontname="F0", fontfile=font_path)

        for block in pages:
            #ブロックごとのループ
            # 文字列挿入用ブロックの定義
            text = block["text"]
            x0, y0, x1, y1 = block["coordinates"]
            fs = block["size"]

            #text = text.replace('\n', '')
            text = text.replace(' ',"\u00A0")
            text_rect = fitz.Rect(x0, y0, x1, y1+15)

            # フォントサイズと行の高さの計算
            fs_min = 3  # 最小フォントサイズ
            fs_max = int(y1-y0)  # 最大フォントサイズ

            #print(text)

            best_result = float('inf')
            best_fs = None

            while fs_min <= fs_max:
                result = temp_page.insert_textbox(text_rect, text, fontname="F0", fontsize=fs, align=3, lineheight=lh_factor)
                #print(f"{fs} / {result}")
                if result < 0:
                    fs-=1
                    if best_fs != None:
                        #print(F"{best_result}/best_fs:{best_fs}")
                        fs = best_fs
                        break
                else:
                    #print(F"{result}/ Font_Size: {fs}")
                    break
            #正規のpdfに描画
            page.insert_textbox(text_rect, text, fontname="F0", fontsize=fs, align=3, lineheight=lh_factor)

    output_buffer = BytesIO()
    await asyncio.to_thread(doc.save, output_buffer, garbage=4, deflate=True, clean=True)
    await asyncio.to_thread(doc.close)
    await asyncio.to_thread(temp_doc.close)
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
