import fitz  # PyMuPDF
import asyncio
import math,re,html
from io import BytesIO
from statistics import median
from modules.spacy_api import *
from collections import defaultdict

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
    """
    pdf バイトデータのテキストファイル座標を取得します。
    """
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
            block["page_no"] = page_num
            block["block_no"] = lines["number"]
            block["coordinates"] = lines["bbox"]
            block["text"] = ""
            for line in lines['lines']:
                for span in line['spans']:
                    if block["text"] == "":
                        block["text"]+=span["text"]
                    else:
                        block["text"]+=" " + span["text"]
                    block["size"]=span['size']
                    block["font"]=span['font']
            # block["text_count"] = len(block["text"])
            page_content.append(block)
        
        content.append(page_content)
    await asyncio.to_thread(document.close)
    return content

async def extract_text_coordinates_dict_dev(pdf_data):
    """
    デバッグ用。dictで取得したデータを出力します。
    """
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
            if lines["type"] != 0:
                # テキストブロック以外はスキップ
                continue
            page_content.append(lines)
        
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


async def preprocess_write_blocks(block_info, to_lang='ja'):
    lh_calc_factor = 1.3

    # フォント選択
    if to_lang == 'en':
        font_path = 'fonts/TIMES.TTF'
        a_text = 'a'
    elif to_lang == 'ja':
        font_path = 'fonts/MSMINCHO.TTC'
        a_text = 'あ'

    # フォントサイズを逆算+ブロックごとにテキストを分割
    any_blocks = []
    for page in block_info:
        for box in page:

            font_size = box["size"][0]

            while True:
                #初期化
                max_chars_per_boxes = []

                # フォントサイズ計算
                font = fitz.Font("F0",font_path)
                a_width=font.text_length(a_text,font_size)

                # BOXに収まるテキスト数を行ごとにリストに格納
                max_chars_per_boxes = []
                for coordinates in box["coordinates"]:
                    x1,y1,x2,y2 = coordinates
                    hight = y2-y1
                    width = x2-x1

                    num_colums = int(hight/(font_size*lh_calc_factor))
                    num_raw = int(width/a_width)
                    max_chars_per_boxes.append([num_raw]*num_colums)
                
                # 文字列を改行ごとに分割してリストに格納
                #text_all = box["text"].replace(' ', '\u00A0') #スペースを改行されないノーブレークスペースに置き換え
                text_all = box["text"]
                text_list = text_all.split('\n')

                text = text_list.pop(0)
                text_num = len(text)
                box_texts = []
                exit_flag = False

                for chars_per_box in max_chars_per_boxes:
                    #各箱ごとを摘出
                    if exit_flag:
                        break
                    box_text = ""

                    for chars_per_line in chars_per_box:
                        #1行あたりに代入できる文字数 : chars_per_line
                        if exit_flag:
                            break
                        # 行に文字を代入した際の残り文字数を計算
                        text_num = text_num - chars_per_line
                        #print(F"{chars_per_line}/{text_num}")
                        if text_num <= 0:
                            #その行にて収まる場合は、次の文字列を取り出す
                            box_text += text + "\n"
                            #print("add str to box")
                            if text_list == []:
                                #次の文字列がない場合はbreak
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
                if text_list == [] and text == "":
                    break
                else:
                    font_size-=0.1
            box_texts = [text.lstrip().rstrip('\n') for text in box_texts]
            for page_no,block_no,coordinates,text in zip(box["page_no"],box["block_no"],box["coordinates"],box_texts):
                result_block = {"page_no":page_no,
                                "block_no":block_no,
                                "coordinates":coordinates,
                                "text":text,
                                "size":font_size}
                any_blocks.append(result_block)
    page_groups = defaultdict(list)
    for block in any_blocks:
        page_groups[block["page_no"]].append(block)
    # 結果としてのリストのリスト
    grouped_pages = list(page_groups.values())
    return grouped_pages

async def write_pdf_text(input_pdf_data, block_info, to_lang='en', debug=False):
    """
    指定されたフォントで、文字を作画します。
    """
    lh_factor = 1.5  # 行の高さの係数

    # フォント選択
    if to_lang == 'en':
        font_path = 'fonts/TIMES.TTF'
    elif to_lang == 'ja':
        font_path = 'fonts/MSMINCHO.TTC'

    doc = await asyncio.to_thread(fitz.open, stream=input_pdf_data, filetype="pdf")

    for page_block in block_info:
        
        for block in page_block:
            #ページ設定
            page_num = block["page_no"]
            page = doc[page_num]
            page.insert_font(fontname="F0", fontfile=font_path)
            # 書き込み実施
            coordinates = list(block["coordinates"])
            text = block["text"]
            font_size = block["size"]
            if debug:
                rect = fitz.Rect(coordinates)
                page.draw_rect(rect,color=[0, 0, 1],width=0,fill=[0,0,1],fill_opacity=0.3)
            while True:
                rect = fitz.Rect(coordinates)
                result = page.insert_textbox(rect, text, fontsize=font_size, fontname="F0", align=3, lineheight=lh_factor)
                if result >=0:
                    break   
                else:
                    coordinates[3]+=1
        
    output_buffer = BytesIO()
    await asyncio.to_thread(doc.save, output_buffer, garbage=4, deflate=True, clean=True)
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
