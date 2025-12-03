# SPDX-License-Identifier: AGPL-3.0-only
import fitz  # PyMuPDF
import asyncio
import math,re,html
from io import BytesIO
from statistics import median
from modules.spacy_api import *
from collections import defaultdict
import numpy as np

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
            sizes = []
            for line in lines['lines']:
                for span in line['spans']:
                    if block["text"] == "":
                        block["text"]+=span["text"]
                    else:
                        block["text"]+=" " + span["text"]
                    sizes.append(span['size'])
                    block["font"]=span['font']
            block["size"] = np.mean(sizes)
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
    import string,copy
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

    # データの分割
    bboxs = [item['coordinates'] for sublist in block_info for item in sublist]
    widths = [x1 - x0 for x0, _, x1, _ in bboxs]

    sizes = [item['size'] for sublist in block_info for item in sublist]

    text_list = [item['text'] for sublist in block_info for item in sublist]
    for i in range(len(text_list)):
        text_list[i] = text_list[i].replace("\n", "")
        text_list[i] = ''.join(char for char in text_list[i] if char not in string.punctuation and char not in string.digits)
    texts = [tokenize_text(lang,text) for text in text_list]
    texts = [len(text) for text in texts]

    # スコア値を換算
    scores = []

    for text in texts:
        if token_threshold <= text:
            scores.append([0])
        else:
            scores.append([1])
    
    for item in [widths,sizes]:
        # IQR:ロバストスケーリング
        item_median = median(item)
        item_75_percentile = np.percentile(item,75)
        item_25_percentile = np.percentile(item,25)
        
        for value,score_list in zip(item,scores):
            score = abs((value-item_median)/(item_75_percentile-item_25_percentile))
            score_list.append(score)
    marge_score = [] #3スコアの中央値を換算
    for list_score in scores:
        score_median = sum(list_score)
        marge_score.append(score_median)
    
    # ヒストグラムから基準値を算出
    # データのサンプルサイズ
    n = len(marge_score)
    # スタージェスの公式を使用してビンの数を計算
    num_bins_sturges = math.ceil(math.log2(n) + 1)
    # IQRを計算
    q75, q25 = np.percentile(marge_score, [75 ,25])
    iqr = q75 - q25
    # フリードマン＝ダイアコニスのルールに基づいてビン幅を計算
    bin_width_fd = 2 * iqr / n ** (1/3)
    # ビン幅を基にビンの数を計算
    bin_range = max(marge_score) - min(marge_score)
    num_bins_fd = math.ceil(bin_range / bin_width_fd)
    # ビンの数を決定（二つの方法のうち小さい方を選択）
    num_bins = min(num_bins_sturges, num_bins_fd)
    # ヒストグラムを計算
    histogram, bin_edges = np.histogram(marge_score, bins=num_bins)
    # 最も頻繁に現れるビンのインデックスを取得
    max_index = np.argmax(histogram)
    # 最も頻繁に現れる範囲を返す
    most_frequent_range = (bin_edges[max_index], bin_edges[max_index + 1])

    i = 0
    for pages in block_info:
        page_filtered_blocks = []
        page_fig_table_blocks = []
        page_removed_blocks = []

        for block in pages:

            #tokenを計算
            block_text = block["text"]
            tokens_list = tokenize_text(lang,block_text)

            # スコア値の取得
            score = marge_score[i]
            size = math.floor((sizes[i])*100)/100
            result=  bool(most_frequent_range[0]<=score<=most_frequent_range[1] and scores[i][0]==0)
            printscore = F"[{math.floor(score * 100) / 100}/{result}] /T:{math.floor((scores[i][0])*100)/100}({texts[i]})/W:{math.floor((scores[i][1])*100)/100}/S:{math.floor((scores[i][2])*100)/100}({size})"

            # tokenリスト３番目までに特定ワードが入ってる場合はグラフ表として認識する
            if lang == 'ja':
                keyword = ['表','グラフ']
            else:
                keyword = ['fig','table']
            table_bool = check_first_num_tokens(tokens_list,keyword)
            
            if table_bool:
                # 図データとしてリストに追加
                page_fig_table_blocks.append(block)
            elif most_frequent_range[0]<=score<=most_frequent_range[1] and scores[i][0]==0:
                # 本文データとしてリストに追加
                page_filtered_blocks.append(block)
            else:
                #データを除外
                swap_text = F"{printscore}"
                add_block = copy.copy(block)
                add_block["text"] = swap_text
                page_removed_blocks.append(add_block)
            i+=1

        fig_table_blocks.append(page_fig_table_blocks)
        filtered_blocks.append(page_filtered_blocks)
        removed_blocks.append(page_removed_blocks)
    
    if debug:
        # 解析用にデータを保存する
        size_median = median(sizes)
        size_mean = np.mean(sizes)

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
        token_mean = np.mean(tokens)

        token_Mean = plot_area_distribution(areas=tokens,labels_values=[{"Median":token_median},
                                                        {"Mean":token_mean}],title="token Mean",xlabel='Token',ylabel='Frequency')

        size_Mean = plot_area_distribution(areas=sizes,labels_values=[{"Median":size_median},
                                                        {"Mean":size_mean}],title="Size Mean",xlabel='font size',ylabel='Frequency')

        socre_Mean = plot_area_distribution(areas=marge_score,labels_values=[{"Histogram Low":most_frequent_range[0]},
                                                        {"Histogram High":most_frequent_range[1]}],title="score Mean",xlabel='score',ylabel='Frequency')
       
        return filtered_blocks,fig_table_blocks,removed_blocks,[token_Mean,size_Mean,socre_Mean]
    return filtered_blocks,fig_table_blocks,removed_blocks,None

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
                text_all = box["text"].replace(' ', '\u00A0') #スペースを改行されないノーブレークスペースに置き換え
                #text_all = box["text"]
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

async def write_pdf_text(input_pdf_data, block_info, to_lang='en',text_color=[0,0,0],font_path=None):
    """
    指定されたフォントで、文字を作画します。
    """
    lh_factor = 1.5  # 行の高さの係数

    # フォント選択
    if to_lang == 'en' and font_path == None:
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
            while True:
                rect = fitz.Rect(coordinates)
                result = page.insert_textbox(rect, text, fontsize=font_size, fontname="F0", align=3, lineheight = lh_factor, color = text_color)
                if result >=0:
                    break   
                else:
                    coordinates[3]+=1
        
    output_buffer = BytesIO()
    await asyncio.to_thread(doc.save, output_buffer, garbage=4, deflate=True, clean=True)
    await asyncio.to_thread(doc.close)
    output_data = output_buffer.getvalue()

    return output_data

async def write_image_data(input_pdf_data,image_data,rect=(10,10,200,200),position=-1,add_new_page=True):
    """
    新しいページを作成し、画像を挿入します。
    """
    from PIL import Image
    import io,os

    doc = await asyncio.to_thread(fitz.open, stream=input_pdf_data, filetype="pdf")

    # 最初のページの寸法を取得
    first_page = doc[0]  # 最初のページを取得
    rect = first_page.rect  # 最初のページの寸法を取得

    # ページの追加
    if add_new_page:
        doc.insert_page(position, width=rect.width, height=rect.height)

    page = doc[position]
    image_byte = Image.open(io.BytesIO(image_data))
    temp_image_path = "temp_image.png"
    image_byte.save(temp_image_path)
    page.insert_image(rect,filename=temp_image_path)

    # 保存と終了処理
    output_buffer = BytesIO()
    await asyncio.to_thread(doc.save, output_buffer, garbage=4, deflate=True, clean=True)

    # 一時ファイルの削除
    os.remove(temp_image_path)  # temp_image.pngを削除
    #ドキュメントのクローズ
    await asyncio.to_thread(doc.close)
    
    # PDFデータをバイトとして返す
    output_data = output_buffer.getvalue()
    return output_data

async def write_logo_data(input_pdf_data):
    """
    PDFにサービスロゴを描画します
    """
    doc = await asyncio.to_thread(fitz.open, stream=input_pdf_data, filetype="pdf")
    rect = (5,5,35,35)
    logo_path = "./data/indqx_qr.png"
    font_path = 'fonts/TIMES.TTF'
    for page in doc:
        page.insert_font(fontname="F0", fontfile=font_path)
        page.insert_image(rect,filename=logo_path)
        page.insert_textbox((37,5,100,35),"Translated by.",fontsize=5,fontname="F0")
        page.insert_textbox((37, 12, 100, 35), "IndQx", fontsize=10, fontname="F0")
        page.insert_textbox((37,25,100,35),"Translation.",fontsize=5,fontname="F0")

    output_buffer = BytesIO()
    await asyncio.to_thread(doc.save, output_buffer, garbage=4, deflate=True, clean=True)
    await asyncio.to_thread(doc.close)
    output_data = output_buffer.getvalue()

    return output_data

async def create_viewing_pdf(base_pdf_path, translated_pdf_path):
    # PDFドキュメントを開く
    doc_base = await asyncio.to_thread(fitz.open, stream=base_pdf_path, filetype="pdf")
    doc_translate = await asyncio.to_thread(fitz.open, stream=translated_pdf_path, filetype="pdf")

    # 新しいPDFドキュメントを作成
    new_doc = fitz.open()

    # 各ページをループ処理
    for page_num in range(len(doc_base)):
        # base_pdfとtranslated_pdfからページを取得
        page_base = doc_base.load_page(page_num)
        page_translate = doc_translate.load_page(page_num)

        # 新しい見開きページを作成
        # ページサイズはそれぞれのPDFの1ページの幅と高さを使う
        rect_base = page_base.rect
        rect_translate = page_translate.rect
        
        # 両ページの高さが異なる場合、高い方に合わせる
        max_height = max(rect_base.height, rect_translate.height)

        # base_pdfのページを左ページに追加
        new_page = new_doc.new_page(width=rect_base.width, height=max_height)
        new_page.show_pdf_page(new_page.rect, doc_base, page_num)
        
        # translated_pdfのページを右ページに追加
        new_page = new_doc.new_page(width=rect_translate.width, height=max_height)
        new_page.show_pdf_page(new_page.rect, doc_translate, page_num)
    
    # ページレイアウトを見開きに設定
    new_doc.set_pagelayout("TwoPageLeft")

    # 新しいPDFファイルを保存
    output_buffer = BytesIO()
    await asyncio.to_thread(new_doc.save, output_buffer, garbage=4, deflate=True, clean=True)
    await asyncio.to_thread(new_doc.close)
    await asyncio.to_thread(doc_base.close)
    await asyncio.to_thread(doc_translate.close)
    output_data = output_buffer.getvalue()
    return output_data

def plot_area_distribution(areas, labels_values, title='Distribution of Areas', xlabel='Area', ylabel='Frequency'):
    """
    デバッグ用、グラフを作画し画像データを返します
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

    # PNGデータとしてグラフをメモリ上に保存
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  # リソースを解放
    buf.seek(0)  # バッファの先頭にシーク
    return buf.read()  # バイトデータとして読み出し
