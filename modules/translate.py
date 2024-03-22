import aiohttp
import os
import re
import fitz  # PyMuPDF
import asyncio
from io import BytesIO
from statistics import median

async def translate_text(text: str, target_lang: str) -> str:
    """
    DeepL APIを使用して、入力されたテキストを指定の言語に翻訳する非同期関数。
    タグハンドリングがXML向けになっているので注意

    Args:
        text (str): 翻訳するテキスト。
        target_lang (str): 翻訳先の言語コード（例: "EN", "JA", "FR"など）。

    Returns:
        str: 翻訳されたテキスト。

    Raises:
        Exception: APIリクエストが失敗した場合。
    """
    api_key = os.environ["DEEPL_API_KEY"]  # 環境変数からDeepL APIキーを取得
    api_url = "https://api-free.deepl.com/v2/translate"

    params = {
        "auth_key": api_key,
        "text": text,
        "target_lang": target_lang,
        'tag_handling': 'xml',
        "split_sentences": "nonewlines",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, data=params) as response:
            if response.status == 200:
                result = await response.json()
                return {'ok':True,'data':result["translations"][0]["text"]}
            else:
                return {'ok':False,'message':f"DeepL API request failed with status code {response.status}"}
            
def calculate_translation_cost(text: str, price_per_character: float) -> float:
    """
    翻訳コストを計算する関数。

    Args:
        text (str): 翻訳するテキスト。
        price_per_character (float): 1文字あたりの料金。

    Returns:
        float: 翻訳コスト。
    """
    character_count = len(text)
    translation_cost = character_count * price_per_character
    return translation_cost

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

def plot_area_distribution(areas, labels_values, title='Distribution of Areas', xlabel='Area', ylabel='Frequency', save_path=None):
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
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

async def remove_blocks_with_few_words(block_info, word_threshold=10):
    import string
    import numpy as np
    """
    英単語数が指定された閾値以下のブロックをリストから削除します。更にブロックの面積中央値を求め中央値以下のブロックもリストから消去します。
    削除されたブロックも返します。

    :param block_info: ブロック情報のリスト
    :param word_threshold: 単語の数の閾値（この数以下の場合、ブロックを削除）
    :return: 更新されたブロック情報のリストと削除されたブロック情報のリスト
    """
    #フィルターに基づいて分離する。
    filtered_blocks = []
    removed_blocks = []

    # boxデータの分割
    bboxs = [item['coordinates'] for sublist in block_info for item in sublist]

    # ブロック幅のしきい値を求める
    widths = [x1 - x0 for x0, _, x1, _ in bboxs]
    width_median = median(widths)
    width_percentile_90 = np.percentile(widths, 90)
    width_percentile_75 = np.percentile(widths, 75)
    width_percentile_80 = np.percentile(widths, 80)
    mean_width = np.mean(widths)
    for i in range(100,-1,-25):
        percentile = np.percentile(widths,i)
        if percentile < 300:
            break
    width_threshold_low = 0.9 * percentile
    width_threshold_high = 1.1 * percentile

    # 文字数のしきい値を求める
    texts = [item['text'] for sublist in block_info for item in sublist]
    text_length_list = [len(s) for s in texts]

    # 10以下の数字をすべて除去
    text_length_list = [num for num in text_length_list if num > 10]
    texts_median = median(text_length_list)
    texts_percentile_90 = np.percentile(text_length_list, 90)
    texts_percentile_75 = np.percentile(text_length_list, 75)
    texts_percentile_80 = np.percentile(text_length_list, 80)
    texts_mean = np.mean(text_length_list)
    text_threshold_low = texts_median *0.9
    text_threshold_high = texts_median * 1.1

    for pages in block_info:
        page_filtered_blocks = []
        page_removed_blocks = []
        for block in pages:
            #widthを計算
            width = (block_coordinates[2] - block_coordinates[0])
            #記号と数字が50%を超える場合は、リストから消去
            no_many_symbol = True
            symbol_and_digit_count = sum(1 for char in block_text if char in string.punctuation or char in string.digits)
            if len(block_text)!=0:
                no_many_symbol = symbol_and_digit_count / len(block_text) < 0.5
            

    save_data = []
    for pages in block_info:
        page_filtered_blocks = []
        page_removed_blocks = []
        for block in pages:
            block_text = block["text"].strip()
            
            #面積を換算
            block_coordinates = block['coordinates']
            #block_area = (block_coordinates[2] - block_coordinates[0]) * (block_coordinates[3] - block_coordinates[1])
            #幅を換算
            width = (block_coordinates[2] - block_coordinates[0])

            #記号と数字が50%を超える場合は、リストから消去
            no_many_symbol = True
            symbol_and_digit_count = sum(1 for char in block_text if char in string.punctuation or char in string.digits)
            if len(block_text)!=0:
                no_many_symbol = symbol_and_digit_count / len(block_text) < 0.5

            text_count_bool = bool(len(block_text.split()) > word_threshold)
            width_bool = bool(width_threshold_high > width > width_threshold_low)
            text_mean_bool = bool(text_threshold_high > len(block_text) > text_threshold_low)
            no_symbol_bool = bool(no_many_symbol)

            save_data.append({"Texts": block_text,
                              "result bool": text_count_bool and no_symbol_bool and width_bool,
                                "width bool": width_bool,
                                "width_threshold_low": float(width_threshold_low),
                                "width_threshold_high": float(width_threshold_high),
                                "text_mean_bool": text_mean_bool,
                                "text threshold low": float(text_threshold_low),
                                "text count": float(len(block_text)),
                                "text_threshold high": float(text_threshold_high),
                                "this with": float(width),
                                "文字数":len(block_text),
                                "単語数Bool": text_count_bool,
                                "シンボルBool": no_symbol_bool})
            
            if text_count_bool and no_symbol_bool and width_bool and text_mean_bool:
                page_filtered_blocks.append(block)
            else:
                page_removed_blocks.append(block)
        filtered_blocks.append(page_filtered_blocks)
        removed_blocks.append(page_removed_blocks)
    
    import json
    with open('save_string.json', 'w', encoding='utf-8') as json_file:
        json.dump(save_data, json_file, ensure_ascii=False, indent=2)
    
    
    # 解析用にデータを保存する
    plot_area_distribution(areas=widths,labels_values=[{"Median":width_median},
                                                       {"threshold_low":width_threshold_low},
                                                       {"threshold_high":width_threshold_high},
                                                       {"Mean":mean_width},
                                                       {"percentile_75":width_percentile_75},
                                                       {"percentile_80":width_percentile_80},
                                                       {"percentile_90":width_percentile_90}],title="Awidth Mean",xlabel='width size',ylabel='Frequency',save_path='grah_With.png')
    
    plot_area_distribution(areas=text_length_list,labels_values=[{"Median":texts_median},
                                                       {"Mean":texts_mean},
                                                       {"percentile_75":texts_percentile_75},
                                                       {"percentile_80":texts_percentile_80},
                                                       {"percentile_90":texts_percentile_90}],title="Texts Mean",xlabel='texts_num',ylabel='Frequency',save_path='grah_texts.png')

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

async def write_pdf_text(input_pdf_data, block_info, font_path='fonts/TIMES.TTF', lang='en', debug=False):
    doc = await asyncio.to_thread(fitz.open, stream=input_pdf_data, filetype="pdf")

    font = await asyncio.to_thread(fitz.Font, fontfile=font_path)

    for i, pages in enumerate(block_info):
        page = doc[i]
        await asyncio.to_thread(page.insert_font, fontfile=font_path, fontname="F0")

        for block in pages:
            text = block["text"]
            x0, y0, x1, y1 = block["coordinates"]
            text_rect = fitz.Rect(x0, y0, x1, y1)

            # フォントサイズと行の高さの計算
            fs = int(y1-y0)  # 初期フォントサイズ
            lh_factor = 1.2 # 行の高さの係数

            if lang == 'en':
                lang_a = 'a'
            elif lang == 'ja':
                lang_a = 'あ'

            while fs > 0:
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
                    break
                else:
                    fs -= 0.5

            while fs > 0:
                result = await asyncio.to_thread(page.insert_textbox, text_rect, text_no_newlines, fontsize=fs, fontname="F0", lineheight=lh_factor)
                if result >= 0:
                    break
                else:
                    fs -= 0.5

            if fs > 0:  # 適切なフォントサイズが見つかった場合
                # テキストをページに追加
                result = await asyncio.to_thread(page.insert_textbox, text_rect, text_no_newlines, fontsize=fs, fontname="F0", lineheight=lh_factor)

    output_buffer = BytesIO()
    await asyncio.to_thread(doc.save, output_buffer)
    await asyncio.to_thread(doc.close)

    output_data = output_buffer.getvalue()
    return output_data

async def translate_document(document_content):
    # 翻訳後のページごとのテキストを格納するリスト
    cost = 0

    # XMLに変換
    xml_data,cost = await convert_to_xml(document_content)

    import aiofiles
    async with aiofiles.open('output.xml', 'w', encoding='utf-8') as file:
        await file.write(xml_data)
    
    translate_xml = await translate(xml_data)
    async with aiofiles.open('output_translate.xml', 'w', encoding='utf-8') as file:
        await file.write(translate_xml)
    restored_json_data = await convert_from_xml(document_content,translate_xml)

    import json
    with open('block_info_translated.json', 'w', encoding='utf-8') as json_file:
        json.dump(restored_json_data, json_file, ensure_ascii=False, indent=2)

    return restored_json_data, cost

async def translate(text):
    result = await translate_text(text, "JA")
    if result['ok']:
        return result['data']
    else:
        raise Exception(result['message'])
    
async def convert_to_xml(json_data):
    cost =0
    price_per_character = 0.0025  # 1文字あたりの料金(円)
    xml_output = ""
    for page in json_data:
        for block in page:
            text = block['text'].replace('\n', '') #改行キーの消去
            xml_output += f"<div>{text}</div>\n"
            temp_cost = calculate_translation_cost(text,price_per_character)
            cost += temp_cost
    return xml_output,cost

async def convert_from_xml(original_json_data, xml_data):
    # <div>タグの内容を抽出する正規表現パターン
    pattern = re.compile(r'<div>(.*?)</div>', re.DOTALL)

    # <div>タグの内容を抽出
    blocks = pattern.findall(xml_data)

    block_index = 0
    for page in original_json_data:
        for block in page:
            if block_index < len(blocks):
                block['text'] = blocks[block_index]
                block_index += 1

    return original_json_data

async def deepl_translate_test():
    """
    DeepL テスト用コード
    """
    text = "Hello, how are you?"
    target_lang = "JA"
    price_per_character = 0.0025  # 1文字あたりの料金(円)

    translated = await translate_text(text, target_lang)
    print(translated)

    translation_cost = calculate_translation_cost(text, price_per_character)
    print(f"Translation cost: {translation_cost:.5f}円")


async def pdf_translate_test():
    import json
    to_lang = 'ja'
    # フォント選択
    if to_lang == 'en':
        font_path = 'fonts/TIMES.TTF'
    elif to_lang == 'ja':
        font_path = 'fonts/ipam.ttf'

    with open("input.pdf", "rb") as f:
        input_pdf_data = f.read()

    block_info = await extract_text_coordinates(input_pdf_data)

    blocked_pdf_data = await pdf_draw_blocks(input_pdf_data,block_info,width=0,fill_opacity=0.3)

    block_info,removed_blocks = await remove_blocks_with_few_words(block_info,10)

    # removed_blockをリストに分解
    leave_str_list = [item['text'] for sublist in removed_blocks for item in sublist]
    translate_blocked_pdf_data = await pdf_draw_blocks(input_pdf_data,block_info,width=0,fill_opacity=0.3)
    removed_textbox_pdf_data = await remove_textbox_for_pdf(input_pdf_data,leave_str_list)
    
    with open("draw_block.pdf", "wb") as f:
        f.write(blocked_pdf_data)
    with open("draw_block_translation.pdf", "wb") as f:
        f.write(translate_blocked_pdf_data)
    with open('block_info.json', 'w', encoding='utf-8') as json_file:
        json.dump(block_info, json_file, ensure_ascii=False, indent=2)
    exit()
    block_info,cost = await translate_document(block_info)
    print(F"翻訳コスト： {cost}円")
    translated_pdf_data = await write_pdf_text(removed_textbox_pdf_data,block_info,font_path,to_lang)

    with open("output.pdf", "wb") as f:
        f.write(translated_pdf_data)

if __name__ == "__main__":
    asyncio.run(pdf_translate_test())