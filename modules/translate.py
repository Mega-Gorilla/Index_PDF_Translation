import aiohttp
import os
import re
import fitz  # PyMuPDF
import asyncio
from io import BytesIO
from statistics import median
from spacy_api import *
from pdf_edit import *

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
        "auth_key": api_key,           # DeepLの認証キー
        "text": text,                  # 翻訳するテキスト
        "target_lang": target_lang,    # 目的の言語コード
        'tag_handling': 'xml',         # タグの扱い
        "split_sentences": "nonewlines", # 文章の分割方法
        "formality": "more"            # 丁寧な口調で翻訳
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


async def translate_document(document_content,lang='ja'):
    # 翻訳後のページごとのテキストを格納するリスト
    cost = 0

    # XMLに変換
    xml_data,cost = await deepl_convert_xml_calc_cost(document_content)
    
    translate_xml = await translate(xml_data,lang)
    """
    import aiofiles
    async with aiofiles.open('output.xml', 'w', encoding='utf-8') as file:
        await file.write(xml_data)
    async with aiofiles.open('output_translate.xml', 'w', encoding='utf-8') as file:
        await file.write(translate_xml)
    """
    restored_json_data = await convert_from_xml(document_content,translate_xml)

    import json
    with open('block_info_translated.json', 'w', encoding='utf-8') as json_file:
        json.dump(restored_json_data, json_file, ensure_ascii=False, indent=2)

    return restored_json_data, cost

async def translate(text,lang):
    result = await translate_text(text, lang)
    if result['ok']:
        return result['data']
    else:
        raise Exception(result['message'])
    
async def deepl_convert_xml_calc_cost(json_data):
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

# ---------------　以下テストコード(不要)------------------------
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
    block_info,cost = await translate_document(block_info)
    print(F"翻訳コスト： {cost}円")
    translated_pdf_data = await write_pdf_text(removed_textbox_pdf_data,block_info,to_lang)

    with open("output.pdf", "wb") as f:
        f.write(translated_pdf_data)

if __name__ == "__main__":
    asyncio.run(pdf_translate_test())