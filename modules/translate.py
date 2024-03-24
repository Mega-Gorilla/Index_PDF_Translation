import aiohttp
import os
import re
import asyncio
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
    api_url = "https://api.deepl.com/v2/translate"

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
                result = result["translations"][0]["text"]
                return {'ok':True,'data':result}
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
    async with aiofiles.open('output_translate.xml', 'w', encoding='utf-8') as file:
        await file.write(translate_xml)
    """
    restored_json_data = await convert_from_xml(document_content,translate_xml)

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
            text = block['text']
            # 翻訳にて問題になる文字列を変換
            text = text.replace('\n', '')

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
    source_lang = 'en'
    to_lang = 'ja'
    debug =True

    with open("input.pdf", "rb") as f:
        input_pdf_data = f.read()
    block_info = await extract_text_coordinates(input_pdf_data,source_lang)

    text_blocks,fig_blocks,removed_blocks = await remove_blocks(block_info,10)

    # removed_blockをリストに分解
    leave_str_list = [item['text'] for sublist in removed_blocks for item in sublist]
    removed_textbox_pdf_data = await remove_textbox_for_pdf(input_pdf_data,leave_str_list)

    if debug:
        """
        with open('all_blocks.json', 'w', encoding='utf-8') as json_file:
            json.dump(block_info, json_file, ensure_ascii=False, indent=2)
        with open('text_block.json', 'w', encoding='utf-8') as json_file:
            json.dump(text_blocks, json_file, ensure_ascii=False, indent=2)
        with open('fig_blocks.json', 'w', encoding='utf-8') as json_file:
            json.dump(fig_blocks, json_file, ensure_ascii=False, indent=2)
        """
        text_block_pdf_data = await pdf_draw_blocks(input_pdf_data,text_blocks,width=0,fill_opacity=0.3,fill_colorRGB=[0,0,1])
        fig_block_pdf_data = await pdf_draw_blocks(text_block_pdf_data,fig_blocks,width=0,fill_opacity=0.3,fill_colorRGB=[0,1,0])
        all_block_pdf_data = await pdf_draw_blocks(fig_block_pdf_data,removed_blocks,width=0,fill_opacity=0.3,fill_colorRGB=[1,0,0])
        with open("show_blocks.pdf", "wb") as f:
            f.write(all_block_pdf_data)

    # 翻訳
    text_blocks,text_cost = await translate_document(text_blocks)
    fig_blocks,fig_cost = await translate_document(fig_blocks)
    cost = text_cost + fig_cost
    print(F"翻訳コスト： {cost}円")
    
    # 翻訳したブロックを結合
    combined_blocks =[]
    for page1, page2 in zip(text_blocks,fig_blocks):
        combined_page = page1 + page2
        sorted_page = sorted(combined_page, key=lambda x: x['block_no'])
        combined_blocks.append(sorted_page)
    
    # 翻訳したPDFを作成
    translated_pdf_data = await write_pdf_text(removed_textbox_pdf_data,combined_blocks,to_lang)

    with open("output.pdf", "wb") as f:
        f.write(translated_pdf_data)

if __name__ == "__main__":
    asyncio.run(pdf_translate_test())