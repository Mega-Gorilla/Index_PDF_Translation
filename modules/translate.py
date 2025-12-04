# SPDX-License-Identifier: AGPL-3.0-only
import aiohttp
import asyncio
from modules.pdf_edit import *

async def translate_str_data(key: str,text: str, target_lang: str,api_url:str) -> str:
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
    api_key = key  # 環境変数からDeepL APIキーを取得

    params = {
        "auth_key": api_key,           # DeepLの認証キー
        "text": text,                  # 翻訳するテキスト
        "target_lang": target_lang,    # 目的の言語コード
        'tag_handling': 'xml',         # タグの扱い
        "formality": "more"            # 丁寧な口調で翻訳
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, data=params) as response:
            if response.status == 200:
                result = await response.json()
                result = result["translations"][0]["text"]
                #print(F"Translate :{result}")
                return {'ok':True,'data':result}
            else:
                return {'ok':False,'message':f"DeepL API request failed with status code {response.status}"}

async def translate_blocks(blocks,key: str, target_lang: str,api_url:str):
    # テキスト検出
    translate_text = ""
    for page in blocks:
        for block in page:
            translate_text += block["text"] + "\n"
    
    # 翻訳
    translated_text = await translate_str_data(key,translate_text,target_lang,api_url)

    if translated_text['ok']:
        translated_text = translated_text['data']
    else:
        raise Exception(translated_text['message'])
    translated_text = translated_text.split('\n')
    
    # 翻訳後テキスト挿入
    for page in blocks:
        for block in page:
            block["text"] = translated_text.pop(0)

    return blocks

async def preprocess_translation_blocks(blocks,end_maker=(".",":",";"),end_maker_enable=True):
    """
    blockの文字列について、end makerがない場合、一括で翻訳できるように変換します。
    変換結果のblockを返します
    """
    results = []

    text = ""
    coordinates = []
    block_no = []
    page_no = []
    font_size = []

    for page in blocks:
        page_results = []
        temp_block_no = 0
        for block in page:
            text += " "+block["text"]
            page_no.append(block["page_no"])
            coordinates.append(block["coordinates"])
            block_no.append(block["block_no"])
            font_size.append(block["size"])

            if text.endswith(end_maker) or block["block_no"] - temp_block_no <= 1 or end_maker_enable == False:
                #マーカーがある場合格納
                page_results.append({"page_no":page_no,
                                     "block_no":block_no,
                                     "coordinates":coordinates,
                                     "text":text,
                                     "size":font_size})
                text = ""
                coordinates = []
                block_no = []
                page_no = []
                font_size = []
            temp_block_no = block["block_no"]
                
        results.append(page_results)
    return results


async def pdf_translate(key, pdf_data, source_lang='en', to_lang='ja',
                        api_url="https://api.deepl.com/v2/translate",
                        debug=False, disable_translate=False, add_logo=True):
    """
    PDFを翻訳し、見開きPDF（オリジナル + 翻訳）を生成します。

    Args:
        key: DeepL APIキー
        pdf_data: 入力PDFのバイナリデータ
        source_lang: 翻訳元言語コード (default: 'en')
        to_lang: 翻訳先言語コード (default: 'ja')
        api_url: DeepL API URL
        debug: デバッグモード（可視化PDF生成）
        disable_translate: 翻訳を無効化（テスト用）
        add_logo: ロゴウォーターマークを追加 (default: True)

    Returns:
        見開きPDFのバイナリデータ
    """
    block_info = await extract_text_coordinates_dict(pdf_data)

    if debug:
        text_blocks, fig_blocks, remove_info, plot_images = await remove_blocks(
            block_info, 10, lang=source_lang, debug=True
        )
    else:
        text_blocks, fig_blocks, _, _ = await remove_blocks(block_info, 10, lang=source_lang)

    # 翻訳部分を消去したPDFデータを制作
    removed_textbox_pdf_data = await remove_textbox_for_pdf(pdf_data, text_blocks)
    removed_textbox_pdf_data = await remove_textbox_for_pdf(removed_textbox_pdf_data, fig_blocks)
    print("1.Generate removed_textbox_pdf_data")

    # 翻訳前のブロック準備
    preprocess_text_blocks = await preprocess_translation_blocks(text_blocks, (".", ":", ";"), True)
    preprocess_fig_blocks = await preprocess_translation_blocks(fig_blocks, (".", ":", ";"), False)
    print("2.Generate Prepress_blocks")

    # 翻訳実施
    if disable_translate is False:
        translate_text_blocks = await translate_blocks(preprocess_text_blocks, key, to_lang, api_url)
        translate_fig_blocks = await translate_blocks(preprocess_fig_blocks, key, to_lang, api_url)
        print("3.translated blocks")

        # pdf書き込みデータ作成
        write_text_blocks = await preprocess_write_blocks(translate_text_blocks, to_lang)
        write_fig_blocks = await preprocess_write_blocks(translate_fig_blocks, to_lang)
        print("4.Generate wirte Blocks")

        # pdfの作成
        if write_text_blocks != []:
            translated_pdf_data = await write_pdf_text(removed_textbox_pdf_data, write_text_blocks, to_lang)
        if write_fig_blocks != []:
            translated_pdf_data = await write_pdf_text(translated_pdf_data, write_fig_blocks, to_lang)

        # ロゴ追加（オプション）
        if add_logo:
            translated_pdf_data = await write_logo_data(translated_pdf_data)
    else:
        print("99.Translate is False")

    # 見開き結合の実施
    marged_pdf_data = await create_viewing_pdf(pdf_data, translated_pdf_data)
    print("5.Generate PDF Data")
    return marged_pdf_data
