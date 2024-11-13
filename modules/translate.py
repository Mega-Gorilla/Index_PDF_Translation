import asyncio

import aiohttp
from tenacity import retry, stop_after_attempt, wait_fixed

from config import *
from modules.pdf_edit import *
from modules.translate_llm import translate_str_data_with_llm


async def translate_str_data(
    key: str, text: str, target_lang: str, api_url: str
) -> str:
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
        "auth_key": api_key,  # DeepLの認証キー
        "text": text,  # 翻訳するテキスト
        "target_lang": target_lang,  # 目的の言語コード
        "tag_handling": "xml",  # タグの扱い
        "formality": "more",  # 丁寧な口調で翻訳
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, data=params) as response:
            if response.status == 200:
                result = await response.json()
                result = result["translations"][0]["text"]
                # print(F"Translate :{result}")
                return {"ok": True, "data": result}
            else:
                return {
                    "ok": False,
                    "message": f"DeepL API request failed with status code {response.status}",
                }


async def translate_blocks(blocks, key: str, target_lang: str, api_url: str):
    @retry(wait=wait_fixed(2), stop=stop_after_attempt(4))
    async def translate_block(block) -> dict:
        """
        ブロックのテキストを翻訳する非同期関数
        ブロックとは、text, coordinates, block_no, page_no, sizeのキーを持つ辞書を指す
        """
        text = block["text"]
        if text.strip() == "":
            return block

        translated_text = await translate_str_data_with_llm(text, target_lang)
        if translated_text["ok"]:
            block["text"] = translated_text["data"]
            return block
        else:
            raise Exception(translated_text["message"])

    def print_progress(*args):
        """
        翻訳進捗状況表示
        """
        print(".", end="", flush=True)

    tasks = []
    try:
        async with asyncio.TaskGroup() as tg:
            for block_idx, page in enumerate(blocks):
                for page_idx, block in enumerate(page):
                    task = tg.create_task(translate_block(block))
                    task.add_done_callback(print_progress)
                    tasks.append(((block_idx, page_idx), task))
            print(f"  generated {len(tasks)} tasks")
            print("  waiting for complete...")
            # show progress while waiting for completion
    except* Exception as e:
        print(f"{e.exceptions=}")
        raise e

    print("  completed all tasks")

    for (block_idx, page_idx), task in tasks:
        blocks[block_idx][page_idx] = task.result()

    return blocks


async def preprocess_translation_blocks(
    blocks, end_maker=(".", ":", ";"), end_maker_enable=True
):
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
            text += " " + block["text"]
            page_no.append(block["page_no"])
            coordinates.append(block["coordinates"])
            block_no.append(block["block_no"])
            font_size.append(block["size"])

            if (
                text.endswith(end_maker)
                or block["block_no"] - temp_block_no <= 1
                or end_maker_enable == False
            ):
                # マーカーがある場合格納
                page_results.append(
                    {
                        "page_no": page_no,
                        "block_no": block_no,
                        "coordinates": coordinates,
                        "text": text,
                        "size": font_size,
                    }
                )
                text = ""
                coordinates = []
                block_no = []
                page_no = []
                font_size = []
            temp_block_no = block["block_no"]

        results.append(page_results)
    return results


async def deepl_convert_xml_calc_cost(json_data):
    """
    翻訳コストを算出します。
    """
    cost = 0
    price_per_character = 0.0025  # 1文字あたりの料金(円)
    xml_output = ""
    for page in json_data:
        for block in page:
            text = block["text"]
            # 翻訳にて問題になる文字列を変換
            # text = text.replace('\n', '')

            xml_output += f"<div>{text}</div>\n"
    return xml_output, cost


async def pdf_translate(
    key,
    pdf_data,
    source_lang="en",
    to_lang="ja",
    api_url="https://api.deepl.com/v2/translate",
    debug=False,
    disable_translate=False,
):
    block_info = await extract_text_coordinates_dict(pdf_data)

    text_blocks, fig_blocks, _excluded_blocks = await remove_blocks(
        block_info, 10, lang=source_lang
    )
    # breakpoint()
    # t = lambda x: '\n'.join([ee.get('text') for e in x for ee in e])

    # 翻訳部分を消去したPDFデータを制作
    removed_textbox_pdf_data = await remove_textbox_for_pdf(pdf_data, text_blocks)
    removed_textbox_pdf_data = await remove_textbox_for_pdf(
        removed_textbox_pdf_data, fig_blocks
    )
    print("1.Generate removed_textbox_pdf_data")

    # 翻訳前のブロック準備
    preprocess_text_blocks = await preprocess_translation_blocks(
        text_blocks, (".", ":", ";"), True
    )
    preprocess_fig_blocks = await preprocess_translation_blocks(
        fig_blocks, (".", ":", ";"), False
    )
    print("2.Generate Prepress_blocks")

    # 翻訳実施
    translated_pdf_data = None
    if disable_translate is False:
        translate_text_blocks = await translate_blocks(
            preprocess_text_blocks, key, to_lang, api_url
        )
        translate_fig_blocks = await translate_blocks(
            preprocess_fig_blocks, key, to_lang, api_url
        )
        print("3.translated blocks")
        # pdf書き込みデータ作成
        write_text_blocks = await preprocess_write_blocks(
            translate_text_blocks, to_lang
        )
        write_fig_blocks = await preprocess_write_blocks(translate_fig_blocks, to_lang)
        print("4.Generate wirte Blocks")
        # pdfの作成
        if write_text_blocks != []:
            print("write text to pdf.")
            print(len(write_text_blocks))
            translated_pdf_data = await write_pdf_text(
                removed_textbox_pdf_data, write_text_blocks, to_lang
            )
        else:
            print("write text to pdf is empty.")
            breakpoint()

        if write_fig_blocks != []:
            print("write fig to pdf.")
            translated_pdf_data = await write_pdf_text(
                translated_pdf_data, write_fig_blocks, to_lang
            )
        else:
            print("write fig to pdf is empty.")
            breakpoint()

    else:
        print("99.Translate is False")

    # 見開き結合の実施
    marged_pdf_data = await create_viewing_pdf(pdf_data, translated_pdf_data)
    print("5.Generate PDF Data")
    return marged_pdf_data


async def PDF_block_check(pdf_data, source_lang="en"):
    """
    ブロックの枠を作画します
    """

    block_info = await extract_text_coordinates_dict(pdf_data)

    text_blocks, fig_blocks, leave_blocks = await remove_blocks(
        block_info, 10, lang=source_lang
    )

    text_block_pdf_data = await pdf_draw_blocks(
        pdf_data, text_blocks, width=0, fill_opacity=0.3, fill_colorRGB=[0, 0, 1]
    )
    fig_block_pdf_data = await pdf_draw_blocks(
        text_block_pdf_data,
        fig_blocks,
        width=0,
        fill_opacity=0.3,
        fill_colorRGB=[0, 1, 0],
    )
    all_block_pdf_data = await pdf_draw_blocks(
        fig_block_pdf_data,
        leave_blocks,
        width=0,
        fill_opacity=0.3,
        fill_colorRGB=[1, 0, 0],
    )

    return all_block_pdf_data
