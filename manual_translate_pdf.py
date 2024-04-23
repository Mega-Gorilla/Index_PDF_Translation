from modules.translate import pdf_translate,pdf_draw_dev
import os, asyncio
import tkinter as tk
from tkinter import filedialog
from config import *

def load_json_to_list(file_path):
    import json
    """ 指定されたJSONファイルを読み込み、Pythonのリストとして返す関数 """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"指定されたファイルが見つかりません: {file_path}")
        return None
    except json.JSONDecodeError:
        print("JSONファイルの形式が正しくありません。")
        return None
    except Exception as e:
        print(f"ファイルの読み込み中にエラーが発生しました: {e}")
        return None

async def translate_test():
    # GUIでファイル選択のための設定
    root = tk.Tk()
    root.withdraw()  # GUIのメインウィンドウを表示しない
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])  # PDFファイルのみ選択

    if not file_path:
        print("ファイルが選択されませんでした。")
        return

    with open(file_path, "rb") as f:
        input_pdf_data = f.read()

    result_pdf = await pdf_translate(os.environ["DEEPL_API_KEY"], input_pdf_data,debug=True)

    _, file_name = os.path.split(file_path)
    output_path = Debug_folder_path + "result_"+file_name

    with open(output_path, "wb") as f:
        f.write(result_pdf)

async def generate_pdf_test():
    # 翻訳済みリストを読み込み
    translated_text_blocks = load_json_to_list(Debug_folder_path + "translate_text_blocks.json")
    translated_fig_blocks = load_json_to_list(Debug_folder_path + "translate_fig_blocks.json")
    # PDFの読み込み 
    with open(Debug_folder_path + "removed_pdf.pdf", "rb") as f:
        input_pdf_data = f.read()
    result_pdf = await pdf_draw_dev(input_pdf_data,translated_text_blocks,translated_fig_blocks)

    output_path = Debug_folder_path + "result_draw_only.pdf"

    with open(output_path, "wb") as f:
        f.write(result_pdf)

if __name__ == "__main__":
    #asyncio.run(translate_test())
    asyncio.run(generate_pdf_test())
