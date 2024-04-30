from modules.translate import pdf_translate,PDF_block_check,write_logo_data
import os, asyncio,time
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
    
    process_time = time.time()
    with open(file_path, "rb") as f:
        input_pdf_data = f.read()

    result_pdf = await pdf_translate(os.environ["DEEPL_API_KEY"], input_pdf_data,debug=True)

    if result_pdf is None:
        return
    _, file_name = os.path.split(file_path)
    output_path = Debug_folder_path + "result_"+file_name

    with open(output_path, "wb") as f:
        f.write(result_pdf)
    print(F"Time:{time.time()-process_time}")

async def test_bench():
    original_directory = os.getcwd()
    directory = ".\Test Bench\\raw"
    import glob
    # カレントディレクトリに変更する場合
    os.chdir(directory)
    # PDFファイルのフルパスを取得
    pdf_files = glob.glob('**/*.pdf', recursive=True)
    # ディレクトリをフルパスで取得するには、以下のように結合する
    pdf_files = [os.path.join(directory, file) for file in pdf_files]
    pdf_files = glob.glob('**/*.pdf', recursive=True)

    # ディレクトリー移動
    os.chdir(original_directory)

    for file_path in pdf_files:
        file_path = directory + "\\"+ file_path
        with open(file_path, "rb") as f:
            input_pdf_data = f.read()
        print(F"Loaded: {file_path}")

        result_pdf = await pdf_translate(os.environ["DEEPL_API_KEY"], input_pdf_data,debug=True)

        if result_pdf is None:
            continue
        _, file_name = os.path.split(file_path)
        output_path = bach_process_path + "result_"+file_name

        with open(output_path, "wb") as f:
            f.write(result_pdf)
        print(F"Saved: {output_path}")

async def pdf_block_test():
    # GUIでファイル選択のための設定
    root = tk.Tk()
    root.withdraw()  # GUIのメインウィンドウを表示しない
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])  # PDFファイルのみ選択

    if not file_path:
        print("ファイルが選択されませんでした。")
        return
    
    with open(file_path, "rb") as f:
        input_pdf_data = f.read()
    result_pdf = await PDF_block_check(input_pdf_data)    

    if result_pdf is None:
        return
    _, file_name = os.path.split(file_path)
    output_path = Debug_folder_path + "Blocks_"+file_name

    with open(output_path, "wb") as f:
        f.write(result_pdf)

async def pdf_block_bach():
    original_directory = os.getcwd()
    directory = ".\Test Bench\\raw"
    import glob
    # カレントディレクトリに変更する場合
    os.chdir(directory)
    # PDFファイルのフルパスを取得
    pdf_files = glob.glob('**/*.pdf', recursive=True)
    # ディレクトリをフルパスで取得するには、以下のように結合する
    pdf_files = [os.path.join(directory, file) for file in pdf_files]
    pdf_files = glob.glob('**/*.pdf', recursive=True)

    # ディレクトリー移動
    os.chdir(original_directory)

    for file_path in pdf_files:
        file_path = directory + "\\"+ file_path
        with open(file_path, "rb") as f:
            input_pdf_data = f.read()
        print(F"Loaded: {file_path}")

        result_pdf = await PDF_block_check(input_pdf_data)
        result_pdf = await write_logo_data(result_pdf)

        if result_pdf is None:
            continue
        _, file_name = os.path.split(file_path)
        output_path = bach_process_path + "Blocks_"+file_name

        with open(output_path, "wb") as f:
            f.write(result_pdf)
        print(F"Saved: {output_path}")

async def marge_test():
    base_path = "./Test Bench/raw/3page-Interactive Video Stylization Using Few-Shot Patch-Based Training.pdf"
    tran_path = "./Test Bench/result/result_3page-Interactive Video Stylization Using Few-Shot Patch-Based Training.pdf"
    with open(base_path, "rb") as f:
        input_pdf_data_base = f.read()
    with open(tran_path, "rb") as f:
        input_pdf_data_tran = f.read()
    from modules.pdf_edit import create_viewing_pdf
    #
    result_pdf = await create_viewing_pdf(input_pdf_data_base,input_pdf_data_tran)

    with open("./marge_test.pdf", "wb") as f:
        f.write(result_pdf)

if __name__ == "__main__":
    #asyncio.run(translate_test())
    #asyncio.run(test_bench())
    #asyncio.run(pdf_block_bach())
    asyncio.run(marge_test())
    