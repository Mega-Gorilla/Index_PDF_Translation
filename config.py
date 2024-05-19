import os
# config.py
# ----- ローカル版 設定 ------
DeepL_API_Key = "Your DeepL API Key"
DeepL_URL = "https://api-free.deepl.com/v2/translate" # DeepL Proの場合は、「https://api.deepl.com/v2/translate」を設定してください
Output_folder_path = "./output/"

# ----- 以下 API用 設定 --------

# 接続許可リスト
CORS_CONFIG = [
    'https://indqx-demo-front.onrender.com',
    'http://localhost:5173'
]

#URLリスト
URL_LIST = {
    'papers_link':'https://indqx-demo-front.onrender.com/abs/'
}

# 翻訳設定
TRANSLATION_CONFIG = {
    'ALLOWED_LANGUAGES':['en', 'ja']
}

#token 計算用
SPACY_CONFIG = {
    'supported_languages': {
        'en': 'en_core_web_sm',
        'ja': 'ja_core_news_sm'
        }
}

# デバッグ用ファイル位置
Debug_folder_path = "./debug/"
bach_process_path = "./Test Bench/result/"