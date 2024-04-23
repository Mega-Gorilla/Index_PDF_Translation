import os
# config.py

# Black BlazeオブジェクトDB設定
BLACK_BLAZE_CONFIG = {
    'public_key_id': os.environ["blackblaze_public_id"],
    'public_key' : os.environ["blackblaze_public_key"],
    'private_key_id': os.environ["blackblaze_private_id"],
    'private_key' : os.environ["blackblaze_private_key"],
    'public_bucket' : 'pdf-public',
    'private_bucket' : 'pdf-private'
}

# 接続許可リスト
CORS_CONFIG = [
    'http://192.168.0.65:5173',
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