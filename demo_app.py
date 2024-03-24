from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException,status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware

from typing import List,Optional,Union
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
import json
import asyncio
from queue import Queue
import psycopg2
from passlib.context import CryptContext

from modules.arxiv_api import get_arxiv_info_async,download_arxiv_pdf
from modules.translate import translate_text
from modules.database import *
from modules.backblaze_api import upload_byte
from modules.translate import pdf_translate

from sqlalchemy.orm import  Session

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, #cookie サポート
    allow_methods=["*"], 
    allow_headers=["*"], #ブラウザからアクセスできるようにするレスポンスヘッダーを示します
)

# --------------- Paper meta DB用処理 ---------------

async def process_translate_arxiv_pdf(target_lang, arxiv_id):
    try:
        # PDFをダウンロードしてバイトデータを取得
        pdf_data = await download_arxiv_pdf(arxiv_id)
        
        #翻訳処理
        translate_data = await pdf_translate(pdf_data,to_lang=target_lang)
        
        return translate_data
    
    except Exception as e:
        print(f"Error processing arxiv_id {arxiv_id}: {str(e)}")
        raise

def load_license_data():
    with open('data/license.json', 'r') as f:
        return json.load(f)

ALLOWED_LANGUAGES = ['en', 'ja']

@app.post("/arxiv/translate/{arxiv_id}")
async def translate_paper_data(arxiv_id: str, deepl_key: str, target_lang: str = "ja"):
    """
    abstract およびPDF を日本語に翻訳します。
    - target_lang:ISO 639-1にて記載のこと
    """
    try:
        # 許可された言語のリストに target_lang が含まれているかを確認
        if target_lang.lower() not in ALLOWED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_lang}. Allowed languages: {', '.join(ALLOWED_LANGUAGES)}")
        
        # ライセンスデータを読み込み
        license_data = load_license_data()
        # Arxiv_データを読み込み
        arxiv_info = await get_arxiv_info_async(arxiv_id)
        paper_license = arxiv_info['license']

        license_ok = license_data.get(paper_license, {}).get("OK", False)

        if not license_ok:
            raise HTTPException(status_code=400, detail="License not permitted for translation")
        
        translate_pdf_data = await process_translate_arxiv_pdf(target_lang, arxiv_id,deepl_key)
            
        return translate_pdf_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo_app:app", host="0.0.0.0", port=8001,reload=True)