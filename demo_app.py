from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import json,re
from aiohttp import web, ClientSession
from datetime import datetime

from modules.backblaze_api import upload_byte
from modules.arxiv_api import get_arxiv_info_async,download_arxiv_pdf
from modules.translate import pdf_translate
from modules.database import *

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA256
import base64

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import  Session

app = FastAPI(timeout=300,version="0.1.1")

# 接続許可設定 -----
origins = [
    "http://localhost:5173",
    'https://indqx-demo-front.onrender.com'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, #cookie サポート
    allow_methods=["*"], 
    allow_headers=["*"], #ブラウザからアクセスできるようにするレスポンスヘッダーを示します
)
# RSA設定 -----
def pseudo_random_generator(seed):
    """
    再現性のある疑似乱数生成器。
    指定されたシードに基づいて、必要に応じてランダムバイト列を生成します。
    """
    while True:
        seed = SHA256.new(seed).digest()  # 新しいシードを生成
        for byte in seed:
            yield byte

def transform_date(current_date):
    """
    current_dateを変換して再現性のある値を生成します。
    この変換はハッシュ関数を使用して行います。
    """
    hash = SHA256.new(current_date.encode()).digest()
    return hash

async def generate_public_key(current_date):
    FIXED_KEY = os.environ["indqx_userdeepl_key"]
    transformed_date = transform_date(current_date)  # 日付を変換
    combined_key = f"{transformed_date.hex()}_{FIXED_KEY}"
    seed = SHA256.new(combined_key.encode()).digest()

    # 疑似乱数生成器を初期化
    rng = pseudo_random_generator(seed)
    randfunc = lambda n: bytes([next(rng) for _ in range(n)])
    
    rsa_key = RSA.generate(2048, randfunc=randfunc)
    public_key = rsa_key.publickey().export_key()
    return public_key

async def generate_private_key(current_date):
    FIXED_KEY = os.environ["indqx_userdeepl_key"]
    transformed_date = transform_date(current_date)  # 日付を変換
    combined_key = f"{transformed_date.hex()}_{FIXED_KEY}"
    seed = SHA256.new(combined_key.encode()).digest()

    # 疑似乱数生成器を初期化
    rng = pseudo_random_generator(seed)
    randfunc = lambda n: bytes([next(rng) for _ in range(n)])
    
    rsa_key = RSA.generate(2048, randfunc=randfunc)
    private_key = rsa_key.export_key()
    return private_key

@app.get("/public-key")
async def get_public_key():
    #　日付に基づいてRSAキーを自動生成
    import time
    process_time = time.time()
    current_date = datetime.now().strftime("%Y%m%d%H%M%S")
    public_key = await generate_public_key(current_date)
    print(F"GenerateKey:{time.time()-process_time}")
    return {"datestamp":current_date,"public_key": public_key.decode()}

# DB設定 -----
Base.metadata.create_all(bind=engine)   #テーブルの作成
def get_db():                           # データベースセッションの取得
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def load_license_data():
    with open('data/license.json', 'r') as f:
        return json.load(f)
    
async def check_deepl_key(deepl_key, deepl_url, session):
    """
    DeepL Keyが使えるKeyか確認し、問題があればエラーを返します
    """
    headers = {"Authorization": f"DeepL-Auth-Key {deepl_key}"}
    async with session.get(f"{deepl_url}/v2/usage", headers=headers) as response:
        if response.status == 403:
            raise web.HTTPBadRequest(reason="Invalid DeepL API Key.")
        elif response.status != 200:
            raise web.HTTPInternalServerError(reason="Error checking DeepL API key.")
        
class translate_task_payload(BaseModel):
    arxiv_id: str
    deepl_url: str
    deepl_hash_key: str
    datestamp: str

@app.post("/add_translate_task")
async def add_user_translate_task(payload: translate_task_payload,db: Session = Depends(get_db)):
    """
    DB にユーザーから入力された翻訳リクエストを記録する
    """
    
    # ----- ライセンスチェック -----
    license_data = load_license_data()
    try:
        # Arxiv_データを読み込み
        arxiv_info = await get_arxiv_info_async(payload.arxiv_id)
    except e:
        raise HTTPException(status_code=500, detail="Failed to connect to the ArXiv Server. Please try your request again after some time.") from e
    
    # 存在しないArxiv IDの場合エラー
    if arxiv_info=={'authors': []}:
        raise HTTPException(status_code=400, detail="Invalid arxiv URL.")
    
    paper_license = arxiv_info['license']
    license_ok = license_data.get(paper_license, {}).get("OK", False)
    if not license_ok:
        raise HTTPException(status_code=400, detail="License not permitted for translation")
    
    # ----- DeepL ライセンス 確認 -----
    cipher = PKCS1_OAEP.new(RSA.import_key(generate_private_key(payload.datestamp)))
    decrypted_deepl_key = cipher.decrypt(base64.b64decode(payload.deepl_hash_key)).decode()
    async with ClientSession() as session:
        await check_deepl_key(decrypted_deepl_key, payload.deepl_url, session)

    # ----- タスクの追加 -----
    try:
        deepl_translate = Deepl_Translate_Task(
            arxiv_id=payload.arxiv_id,
            deepl_hash_key=payload.deepl_hash_key,
            datestamp=payload.datestamp  # 現在の時刻を設定
        )

        db.add(deepl_translate)
        db.commit()
        db.refresh(deepl_translate)

    except SQLAlchemyError as e:
        db.rollback()  # エラー発生時には変更をロールバック
        raise HTTPException(status_code=500, detail="Failed to connect to the database. Please try your request again after some time.") from e

    return {"ok":True,"message": "翻訳タスクを追加しました。", "arxiv_id": payload.arxiv_id}

@app.post("/check/{arxiv_id}")
async def check_arxiv_id(arxiv_id: str):
    """
    ArXiv IDのフォーマットを確認
    """
    # arxiv_idの形式をチェック
    if not re.match(r'^\d{4}\.\d{5}$', arxiv_id):
        raise HTTPException(status_code=400, detail="Invalid arxiv ID.")
    
    return {"ok":True,"message":"The data format of the ArXiv ID was correct."}

async def process_translate_arxiv_pdf(key,target_lang, arxiv_id,api_url):
    """
    PDFを翻訳する
    """
    try:
        # PDFをダウンロードしてバイトデータを取得
        pdf_data = await download_arxiv_pdf(arxiv_id)
        
        #翻訳処理
        translate_data = await pdf_translate(key,pdf_data,to_lang=target_lang,api_url=api_url)
        download_url = await upload_byte(translate_data, 'arxiv_pdf', F"{arxiv_id}_{target_lang}.pdf", content_type='application/pdf')

        return download_url
    
    except Exception as e:
        print(f"Error processing arxiv_id {arxiv_id}: {str(e)}")
        raise

    
        
ALLOWED_LANGUAGES = ['en', 'ja']

class TranslateRequest(BaseModel):
    deepl_url: str = "https://api.deepl.com"
    deepl_key: str
    target_lang: str = "ja"

@app.post("/arxiv/translate/{arxiv_id}")
async def translate_paper_data(arxiv_id: str, request: TranslateRequest):
    """
    abstract およびPDF を日本語に翻訳します。
    - target_lang:ISO 639-1にて記載のこと
    """
    target_lang = request.target_lang
    deepl_key = request.deepl_key
    deepl_url = request.deepl_url
    
    # arxiv_idの形式をチェック
    if not re.match(r'^\d{4}\.\d{5}$', arxiv_id):
        raise HTTPException(status_code=400, detail="Invalid arxiv URL.")
    
    # DeepL APIキーの有効性をチェック
    async with ClientSession() as session:
        await check_deepl_key(deepl_key, deepl_url, session)
        
    try:
        # 許可された言語のリストに target_lang が含まれているかを確認
        if target_lang.lower() not in ALLOWED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_lang}. Allowed languages: {', '.join(ALLOWED_LANGUAGES)}")
        # ライセンスデータを読み込み
        license_data = load_license_data()
        # Arxiv_データを読み込み
        arxiv_info = await get_arxiv_info_async(arxiv_id)
        # 存在しないArxiv IDの場合エラー
        if arxiv_info=={'authors': []}:
            raise HTTPException(status_code=400, detail="Invalid arxiv URL.")
        
        paper_license = arxiv_info['license']

        license_ok = license_data.get(paper_license, {}).get("OK", False)

        if not license_ok:
            raise HTTPException(status_code=400, detail="License not permitted for translation")
        
        try:
            deepl_url = F"{deepl_url}/v2/translate"
            pdf_dl_url = await process_translate_arxiv_pdf(deepl_key,target_lang, arxiv_id,deepl_url)
            return pdf_dl_url
        
        except Exception as e:
            raise e
    
    except HTTPException as e:
        # HTTPExceptionはそのまま投げる
        raise e
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo_app:app", host="0.0.0.0", port=8001,reload=True)