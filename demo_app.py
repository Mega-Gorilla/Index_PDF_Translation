from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

import json,re
from aiohttp import web, ClientSession
from datetime import datetime,timedelta

from modules.backblaze_api import upload_byte
from modules.arxiv_api import get_arxiv_info_async,download_arxiv_pdf
from modules.translate import pdf_translate,translate_text
from modules.database import *

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA256
import base64
import uuid
from typing import List, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import  Session,selectinload

app = FastAPI(timeout=300,version="0.1.2")

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

# メモリ上に保持するリストを定義
private_key_memory: List[dict] = []

async def generate_key_pair():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return public_key, private_key

async def remove_expired_keys():
    """
    リクエストされて20以上経過しているprivate_key_memoryのデータを消去
    """
    current_time = datetime.now()
    global private_key_memory
    private_key_memory = [item for item in private_key_memory if current_time - datetime.strptime(item["datestamp"], "%Y%m%d%H%M%S") <= timedelta(minutes=20)]

@app.get("/public-key")
async def get_public_key():
    current_date = datetime.now().strftime("%Y%m%d%H%M%S")
    public_key, private_key = await generate_key_pair()
    unique_id = str(uuid.uuid4())  # ランダムなUUIDを生成し、文字列に変換
    private_key_memory.append({"id": unique_id, "datestamp": current_date, "private_key": private_key})
    
    # UUIDを返す場合も含め、適宜レスポンスを調整
    return {"id": unique_id, "datestamp": current_date, "public_key": public_key.decode()}

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
    deepl_url: str = "https://api.deepl.com"
    deepl_hash_key: str
    id: str
    target_lang: str = "ja"

ALLOWED_LANGUAGES = ['en', 'ja']

@app.post("/add_translate_task")
async def add_user_translate_task(payload: translate_task_payload, background_tasks: BackgroundTasks,db: Session = Depends(get_db)):
    """
    DB にユーザーから入力された翻訳リクエストを記録する
    """
    # --- 許可された翻訳言語か確認---
    if payload.target_lang.lower() not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {payload.target_lang}. Allowed languages: {', '.join(ALLOWED_LANGUAGES)}")
    # ----- DBに該当Arxivがあるか -----
    existing_paper = db.query(paper_meta_data).filter(paper_meta_data.identifier == f"oai:arXiv.org:{payload.arxiv_id}").first()
    
    if existing_paper:
        if getattr(existing_paper.pdf_url[0], payload.target_lang.lower(), None):
            return {"OK":True,"message":"翻訳済みデータが見つかりました","link":F"https://indqx-demo-front.onrender.com/abs/{payload.arxiv_id}"}
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
        raise HTTPException(status_code=400, detail="This paper cannot be translated as the license does not permit modifications.")
    
    # ----- DeepL ライセンス 確認 -----
    try:
        # DeepL Key 復号化
        private_key = None  # 抜き出されるprivate_keyを初期化
        for index, item in enumerate(private_key_memory):
            if item['id'] == payload.id:
                private_key = item['private_key']
                #private_key_memory.pop(index)  # 見つかった項目をリストから削除
                break  # 最初に見つかった項目を処理した後はループを抜ける
        if private_key is None:
            raise HTTPException(status_code=400, detail="The request has not been sent.")
        await remove_expired_keys() #古いデータを消去
        private_key = RSA.import_key(private_key)
        cipher = PKCS1_OAEP.new(private_key, hashAlgo=SHA256)
        decrypted_deepl_key = cipher.decrypt(base64.b64decode(payload.deepl_hash_key)).decode()
    except ValueError:
        # 復号化できなかった場合のエラーハンドリング
        raise HTTPException(status_code=400, detail="Key decryption failed. Please check the encryption key and try again.")
    
    # DeepL APIに問い合わせしキーが存在するか確認
    try:
        async with ClientSession() as session:
            await check_deepl_key(decrypted_deepl_key, payload.deepl_url, session)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to check DeepL API key.") from e
    # ----- DBに翻訳タスクの追加 -----
    try:
        # arxiv情報をPaperデータベースに追加
        add_paper = await create_paper_meta_data(arxiv_info,db)
        if not add_paper:
            raise HTTPException(status_code=500, detail="Failed to connect to the database. Please try your request again after some time.") from e
        # 翻訳情報をtaskDBに追加
        deepl_translate = Deepl_Translate_Task(
            arxiv_id=payload.arxiv_id,
            deepl_hash_key=payload.deepl_hash_key,
            deepl_url = payload.deepl_url,
            uuid=payload.id,
            target_lang =payload.target_lang
        )

        db.add(deepl_translate)
        db.commit()
        db.refresh(deepl_translate)
        db.close()

        background_tasks.add_task(background_trasnlate_task,payload.id,db)

    except SQLAlchemyError as e:
        db.rollback()  # エラー発生時には変更をロールバック
        db.close()
        print(e)
        raise HTTPException(status_code=500, detail="Failed to connect to the database. Please try your request again after some time.") from e

    return {"ok":True,"message": "翻訳を開始しました。", "arxiv_id": payload.arxiv_id,"link":F"https://indqx-demo-front.onrender.com/abs/{payload.arxiv_id}"}

async def background_trasnlate_task(uuid,db: Session):
    task_data = db.query(Deepl_Translate_Task).filter(Deepl_Translate_Task.uuid==uuid).first()
    if task_data == None:
        print("taskdata_none")
        return #error
    deepl_url = task_data.deepl_url + "/v2/translate"
    paper = db.query(paper_meta_data).filter(paper_meta_data.identifier == f"oai:arXiv.org:{task_data.arxiv_id}").first()
    
    # DeepL Key 復号化
    private_key = None  # 抜き出されるprivate_keyを初期化
    for index, item in enumerate(private_key_memory):
        if item['id'] == uuid:
            private_key = item['private_key']
            private_key_memory.pop(index)  # 見つかった項目をリストから削除
            break  # 最初に見つかった項目を処理した後はループを抜ける
    if private_key is None:
        print("private key none")
        return #error
    await remove_expired_keys() #古いデータを消去
    private_key = RSA.import_key(private_key)
    cipher = PKCS1_OAEP.new(private_key, hashAlgo=SHA256)
    decrypted_deepl_key = cipher.decrypt(base64.b64decode(task_data.deepl_hash_key)).decode()
    # アブストとタイトルを翻訳する
    if not getattr(paper.abstract[0], task_data.target_lang, None):
        translation_result = await translate_text(decrypted_deepl_key,paper.abstract[0].en, task_data.target_lang,deepl_url)
        if translation_result['ok']:
            setattr(paper.abstract[0], task_data.target_lang, translation_result['data'])
        else:
            print(translation_result)
            print("abstract error")
            return #error
    
    if not getattr(paper.title[0], task_data.target_lang, None):
        translation_result = await translate_text(decrypted_deepl_key,paper.title[0].en, task_data.target_lang,deepl_url)
        if translation_result['ok']:
            setattr(paper.title[0], task_data.target_lang, translation_result['data'])
        else:
            print("title error")
            return #error
    db.commit()
    db.refresh(paper)
    # PDFをダウンロードしてバイトデータを取得
    pdf_data = await download_arxiv_pdf(task_data.arxiv_id)
    download_url = await upload_byte(pdf_data, 'arxiv_pdf', F"{task_data.arxiv_id}_en.pdf", content_type='application/pdf')
    #翻訳処理
    translate_data = await pdf_translate(decrypted_deepl_key,pdf_data,to_lang=task_data.target_lang,api_url=deepl_url)
    translate_download_url = await upload_byte(translate_data, 'arxiv_pdf', F"{task_data.arxiv_id}_{task_data.target_lang}.pdf", content_type='application/pdf')
    setattr(paper.pdf_url[0], "en", download_url)
    setattr(paper.pdf_url[0], task_data.target_lang, translate_download_url)
    #データリセット
    translate_data = None
    pdf_data = None
    await remove_expired_keys()

    db.delete(task_data)
    db.commit()
    db.refresh(paper)
    db.close()

@app.get("/arxiv/metadata/{arxiv_id}")
async def Get_Paper_Data(arxiv_id: str, db: Session = Depends(get_db)):
    """
    ArXiv IDより、自社DBのデータを参照します。存在しない場合はArXiv APIに問い合わせ実施する。
    """
    # 自社データベースに問い合わせて、指定された arxiv_id のデータを取得
    existing_paper = db.query(paper_meta_data).options(
        selectinload(paper_meta_data.authors),
        selectinload(paper_meta_data.title),
        selectinload(paper_meta_data.categories_list),
        selectinload(paper_meta_data.abstract),
        selectinload(paper_meta_data.abstract_user),
        selectinload(paper_meta_data.pdf_url),
        selectinload(paper_meta_data.comments),
    ).filter(paper_meta_data.identifier == f"oai:arXiv.org:{arxiv_id}").first()
    #print(existing_paper.abstract)
    return existing_paper
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo_app:app", host="0.0.0.0", port=8001,reload=True)