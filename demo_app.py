from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks,File, UploadFile,Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,ValidationError
import asyncio

import json,re
from aiohttp import web, ClientSession
from datetime import datetime,timedelta

from modules.backblaze_api import *
from modules.arxiv_api import get_arxiv_info_async,download_arxiv_pdf
from modules.translate import pdf_translate,translate_str_data
from modules.database import *

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA256
import base64
import uuid
from typing import List, Optional

from sqlalchemy import and_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import  Session,selectinload

from config import *
from objective_DB_config import *

app = FastAPI(timeout=300,version="0.1.2")

# 接続許可設定 -----
origins = CORS_CONFIG

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
    await asyncio.sleep(2)
    
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
        
async def decrypt_deepl_key(id,deepl_hash_key,deepl_url):
    """ 
    DeepLキーの復号および、問い合わせキーの確認を行います
    """
    try:
        # DeepL Key 復号化
        private_key = None  # 抜き出されるprivate_keyを初期化
        for index, item in enumerate(private_key_memory):
            if item['id'] == id:
                private_key = item['private_key']
                # private_key_memory.pop(index)  # 見つかった項目をリストから削除
                break  # 最初に見つかった項目を処理した後はループを抜ける
        if private_key is None:
            raise HTTPException(status_code=400, detail="The request has not been sent.")
        await remove_expired_keys()  # 古いデータを消去
        private_key = RSA.import_key(private_key)
        cipher = PKCS1_OAEP.new(private_key, hashAlgo=SHA256)
        decrypted_deepl_key = cipher.decrypt(base64.b64decode(deepl_hash_key)).decode()
        
    except ValueError:
        # 復号化できなかった場合のエラーハンドリング
        raise HTTPException(status_code=400, detail="Key decryption failed. Please check the encryption key and try again.")
    
    # DeepL APIに問い合わせしキーが存在するか確認
    try:
        async with ClientSession() as session:
            await check_deepl_key(decrypted_deepl_key, deepl_url, session)
            return decrypted_deepl_key
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to check DeepL API key.") from e
    
ALLOWED_LANGUAGES = TRANSLATION_CONFIG['ALLOWED_LANGUAGES']

async def check_target_lang(target_lang):
    if target_lang.lower() not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_lang}. Allowed languages: {', '.join(ALLOWED_LANGUAGES)}")

class translate_task_payload(BaseModel):
    arxiv_id: str
    deepl_url: str = "https://api.deepl.com"
    deepl_hash_key: str
    id: str
    target_lang: str = "ja"

@app.post("/add_translate_task")
async def add_user_translate_task(payload: translate_task_payload, background_tasks: BackgroundTasks,db: Session = Depends(get_db)):
    """
    DB にユーザーから入力された翻訳リクエストを記録する
    """
    # --- 許可された翻訳言語か確認---
    await check_target_lang(payload.target_lang)

    # ----- DBに該当Arxivがあるか -----
    existing_paper = db.query(paper_meta_data).filter(paper_meta_data.identifier == f"oai:arXiv.org:{payload.arxiv_id}").first()
    
    if existing_paper:
        if getattr(existing_paper.pdf_url[0], payload.target_lang.lower(), None):
            db.close()
            return {"OK":True,"found":True,"message":"翻訳済みデータが見つかりました","link":F"{URL_LIST['papers_link']}{payload.arxiv_id}"}
    # ----- Arxiv ライセンスチェック -----
    license_data = load_license_data()
    try:
        # Arxiv_データを読み込み
        arxiv_info = await get_arxiv_info_async(payload.arxiv_id)
    except e:
        db.close()
        raise HTTPException(status_code=500, detail="Failed to connect to the ArXiv Server. Please try your request again after some time.") from e
    
    # ----- 存在しないArxiv IDの場合エラー ----- 
    if arxiv_info=={'authors': []}:
        db.close()
        raise HTTPException(status_code=400, detail="Invalid arxiv URL.")
    # ----- ライセンスを確認 ----- 
    paper_license = arxiv_info['license']
    license_ok = license_data.get(paper_license, {}).get("OK", False)
    if not license_ok:
        db.close()
        raise HTTPException(status_code=400, detail="This paper cannot be translated as the license does not permit modifications.")
    
    # ----- DeepL 復号 および ライセンス 確認 -----
    await decrypt_deepl_key(payload.id,payload.deepl_hash_key,payload.deepl_url)
    
    # ----- DBに翻訳タスクの追加 -----
    try:
        # arxiv情報をPaperデータベースに追加
        add_paper = await create_paper_meta_data(arxiv_info,db)

        if not add_paper:
            db.close()
            raise HTTPException(status_code=500, detail="Failed to connect to the database. Please try your request again after some time.") from e
        
        # 翻訳情報をtaskDBに追加
        deepl_translate = Deepl_Translate_Task(
            arxiv_id=F"axv_{payload.arxiv_id}",
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

    return {"ok":True,"found":False,"message": "翻訳を開始しました。画面をリロードしないでください。翻訳が完了すると自動的に翻訳リンクが表示されます。", "arxiv_id": payload.arxiv_id,"link":F"{URL_LIST['papers_link']}{payload.arxiv_id}"}

class translate_pdf_file_payload(BaseModel):
    deepl_url: str = "https://api-free.deepl.com"
    deepl_hash_key: str
    id: str
    target_lang: str = "ja"

@app.post("/translate_pdf_file")
async def translate_for_pdf_file(background_tasks: BackgroundTasks,payload: str = Form(...), file: UploadFile = File(...),db: Session = Depends(get_db)):
    # JSON文字列をPydanticモデルにパースする
    try:
        payload_data = json.loads(payload)
        payload_obj = translate_pdf_file_payload(**payload_data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=422, detail="Invalid payload format")
    # PDFファイルをロード
    pdf_data = await file.read()
    # --- 許可された翻訳言語か確認---
    await check_target_lang(payload_obj.target_lang)
    # -- DeepLキーを復号して
    await decrypt_deepl_key(payload_obj.id,payload_obj.deepl_hash_key,payload_obj.deepl_url)
    # -- PDFデータをストレージに上げる
    current_date = datetime.now().strftime("%Y%m%d%H%M%S")
    await upload_byte(BLACK_BLAZE_CONFIG['private_key_id'],BLACK_BLAZE_CONFIG['private_key'],BLACK_BLAZE_CONFIG['private_bucket'],
                                     pdf_data, 'temp', F"{current_date}.pdf", content_type='application/pdf')
    
    # 翻訳テーブルに追加するデータを追加
    deepl_translate = Deepl_Translate_Task(
        arxiv_id=F"pdf_{current_date}.pdf",
        deepl_hash_key=payload_obj.deepl_hash_key,
        deepl_url = payload_obj.deepl_url,
        uuid=payload_obj.id,
        target_lang =payload_obj.target_lang
    )

    try:
        db.add(deepl_translate)
        db.commit()
        db.refresh(deepl_translate)
        db.close()

        background_tasks.add_task(background_trasnlate_task,payload_obj.id,db)
    except SQLAlchemyError as e:
        db.rollback()  # エラー発生時には変更をロールバック
        db.close()
        print(e)
        raise HTTPException(status_code=500, detail="Failed to connect to the database. Please try your request again after some time.") from e

    # ファイルを保存するか、必要な処理を行う
    return {"ok":True,"message": "翻訳を開始しました。画面をリロードしないでください。翻訳完了後、PDFが表示されます。"}

class get_translate_tasks_payload(BaseModel):
    uuid: str
    deepl_hash_key: str

@app.post("/get_task_progress")
async def get_translate_tasks(payload:get_translate_tasks_payload,db: Session = Depends(get_db)):
    """
    タスクが完了している場合URLを返す
    """
    try:
        serch_result = db.query(Translate_logs).filter(
            and_(
                Translate_logs.uuid==payload.uuid,
                Translate_logs.deepl_hash_key==payload.deepl_hash_key
                )
                ).first()
        if serch_result:
            if serch_result.done:
                return {'ok':True,'done':True,'link':serch_result.link}
            else:
                {'ok':True,'done':False,'error':serch_result.link}
        else:
            return {'ok':False,'link':None}
    except Exception as e:
        # ここでエラー処理を行う
        return {'ok': False, 'error': str(e)}

# -------------- バックグラウンドタスク -----------

def delete_expired_translate_logs(db):
    """
    Translate_logsから1時間たったデータを消去する
    """
    try:
        # 現在時刻より1時間以上前の時刻を計算
        one_hour_ago = datetime.now() - timedelta(hours=1)

        # datestampが1時間以上前のレコードを検索
        expired_logs = db.query(Translate_logs).filter(Translate_logs.datestamp < one_hour_ago).all()

        # 該当するすべてのレコードを削除
        for log in expired_logs:
            db.delete(log)

        # 変更をコミット
        db.commit()
        print(f"Deleted {len(expired_logs)} expired logs.")
    except Exception as e:
        db.rollback()
        print(f"An error occurred: {e}")

async def Arxiv_back_gorund_task(arxiv_id, target_lang, decrypted_deepl_key, deepl_url,db):
    message = None
    paper = db.query(paper_meta_data).filter(paper_meta_data.identifier == f"oai:arXiv.org:{arxiv_id}").first()
    # DBのアブストとタイトルを翻訳
    if not getattr(paper.abstract[0], target_lang, None):
        translation_result = await translate_str_data(decrypted_deepl_key, paper.abstract[0].en, target_lang, deepl_url)
        if translation_result['ok']:
            setattr(paper.abstract[0], target_lang, translation_result['data'])
        else:
            print(translation_result)
            print("abstract error")
            return "error"  # DBにエラー追加するコードを後ほど追加

    if not getattr(paper.title[0], target_lang, None):
        translation_result = await translate_str_data(decrypted_deepl_key, paper.title[0].en, target_lang, deepl_url)
        if translation_result['ok']:
            setattr(paper.title[0], target_lang, translation_result['data'])
        else:
            print("title error")
            return "error"  # DBにエラー追加するコードを後ほど追加
        
    # 論文データをDB追加
    db.commit()
    db.refresh(paper)
    
    # PDFデータの取得
    pdf_data = await download_arxiv_pdf(arxiv_id)
    download_url = await upload_byte(BLACK_BLAZE_CONFIG['public_key_id'],BLACK_BLAZE_CONFIG['public_key'],BLACK_BLAZE_CONFIG['public_bucket'],
                                     pdf_data, 'arxiv_pdf', F"{arxiv_id}_en.pdf", content_type='application/pdf')
    #翻訳処理
    try:
        translate_data = await pdf_translate(decrypted_deepl_key,pdf_data,to_lang= target_lang,api_url=deepl_url)
    except Exception as e:
        message = str(e)
        print(f"翻訳中にエラーが発生しました: {message} / {arxiv_id}")
        message = F"翻訳中にエラーが発生しました: {message} / {arxiv_id}"
        return "error", message
    
    translate_download_url = await upload_byte(BLACK_BLAZE_CONFIG['public_key_id'],BLACK_BLAZE_CONFIG['public_key'],BLACK_BLAZE_CONFIG['public_bucket'],
                                               translate_data, 'arxiv_pdf', F"{arxiv_id}_{target_lang}.pdf", content_type='application/pdf')
    #翻訳PDF保存後不要データのリセット
    translate_data = None
    pdf_data = None

    # DB へ翻訳PDFリンクをデータ追加
    setattr(paper.pdf_url[0], "en", download_url)
    setattr(paper.pdf_url[0], target_lang, translate_download_url)
    
    db.commit()
    db.refresh(paper)
    download_url = F"{URL_LIST['papers_link']}{arxiv_id}"
    return download_url, message

async def pdf_back_ground_task(file_name, target_lang, decrypted_deepl_key, deepl_url):
    """
    PDF翻訳BGタスク
    """
    file_path = F'temp/{file_name}'
    id = BLACK_BLAZE_CONFIG['private_key_id']
    key= BLACK_BLAZE_CONFIG['private_key']
    bucket_name = BLACK_BLAZE_CONFIG['private_bucket']
    message = None

    # フォルダー内にある古いデータを消去
    #list_file_data = await list_files_in_folder(id,key,bucket_name,'temp')
    #list_file_data = await find_recent_files(list_file_data,600) #現在時刻より10分以内でないファイルリストを作成
    #await delete_files_from_folder(id,key,bucket_name,list_file_data)
    # トークン発行および翻訳元データのダウンロード

    try:
        auth_token = create_download_auth_token(id,key,bucket_name,file_path,600) # 10分有効なキーを発行
        pdf_byte = await download_file(id,key,bucket_name,file_path,auth_token)
        translate_data = await pdf_translate(decrypted_deepl_key,pdf_byte,to_lang= target_lang,api_url=deepl_url)
        #翻訳データのダウンロードURLを発行
        download_url = await upload_byte(id,key,bucket_name,translate_data,'temp',file_name,'application/pdf')
        
    except Exception as e:
        message = str(e)
        message = F"エラーが発生しました: {message} / {file_name}"
        print(f"{message} / {file_name}")
        return "error", message
    
    download_url = F"{download_url}?Authorization={auth_token}"
    return download_url, message

async def background_trasnlate_task(uuid,db):
    """
    翻訳用のタスクを追加
    """
    # 翻訳用DBよりUUIDから、翻訳データを取得
    task_data = db.query(Deepl_Translate_Task).filter(Deepl_Translate_Task.uuid==uuid).first()
    if task_data is None:
        print("taskdata_none")
        db.close()
        return
    print(F"Back Ground Task: {task_data.arxiv_id}")
    deepl_url = task_data.deepl_url + "/v2/translate"
    
    # DeepL Key 復号化
    private_key = None  # 抜き出されるprivate_keyを初期化
    for index, item in enumerate(private_key_memory):
        if item['id'] == uuid:
            private_key = item['private_key']
            private_key_memory.pop(index)  # 見つかった項目をリストから削除
            break  # 最初に見つかった項目を処理した後はループを抜ける
    
    if private_key is None:
        db.delete(task_data)
        db.commit()
        db.close()
        print("private key none")
        return #error
    
    private_key = RSA.import_key(private_key)
    cipher = PKCS1_OAEP.new(private_key, hashAlgo=SHA256)
    decrypted_deepl_key = cipher.decrypt(base64.b64decode(task_data.deepl_hash_key)).decode()

    # サービスに応じて翻訳分岐する
    mode,translate_parts=task_data.arxiv_id.split('_', 1)
    if mode == "axv":
        link,mes = await Arxiv_back_gorund_task(translate_parts,task_data.target_lang,decrypted_deepl_key,deepl_url,db)
    elif mode == "pdf":
        link,mes = await pdf_back_ground_task(translate_parts,task_data.target_lang,decrypted_deepl_key,deepl_url)
    if mes:
        done_flag = Translate_logs(
            done = False,
            uuid = uuid,
            deepl_hash_key = task_data.deepl_hash_key,
            mode = mode,
            link = mes,
            datestamp = datetime.now()
        )
    else:
        done_flag = Translate_logs(
            done = True,
            uuid = uuid,
            deepl_hash_key = task_data.deepl_hash_key,
            mode = mode,
            link = link,
            datestamp = datetime.now()
        )
    db.add(done_flag)
    # 最後、翻訳taskDBからタスクを消去 & プライベートメモリーリストより使われていないデータがある場合消去
    await remove_expired_keys() #古いデータをリストから消去
    delete_expired_translate_logs(db) # 翻訳済みテーブルから1時間経過したデータを消去
    db.delete(task_data)
    db.commit()
    db.refresh(done_flag)
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