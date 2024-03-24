from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

import json
from modules.backblaze_api import upload_byte

from modules.arxiv_api import get_arxiv_info_async,download_arxiv_pdf
from modules.translate import pdf_translate

app = FastAPI(timeout=300)

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

# --------------- Paper meta DB用処理 ---------------

async def process_translate_arxiv_pdf(key,target_lang, arxiv_id,api_url):
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

def load_license_data():
    with open('data/license.json', 'r') as f:
        return json.load(f)

ALLOWED_LANGUAGES = ['en', 'ja']

@app.post("/arxiv/translate/{arxiv_id}")
async def translate_paper_data(arxiv_id: str,deepl_url:str, deepl_key: str, target_lang: str = "ja"):
    """
    abstract およびPDF を日本語に翻訳します。
    - target_lang:ISO 639-1にて記載のこと
    """
    print("処理開始 - 1")
    try:
        # 許可された言語のリストに target_lang が含まれているかを確認
        if target_lang.lower() not in ALLOWED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_lang}. Allowed languages: {', '.join(ALLOWED_LANGUAGES)}")
        print("翻訳言語確認 -2")
        # ライセンスデータを読み込み
        license_data = load_license_data()
        print("ライセンスデータ取得 -3" )
        # Arxiv_データを読み込み
        arxiv_info = await get_arxiv_info_async(arxiv_id)
        print("Arxiv問い合わせ -4")
        paper_license = arxiv_info['license']

        license_ok = license_data.get(paper_license, {}).get("OK", False)
        print("ライセンス確認 -5")

        if not license_ok:
            raise HTTPException(status_code=400, detail="License not permitted for translation")
        
        print("翻訳開始 -6")
        pdf_dl_url = await process_translate_arxiv_pdf(deepl_key,target_lang, arxiv_id,deepl_url)
            
        print("翻訳終了 -7")
        return pdf_dl_url
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo_app:app", host="0.0.0.0", port=8001,reload=True)