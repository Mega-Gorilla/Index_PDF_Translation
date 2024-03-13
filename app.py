from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from modules.arxiv_api import get_arxiv_info
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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

class Response(BaseModel):
    ok: bool
    text: str

@app.get("/test_api", response_model=Response)
async def read_root():
    return {"ok": True, "text": "Hello world."}

@app.get("/arxiv-license/{arxiv_id}")
async def get_arxiv_license(arxiv_id: str):
    license_info = get_arxiv_info(arxiv_id)
    return {"arxiv_id": arxiv_id, "license_info": license_info}

if __name__ == "__main__":
    #uvicorn API:app --reload   
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001,reload=True)