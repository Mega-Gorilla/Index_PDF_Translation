from fastapi import FastAPI, BackgroundTasks, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from modules.arxiv_api import get_arxiv_info_async
from datetime import datetime
from typing import List,Optional

from queue import Queue
import os

import psycopg2
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Table, ForeignKey,func
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

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

# --------------- DB設定 ----------------
# PostgreSQLデータベースの接続設定
SQLALCHEMY_DATABASE_URL = os.environ.get('render-db-url')
# SQLAlchemyエンジンの作成
engine = create_engine(SQLALCHEMY_DATABASE_URL)
# セッションの作成 (autocommitとautoflushはFalseに設定され、明示的にコミットやフラッシュを行う必要があります。)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Baseモデルの作成
Base = declarative_base()
# データベースモデルの定義

# 中間テーブル
user_authors = Table(
    "user_authors", #テーブル名
    Base.metadata,
    Column("paper_meta", Integer, ForeignKey("paper_meta.id")),
    Column("author_id", String, ForeignKey("authors.id")),
)
class Author(Base):
    __tablename__ = "authors"
    id = Column(String, primary_key=True, index=True)

paper_categories = Table('paper_categories', Base.metadata,
    Column('paper_id', Integer, ForeignKey('paper_meta.id')),
    Column('category_id', String, ForeignKey('categories.id'))
)
class Category(Base):
    __tablename__ = "categories"

    id = Column(String, primary_key=True, index=True)


# 論文メタデータリスト
class paper_meta_data(Base):
    __tablename__ = "paper_meta"

    id = Column(Integer, primary_key=True, index=True)  #2402.10949
    identifier = Column(String, index=True)             #oai:arXiv.org:2402.10949
    datestamp = Column(DateTime, index=True)            #2024-02-21
    setSpec = Column(String)                            #cs
    created = Column(DateTime)                          #2024-02-09
    updated = Column(DateTime)                          #2024-02-20
    authors = relationship("Author", secondary="user_authors") #[{"id": 3,"keyname": "Zhu","forenames": "Lei"},{"id": 4,"keyname": "Wei","forenames": "Fangyun"},{"id": 5,"keyname": "Lu","forenames": "Yanye"}],
    title = Column(String)                              #"The Unreasonable Effectiveness of Eccentric Automatic Prompts",
    categories = Column(String)                         #cs.CL cs.AI cs.LG
    categories_list = relationship("Category", secondary=paper_categories)
    license = Column(String)                            #"http://creativecommons.org/licenses/by/4.0/",
    abstract = Column(String)                           #"  Large Language Models (LLMs) have demonstrated remarkable

class AuthorResponse(BaseModel):
    id: str

    class Config:
        from_attributes = True

class CategoryResponse(BaseModel):
    id: str

    class Config:
        from_attributes = True

class PaperMetaDataResponse(BaseModel):
    id: int
    identifier: str
    datestamp: datetime
    setSpec: str
    created: datetime
    updated: Optional[datetime]
    authors: List[AuthorResponse]
    title: str
    categories: str
    categories_list: List[CategoryResponse]  # 修正
    license: str
    abstract: str

    class Config:
        from_attributes = True

# テーブルの作成
Base.metadata.create_all(bind=engine)

# データベースセッションの取得
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def create_paper_meta_data(arxiv_info: dict, db: Session):
    """
    arxiv_info: ArXivから受信した辞書配列情報
    db: データベース
    """

    # 著者データの作成
    authors = []
    for author in arxiv_info['authors']:
        author_id = f"{author['forenames']} {author['keyname']}"
        db_author = db.query(Author).filter(Author.id == author_id).first()
        if not db_author:
            db_author = Author(id=author_id)
        authors.append(db_author)

    # カテゴリーリストを作成
    categories_list = []
    for category in arxiv_info['categories'].split():
        db_category = db.query(Category).filter(Category.id == category).first()
        if not db_category:
            db_category = Category(id=category)
        categories_list.append(db_category)

    # paper_meta_dataの作成
    paper_meta = paper_meta_data(
        identifier=arxiv_info['identifier'],
        datestamp=datetime.strptime(arxiv_info['datestamp'], '%Y-%m-%d') if 'datestamp' in arxiv_info else None,
        setSpec=arxiv_info['setSpec'],
        created=datetime.strptime(arxiv_info['created'], '%Y-%m-%d') if 'created' in arxiv_info else None,
        updated=datetime.strptime(arxiv_info['updated'], '%Y-%m-%d') if 'updated' in arxiv_info else None,
        authors=authors,
        title=arxiv_info['title'],
        categories=arxiv_info['categories'],
        categories_list=categories_list,
        license=arxiv_info['license'],
        abstract=arxiv_info['abstract']
    )

    # セッションにデータを追加
    db.add(paper_meta)
    db.commit()
    db.refresh(paper_meta)

    return paper_meta

@app.get("/arxiv/metadata/{arxiv_id}")
async def Get_Paper_Data(arxiv_id: str, db: Session = Depends(get_db)):
    # データベースに問い合わせて、指定された arxiv_id のデータを取得
    existing_paper = db.query(paper_meta_data).filter(paper_meta_data.identifier == f"oai:arXiv.org:{arxiv_id}").first()

    if existing_paper:
        # データが既に存在する場合は、そのデータを返す
        return existing_paper
    else:
        # データが存在しない場合は、Arxiv からメタデータを取得し、データベースに保存
        arxiv_info = await get_arxiv_info_async(arxiv_id)
        paper_meta = await create_paper_meta_data(arxiv_info, db)
        return paper_meta

# --------------- DB問い合わせ ----------------

@app.get("/papers/", response_model=List[PaperMetaDataResponse])
async def get_all_papers(db: Session = Depends(get_db)):
    papers = db.query(paper_meta_data).all()
    return papers

@app.get("/papers/search/", response_model=List[PaperMetaDataResponse])
async def search_papers_by_author(author_name: str, db: Session = Depends(get_db)):
    # 著者名が完全一致する場合を検索
    papers = db.query(paper_meta_data).join(paper_meta_data.authors).filter(
        Author.id == author_name
    ).all()
    
    return papers
# -----------------------------------
# ジョブキューを作成
job_queue = Queue()

# ジョブを処理する非同期タスク
async def process_job(arxiv_id: str):
    try:
        # 仮想のメタデータ取得関数を想定
        data = await get_arxiv_info_async(arxiv_id)
        print(f"Processed job for arxiv_id: {arxiv_id}\n{data}")
    except Exception as e:
        print(f"Error occurred while processing job for arxiv_id: {arxiv_id}. Error: {str(e)}")

if __name__ == "__main__":
    #uvicorn API:app --reload   
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001,reload=True)