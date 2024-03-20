from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException,status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware

from typing import List,Optional,Union
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
import json
from queue import Queue
import psycopg2
from passlib.context import CryptContext

from modules.arxiv_api import get_arxiv_info_async
from modules.translate import translate_text
from modules.database import *

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

# --------------- FastAPI用 DB型宣言 ---------------
class AuthorResponse(BaseModel):
    id: str

    class Config:
        from_attributes = True

class TitleResponse(BaseModel):
    en: Optional[str]
    ja: Optional[str]
    paper_meta_id: Optional[int]
    
    class Config:
        from_attributes = True

class CategoryResponse(BaseModel):
    id: str

    class Config:
        from_attributes = True

class CommentResponse(BaseModel):
    id: int
    user_id: str
    content: str
    lang: Optional[str]
    created_at: datetime
    paper_meta_id: Optional[int]

    class Config:
        from_attributes = True

class AbstractResponse(BaseModel):
    en: Optional[str]
    ja: Optional[str]
    paper_meta_id: Optional[int]

    class Config:
        from_attributes = True

class AbstractUserResponse(BaseModel):
    id: int
    user_id: str
    lang: Optional[str]
    like: int
    content: str
    created_at: datetime
    paper_meta_id: Optional[int]

    class Config:
        from_attributes = True

class PdfURLResponse(BaseModel):
    en: Optional[str]
    ja: Optional[str]
    paper_meta_id: Optional[int]

    class Config:
        from_attributes = True

class PaperMetaDataResponse(BaseModel):
    #　追加した場合、paper_meta_dataと、 paper_meta = paper_meta_dataを修正し、DBに列を追加のこと
    id: int
    identifier: str
    datestamp: datetime
    setSpec: str
    created: datetime
    updated: Optional[datetime]
    authors: List[AuthorResponse]
    title: List[TitleResponse]
    categories: str
    categories_list: List[CategoryResponse] 
    license: str
    license_bool : bool
    abstract: List[AbstractResponse]
    abstract_user: List[AbstractUserResponse]
    pdf_url: List[PdfURLResponse]
    comments: List[CommentResponse]
    good: int
    bad:  int
    favorite: int

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

# --------------- User DB用処理 ---------------
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

import logging
logging.getLogger('passlib').setLevel(logging.ERROR)  # 追加

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Union[str, None] = None

class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None

class UserInDB(User):
    hashed_password: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db: Session, username: str):
    return db.query(UserDB).filter(UserDB.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --------------- セキュリティエンドポイント ----------------

@app.post("/token",tags=['UserData'])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> Token:
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str
    password: str

@app.post("/users/",tags=['UserData'])
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # パスワードのハッシュ化
    hashed_password = get_password_hash(user.password)

    # 新しいユーザーを作成
    db_user = UserDB(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
    )

    # ユーザーをデータベースに追加
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return {"message": "User created successfully"}
# --------------- Paper meta DB用処理 ---------------
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
    
    # ライセンスの確認
    license_data = load_license_data()
    license = arxiv_info['license']
    license_ok = license_data.get(license, {}).get("OK", False)

    # paper_meta_dataの作成
    paper_meta = paper_meta_data(
        identifier=arxiv_info['identifier'],
        datestamp=datetime.strptime(arxiv_info['datestamp'], '%Y-%m-%d') if 'datestamp' in arxiv_info else None,
        setSpec=arxiv_info['setSpec'],
        created=datetime.strptime(arxiv_info['created'], '%Y-%m-%d') if 'created' in arxiv_info else None,
        updated=datetime.strptime(arxiv_info['updated'], '%Y-%m-%d') if 'updated' in arxiv_info else None,
        authors=authors,
        title=[],
        categories=arxiv_info['categories'],
        categories_list=categories_list,
        license=arxiv_info['license'],
        license_bool = license_ok,
        abstract=[],
        abstract_user = [],
        pdf_url = [],
        comments = [],
        good = 0,
        bad = 0,
        favorite = 0
    )

    # セッションにデータを追加
    db.add(paper_meta)
    db.commit()

    # Titleインスタンスを作成
    title = TitleMain(
        en=arxiv_info['title'].replace("\n", ""),
        ja=None,
        paper_meta_id=paper_meta.id  # paper_meta_data の id を使用
    )
    db.add(title)

    # AbstractResponseインスタンスを作成
    abstract = AbstractMain(
        en=arxiv_info['abstract'].replace("\n", ""),
        ja=None,
        paper_meta_id=paper_meta.id  # paper_meta_data の id を使用
    )
    db.add(abstract)
    
    # セッションにデータを追加
    db.commit()
    db.refresh(paper_meta)

    return paper_meta

@app.get("/arxiv/metadata/{arxiv_id}")
async def Get_Paper_Data(arxiv_id: str, db: Session = Depends(get_db)):
    """
    ArXiv IDより、自社DBのデータを参照します。存在しない場合はArXiv APIに問い合わせ実施する。
    """
    # 自社データベースに問い合わせて、指定された arxiv_id のデータを取得
    existing_paper = db.query(paper_meta_data).filter(paper_meta_data.identifier == f"oai:arXiv.org:{arxiv_id}").first()

    if existing_paper:
        # データが既に存在する場合は、そのデータを返す
        return existing_paper
    else:
        # データが存在しない場合は、Arxiv からメタデータを取得し、データベースに保存
        arxiv_info = await get_arxiv_info_async(arxiv_id)
        paper_meta = await create_paper_meta_data(arxiv_info, db)
        return paper_meta

@app.get("/papers/", response_model=List[PaperMetaDataResponse])
async def get_all_papers(db: Session = Depends(get_db)):
    """
    DBにあるすべてのデータを取得します。
    """
    papers = db.query(paper_meta_data).all()
    return papers

@app.get("/papers/search/name/{author_name}", response_model=List[PaperMetaDataResponse])
async def search_papers_by_author(author_name: str, db: Session = Depends(get_db)):
    """
    ### 著者検索 - 名前に基づいてDBから著者検索をします。完全一致する論文リストを返します。
    
    - author_name: 検索名をフルネームで入力します。苗字名前の間には必ず半角スペースを入れてください。
    """
    # 著者名が完全一致する場合を検索
    papers = db.query(paper_meta_data).join(paper_meta_data.authors).filter(
        Author.id == author_name
    ).all()
    
    return papers

@app.post("/papers/{paper_id}/comments")
async def add_comment_to_paper(paper_id: int, user_id: str, content: str, lang: str, db: Session = Depends(get_db)):
    """
    コメントを追加する
    """
    #raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="This endpoint is not implemented yet")
    paper = db.query(paper_meta_data).filter(paper_meta_data.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    comment = Comment(user_id=user_id, content=content,lang=lang, created_at=datetime.now(), paper_meta=paper)
    db.add(comment)
    db.commit()
    db.refresh(comment)

    return {"message": "Comment added successfully"}

@app.post("/papers/{paper_id}/usr_abstracts")
async def add_abstract_to_paper(
    paper_id: int,
    lang: str,
    content: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    特定の論文に対するユーザー要約を追加する
    """
    paper = db.query(paper_meta_data).filter(paper_meta_data.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    abstract_user = AbstractUser(
        user_id=current_user.username,
        lang=lang,
        like=0,
        content=content,
        created_at=datetime.now(),
        paper_meta_id=paper_id,
    )
    db.add(abstract_user)
    db.commit()
    db.refresh(abstract_user)

    return {"message": "Abstract added successfully"}

def load_license_data():
    with open('data/license.json', 'r') as f:
        return json.load(f)

@app.post("/papers/{paper_id}/translate")
async def traslate_abstract(paper_id:int,target_lang: str = "JA", db: Session = Depends(get_db)):
    """
    abstract を日本語に翻訳します。
    """
    # ライセンスデータを読み込み
    license_data = load_license_data()
    # データベースに問い合わせて、指定された arxiv_id のデータを取得
    paper = db.query(paper_meta_data).filter(paper_meta_data.id == paper_id).first()
    if paper:

        # ライセンスが許可されているか検証
        license_ok = license_data.get(paper.license, {}).get("OK", False)
        if not license_ok:
            raise HTTPException(status_code=400, detail="License not permitted for translation")
        
        if not paper.abstract[0].ja or not paper.title[0].ja:
            if not paper.abstract[0].ja:
                translation_result = await translate_text(paper.abstract[0].en, target_lang)
                if translation_result['ok']:
                    paper.abstract[0].ja = translation_result['data']
                else:
                    raise HTTPException(status_code=500, detail=translation_result['message'])
            if not paper.title[0].ja:
                translation_result = await translate_text(paper.title[0].en, target_lang)
                if translation_result['ok']:
                    paper.title[0].ja = translation_result['data']
                else:
                    raise HTTPException(status_code=500, detail=translation_result['message'])
        else:
            return paper
        # データを更新した場合
        db.commit()
        db.refresh(paper)
        return paper

    else:
        raise HTTPException(status_code=404, detail="Paper not found")

@app.post("/papers/{paper_id}/vote")
async def update_paper_vote(paper_id: int, vote_type: str, db: Session = Depends(get_db)):
    """
    指定された論文のgood数またはbad数を更新します。呼ばれた際、+1します。(認証追加後実装)
    
    - vote_type: "good" or "bad"
    """
    paper = db.query(paper_meta_data).filter(paper_meta_data.id == paper_id).first()

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    if vote_type == "good":
        paper.good = (paper.good or 0) + 1
    elif vote_type == "bad":
        paper.bad = (paper.bad or 0) + 1
    else:
        raise HTTPException(status_code=400, detail="Invalid vote type")

    db.commit()
    db.refresh(paper)

    return {"message": "Vote updated successfully"}
    
if __name__ == "__main__":
    #uvicorn API:app --reload   
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001,reload=True)