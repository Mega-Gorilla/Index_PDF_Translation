from sqlalchemy import create_engine, Column, Integer, String, DateTime, Table, ForeignKey, Boolean, UUID
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import  Session
import os,json
from datetime import datetime

SQLALCHEMY_DATABASE_URL = os.environ.get('render-db-url')
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def load_license_data():
    with open('data/license.json', 'r') as f:
        return json.load(f)
    
async def create_paper_meta_data(arxiv_info: dict, db: Session):
    """
    arxiv_info: ArXivから受信した辞書配列情報
    db: データベース
    """

    # ヘルパー関数: リストの作成
    def create_or_get(model, data, key_name='id'):
        instances = []
        for item in data:
            item_id = item if isinstance(item, str) else f"{item['forenames']} {item['keyname']}"
            db_instance = db.query(model).filter(getattr(model, key_name) == item_id).first()
            if not db_instance:
                db_instance = model(**{key_name: item_id})
            instances.append(db_instance)
        return instances

    # 著者データの作成
    authors = create_or_get(Author, arxiv_info['authors'])

    # カテゴリーリストを作成
    categories_list = create_or_get(Category, arxiv_info['categories'].split(), 'id')
    
    # ライセンスの確認
    license_data = load_license_data()
    license_ok = license_data.get(arxiv_info['license'], {}).get("OK", False)

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
        license_bool=license_ok,
        abstract=[],
        abstract_user=[],
        pdf_url=[],
        comments=[],
        good=0,
        bad=0,
        favorite=0
    )

    # paper_meta_dataとそれに関連するタイトルとアブストラクトをデータベースに追加
    db.add(paper_meta)
    db.commit()
    db.refresh(paper_meta)

    # Titleインスタンスを作成して追加
    title = TitleMain(en=arxiv_info['title'].replace("\n", ""), ja=None, paper_meta_id=paper_meta.id)
    abstract = AbstractMain(en=arxiv_info['abstract'].replace("\n", ""), ja=None, paper_meta_id=paper_meta.id)
    pdf_url = PdfURL(en=None,ja=None, paper_meta_id=paper_meta.id)
    # 一度に追加してコミット
    db.add_all([title, abstract,pdf_url])
    db.commit()

    return paper_meta

# --------------- SQL テーブル ---------------
# Arxiv論文用テーブル

# 著者用テーブル
user_authors = Table(
    "user_authors",
    Base.metadata,
    Column("paper_meta", Integer, ForeignKey("paper_meta.id")),
    Column("author_id", String, ForeignKey("authors.id")),
)

class Author(Base):
    __tablename__ = "authors"
    id = Column(String, primary_key=True, index=True)

# カテゴリーテーブル
paper_categories = Table('paper_categories', Base.metadata,
    Column('paper_id', Integer, ForeignKey('paper_meta.id')),
    Column('category_id', String, ForeignKey('categories.id'))
)

class Category(Base):
    __tablename__ = "categories"
    id = Column(String, primary_key=True, index=True)

# タイトルテーブル
class TitleMain(Base):
    __tablename__ = "title"
    paper_meta_id = Column(Integer, ForeignKey("paper_meta.id"), primary_key=True)
    en = Column(String)
    ja = Column(String)
    paper_meta = relationship("paper_meta_data", back_populates="title")

# アブストテーブル
class AbstractMain(Base):
    __tablename__ = "abstract"
    paper_meta_id = Column(Integer, ForeignKey("paper_meta.id"), primary_key=True)
    en = Column(String)
    ja = Column(String)
    paper_meta = relationship("paper_meta_data", back_populates="abstract")

# アブストユーザーテーブル
class AbstractUser(Base):
    __tablename__ = "abstract_user"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)
    lang = Column(String, index=True)
    like = Column(Integer)
    content = Column(String)
    created_at = Column(DateTime)
    paper_meta_id = Column(Integer, ForeignKey("paper_meta.id"))
    paper_meta = relationship("paper_meta_data", back_populates="abstract_user")

# コメントテーブル
class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)
    content = Column(String)
    lang = Column(String, index=True)
    created_at = Column(DateTime)
    paper_meta_id = Column(Integer, ForeignKey("paper_meta.id"))
    paper_meta = relationship("paper_meta_data", back_populates="comments")

# PDF URLテーブル
class PdfURL(Base):
    __tablename__ = "pdf_url"
    paper_meta_id = Column(Integer, ForeignKey("paper_meta.id"), primary_key=True)
    en = Column(String)
    ja = Column(String)
    paper_meta = relationship("paper_meta_data", back_populates="pdf_url")

# マスターテーブル
class paper_meta_data(Base):
    __tablename__ = "paper_meta"
    id = Column(Integer, primary_key=True, index=True)
    identifier = Column(String, index=True)
    datestamp = Column(DateTime, index=True)
    setSpec = Column(String)
    created = Column(DateTime)
    updated = Column(DateTime)
    authors = relationship("Author", secondary="user_authors")
    title = relationship("TitleMain", back_populates="paper_meta")
    categories = Column(String)
    categories_list = relationship("Category", secondary=paper_categories)
    license = Column(String)
    license_bool = Column(Boolean)
    abstract = relationship("AbstractMain", back_populates="paper_meta")
    abstract_user = relationship("AbstractUser", back_populates="paper_meta")
    pdf_url = relationship("PdfURL", back_populates="paper_meta")
    comments = relationship("Comment", back_populates="paper_meta")
    good = Column(Integer, index=True)
    bad = Column(Integer)
    favorite = Column(Integer)

# ユーザーテーブル
class UserDB(Base):
    __tablename__ = "user_db"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    lang = Column(String)
    full_name = Column(String)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)

# DeepL翻訳テーブル
class Deepl_Translate_Task(Base):
    __tablename__ = "translate_db"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID,index=True)
    arxiv_id = Column(String)
    deepl_hash_key = Column(String)
    deepl_url = Column(String)
    target_lang = Column(String)
    responce = Column(String)

# 翻訳完了データを収納するテーブル
class Translate_logs(Base):
    __tablename__ = "translate_log"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID,index=True)
    deepl_hash_key = Column(String,index=True)
    mode = Column(String)
    link = Column(String)
    datestamp = Column(DateTime, index=True)