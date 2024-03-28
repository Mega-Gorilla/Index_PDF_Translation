from sqlalchemy import create_engine, Column, Integer, String, DateTime, Table, ForeignKey, Boolean, UUID
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
import os

SQLALCHEMY_DATABASE_URL = os.environ.get('render-db-url')
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

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
    responce = Column(String)