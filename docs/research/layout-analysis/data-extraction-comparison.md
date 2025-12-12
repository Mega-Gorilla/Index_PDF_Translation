# PDF抽出方式のデータ構造比較

> 関連: [Issue #38](https://github.com/Mega-Gorilla/Index_PDF_Translation/issues/38)

## 概要

本ドキュメントでは、PDF翻訳パイプラインで使用可能な3つのデータ抽出方式を比較します。

| 方式 | ライブラリ | ライセンス | 主要出力 |
|------|-----------|-----------|---------|
| 既存方式 | PyMuPDF | AGPL-3.0 | DocumentBlocks |
| pymupdf4llm (layoutなし) | pymupdf4llm | AGPL-3.0 | Markdown |
| pymupdf-layout | pymupdf-layout | PolyForm Noncommercial | ParsedDocument (JSON) |

---

## 1. 既存方式 (extract_text_coordinates_dict)

### 使用ライブラリ
- `PyMuPDF (fitz)` - AGPL-3.0

### API
```python
from index_pdf_translation.core.pdf_edit import extract_text_coordinates_dict

blocks = await extract_text_coordinates_dict(pdf_data: bytes)
```

### 返り値の構造

```
DocumentBlocks = List[PageBlocks]
PageBlocks = List[BlockInfo]
```

```python
# 返り値: List[List[dict]]
[
    # Page 0
    [
        {
            "page_no": 0,
            "block_no": 0,
            "coordinates": [131.83, 99.83, 480.17, 136.97],  # bbox (x0, y0, x1, y1)
            "text": "Chain-of-Thought Prompting...",
            "font": "NimbusRomNo9L-Medi",
            "size": 17.21
        },
        # ... more blocks
    ],
    # Page 1
    [
        # ... blocks
    ],
    # ... more pages
]
```

### 取得できるデータ

| データ | 取得可否 | 説明 |
|--------|---------|------|
| bbox座標 | ✅ | `coordinates` (x0, y0, x1, y1) |
| テキスト | ✅ | `text` |
| フォントサイズ | ✅ | `size` |
| フォント名 | ✅ | `font` |
| ページ番号 | ✅ | `page_no` |
| ブロック番号 | ✅ | `block_no` |
| 構造タグ (boxclass) | ❌ | なし |
| 目次 (TOC) | ❌ | なし |
| 画像データ | ❌ | なし |
| テーブルデータ | ❌ | なし |

### 現在のPDF変換フロー

```
PDF (bytes)
    │
    ▼
extract_text_coordinates_dict()
    │
    ▼
DocumentBlocks
    │
    ├─► remove_blocks() ─► ヒストグラム+IQR分類
    │       │
    │       ▼
    │   text_blocks, fig_blocks, removed_blocks
    │
    ├─► preprocess_translation_blocks() ─► 文境界マージ
    │
    ├─► translate_blocks() ─► 翻訳API
    │
    ├─► preprocess_write_blocks() ─► フォントサイズ調整
    │
    ├─► remove_textbox_for_pdf() ─► 元テキスト削除 (bbox使用)
    │
    ├─► write_pdf_text() ─► 翻訳テキスト挿入 (bbox使用)
    │
    └─► create_viewing_pdf() ─► 見開きPDF生成
```

---

## 2. pymupdf4llm (layoutなし)

### 使用ライブラリ
- `pymupdf4llm` - AGPL-3.0
- `PyMuPDF` - AGPL-3.0

### 有効化方法
```python
import pymupdf4llm  # pymupdf.layoutをインポートしない
```

### API

```python
# Markdown出力
md: str = pymupdf4llm.to_markdown(pdf_path, pages=[0, 1])

# ページごとの構造データ
chunks: List[dict] = pymupdf4llm.to_markdown(pdf_path, pages=[0, 1], page_chunks=True)

# to_json() は使用不可 (NotImplementedError)
```

### 返り値の構造

#### to_markdown() → str
```markdown
## **Chain-of-Thought Prompting Elicits Reasoning** **in Large Language Models**

**Jason Wei** **Xuezhi Wang** **Dale Schuurmans** **Maarten Bosma**

Google Research, Brain Team

**Abstract**

We explore how generating a _chain of thought_ —a series of intermediate reasoning
steps—significantly improves the ability of large language models...
```

#### to_markdown(page_chunks=True) → List[dict]
```python
[
    # Page 0
    {
        "metadata": {
            "format": "PDF 1.5",
            "title": "",
            "author": "",
            "creator": "LaTeX with hyperref",
            "producer": "pdfTeX-1.40.21",
            "creationDate": "D:20230112010630Z",
            "modDate": "D:20230112010630Z",
            "file_path": "paper.pdf",
            "page_count": 43,
            "page": 1
        },
        "toc_items": [],  # Page 0には目次なし
        "tables": [],
        "images": [],
        "graphics": [],
        "text": "## **Chain-of-Thought...",  # Markdown文字列
        "words": []
    },
    # Page 1
    {
        "metadata": {...},
        "toc_items": [
            [1, "1 Introduction", 2],
            [1, "2 Chain-of-Thought Prompting", 2]
        ],
        "tables": [
            {
                "bbox": [386.36, 155.51, 497.43, 244.48],
                "rows": 5,
                "columns": 5
            }
        ],
        "images": [],
        "graphics": [],
        "text": "**1** **Introduction**\n\nThe NLP landscape...",
        "words": []
    }
]
```

### 取得できるデータ

| データ | 取得可否 | 説明 |
|--------|---------|------|
| Markdown | ✅ | `text` (フォーマット済み) |
| メタデータ | ✅ | `metadata` (PDFプロパティ) |
| 目次 (TOC) | ✅ | `toc_items` [level, title, page] |
| テーブルbbox | ✅ | `tables[].bbox` |
| テーブル行列数 | ✅ | `tables[].rows`, `tables[].columns` |
| 画像情報 | △ | `images` (通常は空) |
| **テキストブロックbbox** | ❌ | **なし** |
| **構造タグ (boxclass)** | ❌ | **なし** |
| **フォント情報** | ❌ | **なし** |

### 制限事項

```
❌ to_json() は使用不可
   → NotImplementedError: Function 'to_json' is only available in PyMuPDF-Layout mode

❌ テキストブロックごとのbbox座標が取得できない
   → PDF編集（テキスト削除・挿入）には使用できない

❌ 構造タグ (boxclass) がない
   → 見出し/本文/キャプションの自動分類ができない
```

---

## 3. pymupdf-layout

### 使用ライブラリ
- `pymupdf-layout` - **PolyForm Noncommercial 1.0.0** (商用利用には商用ライセンス必要)
- `pymupdf4llm` - AGPL-3.0
- `PyMuPDF` - AGPL-3.0

### 有効化方法
```python
import pymupdf.layout  # ← これを先にインポート（必須）
import pymupdf4llm
```

### API

```python
# JSON出力 (ParsedDocument)
json_str: str = pymupdf4llm.to_json(pdf_path, pages=[0, 1])
data: dict = json.loads(json_str)

# Markdown出力 (高精度)
md: str = pymupdf4llm.to_markdown(pdf_path, pages=[0, 1])

# Text出力
text: str = pymupdf4llm.to_text(pdf_path, pages=[0, 1])
```

### 返り値の構造

#### to_json() → ParsedDocument (JSON)

```python
{
    # ドキュメント情報
    "filename": "tests/pdf_sample/2201.11903v6.pdf",
    "page_count": 43,
    "from_bytes": False,

    # オプション設定
    "image_dpi": 150,
    "image_format": "png",
    "image_path": "",
    "use_ocr": False,
    "force_text": True,
    "embed_images": False,
    "write_images": False,

    # メタデータ
    "metadata": {
        "format": "PDF 1.5",
        "title": "",
        "author": "",
        "creator": "LaTeX with hyperref",
        "producer": "pdfTeX-1.40.21",
        "creationDate": "D:20230112010630Z",
        "modDate": "D:20230112010630Z"
    },

    # 目次 (TOC)
    "toc": [
        [1, "1 Introduction", 2],
        [1, "2 Chain-of-Thought Prompting", 2],
        [1, "3 Arithmetic Reasoning", 3],
        [2, "3.1 Experimental Setup", 3],
        [2, "3.2 Results", 4],
        # ... 35 items
    ],

    # ページデータ
    "pages": [
        {
            "page_number": 1,  # 1-indexed
            "width": 612.0,
            "height": 792.0,
            "full_ocred": False,
            "text_ocred": False,
            "fulltext": [...],
            "words": [...],
            "links": [...],

            # ★ LayoutBox配列
            "boxes": [
                {
                    # bbox座標
                    "x0": 131.83,
                    "y0": 101.17,
                    "x1": 480.17,
                    "y1": 136.57,

                    # ★ 構造タグ
                    "boxclass": "title",

                    # 画像データ (pictureの場合)
                    "image": null,

                    # テーブルデータ (tableの場合)
                    "table": null,

                    # テキストライン詳細
                    "textlines": [
                        {
                            "bbox": [131.83, 101.15, 480.17, 116.59],
                            "spans": [
                                {
                                    "size": 17.21,
                                    "flags": 20,
                                    "font": "NimbusRomNo9L-Medi",
                                    "color": 0,
                                    "text": "Chain-of-Thought Prompting..."
                                }
                            ]
                        }
                    ]
                },
                # ... more boxes
            ]
        }
    ]
}
```

### LayoutBox (boxes[]) の詳細

```python
{
    # bbox座標
    "x0": float,  # 左
    "y0": float,  # 上
    "x1": float,  # 右
    "y1": float,  # 下

    # ★ 構造タグ (重要)
    "boxclass": str,  # "title", "text", "section-header", "caption", etc.

    # 画像データ (boxclass == "picture" の場合)
    "image": bytes | str | null,

    # テーブルデータ (boxclass == "table" の場合)
    "table": dict | null,

    # テキストライン (テキスト系boxclassの場合)
    "textlines": [
        {
            "bbox": [x0, y0, x1, y1],
            "spans": [
                {
                    "size": float,      # フォントサイズ
                    "flags": int,       # フォントフラグ
                    "font": str,        # フォント名
                    "color": int,       # 色
                    "text": str         # テキスト内容
                }
            ]
        }
    ]
}
```

### boxclassの種類

| boxclass | 説明 | 翻訳対象 |
|----------|------|---------|
| `title` | 文書タイトル | ✅ |
| `section-header` | セクション見出し | ✅ |
| `text` | 本文 | ✅ |
| `caption` | 図表キャプション | ✅ |
| `list-item` | リスト項目 | ✅ |
| `footnote` | 脚注 | △ |
| `page-header` | ページヘッダー | ❌ |
| `page-footer` | ページフッター | ❌ |
| `picture` | 画像 | ❌ |
| `table` | テーブル | ❌ |
| `table-fallback` | テーブル（フォールバック） | ❌ |
| `formula` | 数式 | ❌ |

### 取得できるデータ

| データ | 取得可否 | 説明 |
|--------|---------|------|
| bbox座標 | ✅ | `boxes[].x0, y0, x1, y1` |
| テキスト | ✅ | `boxes[].textlines[].spans[].text` |
| フォントサイズ | ✅ | `boxes[].textlines[].spans[].size` |
| フォント名 | ✅ | `boxes[].textlines[].spans[].font` |
| **構造タグ (boxclass)** | ✅ | `boxes[].boxclass` |
| 目次 (TOC) | ✅ | `toc` |
| 画像データ | ✅ | `boxes[].image` (write_images=True時) |
| テーブルデータ | ✅ | `boxes[].table` |
| メタデータ | ✅ | `metadata` |
| ページサイズ | ✅ | `pages[].width, height` |

---

## 比較サマリー

### データ取得能力

| データ | 既存方式 | pymupdf4llm (layoutなし) | pymupdf-layout |
|--------|---------|------------------------|----------------|
| bbox座標 | ✅ | ❌ (テーブルのみ) | ✅ |
| テキスト | ✅ | ✅ (Markdown) | ✅ |
| フォントサイズ | ✅ | ❌ | ✅ |
| フォント名 | ✅ | ❌ | ✅ |
| **構造タグ** | ❌ | ❌ | ✅ |
| 目次 (TOC) | ❌ | ✅ | ✅ |
| 画像 | ❌ | △ | ✅ |
| テーブル | ❌ | △ (bboxのみ) | ✅ |
| メタデータ | ❌ | ✅ | ✅ |

### ライセンス

| 方式 | ライセンス | 商用利用 |
|------|-----------|---------|
| 既存方式 | AGPL-3.0 | ✅ (ソース公開必要) |
| pymupdf4llm (layoutなし) | AGPL-3.0 | ✅ (ソース公開必要) |
| pymupdf-layout | PolyForm Noncommercial | ❌ (商用ライセンス必要) |

### PDF翻訳に必要な機能

| 機能 | 既存方式 | pymupdf4llm (layoutなし) | pymupdf-layout |
|------|---------|------------------------|----------------|
| テキスト抽出 | ✅ | ✅ | ✅ |
| 翻訳テキスト挿入 | ✅ (bbox使用) | ❌ (bboxなし) | ✅ (bbox使用) |
| ブロック分類 | △ (ヒストグラム+IQR) | ❌ | ✅ (boxclass) |
| Markdown生成 | ❌ | ✅ | ✅ |

---

## 結論

### pymupdf-layoutなしで実装する場合

**既存方式 (extract_text_coordinates_dict)** を維持する必要があります。

理由:
1. `pymupdf4llm (layoutなし)`はテキストブロックのbbox座標を取得できない
2. PDF編集（テキスト削除・挿入）にはbbox座標が必須
3. 既存方式はbbox、フォント情報を取得可能

**改善案**:
- 既存のヒストグラム+IQR分類に、ルールベースの検出を追加
- 位置ベース（ヘッダー/フッター）
- パターンマッチ（セクション番号、図表キャプション）
- フォントサイズベース（見出し検出）

### pymupdf-layoutを使用する場合

**ParsedDocument (to_json)** を使用することで:
- boxclassによる正確なブロック分類
- 統一されたデータ構造
- Markdown生成の高品質化

ただし:
- PolyForm Noncommercialライセンス（商用利用には商用ライセンス必要）
- `import pymupdf.layout`を先にインポートする必要あり
