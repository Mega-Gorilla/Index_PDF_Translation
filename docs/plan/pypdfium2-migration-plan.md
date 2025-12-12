# PyMuPDF → pypdfium2 移行計画

**作成日**: 2025-12-12
**関連Issue**: #42（マージ済み）
**目的**: AGPLライセンス制約を解除し、Apache 2.0ライセンスへ移行

---

## 目次

1. [概要](#1-概要)
2. [現状分析](#2-現状分析)
3. [移行対象関数](#3-移行対象関数)
4. [実装フェーズ](#4-実装フェーズ)
5. [APIマッピング](#5-apiマッピング)
6. [リスクと対策](#6-リスクと対策)
7. [テスト計画](#7-テスト計画)
8. [マイルストーン](#8-マイルストーン)

---

## 1. 概要

### 1.1 背景

pypdfium2のテキスト削除・挿入機能が実用的であることが検証済み（PR #42）。
これにより、PyMuPDF（AGPL-3.0）をpypdfium2（Apache-2.0）で完全に代替可能。

### 1.2 目標

- PyMuPDF依存の完全排除
- プロジェクトライセンスをAGPL-3.0からApache-2.0へ変更
- 既存機能の完全互換性維持

### 1.3 スコープ

| 対象 | 内容 |
|------|------|
| 対象ファイル | `src/index_pdf_translation/core/pdf_edit.py` |
| 影響範囲 | PDF処理全般（抽出・削除・挿入・結合） |
| 非対象 | 翻訳ロジック、NLP処理、CLI |

---

## 2. 現状分析

### 2.1 PyMuPDF使用箇所

`pdf_edit.py`（863行）で以下の関数がPyMuPDFを使用：

| 関数名 | 行数 | 主な用途 | 移行難易度 |
|--------|------|---------|:----------:|
| `extract_text_coordinates_blocks` | 53-96 | テキスト抽出（ブロック形式） | ⭐ 低 |
| `extract_text_coordinates_dict` | 99-146 | テキスト抽出（詳細形式） | ⭐ 低 |
| `remove_textbox_for_pdf` | 353-378 | テキスト削除（墨消し） | ⭐⭐ 中 |
| `pdf_draw_blocks` | 381-428 | デバッグ枠描画 | ⭐⭐ 中 |
| `preprocess_write_blocks` | 474-575 | フォントメトリクス計算 | ⭐⭐ 中 |
| `write_pdf_text` | 578-650 | テキスト挿入 | ⭐⭐⭐ 高 |
| `write_logo_data` | 653-703 | ロゴ挿入 | ⭐⭐ 中 |
| `create_viewing_pdf` | 706-755 | 見開きPDF作成 | ⭐⭐⭐ 高 |
| `create_debug_pdf` | 821-862 | デバッグPDF作成 | ⭐⭐ 中 |

### 2.2 PyMuPDF API使用状況

```python
# 文書操作
fitz.open(stream=data, filetype="pdf")
document.load_page(page_num)
document.close()
doc.save(buffer, garbage=4, deflate=True, clean=True)

# テキスト抽出
page.get_text("blocks")
page.get_text("dict")

# テキスト削除
page.add_redact_annot(rect)
page.apply_redactions()

# テキスト挿入
page.insert_font(fontname, fontfile)
page.insert_textbox(rect, text, fontsize, fontname, ...)

# 描画
page.draw_rect(rect, color, width, fill, fill_opacity)
page.insert_image(rect, filename)

# ページ操作
doc.new_page(width, height)
page.show_pdf_page(rect, src_doc, page_num)
doc.insert_pdf(src_doc)
doc.set_pagelayout("TwoPageLeft")

# フォント
fitz.Font(fontname)
fitz.Font("F0", fontfile)
font.text_length(text, fontsize)

# ユーティリティ
fitz.Rect(coordinates)
```

---

## 3. 移行対象関数

### 3.1 Phase 1: 基本機能（必須）

#### 3.1.1 テキスト抽出

**現在**:
```python
page.get_text("dict")  # 詳細情報付き
page.get_text("blocks")  # ブロック形式
```

**pypdfium2**:
```python
textpage = page.get_textpage()
# 文字単位での抽出
for i in range(textpage.count_chars()):
    char = textpage.get_char(i)
    bbox = textpage.get_charbox(i)
```

#### 3.1.2 テキスト削除

**現在**:
```python
page.add_redact_annot(rect)
page.apply_redactions()
```

**pypdfium2** (検証済み):
```python
for obj in page.get_objects(filter=[FPDF_PAGEOBJ_TEXT]):
    bounds = obj.get_bounds()
    if overlaps(target_rect, bounds):
        page.remove_obj(obj)
page.gen_content()
```

#### 3.1.3 テキスト挿入

**現在**:
```python
page.insert_font(fontname="F0", fontfile=font_path)
page.insert_textbox(rect, text, fontsize, fontname, ...)
```

**pypdfium2** (検証済み):
```python
font_handle = pdfium.raw.FPDFText_LoadFont(doc.raw, font_data, ...)
text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(doc.raw, font_handle, fontsize)
pdfium.raw.FPDFText_SetText(text_obj, text_widestring)
pdfium.raw.FPDFPageObj_Transform(text_obj, ...)
pdfium.raw.FPDFPage_InsertObject(page.raw, text_obj)
page.gen_content()
```

### 3.2 Phase 2: 補助機能

#### 3.2.1 デバッグ描画

**現在**:
```python
page.draw_rect(rect, color, width, fill, fill_opacity)
```

**pypdfium2**:
```python
# パスオブジェクトを作成
path_obj = pdfium.raw.FPDFPageObj_CreateNewPath(x, y)
pdfium.raw.FPDFPath_LineTo(path_obj, ...)
pdfium.raw.FPDFPageObj_SetStrokeColor(path_obj, r, g, b, a)
pdfium.raw.FPDFPage_InsertObject(page.raw, path_obj)
```

#### 3.2.2 画像挿入

**現在**:
```python
page.insert_image(rect, filename=logo_path)
```

**pypdfium2**:
```python
image = pdfium.PdfImage.new(pdf)
image.load_jpeg(path)  # or load_png, etc.
image.set_matrix(matrix)
page.insert_obj(image)
```

### 3.3 Phase 3: 高度な機能

#### 3.3.1 見開きPDF作成

**現在**:
```python
new_page.show_pdf_page(rect, src_doc, page_num)
```

**pypdfium2**:
```python
# ページをXObjectとしてインポート
xobject = pdf.page_as_xobject(page_index)
# または画像としてレンダリングして挿入
bitmap = page.render(scale=2)
```

#### 3.3.2 PDF結合

**現在**:
```python
new_doc.insert_pdf(src_doc)
```

**pypdfium2**:
```python
# import_pages を使用
dst_pdf.import_pages(src_pdf, page_indices)
```

---

## 4. 実装フェーズ

### 4.1 Phase 1: 新規ラッパーモジュール作成

**目標**: pypdfium2のraw APIをラップした使いやすいモジュール

**新規ファイル**: `src/index_pdf_translation/core/pdfium_wrapper.py`

```python
"""
pypdfium2ラッパーモジュール

PyMuPDF互換のAPIを提供し、既存コードの移行を容易にする。
"""

import ctypes
import pypdfium2 as pdfium
from typing import Optional, Iterator

class PdfiumDocument:
    """PDF文書クラス"""

    def __init__(self, data: bytes):
        self._pdf = pdfium.PdfDocument(data)

    def __len__(self) -> int:
        return len(self._pdf)

    def __getitem__(self, index: int) -> 'PdfiumPage':
        return PdfiumPage(self._pdf[index], self)

    def save(self) -> bytes:
        from io import BytesIO
        buf = BytesIO()
        self._pdf.save(buf)
        return buf.getvalue()

    def close(self):
        self._pdf.close()


class PdfiumPage:
    """PDFページクラス"""

    def __init__(self, page, doc: PdfiumDocument):
        self._page = page
        self._doc = doc

    def get_text_blocks(self) -> list[dict]:
        """テキストブロックを取得（PyMuPDF互換形式）"""
        # 実装

    def remove_text_in_rect(self, rect: tuple) -> int:
        """矩形内のテキストを削除"""
        # 実装

    def insert_text(self, text: str, pos: tuple, font_path: str, font_size: float):
        """テキストを挿入"""
        # 実装


class PdfiumFont:
    """フォントクラス"""

    def __init__(self, font_path: str, doc: PdfiumDocument):
        # 実装

    def text_length(self, text: str, font_size: float) -> float:
        """テキスト幅を計算"""
        # 実装
```

### 4.2 Phase 2: 既存関数の移行

**対象**: `pdf_edit.py`の各関数を順次移行

**移行順序**:
1. `extract_text_coordinates_dict` - テキスト抽出（基盤）
2. `remove_textbox_for_pdf` - テキスト削除（検証済み）
3. `write_pdf_text` - テキスト挿入（検証済み）
4. `preprocess_write_blocks` - フォントメトリクス
5. `write_logo_data` - ロゴ挿入
6. `pdf_draw_blocks` - デバッグ描画
7. `create_viewing_pdf` - 見開きPDF
8. `create_debug_pdf` - デバッグPDF
9. `extract_text_coordinates_blocks` - 簡易抽出

### 4.3 Phase 3: テスト・最適化

- 既存テストケースの実行
- パフォーマンス比較
- エッジケース対応

### 4.4 Phase 4: クリーンアップ

- PyMuPDF依存の削除（pyproject.toml）
- ライセンス変更（AGPL-3.0 → Apache-2.0）
- ドキュメント更新

---

## 5. APIマッピング

### 5.1 文書操作

| PyMuPDF | pypdfium2 |
|---------|-----------|
| `fitz.open(stream=data)` | `pdfium.PdfDocument(data)` |
| `len(doc)` | `len(pdf)` |
| `doc[i]` | `pdf[i]` |
| `doc.load_page(i)` | `pdf[i]` |
| `doc.close()` | `pdf.close()` |
| `doc.save(buf)` | `pdf.save(buf)` |
| `doc.new_page(w, h)` | `pdf.new_page(w, h)` |

### 5.2 テキスト操作

| PyMuPDF | pypdfium2 |
|---------|-----------|
| `page.get_text("dict")` | `page.get_textpage()` + iteration |
| `page.get_text("blocks")` | Custom implementation |
| `page.add_redact_annot()` | `page.remove_obj()` |
| `page.apply_redactions()` | `page.gen_content()` |
| `page.insert_textbox()` | Raw API text object creation |

### 5.3 フォント操作

| PyMuPDF | pypdfium2 |
|---------|-----------|
| `fitz.Font(name)` | `FPDFText_LoadStandardFont()` |
| `fitz.Font(name, path)` | `FPDFText_LoadFont()` |
| `font.text_length()` | `FPDFFont_GetGlyphWidth()` based calculation |
| `page.insert_font()` | Font loaded per document |

### 5.4 描画操作

| PyMuPDF | pypdfium2 |
|---------|-----------|
| `page.draw_rect()` | Path object creation |
| `page.insert_image()` | `PdfImage` + `page.insert_obj()` |
| `fitz.Rect()` | Tuple `(x0, y0, x1, y1)` |

### 5.5 ページ操作

| PyMuPDF | pypdfium2 |
|---------|-----------|
| `page.rect` | `(page.get_width(), page.get_height())` |
| `page.show_pdf_page()` | `pdf.page_as_xobject()` or render+insert |
| `doc.insert_pdf()` | `pdf.import_pages()` |
| `doc.set_pagelayout()` | Raw API catalog modification |

---

## 6. リスクと対策

### 6.1 技術的リスク

| リスク | 影響度 | 対策 |
|--------|:------:|------|
| テキストボックス自動改行なし | 高 | 独自実装（文字幅計算＋改行処理） |
| フォントメトリクス精度 | 中 | harfbuzz連携で対応 |
| PDF/A非対応 | 低 | 現状不要、将来検討 |
| 複雑なPDFレイアウト | 中 | 十分なテストケース |

### 6.2 互換性リスク

| リスク | 影響度 | 対策 |
|--------|:------:|------|
| 出力PDF品質の差異 | 中 | ビジュアル比較テスト |
| 処理速度の変化 | 低 | ベンチマーク実施 |
| メモリ使用量 | 低 | プロファイリング |

### 6.3 対策: フォールバック機構

移行初期は両方のバックエンドを切り替え可能に：

```python
PDF_BACKEND = os.environ.get("PDF_BACKEND", "pypdfium2")

if PDF_BACKEND == "pymupdf":
    from .pdf_edit_pymupdf import *
else:
    from .pdf_edit_pypdfium2 import *
```

---

## 7. テスト計画

### 7.1 単体テスト

各関数の動作確認：

```python
# tests/test_pdfium_wrapper.py

def test_text_extraction():
    """テキスト抽出が正しく動作すること"""

def test_text_removal():
    """テキスト削除が正しく動作すること"""

def test_text_insertion():
    """テキスト挿入が正しく動作すること"""

def test_japanese_text():
    """日本語テキストが正しく処理されること"""
```

### 7.2 統合テスト

エンドツーエンドの翻訳パイプライン：

```python
def test_full_translation_pipeline():
    """PDFの翻訳パイプライン全体が動作すること"""
```

### 7.3 比較テスト

PyMuPDF版との出力比較：

```python
def test_output_comparison():
    """PyMuPDF版と同等の出力が得られること"""
```

### 7.4 テストPDFセット

| PDF | 特徴 | テスト観点 |
|-----|------|-----------|
| sample_llama.pdf | 学術論文 | 標準的なレイアウト |
| sample_cot.pdf | 複雑なレイアウト | 多段組、図表 |
| 日本語PDF | 日本語テキスト | CIDフォント対応 |
| 数式含むPDF | LaTeX生成 | 特殊文字 |

---

## 8. マイルストーン

### Milestone 1: ラッパーモジュール完成
- [ ] `pdfium_wrapper.py`作成
- [ ] 基本クラス実装（Document, Page, Font）
- [ ] 単体テスト

### Milestone 2: テキスト処理移行
- [ ] `extract_text_coordinates_dict`移行
- [ ] `remove_textbox_for_pdf`移行
- [ ] `write_pdf_text`移行
- [ ] 統合テスト

### Milestone 3: 補助機能移行
- [ ] `preprocess_write_blocks`移行
- [ ] `write_logo_data`移行
- [ ] `pdf_draw_blocks`移行

### Milestone 4: 高度な機能移行
- [ ] `create_viewing_pdf`移行
- [ ] `create_debug_pdf`移行
- [ ] `extract_text_coordinates_blocks`移行

### Milestone 5: クリーンアップ
- [ ] PyMuPDF依存削除
- [ ] ライセンス変更
- [ ] ドキュメント更新
- [ ] リリース

---

## 参考資料

- [pypdfium2 GitHub](https://github.com/pypdfium2-team/pypdfium2)
- [PDFium API Reference](https://pdfium.googlesource.com/pdfium/+/refs/heads/main/public/)
- [pypdfium2検証結果](../research/pymupdf-alternative-investigation.md)
- [検証スクリプト](../../tests/evaluation/verify_pypdfium2_*.py)
