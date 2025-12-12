# PyMuPDF代替ライブラリ調査レポート

**作成日**: 2025-12-12
**関連Issue**: #40
**目的**: AGPLライセンス制約を回避するため、PyMuPDFの代替手段を調査

---

## 目次

1. [背景と動機](#1-背景と動機)
2. [現在のアーキテクチャ](#2-現在のアーキテクチャ)
3. [代替ライブラリ調査](#3-代替ライブラリ調査)
4. [アーキテクチャ選択肢](#4-アーキテクチャ選択肢)
5. [技術的課題](#5-技術的課題)
6. [推奨アプローチ](#6-推奨アプローチ)
7. [次のステップ](#7-次のステップ)
8. [参考文献](#8-参考文献)

---

## 1. 背景と動機

### 1.1 現状の問題

本プロジェクトは現在 **AGPL-3.0** ライセンスで公開されているが、これは主に **PyMuPDF (fitz)** の依存関係に起因する。

| ライブラリ | ライセンス | 影響 |
|-----------|-----------|------|
| PyMuPDF | **AGPL-3.0** | プロジェクト全体をAGPLに強制 |
| PP-DocLayout | Apache 2.0 | 制限なし |
| PaddleOCR | Apache 2.0 | 制限なし |

### 1.2 目標

PyMuPDFを排除し、**Apache 2.0** または同等の寛容なライセンスのライブラリのみで構成することで、以下を実現：

- **商用ライセンス販売の容易化**
- **クローズドソース派生物の許可**
- **ライセンス条件の簡素化**

---

## 2. 現在のアーキテクチャ

### 2.1 PDF翻訳ワークフロー

```
[入力PDF]
    │
    ├─① テキスト抽出（bbox座標付き）──→ PyMuPDF
    │
    ├─② レイアウト分類 ──────────────→ (新規: PP-DocLayout)
    │
    ├─③ テキスト削除（墨消し）────────→ PyMuPDF ⚠️ 代替困難
    │
    ├─④ 翻訳処理 ─────────────────→ Google/DeepL/OpenAI
    │
    ├─⑤ テキスト挿入 ──────────────→ PyMuPDF
    │
    └─⑥ 出力PDF生成 ──────────────→ PyMuPDF

[出力PDF（原文＋翻訳の並列表示）]
```

### 2.2 PyMuPDFの使用箇所

| 機能 | 使用メソッド | 代替難易度 |
|------|------------|-----------|
| テキスト抽出 | `get_text("dict")` | ⭐ 容易 |
| ページ画像化 | `get_pixmap()` | ⭐ 容易 |
| テキスト削除 | `add_redact_annot()` + `apply_redactions()` | ⭐⭐⭐ 困難 |
| テキスト挿入 | `insert_textbox()` | ⭐⭐ 中程度 |
| PDF結合/分割 | `insert_pdf()` | ⭐⭐ 中程度 |

---

## 3. 代替ライブラリ調査

### 3.1 テキスト抽出（bbox付き）

| ライブラリ | ライセンス | bbox取得 | 精度 | 備考 |
|-----------|-----------|:--------:|:----:|------|
| **pdftext** | **Apache 2.0** | ✅ | 高 | pypdfium2ベース、PyMuPDF同等 |
| pypdfium2 | Apache 2.0 | ✅ | 高 | Google PDFiumバインディング |
| pypdf | BSD-3-Clause | ⚠️ | 中 | 文字単位bbox非対応 |
| pdfplumber | MIT | ✅ | 高 | pdfminerベース |

**推奨**: **pdftext** (Apache 2.0)

```python
from pdftext.extraction import plain_text_output, dictionary_output

# bbox付きテキスト抽出
result = dictionary_output(pdf_path, sort=True, keep_chars=True)
# result["pages"][0]["blocks"][0]["bbox"] → [x1, y1, x2, y2]
```

### 3.2 PDF→画像レンダリング

| ライブラリ | ライセンス | 品質 | 速度 | 備考 |
|-----------|-----------|:----:|:----:|------|
| **pypdfium2** | **Apache 2.0** | 高 | 高速 | Google PDFium |
| pdf2image | MIT | 高 | 中 | Poppler依存 |
| Wand | MIT | 高 | 遅い | ImageMagick依存 |

**推奨**: **pypdfium2** (Apache 2.0)

```python
import pypdfium2 as pdfium

pdf = pdfium.PdfDocument(pdf_path)
for page in pdf:
    image = page.render(scale=2).to_pil()
    # PP-DocLayoutに渡す
```

### 3.3 テキスト削除（墨消し）⚠️ 最難関

| ライブラリ | ライセンス | 墨消し対応 | 品質 | 備考 |
|-----------|-----------|:--------:|:----:|------|
| **PyMuPDF** | AGPL-3.0 | ✅ 完全 | 高 | 業界標準 |
| pdf-redactor | CC0 (PD) | ⚠️ regex | 中 | pdfrwベース |
| pikepdf | MPL-2.0 | ⚠️ 複雑 | 中 | 低レベルAPI |
| pypdfium2 | Apache 2.0 | ❌ | - | 未実装 |

**重要な発見**:
Apache 2.0ライセンスで**確実なテキスト墨消し**を行えるライブラリは**存在しない**。

#### pdf-redactor (CC0)

```python
import pdf_redactor

options = pdf_redactor.RedactorOptions()
options.content_filters = [
    (re.compile(r"翻訳対象テキスト"), lambda m: "")
]
pdf_redactor.redact(input_path, output_path, options)
```

**制限**: 正規表現マッチングベースのため、bbox指定での削除不可。

### 3.4 テキスト挿入

| ライブラリ | ライセンス | テキスト挿入 | フォント対応 | 備考 |
|-----------|-----------|:----------:|:----------:|------|
| **pypdfium2** | **Apache 2.0** | ✅ | ✅ | `insert_text()` |
| reportlab | BSD | ✅ | ✅ | PDF生成専用 |
| pikepdf | MPL-2.0 | ⚠️ | ⚠️ | 低レベル |

**推奨**: **pypdfium2** (Apache 2.0)

```python
import pypdfium2 as pdfium

pdf = pdfium.PdfDocument(pdf_path)
page = pdf[0]
page.insert_text(
    text="翻訳テキスト",
    pos_x=100, pos_y=200,
    font_size=12,
    font_path="/path/to/font.ttf"
)
pdf.save(output_path)
```

---

## 4. アーキテクチャ選択肢

### 4.1 選択肢A: PyMuPDF継続（現状維持）

```
[PDF] → [PyMuPDF] → テキスト抽出/削除/挿入
          ↓
      [PP-DocLayout] → レイアウト分類
          ↓
      [出力PDF]
```

| 評価項目 | 評価 |
|---------|------|
| 実装コスト | ⭐ 低（既存） |
| 品質 | ⭐⭐⭐ 高 |
| ライセンス | ❌ AGPL-3.0強制 |

### 4.2 選択肢B: 画像オーバーレイ方式

```
[PDF] → [pypdfium2] → 画像レンダリング
          ↓
      [PP-DocLayout] → レイアウト検出
          ↓
      [pdftext] → テキスト抽出
          ↓
      [画像処理] → テキスト領域を白塗り
          ↓
      [pypdfium2/reportlab] → 翻訳テキスト描画
          ↓
      [新規PDF生成]
```

| 評価項目 | 評価 |
|---------|------|
| 実装コスト | ⭐⭐⭐ 高 |
| 品質 | ⭐⭐ 中（ラスタライズ） |
| ライセンス | ✅ Apache 2.0可能 |

**課題**:
- 元PDFがラスタライズされ、テキスト検索・コピー不可
- ファイルサイズ増大
- フォント品質低下

### 4.3 選択肢C: ハイブリッドPDF再構築

```
[PDF] → [pdftext] → テキスト抽出（座標付き）
          ↓
      [pypdfium2] → 画像抽出/レンダリング
          ↓
      [PP-DocLayout] → レイアウト分類
          ↓
      [翻訳処理]
          ↓
      [reportlab] → 新規PDF構築
          ├── 翻訳テキスト（テキストレイヤー）
          └── 元画像/図表（画像レイヤー）
```

| 評価項目 | 評価 |
|---------|------|
| 実装コスト | ⭐⭐⭐ 高 |
| 品質 | ⭐⭐ 中〜高 |
| ライセンス | ✅ Apache 2.0可能 |

**課題**:
- 複雑な実装
- 元PDFのレイアウト完全再現が困難
- 数式・表の処理が複雑

### 4.4 選択肢D: OCRベース完全再構築

```
[PDF] → [pypdfium2] → 全ページ画像化
          ↓
      [PPStructureV3] → レイアウト検出 + OCR
          ↓
      [翻訳処理]
          ↓
      [reportlab] → 新規PDF生成
```

| 評価項目 | 評価 |
|---------|------|
| 実装コスト | ⭐⭐ 中 |
| 品質 | ⭐ 低（OCR精度依存） |
| ライセンス | ✅ Apache 2.0可能 |

**課題**:
- 埋め込みテキストがあるPDFでもOCRが必要
- OCR誤認識リスク
- 元PDFの品質を完全に失う

---

## 5. 技術的課題

### 5.1 テキスト削除の難しさ

PDFのテキスト削除（墨消し）が困難な理由：

1. **PDF構造の複雑さ**: PDFはページ記述言語であり、テキストは描画命令として埋め込まれている
2. **座標系の変換**: テキストの座標系は複数の変換行列を経由
3. **フォント埋め込み**: 削除後のフォントサブセット再計算が必要
4. **透明度・重ね合わせ**: 他の要素との重なりを考慮

### 5.2 各ライブラリの墨消し実装状況

| ライブラリ | 実装方式 | 制限 |
|-----------|---------|------|
| PyMuPDF | MuPDF C APIラップ | 完全な墨消し |
| pdf-redactor | pdfrwでストリーム書き換え | 正規表現マッチのみ |
| pikepdf | QPDFでコンテンツストリーム操作 | 低レベルAPI、複雑 |

### 5.3 pypdfium2の墨消し可能性

pypdfium2のPDFium APIには `FPDFPage_RemoveObject()` が存在するが、テキストオブジェクトの特定と削除は複雑。

```python
# 理論上の実装（未検証）
import pypdfium2 as pdfium
from pypdfium2._helpers import LoopFormObject

pdf = pdfium.PdfDocument(path)
page = pdf[0]

for obj in LoopFormObject(page):
    if obj.type == pdfium.FPDF_PAGEOBJ_TEXT:
        # テキストオブジェクトを削除
        page.remove_obj(obj)
```

**要検証**: この方法が実用的かどうかは追加調査が必要。

---

## 6. 推奨アプローチ

### 6.1 短期（現実的）

**選択肢A（PyMuPDF継続）+ PP-DocLayoutハイブリッド**

```
[PDF] → [pdftext] → テキスト抽出（Apache 2.0）
          ↓
      [pypdfium2] → 画像レンダリング（Apache 2.0）
          ↓
      [PP-DocLayout] → レイアウト分類（Apache 2.0）
          ↓
      [PyMuPDF] → テキスト削除/挿入（AGPL-3.0）⚠️
          ↓
      [出力PDF]
```

**メリット**:
- 実装変更最小限
- PP-DocLayoutの恩恵を受けられる
- PyMuPDF依存を削除/挿入のみに限定

**デメリット**:
- 引き続きAGPL-3.0制約

### 6.2 中期（要プロトタイプ検証）

**pypdfium2の墨消し機能調査**

1. `FPDFPage_RemoveObject()` の実用性検証
2. テキストオブジェクト特定アルゴリズム開発
3. パフォーマンス・品質評価

**成功すれば**: 完全Apache 2.0化が可能

### 6.3 長期（代替アプローチ）

**選択肢C（ハイブリッドPDF再構築）の段階的実装**

1. 単純なテキストのみのPDFでプロトタイプ
2. 図表・数式の扱いを段階的に追加
3. 品質とパフォーマンスの最適化

---

## 7. 次のステップ

### 7.1 即座に実施可能

- [ ] pypdfium2の `FPDFPage_RemoveObject()` 検証プロトタイプ作成
- [ ] pdftext + PP-DocLayout統合のPoC
- [ ] 選択肢Bの画像オーバーレイ方式プロトタイプ

### 7.2 調査継続

- [ ] pikepdfでのテキスト削除可能性の詳細調査
- [ ] pdf-redactorのbbox対応拡張可能性
- [ ] 商用ライブラリ（Apryse等）のライセンス条件確認

### 7.3 Issue作成

PyMuPDF代替の本格検討を進めるため、以下のIssueを作成：

1. **pypdfium2墨消し機能検証** (優先度: 高)
2. **pdftext + PP-DocLayout統合** (優先度: 中)
3. **PDF再構築アーキテクチャ設計** (優先度: 低)

---

## 8. 参考文献

### ライブラリ

- [pdftext - PyPI](https://pypi.org/project/pdftext/) - Apache 2.0
- [pypdfium2 - GitHub](https://github.com/pypdfium2-team/pypdfium2) - Apache 2.0 / BSD-3-Clause
- [PaddleOCR - GitHub](https://github.com/PaddlePaddle/PaddleOCR) - Apache 2.0
- [PP-DocLayout - Hugging Face](https://huggingface.co/PaddlePaddle/PP-DocLayout-L) - Apache 2.0
- [pdf-redactor - GitHub](https://github.com/JoshData/pdf-redactor) - CC0
- [pikepdf - Documentation](https://pikepdf.readthedocs.io/) - MPL-2.0

### 技術資料

- [PDFium API Reference](https://pdfium.googlesource.com/pdfium/+/refs/heads/main/public/)
- [PDF Reference 1.7](https://www.adobe.com/content/dam/acom/en/devnet/pdf/pdfs/PDF32000_2008.pdf)

---

## 付録: ライセンス比較表

| ライセンス | 商用利用 | 派生物公開義務 | 修正部分公開 | 評価 |
|-----------|:-------:|:------------:|:----------:|------|
| **Apache 2.0** | ✅ | ❌ | ❌ | 最も寛容 |
| **BSD-3-Clause** | ✅ | ❌ | ❌ | 寛容 |
| **MIT** | ✅ | ❌ | ❌ | 寛容 |
| **CC0** | ✅ | ❌ | ❌ | パブリックドメイン |
| **MPL-2.0** | ✅ | ❌ | ✅ ファイル単位 | 中程度 |
| **LGPL-3.0** | ✅ | ❌ | ✅ ライブラリ部分 | やや制限的 |
| **AGPL-3.0** | ✅ | ✅ 全体 | ✅ 全体 | **制限的** |
