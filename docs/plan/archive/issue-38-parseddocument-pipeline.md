# Issue #38: PyMuPDF4LLM ParsedDocumentベース翻訳パイプライン

> **関連Issue**: [#38](https://github.com/Mega-Gorilla/Index_PDF_Translation/issues/38)
> **関連調査**: [PyMuPDF4LLM評価レポート](../research/evaluations/pymupdf4llm-evaluation.md)

---

## ⚠️ アーカイブ: ライセンス上の問題により本計画は保留

### 問題の概要

本計画で使用を予定していた`pymupdf-layout`パッケージに**ライセンス上の問題**があることが判明しました。

### ライセンス調査結果

| パッケージ | ライセンス | 商用利用 |
|-----------|-----------|---------|
| PyMuPDF | AGPL-3.0 / 商用 | ✅ AGPL遵守で可 |
| pymupdf4llm | AGPL-3.0 / 商用 | ✅ AGPL遵守で可 |
| **pymupdf-layout** | **PolyForm Noncommercial 1.0.0** | ❌ 商用ライセンス必要 |

### 技術的制約

`pymupdf-layout`なしでは、本計画の核となる機能が使用できません：

| 機能 | pymupdf-layoutなし | pymupdf-layoutあり |
|------|-------------------|-------------------|
| `to_json()` (ParsedDocument) | ❌ NotImplementedError | ✅ |
| boxclass (構造タグ) | ❌ | ✅ |
| テキストブロックbbox | ❌ | ✅ |

### 結論

- **ParsedDocumentベースのアーキテクチャ**は`pymupdf-layout`が必須
- `pymupdf-layout`は**PolyForm Noncommercial**ライセンスのため、商用利用には商用ライセンスが必要
- 本プロジェクト（AGPL-3.0）での商用利用を想定する場合、別のアプローチを検討する必要がある

### 代替案

1. **既存方式の改善**: ヒストグラム+IQR分類にルールベース検出を追加
2. **他のOSSツール検討**: DocLayout-YOLO (Apache-2.0)、GROBID (Apache-2.0) 等
3. **商用ライセンス取得**: Artifexから`pymupdf-layout`の商用ライセンスを購入

### 関連ドキュメント

- [データ抽出方式比較](../research/data-extraction-comparison.md)

---

## 1. 概要

### 1.1 目的

現行のPyMuPDF直接操作ベースのアーキテクチャから、**PyMuPDF4LLM ParsedDocumentベース**のアーキテクチャへ完全移行する。

### 1.2 主要な変更点

| 項目 | 現行 | 新規 |
|------|------|------|
| データソース | PyMuPDF `page.get_text("dict")` | PyMuPDF4LLM `parse_document()` |
| 中間データ | `DocumentBlocks` (独自形式) | `ParsedDocument` (PyMuPDF4LLM標準) |
| ブロック分類 | ヒストグラム+IQR | `boxclass` (PyMuPDF4LLM) |
| 主要出力 | PDF | **Markdown** (PDFは付属) |
| レイアウト情報 | 座標のみ | 座標 + 構造タグ + TOC |

### 1.3 期待される効果

1. **メンテナンス性向上**: PyMuPDF4LLMのアップデートに追従しやすい
2. **構造認識の精度向上**: `boxclass`による明示的なブロック分類
3. **出力形式の多様化**: Markdown, PDF, JSON
4. **コードの簡素化**: 独自分類ロジックの削減

## 2. アーキテクチャ設計

### 2.1 現行アーキテクチャ

```
PDF
 │
 ▼
PyMuPDF page.get_text("dict")
 │
 ▼
DocumentBlocks (独自形式)
 │
 ├─► remove_blocks() ─► ヒストグラム+IQRスコアリング
 │                       │
 │                       ▼
 │                   分類結果 (text_blocks, fig_blocks, removed_blocks)
 │
 ├─► 翻訳処理
 │
 ▼
PyMuPDF 直接編集 (remove_textbox_for_pdf, write_pdf_text)
 │
 ▼
見開きPDF出力
```

### 2.2 新アーキテクチャ

```
PDF
 │
 ▼
PyMuPDF4LLM parse_document()
 │
 ▼
ParsedDocument (PyMuPDF4LLM標準形式)
 ├─ pages: List[PageLayout]
 │   └─ boxes: List[LayoutBox]
 │       ├─ boxclass: str (title, section-header, text, caption, footer, picture, table...)
 │       ├─ bbox: (x0, y0, x1, y1)
 │       └─ textlines: List[TextLine]
 ├─ toc: List (目次構造)
 └─ metadata: Dict
 │
 ├─► boxclassによるフィルタリング
 │   ├─ 翻訳対象: text, title, section-header, caption
 │   └─ 除外: footer, page-header, picture, table
 │
 ├─► 翻訳処理 (既存ロジック活用)
 │
 ▼
翻訳済みParsedDocument
 │
 ├─────────────────┬────────────────────┐
 ▼                 ▼                    ▼
Markdown出力   PDF出力                JSON出力
(主要出力)     (bbox座標使用)         (構造データ)
```

### 2.3 データフロー詳細

```python
# Phase 1: 抽出
parsed_doc = pymupdf4llm.parse_document(pdf_path)

# Phase 2: フィルタリング・分類
extraction_result = extract_translatable_blocks(parsed_doc)
# → TranslatableBlocks(text_blocks, heading_blocks, caption_blocks, excluded_blocks)

# Phase 3: 翻訳
translated_blocks = await translate_blocks(extraction_result.all_translatable, translator, target_lang)

# Phase 4: 出力生成
markdown_output = generate_markdown(parsed_doc, translated_blocks)
pdf_output = generate_pdf(original_pdf, parsed_doc, translated_blocks)  # bbox座標使用
```

## 3. データ構造設計

### 3.1 ParsedDocument構造 (PyMuPDF4LLM)

```python
# PyMuPDF4LLMの出力構造 (to_json())
{
    "filename": "paper.pdf",
    "page_count": 10,
    "toc": [
        [1, "1 Introduction", 2],
        [1, "2 Methods", 5],
        [2, "2.1 Data Collection", 5],
        ...
    ],
    "pages": [
        {
            "page_number": 0,
            "width": 612.0,
            "height": 792.0,
            "boxes": [
                {
                    "boxclass": "title",
                    "x0": 131.83, "y0": 101.17, "x1": 480.17, "y1": 136.57,
                    "textlines": [
                        {
                            "spans": [
                                {"text": "Paper Title", "font": "TimesNewRoman-Bold", "size": 17.0}
                            ]
                        }
                    ]
                },
                {
                    "boxclass": "text",
                    "x0": 72.0, "y0": 200.0, "x1": 540.0, "y1": 300.0,
                    "textlines": [...]
                },
                ...
            ]
        }
    ]
}
```

### 3.2 新規データクラス

```python
# src/index_pdf_translation/core/models.py

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class BlockType(Enum):
    """ブロックタイプ（PyMuPDF4LLM boxclassに対応）"""
    TITLE = "title"
    SECTION_HEADER = "section-header"
    TEXT = "text"
    CAPTION = "caption"
    FOOTER = "footer"
    PAGE_HEADER = "page-header"
    PICTURE = "picture"
    TABLE = "table"
    LIST = "list"
    FORMULA = "formula"
    UNKNOWN = "unknown"

@dataclass
class TextBlock:
    """翻訳対象テキストブロック"""
    page_num: int
    block_index: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    text: str
    block_type: BlockType
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    heading_level: Optional[int] = None  # 1-6 for headings

@dataclass
class TranslatedBlock(TextBlock):
    """翻訳済みブロック"""
    original_text: str = ""
    translated_text: str = ""

@dataclass
class ExtractionResult:
    """ParsedDocumentからの抽出結果"""
    # 翻訳対象
    text_blocks: list[TextBlock] = field(default_factory=list)
    heading_blocks: list[TextBlock] = field(default_factory=list)
    caption_blocks: list[TextBlock] = field(default_factory=list)

    # 翻訳対象外（座標情報は保持）
    figure_blocks: list[TextBlock] = field(default_factory=list)
    table_blocks: list[TextBlock] = field(default_factory=list)
    excluded_blocks: list[TextBlock] = field(default_factory=list)

    # メタデータ
    toc: list[tuple[int, str, int]] = field(default_factory=list)
    page_count: int = 0

    @property
    def all_translatable(self) -> list[TextBlock]:
        """全翻訳対象ブロック"""
        return self.text_blocks + self.heading_blocks + self.caption_blocks

@dataclass
class TranslationOutput:
    """翻訳出力"""
    markdown: str
    pdf: Optional[bytes] = None
    json_data: Optional[dict] = None
    debug_pdf: Optional[bytes] = None
```

### 3.3 boxclass → BlockType マッピング

```python
BOXCLASS_MAPPING = {
    # 翻訳対象
    "title": BlockType.TITLE,
    "section-header": BlockType.SECTION_HEADER,
    "text": BlockType.TEXT,
    "caption": BlockType.CAPTION,
    "list": BlockType.LIST,

    # 翻訳対象外
    "footer": BlockType.FOOTER,
    "page-header": BlockType.PAGE_HEADER,
    "picture": BlockType.PICTURE,
    "table": BlockType.TABLE,
    "formula": BlockType.FORMULA,
}

TRANSLATABLE_TYPES = {
    BlockType.TITLE,
    BlockType.SECTION_HEADER,
    BlockType.TEXT,
    BlockType.CAPTION,
    BlockType.LIST,
}

SKIP_TYPES = {
    BlockType.FOOTER,
    BlockType.PAGE_HEADER,
    BlockType.PICTURE,
    BlockType.TABLE,
    BlockType.FORMULA,
}
```

## 4. 実装計画

### Phase 1: ParsedDocument抽出モジュール

**目標**: PyMuPDF4LLMからParsedDocumentを取得し、翻訳対象ブロックを抽出

**新規ファイル**: `src/index_pdf_translation/core/extractor.py`

```python
# src/index_pdf_translation/core/extractor.py

import json
from pathlib import Path
from typing import Union
import pymupdf4llm

from index_pdf_translation.core.models import (
    TextBlock, BlockType, ExtractionResult,
    BOXCLASS_MAPPING, TRANSLATABLE_TYPES
)

async def extract_from_pdf(
    pdf_input: Union[bytes, Path, str],
) -> tuple[dict, ExtractionResult]:
    """
    PDFからParsedDocumentを抽出し、翻訳対象ブロックを分類する。

    Args:
        pdf_input: PDFバイナリデータまたはファイルパス

    Returns:
        (parsed_document_dict, extraction_result)
    """
    # PyMuPDF4LLMでパース
    if isinstance(pdf_input, bytes):
        # バイナリの場合は一時ファイルに書き出し
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_input)
            temp_path = f.name
        json_output = pymupdf4llm.to_json(temp_path)
        Path(temp_path).unlink()
    else:
        json_output = pymupdf4llm.to_json(str(pdf_input))

    parsed_doc = json.loads(json_output)

    # ブロック抽出・分類
    result = _classify_blocks(parsed_doc)

    return parsed_doc, result

def _classify_blocks(parsed_doc: dict) -> ExtractionResult:
    """ParsedDocumentからブロックを分類"""
    result = ExtractionResult(
        toc=parsed_doc.get("toc", []),
        page_count=parsed_doc.get("page_count", 0),
    )

    for page in parsed_doc.get("pages", []):
        page_num = page.get("page_number", 0)

        for idx, box in enumerate(page.get("boxes", [])):
            boxclass = box.get("boxclass", "unknown")
            block_type = BOXCLASS_MAPPING.get(boxclass, BlockType.UNKNOWN)

            # テキスト抽出
            text = _extract_text_from_box(box)
            if not text.strip():
                continue

            # フォント情報抽出
            font_size, font_name = _extract_font_info(box)

            # TextBlock作成
            block = TextBlock(
                page_num=page_num,
                block_index=idx,
                bbox=(box.get("x0", 0), box.get("y0", 0),
                      box.get("x1", 0), box.get("y1", 0)),
                text=text,
                block_type=block_type,
                font_size=font_size,
                font_name=font_name,
                heading_level=_detect_heading_level(boxclass),
            )

            # 分類
            if block_type in TRANSLATABLE_TYPES:
                if block_type == BlockType.TITLE:
                    result.heading_blocks.append(block)
                elif block_type == BlockType.SECTION_HEADER:
                    result.heading_blocks.append(block)
                elif block_type == BlockType.CAPTION:
                    result.caption_blocks.append(block)
                else:
                    result.text_blocks.append(block)
            elif block_type == BlockType.PICTURE:
                result.figure_blocks.append(block)
            elif block_type == BlockType.TABLE:
                result.table_blocks.append(block)
            else:
                result.excluded_blocks.append(block)

    return result

def _extract_text_from_box(box: dict) -> str:
    """ボックスからテキストを抽出"""
    texts = []
    for textline in box.get("textlines", []) or []:
        for span in textline.get("spans", []) or []:
            texts.append(span.get("text", ""))
    return " ".join(texts)

def _extract_font_info(box: dict) -> tuple[Optional[float], Optional[str]]:
    """フォント情報を抽出"""
    for textline in box.get("textlines", []) or []:
        for span in textline.get("spans", []) or []:
            return span.get("size"), span.get("font")
    return None, None

def _detect_heading_level(boxclass: str) -> Optional[int]:
    """見出しレベルを検出"""
    if boxclass == "title":
        return 1
    elif boxclass == "section-header":
        return 2
    return None
```

### Phase 2: 翻訳処理の適応

**目標**: 既存の翻訳ロジックをTextBlockベースに適応

**変更ファイル**: `src/index_pdf_translation/core/translate.py`

```python
# 新規関数追加

async def translate_text_blocks(
    blocks: list[TextBlock],
    translator: "TranslatorBackend",
    target_lang: str,
) -> list[TranslatedBlock]:
    """
    TextBlockリストを翻訳する。

    既存のchunk_texts_for_translation、translate_chunk_with_retryを活用。
    """
    if not blocks:
        return []

    # テキスト抽出
    texts = [block.text for block in blocks]

    # チャンク分割（既存ロジック）
    chunks = chunk_texts_for_translation(texts, BLOCK_SEPARATOR, MAX_CHUNK_SIZE)

    # 翻訳（既存ロジック）
    translated_texts: list[str] = []
    for chunk in chunks:
        combined = BLOCK_SEPARATOR.join(chunk)
        translated = await translate_chunk_with_retry(translator, combined, target_lang)
        translated_texts.extend(translated.split(BLOCK_SEPARATOR))

    # TranslatedBlock作成
    result = []
    for block, translated_text in zip(blocks, translated_texts):
        result.append(TranslatedBlock(
            page_num=block.page_num,
            block_index=block.block_index,
            bbox=block.bbox,
            text=translated_text.strip(),
            block_type=block.block_type,
            font_size=block.font_size,
            font_name=block.font_name,
            heading_level=block.heading_level,
            original_text=block.text,
            translated_text=translated_text.strip(),
        ))

    return result
```

### Phase 3: Markdown出力生成

**目標**: 翻訳済みParsedDocumentからMarkdownを生成

**新規ファイル**: `src/index_pdf_translation/core/markdown_generator.py`

```python
# src/index_pdf_translation/core/markdown_generator.py

from index_pdf_translation.core.models import (
    TranslatedBlock, BlockType, ExtractionResult
)

def generate_markdown(
    extraction_result: ExtractionResult,
    translated_blocks: list[TranslatedBlock],
    include_toc: bool = True,
    include_images: bool = True,
    image_base_path: str = "images",
) -> str:
    """
    翻訳結果からMarkdownを生成する。

    Args:
        extraction_result: 抽出結果（構造情報）
        translated_blocks: 翻訳済みブロック
        include_toc: 目次を含める
        include_images: 画像参照を含める
        image_base_path: 画像ファイルのベースパス

    Returns:
        Markdown文字列
    """
    lines: list[str] = []

    # 目次生成（オプション）
    if include_toc and extraction_result.toc:
        lines.append("## 目次\n")
        for level, title, page in extraction_result.toc:
            indent = "  " * (level - 1)
            lines.append(f"{indent}- {title}")
        lines.append("\n---\n")

    # 翻訳ブロックをページ順・位置順にソート
    sorted_blocks = sorted(
        translated_blocks,
        key=lambda b: (b.page_num, b.bbox[1], b.bbox[0])  # page, y, x
    )

    current_page = -1
    for block in sorted_blocks:
        # ページ区切り
        if block.page_num != current_page:
            if current_page >= 0:
                lines.append("\n---\n")
            current_page = block.page_num

        # ブロックタイプに応じたフォーマット
        if block.block_type == BlockType.TITLE:
            lines.append(f"# {block.translated_text}\n")
        elif block.block_type == BlockType.SECTION_HEADER:
            level = block.heading_level or 2
            prefix = "#" * level
            lines.append(f"{prefix} {block.translated_text}\n")
        elif block.block_type == BlockType.CAPTION:
            lines.append(f"*{block.translated_text}*\n")
        elif block.block_type == BlockType.LIST:
            # リスト項目の処理
            for item in block.translated_text.split("\n"):
                if item.strip():
                    lines.append(f"- {item.strip()}")
            lines.append("")
        else:
            # 本文
            lines.append(f"{block.translated_text}\n")

    return "\n".join(lines)
```

### Phase 4: PDF出力生成

**目標**: bbox座標を使用してレイアウト維持PDFを生成

**変更ファイル**: `src/index_pdf_translation/core/pdf_edit.py`

```python
# 新規関数追加

async def generate_translated_pdf(
    original_pdf: bytes,
    translated_blocks: list[TranslatedBlock],
    target_lang: str = "ja",
) -> bytes:
    """
    翻訳済みブロックから、元のレイアウトを維持したPDFを生成する。

    既存のremove_textbox_for_pdf、write_pdf_textを活用。
    """
    # 1. 翻訳対象ブロックの元テキストを削除
    blocks_to_remove = _convert_to_document_blocks(translated_blocks)
    pdf_data = await remove_textbox_for_pdf(original_pdf, blocks_to_remove)

    # 2. 翻訳テキストを挿入
    blocks_to_write = _prepare_write_blocks(translated_blocks, target_lang)
    pdf_data = await write_pdf_text(pdf_data, blocks_to_write, target_lang)

    return pdf_data

def _convert_to_document_blocks(blocks: list[TranslatedBlock]) -> DocumentBlocks:
    """TranslatedBlockを既存のDocumentBlocks形式に変換"""
    from collections import defaultdict

    page_groups: dict[int, list[dict]] = defaultdict(list)
    for block in blocks:
        page_groups[block.page_num].append({
            "page_no": block.page_num,
            "block_no": block.block_index,
            "coordinates": block.bbox,
            "text": block.original_text,
            "size": block.font_size or 10.0,
        })

    return [page_groups[i] for i in sorted(page_groups.keys())]

def _prepare_write_blocks(
    blocks: list[TranslatedBlock],
    target_lang: str,
) -> DocumentBlocks:
    """翻訳済みブロックをPDF書き込み用に準備"""
    from collections import defaultdict

    page_groups: dict[int, list[dict]] = defaultdict(list)
    for block in blocks:
        page_groups[block.page_num].append({
            "page_no": block.page_num,
            "block_no": block.block_index,
            "coordinates": block.bbox,
            "text": block.translated_text,
            "size": block.font_size or 10.0,
        })

    return [page_groups[i] for i in sorted(page_groups.keys())]
```

## 5. 新規エントリーポイント

### 5.1 メイン翻訳関数

```python
# src/index_pdf_translation/core/translate.py に追加

async def translate_pdf_v2(
    pdf_input: Union[bytes, Path, str],
    *,
    config: TranslationConfig,
) -> TranslationOutput:
    """
    新アーキテクチャによるPDF翻訳（ParsedDocumentベース）。

    Args:
        pdf_input: PDFバイナリデータまたはファイルパス
        config: 翻訳設定

    Returns:
        TranslationOutput (markdown, pdf, json_data, debug_pdf)
    """
    from index_pdf_translation.core.extractor import extract_from_pdf
    from index_pdf_translation.core.markdown_generator import generate_markdown
    from index_pdf_translation.core.pdf_edit import (
        generate_translated_pdf, create_viewing_pdf
    )

    # 1. 抽出
    logger.info("1. Extracting blocks from PDF...")
    parsed_doc, extraction_result = await extract_from_pdf(pdf_input)
    logger.info(f"   Found {len(extraction_result.all_translatable)} translatable blocks")

    # 2. 翻訳
    logger.info("2. Translating blocks...")
    translator = config.create_translator()
    translated_blocks = await translate_text_blocks(
        extraction_result.all_translatable,
        translator,
        config.target_lang,
    )
    logger.info(f"   Translated {len(translated_blocks)} blocks")

    # 3. Markdown生成（主要出力）
    logger.info("3. Generating Markdown...")
    markdown_output = generate_markdown(extraction_result, translated_blocks)

    # 4. PDF生成（オプション）
    pdf_output = None
    if config.generate_pdf:
        logger.info("4. Generating PDF...")
        original_pdf = pdf_input if isinstance(pdf_input, bytes) else Path(pdf_input).read_bytes()
        translated_pdf = await generate_translated_pdf(
            original_pdf, translated_blocks, config.target_lang
        )

        if config.side_by_side:
            pdf_output = await create_viewing_pdf(original_pdf, translated_pdf)
        else:
            pdf_output = translated_pdf

    # 5. JSON出力（オプション）
    json_data = None
    if config.generate_json:
        json_data = _create_json_output(parsed_doc, translated_blocks)

    return TranslationOutput(
        markdown=markdown_output,
        pdf=pdf_output,
        json_data=json_data,
    )
```

### 5.2 Config拡張

```python
# src/index_pdf_translation/config.py に追加

@dataclass
class TranslationConfig:
    # 既存フィールド...

    # 新規: 出力オプション
    generate_pdf: bool = True          # PDF出力を生成
    generate_json: bool = False        # JSON出力を生成
    side_by_side: bool = True          # 見開きPDF

    # 新規: Markdownオプション
    include_toc: bool = True           # 目次を含める
    include_images: bool = True        # 画像参照を含める

    # 新規: 翻訳対象オプション
    translate_headings: bool = True    # 見出しを翻訳
    translate_captions: bool = True    # キャプションを翻訳
```

### 5.3 CLI拡張

```python
# src/index_pdf_translation/cli.py に追加

@click.option("--output-format", "-f",
              type=click.Choice(["markdown", "pdf", "both"]),
              default="both",
              help="Output format (default: both)")
@click.option("--no-pdf", is_flag=True,
              help="Skip PDF generation (Markdown only)")
@click.option("--json", "generate_json", is_flag=True,
              help="Also generate JSON output")
@click.option("--no-toc", is_flag=True,
              help="Exclude table of contents from Markdown")
```

## 6. ファイル構成

### 6.1 新規ファイル

```
src/index_pdf_translation/
├── core/
│   ├── models.py              # 新規: データクラス定義
│   ├── extractor.py           # 新規: ParsedDocument抽出
│   ├── markdown_generator.py  # 新規: Markdown生成
│   ├── pdf_edit.py            # 変更: PDF生成関数追加
│   └── translate.py           # 変更: translate_pdf_v2追加
├── config.py                   # 変更: 新オプション追加
└── cli.py                      # 変更: CLIオプション追加
```

### 6.2 削除候補（Phase 4以降）

```
src/index_pdf_translation/core/pdf_edit.py:
  - extract_text_coordinates_blocks()  # → extractor.pyに移行
  - extract_text_coordinates_dict()    # → extractor.pyに移行
  - remove_blocks()                    # → boxclass分類に置換
```

## 7. 移行計画

### 7.1 段階的移行

| Phase | 内容 | 既存機能 |
|-------|------|----------|
| Phase 1 | extractor.py実装 | 維持 |
| Phase 2 | translate_text_blocks実装 | 維持 |
| Phase 3 | markdown_generator.py実装 | 維持 |
| Phase 4 | PDF生成関数追加 | 維持 |
| Phase 5 | translate_pdf_v2を新デフォルトに | 非推奨化 |
| Phase 6 | 旧関数削除 | 削除 |

### 7.2 CLI移行

```bash
# Phase 1-4: 既存CLIは変更なし
uv run translate-pdf paper.pdf

# Phase 5: 新CLIがデフォルト
uv run translate-pdf paper.pdf              # → translate_pdf_v2使用
uv run translate-pdf paper.pdf --legacy     # → 旧translate_pdf使用

# Phase 6: --legacy削除
```

### 7.3 後方互換性

- `translate_pdf()`は内部的に維持（Phase 5まで）
- `DocumentBlocks`型は変換レイヤーで対応
- 既存のテストはそのまま動作

## 8. テスト計画

### 8.1 単体テスト

```python
# tests/test_extractor.py
class TestExtractor:
    async def test_extract_from_pdf(self):
        """PDF抽出のテスト"""

    async def test_boxclass_classification(self):
        """boxclass分類のテスト"""

    async def test_text_extraction_from_box(self):
        """ボックスからのテキスト抽出テスト"""

# tests/test_markdown_generator.py
class TestMarkdownGenerator:
    def test_generate_markdown_basic(self):
        """基本的なMarkdown生成テスト"""

    def test_heading_levels(self):
        """見出しレベルのテスト"""

    def test_toc_generation(self):
        """目次生成のテスト"""
```

### 8.2 統合テスト

```python
# tests/test_integration_v2.py
class TestTranslatePdfV2:
    @pytest.mark.parametrize("pdf_file", PDF_FIXTURES)
    async def test_full_pipeline(self, pdf_file):
        """完全なパイプラインテスト"""

    async def test_markdown_output(self):
        """Markdown出力テスト"""

    async def test_pdf_output(self):
        """PDF出力テスト"""
```

### 8.3 比較テスト

```python
# tests/test_comparison.py
class TestV1V2Comparison:
    async def test_translation_quality_comparison(self):
        """v1とv2の翻訳品質比較"""

    async def test_block_classification_comparison(self):
        """ブロック分類精度の比較"""
```

## 9. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| PyMuPDF4LLM APIの変更 | 高 | バージョン固定、変更追跡 |
| boxclass精度不足 | 中 | フォールバックルール追加 |
| 一時ファイル管理 | 低 | contextmanagerで確実に削除 |
| メモリ使用量増加 | 中 | ストリーミング処理検討 |
| 既存テスト破損 | 高 | 段階的移行、並行稼働 |

## 10. 依存関係

### 10.1 必須依存

```toml
# pyproject.toml
dependencies = [
    "pymupdf>=1.24.0",
    "pymupdf4llm>=0.0.17",  # 追加
    # 既存の依存関係...
]
```

### 10.2 バージョン要件

| パッケージ | 最小バージョン | 理由 |
|-----------|---------------|------|
| pymupdf4llm | 0.0.17 | to_json()サポート |
| pymupdf | 1.24.0 | pymupdf4llm互換性 |

## 11. 参考資料

- [PyMuPDF4LLM Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [PyMuPDF4LLM評価レポート](../research/evaluations/pymupdf4llm-evaluation.md)
- [Issue #38: ParsedDocumentベース翻訳パイプライン](https://github.com/Mega-Gorilla/Index_PDF_Translation/issues/38)
- [Issue #31: ブロック分類改善調査](https://github.com/Mega-Gorilla/Index_PDF_Translation/issues/31)
