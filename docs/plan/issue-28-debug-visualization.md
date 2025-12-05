# Issue #28: デバッグモードでブロック分類の可視化PDF出力

## 概要

`--debug` オプション使用時に、ブロック分類結果を視覚的に確認できるPDFを自動出力する機能を実装する。

**Issue**: https://github.com/Mega-Gorilla/Index_PDF_Translation/issues/28

## 現状分析

### 問題点

| ファイル | 現状 | 問題点 |
|---------|------|--------|
| `cli.py:268` | `pdf_translate()` を呼び出し | デバッグ情報を受け取れない |
| `translate.py:328-335` | `plot_images` を取得 | **使用せずに破棄** |
| `pdf_edit.py:381-428` | `pdf_draw_blocks()` 存在 | **呼び出されていない** |

### 現在のコードフロー

```
cli.py
  └── pdf_translate(pdf_data, config)  # config.debug=True
        └── translate.py
              └── remove_blocks(..., debug=True)
                    └── plot_images を生成 ← 使用されていない！
```

## 設計方針

### 戻り値の変更: `TranslationResult` dataclass

**Before:**
```python
async def pdf_translate(...) -> Optional[bytes]:
```

**After:**
```python
@dataclass
class TranslationResult:
    pdf: bytes                              # 翻訳済みPDF（見開き）
    debug_pdf: Optional[bytes] = None       # デバッグ可視化PDF（統合版）

async def pdf_translate(...) -> Optional[TranslationResult]:
```

### デバッグPDF構成（統合版）

ヒストグラム画像を別ファイルではなく、デバッグPDFの先頭ページとして統合:

```
debug_blocks.pdf
├── Page 1: トークン分布ヒストグラム
├── Page 2: フォントサイズ分布ヒストグラム
├── Page 3: スコア分布ヒストグラム
├── Page 4: 元PDF 1ページ目（ブロック枠付き）
├── Page 5: 元PDF 2ページ目（ブロック枠付き）
└── ...
```

**メリット:**
- 出力ファイルが1つに統合（管理しやすい）
- デバッグ情報が1ファイルで完結
- PDFビューアで連続確認可能

### 色分け設計

| ブロック種類 | 色 | RGB | 用途 |
|-------------|-----|-----|------|
| text_blocks | 緑 | `[0, 0.7, 0]` | 本文（翻訳対象） |
| fig_blocks | 黄 | `[1, 0.8, 0]` | 図表キャプション |
| removed_blocks | 赤 | `[1, 0, 0]` | 除外ブロック |

全ブロック共通: `fill_opacity=0.2`, `width=1`

### 出力ファイル命名規則

```
入力: paper.pdf
出力:
  - paper_translated.pdf        # 翻訳済み見開きPDF（既存）
  - paper_debug.pdf             # デバッグ可視化PDF（新規・統合版）
```

## 実装フェーズ

### Phase 1: 型定義の追加

**ファイル**: `src/index_pdf_translation/core/translate.py`

```python
from dataclasses import dataclass

@dataclass
class TranslationResult:
    """翻訳結果を格納するデータクラス。"""
    pdf: bytes                              # 翻訳済みPDF（見開き）
    debug_pdf: Optional[bytes] = None       # デバッグ可視化PDF（統合版）
```

**ファイル**: `src/index_pdf_translation/__init__.py`

```python
from index_pdf_translation.core.translate import pdf_translate, TranslationResult

__all__ = [
    "__version__",
    "pdf_translate",
    "TranslationConfig",
    "TranslationResult",  # 追加
]
```

### Phase 2: `pdf_translate()` の修正

**ファイル**: `src/index_pdf_translation/core/translate.py`

デバッグモード時の処理を追加:

```python
async def pdf_translate(...) -> Optional[TranslationResult]:
    # ... 既存の処理 ...

    # デバッグPDF生成
    debug_pdf = None

    if config.debug:
        # 1. 元のPDFにブロック枠を描画
        blocks_pdf = pdf_data

        # text_blocks: 緑
        blocks_pdf = await pdf_draw_blocks(
            blocks_pdf, text_blocks,
            line_color_rgb=[0, 0.7, 0],
            fill_color_rgb=[0, 0.7, 0],
            fill_opacity=0.2,
            width=1,
        )

        # fig_blocks: 黄
        blocks_pdf = await pdf_draw_blocks(
            blocks_pdf, fig_blocks,
            line_color_rgb=[1, 0.8, 0],
            fill_color_rgb=[1, 0.8, 0],
            fill_opacity=0.2,
            width=1,
        )

        # removed_blocks: 赤
        blocks_pdf = await pdf_draw_blocks(
            blocks_pdf, remove_info,
            line_color_rgb=[1, 0, 0],
            fill_color_rgb=[1, 0, 0],
            fill_opacity=0.2,
            width=1,
        )

        # 2. ヒストグラム画像をPDFページとして統合
        debug_pdf = await create_debug_pdf(blocks_pdf, plot_images)

    return TranslationResult(
        pdf=merged_pdf_data,
        debug_pdf=debug_pdf,
    )
```

### Phase 2.5: ヒストグラム統合関数の追加

**ファイル**: `src/index_pdf_translation/core/pdf_edit.py`

PNG画像をPDFの先頭ページとして追加する関数:

```python
async def create_debug_pdf(
    blocks_pdf: bytes,
    histogram_images: list[bytes],
) -> bytes:
    """
    ヒストグラム画像をPDFの先頭ページとして追加します。

    Args:
        blocks_pdf: ブロック枠付きPDFデータ
        histogram_images: ヒストグラムPNG画像のリスト

    Returns:
        統合されたデバッグPDFデータ
    """
    # 新しいPDFドキュメントを作成
    new_doc = fitz.open()

    # 1. ヒストグラム画像を先頭ページとして追加
    for img_data in histogram_images:
        # PNG画像からサイズを取得
        img = fitz.Pixmap(img_data)

        # A4サイズのページを追加（画像をフィット）
        page = new_doc.new_page(width=595, height=842)  # A4 portrait

        # 画像を中央に配置
        img_rect = fitz.Rect(50, 50, 545, 792)
        page.insert_image(img_rect, stream=img_data)

    # 2. ブロック枠付きPDFのページを追加
    blocks_doc = fitz.open(stream=blocks_pdf, filetype="pdf")
    new_doc.insert_pdf(blocks_doc)
    blocks_doc.close()

    # 出力
    output_buffer = BytesIO()
    new_doc.save(output_buffer, garbage=4, deflate=True, clean=True)
    new_doc.close()

    return output_buffer.getvalue()
```

### Phase 3: CLI の修正

**ファイル**: `src/index_pdf_translation/cli.py`

```python
async def run(args: argparse.Namespace) -> int:
    # ... 既存の処理 ...

    # Execute translation
    try:
        result = await pdf_translate(pdf_data, config=config)
    except Exception as e:
        print(f"Error: Translation failed: {e}", file=sys.stderr)
        return 1

    if result is None:
        print("Error: Translation failed", file=sys.stderr)
        return 1

    # Save result
    with open(output_path, "wb") as f:
        f.write(result.pdf)

    # Save debug PDF (統合版: ヒストグラム + ブロック枠)
    if args.debug and result.debug_pdf:
        debug_path = output_path.with_stem(output_path.stem.replace("translated_", "") + "_debug")
        with open(debug_path, "wb") as f:
            f.write(result.debug_pdf)
        print(f"Debug PDF: {debug_path}")

    print()
    print(f"Complete: {output_path}")
    return 0
```

**出力例:**
```
Input: paper.pdf
Output: output/translated_paper.pdf
Backend: google
Debug mode: enabled

Complete: output/translated_paper.pdf
Debug PDF: output/paper_debug.pdf
```

### Phase 4: テストの追加

**ファイル**: `tests/test_translate.py`（新規または追記）

```python
import pytest
from index_pdf_translation import pdf_translate, TranslationConfig, TranslationResult


class TestTranslationResult:
    """Tests for TranslationResult dataclass."""

    def test_translation_result_has_pdf(self) -> None:
        """TranslationResult should have pdf field."""
        result = TranslationResult(pdf=b"test")
        assert result.pdf == b"test"

    def test_translation_result_debug_pdf_optional(self) -> None:
        """debug_pdf field should be optional."""
        result = TranslationResult(pdf=b"test")
        assert result.debug_pdf is None


class TestPdfTranslateDebugMode:
    """Tests for pdf_translate with debug mode."""

    @pytest.mark.asyncio
    async def test_debug_mode_returns_debug_pdf(
        self, sample_pdf: bytes
    ) -> None:
        """Debug mode should return debug_pdf with histograms + block frames."""
        config = TranslationConfig(debug=True)
        result = await pdf_translate(
            sample_pdf, config=config, disable_translate=True
        )
        assert result is not None
        assert result.debug_pdf is not None
        # デバッグPDFがヒストグラム3ページ + 元PDFページを含むことを確認
        import fitz
        doc = fitz.open(stream=result.debug_pdf, filetype="pdf")
        assert len(doc) >= 4  # 最低4ページ（ヒストグラム3 + 元PDF1）
        doc.close()

    @pytest.mark.asyncio
    async def test_non_debug_mode_no_debug_pdf(
        self, sample_pdf: bytes
    ) -> None:
        """Non-debug mode should not return debug_pdf."""
        config = TranslationConfig(debug=False)
        result = await pdf_translate(
            sample_pdf, config=config, disable_translate=True
        )
        assert result is not None
        assert result.debug_pdf is None
```

### Phase 5: ドキュメント更新

**ファイル**: `docs/architecture/block-detection-algorithm.md` Section 8

Section 8「デバッグ機能」を更新し、新しい出力ファイルについて記載。

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/index_pdf_translation/core/translate.py` | `TranslationResult` 追加、`pdf_translate()` 修正 |
| `src/index_pdf_translation/core/pdf_edit.py` | `create_debug_pdf()` 関数追加 |
| `src/index_pdf_translation/__init__.py` | `TranslationResult` エクスポート |
| `src/index_pdf_translation/cli.py` | デバッグファイル保存処理 |
| `tests/test_translate.py` | デバッグモードテスト追加 |
| `docs/architecture/block-detection-algorithm.md` | Section 8 更新 |

## Breaking Changes

- `pdf_translate()` の戻り値が `Optional[bytes]` から `Optional[TranslationResult]` に変更
- ライブラリとして使用している場合、`result.pdf` でPDFデータにアクセスする必要がある

## 完了条件

- [ ] Phase 1: `TranslationResult` dataclass の追加
- [ ] Phase 2: `pdf_translate()` でデバッグPDF生成
- [ ] Phase 2.5: `create_debug_pdf()` 関数追加（ヒストグラム統合）
- [ ] Phase 3: CLI でデバッグファイル保存
- [ ] Phase 4: テスト追加・既存テスト修正
- [ ] Phase 5: ドキュメント更新
- [ ] 全テスト通過
- [ ] E2Eテスト（実際のPDFで確認）

## 参考資料

- Issue: https://github.com/Mega-Gorilla/Index_PDF_Translation/issues/28
- 関連ドキュメント: `docs/architecture/block-detection-algorithm.md`
