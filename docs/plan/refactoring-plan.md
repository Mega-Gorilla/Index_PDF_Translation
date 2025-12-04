# リファクタリング計画: CLIツールへの簡素化

> **Issue**: #8
> **作成日**: 2024-12-04
> **ステータス**: 計画策定中

---

## 1. 概要

### 1.1 目的

本リポジトリをWebサーバー向けライブラリから、シンプルな **PDF翻訳CLIツール** へリファクタリングする。

### 1.2 背景

- 元々Webサービス（Indqx PDF翻訳）向けに開発
- 2024年5月31日にサービス終了
- 開発初期（技術習得段階）に作成されたコードが多く残存
- 不要なWeb関連コード、デバッグコードが混在
- メンテナンス性・可読性の向上が必要

### 1.3 ゴール

```
入力: PDF ファイル
  ↓
処理: DeepL API で翻訳
  ↓
出力: 見開きPDF（オリジナル + 翻訳）
```

- 最小限の依存関係
- シンプルで理解しやすいコード構造
- CLIツールとしての使いやすさ

---

## 2. 現状分析

### 2.1 ファイル構成

```
Index_PDF_Translation/
├── manual_translate_pdf.py      # CLI ツール（186行）
├── demo_app.py                  # FastAPI Webサーバー（483行）
├── config.py                    # 設定ファイル（36行）
├── objective_DB_config.py       # Backblaze/DB設定
├── requirements.txt             # 依存パッケージ
├── modules/
│   ├── translate.py             # 翻訳オーケストレーション（222行）
│   ├── pdf_edit.py              # PDF処理エンジン（634行）
│   ├── spacy_api.py             # トークン化（46行）
│   ├── arxiv_api.py             # ArXiv連携（154行）
│   ├── backblaze_api.py         # クラウドストレージ（266行）
│   ├── database.py              # SQLAlchemy ORM（211行）
│   └── generate_fixed_key.py    # RSAキー生成
├── fonts/                       # フォントファイル
├── data/                        # ライセンスJSON、QR画像
├── output/                      # 出力先
└── debug/                       # デバッグ出力
```

**総コード行数**: 約2,200行（Pythonファイルのみ）

### 2.2 モジュール依存関係

```
manual_translate_pdf.py
    └── modules/translate.py
            ├── modules/pdf_edit.py
            │       └── modules/spacy_api.py
            └── aiohttp (DeepL API)

demo_app.py ← 削除対象
    ├── fastapi
    ├── modules/backblaze_api.py ← 削除対象
    ├── modules/arxiv_api.py ← 削除対象
    ├── modules/database.py ← 削除対象
    └── modules/translate.py
```

### 2.3 コア翻訳フロー

`modules/translate.py::pdf_translate()` が中心：

```python
async def pdf_translate(key, pdf_data, source_lang='en', to_lang='ja', api_url, debug=False):
    # 1. テキスト抽出
    block_info = await extract_text_coordinates_dict(pdf_data)

    # 2. ブロック分類（本文/図表/削除対象）
    body_blocks, figure_blocks, remove_blocks, _ = await remove_blocks(block_info)

    # 3. 元PDFからテキスト削除
    removed_pdf = await remove_textbox_for_pdf(pdf_data, all_blocks)

    # 4. DeepL翻訳
    translated_body = await translate_blocks(body_blocks, key, to_lang, api_url)
    translated_figures = await translate_blocks(figure_blocks, key, to_lang, api_url)

    # 5. 翻訳テキスト挿入
    translated_pdf = await write_pdf_text(removed_pdf, translated_blocks, to_lang)

    # 6. ロゴ追加
    translated_pdf = await write_logo_data(translated_pdf)

    # 7. 見開きPDF生成
    final_pdf = await create_viewing_pdf(pdf_data, translated_pdf)

    return final_pdf
```

### 2.4 外部依存（現状）

| パッケージ | 用途 | CLI必要性 |
|-----------|------|----------|
| PyMuPDF (fitz) | PDF処理 | ✅ 必須 |
| spacy | トークン化 | ✅ 必須 |
| aiohttp | HTTP通信（DeepL） | ✅ 必須 |
| numpy | 数値計算 | ✅ 必須 |
| matplotlib | グラフ描画（デバッグ） | ❌ 削除可 |
| fastapi | Webフレームワーク | ❌ 削除 |
| uvicorn | ASGIサーバー | ❌ 削除 |
| sqlalchemy | ORM | ❌ 削除 |
| b2sdk | Backblaze連携 | ❌ 削除 |
| pycryptodome | RSA暗号 | ❌ 削除 |
| psycopg2 | PostgreSQL | ❌ 削除 |

---

## 3. リファクタリング後の構成

### 3.1 ターゲット構成

```
Index_PDF_Translation/
├── translate_pdf.py             # CLIエントリーポイント（新規）
├── config.py                    # DeepL設定のみ
├── requirements.txt             # 最小依存
├── modules/
│   ├── translate.py             # 翻訳オーケストレーション（整理）
│   ├── pdf_edit.py              # PDF処理（整理）
│   └── spacy_api.py             # トークン化（維持）
├── fonts/                       # フォントファイル
│   ├── LiberationSerif-*.ttf
│   ├── ipam.ttf
│   ├── OFL.txt
│   └── IPA_Font_License_Agreement_v1.0.txt
├── output/                      # デフォルト出力先
├── docs/                        # ドキュメント
├── README.md
├── LICENSE
└── CLAUDE.md
```

### 3.2 新しいCLIインターフェース

```bash
# 基本使用法
python translate_pdf.py input.pdf

# オプション付き
python translate_pdf.py input.pdf --output ./translated.pdf
python translate_pdf.py input.pdf --source en --target ja
python translate_pdf.py input.pdf --no-side-by-side  # 見開き無効
python translate_pdf.py input.pdf --no-logo          # ロゴ無効
```

### 3.3 予想コード量

| ファイル | Before | After | 削減 |
|----------|--------|-------|------|
| エントリーポイント | 186行 | ~80行 | -57% |
| translate.py | 222行 | ~150行 | -32% |
| pdf_edit.py | 634行 | ~400行 | -37% |
| spacy_api.py | 46行 | 46行 | 0% |
| config.py | 36行 | ~15行 | -58% |
| **合計** | **1,124行** | **~690行** | **-39%** |

---

## 4. 削除対象の詳細

### 4.1 完全削除ファイル

| ファイル | 行数 | 削除理由 |
|----------|------|----------|
| `demo_app.py` | 483 | FastAPI Webサーバー全体 |
| `objective_DB_config.py` | ~20 | Backblaze/DB認証情報 |
| `modules/arxiv_api.py` | 154 | ArXiv連携（Web専用） |
| `modules/backblaze_api.py` | 266 | クラウドストレージ（Web専用） |
| `modules/database.py` | 211 | SQLAlchemy ORM（Web専用） |
| `modules/generate_fixed_key.py` | ~50 | RSAキー生成（Web専用） |

### 4.2 削除する関数・コード

#### manual_translate_pdf.py

```python
# 削除対象
def translate_test()      # デバッグ用
def test_bench()          # バッチテスト
def pdf_block_test()      # ブロック分類テスト
def pdf_block_bach()      # バッチ分類テスト
def marge_test()          # 見開き結合テスト

# 維持（整理）
def translate_local()     # → translate_pdf.py に移行
```

#### modules/pdf_edit.py

```python
# 削除対象
async def extract_text_coordinates_dict_dev()  # デバッグ用
async def pdf_draw_blocks()                    # ブロック枠描画
async def write_image_data()                   # 画像埋め込み（未使用）
def plot_area_distribution()                   # matplotlib可視化

# 維持
async def extract_text_coordinates_dict()      # テキスト抽出
async def remove_blocks()                      # ブロック分類
async def remove_textbox_for_pdf()             # テキスト削除
async def preprocess_write_blocks()            # フォント計算
async def write_pdf_text()                     # テキスト挿入
async def write_logo_data()                    # ロゴ追加（オプション化検討）
async def create_viewing_pdf()                 # 見開き生成
```

#### modules/translate.py

```python
# 削除対象
async def PDF_block_check()                    # ブロック可視化（デバッグ）
# コメントアウトされたデバッグコード（約60行）

# 維持
async def pdf_translate()                      # メイン翻訳フロー
async def translate_str_data()                 # DeepL API呼び出し
async def translate_blocks()                   # 複数ブロック翻訳
def preprocess_translation_blocks()            # 前処理
```

### 4.3 削除するフォルダ/ファイル

```
削除:
├── data/license.json        # ArXivライセンス検証（Web専用）
├── data/indqx_qr.png        # QRコード（ロゴ機能削除時）
├── debug/                   # デバッグ出力フォルダ
└── Test Bench/              # テストベンチマーク
```

---

## 5. 実装フェーズ

### Phase 1: 不要ファイルの削除

**目的**: Web専用モジュールを完全削除

**タスク**:
- [ ] `demo_app.py` を削除
- [ ] `objective_DB_config.py` を削除
- [ ] `modules/arxiv_api.py` を削除
- [ ] `modules/backblaze_api.py` を削除
- [ ] `modules/database.py` を削除
- [ ] `modules/generate_fixed_key.py` を削除
- [ ] `data/license.json` を削除

**検証**: インポートエラーが発生しないことを確認

---

### Phase 2: デバッグコードの整理

**目的**: 不要なデバッグ/テストコードを削除

**タスク**:
- [ ] `manual_translate_pdf.py` のテスト関数を削除
- [ ] `modules/pdf_edit.py` のデバッグ関数を削除
- [ ] `modules/translate.py` のコメントアウトコードを削除
- [ ] matplotlib依存を削除

**検証**: コア機能が動作することを確認

---

### Phase 3: config.py の簡素化

**目的**: CLI に必要な設定のみに絞る

**Before**:
```python
DeepL_API_Key = "xxx"
DeepL_URL = "https://api-free.deepl.com/v2/translate"
output_folder_path = "./output/"
Debug_folder_path = "./debug/"
bach_process_path = "./Test Bench/"
CORS_CONFIG = {...}
URL_LIST = [...]
Language = {"en": "EN", "ja": "JA"}
SPACY_MODEL = {...}
```

**After**:
```python
# DeepL API 設定
DEEPL_API_KEY = "your-api-key"
DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"

# 出力設定
OUTPUT_DIR = "./output"

# 言語設定
SUPPORTED_LANGUAGES = {
    "en": {"deepl": "EN", "spacy": "en_core_web_sm"},
    "ja": {"deepl": "JA", "spacy": "ja_core_news_sm"},
}
```

---

### Phase 4: requirements.txt の整理

**Before**:
```
uvicorn
fastapi
PyMuPDF
spacy
aiohttp
sqlalchemy
b2sdk
pycryptodome
psycopg2-binary
numpy
matplotlib
# spaCy models...
```

**After**:
```
PyMuPDF>=1.24.0
spacy>=3.7.0
aiohttp>=3.9.0
numpy>=1.26.0

# spaCy language models
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.7.0/ja_core_news_sm-3.7.0-py3-none-any.whl
```

---

### Phase 5: CLIエントリーポイントの作成

**目的**: 使いやすいCLIインターフェースを提供

**新規ファイル**: `translate_pdf.py`

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
PDF Translation CLI Tool

Usage:
    python translate_pdf.py <input.pdf> [options]

Options:
    --output, -o    Output file path (default: ./output/translated_<input>.pdf)
    --source, -s    Source language (default: en)
    --target, -t    Target language (default: ja)
    --no-side-by-side   Disable side-by-side PDF generation
    --no-logo           Disable logo watermark
"""

import argparse
import asyncio
from pathlib import Path

from modules.translate import pdf_translate
from config import DEEPL_API_KEY, DEEPL_API_URL, OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Translate PDF documents")
    parser.add_argument("input", help="Input PDF file path")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-s", "--source", default="en", help="Source language")
    parser.add_argument("-t", "--target", default="ja", help="Target language")
    parser.add_argument("--no-side-by-side", action="store_true")
    parser.add_argument("--no-logo", action="store_true")

    args = parser.parse_args()

    # Run translation
    asyncio.run(translate(args))


async def translate(args):
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return

    # Read PDF
    with open(input_path, "rb") as f:
        pdf_data = f.read()

    print(f"Translating: {input_path.name}")
    print(f"  Source: {args.source} → Target: {args.target}")

    # Translate
    result = await pdf_translate(
        key=DEEPL_API_KEY,
        pdf_data=pdf_data,
        source_lang=args.source,
        to_lang=args.target,
        api_url=DEEPL_API_URL,
        side_by_side=not args.no_side_by_side,
        add_logo=not args.no_logo,
    )

    # Save output
    output_path = args.output or Path(OUTPUT_DIR) / f"translated_{input_path.name}"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(result)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
```

---

### Phase 6: コードリファクタリング

**目的**: 残存コードの整理・改善

**タスク**:
- [ ] 関数のドキュメント文字列を追加/更新
- [ ] 型ヒントを追加
- [ ] エラーハンドリングの改善
- [ ] ログ出力の整理（print → logging）
- [ ] 定数の整理

---

## 6. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| 削除漏れによる参照エラー | 高 | 段階的削除＆各フェーズで動作確認 |
| 翻訳品質の劣化 | 中 | リファクタリング前後でテストPDFを比較 |
| 依存関係の破損 | 中 | requirements.txt更新後にクリーンインストールテスト |
| フォント処理の不具合 | 中 | 日英両言語でテスト |

---

## 7. テスト計画

### 7.1 テストケース

| テストケース | 内容 | 優先度 |
|--------------|------|--------|
| 基本翻訳 | 英語PDF → 日本語翻訳 | 高 |
| 逆翻訳 | 日本語PDF → 英語翻訳 | 高 |
| 見開きPDF | 左右並列出力の確認 | 高 |
| フォントフォールバック | フォント不在時の警告確認 | 中 |
| 大容量PDF | 10ページ以上のPDF処理 | 中 |
| CLI引数 | 各オプションの動作確認 | 中 |

### 7.2 テスト用PDF

- 英語学術論文（arXiv等）
- 日本語文書
- 図表を含むPDF
- 数式を含むPDF

---

## 8. スケジュール

| Phase | 内容 | 見積もり |
|-------|------|----------|
| Phase 1 | 不要ファイル削除 | 0.5日 |
| Phase 2 | デバッグコード整理 | 0.5日 |
| Phase 3 | config.py簡素化 | 0.5日 |
| Phase 4 | requirements.txt整理 | 0.5日 |
| Phase 5 | CLI作成 | 1日 |
| Phase 6 | コードリファクタリング | 1-2日 |
| テスト | 動作確認 | 0.5日 |
| **合計** | | **4-5日** |

---

## 9. 成果指標

| 指標 | Before | After | 目標 |
|------|--------|-------|------|
| Pythonファイル数 | 10 | 4-5 | -50% |
| 総コード行数 | ~2,200 | ~700 | -68% |
| 依存パッケージ | 15+ | 5-6 | -60% |
| クリーンインストール時間 | - | 短縮 | 改善 |

---

## 10. 未決定事項

### 10.1 検討が必要な項目

1. **write_logo_data() の扱い**
   - オプション A: 削除（シンプル化優先）
   - オプション B: オプション化（`--no-logo`）
   - **提案**: オプション B

2. **見開きPDF生成**
   - オプション A: 必須機能として維持
   - オプション B: オプション化（`--no-side-by-side`）
   - **提案**: オプション B

3. **data/indqx_qr.png の扱い**
   - ロゴ機能を残す場合は維持
   - 削除する場合は不要

4. **debug/ フォルダの扱い**
   - 開発時には便利だが、リリース版には不要
   - **提案**: .gitignore に追加し、フォルダは残す

---

## 11. 参考資料

- Issue #8: リファクタリング計画
- CLAUDE.md: プロジェクト概要
- 元Webサービス: https://indqx-demo-front.onrender.com/ (終了)

---

## 変更履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|----------|
| 2024-12-04 | 1.0 | 初版作成 |
