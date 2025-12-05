# Index PDF Translation

[![Tests](https://github.com/Mega-Gorilla/Index_PDF_Translation/actions/workflows/test.yml/badge.svg)](https://github.com/Mega-Gorilla/Index_PDF_Translation/actions/workflows/test.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> **Note**: 本プロジェクトは、2024年5月31日までWebサービスとして提供されていた「Indqx PDF 翻訳」のソースコードです。**Webサーバー機能は削除済み**で、現在はローカルCLIツールとしてのみ動作します。

Index PDF Translationは、PDFのフォーマットを崩さずに翻訳できる論文向けPDF翻訳ツールです。

## 主な機能

- **本文自動認識**: 論文データの数式やタイトルなどの翻訳不要部分を無視して本文のみを翻訳
  - ブロックの幅、文字数、フォントサイズによる分析
- **クロスブロック翻訳**: 終了記号(`.``:``/`など)がない場合、複数ブロックを1ブロックとして翻訳
  - ブロック間やページ間で翻訳文が途切れる問題を解決
- **図表キャプション対応**: 図や表の説明文ブロックを自動認識し、本文と分割して翻訳
- **見開きPDF出力**: オリジナルと翻訳を並べた見開きPDFを生成

<img src="https://github.com/Mega-Gorilla/Index_PDF_Translation/blob/main/images/GMtDCedbsAAIkDO.jpg?raw=true" width="500" />

## インストール

本リポジトリをクローン後、以下のコマンドで必要ライブラリをインストールしてください。
Python 3.11以上が必要です。

### uv を使用する場合（推奨）

[uv](https://docs.astral.sh/uv/) を使用すると、高速にインストールできます。

```bash
uv sync
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download ja_core_news_sm
```

### pip を使用する場合

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download ja_core_news_sm
```

## 使用方法

### 基本的な使い方

```bash
# Google翻訳を使用（デフォルト、APIキー不要）
uv run translate-pdf paper.pdf

# pip でインストールした場合
translate-pdf paper.pdf
```

翻訳が完了すると、`./output/translated_<ファイル名>.pdf` に見開きPDF（オリジナル + 翻訳）が保存されます。

### 翻訳バックエンド

#### Google 翻訳（デフォルト）

APIキー不要で即座に使用可能：

```bash
translate-pdf paper.pdf
translate-pdf paper.pdf --backend google  # 明示的に指定
```

#### DeepL（高品質）

高品質な翻訳が必要な場合：

```bash
export DEEPL_API_KEY="your-api-key"
translate-pdf paper.pdf --backend deepl

# または
translate-pdf paper.pdf --backend deepl --api-key "your-api-key"
```

DeepL を使用するには追加の依存関係が必要：

```bash
uv pip install index-pdf-translation[deepl]
```

### オプション

| オプション | 説明 | デフォルト |
|------------|------|-----------|
| `-o, --output` | 出力ファイルのパス | `./output/translated_<input>.pdf` |
| `-b, --backend` | 翻訳バックエンド (google/deepl) | `google` |
| `-s, --source` | 翻訳元の言語 (en/ja) | `en` |
| `-t, --target` | 翻訳先の言語 (en/ja) | `ja` |
| `--api-key` | DeepL APIキー | 環境変数 `DEEPL_API_KEY` |
| `--api-url` | DeepL API URL | `https://api-free.deepl.com/v2/translate` |
| `--no-logo` | ロゴウォーターマークを無効化 | - |
| `--debug` | デバッグモード（ブロック分類の可視化） | - |

### 使用例

```bash
# 出力ファイルを指定
uv run translate-pdf paper.pdf -o ./result.pdf

# DeepLで翻訳
uv run translate-pdf paper.pdf --backend deepl

# 日本語から英語に翻訳
uv run translate-pdf paper.pdf -s ja -t en

# ロゴなし + デバッグモード
uv run translate-pdf paper.pdf --no-logo --debug
```

### 環境変数の設定（オプション）

DeepL を使用する場合、環境変数で API キーを設定できます。

#### 方法 1: .env ファイルを使用

```bash
# .env.example をコピー
cp .env.example .env

# .env を編集して API キーを設定
# DEEPL_API_KEY=your-api-key-here

# シェルで読み込んで実行
source .env && translate-pdf paper.pdf --backend deepl
```

#### 方法 2: export コマンド

```bash
export DEEPL_API_KEY="your-api-key"
translate-pdf paper.pdf --backend deepl
```

> **Note**: Google 翻訳（デフォルト）は API キー不要のため、環境変数の設定なしで使用できます。

## アーキテクチャ

### ファイル構成

```
Index_PDF_Translation/
├── src/index_pdf_translation/         # パッケージ本体
│   ├── __init__.py                    # 公開API
│   ├── cli.py                         # CLIエントリーポイント
│   ├── config.py                      # 設定管理
│   ├── core/
│   │   ├── translate.py               # 翻訳オーケストレーション
│   │   └── pdf_edit.py                # PDF処理エンジン（PyMuPDF）
│   ├── nlp/
│   │   └── tokenizer.py               # テキストトークン化（spaCy）
│   ├── resources/
│   │   ├── fonts/                     # フォントファイル
│   │   └── data/                      # ロゴ画像
│   └── logger.py                      # 集中ログ管理
└── output/                            # デフォルト出力先
```

### 翻訳フロー

```
pdf_translate()
├── extract_text_coordinates_dict()  # テキストブロック抽出
├── remove_blocks()                   # ブロック分類（本文/図表/除外）
├── remove_textbox_for_pdf()          # 元テキスト削除
├── translate_blocks()                # 翻訳（Google/DeepL）
├── preprocess_write_blocks()         # フォントサイズ計算
├── write_pdf_text()                  # 翻訳テキスト挿入
├── write_logo_data()                 # ロゴ追加（オプション）
└── create_viewing_pdf()              # 見開きPDF生成
```

### ブロック分類アルゴリズム

1. **テキストブロック抽出**: PyMuPDFで座標・フォント情報付きブロックを取得
2. **スコア計算**: トークン数、ブロック幅（IQR外れ値検出）、フォントサイズ偏差でスコアリング
3. **分類**: 本文 | 図表キャプション（キーワード検出） | 除外
4. **前処理**: 終端句読点のない連続ブロックをマージ（文脈保持）

## トラブルシューティング

### spaCyモデルが見つからない

```
OSError: [E050] Can't find model 'en_core_web_sm'
```

**解決方法**: spaCyモデルをダウンロードしてください。

```bash
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download ja_core_news_sm
```

### DeepL APIエラー

```
DeepL API error (status 403)
```

**解決方法**:
- `--backend deepl` を指定した場合、`DEEPL_API_KEY` または `--api-key` でAPIキーが設定されているか確認
- Free APIを使用している場合、`DEEPL_API_URL` が `https://api-free.deepl.com/v2/translate` になっているか確認（デフォルト）
- Pro APIを使用する場合は `--api-url https://api.deepl.com/v2/translate` を指定
- Google翻訳を使用したい場合は `--backend google` を指定（APIキー不要）

### フォントが見つからない警告

```
Font file not found: LiberationSerif-Regular.ttf
```

**解決方法**:
- パッケージのリソースディレクトリ（`src/index_pdf_translation/resources/fonts/`）にフォントファイルが含まれているか確認
- フォールバックフォント（PyMuPDF組み込み）が自動的に使用されるため、翻訳は継続されます

### 出力PDFのテキストが欠落する

**考えられる原因**:
- 入力PDFがスキャン画像（OCRなし）の場合、テキスト抽出ができません
- 複雑なレイアウトのPDFでブロック認識が正確でない可能性があります

**解決方法**: `--debug`オプションでブロック分類を可視化し、認識状況を確認してください。

## フォント要件

本プロジェクトでは以下のフォントを使用しています。これらはパッケージに同梱されており、追加のインストールは不要です。

| フォント | 用途 | ライセンス |
|----------|------|-----------|
| Liberation Serif | 英語テキスト | [SIL Open Font License 1.1](src/index_pdf_translation/resources/fonts/OFL.txt) |
| IPA明朝 (ipam.ttf) | 日本語テキスト | [IPA Font License v1.0](src/index_pdf_translation/resources/fonts/IPA_Font_License_Agreement_v1.0.txt) |

フォントファイルが見つからない場合、PyMuPDF組み込みフォントにフォールバックします。

## 開発者向け情報

### 依存関係

- **PyMuPDF**: PDF処理（テキスト抽出、編集、結合）
- **spaCy**: 自然言語処理（トークン化、言語検出）
- **deep-translator**: Google翻訳バックエンド
- **aiohttp**: 非同期HTTPクライアント（DeepL API通信、オプション）
- **numpy**: 数値計算（ヒストグラム分析）
- **matplotlib**: デバッグ用可視化

### 開発環境セットアップ

```bash
# 開発用依存関係を含めてインストール
uv sync --extra dev

# 構文チェック
uv run python -m py_compile src/index_pdf_translation/cli.py
```

### コード規約

- **型ヒント**: 全ての関数に型アノテーションを使用
- **ドキュメント**: Google スタイルのdocstringを使用
- **ロギング**: `index_pdf_translation.logger`の`get_logger()`を使用
- **ライセンス**: ソースファイルに`# SPDX-License-Identifier: AGPL-3.0-only`を記載

### モジュール拡張

新しい言語を追加する場合、`src/index_pdf_translation/config.py`の`SUPPORTED_LANGUAGES`に追加してください:

```python
SUPPORTED_LANGUAGES: dict[str, LanguageConfig] = {
    "en": {"spacy": "en_core_web_sm"},
    "ja": {"spacy": "ja_core_news_sm"},
    # 新しい言語を追加
    "de": {"spacy": "de_core_news_sm"},
}
```

## ライセンス

このプロジェクトは [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE) の下でライセンスされています。

PyMuPDFがAGPL-3.0でライセンスされているため、本プロジェクトもAGPL-3.0を採用しています。

## 謝辞

- [Google Translate](https://translate.google.com/) - 無料の機械翻訳サービス
- [DeepL](https://www.deepl.com/) - 高品質な機械翻訳API
- [deep-translator](https://github.com/nidhaloff/deep-translator) - 翻訳APIラッパーライブラリ
- [PyMuPDF](https://pymupdf.readthedocs.io/) - 強力なPDF処理ライブラリ
- [spaCy](https://spacy.io/) - 産業用自然言語処理
