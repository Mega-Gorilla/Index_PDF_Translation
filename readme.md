# Index PDF Translation

> **Note**: 本プロジェクトは、2024年5月31日までWebサービスとして提供されていた「Indqx PDF 翻訳」のソースコードです。**Webサーバー機能は削除済み**で、現在はローカルCLIツールとしてのみ動作します。

Index PDF Translationは、PDFのフォーマットを崩さずに翻訳できる論文向けPDF翻訳ツールです。以下の機能があります。
- 本文自動認識機能により、論文データの数式やタイトルなどの翻訳不要部分を無視して本文のみを翻訳します。
    - 本文認識はブロックの幅、文字数、フォントサイズにより、分析を行います。
- 本文ブロックを認識時、終了記号(.:/など)がない場合、複数ブロックを1ブロックとして翻訳します。これによりブロック間やページ間にて翻訳文が途切れる等の問題を解決します。
- 図や表の説明文のブロックを自動認識し、本文と分割して翻訳します。

<img src="https://github.com/Mega-Gorilla/Index_PDF_Translation/blob/main/images/GMtDCedbsAAIkDO.jpg?raw=true" width = "500" />

# インストール

本レポジトリをクローン後、以下のコマンドで必要ライブラリをインストールしてください。
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

### APIキーの設定

config.pyを開き、`DEEPL_API_KEY`を変更し、[https://www.deepl.com/ja/your-account/keys](https://www.deepl.com/ja/your-account/keys)より取得したDeepL API Keyを入力してください。
また、DeepL API Proユーザーの場合、`DEEPL_API_URL`をPro API用URLに変更し保存してください。
```python
DEEPL_API_KEY = "your-api-key"
DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"  # Pro: https://api.deepl.com/v2/translate
```

### フォント要件

本プロジェクトでは以下のフォントを使用しています。これらはリポジトリに同梱されており、追加のインストールは不要です。

| フォント | 用途 | ライセンス |
|----------|------|-----------|
| Liberation Serif | 英語テキスト | [SIL Open Font License 1.1](fonts/OFL.txt) |
| IPA明朝 (ipam.ttf) | 日本語テキスト | [IPA Font License v1.0](fonts/IPA_Font_License_Agreement_v1.0.txt) |

フォントファイルが見つからない場合、PyMuPDF組み込みフォントにフォールバックします。

# 使用方法

## 基本的な使い方

```bash
# uv を使用する場合（推奨）
uv run python translate_pdf.py paper.pdf

# pip を使用する場合
python translate_pdf.py paper.pdf
```

翻訳が完了すると、`./output/translated_<ファイル名>.pdf` に見開きPDF（オリジナル + 翻訳）が保存されます。

## オプション

| オプション | 説明 | デフォルト |
|------------|------|-----------|
| `-o, --output` | 出力ファイルのパス | `./output/translated_<input>.pdf` |
| `-s, --source` | 翻訳元の言語 (en/ja) | `en` |
| `-t, --target` | 翻訳先の言語 (en/ja) | `ja` |
| `--no-logo` | ロゴウォーターマークを無効化 | - |
| `--debug` | デバッグモード（ブロック分類の可視化） | - |

## 使用例

```bash
# 出力ファイルを指定
uv run python translate_pdf.py paper.pdf -o ./result.pdf

# 日本語から英語に翻訳
uv run python translate_pdf.py paper.pdf -s ja -t en

# ロゴなし + デバッグモード
uv run python translate_pdf.py paper.pdf --no-logo --debug
```
