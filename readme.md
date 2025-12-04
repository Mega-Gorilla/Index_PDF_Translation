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

# コードの実行

以下のコマンドを実行して、PDF翻訳を実行します。アプリケーションが起動すると、ファイルエクスプローラが表示されます。
翻訳するPDFを選択してください。翻訳が完了すると、./outputに翻訳後のPDFデータが保存されます。
```
 python manual_translate_pdf.py
 ```
