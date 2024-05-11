# 概要

[Index PDF 翻訳](https://indqx-demo-front.onrender.com/)は、2024年5月31までwebにて翻訳サービスを提供していた、PDF翻訳のソースコードです。

サービス終了に伴い、Index PDF翻訳を、ローカルで実行可能なコードも提供しています。

Index PDF 翻訳は、PDFのフォーマットを崩さずに、翻訳することができる論文向けPDF翻訳サービスです。以下の機能があります。
- 本文自動認識機能により、論文データの数式やタイトルなどの翻訳不要部分を無視して本文のみを翻訳します。
    - 本文認識はブロックの幅、文字数、フォントサイズにより、分析を行います。
- 本文ブロックを認識時、終了記号(.:/など)がない場合、複数ブロックを1ブロックとして翻訳します。これによりブロック間やページ間にて翻訳文が途切れる等の問題を解決します。
- 図や表の説明文のブロックを自動認識し、本文と分割して翻訳します。

<img src="https://github.com/Mega-Gorilla/Index_PDF_Translation/blob/main/images/GMtDCedbsAAIkDO.jpg?raw=true" width = "300" />

# ローカル版のインストール

本、レポジトリをクローン後、
以下のコマンドで必要ライブラリをインストールしてください。実行には、Python 3.11環境が必要です。

### ライブラリーをインストール
```
pip install -r requirements.txt
```

### APIキーの設定

config.pyを開き、以下のDeepL_API_Keyを変更し、[https://www.deepl.com/ja/your-account/keys](https://www.deepl.com/ja/your-account/keys)より取得したDeepL API Keyを入力してください。
また、DeepL API Proユーザーの場合、DeepL_URLをProAPI用URLに変更し保存してください。
```
DeepL_API_Key = "xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxx:fx"
```

### コードの実行

以下のコマンドを実行して、PDF翻訳を実行します。アプリケーションが起動すると、ファイルエクスプローラが表示されます。
翻訳するPDFを選択してください。翻訳が完了すると、./outputに翻訳後のPDFデータが保存されます。
```
 python manual_translate_pdf.py
 ```