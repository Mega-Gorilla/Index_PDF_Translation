# 評価用PDFフィクスチャ

このディレクトリには、レイアウト解析ツールの評価に使用するテスト用PDFを配置します。

## ディレクトリ構造

```
fixtures/
├── README.md (このファイル)
├── simple/           # シンプルなレイアウトのPDF
│   └── *.pdf
├── complex/          # 複雑なレイアウトのPDF
│   └── *.pdf
└── japanese/         # 日本語論文
    └── *.pdf
```

## テストPDFの要件

### 1. シンプルな論文 (simple/)

- 1カラムレイアウト
- 図表が少ない
- 明確なセクション見出し
- 英語または日本語

### 2. 複雑な論文 (complex/)

- 2カラムレイアウト
- 多数の図表
- 数式を含む
- ヘッダー/フッターあり
- 脚注あり

### 3. 日本語論文 (japanese/)

- 日本語の学術論文
- 横書き
- 和英混在

## 注意事項

- 著作権に配慮し、オープンアクセス論文またはCC-BYライセンスの論文を使用
- arXiv, PubMed Central, J-STAGE等からの取得を推奨
- 個人情報を含むPDFは使用しない

## 推奨取得元

- [arXiv](https://arxiv.org/) - プレプリントサーバー (CC-BY)
- [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/) - 生物医学論文
- [J-STAGE](https://www.jstage.jst.go.jp/) - 日本の学術論文

## テスト実行方法

```bash
# 単一PDFの評価
uv run python scripts/evaluate_layout.py tests/evaluation/fixtures/simple/paper.pdf

# 結果をJSONに出力
uv run python scripts/evaluate_layout.py tests/evaluation/fixtures/simple/paper.pdf --output results.json
```
