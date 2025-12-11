# DocLayout-YOLO 評価結果サマリー

**評価日時**: 2025-12-11 12:36:31

## 1. 評価概要

- 評価PDF数: 3
- デバイス: cuda:0

## 2. 検出クラス

| クラスID | クラス名 | 説明 |
|----------|----------|------|
| 0 | title | タイトル |
| 1 | plain_text | 本文 |
| 2 | abandoned_text | 削除テキスト |
| 3 | figure | 図 |
| 4 | figure_caption | 図のキャプション |
| 5 | table | 表 |
| 6 | table_caption | 表のキャプション |
| 7 | table_footnote | 表の脚注 |
| 8 | isolated_formula | 数式 |
| 9 | formula_caption | 数式のキャプション |

## 3. PDF別結果

| PDF | ページ数 | ブロック数 | 処理時間 | タイプ分布 |
|-----|---------|-----------|---------|-----------|
| 2201.11903v6 | 43 | 484 | 6.154s | abandoned_text: 51, figure: 12, figure_caption: 12, plain_text: 306, table: 25, table_caption: 29, title: 49 |
| 2302.13971v1 | 27 | 372 | 3.854s | abandoned_text: 10, figure: 2, figure_caption: 9, isolated_formula: 1, plain_text: 257, table: 17, table_caption: 12, table_footnote: 7, title: 57 |
| 2308.08155v2 | 43 | 435 | 5.774s | abandoned_text: 58, figure: 20, figure_caption: 20, formula_caption: 1, isolated_formula: 15, plain_text: 239, table: 21, table_caption: 19, table_footnote: 3, title: 39 |

**合計**: 1291 ブロック, 15.782s

## 4. 処理速度

| PDF | ページ数 | 処理時間 | ページ/秒 |
|-----|---------|---------|----------|
| 2201.11903v6 | 43 | 6.154s | 6.99 |
| 2302.13971v1 | 27 | 3.854s | 7.01 |
| 2308.08155v2 | 43 | 5.774s | 7.45 |