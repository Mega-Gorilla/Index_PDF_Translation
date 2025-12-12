# YOLO-DocLayNet 評価結果サマリー

**評価日時**: 2025-12-12 13:59:38

## 1. 評価概要

- 評価PDF数: 3
- モデル: yolov11l-doclaynet
- デバイス: cuda:0

## 2. 検出クラス (DocLayNet 11カテゴリ)

| クラスID | クラス名 | 説明 |
|----------|----------|------|
| 0 | Caption | - |
| 1 | Footnote | - |
| 2 | Formula | - |
| 3 | List-item | - |
| 4 | Page-footer | - |
| 5 | Page-header | - |
| 6 | Picture | - |
| 7 | Section-header | - |
| 8 | Table | - |
| 9 | Text | - |
| 10 | Title | - |

## 3. PDF別結果

| PDF | ページ数 | ブロック数 | 処理時間 | タイプ分布 |
|-----|---------|-----------|---------|-----------|
| 2201.11903v6 | 43 | 697 | 6.171s | Caption: 12, Footnote: 7, List-item: 140, Page-footer: 40, Page-header: 1, Picture: 11, Section-header: 66, Table: 8, Text: 412 |
| 2302.13971v1 | 27 | 412 | 3.597s | Caption: 7, Footnote: 6, List-item: 93, Page-header: 2, Picture: 6, Section-header: 48, Table: 17, Text: 231, Title: 2 |
| 2308.08155v2 | 43 | 368 | 5.073s | Caption: 17, Footnote: 7, Formula: 1, List-item: 94, Page-footer: 37, Page-header: 4, Picture: 24, Section-header: 34, Table: 17, Text: 132, Title: 1 |

**合計**: 1477 ブロック, 14.841s

## 4. 処理速度

| PDF | ページ数 | 処理時間 | ページ/秒 |
|-----|---------|---------|----------|
| 2201.11903v6 | 43 | 6.171s | 6.97 |
| 2302.13971v1 | 27 | 3.597s | 7.51 |
| 2308.08155v2 | 43 | 5.073s | 8.48 |
