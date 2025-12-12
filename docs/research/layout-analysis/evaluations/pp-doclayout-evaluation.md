# PP-DocLayout 評価レポート

**評価日**: 2025-12-12
**関連Issue**: #40

## 1. 概要

PP-DocLayout は PaddlePaddle/Baidu が開発したドキュメントレイアウト検出モデルです。RT-DETR をベースとし、23カテゴリの詳細な分類が可能です。

### 基本情報

| 項目 | 値 |
|------|-----|
| 開発元 | PaddlePaddle (Baidu) |
| ライセンス | **Apache 2.0** |
| ベースモデル | RT-DETR-L |
| カテゴリ数 | 23 |
| 公称精度 | 90.4% mAP@0.5 |
| リリース日 | 2025年3月 |

### インストール

```bash
# CUDA 12.6の場合
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install paddleocr
```

## 2. 評価環境

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4070 Ti |
| CUDA | 13.0 |
| モデル | PP-DocLayout-L |
| 評価PDF数 | 3 |
| 総ページ数 | 113 |

## 3. 検出クラス (23カテゴリ)

PP-DocLayoutは以下の23カテゴリを検出可能です：

| カテゴリ | 説明 | 検出数 |
|---------|------|--------|
| text | 本文テキスト | 455 |
| paragraph_title | 段落タイトル | 115 |
| formula | 数式 | 97 |
| number | ページ番号等 | 83 |
| table | 表 | 64 |
| table_title | 表タイトル | 68 |
| reference | 参考文献 | 19 |
| footnote | 脚注 | 18 |
| image | 画像 | 20 |
| figure_title | 図タイトル | 22 |
| chart | グラフ/チャート | 15 |
| chart_title | チャートタイトル | 16 |
| header | ヘッダー | 6 |
| footer | フッター | 1 |
| aside_text | サイドテキスト | 3 |
| algorithm | アルゴリズム | 4 |
| abstract | 要約 | 3 |
| doc_title | 文書タイトル | 3 |
| formula_number | 数式番号 | 1 |

## 4. 評価結果

### 4.1 PDF別検出結果

| PDF | ページ数 | ブロック数 | 処理時間 | ページ/秒 |
|-----|---------|-----------|---------|----------|
| 2201.11903v6 (LLaMA) | 43 | 369 | 8.583s | 5.01 |
| 2302.13971v1 (LLaMA2) | 27 | 250 | 5.522s | 4.89 |
| 2308.08155v2 (Code Llama) | 43 | 394 | 8.172s | 5.26 |
| **合計** | **113** | **1013** | **22.276s** | **5.07** |

### 4.2 処理速度

- **平均処理速度**: 5.07 ページ/秒
- **1ページあたり処理時間**: 約197ms

### 4.3 タイプ別検出分布

```
text:            455 (44.9%)
paragraph_title: 115 (11.4%)
formula:          97 (9.6%)
number:           83 (8.2%)
table_title:      68 (6.7%)
table:            64 (6.3%)
figure_title:     22 (2.2%)
image:            20 (2.0%)
reference:        19 (1.9%)
footnote:         18 (1.8%)
chart_title:      16 (1.6%)
chart:            15 (1.5%)
その他:           21 (2.1%)
```

## 5. DocLayout-YOLO との比較

| 項目 | PP-DocLayout | DocLayout-YOLO |
|------|-------------|----------------|
| ライセンス | **Apache 2.0** | AGPL-3.0 |
| カテゴリ数 | **23** | 10 |
| 総ブロック数 | 1013 | 1291 |
| 処理時間 | 22.3s | 15.8s |
| ページ/秒 | 5.07 | **7.16** |
| 数式検出 | formula (97) | isolated_formula (16) |
| テーブル検出 | table + table_title (132) | table + table_caption (95) |

## 6. 長所と短所

### 長所

1. **Apache 2.0ライセンス**: 商用利用が自由
2. **23カテゴリの詳細分類**: formula, algorithm, abstract等の細かい分類が可能
3. **高精度**: 公称90.4% mAP
4. **数式検出**: 数式を独立カテゴリとして検出（本文と分離可能）
5. **最新モデル**: 2025年3月リリース

### 短所

1. **処理速度**: DocLayout-YOLOより約30%遅い
2. **PaddlePaddleフレームワーク依存**: PyTorchではなくPaddleを使用
3. **CUDAバージョン制約**: 特定のCUDAバージョンに対応したパッケージが必要
4. **検出数**: DocLayout-YOLOより検出ブロック数が少ない（より保守的）

## 7. 本プロジェクトへの適合性

### 適合する点

- **商用利用可能**: Apache 2.0でライセンス問題なし
- **数式分離**: 数式を独立検出することで翻訳対象から除外可能
- **詳細分類**: 23カテゴリによる細かい制御が可能

### 課題

- **フレームワーク**: 既存のPyTorchベースコードとの統合に工夫が必要
- **処理速度**: 大量のPDF処理時にボトルネックになる可能性

## 8. 結論

PP-DocLayoutは**Apache 2.0ライセンス**と**23カテゴリの詳細分類**が最大の強みです。特に数式やアルゴリズムを独立カテゴリとして検出できる点は、学術論文の翻訳において非常に有用です。

処理速度はDocLayout-YOLOより劣りますが、ライセンスの自由度と分類の詳細さを考慮すると、本プロジェクトの**第一候補**として推奨します。

## 9. 参考リンク

- [Hugging Face - PP-DocLayout-L](https://huggingface.co/PaddlePaddle/PP-DocLayout-L)
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleX Documentation](https://github.com/PaddlePaddle/PaddleX)
