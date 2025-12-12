# YOLO-DocLayNet 評価レポート

**評価日**: 2025-12-12
**関連Issue**: #40

## 1. 概要

YOLO-DocLayNet は DocLayNet データセットで訓練された YOLO シリーズのモデル群です。hantian/yolo-doclaynet リポジトリで YOLOv8〜v12 の各バージョンが提供されています。

### 基本情報

| 項目 | 値 |
|------|-----|
| 開発元 | hantian (Hugging Face) |
| ライセンス | **AGPL-3.0** (Ultralytics) |
| ベースモデル | YOLOv8〜v12 |
| カテゴリ数 | 11 (DocLayNet) |
| 公称精度 | 79.4% mAP (YOLOv10x) |

### 利用可能なモデル

| シリーズ | バリエーション |
|---------|---------------|
| YOLOv8 | n, s, m, l, x |
| YOLOv9 | t, s, m |
| YOLOv10 | n, s, m, b, l |
| YOLOv11 | n, s, m, l |
| YOLOv12 | n, s, m, l |

### インストール

```bash
pip install ultralytics huggingface_hub

# モデルのダウンロード
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="hantian/yolo-doclaynet",
    filename="yolov11l-doclaynet.pt"
)
```

## 2. 評価環境

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4070 Ti |
| CUDA | 13.0 |
| モデル | yolov11l-doclaynet |
| 評価PDF数 | 3 |
| 総ページ数 | 113 |

## 3. 検出クラス (DocLayNet 11カテゴリ)

DocLayNetデータセットは以下の11カテゴリを定義しています：

| クラスID | クラス名 | 説明 | 検出数 |
|---------|---------|------|--------|
| 0 | Caption | 図表のキャプション | 36 |
| 1 | Footnote | 脚注 | 20 |
| 2 | Formula | 数式 | 1 |
| 3 | List-item | リスト項目 | 327 |
| 4 | Page-footer | ページフッター | 77 |
| 5 | Page-header | ページヘッダー | 7 |
| 6 | Picture | 図/画像 | 41 |
| 7 | Section-header | セクション見出し | 148 |
| 8 | Table | 表 | 42 |
| 9 | Text | 本文テキスト | 775 |
| 10 | Title | タイトル | 3 |

## 4. 評価結果

### 4.1 PDF別検出結果

| PDF | ページ数 | ブロック数 | 処理時間 | ページ/秒 |
|-----|---------|-----------|---------|----------|
| 2201.11903v6 (LLaMA) | 43 | 697 | 6.171s | 6.97 |
| 2302.13971v1 (LLaMA2) | 27 | 412 | 3.597s | 7.51 |
| 2308.08155v2 (Code Llama) | 43 | 368 | 5.073s | 8.48 |
| **合計** | **113** | **1477** | **14.841s** | **7.61** |

### 4.2 処理速度

- **平均処理速度**: 7.61 ページ/秒
- **1ページあたり処理時間**: 約131ms

### 4.3 タイプ別検出分布

```
Text:           775 (52.5%)
List-item:      327 (22.1%)
Section-header: 148 (10.0%)
Page-footer:     77 (5.2%)
Table:           42 (2.8%)
Picture:         41 (2.8%)
Caption:         36 (2.4%)
Footnote:        20 (1.4%)
Page-header:      7 (0.5%)
Title:            3 (0.2%)
Formula:          1 (0.1%)
```

## 5. DocLayout-YOLO との比較

| 項目 | YOLO-DocLayNet (v11l) | DocLayout-YOLO |
|------|----------------------|----------------|
| ライセンス | AGPL-3.0 | AGPL-3.0 |
| カテゴリ数 | 11 | 10 |
| 総ブロック数 | **1477** | 1291 |
| 処理時間 | **14.8s** | 15.8s |
| ページ/秒 | **7.61** | 7.16 |
| 数式検出 | Formula (1) | isolated_formula (16) |
| リスト検出 | **List-item (327)** | なし |

## 6. PP-DocLayout との比較

| 項目 | YOLO-DocLayNet | PP-DocLayout |
|------|---------------|--------------|
| ライセンス | AGPL-3.0 | **Apache 2.0** |
| カテゴリ数 | 11 | **23** |
| 総ブロック数 | **1477** | 1013 |
| 処理時間 | **14.8s** | 22.3s |
| ページ/秒 | **7.61** | 5.07 |
| フレームワーク | **PyTorch** | PaddlePaddle |

## 7. 長所と短所

### 長所

1. **高速処理**: PP-DocLayoutより約50%高速
2. **PyTorchベース**: 既存コードとの統合が容易
3. **複数バージョン**: v8〜v12まで選択可能
4. **リスト検出**: List-itemカテゴリによるリスト項目の検出
5. **Ultralytics統合**: 豊富なドキュメントとコミュニティサポート

### 短所

1. **AGPL-3.0ライセンス**: Ultralytics依存のため商用利用に制約
2. **カテゴリ数**: 11カテゴリのみ（PP-DocLayoutの約半分）
3. **数式検出が弱い**: Formulaの検出数が極めて少ない(1件)
4. **アルゴリズム未対応**: algorithm等の特殊カテゴリなし

## 8. 本プロジェクトへの適合性

### 適合する点

- **ライセンス互換**: 本プロジェクト(AGPL-3.0)と同一ライセンス
- **高速処理**: 大量PDF処理に適している
- **PyTorch**: 既存コードベースとの親和性が高い
- **リスト検出**: List-itemの検出は論文構造理解に有用

### 課題

- **数式検出**: Formulaの検出精度が低く、数式の翻訳除外に不適
- **カテゴリ不足**: abstract, reference等の学術論文特有のカテゴリなし

## 9. モデルバージョン比較（参考）

hantian/yolo-doclaynetで提供されているモデルの比較：

| モデル | サイズ | 推定精度 |
|--------|--------|---------|
| yolov8x | 137 MB | 高 |
| yolov10l | 52 MB | 高 |
| yolov11l | 51 MB | 高 |
| yolov12l | 54 MB | 最新 |

本評価ではyolov11lを使用しましたが、yolov12lも同等の性能が期待できます。

## 10. 結論

YOLO-DocLayNetは**高速処理**と**PyTorchベース**が最大の強みです。本プロジェクト(AGPL-3.0)とライセンスが同一のため、統合に問題はありません。

ただし、**数式検出の精度が低い**点は学術論文翻訳において重大な課題です。数式を本文と分離して翻訳対象から除外する機能が重要な本プロジェクトでは、PP-DocLayoutの方が適している可能性があります。

**推奨**: 速度重視の場合はYOLO-DocLayNet、精度・カテゴリ重視の場合はPP-DocLayoutを使用。

## 11. 参考リンク

- [Hugging Face - hantian/yolo-doclaynet](https://huggingface.co/hantian/yolo-doclaynet)
- [GitHub - ppaanngggg/yolo-doclaynet](https://github.com/ppaanngggg/yolo-doclaynet)
- [DocLayNet Dataset](https://github.com/DS4SD/DocLayNet)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
