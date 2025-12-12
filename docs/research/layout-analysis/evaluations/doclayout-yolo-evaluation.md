# DocLayout-YOLO 評価レポート

**評価日**: 2025-12-11
**関連Issue**: #31
**PR**: TBD

## 1. 概要

DocLayout-YOLO を使用したドキュメントレイアウト解析の評価結果。
YOLOv10ベースの物体検出モデルで、ドキュメント要素を10クラスに分類。

### 評価目的

1. 見出し（title）と本文（plain_text）の区別精度
2. 表（table）・図（figure）の検出精度
3. 処理速度とGPU効率
4. PyMuPDF4LLM + Layout との比較

## 2. テスト環境

| 項目 | 値 |
|------|-----|
| OS | Linux (Ubuntu) |
| Python | 3.12 |
| doclayout-yolo | 0.0.4 |
| torch | 2.9.1 (CUDA) |
| GPU | NVIDIA RTX 4070 Ti (12GB VRAM) |
| Device | cuda:0 |

## 3. 検出クラス（10種類）

| クラスID | クラス名 | 日本語 | 用途 |
|----------|----------|--------|------|
| 0 | title | タイトル | 見出し検出 |
| 1 | plain_text | 本文 | 翻訳対象 |
| 2 | abandoned_text | 削除テキスト | 除外候補（ヘッダー/フッター等） |
| 3 | figure | 図 | 除外 |
| 4 | figure_caption | 図のキャプション | 翻訳対象（オプション） |
| 5 | table | 表 | 特殊処理 |
| 6 | table_caption | 表のキャプション | 翻訳対象（オプション） |
| 7 | table_footnote | 表の脚注 | 翻訳対象（オプション） |
| 8 | isolated_formula | 数式 | 除外 |
| 9 | formula_caption | 数式のキャプション | 翻訳対象（オプション） |

## 4. テスト対象PDF

### 4.1 fixtures/ (シンプルなPDF)

| ファイル | ページ数 | 特徴 |
|---------|---------|------|
| sample_autogen.pdf | 1 | AutoGen論文、図含む |
| sample_cot.pdf | 1 | Chain-of-Thought論文 |
| sample_llama.pdf | 1 | LLaMA論文 |

### 4.2 pdf_sample/ (arXiv論文)

| ファイル | ページ数 | 特徴 |
|---------|---------|------|
| 2201.11903v6.pdf | 43 | 長文論文、図表多め |
| 2302.13971v1.pdf | 27 | 中程度の論文 |
| 2308.08155v2.pdf | 43 | コードブロック含む |

## 5. 評価結果

### 5.1 全体サマリー

| 指標 | DocLayout-YOLO |
|------|----------------|
| 総ブロック数 | 1,334 |
| 総処理時間 | 18.58s |
| title検出 | 165 |
| plain_text検出 | 843 |
| abandoned_text検出 | 129 |
| figure検出 | 36 |
| table検出 | 63 |
| isolated_formula検出 | 16 |

### 5.2 PDF別詳細結果

#### fixtures/ (シンプルなPDF)

| PDF | ブロック | title | plain_text | abandoned | figure | table | 時間 |
|-----|---------|-------|------------|-----------|--------|-------|------|
| sample_autogen | 15 | 2 | 7 | 4 | 1 | 0 | 1.35s |
| sample_cot | 13 | 2 | 6 | 3 | 1 | 0 | 0.79s |
| sample_llama | 15 | 3 | 9 | 3 | 0 | 0 | 0.65s |

#### pdf_sample/ (arXiv論文)

| PDF | ページ | ブロック | title | plain_text | abandoned | figure | table | 時間 |
|-----|--------|---------|-------|------------|-----------|--------|-------|------|
| 2201.11903v6 | 43 | 484 | 49 | 306 | 51 | 12 | 25 | 6.15s |
| 2302.13971v1 | 27 | 372 | 57 | 257 | 10 | 2 | 17 | 3.85s |
| 2308.08155v2 | 43 | 435 | 39 | 239 | 58 | 20 | 21 | 5.77s |

### 5.3 処理速度分析

| PDF | ページ数 | 処理時間 | ページ/秒 |
|-----|---------|---------|----------|
| sample_autogen | 1 | 1.35s | 0.74 |
| sample_cot | 1 | 0.79s | 1.27 |
| sample_llama | 1 | 0.65s | 1.54 |
| 2201.11903v6 | 43 | 6.15s | **6.99** |
| 2302.13971v1 | 27 | 3.85s | **7.01** |
| 2308.08155v2 | 43 | 5.77s | **7.45** |

**平均**: 約 7.0 ページ/秒（GPU使用、長文PDF）

※ 1ページPDFは初期化オーバーヘッドが大きいため遅くなる

## 6. PyMuPDF4LLM + Layout との比較

### 6.1 処理速度比較

| PDF | PyMuPDF4LLM | DocLayout-YOLO | 速度比 |
|-----|-------------|----------------|--------|
| sample_llama (1p) | 0.24s | 0.65s | 0.37x (YOLO遅い) |
| 2201.11903v6 (43p) | 32.96s | 6.15s | **5.4x (YOLO速い)** |
| 2302.13971v1 (27p) | 15.44s | 3.85s | **4.0x (YOLO速い)** |
| 2308.08155v2 (43p) | 34.85s | 5.77s | **6.0x (YOLO速い)** |

**結論**: 長文PDFではDocLayout-YOLOが4〜6倍高速

### 6.2 検出クラス比較

| 機能 | PyMuPDF4LLM | DocLayout-YOLO |
|------|-------------|----------------|
| 見出し検出 | ✅ H1-H4区別 | ✅ title統一 |
| 本文検出 | ✅ body | ✅ plain_text |
| ヘッダー/フッター | ✅ page-header | ✅ abandoned_text |
| 図検出 | ❌ なし | ✅ figure + caption |
| 表検出 | ✅ table | ✅ table + caption + footnote |
| 数式検出 | ❌ なし | ✅ isolated_formula |
| リスト検出 | ✅ list | ❌ なし |
| コード検出 | ✅ code | ❌ なし |

### 6.3 座標情報比較

| 項目 | PyMuPDF4LLM | DocLayout-YOLO |
|------|-------------|----------------|
| bbox | ✅ to_json()で取得 | ✅ 常に取得 |
| 信頼度 | ❌ なし | ✅ 0.0〜1.0 |
| テキスト | ✅ 取得可能 | ❌ 別途抽出必要 |
| フォント情報 | ✅ size, font名 | ❌ なし |

## 7. 定性的評価

### 7.1 良かった点

| 項目 | 評価 | 詳細 |
|------|------|------|
| 処理速度 | ✅ 優秀 | 長文PDFで4〜6倍高速 |
| 図・表検出 | ✅ 優秀 | caption、footnoteも検出 |
| 数式検出 | ✅ 対応 | isolated_formula検出 |
| 座標情報 | ✅ 常に取得 | bbox + confidence |
| ヘッダー/フッター | ✅ 検出 | abandoned_textとして |

### 7.2 問題点・制限

| 項目 | 評価 | 詳細 |
|------|------|------|
| GPU必須 | ⚠️ | CPUでは大幅に遅くなる |
| テキスト抽出 | ❌ 不可 | 別途PyMuPDF等が必要 |
| 見出しレベル | ❌ 区別不可 | H1-H4の区別なし |
| リスト検出 | ❌ なし | plain_textに含まれる |
| コード検出 | ❌ なし | plain_textに含まれる |
| 依存関係 | ⚠️ 重い | torch, CUDA等 |

### 7.3 本プロジェクトへの適合性

| 観点 | 評価 | 理由 |
|------|------|------|
| 見出し検出要件 | ⚠️ 部分適合 | titleは検出、レベル区別不可 |
| 本文抽出要件 | ⚠️ 部分適合 | 領域検出のみ、テキスト別途 |
| 翻訳ワークフロー | ⚠️ 要検討 | PyMuPDFとの組み合わせ必要 |
| 処理速度 | ✅ 優秀 | 長文PDFで高速 |
| ライセンス | ✅ 互換 | AGPL-3.0 |

## 8. 結論

### 推奨度

- [ ] **単体では非推奨** - テキスト抽出機能がない
- [x] **PyMuPDFとの組み合わせで推奨** - 高速な領域検出として

### 理由

1. **長文PDFで高速**: 43ページPDFを6秒で処理（PyMuPDF4LLMの5〜6倍）
2. **豊富な検出クラス**: 図、表、数式、キャプションを区別
3. **座標+信頼度**: 全ブロックにbboxと信頼度スコア
4. **ライセンス互換**: AGPL-3.0

### 使用方法の提案

```python
# DocLayout-YOLOで領域検出 + PyMuPDFでテキスト抽出
import fitz
from doclayout_yolo import YOLOv10

# 1. DocLayout-YOLOで領域検出
model = YOLOv10("doclayout_yolo_docstructbench_imgsz1024.pt")
results = model.predict(page_image, imgsz=1024, conf=0.2)

# 2. 検出領域からテキスト抽出（PyMuPDF使用）
doc = fitz.open(pdf_path)
page = doc[page_num]

for box in results[0].boxes:
    bbox = box.xyxy[0].tolist()
    block_type = CLASSES[int(box.cls[0])]

    if block_type in ["plain_text", "title"]:
        # bboxからテキスト抽出
        text = page.get_text(clip=fitz.Rect(bbox))
        # 翻訳処理...
```

## 9. PyMuPDF4LLM vs DocLayout-YOLO 総合比較

| 項目 | PyMuPDF4LLM | DocLayout-YOLO | 推奨 |
|------|-------------|----------------|------|
| **処理速度** | 1.3p/s | 7.0p/s | DocLayout |
| **見出しレベル** | H1-H4 | なし | PyMuPDF4LLM |
| **テキスト抽出** | ✅ | ❌ | PyMuPDF4LLM |
| **図・表・数式** | 一部 | ✅ | DocLayout |
| **信頼度スコア** | ❌ | ✅ | DocLayout |
| **GPU不要** | ✅ | ❌ | PyMuPDF4LLM |
| **依存関係** | 軽い | 重い | PyMuPDF4LLM |

### 最終推奨

**Issue #31 の目的（見出し検出・ブロック分類改善）には PyMuPDF4LLM を推奨**

理由:
1. テキスト抽出が可能（翻訳ワークフローに直接統合）
2. 見出しレベル（H1-H4）を区別
3. GPU不要で軽量

DocLayout-YOLOは、高速な領域検出や図・表の検出が重要な場合の補助ツールとして検討。

## 10. 出力ファイル一覧

```
tests/evaluation/outputs/
├── DocLayout_YOLO/
│   ├── sample_autogen/
│   │   ├── metadata.json
│   │   ├── blocks.json
│   │   └── visualizations/
│   │       └── page_000.png
│   ├── sample_cot/
│   ├── sample_llama/
│   ├── 2201.11903v6/
│   ├── 2302.13971v1/
│   └── 2308.08155v2/
└── doclayout_yolo_summary.md
```

## 11. 参考

- [DocLayout-YOLO GitHub](https://github.com/opendatalab/DocLayout-YOLO)
- [DocLayout-YOLO Paper (arXiv:2410.12628)](https://arxiv.org/abs/2410.12628)
- [HuggingFace Model](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench)
- [評価スクリプト](../../../scripts/evaluate_doclayout_yolo.py)
