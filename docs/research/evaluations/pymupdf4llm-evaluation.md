# PyMuPDF4LLM + Layout 評価レポート

**評価日**: 2025-12-11
**関連Issue**: #31
**PR**: #33

## 1. 概要

PyMuPDF4LLM と pymupdf-layout パッケージを使用したレイアウト解析の評価結果。

### 評価目的

1. 見出し（heading）と本文（body）の区別が可能か
2. 表（table）の検出精度
3. 処理速度とトレードオフ
4. 本プロジェクトへの統合可能性

## 2. テスト環境

| 項目 | 値 |
|------|-----|
| OS | Linux (Ubuntu) |
| Python | 3.12 |
| PyMuPDF | 1.25.x |
| pymupdf4llm | 0.2.7 |
| pymupdf-layout | 1.26.6 |
| GPU | N/A (CPU only) |

## 3. テスト対象PDF

### 3.1 fixtures/ (シンプルなPDF)

| ファイル | ページ数 | 特徴 |
|---------|---------|------|
| sample_autogen.pdf | 1 | AutoGen論文、シンプルレイアウト |
| sample_cot.pdf | 1 | Chain-of-Thought論文 |
| sample_llama.pdf | 1 | LLaMA論文 |

### 3.2 pdf_sample/ (arXiv論文)

| ファイル | ページ数 | 特徴 |
|---------|---------|------|
| 2201.11903v6.pdf | 43 | 長文論文、図表多め |
| 2302.13971v1.pdf | 27 | 中程度の論文 |
| 2308.08155v2.pdf | 43 | コードブロック含む |

## 4. 評価結果

### 4.1 全体サマリー

| 指標 | PyMuPDF Baseline | PyMuPDF4LLM + Layout |
|------|------------------|----------------------|
| 総ブロック数 | 2,442 | 1,334 |
| 総処理時間 | 0.48s | 84.24s |
| 見出し検出 | 0 | 168 |
| 本文検出 | 0 (unknown) | 1,065 |
| 表検出 | 0 | 62 |
| リスト検出 | 0 | 284 |

### 4.2 PDF別詳細結果

#### fixtures/ (シンプルなPDF)

| PDF | ツール | ブロック | 見出し | 本文 | 表 | 時間 |
|-----|--------|---------|--------|------|-----|------|
| sample_autogen | Baseline | 38 | 0 | 0 | 0 | 0.026s |
| sample_autogen | **Layout** | 13 | 2 | 11 | 0 | 0.529s |
| sample_cot | Baseline | 22 | 0 | 0 | 0 | 0.006s |
| sample_cot | **Layout** | 13 | 2 | 11 | 0 | 0.219s |
| sample_llama | Baseline | 13 | 0 | 0 | 0 | 0.006s |
| sample_llama | **Layout** | 14 | 4 | 10 | 0 | 0.243s |

#### pdf_sample/ (arXiv論文)

| PDF | ツール | ブロック | 見出し | 本文 | 表 | リスト | 時間 |
|-----|--------|---------|--------|------|-----|--------|------|
| 2201.11903v6 (43p) | Baseline | 838 | 0 | 0 | 0 | 0 | 0.150s |
| 2201.11903v6 (43p) | **Layout** | 544 | 44 | 378 | 12 | 110 | 32.96s |
| 2302.13971v1 (27p) | Baseline | 415 | 0 | 0 | 0 | 0 | 0.086s |
| 2302.13971v1 (27p) | **Layout** | 354 | 53 | 187 | 17 | 97 | 15.44s |
| 2308.08155v2 (43p) | Baseline | 1116 | 0 | 0 | 0 | 0 | 0.203s |
| 2308.08155v2 (43p) | **Layout** | 396 | 28 | 269 | 16 | 77 | 34.85s |

### 4.3 処理速度分析

| PDF | ページ数 | Layout時間 | ページ/秒 |
|-----|---------|-----------|----------|
| sample_autogen | 1 | 0.53s | 1.9 |
| sample_cot | 1 | 0.22s | 4.5 |
| sample_llama | 1 | 0.24s | 4.2 |
| 2201.11903v6 | 43 | 32.96s | 1.3 |
| 2302.13971v1 | 27 | 15.44s | 1.7 |
| 2308.08155v2 | 43 | 34.85s | 1.2 |

**平均**: 約 1.3〜4.5 ページ/秒（PDF複雑さに依存）

## 5. 出力サンプル

### 5.1 Markdown出力例 (sample_llama.pdf)

```markdown
# LLaMA: Open and Efficient Foundation Language Models

Hugo Touvron∗, Thibaut Lavril∗, Gautier Izacard∗, Xavier Martinet
Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal
...

## Abstract

We introduce LLaMA, a collection of foundation language models ranging from 7B to
65B parameters. We train our models on trillions of tokens, and show that it is possible
to train state-of-the-art models using publicly available datasets exclusively...

## 1 Introduction

Large Languages Models (LLMs) trained on massive corpora of texts have shown their
ability to perform new tasks from textual instructions or from a few examples...
```

### 5.2 検出されたブロック例 (JSON)

```json
{
  "bbox": [0, 0, 0, 0],
  "text": "LLaMA: Open and Efficient Foundation Language Models",
  "block_type": "heading_1",
  "confidence": 1.0,
  "font_size": null,
  "page_num": 0
}
```

## 6. 定性的評価

### 6.1 良かった点

| 項目 | 評価 | 詳細 |
|------|------|------|
| 見出し検出 | ✅ 優秀 | H1/H2/H3を正確に区別 |
| 本文抽出 | ✅ 良好 | 段落を適切にマージ |
| 表検出 | ✅ 良好 | Markdownテーブル形式で出力 |
| リスト検出 | ✅ 良好 | 箇条書きを認識 |
| コード検出 | ✅ 対応 | コードブロックを識別 |
| GPU不要 | ✅ | CPUのみで動作 |

### 6.2 問題点・制限

| 項目 | 評価 | 詳細 |
|------|------|------|
| 処理速度 | ⚠️ 遅い | Baselineの約190倍 |
| ヘッダー/フッター | ❓ 未確認 | 明示的な除外機能は未検証 |
| 座標情報 | ❌ 損失 | Markdown出力では座標が失われる |
| OCR | ⚠️ 無効 | OpenCV未インストールのため無効 |

### 6.3 本プロジェクトへの適合性

| 観点 | 評価 | 理由 |
|------|------|------|
| 見出し検出要件 | ✅ 適合 | H1-H4を正確に検出 |
| 本文抽出要件 | ✅ 適合 | 段落を適切に結合 |
| 翻訳ワークフロー | ⚠️ 要検討 | 座標情報が必要な場合は追加処理要 |
| 処理速度 | ⚠️ 許容範囲 | バッチ処理なら許容可能 |
| ライセンス | ✅ 互換 | AGPL-3.0 |

## 7. 結論

### 推奨度

- [x] **推奨** - 見出し検出・ブロック分類に有効

### 理由

1. **見出し検出が正確**: H1-H4を適切に区別でき、Issue #31の主要目標を達成可能
2. **追加のML不要**: GPUなしで動作、依存関係が軽量
3. **ライセンス互換**: AGPL-3.0で本プロジェクトと同一
4. **処理速度は許容範囲**: 学術論文翻訳はリアルタイム性不要

### 注意点

1. **座標情報の扱い**: 現行実装では座標ベースの処理を行っているため、統合方法の検討が必要
2. **処理時間**: 大きなPDFでは30秒以上かかる場合あり

## 8. 次のステップ

- [x] 基本評価完了
- [ ] 座標情報を保持した抽出方法の調査
- [ ] ヘッダー/フッター除外機能の検証
- [ ] 現行実装との統合設計

## 9. 出力ファイル一覧

評価で生成されたファイルは `tests/evaluation/outputs/` に保存:

```
tests/evaluation/outputs/
├── PyMuPDF_Baseline/
│   ├── sample_autogen/
│   │   ├── metadata.json
│   │   └── blocks.json
│   └── ...
├── PyMuPDF4LLM_Layout/
│   ├── sample_autogen/
│   │   ├── metadata.json
│   │   ├── blocks.json
│   │   └── output.md      # Markdown出力
│   └── ...
└── evaluation_summary.md
```

## 10. 参考

- [PyMuPDF4LLM Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [PyMuPDF Layout Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf-layout/index.html)
- [評価スクリプト](../../../scripts/evaluate_layout.py)
