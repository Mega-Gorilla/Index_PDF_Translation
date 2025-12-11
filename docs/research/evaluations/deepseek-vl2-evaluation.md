# DeepSeek-VL2 評価レポート

**評価日時**: 2025-12-11
**評価対象**: DeepSeek-VL2-Tiny (deepseek-ai/deepseek-vl2-tiny)
**評価目的**: Issue #31 ブロック分類改善のためのOCR/レイアウト検出ツール評価

## 1. 概要

DeepSeek-VL2は、DeepSeek社が開発したVision-Language Model (VLM)で、画像理解とテキスト生成を組み合わせた機能を提供します。OCR機能とレイアウト検出機能を持ち、学術論文PDFの翻訳に適用可能かを評価しました。

## 2. モデル情報

| 項目 | 詳細 |
|------|------|
| モデル名 | DeepSeek-VL2-Tiny |
| モデルID | deepseek-ai/deepseek-vl2-tiny |
| パラメータ数 | 約3.37B |
| VRAM使用量 | 約6.4GB (BF16) |
| ライセンス | MIT (コード), DeepSeek License (モデル) |
| リポジトリ | https://github.com/deepseek-ai/DeepSeek-VL2 |

## 3. 環境・互換性

### 3.1 要求環境

DeepSeek-VL2の公式要件：
```
torch==2.0.1
transformers==4.38.2
xformers>=0.0.21
timm>=0.9.16
accelerate
sentencepiece
attrdict
einops
```

### 3.2 互換性問題

**重大な問題**: DeepSeek-VL2は`transformers==4.38.2`を厳格に要求し、新しいバージョン（例: 4.57.3）では動作しません。

発見された問題：

1. **`LlamaFlashAttention2`の削除**
   - transformers 4.50以降で`LlamaFlashAttention2`クラスが削除された
   - 影響: モジュールのインポート失敗
   - 対処: 条件付きインポートでパッチ可能

2. **`GenerationMixin`の継承変更**
   - transformers 4.50以降で`PreTrainedModel`が`GenerationMixin`を継承しなくなった
   - 影響: `model.generate()`メソッドが使用不可
   - 対処: クラス定義の修正が必要（複雑）

3. **`generation_config`の初期化**
   - 新しいtransformersでは`generation_config`が必須
   - 影響: generate呼び出し時にAttributeError
   - 対処: モデルロード後に手動で設定（部分的に解決）

4. **Config属性の不一致**
   - `DeepseekVLV2Config`に`num_hidden_layers`属性がない
   - 影響: DynamicCacheの初期化失敗
   - 対処: 根本的なコード修正が必要

### 3.3 解決策

**隔離環境での評価**を実施：
- `/tmp/deepseek-vl2-eval-env/` に専用仮想環境を作成
- transformers==4.38.2をインストール
- プロジェクト本体の環境には影響なし

## 4. ライセンス互換性

| ライセンス | 対象 | AGPL-3.0との互換性 |
|------------|------|-------------------|
| MIT | コード | ✅ 互換 |
| DeepSeek License | モデル重み | ⚠️ 要確認 |

DeepSeek Licenseはモデル重みに適用され、商用利用に制限がある可能性があります。

## 5. 評価結果

### 5.1 インストール・セットアップ

| 項目 | 結果 |
|------|------|
| インストール | ⚠️ 複雑（バージョン固定必要） |
| GPUメモリ | ✅ 12GB GPUで動作可能（6.4GB使用） |
| 依存関係 | ❌ transformers 4.38.2固定 |
| モデルロード | ✅ 成功 |
| 推論実行 | ✅ 成功（隔離環境で） |

### 5.2 OCR性能評価

3つのテストPDF（各1ページ）で評価：

| PDF | OCR時間 | 文字数 | 速度 (pages/min) |
|-----|---------|--------|------------------|
| sample_cot.pdf | 7.43s | 1,280 | 8.08 |
| sample_autogen.pdf | 8.70s | 1,623 | 6.90 |
| sample_llama.pdf | 12.84s | 2,245 | 4.67 |
| **平均** | **9.66s** | **1,716** | **6.21** |

### 5.3 OCR品質評価

**出力例 (sample_llama.pdf)**:
```
LLaMA: Open and Efficient Foundation Language Models Hugo Touvron; Thibaut
Lavril; Gautier Izacard; Xavier Martinet Marie-Anne Lachaux, Timothee Lacroix,
Baptiste Roziere, Naman Goyal Eric Hambro, Faisal Azhar, Aurelien Rodriguez,
Armand Joulin Edouard Grave; Guillaume Lample* Meta AI

Abstract

We introduce LLaMA, a collection of foundation language models ranging from 7B
to 65B parameters. We train our models on trillions of tokens, and show that it
is possible to train state-of-the-art models using publicly available datasets
exclusively, without resorting to proprietary and inaccessible datasets...
```

品質評価：
- ✅ タイトル抽出: 正確
- ✅ 著者名抽出: 正確
- ✅ 本文抽出: 高精度
- ✅ セクション番号: 保持 ("1 Introduction")
- ✅ 引用形式: 正確 ("Brown et al., 2020")
- ⚠️ 座標情報: なし（VLMの特性上）

### 5.4 レイアウト検出評価

| PDF | 検出時間 | 速度 (pages/min) |
|-----|---------|------------------|
| sample_cot.pdf | 5.21s | 11.51 |
| sample_autogen.pdf | 6.27s | 9.57 |
| sample_llama.pdf | 4.49s | 13.37 |
| **平均** | **5.32s** | **11.27** |

**レイアウト検出出力例**:
```
Title: LLaMA: Open and Efficient Foundation Language Models

Section headers:
- 1. Introduction

Body text:
- We introduce LLaMA, a collection of foundation language models...

Page headers/footers:
- Meta AI
```

注意点：
- ⚠️ 存在しないセクション（"2. Model architecture details"等）を推測で出力する場合がある
- これはVLMの「推論」能力によるもので、プロンプトで制御可能

### 5.5 Issue #31への適用性

| 評価項目 | スコア | コメント |
|----------|--------|----------|
| 見出し検出 | ✅ | タイトル、セクションヘッダーを検出 |
| 本文/キャプション分類 | ✅ | プロンプトで分類可能 |
| 座標情報取得 | ❌ | VLMのため座標情報なし |
| 処理速度 | ❌ | 9.66秒/ページ（PyMuPDF4LLMの約12倍遅い） |
| 統合の容易さ | ❌ | 依存関係の制約が大きい |

## 6. 他ツールとの比較

| ツール | 速度 | 速度比 | OCR品質 | 座標情報 | 統合容易性 | 推奨度 |
|--------|------|--------|---------|----------|------------|--------|
| DocLayout-YOLO | ~420 p/min | 1x | - | ✅ | ✅ | ⭐⭐⭐⭐ |
| PyMuPDF4LLM | ~78 p/min | 5x遅い | - | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| **DeepSeek-VL2** | **~6.2 p/min** | **68x遅い** | **高精度** | ❌ | ❌ | ⭐⭐ |

※ DocLayout-YOLOとPyMuPDF4LLMはOCRではなくレイアウト検出/テキスト抽出ツール

## 7. 結論と推奨事項

### 7.1 結論

DeepSeek-VL2は**高品質なOCR能力**を持ちますが、以下の理由によりIssue #31での採用は**推奨しません**：

1. **処理速度**: 平均9.66秒/ページはPyMuPDF4LLMの約12倍遅い
2. **座標情報の欠如**: VLMの特性上、ブロックの座標情報を取得できない
3. **依存関係の制約**: transformers 4.38.2固定は保守コストが高い
4. **VRAM要件**: 12GB GPUでもメモリ制約あり

### 7.2 推奨事項

Issue #31のブロック分類改善には以下を推奨：

1. **第一選択**: PyMuPDF4LLM
   - `to_json()`による座標情報取得
   - `boxclass`によるヘッダー/フッター検出
   - プロジェクトの既存PyMuPDF使用との親和性

2. **補助ツール**: DocLayout-YOLO
   - 高速なレイアウト検出（7ページ/秒）
   - 10種類のブロック分類
   - PyMuPDF4LLMと組み合わせて使用可能

### 7.3 DeepSeek-VL2の適切なユースケース

以下の場合にはDeepSeek-VL2が有用：
- スキャンPDFなど、テキスト抽出が困難な文書のOCR
- 画像内のテキスト抽出
- レイアウト分析が不要で高品質OCRのみが必要な場合
- バッチ処理で処理時間が問題にならない場合

## 8. 評価環境

### 本評価（隔離環境）
```
OS: Linux 6.14.0-36-generic
GPU: NVIDIA GeForce RTX 4070 Ti (12GB)
Python: 3.12
torch: 2.9.1+cu128
transformers: 4.38.2
xformers: 0.0.33.post2
Image DPI: 100
Max image size: 1024px
Max new tokens: 512
```

### プロジェクト環境（互換性問題あり）
```
transformers: 4.57.3
結果: 動作せず（パッチ適用でも解決不可）
```

## 9. 参考資料

- [DeepSeek-VL2 GitHub](https://github.com/deepseek-ai/DeepSeek-VL2)
- [DeepSeek-VL2 Paper](https://arxiv.org/abs/2412.10302)
- [HuggingFace Model](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny)

## 10. 添付ファイル

評価結果は以下に保存：
- `tests/evaluation/outputs/DeepSeek_VL2/evaluation_summary.json`
- `tests/evaluation/outputs/DeepSeek_VL2/sample_*/page_*_ocr.txt`
- `tests/evaluation/outputs/DeepSeek_VL2/sample_*/page_*_layout.txt`
