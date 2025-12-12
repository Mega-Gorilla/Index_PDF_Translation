# 文書レイアウト解析技術調査レポート

**作成日**: 2025-12-05
**関連Issue**: #31 ブロック分類アルゴリズム改善
**目的**: PDF文書のレイアウト解析における最新技術の包括的調査

---

## 目次

1. [概要](#1-概要)
2. [オープンソースツール](#2-オープンソースツール)
   - [2.1 PyMuPDF4LLM + Layout](#21-pymupdf4llm--layout)
   - [2.2 DocLayout-YOLO](#22-doclayout-yolo)
   - [2.3 DeepSeek-OCR](#23-deepseek-ocr)
   - [2.4 GROBID](#24-grobid)
   - [2.5 Unstructured](#25-unstructured)
   - [2.6 Marker](#26-marker)
3. [学術研究](#3-学術研究)
   - [3.1 LayoutLMファミリー](#31-layoutlmファミリー)
   - [3.2 LiLT](#32-lilt)
4. [データセット](#4-データセット)
   - [4.1 PubLayNet](#41-publaynet)
   - [4.2 DocLayNet](#42-doclaynet)
5. [ツール比較](#5-ツール比較)
6. [本プロジェクトへの適用考察](#6-本プロジェクトへの適用考察)
7. [参考文献](#7-参考文献)

---

## 1. 概要

本レポートは、PDF文書のレイアウト解析（本文・見出し・ヘッダー/フッター・図表の分類）に関する最新技術を調査し、Index PDF Translationプロジェクトへの適用可能性を評価する。

### 調査対象

| カテゴリ | 対象 |
|---------|------|
| オープンソースツール | PyMuPDF4LLM, DocLayout-YOLO, DeepSeek-OCR, GROBID, Unstructured, Marker |
| 学術研究 | LayoutLM, LayoutLMv3, LiLT, DiT |
| データセット | PubLayNet, DocLayNet, OmniDocBench |

### 評価観点

1. **精度**: レイアウト要素の検出・分類精度
2. **速度**: 処理スループット
3. **依存関係**: 必要なライブラリ、GPU要件
4. **ライセンス**: AGPL-3.0との互換性
5. **統合容易性**: 本プロジェクトへの組み込みやすさ

---

## 2. オープンソースツール

### 2.1 PyMuPDF4LLM + Layout

#### 概要

PyMuPDF4LLMは、LLM/RAG環境向けに最適化されたPDFコンテンツ抽出ライブラリ。`pymupdf-layout`パッケージと組み合わせることで、AIベースのレイアウト解析が可能になる。

| 項目 | 内容 |
|------|------|
| **リリース** | 2024年 |
| **ライセンス** | AGPL-3.0 |
| **GPU要件** | 不要 |
| **アプローチ** | ヒューリスティクス + 機械学習 |

#### 主要機能

- **マルチカラム対応**: 複数カラムのページレイアウトを正確に解析
- **ヘッダー/フッター検出**: ページ間で繰り返される要素（ロゴ、ページ番号）を自動識別・除外
- **見出し検出**: フォントサイズに基づく見出しの自動検出
- **図表抽出**: 画像・ベクターグラフィックスの抽出とMarkdown参照
- **OCR統合**: 画像のみのPDFに対してOCRを選択的に適用

#### インストール

```bash
pip install pymupdf4llm
pip install pymupdf-layout  # AIベースレイアウト解析を有効化
```

#### 使用例

```python
# 重要: pymupdf.layout を先にインポートする必要がある
import pymupdf.layout
import pymupdf4llm

# Markdown形式で抽出
md_text = pymupdf4llm.to_markdown("input.pdf")

# JSON形式で抽出（レイアウト情報含む）
json_data = pymupdf4llm.to_json("input.pdf")

# LlamaIndex統合
llama_reader = pymupdf4llm.LlamaMarkdownReader()
llama_docs = llama_reader.load_data("input.pdf")
```

#### 出力形式

| 形式 | 説明 |
|------|------|
| Markdown | フォーマット済みテキスト、ヘッダー、表、リスト、コードブロック |
| JSON | レイアウトメタデータ付き構造化データ |
| Plain Text | プレーンテキスト |
| LlamaIndex Documents | RAGパイプライン用ドキュメント |

#### 制限事項

- `pymupdf-layout`のインポート順序が重要（先にインポートしないと標準抽出にフォールバック）
- 複雑なレイアウトの精度は文書タイプに依存
- 詳細なベンチマーク結果は公開されていない

#### 参考リンク

- [PyMuPDF4LLM Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [PyMuPDF Layout Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf-layout/index.html)

---

### 2.2 DocLayout-YOLO

#### 概要

DocLayout-YOLOは、OpenDataLabが開発した文書レイアウト解析に特化したYOLOベースのモデル。YOLOv10をベースに、文書解析向けの最適化を施している。

| 項目 | 内容 |
|------|------|
| **リリース** | 2024年 |
| **ライセンス** | Apache-2.0 |
| **GPU要件** | 推奨（CPUでも動作可能） |
| **アプローチ** | YOLOv10 + Global-to-Local Controllability |

#### アーキテクチャ

**Global-to-Local Controllability (G2L) モジュール**:
- 文書要素のスケール変動に対応するための核心技術
- グローバルな視点で文書を把握しつつ、ローカルな精度を維持
- Backbone → Neck → Head の階層構造

**事前学習データセット**:
- **DocSynth-300K**: 大規模・多様な文書レイアウト事前学習データセット
- Mesh-candidate BestFitアルゴリズムによる合成（2Dビンパッキング問題として定式化）
- 2024年10月公開

#### ベンチマーク結果

| データセット | AP50 | mAP | 事前学習 |
|-------------|------|-----|---------|
| D4LA | 82.4% | 70.3% | DocSynth300K |
| DocLayNet | 93.4% | 79.7% | DocSynth300K |

#### インストール

```bash
# Conda環境作成
conda create -n doclayout_yolo python=3.10
conda activate doclayout_yolo

# 開発用インストール
git clone https://github.com/opendatalab/DocLayout-YOLO.git
cd DocLayout-YOLO
pip install -e .

# 推論のみ
pip install doclayout-yolo
```

#### 使用例

```python
from doclayout_yolo import YOLOv10

# モデル読み込み
model = YOLOv10("path/to/model.pt")

# 推論
results = model.predict(
    "document.png",
    imgsz=1024,
    conf=0.2,  # 信頼度閾値
)

# 結果の処理
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = box.cls  # クラスID
        conf = box.conf  # 信頼度
        xyxy = box.xyxy  # バウンディングボックス
```

**CLI使用**:

```bash
yolo predict model=path/to/model.pt source=document.png imgsz=1024
```

#### 検出クラス

DocStructBenchモデルでの対応クラス（文書タイプにより異なる）:
- Title, Text, Figure, Table, Caption
- Header, Footer, Page Number
- List, Formula, Code Block

#### 制限事項

- GPUがないと処理速度が遅い
- 大きなモデルサイズ
- PDFを直接処理するにはPDF-Extract-Kitとの統合が必要

#### 参考リンク

- [GitHub Repository](https://github.com/opendatalab/DocLayout-YOLO)
- [Hugging Face Models](https://huggingface.co/opendatalab/DocLayout-YOLO-DocStructBench)
- [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)

---

### 2.3 DeepSeek-OCR

#### 概要

DeepSeek-OCRは、DeepSeekが2025年10月にリリースしたVision-Languageモデル。「Context Optical Compression」というコンセプトで、画像を効率的に圧縮しながら高精度OCRを実現する。

| 項目 | 内容 |
|------|------|
| **リリース** | 2025年10月 |
| **ライセンス** | MIT |
| **GPU要件** | 必須（A100推奨） |
| **モデルサイズ** | 3B (MoE, 570Mアクティブパラメータ) |

#### アーキテクチャ

**DeepEncoder**:
- SAM-base (80M) + CLIP-large (300M) の組み合わせ
- ローカル認識（SAM）とグローバル理解（CLIP）の統合
- 高解像度入力でも低アクティベーションを維持

**DeepSeek3B-MoE-A570M Decoder**:
- Mixture of Experts アーキテクチャ
- 高効率な推論

#### 解像度モード

| モード | 解像度 | トークン数 |
|--------|--------|-----------|
| Native 512 | 512×512 | 64 |
| Native 640 | 640×640 | 100 |
| Native 1024 | 1024×1024 | 256 |
| Native 1280 | 1280×1280 | 400 |
| Gundam (動的) | n×640×640 + 1×1024×1024 | 可変 |

#### 性能

| 指標 | 値 |
|------|-----|
| 圧縮率 10× | 97% 精度 |
| 圧縮率 20× | ~60% 精度 |
| スループット | ~2,500 tokens/s (A100-40G) |
| 日次処理能力 | 200,000+ ページ (A100-40G) |

**比較**:
- GOT-OCR2.0 (256トークン/ページ) を100トークンで上回る
- MinerU2.0 (6000+トークン/ページ) を800トークン未満で上回る

#### GPU/VRAM要件

**モデルサイズとVRAM消費**:

| 項目 | 値 |
|------|-----|
| パラメータ数 | 3B (MoEで570Mアクティブ) |
| BF16重み | ~6.7 GB |
| 推論時合計 (512トークン) | ~13 GB |
| 推奨VRAM | 16 GB以上 |

**量子化によるVRAM削減**:

| 精度 | 重みサイズ | 推論時VRAM | 品質への影響 |
|------|-----------|-----------|-------------|
| BF16 (フル) | ~6.7 GB | ~13 GB | なし（最高品質） |
| INT8 (Q8_0) | ~3.4 GB | ~7 GB | 軽微 |
| INT4 (Q4_K) | ~1.7 GB | ~4 GB | 中程度 |

**コンシューマGPU動作可否**:

| GPU | VRAM | BF16 | INT8 | INT4 |
|-----|------|------|------|------|
| RTX 3060 | 12 GB | ❌ | ✅ | ✅ |
| RTX 3080 | 10 GB | ❌ | ⚠️ | ✅ |
| RTX 3080 Ti | 12 GB | ❌ | ✅ | ✅ |
| RTX 4070 Ti | 12 GB | ❌ | ✅ | ✅ |
| RTX 4080 | 16 GB | ⚠️ | ✅ | ✅ |
| RTX 4090 | 24 GB | ✅ | ✅ | ✅ |
| T4 (Colab無料) | 16 GB | ⚠️ | ✅ | ✅ |
| A100 | 40/80 GB | ✅ | ✅ | ✅ |

凡例: ✅ 動作可能 / ⚠️ ギリギリ / ❌ VRAM不足

**4-bit量子化での実行方法**:

```python
# 4-bit量子化でのロード例
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    quantization_config=quantization_config,
    trust_remote_code=True,
)
```

**参考**: [4-bit推論 Colab Notebook](https://colab.research.google.com/github/Alireza-Akhavan/LLM/blob/main/deepseek_ocr_inference_4bit.ipynb)

#### インストール

```bash
# 環境要件
# - CUDA 11.8
# - Python 3.12.9
# - PyTorch 2.6.0

# 依存関係
pip install vllm==0.8.5
pip install flash-attn==2.7.3
pip install transformers

# モデルダウンロード
# Hugging Face: deepseek-ai/DeepSeek-OCR
```

#### 使用例

**vLLM推論**:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    trust_remote_code=True,
)

# 画像OCR
outputs = llm.generate(
    prompts=[{"prompt": "OCR this document", "multi_modal_data": {"image": image}}],
    sampling_params=SamplingParams(temperature=0, max_tokens=4096),
)
```

**Transformers推論**:

```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained("deepseek-ai/DeepSeek-OCR")

# 推論
inputs = processor(images=image, return_tensors="pt")
outputs = model.generate(**inputs)
```

#### 対応タスク

- 文書OCR（Markdown変換）
- 図表解析
- 数式認識
- 一般画像説明
- 空間的グラウンディング

#### 制限事項

- GPU必須（A100クラス推奨）
- 複雑な表でのエラー報告あり（マルチヘッダー表、ページ跨ぎ）
- 比較的新しいツールのため、実運用での検証事例が少ない

#### 参考リンク

- [GitHub Repository](https://github.com/deepseek-ai/DeepSeek-OCR)
- [Hugging Face Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [arXiv Paper](https://arxiv.org/abs/2510.18234)

---

### 2.4 GROBID

#### 概要

GROBID (GeneRation Of BIbliographic Data) は、学術論文PDFの構造化抽出に特化した機械学習ライブラリ。2009年のECDLで発表され、現在も活発に開発が続いている。

| 項目 | 内容 |
|------|------|
| **初版** | ECDL 2009 |
| **ライセンス** | Apache-2.0 |
| **GPU要件** | オプション（Deep Learning使用時） |
| **アプローチ** | CRF + Deep Learning (カスケード型) |

#### アーキテクチャ

**カスケード型シーケンスラベリング**:
- 単一のEnd-to-Endモデルではなく、複数の専門モデルの連鎖
- 68の異なるラベルで詳細な構造識別
- 「Layout Tokens」（視覚・空間情報を含む）を入力として使用

**モデルタイプ**:

| タイプ | 説明 | 精度 | 速度 |
|--------|------|------|------|
| CRF | デフォルト設定 | 良好 | 高速 |
| Deep Learning | DeLFTフレームワーク（RNN, Transformer） | 最高 | 中程度 |
| ハイブリッド | CRF + DL の組み合わせ | 高い | バランス |

#### 抽出機能

**ヘッダー・メタデータ**:
- タイトル、アブストラクト、著者、所属、キーワード
- 著者名のパース（ヘッダー用・参考文献用の別モデル）
- 日付のISO正規化
- 著作権・ライセンス識別

**参考文献処理**:
- 参考文献抽出: ~0.87 F1スコア (PubMed Central)
- 引用コンテキスト認識: 0.76-0.91 F1スコア
- DOI/PMID解決: 0.95+ F1スコア

**フルテキスト構造化**:
- 段落、セクションタイトル、図、表、脚注の認識
- 資金情報の抽出（CrossRef Funder Registry連携）
- PDF座標の保持（インタラクティブPDF生成用）

#### ベンチマーク結果

| タスク | データセット | F1スコア | モデル |
|--------|-------------|---------|--------|
| 参考文献抽出 | PubMed Central (1943 PDFs) | ~0.87 | Deep Learning |
| 参考文献抽出 | bioRxiv | ~0.90 | Deep Learning |
| 参考文献パース（インスタンス） | - | 0.90+ | Deep Learning |
| 参考文献パース（フィールド） | - | 0.95+ | Deep Learning |

#### インストール

**要件**:
- OpenJDK 21
- Linux 64-bit または macOS (Intel/ARM)
- オプション: Python 3.8+ (Deep Learning用)
- オプション: NVIDIA GPU + CUDA

**Docker使用**:

```bash
docker pull grobid/grobid:latest
docker run -p 8070:8070 grobid/grobid:latest
```

**ソースからビルド**:

```bash
git clone https://github.com/kermitt2/grobid.git
cd grobid
./gradlew clean install
```

#### 使用例

**REST API**:

```bash
# サービス起動
./gradlew run

# PDF処理
curl -X POST \
  -F "input=@paper.pdf" \
  http://localhost:8070/api/processFulltextDocument
```

**Pythonクライアント**:

```python
from grobid_client.grobid_client import GrobidClient

client = GrobidClient(config_path="./config.json")
client.process("processFulltextDocument", "./pdfs/", output="./outputs/")
```

#### 出力形式

- **TEI XML**: Text Encoding Initiative準拠のXML
- **BibTeX**: 参考文献のBibTeX形式
- **JSON**: 構造化JSON

#### 本番運用実績

- ResearchGate
- Semantic Scholar
- HAL (フランス国立オープンアーカイブ)
- scite.ai
- Academia.edu
- Internet Archive Scholar
- CERN Invenio

#### 関連モジュール

| モジュール | 機能 |
|-----------|------|
| software-mention | 論文内のソフトウェア言及認識 |
| datastet | データセット識別・分類 |
| grobid-quantities | 物理量認識 |
| entity-fishing | Wikidataエンティティ抽出 |
| grobid-ner | 固有表現認識 |

#### 制限事項

- Java依存（JVM必要）
- 学術論文に特化（一般文書には最適化されていない）
- 初期セットアップが複雑

#### 参考リンク

- [GitHub Repository](https://github.com/kermitt2/grobid)
- [Documentation](https://grobid.readthedocs.io/)
- [ECDL 2009 Paper](https://hal.science/hal-01673305)

---

### 2.5 Unstructured

#### 概要

Unstructuredは、複雑な文書を構造化データに変換するオープンソースETLライブラリ。LLM/AIアプリケーション向けのデータ前処理に特化。

| 項目 | 内容 |
|------|------|
| **リリース** | 2022年 |
| **ライセンス** | Apache-2.0 |
| **GPU要件** | オプション |
| **アプローチ** | Detectron2 + ルールベース |

#### 対応フォーマット

| カテゴリ | フォーマット |
|---------|-------------|
| ドキュメント | PDF, DOCX, PPTX, XLSX, RTF, EPUB |
| ウェブ | HTML, XML, JSON |
| メール | EML, MSG |
| 画像 | PNG, JPG (OCR) |

#### インストール

```bash
# 基本インストール（プレーンテキスト、HTML、XML、JSON、メール）
pip install unstructured

# フルサポート
pip install "unstructured[all-docs]"

# 選択的インストール
pip install "unstructured[pdf,docx]"
```

**システム依存関係**:

```bash
# Ubuntu/Debian
apt-get install libmagic-dev poppler-utils tesseract-ocr libreoffice pandoc

# macOS
brew install libmagic poppler tesseract libreoffice pandoc
```

#### 使用例

```python
from unstructured.partition.auto import partition

# 自動ファイルタイプ検出
elements = partition(filename="document.pdf")

# 要素の処理
for element in elements:
    print(f"Type: {type(element).__name__}")
    print(f"Text: {element.text}")
    print(f"Metadata: {element.metadata}")
```

**PDF特化**:

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="paper.pdf",
    strategy="hi_res",  # 高解像度戦略
    infer_table_structure=True,
)
```

#### 出力要素タイプ

- `Title`: タイトル
- `NarrativeText`: 本文テキスト
- `ListItem`: リスト項目
- `Table`: 表
- `Image`: 画像
- `Header`: ヘッダー
- `Footer`: フッター
- `PageBreak`: ページ区切り

#### 処理戦略

| 戦略 | 説明 | 精度 | 速度 |
|------|------|------|------|
| `fast` | テキスト抽出のみ | 低 | 高速 |
| `hi_res` | レイアウト解析 + OCR | 高 | 低速 |
| `ocr_only` | OCRのみ | 中 | 中 |

#### 制限事項

- `hi_res`戦略は処理が遅い
- 一部依存関係がAGPL（ultralytics等）
- 複雑なレイアウトでの精度にばらつき

#### 参考リンク

- [GitHub Repository](https://github.com/Unstructured-IO/unstructured)
- [Documentation](https://unstructured-io.github.io/unstructured/)
- [Unstructured Platform](https://unstructured.io/) (エンタープライズ版)

---

### 2.6 Marker

#### 概要

Markerは、PDF・画像・各種ドキュメントをMarkdown、JSON、HTML形式に変換する高精度ツール。LLM統合による精度向上機能を持つ。

| 項目 | 内容 |
|------|------|
| **リリース** | 2023年 |
| **ライセンス** | GPL-3.0 (コード), AI Pubs Open Rail-M (モデル) |
| **GPU要件** | 推奨（CPU/MPSも対応） |
| **アプローチ** | マルチモデル統合 |

#### 主要機能

- **マルチフォーマット**: PDF, PPTX, DOCX, XLSX, HTML, EPUB, 画像
- **多言語対応**: 全言語をサポート
- **コンテンツ抽出**: 表、数式、インラインmath、コードブロック、リンク
- **アーティファクト除去**: ヘッダー、フッター等の不要要素を自動除去
- **LLM統合**: オプションでLLMによる精度向上

#### 性能

| 指標 | 値 |
|------|-----|
| バッチ処理速度 | ~25 ページ/秒 (H100) |
| 精度 | Llamaparse, Mathpixと競合 |

#### インストール

```bash
# 基本インストール
pip install marker-pdf

# 非PDFフォーマットサポート
pip install marker-pdf[full]
```

**要件**:
- Python 3.10+
- PyTorch

#### 使用例

**CLI**:

```bash
# 単一ファイル変換
marker_single /path/to/file.pdf

# バッチ処理
marker /path/to/input/folder

# 出力形式指定
marker_single document.pdf --output_format json

# LLM使用（精度向上）
marker_single document.pdf --use_llm
```

**Python API**:

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

# コンバーター作成
converter = PdfConverter(artifact_dict=create_model_dict())

# 変換実行
rendered = converter("document.pdf")

# 結果取得
markdown = rendered.markdown
images = rendered.images
metadata = rendered.metadata
```

**マルチGPU処理**:

```bash
NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert ./input ./output
```

#### 出力形式

| 形式 | 内容 |
|------|------|
| Markdown | 画像、表、LaTeX数式、コードブロック |
| HTML | セマンティックマークアップ |
| JSON | ツリー構造、メタデータ、バウンディングボックス |
| Chunks | RAGシステム向けフラット形式 |

#### LLMサービス統合

- Google Gemini (デフォルト: gemini-2.0-flash)
- Ollama
- カスタムLLMバックエンド

#### ライセンス詳細

| 対象 | ライセンス | 商用利用 |
|------|-----------|---------|
| コード | GPL-3.0 | 可能（GPL遵守） |
| モデル重み | AI Pubs Open Rail-M | 研究・個人・$2M未満スタートアップのみ無料 |

**注意**: 商用利用には[DataLab](https://datalab.to/)からのライセンス取得が必要な場合あり。

#### 制限事項

- モデル重みに収益制限あり
- GPU推奨（CPUでは遅い）
- LLM統合にはAPI費用が発生

#### 参考リンク

- [GitHub Repository](https://github.com/VikParuchuri/marker)
- [DataLab Platform](https://datalab.to/)

---

## 3. 学術研究

### 3.1 LayoutLMファミリー

#### LayoutLM (KDD 2020)

| 項目 | 内容 |
|------|------|
| **発表** | KDD 2020 |
| **著者** | Xu et al. (Microsoft Research) |
| **ライセンス** | MIT |

**アーキテクチャ**:
- テキスト + レイアウト情報の事前学習
- スキャン文書画像からの視覚特徴統合
- BERT風のTransformerエンコーダー

**ベンチマーク結果**:

| タスク | 改善前 | 改善後 |
|--------|--------|--------|
| Form Understanding | 70.72% | 79.27% |
| Receipt Understanding | 94.02% | 95.24% |
| Document Image Classification | 93.07% | 94.42% |

**参考**: [arXiv:1912.13318](https://arxiv.org/abs/1912.13318)

#### LayoutLMv3 (ACM MM 2022)

| 項目 | 内容 |
|------|------|
| **発表** | ACM Multimedia 2022 |
| **ライセンス** | CC BY-NC-SA 4.0 (非商用) |

**改善点**:
- **統一マスキング**: テキストと画像に同一のマスキング戦略を適用
- **Word-Patch Alignment**: テキストと対応する画像パッチの関係学習
- **マルチモーダル**: テキスト・レイアウト・ビジョンの統合

**対応タスク**:
- テキスト中心: フォーム理解、レシート理解、文書VQA
- 画像中心: 文書画像分類、文書レイアウト解析

**注意**: モデル重みはCC BY-NC-SA 4.0ライセンスのため、商用利用には使用不可。

**参考**: [arXiv:2204.08387](https://arxiv.org/abs/2204.08387)

---

### 3.2 LiLT

| 項目 | 内容 |
|------|------|
| **発表** | ACL 2022 |
| **著者** | Wang et al. |
| **ライセンス** | MIT |

#### 特徴

**言語非依存アーキテクチャ**:
- レイアウト理解とテキスト処理を分離
- 英語で事前学習後、他言語に適用可能
- テキストエンコーダーを言語別に差し替え可能

**利用可能なチェックポイント**:

| モデル | サイズ | 説明 |
|--------|--------|------|
| lilt-roberta-en-base | 293MB | 英語特化 |
| lilt-infoxlm-base | 846MB | 多言語対応 |
| lilt-only-base | 21MB | レイアウトのみ（カスタム組み合わせ用） |

#### ベンチマーク

- FUNSD (フォーム理解): セマンティックエンティティ認識
- XFUND: 多言語ベンチマーク、ゼロショット言語間転移

#### インストール・使用

```bash
# 要件
# Python 3.7, PyTorch 1.7.1, CUDA 11.0+, Detectron2 0.5

git clone https://github.com/jpWang/LiLT.git
cd LiLT
pip install -r requirements.txt
```

**商用利用**: MITライセンスのため制限なし

**参考**: [ACL Anthology](https://aclanthology.org/2022.acl-long.534/)

---

## 4. データセット

### 4.1 PubLayNet

| 項目 | 内容 |
|------|------|
| **発表** | ICDAR 2019 (Best Paper Award) |
| **ライセンス** | CDLA-Permissive-1.0 |
| **サイズ** | 360,000+ ページ |

#### 概要

PubMed Centralの100万以上のPDF論文から自動生成された大規模データセット。XMLとPDFコンテンツの自動マッチングによりアノテーションを作成。

#### クラス

5つのレイアウト要素クラス:
1. Title
2. Text
3. List
4. Table
5. Figure

#### 特徴

- **自動アノテーション**: 手動ラベリング不要
- **大規模**: コンピュータビジョンデータセットに匹敵
- **転移学習**: 他のドメインへの転移学習のベースモデルとして有効

**参考**: [arXiv:1908.07836](https://arxiv.org/abs/1908.07836), [GitHub](https://github.com/ibm-aur-nlp/PubLayNet)

---

### 4.2 DocLayNet

| 項目 | 内容 |
|------|------|
| **発表** | KDD 2022 |
| **ライセンス** | CDLA-Permissive-1.0 |
| **サイズ** | 80,863 ページ |

#### 概要

多様なデータソースから手動アノテーションされたデータセット。PubLayNet/DocBankの「学術論文のみ」という制限を克服。

#### クラス

11の詳細なクラス:
1. Caption
2. Footnote
3. Formula
4. List-item
5. Page-footer
6. Page-header
7. Picture
8. Section-header
9. Table
10. Text
11. Title

#### 特徴

- **多様性**: 学術論文以外の文書タイプを含む
- **手動アノテーション**: 高品質なラベル
- **Inter-annotator Agreement**: 二重・三重アノテーションによる品質検証
- **汎用性**: DocLayNetで学習したモデルはより堅牢

**比較**: 一般的な物体検出モデルはInter-annotator Agreementより約10%低い精度

**参考**: [arXiv:2206.01062](https://arxiv.org/abs/2206.01062), [GitHub](https://github.com/DS4SD/DocLayNet)

---

## 5. ツール比較

### 総合比較表

| ツール | ライセンス | AGPL互換 | GPU要件 | 精度 | 速度 | 統合容易性 | 学術論文特化 |
|--------|-----------|---------|---------|------|------|-----------|-------------|
| **PyMuPDF4LLM** | AGPL-3.0 | ✅ | 不要 | 中〜高 | 高速 | ★★★★★ | ○ |
| **DocLayout-YOLO** | Apache-2.0 | ✅ | 推奨 | 高 | 高速 | ★★★☆☆ | ○ |
| **DeepSeek-OCR** | MIT | ✅ | 必須 | 最高 | 高速 | ★★☆☆☆ | △ |
| **GROBID** | Apache-2.0 | ✅ | オプション | 高 | 中 | ★★★☆☆ | ★★★★★ |
| **Unstructured** | Apache-2.0 | ✅ | オプション | 中〜高 | 中 | ★★★★☆ | ○ |
| **Marker** | GPL-3.0 | ✅ | 推奨 | 高 | 高速 | ★★★☆☆ | ○ |

### ユースケース別推奨

| ユースケース | 推奨ツール | 理由 |
|-------------|-----------|------|
| 軽量・高速処理 | PyMuPDF4LLM | GPU不要、既存PyMuPDF統合 |
| 最高精度（GPU有） | DeepSeek-OCR | 97%精度、省トークン |
| 学術論文特化 | GROBID | 参考文献、メタデータ抽出に最適 |
| 汎用文書処理 | Unstructured | 多フォーマット対応 |
| Markdown出力 | Marker | 高品質Markdown生成 |
| リアルタイム検出 | DocLayout-YOLO | YOLOベースの高速推論 |

### ライセンス互換性

| ライセンス | AGPL-3.0互換 | 商用利用 | 該当ツール |
|-----------|-------------|---------|-----------|
| AGPL-3.0 | ✅ 同一 | ✅ | PyMuPDF4LLM |
| Apache-2.0 | ✅ 互換 | ✅ | GROBID, Unstructured, DocLayout-YOLO |
| MIT | ✅ 互換 | ✅ | LiLT, DeepSeek-OCR |
| GPL-3.0 | ✅ 互換 | ✅ | Marker (コード) |
| CC BY-NC-SA 4.0 | ⚠️ 非商用のみ | ❌ | LayoutLMv3 (モデル) |

---

## 6. 本プロジェクトへの適用考察

### 現状の課題

Index PDF Translationの現在のブロック分類アルゴリズムには以下の課題がある:

1. **見出し検出不可**: トークン数・フォントサイズだけでは見出しと本文を区別困難
2. **位置情報未使用**: ヘッダー/フッターの位置パターンを活用していない
3. **ヒストグラム選択の不安定性**: 最頻出ビン≠本文ブロックとなる場合がある

### 推奨アプローチ

#### Phase 1: PyMuPDF4LLM + Layout 統合（推奨）

**理由**:
- ✅ 同一ライセンス（AGPL-3.0）
- ✅ GPU不要
- ✅ 既存PyMuPDF使用のため統合容易
- ✅ ヘッダー/フッター/見出し検出をサポート

**実装方針**:
```python
# オプショナル依存として追加
try:
    import pymupdf.layout
    import pymupdf4llm
    USE_LAYOUT = True
except ImportError:
    USE_LAYOUT = False

# フォールバック付き抽出
if USE_LAYOUT:
    # PyMuPDF Layout による高精度抽出
    ...
else:
    # 既存アルゴリズム
    ...
```

#### Phase 2: カスタムルール追加

PyMuPDF Layoutの結果を補完するルールベース処理:
- セクション番号パターン検出 (`1.`, `1.1`, `A.`)
- 位置ベースのヘッダー/フッター検出
- 参考文献パターン検出 (`[1]`, `et al.`)
- 図表キャプションパターン強化

#### Phase 3: オプショナルML強化（将来）

高精度が必要な場合のオプション:
- **LiLT**: MITライセンス、言語非依存、商用可
- **DocLayout-YOLO**: Apache-2.0、高速・高精度

**避けるべきモデル**:
- LayoutLMv3: CC BY-NC-SA 4.0（非商用のみ）
- DiT: CC BY-NC-SA 4.0（非商用のみ）

### 依存関係への影響

| Phase | 追加依存 | サイズ | GPU |
|-------|---------|--------|-----|
| Phase 1 | pymupdf-layout | 軽量 | 不要 |
| Phase 2 | なし | - | 不要 |
| Phase 3 | scikit-learn / LiLT | 中〜大 | オプション |

---

## 7. 参考文献

### 学術論文

#### Document AI モデル
- Xu, Y., et al. (2020). "LayoutLM: Pre-training of Text and Layout for Document Image Understanding." KDD 2020. [arXiv:1912.13318](https://arxiv.org/abs/1912.13318)
- Huang, Y., et al. (2022). "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking." ACM MM 2022. [arXiv:2204.08387](https://arxiv.org/abs/2204.08387)
- Wang, J., et al. (2022). "LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding." ACL 2022. [ACL Anthology](https://aclanthology.org/2022.acl-long.534/)
- Li, J., et al. (2022). "DiT: Self-supervised Pre-training for Document Image Transformer." ACM MM 2022. [arXiv:2203.02378](https://arxiv.org/abs/2203.02378)

#### データセット
- Zhong, X., et al. (2019). "PubLayNet: Largest Dataset Ever for Document Layout Analysis." ICDAR 2019 (Best Paper). [arXiv:1908.07836](https://arxiv.org/abs/1908.07836)
- Pfitzmann, B., et al. (2022). "DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation." KDD 2022. [arXiv:2206.01062](https://arxiv.org/abs/2206.01062)

#### OCR・Vision-Language
- Wei, H., et al. (2025). "DeepSeek-OCR: Contexts Optical Compression." [arXiv:2510.18234](https://arxiv.org/abs/2510.18234)

### オープンソースツール

- [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) - AGPL-3.0
- [PyMuPDF Layout](https://pymupdf.readthedocs.io/en/latest/pymupdf-layout/index.html) - AGPL-3.0
- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) - Apache-2.0
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) - MIT
- [GROBID](https://github.com/kermitt2/grobid) - Apache-2.0
- [Unstructured](https://github.com/Unstructured-IO/unstructured) - Apache-2.0
- [Marker](https://github.com/VikParuchuri/marker) - GPL-3.0
- [LiLT](https://github.com/jpWang/LiLT) - MIT

### データセット

- [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) - CDLA-Permissive-1.0
- [DocLayNet](https://github.com/DS4SD/DocLayNet) - CDLA-Permissive-1.0
