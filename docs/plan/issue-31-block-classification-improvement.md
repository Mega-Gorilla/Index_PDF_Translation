# Issue #31: ブロック分類アルゴリズム改善計画

## 1. 概要

現在のブロック分類アルゴリズムを改善し、以下の目標を達成する：

1. **不要テキストの誤検出削減**: ヘッダー/フッター/ページ番号をより正確に除外
2. **見出し検出**: セクション見出しと本文を区別可能にする
3. **見出し翻訳オプション**: 検出した見出しを翻訳対象に含めるオプションを追加
4. **分類精度向上**: 全体的な分類精度を現状より向上

## 2. 現状分析

### 2.1 現在のアルゴリズム

```
特徴量: トークン数, ブロック幅, フォントサイズ
     ↓
スコア計算: トークン閾値(binary) + IQR偏差(width, size)
     ↓
ヒストグラム分析: 最頻出ビンを本文と判定
     ↓
分類: 本文 / 図表キャプション / 除外
```

### 2.2 現在の問題点

| 問題 | 詳細 | 影響度 |
|------|------|--------|
| 見出し検出不可 | トークン数/フォントサイズだけでは区別困難 | 高 |
| 位置情報未使用 | ヘッダー/フッターの位置パターンを無視 | 高 |
| パターン認識不足 | セクション番号、参考文献パターン未検出 | 中 |
| ヒストグラム選択の不安定性 | 最頻出ビン≠本文ブロックの場合あり | 中 |
| 文脈分析なし | ブロック間の関係性を無視 | 低 |

## 3. 調査結果: 最新の研究・実装

### 3.1 学術研究

#### Deep Learning ベースのアプローチ

| 手法 | 概要 | 精度 | ライセンス | 参考 |
|------|------|------|-----------|------|
| **LayoutLMv3** | Text + Layout + Vision のマルチモーダルTransformer | 95% (RVL-CDIP) | ⚠️ CC BY-NC-SA 4.0 (モデル) | [Hugging Face](https://huggingface.co/docs/transformers/model_doc/layoutlm) |
| **LiLT** | 言語非依存レイアウトTransformer | 90%+ | ✅ MIT | [GitHub](https://github.com/jpWang/LiLT) |
| **DiT** | Document Image Transformer (Vision only) | 92% (RVL-CDIP) | ⚠️ CC BY-NC-SA 4.0 (モデル) | [Papers with Code](https://paperswithcode.com/task/document-layout-analysis/latest) |
| **YOLOv8** | 物体検出ベースのレイアウト分析 | 高速・高精度 | ⚠️ AGPL-3.0 | [Nature 2025](https://www.nature.com/articles/s41598-025-07439-y) |

#### ルールベース/ハイブリッドアプローチ

| 手法 | 概要 | 精度 | 参考 |
|------|------|------|------|
| **TBRF** | Rule-based + ML ハイブリッド | 95%+ | [arXiv](https://arxiv.org/html/2305.17401) |
| **Multi-pass Sieve** | 多段階ルール適用 | 92.6% | [ResearchGate](https://www.researchgate.net/publication/265487498) |
| **PDFX** | 学術論文特化ルールベース | 77.45 F1 (heading) | - |
| **SED** | 教師なしルールベース | 82%+ | [IJCSI](https://www.ijcsi.org/papers/IJCSI-17-6-40-47.pdf) |

### 3.2 主要データセット

| データセット | サイズ | クラス数 | ライセンス | 特徴 |
|-------------|--------|---------|-----------|------|
| **PubLayNet** | 360K+ pages | 5 | ✅ CDLA-Permissive-1.0 | 学術論文特化、自動アノテーション |
| **DocLayNet** | 80K+ pages | 11 | ✅ CDLA-Permissive-1.0 | 多様な文書、手動アノテーション |
| **OmniDocBench** | 1,355 pages | 15 block + 4 span | 要確認 | CVPR 2025、多言語対応 |

### 3.3 オープンソースツール

| ツール | アプローチ | ライセンス | AGPL互換 | 特徴 | 参考 |
|--------|-----------|-----------|---------|------|------|
| **PyMuPDF4LLM + Layout** | ルールベース + AI | AGPL-3.0 | ✅ 同一 | ヘッダー/フッター検出、見出し識別 | [PyMuPDF Docs](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) |
| **GROBID** | CRF + Deep Learning | Apache-2.0 | ✅ 互換 | 学術論文特化、90%+ F1 | [GitHub](https://github.com/kermitt2/grobid) |
| **Unstructured** | Detectron2 + ルール | Apache-2.0 | ✅ 互換 | 汎用文書対応 | [GitHub](https://github.com/Unstructured-IO/unstructured) |
| **Marker** | マルチモデル統合 | GPL-3.0 (コード) | ✅ 互換 | Markdown出力、LLM統合 | [GitHub](https://github.com/VikParuchuri/marker) |
| **Detectron2** | 物体検出フレームワーク | Apache-2.0 | ✅ 互換 | Facebook AI Research | [GitHub](https://github.com/facebookresearch/detectron2) |

### 3.4 ライセンス互換性分析

本プロジェクトは **AGPL-3.0** でライセンスされているため、依存関係のライセンス互換性を確認する必要がある。

#### ライセンス互換性マトリックス

| ライセンス | AGPL-3.0との互換性 | 商用利用 | 備考 |
|-----------|-------------------|---------|------|
| **AGPL-3.0** | ✅ 同一 | ✅ 可能 | PyMuPDF, PyMuPDF4LLM |
| **Apache-2.0** | ✅ 互換 | ✅ 可能 | GROBID, Unstructured, Detectron2 |
| **MIT** | ✅ 互換 | ✅ 可能 | LiLT, scikit-learn的な依存 |
| **BSD-3-Clause** | ✅ 互換 | ✅ 可能 | scikit-learn |
| **GPL-3.0** | ✅ 互換 | ✅ 可能 | Marker (コード部分) |
| **CDLA-Permissive-1.0** | ✅ 互換 | ✅ 可能 | PubLayNet, DocLayNet |
| **CC BY-NC-SA 4.0** | ⚠️ 非商用のみ | ❌ 不可 | LayoutLMv3/DiTモデル重み |

#### 推奨ツールのライセンス詳細

| ツール | コードライセンス | モデル/データライセンス | 商用利用 |
|--------|----------------|---------------------|---------|
| **PyMuPDF4LLM** | AGPL-3.0 | AGPL-3.0 | ✅ (AGPL遵守で可) |
| **pymupdf-layout** | AGPL-3.0 (PyMuPDF Pro) | - | ✅ (AGPL遵守で可) |
| **scikit-learn** | BSD-3-Clause | - | ✅ 制限なし |
| **LiLT** | MIT | MIT | ✅ 制限なし |
| **GROBID** | Apache-2.0 | Apache-2.0 | ✅ 制限なし |

#### 注意が必要なツール

| ツール | 問題点 | 代替案 |
|--------|--------|--------|
| **LayoutLMv3** | モデル重みが CC BY-NC-SA 4.0（非商用） | LiLT (MIT) を使用 |
| **DiT** | モデル重みが CC BY-NC-SA 4.0（非商用） | LiLT (MIT) を使用 |
| **Marker** | モデル重みに収益制限あり（$2M未満のみ無料） | PyMuPDF4LLM を使用 |
| **Unstructured** | 一部依存関係がAGPL（ultralytics等） | 依存関係を確認して使用 |

#### 結論

**推奨アプローチ（Phase 1: PyMuPDF4LLM + Layout）は完全にAGPL-3.0互換**であり、ライセンス上の問題はない。

Phase 3でML強化を行う場合：
- ✅ scikit-learn (BSD-3-Clause) - 互換
- ✅ LiLT (MIT) - 互換、商用可
- ❌ LayoutLMv3 (CC BY-NC-SA 4.0) - 非商用のみ、避けるべき

### 3.5 比較評価 (2024年論文より)

"A Comparative Study of PDF Parsing Tools" の結果:

| ツール | 学術論文 | 特許 | 法律文書 |
|--------|---------|------|---------|
| PyMuPDF | 良好 | 困難 | 良好 |
| Unstructured | 中程度 | 中程度 | 良好 |
| Nougat | 良好 | 困難 | 困難 |

**結論**: 学術論文特化であればPyMuPDFベースで十分な精度が期待できる

## 4. 改善アプローチの選択

### 4.1 アプローチ比較

| アプローチ | 実装コスト | 精度向上 | 依存関係 | 推奨度 |
|-----------|-----------|---------|---------|--------|
| **A: ルールベース強化** | 低 | 中 | なし | ★★★ |
| **B: PyMuPDF4LLM統合** | 低 | 高 | pymupdf-layout | ★★★★ |
| **C: 軽量ML導入** | 中 | 高 | scikit-learn | ★★★ |
| **D: LayoutLM統合** | 高 | 最高 | transformers, torch | ★★ |
| **E: GROBID統合** | 中 | 高 | Java, Docker | ★★ |

### 4.2 推奨アプローチ: B + A のハイブリッド

**Phase 1**: PyMuPDF4LLM + Layout 統合（即効性）
**Phase 2**: カスタムルール追加（学術論文特化）
**Phase 3**: オプショナルML強化（将来拡張）

## 5. 詳細実装計画

### Phase 1: PyMuPDF4LLM + Layout 統合

#### 5.1.1 目標
- PyMuPDF Layout の AI ベースレイアウト分析を活用
- ヘッダー/フッター/見出しの自動検出
- 既存APIとの互換性維持

#### 5.1.2 変更ファイル

```
src/index_pdf_translation/
├── core/
│   ├── pdf_edit.py          # 抽出関数の拡張
│   └── translate.py         # 新しい抽出結果の利用
├── config.py                 # 新オプション追加
└── cli.py                    # CLIオプション追加

pyproject.toml                # オプション依存関係追加
```

#### 5.1.3 実装詳細

**Step 1: 依存関係追加**

```toml
# pyproject.toml
[project.optional-dependencies]
layout = ["pymupdf-layout>=0.2.0", "pymupdf4llm>=0.0.17"]
```

**Step 2: 新しい抽出関数**

```python
# pdf_edit.py

async def extract_with_layout(
    pdf_data: bytes,
    include_headers: bool = False,
    include_footers: bool = False,
) -> tuple[DocumentBlocks, LayoutMetadata]:
    """
    PyMuPDF Layout を使用した高精度テキスト抽出。

    Returns:
        (blocks, metadata) where metadata contains:
        - headers: list of header blocks
        - footers: list of footer blocks
        - headings: list of heading blocks with level
        - body: list of body text blocks
        - figures: list of figure/table blocks
    """
    try:
        import pymupdf.layout
        import pymupdf4llm
        use_layout = True
    except ImportError:
        use_layout = False
        logger.warning("pymupdf-layout not installed, using fallback")

    if use_layout:
        # PyMuPDF Layout による高精度抽出
        ...
    else:
        # フォールバック: 既存の extract_text_coordinates_dict
        ...
```

**Step 3: LayoutMetadata データクラス**

```python
@dataclass
class BlockMetadata:
    """個別ブロックのメタデータ"""
    block_type: str  # "heading", "body", "header", "footer", "figure", "table"
    heading_level: Optional[int] = None  # 1-6 for headings
    confidence: float = 1.0

@dataclass
class LayoutMetadata:
    """ページ全体のレイアウトメタデータ"""
    headers: DocumentBlocks
    footers: DocumentBlocks
    headings: DocumentBlocks  # with heading_level
    body: DocumentBlocks
    figures: DocumentBlocks
    tables: DocumentBlocks
```

**Step 4: TranslationConfig 拡張**

```python
@dataclass
class TranslationConfig:
    # 既存フィールド...

    # 新規: レイアウト分析オプション
    use_layout_analysis: bool = True  # PyMuPDF Layout を使用
    translate_headings: bool = True   # 見出しを翻訳対象に含める
    include_headers: bool = False     # ヘッダーを翻訳対象に含める
    include_footers: bool = False     # フッターを翻訳対象に含める
```

#### 5.1.4 フォールバック戦略

```
PyMuPDF Layout インストール済み?
    ├─ Yes → Layout ベース抽出
    └─ No  → 既存アルゴリズム (remove_blocks)
```

### Phase 2: カスタムルール追加

#### 5.2.1 目標
- 学術論文特有のパターンを検出
- PyMuPDF Layout の結果を補完
- フォールバック時の精度向上

#### 5.2.2 追加ルール

**Rule 1: セクション番号パターン**

```python
SECTION_PATTERNS = [
    r"^\d+\.\s",           # "1. Introduction"
    r"^\d+\.\d+\s",        # "1.1 Background"
    r"^\d+\.\d+\.\d+\s",   # "1.1.1 Details"
    r"^[A-Z]\.\s",         # "A. Appendix"
    r"^[IVX]+\.\s",        # "I. Introduction" (Roman numerals)
]

def detect_section_heading(text: str, font_size: float, median_size: float) -> bool:
    """セクション見出しを検出"""
    # パターンマッチ
    for pattern in SECTION_PATTERNS:
        if re.match(pattern, text.strip()):
            return True

    # フォントサイズが本文より大きい + 短いテキスト
    if font_size > median_size * 1.1 and len(text) < 100:
        return True

    return False
```

**Rule 2: ヘッダー/フッター位置検出**

```python
def detect_header_footer_by_position(
    block: BlockInfo,
    page_height: float,
    header_threshold: float = 0.08,  # 上部8%
    footer_threshold: float = 0.05,  # 下部5%
) -> Optional[str]:
    """位置ベースのヘッダー/フッター検出"""
    _, y0, _, y1 = block["coordinates"]

    # 上部領域
    if y1 < page_height * header_threshold:
        return "header"

    # 下部領域
    if y0 > page_height * (1 - footer_threshold):
        return "footer"

    return None
```

**Rule 3: 参考文献パターン**

```python
REFERENCE_PATTERNS = [
    r"^\[\d+\]",              # [1], [2], ...
    r"^\d+\.\s+[A-Z]",        # "1. Author..."
    r"^[A-Za-z]+\s+et\s+al",  # "Smith et al."
]

def detect_reference_block(text: str, y_position: float, page_height: float) -> bool:
    """参考文献ブロックを検出"""
    # ページ下部に集中
    if y_position < page_height * 0.5:
        return False

    # パターンマッチ
    for pattern in REFERENCE_PATTERNS:
        if re.match(pattern, text.strip()):
            return True

    return False
```

**Rule 4: 図表キャプション強化**

```python
FIG_PATTERNS = [
    r"^(?:Figure|Fig\.?)\s*\d+",
    r"^(?:Table|Tab\.?)\s*\d+",
    r"^(?:Algorithm|Alg\.?)\s*\d+",
    r"^(?:Equation|Eq\.?)\s*\d+",
    # 日本語
    r"^図\s*\d+",
    r"^表\s*\d+",
]
```

#### 5.2.3 ルール統合

```python
async def classify_blocks_with_rules(
    blocks: DocumentBlocks,
    page_heights: list[float],
    lang: str = "en",
) -> ClassificationResult:
    """ルールベースのブロック分類"""

    # 1. 統計情報の計算
    font_sizes = [b["size"] for page in blocks for b in page]
    median_size = median(font_sizes)

    result = ClassificationResult()

    for page_idx, page_blocks in enumerate(blocks):
        page_height = page_heights[page_idx]

        for block in page_blocks:
            # Rule 1: 位置ベースのヘッダー/フッター
            pos_type = detect_header_footer_by_position(block, page_height)
            if pos_type:
                result.add(block, pos_type)
                continue

            # Rule 2: セクション見出し
            if detect_section_heading(block["text"], block["size"], median_size):
                result.add(block, "heading", heading_level=guess_level(block))
                continue

            # Rule 3: 図表キャプション
            if detect_figure_caption(block["text"], lang):
                result.add(block, "figure")
                continue

            # Rule 4: 参考文献
            if detect_reference_block(block["text"], block["coordinates"][1], page_height):
                result.add(block, "reference")
                continue

            # デフォルト: 本文
            result.add(block, "body")

    return result
```

### Phase 3: オプショナルML強化（将来）

#### 5.3.1 目標
- ルールで判定困難なケースをMLで補完
- 軽量モデル（scikit-learn）で実装
- GPU不要

#### 5.3.2 特徴量設計

```python
def extract_features(block: BlockInfo, context: DocumentContext) -> np.ndarray:
    """ブロックの特徴量を抽出"""
    return np.array([
        # 基本特徴量
        block["size"] / context.median_font_size,  # 相対フォントサイズ
        len(block["text"]),                         # 文字数
        len(tokenize_text(context.lang, block["text"])),  # トークン数

        # 位置特徴量
        block["coordinates"][1] / context.page_height,  # 相対Y位置
        (block["coordinates"][2] - block["coordinates"][0]) / context.page_width,  # 相対幅

        # テキスト特徴量
        int(bool(re.match(r"^\d+\.", block["text"]))),  # 数字開始
        int(block["text"].isupper()),                    # 全大文字
        int(bool(re.search(r"[.!?]$", block["text"]))), # 文末記号

        # 文脈特徴量
        context.prev_block_type_encoded,  # 前ブロックタイプ
        context.next_block_type_encoded,  # 次ブロックタイプ
    ])
```

#### 5.3.3 モデル選択

```python
# 軽量で高速なモデル
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
)
```

## 6. CLI/API インターフェース

### 6.1 CLI オプション追加

```bash
# 見出しを翻訳対象に含める（デフォルト: 含める）
translate-pdf paper.pdf --translate-headings
translate-pdf paper.pdf --no-translate-headings

# レイアウト分析を無効化（レガシーモード）
translate-pdf paper.pdf --no-layout-analysis

# ヘッダー/フッターを含める
translate-pdf paper.pdf --include-headers --include-footers
```

### 6.2 Python API

```python
from index_pdf_translation import pdf_translate, TranslationConfig

# 見出しを翻訳対象に含める
config = TranslationConfig(
    translate_headings=True,
    use_layout_analysis=True,
)
result = await pdf_translate(pdf_data, config=config)

# レガシーモード（既存アルゴリズム）
config = TranslationConfig(
    use_layout_analysis=False,
)
```

## 7. テスト計画

### 7.1 単体テスト

```python
# tests/test_layout_analysis.py

class TestLayoutAnalysis:
    """レイアウト分析のテスト"""

    async def test_heading_detection(self):
        """見出し検出のテスト"""
        ...

    async def test_header_footer_detection(self):
        """ヘッダー/フッター検出のテスト"""
        ...

    async def test_fallback_without_layout(self):
        """Layout未インストール時のフォールバック"""
        ...
```

### 7.2 統合テスト

```python
# tests/test_integration_layout.py

class TestLayoutIntegration:
    """レイアウト分析の統合テスト"""

    @pytest.mark.parametrize("pdf_file", PDF_FIXTURES)
    async def test_classification_accuracy(self, pdf_file):
        """分類精度のテスト"""
        ...
```

### 7.3 評価指標

| 指標 | 目標値 | 測定方法 |
|------|--------|---------|
| 見出し検出 Precision | > 90% | 手動ラベル付きテストセット |
| 見出し検出 Recall | > 85% | 同上 |
| ヘッダー/フッター除外率 | > 95% | 同上 |
| 本文検出 F1 | > 90% | 同上 |

## 8. マイグレーション計画

### 8.1 後方互換性

- `use_layout_analysis=False` で既存動作を維持
- 既存の `remove_blocks()` 関数は内部的に保持
- APIシグネチャの変更なし

### 8.2 デフォルト値

```python
# v3.2.0 での変更
use_layout_analysis = True   # デフォルトで新アルゴリズム
translate_headings = True    # 見出しも翻訳
```

### 8.3 非推奨化スケジュール

| バージョン | 変更内容 |
|-----------|---------|
| v3.2.0 | Layout分析をデフォルト有効化 |
| v3.3.0 | 旧アルゴリズムを deprecated 警告 |
| v4.0.0 | 旧アルゴリズムを削除（オプション） |

## 9. 実装スケジュール

### Phase 1: PyMuPDF4LLM統合（優先度: 高）

| タスク | 見積もり |
|--------|---------|
| 依存関係追加 | 0.5h |
| 新抽出関数実装 | 2h |
| Config拡張 | 1h |
| CLI拡張 | 1h |
| フォールバック実装 | 1h |
| テスト | 2h |
| **合計** | **7.5h** |

### Phase 2: カスタムルール追加（優先度: 中）

| タスク | 見積もり |
|--------|---------|
| ルール関数実装 | 3h |
| 統合ロジック | 2h |
| テスト | 2h |
| **合計** | **7h** |

### Phase 3: ML強化（優先度: 低、将来）

| タスク | 見積もり |
|--------|---------|
| 特徴量設計 | 2h |
| モデル実装 | 3h |
| 訓練データ準備 | 4h |
| テスト・評価 | 3h |
| **合計** | **12h** |

## 10. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| PyMuPDF Layout の精度不足 | 中 | カスタムルールで補完 |
| 依存関係の増加 | 低 | オプショナル依存として追加 |
| 後方互換性の破壊 | 高 | フォールバックモード提供 |
| パフォーマンス低下 | 中 | ベンチマーク実施、最適化 |

## 11. 参考資料

### 研究論文
- [LayoutLMv3](https://arxiv.org/abs/2204.08387) - Document AI Transformer
- [DocLayNet](https://arxiv.org/abs/2206.01062) - Layout Analysis Dataset
- [TBRF](https://arxiv.org/html/2305.17401) - Rule-based + ML Hybrid
- [PDF Parsing Comparative Study](https://arxiv.org/html/2410.09871v1) - 2024年比較研究

### オープンソースツール
- [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [GROBID](https://github.com/kermitt2/grobid)
- [Unstructured](https://github.com/Unstructured-IO/unstructured)

### データセット
- [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)
- [DocLayNet](https://github.com/DS4SD/DocLayNet)
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
