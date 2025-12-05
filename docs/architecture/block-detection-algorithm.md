# PDFブロック検出・分類アルゴリズム

本ドキュメントでは、Index PDF Translationがどのようにpdfからテキストブロックを検出し、翻訳対象となる「本文」ブロックを識別しているかを説明します。

## 1. 概要

学術論文PDFには以下のような様々な要素が含まれます：

- **本文テキスト**: 翻訳対象（論文のメインコンテンツ）
- **タイトル・著者名**: 通常大きなフォントで短い
- **図表キャプション**: 「Fig.」「Table」などで始まる
- **数式・記号**: 翻訳不要
- **ヘッダー・フッター**: ページ番号など
- **参考文献**: 短い行が多数

本システムは、統計的手法を用いてこれらを自動分類し、本文テキストのみを翻訳対象として抽出します。

## 2. 処理フロー

```
PDF入力
    │
    ▼
┌─────────────────────────────┐
│  1. テキストブロック抽出     │  extract_text_coordinates_dict()
│     PyMuPDFでブロック情報取得 │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  2. 特徴量抽出              │
│     - トークン数            │
│     - ブロック幅            │
│     - フォントサイズ        │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  3. スコア計算              │  remove_blocks()
│     IQR（四分位範囲）ベース  │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  4. ヒストグラム分析        │
│     最頻出スコア範囲を特定   │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  5. ブロック分類            │
│     - 本文ブロック          │
│     - 図表ブロック          │
│     - 除外ブロック          │
└─────────────────────────────┘
    │
    ▼
翻訳処理へ
```

## 3. テキストブロック抽出

### 3.1 PyMuPDFによる抽出

`extract_text_coordinates_dict()` 関数がPDFからテキストブロックを抽出します。

```python
# pdf_edit.py:99-146
async def extract_text_coordinates_dict(pdf_data: bytes) -> DocumentBlocks:
    document = fitz.open(stream=pdf_data, filetype="pdf")

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text_instances_dict = page.get_text("dict")  # 詳細形式で取得
        text_instances = text_instances_dict["blocks"]

        for lines in text_instances:
            if lines["type"] != 0:  # テキストブロック以外はスキップ
                continue

            block = {
                "page_no": page_num,
                "block_no": lines["number"],
                "coordinates": lines["bbox"],  # (x0, y0, x1, y1)
                "text": "",
                "size": 0.0,  # フォントサイズ（平均値）
                "font": "",
            }
```

### 3.2 抽出される情報

各ブロックには以下の情報が含まれます：

| フィールド | 説明 |
|-----------|------|
| `page_no` | ページ番号（0始まり） |
| `block_no` | ブロック番号 |
| `coordinates` | バウンディングボックス `(x0, y0, x1, y1)` |
| `text` | ブロック内のテキスト |
| `size` | フォントサイズ（span平均値） |
| `font` | フォント名 |

## 4. スコア計算アルゴリズム

### 4.1 3つの特徴量

ブロック分類には以下の3つの特徴量を使用します：

1. **トークン数 (T)**: spaCyでトークン化した単語数
2. **ブロック幅 (W)**: `x1 - x0`（ピクセル）
3. **フォントサイズ (S)**: ブロック内の平均フォントサイズ

### 4.2 トークン数スコア

```python
# pdf_edit.py:212-218
token_threshold = 10  # デフォルト

for text in texts:
    if token_threshold <= text:
        scores.append([0.0])  # 十分なトークン数 → 本文候補
    else:
        scores.append([1.0])  # 少ないトークン数 → 除外候補
```

**解釈**:
- トークン数 ≥ 10: スコア `0.0`（本文らしい）
- トークン数 < 10: スコア `1.0`（タイトルや短いテキスト）

### 4.3 IQR（四分位範囲）ベースのスコア

幅とフォントサイズには **ロバストスケーリング** を適用します。

```python
# pdf_edit.py:220-232
for item in [widths, sizes]:
    item_median = median(item)
    item_75_percentile = np.percentile(item, 75)
    item_25_percentile = np.percentile(item, 25)

    for value, score_list in zip(item, scores):
        iqr = item_75_percentile - item_25_percentile
        if iqr > 0:
            score = abs((value - item_median) / iqr)
        else:
            score = 0.0
        score_list.append(score)
```

**IQRスコアの計算式**:

```
score = |value - median| / IQR
```

ここで：
- `IQR = Q3 - Q1`（第3四分位数 - 第1四分位数）
- 中央値から離れるほどスコアが高くなる
- 外れ値（タイトルなど）は高スコア

### 4.4 合計スコア

```python
# pdf_edit.py:234-235
marge_score = [sum(list_score) for list_score in scores]
```

各ブロックの合計スコア = トークンスコア + 幅スコア + サイズスコア

**スコアの解釈**:
- **低スコア** (0に近い): 本文ブロックの可能性が高い
- **高スコア**: タイトル、キャプション、その他の要素

## 5. ヒストグラム分析

### 5.1 ビン数の決定

2つの方法を組み合わせて最適なビン数を決定します：

```python
# pdf_edit.py:237-249
# スタージェスの公式
num_bins_sturges = math.ceil(math.log2(n) + 1)

# フリードマン・ダイアコニスの規則
bin_width_fd = 2 * iqr / n ** (1/3)
num_bins_fd = math.ceil(bin_range / bin_width_fd)

# 小さい方を採用
num_bins = min(num_bins_sturges, num_bins_fd)
```

### 5.2 最頻出範囲の特定

```python
# pdf_edit.py:251-253
histogram, bin_edges = np.histogram(marge_score, bins=num_bins)
max_index = np.argmax(histogram)
most_frequent_range = (bin_edges[max_index], bin_edges[max_index + 1])
```

**論理的根拠**:
- 学術論文では本文ブロックが最も多い
- 本文ブロックは類似した特徴を持つ（類似スコア）
- ヒストグラムの最頻出ビン = 本文ブロックのスコア範囲

## 6. ブロック分類

### 6.1 分類条件

```python
# pdf_edit.py:268-290
# 本文ブロックの条件
result = (
    most_frequent_range[0] <= score <= most_frequent_range[1]  # 最頻出範囲内
    and scores[i][0] == 0  # トークン数が閾値以上
)

# 図表キーワードチェック
keywords = ["fig", "table"]  # 英語の場合
is_figure = _check_first_num_tokens(tokens_list, keywords)

if is_figure:
    page_fig_table_blocks.append(block)
elif result:
    page_filtered_blocks.append(block)  # 本文
else:
    page_removed_blocks.append(block)   # 除外
```

### 6.2 分類結果

| カテゴリ | 条件 | 処理 |
|---------|------|------|
| **本文ブロック** | 最頻出スコア範囲内 AND トークン数≥閾値 | 翻訳対象 |
| **図表ブロック** | 先頭トークンに "fig", "table" など | 翻訳対象（別処理） |
| **除外ブロック** | 上記以外 | 翻訳対象外 |

### 6.3 図表キーワード

```python
# pdf_edit.py:42-44
FIG_KEYWORDS_EN = ["fig", "table"]
FIG_KEYWORDS_JA = ["表", "グラフ"]
```

先頭2トークンにこれらのキーワードが含まれる場合、図表キャプションとして分類されます。

## 7. spaCyトークナイザー

### 7.1 トークン化処理

```python
# tokenizer.py:55-75
def tokenize_text(lang_code: str, text: str) -> list[str]:
    nlp = load_model(lang_code)  # en_core_web_sm or ja_core_news_sm

    if nlp is None:
        return []

    doc = nlp(text)
    # アルファベット文字で構成されるトークンのみを抽出
    tokens = [token.text for token in doc if token.is_alpha]

    return tokens
```

### 7.2 前処理

トークン化前にテキストをクリーニングします：

```python
# pdf_edit.py:202-208
text = text.replace("\n", "")
text = "".join(
    char
    for char in text
    if char not in string.punctuation and char not in string.digits
)
```

- 改行を削除
- 句読点を削除
- 数字を削除

## 8. デバッグモード

`--debug` オプションを有効にすると、以下の可視化画像が生成されます：

1. **トークン数分布**: 各ブロックのトークン数ヒストグラム
2. **フォントサイズ分布**: フォントサイズのヒストグラム
3. **スコア分布**: 合計スコアのヒストグラム（分類閾値を表示）

```bash
uv run translate-pdf paper.pdf --debug
```

## 9. アルゴリズムの特徴

### 9.1 長所

- **統計的アプローチ**: 固定閾値ではなく、文書ごとの分布から閾値を動的に決定
- **ロバスト性**: IQRを使用することで外れ値の影響を軽減
- **言語対応**: spaCyモデルにより英語・日本語の両方に対応

### 9.2 制限事項

- **spaCy依存**: モデルがインストールされていないと機能しない
- **本文が最多の前提**: 図表中心のPDFでは精度が低下する可能性
- **単一列レイアウト最適化**: 複雑なマルチカラムレイアウトには未対応

## 10. 関連ファイル

| ファイル | 役割 |
|---------|------|
| `core/pdf_edit.py` | PDFブロック抽出・分類のメイン処理 |
| `nlp/tokenizer.py` | spaCyによるトークン化 |
| `config.py` | サポート言語・spaCyモデル設定 |

## 11. 参考

- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [spaCy Models](https://spacy.io/models)
- [IQR (Interquartile Range)](https://en.wikipedia.org/wiki/Interquartile_range)
- [Sturges' Rule](https://en.wikipedia.org/wiki/Sturges%27s_rule)
- [Freedman-Diaconis Rule](https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule)
