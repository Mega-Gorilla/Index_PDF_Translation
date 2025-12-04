# Issue #25: deep-translator を使用した Google 翻訳対応

## 概要

翻訳機能を `deep-translator` ライブラリベースに刷新し、**デフォルトで Google 翻訳**を使用する。
APIキー不要で即座に翻訳が可能となり、開発者体験が大幅に向上する。
DeepL は高品質オプションとして引き続きサポートする。

## 設計方針

### 基本方針

| 項目 | 変更前 | 変更後 |
|------|--------|--------|
| デフォルト翻訳エンジン | DeepL（APIキー必須） | **Google（APIキー不要）** |
| 後方互換性 | - | **無視**（Breaking Change許容） |
| 翻訳ライブラリ | aiohttp 直接 | **deep-translator** |

### リリース戦略

**Breaking Change のためメジャーバージョンアップ**: `v2.x.x` → `v3.0.0`

#### 影響範囲

| 対象 | 影響 | 対応 |
|------|------|------|
| CLI | `--api-key` 必須 → 不要に | 引数は残す（DeepL用）、エラーメッセージ変更 |
| テスト | DEEPL_API_KEY 前提のテスト | モック化 or スキップ条件追加 |
| 既存利用者 | `TranslationConfig(api_key=...)` が必須 | `backend="deepl"` 明示で動作 |
| CI | API キー未設定で失敗 | Google デフォルトで通過可能に |

#### リリースチェックリスト

- [ ] CHANGELOG.md に Breaking Changes を明記
- [ ] README.md にマイグレーションガイド追加
- [ ] PyPI リリースノートに互換性情報記載

### アーキテクチャ

```
cli.py
  └── TranslationConfig (config.py)
        └── pdf_translate (core/translate.py)
              └── translate_blocks() ─── 改行連結ロジック
                    └── TranslatorBackend.translate()
                          ├── GoogleTranslator (デフォルト)
                          └── DeepLTranslator (オプション)
```

### Strategy パターン

翻訳バックエンドを抽象化し、実行時に切り替え可能にする。

```
TranslatorBackend (Protocol)
    ├── GoogleTranslator  - デフォルト、APIキー不要
    └── DeepLTranslator   - 高品質、APIキー必要
```

---

## 設計判断: translate_batch の廃止

### 問題分析

deep-translator の `translate_batch()` 内部実装を調査した結果：

```python
# deep_translator/base.py の実装
def _translate_batch(self, batch, **kwargs):
    for i, text in enumerate(batch):
        translated = self.translate(text, **kwargs)  # 個別に translate() を呼び出し
```

**`translate_batch()` は内部でループして個別に `translate()` を呼んでいるだけ。**

### 比較分析

| 観点 | 改行連結方式（採用） | translate_batch（不採用） |
|------|---------------------|--------------------------|
| API コール数 | **1回** | N回（テキスト数分） |
| レート制限リスク | **低** | 高（大量リクエスト） |
| 翻訳品質 | **文脈維持可能** | 文脈断絶（個別翻訳） |
| パフォーマンス | **高速** | 低速（逐次処理） |

### 学術論文での影響

- 典型的な論文: 100〜500 ブロック
- `translate_batch` 使用時: 100〜500回の HTTP リクエスト
- Google の無料翻訳はレート制限あり → **ブロックされるリスク大**

### 結論

**`translate_batch` インターフェースは不要。`translate()` のみで改行連結方式を採用。**

- 改行連結ロジックは `translate_blocks()` 側で実装
- 各 TranslatorBackend は単純な `translate(text, target_lang)` のみ実装
- 責務の分離：バックエンドは翻訳のみ、バッチ処理は呼び出し側

---

## 設計判断: LANG_MAP の廃止

### 問題分析

当初の設計では各 Translator に `LANG_MAP` を定義していた：

```python
# 当初の設計（問題あり）
class GoogleTranslator:
    LANG_MAP = {"en": "en", "ja": "ja"}  # 無意味な変換

class DeepLTranslator:
    LANG_MAP = {"en": "EN", "ja": "JA"}  # config.py と重複
```

### 問題点

1. **重複**: `config.py` の `SUPPORTED_LANGUAGES` と各 Translator で言語コードが重複
2. **不整合リスク**: 言語追加時に3箇所の更新が必要
3. **Google の LANG_MAP は無意味**: `"en" -> "en"` は何もしていない

### 調査結果

- **deep-translator**: `"en"`, `"ja"` をそのまま受け付ける
- **DeepL API**: 言語コードは大文字小文字を区別しない（ISO 639-1）

### 結論

**`LANG_MAP` を完全廃止。言語コード変換はシンプルに処理。**

| バックエンド | 変換方法 |
|-------------|---------|
| Google | `target_lang` をそのまま使用 |
| DeepL | `target_lang.upper()` で変換 |

### メリット

| 観点 | LANG_MAP あり | LANG_MAP なし（採用） |
|------|--------------|---------------------|
| コード量 | 各 Translator に定義 | **不要** |
| 言語追加 | 3箇所更新 | **1箇所のみ**（SUPPORTED_LANGUAGES） |
| 不整合リスク | あり | **なし** |
| 保守性 | 低 | **高** |

---

## 設計判断: セパレータトークン方式の採用

### 問題分析

改行連結方式 (`"\n".join(texts)`) には以下のリスクがある：

```python
# 入力
texts = ["Hello", "", "World"]  # 3行（空行含む）
combined = "Hello\n\nWorld"

# Google翻訳後（空行が消える可能性）
translated = "こんにちは\n世界"  # 2行に

# split後
lines = translated.split("\n")  # len=2 ≠ 3 → ブロック対応崩壊
```

**Google翻訳は改行や空行を保持しない場合がある**

### 検証テスト結果

複数のセパレータ候補を実際の Google 翻訳 API でテスト：

| セパレータ | 成功率 | 問題点 |
|-----------|--------|--------|
| **`[[[BR]]]`** | **100%** | なし ✅ **推奨** |
| `###SPLIT###` | 100% | なし |
| `<<SEP>>` | 60% | SEP→「9月」と翻訳される |
| `\n---\n` | 60% | 空行で崩壊 |
| `|||BLOCK|||` | 0% | BLOCK→「ブロック」と翻訳される |

テストケース：
1. 基本的な複数行: `["Hello", "World", "Good morning"]`
2. 空文字列を含む: `["Hello", "", "World"]`
3. 空白のみを含む: `["Hello", "   ", "World"]`
4. 長めの学術論文風テキスト

### 重要な発見

- **英単語を含むトークンは翻訳される**: `BLOCK` → `ブロック`、`SEP` → `9月`
- **記号のみのトークンは保持される**: `[[[BR]]]` は翻訳されない
- **空行も正しく処理される**: `[[[BR]]]` 使用時

### 結論

**セパレータトークン `[[[BR]]]` を採用**

```python
BLOCK_SEPARATOR = "[[[BR]]]"

# 連結
combined = BLOCK_SEPARATOR.join(texts)
# "Hello[[[BR]]][[[BR]]]World"

# 翻訳後
translated = "こんにちは[[[BR]]][[[BR]]]世界"

# 分割
lines = translated.split(BLOCK_SEPARATOR)
# ["こんにちは", "", "世界"] ← 正確に3要素
```

### メリット

| 観点 | 改行方式 | セパレータ方式（採用） |
|------|----------|----------------------|
| 空行対応 | ❌ 崩壊リスク | ✅ **正確に保持** |
| 改行保持 | ❌ 不安定 | ✅ **100% 保持** |
| API コール数 | 1回 | 1回（同じ） |
| 実装複雑度 | 低 | 低（join/split のみ） |

---

## 設計判断: 文字数制限とチャンキング戦略

### 問題分析

deep-translator の GoogleTranslator には **1回のリクエストで 5,000 文字** という制限がある。

参考: [deep-translator Issue #100](https://github.com/nidhaloff/deep-translator/discussions/100)

```
学術論文の例:
- 100ブロック × 平均150文字 = 15,000文字
- セパレータ「[[[BR]]]」× 99 = 約800文字
- 合計: 約16,000文字 → 制限の3倍超
```

### 解決策: チャンキング

テキストを文字数制限内の複数チャンクに分割し、それぞれを翻訳後に結合する。

```python
MAX_CHUNK_SIZE = 4500  # 5000より余裕を持たせる（セパレータ分も考慮）

def chunk_texts_for_translation(
    texts: list[str],
    separator: str,
    max_size: int = MAX_CHUNK_SIZE,
) -> list[list[str]]:
    """
    テキストリストを文字数制限内のチャンクに分割。

    Args:
        texts: 翻訳するテキストのリスト
        separator: セパレータトークン
        max_size: 1チャンクの最大文字数

    Returns:
        チャンクに分割されたテキストリストのリスト

    Note:
        単一ブロックが max_size を超える場合はそのまま単独チャンクとして扱う。
        API 側でエラーになる可能性があるが、学術論文では稀なケースのため
        Phase 1 ではハードエラーとして処理し、ログで警告を出力する。
    """
    chunks: list[list[str]] = []
    current_chunk: list[str] = []
    current_size = 0
    separator_len = len(separator)

    for text in texts:
        # 単一ブロックが制限を超える場合は警告してそのまま追加
        if len(text) > max_size:
            # 現在のチャンクを先に保存
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            # 巨大ブロックを単独チャンクとして追加（APIエラーになる可能性あり）
            chunks.append([text])
            logger.warning(
                f"Single block exceeds MAX_CHUNK_SIZE ({len(text)} > {max_size}). "
                f"May fail at translation API."
            )
            continue

        # セパレータを含めたサイズを計算
        item_size = len(text)
        if current_chunk:
            item_size += separator_len

        # 現在のチャンクに追加すると制限を超える場合
        if current_size + item_size > max_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
            item_size = len(text)  # 新チャンクなのでセパレータ不要

        current_chunk.append(text)
        current_size += item_size

    # 最後のチャンクを追加
    if current_chunk:
        chunks.append(current_chunk)

    return chunks
```

### 巨大ブロック（単一ブロック > MAX_CHUNK_SIZE）の対応方針

| 方針 | 採用 | 理由 |
|------|------|------|
| **ハードエラー（警告ログ）** | ✅ Phase 1 | シンプル、学術論文では稀 |
| 句点で分割 | ❌ | 分割後の再結合が複雑、翻訳品質低下 |
| 別 API 使用 | ❌ | 実装複雑度増加 |

**学術論文での実態**:
- 一般的な段落: 200〜500 文字
- 長めの段落: 1,000〜2,000 文字
- 4,500 文字超の単一ブロック: **極めて稀**（Abstract 全体でも 2,000 文字程度）

**Phase 1 での対応**:
1. 警告ログを出力
2. そのまま API に送信（エラーになる可能性あり）
3. エラー時は `TranslationError` として上位に伝播

**将来の改善案（Phase 2 以降）**:
- 句点・改行での自動分割
- ユーザーへの警告メッセージ表示
- 手動分割のガイダンス

### 翻訳フロー

```
入力: 100ブロック（合計16,000文字）
  ↓
チャンキング: 4チャンク（各約4,000文字）
  ↓
翻訳: 4回のAPIコール
  ↓
結合: 100ブロックの翻訳結果
```

### トレードオフ

| 観点 | チャンキングなし | チャンキングあり（採用） |
|------|-----------------|------------------------|
| APIコール数 | 1回 | N回（チャンク数） |
| 文字数制限 | ❌ 超過でエラー | ✅ 対応 |
| 文脈維持 | 最大 | チャンク境界で分断 |
| 実装複雑度 | 低 | 中 |

**注意**: チャンク境界での文脈分断は許容。学術論文では段落単位でブロックが分かれるため、実用上の影響は小さい。

---

## 設計判断: エラーリトライ戦略

### 想定されるエラー

| バックエンド | エラー種別 | 原因 |
|-------------|-----------|------|
| Google | 429 Too Many Requests | レート制限 |
| Google | 503 Service Unavailable | 一時的な障害 |
| DeepL | 429 Too Many Requests | APIクォータ超過 |
| DeepL | 456 Quota Exceeded | 月間クォータ超過 |

### Phase 1 での実装（固定遅延リトライ）

**採用方針**: シンプルな固定遅延リトライを実装。

| パラメータ | 値 | 理由 |
|-----------|-----|------|
| max_retries | 3 | 一時的なエラーに対応 |
| retry_delay | 1.0秒 | シンプル、過度な待機を避ける |

```python
async def translate_chunk_with_retry(
    translator: "TranslatorBackend",
    text: str,
    target_lang: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """
    固定遅延リトライ付きで翻訳を実行（Phase 1 実装）。

    Args:
        translator: 翻訳バックエンド
        text: 翻訳するテキスト
        target_lang: 翻訳先言語コード
        max_retries: 最大リトライ回数
        retry_delay: リトライ間隔（秒）- 固定

    Returns:
        翻訳されたテキスト

    Raises:
        TranslationError: リトライ後も失敗した場合
    """
    from index_pdf_translation.translators.base import TranslationError

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await translator.translate(text, target_lang)
        except TranslationError as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    f"Translation failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {retry_delay}s: {e}"
                )
                await asyncio.sleep(retry_delay)  # 固定遅延
            else:
                raise

    raise last_error  # 型チェック用（到達しない）
```

### 将来の改善案（Phase 2 以降）

指数バックオフリトライへの移行を検討:

```python
# 将来の実装イメージ
delay = min(base_delay * (2 ** attempt), max_delay)  # 1, 2, 4, 8... 秒
```

メリット:
- レート制限時に段階的に待機時間を増加
- API 側の負荷軽減
- 成功確率向上

---

## 設計判断: DeepL セパレータ互換性

### 確認事項

`[[[BR]]]` セパレータは Google 翻訳で検証済みだが、DeepL での動作確認が必要。

### 検証項目

1. **セパレータ保持**: DeepL が `[[[BR]]]` をそのまま保持するか
2. **tag_handling との相互作用**: `tag_handling: xml` オプションの影響
3. **空行処理**: 連続セパレータ `[[[BR]]][[[BR]]]` の扱い

### 検証タイミング

**実装時（Phase 4）に DeepL でも同じテストを実施**:

```python
# tests/test_separator_deepl.py（手動実行用）
@pytest.mark.integration
async def test_deepl_separator_preserved():
    """DeepL でセパレータが保持されるか確認"""
    translator = DeepLTranslator(api_key=os.environ["DEEPL_API_KEY"])

    text = "Hello[[[BR]]]World[[[BR]]]Good morning"
    result = await translator.translate(text, "ja")

    parts = result.split("[[[BR]]]")
    assert len(parts) == 3, f"Expected 3 parts, got {len(parts)}: {result}"
```

### フォールバック

DeepL でセパレータが翻訳される場合の対策:
- 別のセパレータ候補: `⟦BR⟧`（Unicode 括弧）
- XML タグ: `<br/>` （tag_handling: xml で保護）

---

## 設計判断: 環境変数と .env ファイル

### 方針

**`.env` はオプショナルとしてサポート**。必須にはしない。

| 項目 | 方針 | 理由 |
|------|------|------|
| `.env` サポート | ✅ あり | 開発者体験向上 |
| python-dotenv | ❌ 追加しない | 依存関係を最小限に保つ |
| 必須化 | ❌ しない | Google デフォルトの強み（APIキー不要）を活かす |

### 背景

このプロジェクトの最大の強みは「**APIキー不要で動作する**」こと。
`.env` を必須にすると、その強みが薄れる。
DeepL/OpenAI 使用時の**便利機能**として位置づける。

### ファイル構成

```
.gitignore          # .env を含める（セキュリティ）
.env.example        # テンプレート（コミット対象）
.env                # 実際の値（gitignore 対象）
```

### `.env.example` テンプレート

```bash
# Index PDF Translation - Environment Variables
# Copy this file to .env and fill in your API keys.
# All variables are OPTIONAL - Google Translate works without any API key.

# ============================================================
# DeepL API (for --backend deepl)
# ============================================================
# Get your API key at: https://www.deepl.com/pro-api
# DEEPL_API_KEY=your-deepl-api-key-here

# For DeepL Pro users (default is Free API):
# DEEPL_API_URL=https://api.deepl.com/v2/translate

# ============================================================
# OpenAI API (for --backend openai, future feature)
# ============================================================
# Get your API key at: https://platform.openai.com/api-keys
# OPENAI_API_KEY=your-openai-api-key-here
```

### 読み込み方法

**python-dotenv は使用しない**。ユーザーは以下のいずれかで設定:

| 方法 | コマンド例 |
|------|-----------|
| シェル読み込み | `source .env && translate-pdf paper.pdf --backend deepl` |
| export | `export DEEPL_API_KEY=xxx && translate-pdf ...` |
| CLI オプション | `translate-pdf paper.pdf --backend deepl --api-key xxx` |
| IDE 設定 | VS Code の `.vscode/launch.json` 等 |

### python-dotenv を追加しない理由

| 観点 | 説明 |
|------|------|
| 依存関係 | パッケージサイズ増加を避ける |
| ライブラリ利用 | `import` 時に自動読み込みは副作用になりうる |
| シンプルさ | 現在の `os.environ.get()` で十分 |
| 代替手段 | シェルの `source` や IDE 設定で対応可能 |

### 将来の検討事項

ユーザーからの要望が多い場合、以下の方法で対応可能:

```python
# cli.py のみで読み込み（ライブラリ使用時は影響なし）
def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv がなくても動作
    ...
```

```toml
# pyproject.toml - オプショナル依存として追加
[project.optional-dependencies]
dotenv = ["python-dotenv>=1.0.0"]
```

---

## 拡張計画: OpenAI GPT 翻訳対応

### 背景・目的

GPT モデルを使用した翻訳機能を追加し、以下のメリットを提供する：

1. **専門的な翻訳**: プロンプトで専門分野を指定可能
2. **用語集対応**: プロンプトに用語集を含めて一貫した訳語を使用
3. **長文コンテキスト**: 論文全体を一括送信し、前後の文脈を踏まえた翻訳
4. **カスタマイズ性**: 翻訳スタイルの調整が可能

### モデル調査結果（2025年12月時点）

#### GPT モデル比較

| モデル | 入力（/1M tokens） | 出力（/1M tokens） | コンテキスト長 | 備考 |
|--------|-------------------|-------------------|----------------|------|
| **GPT-4.1-nano** | **$0.10** | **$0.40** | **1M** | ⭐ 最安、推奨 |
| GPT-4.1-mini | $0.40 | $1.60 | 1M | バランス型 |
| GPT-4.1 | $2.00 | $8.00 | 1M | 高品質 |
| GPT-4o-mini | $0.15 | $0.60 | 128K | 旧モデル |
| GPT-5-mini | $0.25 | $2.00 | 400K | 最新 |

参考: [OpenAI Pricing](https://openai.com/api/pricing/), [GPT-4.1 Pricing Calculator](https://livechatai.com/gpt-4-1-pricing-calculator)

#### 翻訳品質（WMT24 コンペティション）

| モデル | 勝利言語ペア数（/11） |
|--------|----------------------|
| Claude 3.5 Sonnet | **9** |
| GPT-4 | 5 |

参考: [Best LLMs for Translation](https://www.getblend.com/blog/which-llm-is-best-for-translation/), [Lokalise LLM Comparison](https://lokalise.com/blog/what-is-the-best-llm-for-translation/)

#### 推奨モデル: GPT-4.1-nano

- **最安価**: GPT-4o-mini より安い（$0.10 vs $0.15）
- **大容量コンテキスト**: 1M トークン（論文全体を一括処理可能）
- **高性能**: GPT-4o-mini より高いインテリジェンススコア
- **キャッシュ割引**: 同一入力の再利用で 75% 割引（$0.025/1M）

### 改行保持問題と解決策

#### 問題

現在の設計は改行ベースのブロック対応に依存：

```python
texts = ["Hello", "World", "Good morning"]
combined = "Hello\nWorld\nGood morning"
translated = await translator.translate(combined, target_lang)
lines = translated.split("\n")  # len(lines) == 3 を期待
```

**LLM は改行を 100% 保持する保証がない**

#### 解決策: Structured Outputs（JSON 配列）

OpenAI の [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) 機能を使用：

```python
from pydantic import BaseModel

class TranslationResult(BaseModel):
    translations: list[str]

response = await client.beta.chat.completions.parse(
    model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(texts)}
    ],
    response_format=TranslationResult,
)
```

**メリット**:
- gpt-4o-2024-08-06 以降、100% のスキーマ準拠率
- 配列の要素数が保証される
- 改行問題を完全に回避

参考: [Structured Outputs Intro](https://cookbook.openai.com/examples/structured_outputs_intro)

### 翻訳プロンプト設計

#### システムプロンプト（デフォルト）

```
You are a professional translator specializing in academic papers.
Translate the following texts from {source_lang} to {target_lang}.

Requirements:
- Preserve technical terminology accurately
- Maintain the exact number of input texts in the output array
- Each input text corresponds to one output text at the same index
- Do not merge or split texts
- Preserve any special characters or formatting within each text

Return a JSON object with a "translations" array containing the translated texts.
```

#### プロンプトのカスタマイズ

**設計方針**: プロンプトは3つのレベルでオーバーライド可能。

| レベル | 用途 | 例 |
|--------|------|-----|
| **デフォルト** | 学術論文向けの汎用プロンプト | 上記テンプレート |
| **コンストラクタ** | プロジェクト固有の固定設定 | 医学論文向け、法律文書向け |
| **メソッド引数** | 動的な切り替え（A/Bテスト等） | 特定ドキュメントの翻訳スタイル変更 |

#### プレースホルダー

プロンプト内で以下のプレースホルダーを使用可能：

| プレースホルダー | 展開例 | 説明 |
|-----------------|--------|------|
| `{source_lang}` | "English" | 翻訳元言語名 |
| `{target_lang}` | "Japanese" | 翻訳先言語名 |

#### カスタムプロンプトの例

```python
# 医学論文向け
MEDICAL_PROMPT = """You are a medical translator specializing in clinical research papers.
Translate from {source_lang} to {target_lang}.

Requirements:
- Use standard medical terminology (MeSH terms when applicable)
- Preserve drug names, dosages, and clinical measurements exactly
- Maintain the exact number of input/output texts

Return a JSON object with a "translations" array."""

# 用語集指定
GLOSSARY_PROMPT = """You are a professional translator.
Translate from {source_lang} to {target_lang}.

Glossary (use these translations consistently):
- "machine learning" → "機械学習"
- "neural network" → "ニューラルネットワーク"
- "gradient descent" → "勾配降下法"

Return a JSON object with a "translations" array."""
```

#### 推奨パラメータ

| パラメータ | 値 | 理由 |
|-----------|-----|------|
| temperature | 0.2 | 決定論的な翻訳結果 |
| top_p | 0.6 | 安定した出力 |
| response_format | Structured Output | 配列構造を保証 |

参考: [GPT-4o Prompt Strategies](https://medium.com/@michalmikuli/gpt-4o-prompt-strategies-in-2025-d2f418cf0a79)

### コスト試算

学術論文1本（約10,000語 ≈ 15,000トークン）の翻訳コスト：

| バックエンド | コスト | 備考 |
|-------------|--------|------|
| Google 翻訳 | **無料** | レート制限あり |
| DeepL Free | **無料** | 月50万文字まで |
| GPT-4.1-nano | **約 $0.006** | 入力+出力 |
| GPT-4.1-mini | 約 $0.024 | |
| GPT-4.1 | 約 $0.12 | |
| DeepL Pro | 約 $0.27 | |

**GPT-4.1-nano は非常に低コスト**（論文1本約0.6円）

### 実装設計

#### `src/index_pdf_translation/translators/openai.py`

```python
# SPDX-License-Identifier: AGPL-3.0-only
"""OpenAI GPT 翻訳バックエンド"""

import json
from typing import Optional

try:
    from openai import AsyncOpenAI
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "openai and pydantic are required for OpenAI backend. "
        "Install with: pip install index-pdf-translation[openai]"
    )

from .base import TranslationError


class TranslationResult(BaseModel):
    """Structured Output 用のスキーマ"""
    translations: list[str]


class OpenAITranslator:
    """
    OpenAI GPT を使用した翻訳バックエンド。

    Structured Outputs で配列構造を保証し、改行問題を回避。
    プロンプトは3レベルでカスタマイズ可能：
    - デフォルト: 学術論文向け汎用プロンプト
    - コンストラクタ: プロジェクト固有の固定設定
    - メソッド引数: 動的な切り替え
    """

    DEFAULT_MODEL = "gpt-4.1-nano"

    DEFAULT_SYSTEM_PROMPT = """You are a professional translator specializing in academic papers.
Translate the following texts from {source_lang} to {target_lang}.

Requirements:
- Preserve technical terminology accurately
- Maintain the exact number of input texts in the output array
- Each input text corresponds to one output text at the same index
- Do not merge or split texts

Return a JSON object with a "translations" array."""

    # 言語コード -> 言語名
    LANG_NAMES = {"en": "English", "ja": "Japanese"}

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        source_lang: str = "en",
        system_prompt: str | None = None,  # コンストラクタでプロンプトをオーバーライド
    ):
        """
        Args:
            api_key: OpenAI API キー
            model: 使用するモデル（デフォルト: gpt-4.1-nano）
            source_lang: 翻訳元言語コード
            system_prompt: カスタムシステムプロンプト（{source_lang}, {target_lang} プレースホルダー使用可能）
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._source_lang = source_lang
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    @property
    def name(self) -> str:
        return "openai"

    def _format_prompt(self, prompt_template: str, target_lang: str) -> str:
        """プロンプトテンプレートをフォーマット"""
        return prompt_template.format(
            source_lang=self.LANG_NAMES.get(self._source_lang, self._source_lang),
            target_lang=self.LANG_NAMES.get(target_lang, target_lang),
        )

    async def translate(
        self,
        text: str,
        target_lang: str,
        *,
        system_prompt: str | None = None,  # メソッドレベルでオーバーライド
    ) -> str:
        """
        単一テキストを翻訳（改行区切りで複数テキストとして処理）

        Args:
            text: 翻訳するテキスト
            target_lang: 翻訳先言語コード
            system_prompt: カスタムプロンプト（指定時はコンストラクタ設定を上書き）
        """
        if not text.strip():
            return text

        # 改行で分割して配列として送信
        texts = text.split("\n")
        translated_texts = await self.translate_texts(
            texts, target_lang, system_prompt=system_prompt
        )
        return "\n".join(translated_texts)

    async def translate_texts(
        self,
        texts: list[str],
        target_lang: str,
        *,
        system_prompt: str | None = None,  # メソッドレベルでオーバーライド
    ) -> list[str]:
        """
        複数テキストを Structured Outputs で翻訳

        Args:
            texts: 翻訳するテキストのリスト
            target_lang: 翻訳先言語コード
            system_prompt: カスタムプロンプト（指定時はコンストラクタ設定を上書き）
        """
        if not texts:
            return []

        # 空文字列のインデックスを記録
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return texts

        # プロンプト解決: メソッド引数 > コンストラクタ > デフォルト
        effective_prompt = system_prompt or self._system_prompt
        formatted_prompt = self._format_prompt(effective_prompt, target_lang)

        try:
            response = await self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": formatted_prompt},
                    {"role": "user", "content": json.dumps(non_empty_texts)},
                ],
                response_format=TranslationResult,
                temperature=0.2,
                top_p=0.6,
            )

            result = response.choices[0].message.parsed
            translated_parts = result.translations

            # 結果を元の位置に戻す
            results = list(texts)
            for idx, translated in zip(non_empty_indices, translated_parts):
                if idx < len(results):
                    results[idx] = translated

            return results

        except Exception as e:
            raise TranslationError(f"OpenAI API error: {e}")
```

#### TranslationConfig 更新

```python
TranslatorBackendType = Literal["google", "deepl", "openai"]

@dataclass
class TranslationConfig:
    backend: TranslatorBackendType = "google"
    api_key: str = field(default_factory=lambda: os.environ.get("DEEPL_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_model: str = "gpt-4.1-nano"
    openai_system_prompt: str | None = None  # カスタムプロンプト（None でデフォルト使用）
    # ...

    def create_translator(self) -> "TranslatorBackend":
        if self.backend == "openai":
            from index_pdf_translation.translators import get_openai_translator
            OpenAITranslator = get_openai_translator()
            return OpenAITranslator(
                api_key=self.openai_api_key,
                model=self.openai_model,
                source_lang=self.source_lang,
                system_prompt=self.openai_system_prompt,  # カスタムプロンプトを渡す
            )
        # ...
```

#### CLI 更新

```bash
# 使用例
translate-pdf paper.pdf --backend openai --openai-model gpt-4.1-nano
translate-pdf paper.pdf --backend openai --openai-model gpt-4.1  # 高品質

# カスタムプロンプト（ファイルから読み込み）
translate-pdf paper.pdf --backend openai --openai-prompt-file prompts/medical.txt

# カスタムプロンプト（直接指定）
translate-pdf paper.pdf --backend openai --openai-prompt "You are a translator for {source_lang} to {target_lang}..."
```

CLI オプション追加:

```python
parser.add_argument(
    "--openai-prompt",
    help="OpenAI バックエンドのカスタムシステムプロンプト（{source_lang}, {target_lang} プレースホルダー使用可能）",
)

parser.add_argument(
    "--openai-prompt-file",
    type=Path,
    help="OpenAI バックエンドのカスタムシステムプロンプトを読み込むファイル",
)
```

### 依存関係

```toml
[project.optional-dependencies]
openai = ["openai>=1.0.0", "pydantic>=2.0.0"]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.1.0",
    "aiohttp>=3.9.0",
    "openai>=1.0.0",
    "pydantic>=2.0.0",
]
```

### メリット・デメリット

| 観点 | メリット | デメリット |
|------|----------|-----------|
| 翻訳品質 | 専門用語の文脈理解、カスタマイズ可能 | Google/DeepL との明確な優位性は要検証 |
| コスト | GPT-4.1-nano は非常に安価 | 無料ではない |
| 改行保持 | Structured Outputs で 100% 保証 | JSON パースのオーバーヘッド |
| 依存関係 | openai ライブラリが必要 | パッケージサイズ増加 |
| API キー | 必須 | Google 翻訳の「APIキー不要」のメリットが薄れる |
| プロンプト | 3レベルでカスタマイズ可能、用語集対応 | プロンプト設計の学習コスト |

### 実装優先度

**Phase 1（Issue #25）で実装する範囲**:
- [x] Google 翻訳（デフォルト）
- [x] DeepL 翻訳（オプション）

**Phase 2（別 Issue）で実装を検討**:
- [ ] OpenAI GPT 翻訳（オプション）

理由：
1. Issue #25 の主目的は「APIキー不要の翻訳」であり、OpenAI は要件を満たさない
2. Structured Outputs の実装は追加の検証が必要
3. まずは Google/DeepL で安定稼働させてから拡張

---

## 実装フェーズ

### Phase 1: 依存関係の更新

#### 1.1 `pyproject.toml` 更新

```toml
dependencies = [
    "PyMuPDF>=1.24.0",
    "spacy>=3.7.0",
    "numpy>=1.26.0",
    "matplotlib>=3.8.0",
    "deep-translator>=1.11.0",  # 追加
]
# aiohttp は DeepL バックエンドでのみ使用するためオプショナルに移動
[project.optional-dependencies]
deepl = ["aiohttp>=3.9.0"]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.1.0",
    "aiohttp>=3.9.0",  # テスト用
]
```

> **Note**: aiohttp を完全に削除せず `[deepl]` extra として残す。
> DeepL を使用するユーザーは `pip install index-pdf-translation[deepl]` でインストール。

---

### Phase 2: 翻訳バックエンド抽象化

#### 2.1 `src/index_pdf_translation/translators/__init__.py`

```python
# SPDX-License-Identifier: AGPL-3.0-only
"""翻訳バックエンドモジュール"""

from .base import TranslatorBackend, TranslationError
from .google import GoogleTranslator

__all__ = [
    "TranslatorBackend",
    "TranslationError",
    "GoogleTranslator",
]


def get_deepl_translator():
    """DeepLTranslator を取得（aiohttp が必要）"""
    from .deepl import DeepLTranslator
    return DeepLTranslator
```

#### 2.2 `src/index_pdf_translation/translators/base.py`

```python
# SPDX-License-Identifier: AGPL-3.0-only
"""翻訳バックエンドの基底クラス"""

from typing import Protocol, runtime_checkable


class TranslationError(Exception):
    """翻訳処理中に発生したエラー"""
    pass


@runtime_checkable
class TranslatorBackend(Protocol):
    """
    翻訳バックエンドのプロトコル定義。

    各バックエンドは translate() メソッドのみ実装する。
    バッチ処理（改行連結）は translate_blocks() 側で行う。
    """

    @property
    def name(self) -> str:
        """バックエンド名（"google", "deepl"）"""
        ...

    async def translate(self, text: str, target_lang: str) -> str:
        """
        テキストを翻訳する。

        改行を含むテキストも受け付け、改行を保持して翻訳する。

        Args:
            text: 翻訳するテキスト（改行含む場合あり）
            target_lang: 翻訳先言語（"en", "ja"）

        Returns:
            翻訳されたテキスト

        Raises:
            TranslationError: 翻訳に失敗した場合
        """
        ...
```

#### 2.3 `src/index_pdf_translation/translators/google.py`

```python
# SPDX-License-Identifier: AGPL-3.0-only
"""Google 翻訳バックエンド（deep-translator 使用）"""

import asyncio

from deep_translator import GoogleTranslator as DTGoogleTranslator
from deep_translator.exceptions import TranslationNotFound

from .base import TranslationError


class GoogleTranslator:
    """
    Google 翻訳を使用した翻訳バックエンド。

    APIキー不要で使用可能。deep-translator ライブラリ経由。
    言語コードは内部コード（"en", "ja"）をそのまま使用。
    """

    @property
    def name(self) -> str:
        return "google"

    async def translate(self, text: str, target_lang: str) -> str:
        """
        テキストを翻訳する。

        改行を含むテキストも対応。1回の API コールで処理。

        Args:
            text: 翻訳するテキスト
            target_lang: 翻訳先言語（"en", "ja"）- そのまま使用
        """
        if not text.strip():
            return text

        def _translate() -> str:
            try:
                # deep-translator は "en", "ja" をそのまま受け付ける
                translator = DTGoogleTranslator(
                    source="auto",
                    target=target_lang
                )
                return translator.translate(text)
            except TranslationNotFound as e:
                raise TranslationError(f"Translation failed: {e}")
            except Exception as e:
                raise TranslationError(f"Google Translate error: {e}")

        return await asyncio.to_thread(_translate)
```

#### 2.4 `src/index_pdf_translation/translators/deepl.py`

```python
# SPDX-License-Identifier: AGPL-3.0-only
"""DeepL 翻訳バックエンド"""

try:
    import aiohttp
except ImportError:
    raise ImportError(
        "aiohttp is required for DeepL backend. "
        "Install with: pip install index-pdf-translation[deepl]"
    )

from .base import TranslationError


class DeepLTranslator:
    """
    DeepL API を使用した翻訳バックエンド。

    高品質な翻訳が可能だが、APIキーが必要。
    言語コードは .upper() で変換（"en" -> "EN"）。
    """

    DEFAULT_API_URL = "https://api-free.deepl.com/v2/translate"

    def __init__(self, api_key: str, api_url: str | None = None):
        """
        Args:
            api_key: DeepL API キー
            api_url: DeepL API URL（None の場合は Free API を使用）
        """
        if not api_key:
            raise ValueError("DeepL API key is required")
        self._api_key = api_key
        self._api_url = api_url or self.DEFAULT_API_URL

    @property
    def name(self) -> str:
        return "deepl"

    async def translate(self, text: str, target_lang: str) -> str:
        """
        テキストを翻訳する。

        改行を含むテキストも対応。1回の API コールで処理。

        Args:
            text: 翻訳するテキスト
            target_lang: 翻訳先言語（"en", "ja"）- .upper() で変換
        """
        if not text.strip():
            return text

        params = {
            "auth_key": self._api_key,
            "text": text,
            "target_lang": target_lang.upper(),  # "en" -> "EN", "ja" -> "JA"
            "tag_handling": "xml",
            "formality": "more",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self._api_url, data=params) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["translations"][0]["text"]
                else:
                    error_text = await response.text()
                    raise TranslationError(
                        f"DeepL API error (status {response.status}): {error_text}"
                    )
```

---

### Phase 3: Config の更新

#### 3.1 `src/index_pdf_translation/config.py`

```python
# SPDX-License-Identifier: AGPL-3.0-only
"""翻訳設定モジュール"""

import os
from dataclasses import dataclass, field
from typing import Literal, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from index_pdf_translation.translators import TranslatorBackend


class LanguageConfig(TypedDict):
    """言語設定の型定義"""
    spacy: str  # spaCyモデル名


# 言語設定
# - 言語コード変換は各 TranslatorBackend が担当
# - Google: "en", "ja" をそのまま使用
# - DeepL: .upper() で "EN", "JA" に変換
SUPPORTED_LANGUAGES: dict[str, LanguageConfig] = {
    "en": {"spacy": "en_core_web_sm"},
    "ja": {"spacy": "ja_core_news_sm"},
}

DEFAULT_OUTPUT_DIR: str = "./output/"

# 翻訳バックエンドの型
TranslatorBackendType = Literal["google", "deepl"]


@dataclass
class TranslationConfig:
    """
    翻訳設定を管理する dataclass。

    デフォルトは Google 翻訳（APIキー不要）。
    高品質な翻訳が必要な場合は DeepL を使用。

    Attributes:
        backend: 翻訳バックエンド ("google" or "deepl")
        api_key: DeepL APIキー（backend="deepl" の場合のみ必要）
        api_url: DeepL API URL（backend="deepl" の場合のみ使用）
        source_lang: 翻訳元言語コード (default: "en")
        target_lang: 翻訳先言語コード (default: "ja")
        add_logo: ロゴウォーターマークを追加 (default: True)
        debug: デバッグモード (default: False)

    Examples:
        >>> # Google 翻訳（デフォルト、APIキー不要）
        >>> config = TranslationConfig()

        >>> # DeepL 翻訳（高品質）
        >>> config = TranslationConfig(
        ...     backend="deepl",
        ...     api_key="your-deepl-key"
        ... )
    """

    backend: TranslatorBackendType = "google"
    api_key: str = field(
        default_factory=lambda: os.environ.get("DEEPL_API_KEY", "")
    )
    api_url: str = field(
        default_factory=lambda: os.environ.get(
            "DEEPL_API_URL",
            "https://api-free.deepl.com/v2/translate"
        )
    )
    source_lang: str = "en"
    target_lang: str = "ja"
    add_logo: bool = True
    debug: bool = False

    def __post_init__(self) -> None:
        """設定値のバリデーション"""
        # DeepL バックエンドの場合は API キーが必須
        if self.backend == "deepl" and not self.api_key:
            raise ValueError(
                "DeepL API key required when using 'deepl' backend. "
                "Set DEEPL_API_KEY environment variable or pass api_key parameter. "
                "Or use backend='google' for API-key-free translation."
            )

        # 言語コードの検証
        if self.source_lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported source language: {self.source_lang}. "
                f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )
        if self.target_lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported target language: {self.target_lang}. "
                f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )

    def create_translator(self) -> "TranslatorBackend":
        """
        設定に基づいて翻訳バックエンドを作成する。

        Returns:
            TranslatorBackend インスタンス

        Raises:
            ValueError: 未知のバックエンドが指定された場合
            ImportError: DeepL バックエンドで aiohttp がない場合
        """
        from index_pdf_translation.translators import GoogleTranslator

        if self.backend == "google":
            return GoogleTranslator()
        elif self.backend == "deepl":
            from index_pdf_translation.translators import get_deepl_translator
            DeepLTranslator = get_deepl_translator()
            return DeepLTranslator(self.api_key, self.api_url)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
```

---

### Phase 4: translate.py の更新

#### 4.1 `src/index_pdf_translation/core/translate.py`

主な変更点：
- `translate_str_data()` を削除（aiohttp 直接使用を廃止）
- `translate_blocks()` でセパレータトークン方式を実装
- `pdf_translate()` を簡素化
- **aiohttp の import を削除**（DeepL 専用モジュールに移動）

```python
# SPDX-License-Identifier: AGPL-3.0-only
"""翻訳オーケストレーション"""

from typing import Any, Optional, TYPE_CHECKING

from index_pdf_translation.config import TranslationConfig
from index_pdf_translation.logger import get_logger
from index_pdf_translation.core.pdf_edit import (
    DocumentBlocks,
    create_viewing_pdf,
    extract_text_coordinates_dict,
    preprocess_write_blocks,
    remove_blocks,
    remove_textbox_for_pdf,
    write_logo_data,
    write_pdf_text,
)

if TYPE_CHECKING:
    from index_pdf_translation.translators import TranslatorBackend

logger = get_logger("translate")

# セパレータトークン（Google翻訳で翻訳されない記号のみのトークン）
# 検証済み: 空行・空白を含む複数テキストで100%の成功率
BLOCK_SEPARATOR = "[[[BR]]]"

# 文字数制限（deep-translator の Google 翻訳は 5,000 文字制限）
MAX_CHUNK_SIZE = 4500  # 余裕を持たせる


def chunk_texts_for_translation(
    texts: list[str],
    separator: str = BLOCK_SEPARATOR,
    max_size: int = MAX_CHUNK_SIZE,
) -> list[list[str]]:
    """
    テキストリストを文字数制限内のチャンクに分割。

    Args:
        texts: 翻訳するテキストのリスト
        separator: セパレータトークン
        max_size: 1チャンクの最大文字数

    Returns:
        チャンクに分割されたテキストリストのリスト

    Note:
        単一ブロックが max_size を超える場合はそのまま単独チャンクとして扱う。
        API 側でエラーになる可能性があるが、学術論文では稀なケースのため
        Phase 1 ではハードエラーとして処理し、ログで警告を出力する。
    """
    chunks: list[list[str]] = []
    current_chunk: list[str] = []
    current_size = 0
    separator_len = len(separator)

    for text in texts:
        # 単一ブロックが制限を超える場合は警告してそのまま追加
        if len(text) > max_size:
            # 現在のチャンクを先に保存
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            # 巨大ブロックを単独チャンクとして追加（APIエラーになる可能性あり）
            chunks.append([text])
            logger.warning(
                f"Single block exceeds MAX_CHUNK_SIZE ({len(text)} > {max_size}). "
                f"May fail at translation API."
            )
            continue

        # セパレータを含めたサイズを計算
        item_size = len(text)
        if current_chunk:
            item_size += separator_len

        # 現在のチャンクに追加すると制限を超える場合
        if current_size + item_size > max_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
            item_size = len(text)  # 新チャンクなのでセパレータ不要

        current_chunk.append(text)
        current_size += item_size

    # 最後のチャンクを追加
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def translate_chunk_with_retry(
    translator: "TranslatorBackend",
    text: str,
    target_lang: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """
    固定遅延リトライ付きで翻訳を実行（Phase 1 実装）。

    Args:
        translator: 翻訳バックエンド
        text: 翻訳するテキスト
        target_lang: 翻訳先言語コード
        max_retries: 最大リトライ回数
        retry_delay: リトライ間隔（秒）- 固定

    Returns:
        翻訳されたテキスト

    Raises:
        TranslationError: リトライ後も失敗した場合
    """
    from index_pdf_translation.translators.base import TranslationError

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await translator.translate(text, target_lang)
        except TranslationError as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    f"Translation failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {retry_delay}s: {e}"
                )
                await asyncio.sleep(retry_delay)  # 固定遅延
            else:
                raise

    raise last_error  # 型チェック用（到達しない）


async def translate_blocks(
    blocks: DocumentBlocks,
    translator: "TranslatorBackend",
    target_lang: str,
) -> DocumentBlocks:
    """
    複数のテキストブロックを一括翻訳する。

    セパレータトークン方式でテキストを連結し、チャンク単位で翻訳する。
    これにより：
    - 文字数制限（5,000文字）に対応
    - API コール数を最小化（レート制限回避）
    - 文脈を維持した高品質な翻訳
    - 空行・空白を含むブロックも正確に保持

    Args:
        blocks: 翻訳するブロック情報のリスト
        translator: 翻訳バックエンド
        target_lang: 翻訳先言語コード

    Returns:
        翻訳後のブロック情報
    """
    # 全テキストを抽出
    texts: list[str] = []
    for page in blocks:
        for block in page:
            texts.append(block["text"])

    if not texts:
        return blocks

    # テキストをチャンクに分割
    chunks = chunk_texts_for_translation(texts, BLOCK_SEPARATOR, MAX_CHUNK_SIZE)
    logger.info(f"Split {len(texts)} blocks into {len(chunks)} chunks for translation")

    # 各チャンクを翻訳
    translated_texts: list[str] = []
    for i, chunk in enumerate(chunks):
        combined_text = BLOCK_SEPARATOR.join(chunk)
        logger.debug(f"Translating chunk {i + 1}/{len(chunks)} ({len(combined_text)} chars)")

        translated_combined = await translate_chunk_with_retry(
            translator, combined_text, target_lang
        )

        # 翻訳結果を分割
        chunk_results = translated_combined.split(BLOCK_SEPARATOR)

        # チャンク内の行数検証
        if len(chunk_results) != len(chunk):
            logger.warning(
                f"Chunk {i + 1} block count mismatch: "
                f"expected {len(chunk)}, got {len(chunk_results)}"
            )

        translated_texts.extend(chunk_results)

    # 全体の行数検証
    if len(translated_texts) != len(texts):
        logger.warning(
            f"Total block count mismatch after translation: "
            f"expected {len(texts)}, got {len(translated_texts)}"
        )

    # 翻訳結果を各ブロックに割り当て
    idx = 0
    for page in blocks:
        for block in page:
            if idx < len(translated_texts):
                block["text"] = translated_texts[idx]
            else:
                block["text"] = ""
            idx += 1

    return blocks


async def preprocess_translation_blocks(
    blocks: DocumentBlocks,
    end_markers: tuple[str, ...] = (".", ":", ";"),
    end_marker_enable: bool = True,
) -> DocumentBlocks:
    """翻訳前のブロック前処理"""
    # 既存実装を維持
    results: DocumentBlocks = []

    text = ""
    coordinates: list[Any] = []
    block_no: list[int] = []
    page_no: list[int] = []
    font_size: list[float] = []

    for page in blocks:
        page_results: list[dict[str, Any]] = []
        temp_block_no = 0

        for block in page:
            text += " " + block["text"]
            page_no.append(block["page_no"])
            coordinates.append(block["coordinates"])
            block_no.append(block["block_no"])
            font_size.append(block["size"])

            should_save = (
                text.endswith(end_markers)
                or block["block_no"] - temp_block_no <= 1
                or not end_marker_enable
            )

            if should_save:
                page_results.append({
                    "page_no": page_no,
                    "block_no": block_no,
                    "coordinates": coordinates,
                    "text": text,
                    "size": font_size,
                })
                # リセット
                text = ""
                coordinates = []
                block_no = []
                page_no = []
                font_size = []

            temp_block_no = block["block_no"]

        results.append(page_results)

    return results


async def pdf_translate(
    pdf_data: bytes,
    *,
    config: TranslationConfig,
    disable_translate: bool = False,
) -> Optional[bytes]:
    """
    PDFを翻訳し、見開きPDF（オリジナル + 翻訳）を生成する。

    Args:
        pdf_data: 入力PDFのバイナリデータ
        config: 翻訳設定
        disable_translate: 翻訳を無効化（テスト用）

    Returns:
        見開きPDFのバイナリデータ、または失敗時はNone

    Examples:
        >>> # Google 翻訳（デフォルト）
        >>> config = TranslationConfig()
        >>> result = await pdf_translate(pdf_data, config=config)

        >>> # DeepL 翻訳
        >>> config = TranslationConfig(backend="deepl", api_key="xxx")
        >>> result = await pdf_translate(pdf_data, config=config)
    """
    # 翻訳バックエンドを作成
    translator = config.create_translator()
    logger.info(f"Using translator: {translator.name}")

    # 1. テキストブロック抽出
    block_info = await extract_text_coordinates_dict(pdf_data)

    # 2. ブロック分類
    if config.debug:
        text_blocks, fig_blocks, remove_info, plot_images = await remove_blocks(
            block_info, 10, lang=config.source_lang, debug=True
        )
    else:
        text_blocks, fig_blocks, _, _ = await remove_blocks(
            block_info, 10, lang=config.source_lang
        )

    # 3. テキスト削除
    removed_textbox_pdf_data = await remove_textbox_for_pdf(pdf_data, text_blocks)
    removed_textbox_pdf_data = await remove_textbox_for_pdf(
        removed_textbox_pdf_data, fig_blocks
    )
    logger.info("1. テキストボックス削除完了")

    # 翻訳前のブロック準備
    preprocess_text_blocks = await preprocess_translation_blocks(
        text_blocks, (".", ":", ";"), True
    )
    preprocess_fig_blocks = await preprocess_translation_blocks(
        fig_blocks, (".", ":", ";"), False
    )
    logger.info("2. ブロック前処理完了")

    # 4. 翻訳実施
    if not disable_translate:
        translate_text_blocks = await translate_blocks(
            preprocess_text_blocks,
            translator,
            config.target_lang,
        )
        translate_fig_blocks = await translate_blocks(
            preprocess_fig_blocks,
            translator,
            config.target_lang,
        )
        logger.info("3. 翻訳完了")

        # 5. PDF書き込みデータ作成
        write_text_blocks = await preprocess_write_blocks(
            translate_text_blocks, config.target_lang
        )
        write_fig_blocks = await preprocess_write_blocks(
            translate_fig_blocks, config.target_lang
        )
        logger.info("4. 書き込みブロック生成完了")

        # PDFの作成
        translated_pdf_data = removed_textbox_pdf_data
        if write_text_blocks:
            translated_pdf_data = await write_pdf_text(
                translated_pdf_data, write_text_blocks, config.target_lang
            )
        if write_fig_blocks:
            translated_pdf_data = await write_pdf_text(
                translated_pdf_data, write_fig_blocks, config.target_lang
            )

        # 6. ロゴ追加（オプション）
        if config.add_logo:
            translated_pdf_data = await write_logo_data(translated_pdf_data)
    else:
        logger.info("翻訳スキップ（disable_translate=True）")
        translated_pdf_data = removed_textbox_pdf_data

    # 7. 見開き結合
    merged_pdf_data = await create_viewing_pdf(pdf_data, translated_pdf_data)
    logger.info("5. 見開きPDF生成完了")

    return merged_pdf_data
```

---

### Phase 5: CLI の更新

#### 5.1 `src/index_pdf_translation/cli.py`

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CLI ツール"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import NoReturn

from index_pdf_translation import pdf_translate
from index_pdf_translation.config import (
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_LANGUAGES,
    TranslationConfig,
)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        prog="translate-pdf",
        description="PDF翻訳ツール - 学術論文PDFを翻訳し見開きPDFを生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s paper.pdf                        # Google翻訳（デフォルト）
  %(prog)s paper.pdf --backend deepl        # DeepL翻訳（高品質）
  %(prog)s paper.pdf -o result.pdf          # 出力ファイル指定
  %(prog)s paper.pdf -s en -t ja            # 英語→日本語

Environment Variables:
  DEEPL_API_KEY    DeepL APIキー (--backend deepl 時に必要)
  DEEPL_API_URL    DeepL API URL (オプション)
""",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="翻訳するPDFファイルのパス",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        help=f"出力ファイルのパス (デフォルト: {DEFAULT_OUTPUT_DIR}translated_<input>.pdf)",
    )

    parser.add_argument(
        "-b", "--backend",
        default="google",
        choices=["google", "deepl"],
        help="翻訳バックエンド (デフォルト: google)",
    )

    parser.add_argument(
        "-s", "--source",
        default="en",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help="翻訳元の言語 (デフォルト: en)",
    )

    parser.add_argument(
        "-t", "--target",
        default="ja",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help="翻訳先の言語 (デフォルト: ja)",
    )

    parser.add_argument(
        "--api-key",
        help="DeepL APIキー (--backend deepl 時に必要)",
    )

    parser.add_argument(
        "--api-url",
        help="DeepL API URL (オプション)",
    )

    parser.add_argument(
        "--no-logo",
        action="store_true",
        help="ロゴウォーターマークを無効化",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモード（ブロック分類の可視化PDFを生成）",
    )

    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    """PDFを翻訳"""
    input_path: Path = args.input

    # 入力ファイルの検証
    if not input_path.exists():
        print(f"エラー: ファイルが見つかりません: {input_path}", file=sys.stderr)
        return 1

    if input_path.suffix.lower() != ".pdf":
        print(f"エラー: PDFファイルではありません: {input_path}", file=sys.stderr)
        return 1

    # API キーの取得（DeepL の場合のみ）
    api_key = ""
    api_url = ""
    if args.backend == "deepl":
        api_key = args.api_key or os.environ.get("DEEPL_API_KEY", "")
        if not api_key:
            print(
                "エラー: DeepL バックエンドには APIキーが必要です。\n"
                "  --api-key オプションまたは環境変数 DEEPL_API_KEY を設定してください。\n"
                "  または --backend google でAPIキー不要の翻訳を使用できます。",
                file=sys.stderr,
            )
            return 1
        api_url = args.api_url or os.environ.get(
            "DEEPL_API_URL", "https://api-free.deepl.com/v2/translate"
        )

    # 出力パスの決定
    if args.output:
        output_path: Path = args.output
    else:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        output_path = output_dir / f"translated_{input_path.name}"

    # 出力ディレクトリの作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 進捗表示
    print(f"入力: {input_path}")
    print(f"出力: {output_path}")
    print(f"バックエンド: {args.backend}")
    print(f"翻訳: {args.source.upper()} → {args.target.upper()}")
    if args.no_logo:
        print("ロゴ: 無効")
    if args.debug:
        print("デバッグモード: 有効")
    print()

    # TranslationConfig を作成
    try:
        config = TranslationConfig(
            backend=args.backend,
            api_key=api_key,
            api_url=api_url,
            source_lang=args.source,
            target_lang=args.target,
            add_logo=not args.no_logo,
            debug=args.debug,
        )
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1

    # PDFの読み込み
    with open(input_path, "rb") as f:
        pdf_data = f.read()

    # 翻訳の実行
    try:
        result_pdf = await pdf_translate(pdf_data, config=config)
    except Exception as e:
        print(f"エラー: 翻訳中にエラーが発生しました: {e}", file=sys.stderr)
        return 1

    if result_pdf is None:
        print("エラー: 翻訳に失敗しました", file=sys.stderr)
        return 1

    # 結果の保存
    with open(output_path, "wb") as f:
        f.write(result_pdf)

    print()
    print(f"完了: {output_path}")
    return 0


def main() -> NoReturn:
    """メインエントリーポイント"""
    args = parse_args()
    exit_code = asyncio.run(run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
```

---

### Phase 6: テスト追加

#### テスト戦略

| テスト種別 | 実行タイミング | 外部API | 目的 |
|-----------|---------------|---------|------|
| **ユニットテスト** | CI（常時） | モック | 高速・安定・カバレッジ |
| **統合テスト** | ローカル/手動 | 実API | 実際の動作確認 |

#### 6.0 既存テストの更新（Breaking Changes 対応）

**削除/更新が必要なテスト（`tests/test_config.py`）**:

| 行番号 | テスト | 対応 |
|--------|--------|------|
| L22-23 | `SUPPORTED_LANGUAGES["en"]["deepl"]` | キーを `spacy` のみに変更 |
| L99-105 | `test_deepl_target_lang_property` | **削除** |
| L107-113 | `test_deepl_source_lang_property` | **削除** |
| L46-52 | `test_config_missing_key_raises` | DeepL バックエンド用に変更 |
| L62 | `api_url` 検証 | 条件付きに変更（DeepL 時のみ） |

**更新後の `tests/test_config.py`（抜粋）**:

```python
class TestSupportedLanguages:
    def test_english_supported(self) -> None:
        assert "en" in SUPPORTED_LANGUAGES
        # deepl キーは削除されているため spacy のみ確認
        assert SUPPORTED_LANGUAGES["en"]["spacy"] == "en_core_web_sm"

    def test_japanese_supported(self) -> None:
        assert "ja" in SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES["ja"]["spacy"] == "ja_core_news_sm"


class TestTranslationConfig:
    def test_config_google_default_no_api_key_required(self) -> None:
        """Google バックエンド（デフォルト）は API キー不要"""
        config = TranslationConfig()  # api_key なしでもエラーにならない
        assert config.backend == "google"

    def test_config_deepl_requires_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DeepL バックエンドは API キー必須"""
        monkeypatch.delenv("DEEPL_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DeepL API key required"):
            TranslationConfig(backend="deepl", api_key="")
```

---

#### 6.1 `tests/test_translators.py` 新規作成

```python
# SPDX-License-Identifier: AGPL-3.0-only
"""翻訳バックエンドのテスト（モック使用）"""

import os

import pytest
from unittest.mock import patch, MagicMock

from index_pdf_translation.translators import GoogleTranslator, TranslationError


class TestGoogleTranslator:
    """Google 翻訳バックエンドのテスト"""

    def test_name(self):
        """バックエンド名の確認"""
        translator = GoogleTranslator()
        assert translator.name == "google"

    @pytest.mark.asyncio
    async def test_translate_empty_string(self):
        """空文字列の翻訳（APIコールなし）"""
        translator = GoogleTranslator()
        result = await translator.translate("", "ja")
        assert result == ""

    @pytest.mark.asyncio
    async def test_translate_whitespace_only(self):
        """空白のみの文字列（APIコールなし）"""
        translator = GoogleTranslator()
        result = await translator.translate("   ", "ja")
        assert result == "   "

    @pytest.mark.asyncio
    async def test_translate_simple_mocked(self):
        """基本的な翻訳テスト（モック）"""
        translator = GoogleTranslator()

        with patch("deep_translator.GoogleTranslator") as mock_dt:
            mock_instance = MagicMock()
            mock_instance.translate.return_value = "こんにちは"
            mock_dt.return_value = mock_instance

            result = await translator.translate("Hello", "ja")

            assert result == "こんにちは"
            mock_dt.assert_called_once_with(source="auto", target="ja")
            mock_instance.translate.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_translate_with_separator_mocked(self):
        """セパレータを含むテキストの翻訳（モック）"""
        translator = GoogleTranslator()

        with patch("deep_translator.GoogleTranslator") as mock_dt:
            mock_instance = MagicMock()
            # セパレータが保持されることをシミュレート
            mock_instance.translate.return_value = "こんにちは[[[BR]]]世界[[[BR]]]おはよう"
            mock_dt.return_value = mock_instance

            result = await translator.translate("Hello[[[BR]]]World[[[BR]]]Good morning", "ja")

            assert "[[[BR]]]" in result
            parts = result.split("[[[BR]]]")
            assert len(parts) == 3

    @pytest.mark.asyncio
    async def test_translate_error_handling(self):
        """エラーハンドリングのテスト"""
        translator = GoogleTranslator()

        with patch("deep_translator.GoogleTranslator") as mock_dt:
            mock_instance = MagicMock()
            mock_instance.translate.side_effect = Exception("API Error")
            mock_dt.return_value = mock_instance

            with pytest.raises(TranslationError, match="Google Translate error"):
                await translator.translate("Hello", "ja")


class TestDeepLTranslator:
    """DeepL 翻訳バックエンドのテスト（APIキー必須のため限定的）"""

    def test_requires_api_key(self):
        """APIキー必須の確認"""
        from index_pdf_translation.translators import get_deepl_translator
        DeepLTranslator = get_deepl_translator()

        with pytest.raises(ValueError, match="API key is required"):
            DeepLTranslator(api_key="")

    def test_name(self):
        """バックエンド名の確認"""
        from index_pdf_translation.translators import get_deepl_translator
        DeepLTranslator = get_deepl_translator()

        translator = DeepLTranslator(api_key="dummy-key")
        assert translator.name == "deepl"


# 統合テスト（実API使用、CI ではスキップ）
@pytest.mark.integration
@pytest.mark.skipif(
    "CI" in os.environ,
    reason="Skip integration tests in CI"
)
class TestGoogleTranslatorIntegration:
    """Google 翻訳の統合テスト（実API使用）"""

    @pytest.mark.asyncio
    async def test_real_translation(self):
        """実際のGoogle翻訳API呼び出し"""
        translator = GoogleTranslator()
        result = await translator.translate("Hello", "ja")
        assert result
        assert result != "Hello"

    @pytest.mark.asyncio
    async def test_separator_preserved(self):
        """セパレータが実際に保持されるか確認"""
        translator = GoogleTranslator()
        text = "Hello[[[BR]]]World[[[BR]]]Good morning"
        result = await translator.translate(text, "ja")

        parts = result.split("[[[BR]]]")
        assert len(parts) == 3, f"Expected 3 parts, got {len(parts)}: {result}"
```

#### 6.2 `tests/test_config.py` 更新

```python
# 追加テストケース

def test_config_default_backend_is_google():
    """デフォルトバックエンドが Google であること"""
    config = TranslationConfig()
    assert config.backend == "google"


def test_config_google_backend_no_api_key_required():
    """Google バックエンドは API キー不要"""
    config = TranslationConfig(backend="google")
    assert config.backend == "google"
    # api_key が空でもエラーにならない


def test_config_deepl_backend_requires_api_key():
    """DeepL バックエンドは API キー必須"""
    with pytest.raises(ValueError, match="DeepL API key required"):
        TranslationConfig(backend="deepl", api_key="")


def test_config_deepl_backend_with_api_key():
    """DeepL バックエンドに API キーを渡す"""
    config = TranslationConfig(backend="deepl", api_key="test-key")
    assert config.backend == "deepl"
    assert config.api_key == "test-key"


def test_config_create_translator_google():
    """create_translator() で Google バックエンドを作成"""
    config = TranslationConfig(backend="google")
    translator = config.create_translator()
    assert translator.name == "google"


def test_config_create_translator_deepl():
    """create_translator() で DeepL バックエンドを作成"""
    config = TranslationConfig(backend="deepl", api_key="test-key")
    translator = config.create_translator()
    assert translator.name == "deepl"
```

---

### Phase 7: ドキュメント更新

#### 7.0 `CHANGELOG.md` 新規作成

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.0.0] - YYYY-MM-DD

### ⚠️ Breaking Changes

- **Default translator changed**: DeepL → Google Translate
  - API key is no longer required for basic usage
  - Use `--backend deepl` or `backend="deepl"` for DeepL
- **`pdf_translate()` signature changed**: Individual parameters replaced with `config` parameter
  - Before: `pdf_translate(pdf_data, api_key="xxx", target_lang="ja")`
  - After: `pdf_translate(pdf_data, config=TranslationConfig(...))`
- **`TranslationConfig` changes**:
  - `api_key` only required when `backend="deepl"`
  - Removed `deepl_target_lang` and `deepl_source_lang` properties
- **`SUPPORTED_LANGUAGES` structure changed**:
  - Removed `deepl` key (now only contains `spacy`)
- **`aiohttp` dependency moved to optional**:
  - Install with `pip install index-pdf-translation[deepl]` for DeepL support

### Added

- Google Translate backend (default, no API key required)
- `--backend` CLI option to select translator (`google` or `deepl`)
- Separator token method for reliable block translation
- Character limit chunking for large documents
- Retry mechanism for translation errors

### Changed

- Translation backend abstraction using Strategy pattern
- Improved error messages with migration guidance

### Migration Guide

```python
# Before (v2.x)
from index_pdf_translation import pdf_translate
result = await pdf_translate(
    pdf_data,
    api_key="xxx",
    target_lang="ja"
)

# After (v3.0.0) - Google Translate (default)
from index_pdf_translation import pdf_translate, TranslationConfig
config = TranslationConfig()
result = await pdf_translate(pdf_data, config=config)

# After (v3.0.0) - DeepL
config = TranslationConfig(backend="deepl", api_key="xxx")
result = await pdf_translate(pdf_data, config=config)
```

## [2.0.0] - Previous Release

- Initial release with DeepL-only support
```

---

#### 7.1 `readme.md` 更新

```markdown
## Quick Start

```bash
# インストール
uv sync
uv run python -m spacy download en_core_web_sm

# 翻訳実行（Google翻訳、APIキー不要）
uv run translate-pdf paper.pdf
```

## 翻訳バックエンド

### Google 翻訳（デフォルト）

APIキー不要で即座に使用可能：

```bash
translate-pdf paper.pdf
translate-pdf paper.pdf --backend google  # 明示的に指定
```

### DeepL（高品質）

高品質な翻訳が必要な場合：

```bash
export DEEPL_API_KEY="your-api-key"
translate-pdf paper.pdf --backend deepl

# または
translate-pdf paper.pdf --backend deepl --api-key "your-api-key"
```

DeepL を使用するには追加の依存関係が必要：

```bash
uv pip install index-pdf-translation[deepl]
```

## 環境変数の設定（オプション）

DeepL や OpenAI（将来対応）を使用する場合、環境変数で API キーを設定できます。

### 方法 1: .env ファイルを使用

```bash
# .env.example をコピー
cp .env.example .env

# .env を編集して API キーを設定
# DEEPL_API_KEY=your-api-key-here

# シェルで読み込んで実行
source .env && translate-pdf paper.pdf --backend deepl
```

### 方法 2: export コマンド

```bash
export DEEPL_API_KEY="your-api-key"
translate-pdf paper.pdf --backend deepl
```

### 方法 3: CLI オプション

```bash
translate-pdf paper.pdf --backend deepl --api-key "your-api-key"
```

> **Note**: Google 翻訳（デフォルト）は API キー不要のため、環境変数の設定なしで使用できます。
```

#### 7.2 `CLAUDE.md` 更新

CLI オプションセクションを更新：

```markdown
### CLI Options
- `-o, --output`: Output file path
- `-b, --backend`: Translation backend (google/deepl, default: google)
- `-s, --source`: Source language (en/ja, default: en)
- `-t, --target`: Target language (en/ja, default: ja)
- `--api-key`: DeepL API key (required for --backend deepl)
- `--api-url`: DeepL API URL (for Pro users)
- `--no-logo`: Disable logo watermark
- `--debug`: Enable debug mode
```

#### 7.3 `.env.example` 新規作成

```bash
# Index PDF Translation - Environment Variables
# Copy this file to .env and fill in your API keys.
# All variables are OPTIONAL - Google Translate works without any API key.

# ============================================================
# DeepL API (for --backend deepl)
# ============================================================
# Get your API key at: https://www.deepl.com/pro-api
# DEEPL_API_KEY=your-deepl-api-key-here

# For DeepL Pro users (default is Free API):
# DEEPL_API_URL=https://api.deepl.com/v2/translate

# ============================================================
# OpenAI API (for --backend openai, future feature)
# ============================================================
# Get your API key at: https://platform.openai.com/api-keys
# OPENAI_API_KEY=your-openai-api-key-here
```

#### 7.4 `.gitignore` 更新

以下を追加:

```gitignore
# Environment variables (API keys)
.env
.env.local
.env.*.local
```

---

## Breaking Changes

この実装は以下の Breaking Change を含む：

| 項目 | 変更内容 |
|------|----------|
| デフォルトバックエンド | `deepl` → `google` |
| `pdf_translate()` 引数 | 個別パラメータ廃止、`config` パラメータ必須 |
| `TranslationConfig` | `api_key` は `backend="deepl"` 時のみ必須 |
| `aiohttp` 依存 | オプショナル（`[deepl]` extra）に移動 |
| `deepl_target_lang` プロパティ | 削除 |
| `SUPPORTED_LANGUAGES` | `deepl` キー削除（`spacy` のみ） |
| Translator `LANG_MAP` | 廃止（各バックエンドで直接変換） |

### マイグレーションガイド

```python
# Before (v2.x)
from index_pdf_translation import pdf_translate
result = await pdf_translate(
    pdf_data,
    api_key="xxx",
    target_lang="ja"
)

# After (v3.x)
from index_pdf_translation import pdf_translate, TranslationConfig

# Google 翻訳（デフォルト、APIキー不要）
config = TranslationConfig()
result = await pdf_translate(pdf_data, config=config)

# DeepL 翻訳
config = TranslationConfig(backend="deepl", api_key="xxx")
result = await pdf_translate(pdf_data, config=config)
```

---

## 完了条件

### Phase 1: 依存関係
- [ ] `pyproject.toml` 更新（deep-translator 追加、aiohttp をオプショナルに）

### Phase 2: 翻訳バックエンド
- [ ] `translators/` モジュール作成
  - [ ] `__init__.py` - エクスポート定義
  - [ ] `base.py` - プロトコル定義（translate のみ）
  - [ ] `google.py` - Google 翻訳実装
  - [ ] `deepl.py` - DeepL 翻訳実装（aiohttp 遅延 import）

### Phase 3: Config
- [ ] `config.py` 更新（backend オプション、デフォルト google）

### Phase 4: translate.py
- [ ] `core/translate.py` 更新
  - [ ] aiohttp import 削除
  - [ ] セパレータトークン方式（`[[[BR]]]`）実装
  - [ ] チャンキング関数（`chunk_texts_for_translation`）実装
  - [ ] リトライ関数（`translate_chunk_with_retry`）実装
  - [ ] `translate_str_data()` 削除
- [ ] DeepL セパレータ互換性検証（実装時）

### Phase 5: CLI
- [ ] `cli.py` 更新（--backend オプション）

### Phase 6: テスト
- [ ] 既存テスト更新（`test_config.py`）
  - [ ] `SUPPORTED_LANGUAGES` テスト修正（deepl キー削除対応）
  - [ ] `deepl_target_lang` / `deepl_source_lang` テスト削除
  - [ ] API キー必須テストを DeepL バックエンド用に変更
- [ ] `test_translators.py` 新規作成（モック使用）
- [ ] 統合テスト（`@pytest.mark.integration`）
- [ ] チャンキング関数のユニットテスト追加

### Phase 7: ドキュメント
- [ ] `CHANGELOG.md` 新規作成（Breaking Changes）
- [ ] `readme.md` 更新（環境変数設定セクション追加）
- [ ] `CLAUDE.md` 更新
- [ ] `.env.example` 新規作成（APIキーテンプレート）
- [ ] `.gitignore` 更新（`.env` 追加）

### 最終確認
- [ ] `__init__.py` 更新（エクスポート追加）
- [ ] CI 通過確認（API キー不要で通過）
- [ ] ローカルで E2E テスト（実際の PDF 翻訳）

---

## ファイル変更一覧

### 新規作成
| ファイル | 説明 |
|---------|------|
| `src/index_pdf_translation/translators/__init__.py` | エクスポート定義 |
| `src/index_pdf_translation/translators/base.py` | TranslatorBackend プロトコル |
| `src/index_pdf_translation/translators/google.py` | Google 翻訳バックエンド |
| `src/index_pdf_translation/translators/deepl.py` | DeepL 翻訳バックエンド |
| `tests/test_translators.py` | 翻訳バックエンドのテスト |
| `CHANGELOG.md` | 変更履歴（Breaking Changes 記載） |
| `.env.example` | 環境変数テンプレート（APIキー設定用） |

### 更新
| ファイル | 主な変更 |
|---------|---------|
| `pyproject.toml` | deep-translator 追加、aiohttp をオプショナルに |
| `src/index_pdf_translation/__init__.py` | エクスポート追加 |
| `src/index_pdf_translation/config.py` | backend オプション、デフォルト google |
| `src/index_pdf_translation/core/translate.py` | チャンキング、リトライ、セパレータ方式 |
| `src/index_pdf_translation/cli.py` | --backend オプション |
| `tests/test_config.py` | Breaking Changes 対応 |
| `readme.md` | Quick Start、翻訳バックエンド、環境変数設定 |
| `CLAUDE.md` | CLI オプション更新 |
| `.gitignore` | `.env` 追加（セキュリティ） |
