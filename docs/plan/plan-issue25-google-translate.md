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

#### システムプロンプト（案）

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
    """

    DEFAULT_MODEL = "gpt-4.1-nano"

    SYSTEM_PROMPT_TEMPLATE = """You are a professional translator specializing in academic papers.
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
    ):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._source_lang = source_lang

    @property
    def name(self) -> str:
        return "openai"

    async def translate(self, text: str, target_lang: str) -> str:
        """単一テキストを翻訳（改行区切りで複数テキストとして処理）"""
        if not text.strip():
            return text

        # 改行で分割して配列として送信
        texts = text.split("\n")
        translated_texts = await self.translate_texts(texts, target_lang)
        return "\n".join(translated_texts)

    async def translate_texts(
        self,
        texts: list[str],
        target_lang: str
    ) -> list[str]:
        """複数テキストを Structured Outputs で翻訳"""
        if not texts:
            return []

        # 空文字列のインデックスを記録
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return texts

        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            source_lang=self.LANG_NAMES.get(self._source_lang, self._source_lang),
            target_lang=self.LANG_NAMES.get(target_lang, target_lang),
        )

        try:
            response = await self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
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
    # ...

    def create_translator(self) -> "TranslatorBackend":
        if self.backend == "openai":
            from index_pdf_translation.translators import get_openai_translator
            OpenAITranslator = get_openai_translator()
            return OpenAITranslator(
                api_key=self.openai_api_key,
                model=self.openai_model,
                source_lang=self.source_lang,
            )
        # ...
```

#### CLI 更新

```bash
# 使用例
translate-pdf paper.pdf --backend openai --openai-model gpt-4.1-nano
translate-pdf paper.pdf --backend openai --openai-model gpt-4.1  # 高品質
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
- `translate_str_data()` を削除
- `translate_blocks()` で改行連結ロジックを実装
- `pdf_translate()` を簡素化

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


async def translate_blocks(
    blocks: DocumentBlocks,
    translator: "TranslatorBackend",
    target_lang: str,
) -> DocumentBlocks:
    """
    複数のテキストブロックを一括翻訳する。

    全テキストを改行で連結し、1回の API コールで翻訳する。
    これにより：
    - API コール数を最小化（レート制限回避）
    - 文脈を維持した高品質な翻訳

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

    # 改行で連結して一括翻訳（1回の API コール）
    combined_text = "\n".join(texts)
    translated_combined = await translator.translate(combined_text, target_lang)

    # 翻訳結果を分割して各ブロックに割り当て
    translated_lines = translated_combined.split("\n")

    idx = 0
    for page in blocks:
        for block in page:
            if idx < len(translated_lines):
                block["text"] = translated_lines[idx]
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

#### 6.1 `tests/test_translators.py` 新規作成

```python
# SPDX-License-Identifier: AGPL-3.0-only
"""翻訳バックエンドのテスト"""

import pytest
from index_pdf_translation.translators import GoogleTranslator, TranslationError


class TestGoogleTranslator:
    """Google 翻訳バックエンドのテスト"""

    def test_name(self):
        """バックエンド名の確認"""
        translator = GoogleTranslator()
        assert translator.name == "google"

    @pytest.mark.asyncio
    async def test_translate_simple(self):
        """基本的な翻訳テスト"""
        translator = GoogleTranslator()
        result = await translator.translate("Hello", "ja")
        assert result
        assert result != "Hello"

    @pytest.mark.asyncio
    async def test_translate_empty_string(self):
        """空文字列の翻訳"""
        translator = GoogleTranslator()
        result = await translator.translate("", "ja")
        assert result == ""

    @pytest.mark.asyncio
    async def test_translate_with_newlines(self):
        """改行を含むテキストの翻訳（一括翻訳のテスト）"""
        translator = GoogleTranslator()
        text = "Hello\nWorld\nGood morning"
        result = await translator.translate(text, "ja")
        # 改行が保持されていることを確認
        assert "\n" in result
        lines = result.split("\n")
        assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_translate_whitespace_only(self):
        """空白のみの文字列"""
        translator = GoogleTranslator()
        result = await translator.translate("   ", "ja")
        assert result == "   "


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

- [ ] `pyproject.toml` 更新（deep-translator 追加、aiohttp をオプショナルに）
- [ ] `translators/` モジュール作成
  - [ ] `base.py` - プロトコル定義（translate のみ）
  - [ ] `google.py` - Google 翻訳実装
  - [ ] `deepl.py` - DeepL 翻訳実装
- [ ] `config.py` 更新（backend オプション、デフォルト google）
- [ ] `core/translate.py` 更新（改行連結ロジック実装）
- [ ] `cli.py` 更新（--backend オプション）
- [ ] `__init__.py` 更新
- [ ] テスト追加
  - [ ] `test_translators.py`
  - [ ] `test_config.py` 更新
- [ ] ドキュメント更新
  - [ ] `readme.md`
  - [ ] `CLAUDE.md`
- [ ] CI 通過確認

---

## ファイル変更一覧

### 新規作成
- `src/index_pdf_translation/translators/__init__.py`
- `src/index_pdf_translation/translators/base.py`
- `src/index_pdf_translation/translators/google.py`
- `src/index_pdf_translation/translators/deepl.py`
- `tests/test_translators.py`

### 更新
- `pyproject.toml`
- `src/index_pdf_translation/__init__.py`
- `src/index_pdf_translation/config.py`
- `src/index_pdf_translation/core/translate.py`
- `src/index_pdf_translation/cli.py`
- `tests/test_config.py`
- `readme.md`
- `CLAUDE.md`
