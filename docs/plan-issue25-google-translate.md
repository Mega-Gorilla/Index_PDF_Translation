# Issue #25: deep-translator を使用した Google 翻訳対応

## 概要

現在 DeepL API のみに対応している翻訳機能を、`deep-translator` ライブラリを使用して Google 翻訳にも対応させる。これにより、APIキーなしで翻訳テスト・デバッグが可能になる。

## 現状分析

### 現在のアーキテクチャ

```
cli.py
  └── TranslationConfig (config.py)
        └── pdf_translate (core/translate.py)
              └── translate_str_data() - DeepL API直接呼び出し
              └── translate_blocks()
```

### 問題点

1. **翻訳ロジックがDeepL APIにハードコード**
   - `translate_str_data()` が aiohttp で DeepL API を直接呼び出し
   - API URL、パラメータ形式が DeepL 固有

2. **TranslationConfig が DeepL 専用**
   - `api_key` が必須バリデーション（空だとエラー）
   - `deepl_target_lang` プロパティが DeepL 固有

3. **テストが困難**
   - 実際の翻訳テストには DeepL APIキーが必要
   - CI/CD での統合テストが制限される

## 設計方針

### Strategy パターンの採用

翻訳バックエンドを抽象化し、実行時に切り替え可能にする。

```
TranslatorBackend (Protocol/ABC)
    ├── DeepLTranslator     - 既存の DeepL API 実装
    ├── GoogleTranslator    - deep-translator 経由
    └── MockTranslator      - テスト用
```

### 言語コード統一

各バックエンドで言語コード形式が異なるため、統一インターフェースを提供：

| 内部コード | DeepL | Google (deep-translator) |
|-----------|-------|--------------------------|
| `en`      | `EN`  | `english` または `en`     |
| `ja`      | `JA`  | `japanese` または `ja`    |

---

## 実装フェーズ

### Phase 1: 翻訳バックエンド抽象化

#### 1.1 `src/index_pdf_translation/translators/__init__.py` 作成

```python
from .base import TranslatorBackend
from .deepl import DeepLTranslator
from .google import GoogleTranslator

__all__ = ["TranslatorBackend", "DeepLTranslator", "GoogleTranslator"]
```

#### 1.2 `src/index_pdf_translation/translators/base.py` 作成

```python
from abc import ABC, abstractmethod
from typing import Protocol

class TranslatorBackend(Protocol):
    """翻訳バックエンドのプロトコル"""

    @abstractmethod
    async def translate(self, text: str, target_lang: str, source_lang: str = "auto") -> str:
        """
        テキストを翻訳する。

        Args:
            text: 翻訳するテキスト
            target_lang: 翻訳先言語（内部コード: "en", "ja"）
            source_lang: 翻訳元言語（内部コード、"auto"で自動検出）

        Returns:
            翻訳されたテキスト

        Raises:
            TranslationError: 翻訳に失敗した場合
        """
        ...

    @abstractmethod
    async def translate_batch(self, texts: list[str], target_lang: str, source_lang: str = "auto") -> list[str]:
        """複数テキストを一括翻訳"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """バックエンド名（"deepl", "google"）"""
        ...

    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """APIキーが必要かどうか"""
        ...


class TranslationError(Exception):
    """翻訳エラー"""
    pass
```

#### 1.3 `src/index_pdf_translation/translators/deepl.py` 作成

既存の `translate_str_data()` ロジックを移植：

```python
import aiohttp
from .base import TranslatorBackend, TranslationError

class DeepLTranslator(TranslatorBackend):
    """DeepL API を使用した翻訳"""

    # 内部コード -> DeepL コード
    LANG_MAP = {"en": "EN", "ja": "JA"}

    def __init__(self, api_key: str, api_url: str = "https://api-free.deepl.com/v2/translate"):
        if not api_key:
            raise ValueError("DeepL API key is required")
        self._api_key = api_key
        self._api_url = api_url

    @property
    def name(self) -> str:
        return "deepl"

    @property
    def requires_api_key(self) -> bool:
        return True

    async def translate(self, text: str, target_lang: str, source_lang: str = "auto") -> str:
        # 既存ロジック移植
        ...

    async def translate_batch(self, texts: list[str], target_lang: str, source_lang: str = "auto") -> list[str]:
        # 改行で連結 -> 翻訳 -> 分割（既存方式）
        ...
```

#### 1.4 `src/index_pdf_translation/translators/google.py` 作成

```python
import asyncio
from deep_translator import GoogleTranslator as DTGoogleTranslator
from .base import TranslatorBackend, TranslationError

class GoogleTranslator(TranslatorBackend):
    """Google 翻訳を使用（APIキー不要）"""

    # 内部コード -> deep-translator コード
    LANG_MAP = {"en": "en", "ja": "ja"}

    def __init__(self):
        pass  # APIキー不要

    @property
    def name(self) -> str:
        return "google"

    @property
    def requires_api_key(self) -> bool:
        return False

    async def translate(self, text: str, target_lang: str, source_lang: str = "auto") -> str:
        # deep-translator は同期API、asyncio.to_thread() でラップ
        def _translate():
            translator = DTGoogleTranslator(
                source=self.LANG_MAP.get(source_lang, "auto"),
                target=self.LANG_MAP[target_lang]
            )
            return translator.translate(text)

        return await asyncio.to_thread(_translate)

    async def translate_batch(self, texts: list[str], target_lang: str, source_lang: str = "auto") -> list[str]:
        def _translate_batch():
            translator = DTGoogleTranslator(
                source=self.LANG_MAP.get(source_lang, "auto"),
                target=self.LANG_MAP[target_lang]
            )
            return translator.translate_batch(texts)

        return await asyncio.to_thread(_translate_batch)
```

---

### Phase 2: Config と translate.py の更新

#### 2.1 `config.py` 更新

```python
from typing import Literal

TranslatorBackendType = Literal["deepl", "google"]

@dataclass
class TranslationConfig:
    backend: TranslatorBackendType = "deepl"
    api_key: str = field(default_factory=lambda: os.environ.get("DEEPL_API_KEY", ""))
    api_url: str = field(default_factory=lambda: os.environ.get("DEEPL_API_URL", "https://api-free.deepl.com/v2/translate"))
    source_lang: str = "en"
    target_lang: str = "ja"
    add_logo: bool = True
    debug: bool = False

    def __post_init__(self) -> None:
        # backend が "deepl" の場合のみ api_key を必須にする
        if self.backend == "deepl" and not self.api_key:
            raise ValueError(
                "DeepL API key required when using 'deepl' backend. "
                "Set DEEPL_API_KEY or use --backend google"
            )
        # 言語バリデーション（既存通り）
        ...

    def create_translator(self) -> "TranslatorBackend":
        """設定に基づいて翻訳バックエンドを作成"""
        from index_pdf_translation.translators import DeepLTranslator, GoogleTranslator

        if self.backend == "deepl":
            return DeepLTranslator(self.api_key, self.api_url)
        elif self.backend == "google":
            return GoogleTranslator()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
```

#### 2.2 `core/translate.py` 更新

`translate_str_data()` と `translate_blocks()` をリファクタリング：

```python
async def translate_blocks(
    blocks: DocumentBlocks,
    translator: TranslatorBackend,  # 変更: key, api_url → translator
    target_lang: str,
) -> DocumentBlocks:
    """複数のテキストブロックを一括翻訳"""
    texts = [block["text"] for page in blocks for block in page]

    translated_texts = await translator.translate_batch(texts, target_lang)

    # 翻訳結果を各ブロックに割り当て
    idx = 0
    for page in blocks:
        for block in page:
            block["text"] = translated_texts[idx] if idx < len(translated_texts) else ""
            idx += 1

    return blocks


async def pdf_translate(
    pdf_data: bytes,
    *,
    config: Optional[TranslationConfig] = None,
    # 後方互換性のため既存パラメータも維持
    api_key: Optional[str] = None,
    ...
) -> Optional[bytes]:
    # config から translator を作成
    if config is not None:
        translator = config.create_translator()
    else:
        # 後方互換: 個別パラメータから DeepLTranslator を作成
        from index_pdf_translation.translators import DeepLTranslator
        translator = DeepLTranslator(api_key, api_url)

    # 翻訳時に translator を使用
    translate_text_blocks = await translate_blocks(
        preprocess_text_blocks,
        translator,  # 変更
        effective_target_lang,
    )
```

---

### Phase 3: CLI 更新

#### 3.1 `cli.py` 更新

```python
parser.add_argument(
    "--backend",
    default="deepl",
    choices=["deepl", "google"],
    help="翻訳バックエンド (デフォルト: deepl)",
)

# run() 内
api_key = args.api_key or os.environ.get("DEEPL_API_KEY", "")

# backend が google なら api_key 不要
if args.backend == "deepl" and not api_key:
    print("エラー: DeepL APIキーが必要です。--backend google でAPIキー不要の翻訳も可能です。", ...)
    return 1

config = TranslationConfig(
    backend=args.backend,
    api_key=api_key,
    ...
)
```

---

### Phase 4: テスト追加

#### 4.1 `tests/test_translators.py` 作成

```python
import pytest
from index_pdf_translation.translators import DeepLTranslator, GoogleTranslator, TranslationError

class TestGoogleTranslator:
    """Google翻訳のテスト（APIキー不要）"""

    @pytest.mark.asyncio
    async def test_translate_simple(self):
        translator = GoogleTranslator()
        result = await translator.translate("Hello", target_lang="ja")
        assert result  # 何らかの翻訳結果が返る
        assert result != "Hello"  # 原文とは異なる

    @pytest.mark.asyncio
    async def test_translate_batch(self):
        translator = GoogleTranslator()
        texts = ["Hello", "World"]
        results = await translator.translate_batch(texts, target_lang="ja")
        assert len(results) == 2

    def test_properties(self):
        translator = GoogleTranslator()
        assert translator.name == "google"
        assert translator.requires_api_key is False


class TestDeepLTranslator:
    """DeepL翻訳のテスト（APIキー必須のため一部スキップ）"""

    def test_requires_api_key(self):
        with pytest.raises(ValueError):
            DeepLTranslator(api_key="")

    def test_properties(self):
        translator = DeepLTranslator(api_key="dummy")
        assert translator.name == "deepl"
        assert translator.requires_api_key is True
```

#### 4.2 `tests/test_config.py` 更新

```python
def test_config_google_backend_no_api_key():
    """Google バックエンドなら API キー不要"""
    config = TranslationConfig(backend="google")
    assert config.backend == "google"
    # エラーにならない


def test_config_deepl_backend_requires_api_key():
    """DeepL バックエンドは API キー必須"""
    with pytest.raises(ValueError):
        TranslationConfig(backend="deepl", api_key="")


def test_config_create_translator():
    """create_translator() のテスト"""
    config = TranslationConfig(backend="google")
    translator = config.create_translator()
    assert translator.name == "google"
```

#### 4.3 統合テスト（オプション）

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_translate_with_google(sample_llama_pdf):
    """Google翻訳での実際のPDF翻訳テスト"""
    config = TranslationConfig(backend="google", source_lang="en", target_lang="ja")

    with open(sample_llama_pdf, "rb") as f:
        pdf_data = f.read()

    result = await pdf_translate(pdf_data, config=config)
    assert result is not None
    assert len(result) > 0
```

---

### Phase 5: ドキュメント更新

#### 5.1 `readme.md` 更新

```markdown
### 翻訳バックエンド

#### DeepL（デフォルト、高品質）
```bash
export DEEPL_API_KEY="your-api-key"
translate-pdf paper.pdf
```

#### Google 翻訳（APIキー不要）
```bash
translate-pdf paper.pdf --backend google
```
```

#### 5.2 `CLAUDE.md` 更新

CLI オプションに `--backend` を追加。

---

## 依存関係

### pyproject.toml 更新

```toml
dependencies = [
    "PyMuPDF>=1.24.0",
    "spacy>=3.7.0",
    "aiohttp>=3.9.0",
    "numpy>=1.26.0",
    "matplotlib>=3.8.0",
    "deep-translator>=1.11.0",  # 追加
]
```

---

## リスク・考慮事項

### 1. Google 翻訳の品質

- DeepL より品質が劣る可能性
- 対策: デフォルトは DeepL を維持、Google は開発・テスト用途を推奨

### 2. Rate Limiting

- deep-translator の Google 翻訳はレート制限あり（具体的な制限は未公開）
- 対策: 大量テキストの場合は適度な待機を入れる検討

### 3. 後方互換性

- 既存の `api_key`, `api_url` パラメータは維持
- `backend` パラメータはデフォルト `"deepl"` で既存動作を保持

### 4. エラーハンドリング

- deep-translator のエラーを `TranslationError` に統一
- ネットワークエラー、言語未対応などの適切な処理

---

## 完了条件（Issue #25 より）

- [x] 設計完了
- [ ] deep-translator を依存関係に追加
- [ ] TranslatorBackend 抽象クラス作成
- [ ] DeepLTranslator 実装
- [ ] GoogleTranslator 実装
- [ ] TranslationConfig に backend オプション追加
- [ ] CLI に --backend オプション追加
- [ ] テスト追加（Google翻訳を使用）
- [ ] README 更新

---

## ファイル変更一覧

### 新規作成
- `src/index_pdf_translation/translators/__init__.py`
- `src/index_pdf_translation/translators/base.py`
- `src/index_pdf_translation/translators/deepl.py`
- `src/index_pdf_translation/translators/google.py`
- `tests/test_translators.py`

### 更新
- `pyproject.toml` - deep-translator 依存追加
- `src/index_pdf_translation/config.py` - backend オプション追加
- `src/index_pdf_translation/core/translate.py` - translator 抽象化
- `src/index_pdf_translation/cli.py` - --backend オプション追加
- `src/index_pdf_translation/__init__.py` - エクスポート追加
- `tests/test_config.py` - backend テスト追加
- `readme.md` - 使用方法更新
- `CLAUDE.md` - CLI オプション更新
