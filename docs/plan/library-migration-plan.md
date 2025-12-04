# ライブラリ構造移行計画

> Issue #17: feat: ライブラリ構造への移行（パッケージ化）

## 1. 概要

### 目的

- `pip install` / `uv add` でインストール可能なPythonパッケージ化
- `from index_pdf_translation import pdf_translate` で利用可能に
- CLIツールとしての使い勝手は維持

### 成功基準

1. `uv pip install .` でローカルインストール可能
2. `from index_pdf_translation import pdf_translate` が動作
3. `translate-pdf paper.pdf` (CLI) が動作

> **Note**: 後方互換性は考慮しない。`translate_pdf.py` は Phase 5 で削除し、`translate-pdf` コマンドに完全移行する。

---

## 2. 現状分析

### 2.1 現在のファイル構造

```
Index_PDF_Translation/
├── translate_pdf.py      # CLI (171行)
├── config.py             # 設定 (29行) - ハードコード
├── modules/
│   ├── logger.py         # ロギング (69行)
│   ├── pdf_edit.py       # PDF処理 (798行) - 最大
│   ├── spacy_api.py      # NLP (86行)
│   └── translate.py      # 翻訳 (278行)
├── fonts/                # フォントファイル
├── data/                 # ロゴ画像
└── pyproject.toml        # package = false
```

**合計: 1,431行 (Python)**

### 2.2 依存関係マップ

```
translate_pdf.py
├── config.py (DEEPL_API_KEY, DEEPL_API_URL, OUTPUT_DIR, SUPPORTED_LANGUAGES)
└── modules/translate.py (pdf_translate)
    ├── modules/logger.py (get_logger)
    └── modules/pdf_edit.py (複数関数)
        ├── modules/logger.py (get_logger)
        └── modules/spacy_api.py (tokenize_text)
            ├── config.py (SUPPORTED_LANGUAGES)
            └── modules/logger.py (get_logger)
```

### 2.3 ハードコードされたパス（要修正）

| ファイル | 変数 | パス |
|----------|------|------|
| `pdf_edit.py:36` | `FONT_PATH_EN` | `fonts/LiberationSerif-Regular.ttf` |
| `pdf_edit.py:37` | `FONT_PATH_JA` | `fonts/ipam.ttf` |
| `pdf_edit.py:41` | `LOGO_PATH` | `./data/indqx_qr.png` |

### 2.4 現在の pyproject.toml

```toml
[tool.uv]
package = false  # ← これを削除してパッケージ化
```

---

## 3. 目標構造

```
Index_PDF_Translation/
├── src/
│   └── index_pdf_translation/
│       ├── __init__.py           # 公開API
│       ├── _version.py           # バージョン管理
│       ├── core/
│       │   ├── __init__.py
│       │   ├── translate.py      # 翻訳ロジック
│       │   └── pdf_edit.py       # PDF処理
│       ├── nlp/
│       │   ├── __init__.py
│       │   └── tokenizer.py      # spaCy統合
│       ├── config.py             # 設定クラス (dataclass)
│       ├── logger.py             # ロギング
│       ├── cli.py                # CLIエントリーポイント
│       ├── resources/
│       │   ├── fonts/            # フォントファイル
│       │   └── data/             # ロゴ等
│       └── py.typed              # PEP 561 マーカー
├── tests/
│   ├── __init__.py
│   ├── test_translate.py
│   └── test_pdf_edit.py
├── pyproject.toml                # src-layout + entry_points
├── README.md
└── LICENSE
```

---

## 4. フェーズ別実装計画

### Phase 1: 基盤整備（src-layout移行）

**目標**: パッケージとしてインストール可能な状態を作る

#### タスク

| # | タスク | 詳細 |
|---|--------|------|
| 1.1 | ディレクトリ作成 | `src/index_pdf_translation/` |
| 1.2 | `__init__.py` 作成 | 公開APIの定義 |
| 1.3 | `_version.py` 作成 | `__version__ = "2.0.0"` |
| 1.4 | `pyproject.toml` 更新 | src-layout設定 |
| 1.5 | `py.typed` 作成 | PEP 561準拠 |

#### `pyproject.toml` 変更点

```toml
[project]
name = "index-pdf-translation"
version = "2.0.0"
# ...

[project.scripts]
translate-pdf = "index_pdf_translation.cli:main"

[tool.uv]
# package = false を削除

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/index_pdf_translation"]
```

#### `__init__.py` 内容

```python
"""Index PDF Translation - PDF翻訳ライブラリ"""

from index_pdf_translation._version import __version__
from index_pdf_translation.core.translate import pdf_translate
from index_pdf_translation.config import TranslationConfig

__all__ = [
    "__version__",
    "pdf_translate",
    "TranslationConfig",
]
```

#### 完了条件

- [ ] `uv pip install -e .` が成功
- [ ] `python -c "from index_pdf_translation import pdf_translate"` が成功

---

### Phase 2: モジュール移行

**目標**: 既存コードをパッケージ内に移動し、インポートを修正

#### タスク

| # | タスク | 移行元 | 移行先 |
|---|--------|--------|--------|
| 2.1 | logger移行 | `modules/logger.py` | `src/.../logger.py` |
| 2.2 | pdf_edit移行 | `modules/pdf_edit.py` | `src/.../core/pdf_edit.py` |
| 2.3 | translate移行 | `modules/translate.py` | `src/.../core/translate.py` |
| 2.4 | spacy_api移行 | `modules/spacy_api.py` | `src/.../nlp/tokenizer.py` |
| 2.5 | フォント移行 | `fonts/` | `src/.../resources/fonts/` |
| 2.6 | データ移行 | `data/` | `src/.../resources/data/` |
| 2.7 | インポート修正 | 全ファイル | 相対インポートに変更 |

#### インポート修正例

**Before (modules/pdf_edit.py)**:
```python
from modules.logger import get_logger
from modules.spacy_api import tokenize_text
```

**After (src/index_pdf_translation/core/pdf_edit.py)**:
```python
from index_pdf_translation.logger import get_logger
from index_pdf_translation.nlp.tokenizer import tokenize_text
```

#### 完了条件

- [ ] `modules/` ディレクトリ削除完了
- [ ] 全インポートが新パスで動作
- [ ] `uv run python -c "from index_pdf_translation.core.translate import pdf_translate"` 成功

---

### Phase 3: リソースパス解決

**目標**: ハードコードパスを `importlib.resources` に置換

#### タスク

| # | タスク | 詳細 |
|---|--------|------|
| 3.1 | resources モジュール作成 | パス解決ヘルパー関数 |
| 3.2 | pdf_edit.py 修正 | フォントパス動的取得 |
| 3.3 | ロゴパス修正 | `indqx_qr.png` 動的取得 |
| 3.4 | MANIFEST.in 作成 | 非Pythonファイルの同梱設定 |

#### resources.py 実装

```python
"""パッケージリソースへのアクセスを提供"""

from importlib import resources
from pathlib import Path

def get_font_path(font_name: str) -> Path:
    """フォントファイルのパスを取得"""
    with resources.as_file(
        resources.files("index_pdf_translation.resources.fonts").joinpath(font_name)
    ) as path:
        return Path(path)

def get_logo_path() -> Path:
    """ロゴ画像のパスを取得"""
    with resources.as_file(
        resources.files("index_pdf_translation.resources.data").joinpath("indqx_qr.png")
    ) as path:
        return Path(path)
```

#### pdf_edit.py 修正

```python
# Before
FONT_PATH_EN = "fonts/LiberationSerif-Regular.ttf"

# After
from index_pdf_translation.resources import get_font_path
FONT_PATH_EN = get_font_path("LiberationSerif-Regular.ttf")
```

#### pyproject.toml 追加設定

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/index_pdf_translation"]

# 非Pythonファイル（フォント・ロゴ）をwheelに同梱
[tool.hatch.build.targets.wheel.force-include]
"src/index_pdf_translation/resources/fonts" = "index_pdf_translation/resources/fonts"
"src/index_pdf_translation/resources/data" = "index_pdf_translation/resources/data"

[tool.hatch.build.targets.sdist]
include = [
    "src/index_pdf_translation/resources/**/*",
]
```

#### Wheel内容確認コマンド

```bash
# ビルド後、wheelにリソースが含まれているか確認
uv build
unzip -l dist/*.whl | grep -E '\.(ttf|png)$'
```

#### 完了条件

- [ ] フォントがパッケージから正しくロード
- [ ] ロゴがパッケージから正しくロード
- [ ] `pip install` 後もリソースアクセス可能
- [ ] wheel内にフォント（.ttf）とロゴ（.png）が含まれていることを確認

---

### Phase 4: 設定の外部化

**目標**: ハードコード設定を環境変数/引数ベースに変更

#### タスク

| # | タスク | 詳細 |
|---|--------|------|
| 4.1 | TranslationConfig dataclass作成 | 設定オブジェクト |
| 4.2 | 環境変数サポート | `DEEPL_API_KEY` 等 |
| 4.3 | pdf_translate シグネチャ変更 | config引数追加 |
| 4.4 | 旧config.py 削除 | ルートから削除 |

#### config.py 新設計

```python
"""設定管理"""

import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TranslationConfig:
    """翻訳設定"""

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

    def __post_init__(self):
        if not self.api_key:
            raise ValueError(
                "DeepL API key required. Set DEEPL_API_KEY environment variable "
                "or pass api_key parameter."
            )

@dataclass
class LanguageConfig:
    """言語設定"""
    deepl: str
    spacy: str

SUPPORTED_LANGUAGES: dict[str, LanguageConfig] = {
    "en": LanguageConfig(deepl="EN", spacy="en_core_web_sm"),
    "ja": LanguageConfig(deepl="JA", spacy="ja_core_news_sm"),
}
```

#### pdf_translate 新シグネチャ

```python
# Before
async def pdf_translate(
    key: str,
    pdf_data: bytes,
    source_lang: str = "en",
    to_lang: str = "ja",
    api_url: str = "https://api.deepl.com/v2/translate",
    debug: bool = False,
    add_logo: bool = True,
) -> Optional[bytes]:

# After (オーバーロード提供)
async def pdf_translate(
    pdf_data: bytes,
    *,
    config: Optional[TranslationConfig] = None,
    # 個別パラメータも許容（後方互換）
    api_key: Optional[str] = None,
    source_lang: str = "en",
    target_lang: str = "ja",
    **kwargs,
) -> Optional[bytes]:
```

#### 完了条件

- [ ] `DEEPL_API_KEY` 環境変数で動作
- [ ] `TranslationConfig` オブジェクトで動作
- [ ] 旧 `config.py` 削除済み

---

### Phase 5: CLI移行

**目標**: entry_points ベースのCLIに移行

#### タスク

| # | タスク | 詳細 |
|---|--------|------|
| 5.1 | cli.py 作成 | パッケージ内にCLI実装 |
| 5.2 | entry_points 設定 | `translate-pdf` コマンド |
| 5.3 | 環境変数対応 | API key読み込み |
| 5.4 | translate_pdf.py 削除 | ルートから削除 |

#### cli.py 実装

```python
"""CLI エントリーポイント"""

import argparse
import asyncio
import sys
from pathlib import Path

from index_pdf_translation import pdf_translate
from index_pdf_translation.config import TranslationConfig, SUPPORTED_LANGUAGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="translate-pdf",
        description="PDF翻訳ツール - 学術論文PDFを翻訳し見開きPDFを生成",
    )
    parser.add_argument("input", type=Path, help="入力PDFファイル")
    parser.add_argument("-o", "--output", type=Path, help="出力ファイルパス")
    parser.add_argument("-s", "--source", default="en",
                        choices=list(SUPPORTED_LANGUAGES.keys()))
    parser.add_argument("-t", "--target", default="ja",
                        choices=list(SUPPORTED_LANGUAGES.keys()))
    parser.add_argument("--api-key", help="DeepL API Key (環境変数 DEEPL_API_KEY も可)")
    parser.add_argument("--no-logo", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    # 設定作成
    config = TranslationConfig(
        api_key=args.api_key or "",  # 環境変数フォールバックは TranslationConfig 内
        source_lang=args.source,
        target_lang=args.target,
        add_logo=not args.no_logo,
        debug=args.debug,
    )

    # 翻訳実行
    with open(args.input, "rb") as f:
        pdf_data = f.read()

    result = await pdf_translate(pdf_data, config=config)

    if result is None:
        print("翻訳に失敗しました", file=sys.stderr)
        return 1

    # 出力
    output_path = args.output or Path(f"./output/translated_{args.input.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(result)

    print(f"完了: {output_path}")
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
```

#### pyproject.toml

```toml
[project.scripts]
translate-pdf = "index_pdf_translation.cli:main"
```

#### 完了条件

- [ ] `translate-pdf paper.pdf` が動作
- [ ] `DEEPL_API_KEY` 環境変数で認証
- [ ] ルートの `translate_pdf.py` 削除済み

---

### Phase 6: テスト・ドキュメント

**目標**: 品質保証とドキュメント整備

#### タスク

| # | タスク | 詳細 |
|---|--------|------|
| 6.1 | pytest 設定 | `tests/` ディレクトリ構成 |
| 6.2 | 単体テスト | config, resources, tokenizer |
| 6.3 | 統合テスト | pdf_translate (モック使用) |
| 6.4 | README 更新 | ライブラリ使用例追加 |
| 6.5 | API ドキュメント | docstring確認 |

#### テスト例

```python
# tests/test_config.py
import os
import pytest
from index_pdf_translation.config import TranslationConfig

def test_config_from_env(monkeypatch):
    monkeypatch.setenv("DEEPL_API_KEY", "test-key")
    config = TranslationConfig()
    assert config.api_key == "test-key"

def test_config_missing_key():
    with pytest.raises(ValueError, match="API key required"):
        TranslationConfig(api_key="")
```

#### README 追記内容

```markdown
## ライブラリとして使用

### インストール

```bash
pip install index-pdf-translation
# または
uv add index-pdf-translation
```

### Python から利用

```python
import asyncio
from index_pdf_translation import pdf_translate, TranslationConfig

config = TranslationConfig(
    api_key="your-deepl-key",
    source_lang="en",
    target_lang="ja",
)

async def translate():
    with open("paper.pdf", "rb") as f:
        pdf_data = f.read()

    result = await pdf_translate(pdf_data, config=config)

    with open("translated.pdf", "wb") as f:
        f.write(result)

asyncio.run(translate())
```
```

#### 完了条件

- [ ] `uv run pytest` 全テストパス
- [ ] README にライブラリ使用例
- [ ] 全公開関数に docstring

---

## 5. マイルストーン

| Phase | 内容 | 見積もり PR数 |
|-------|------|--------------|
| Phase 1 | 基盤整備 | 1 |
| Phase 2 | モジュール移行 | 1-2 |
| Phase 3 | リソースパス解決 | 1 |
| Phase 4 | 設定の外部化 | 1 |
| Phase 5 | CLI移行 | 1 |
| Phase 6 | テスト・ドキュメント | 1 |
| **合計** | | **6-7 PR** |

---

## 6. リスク評価

| リスク | 影響度 | 対策 |
|--------|--------|------|
| インポートパス変更による破壊 | 高 | Phase毎に動作確認 |
| フォント読み込み失敗 | 中 | フォールバック維持 |
| 環境変数未設定エラー | 中 | 明確なエラーメッセージ |
| PyPI名の競合 | 低 | 事前に `pip search` 確認 |

---

## 7. 後方互換性

### 維持するもの

- CLI の基本的な使い方 (`translate-pdf paper.pdf`)
- オプション (`-o`, `-s`, `-t`, `--no-logo`, `--debug`)

### 破壊的変更

- `from modules.translate import pdf_translate` → `from index_pdf_translation import pdf_translate`
- `config.py` のハードコード設定 → 環境変数 or パラメータ
- `python translate_pdf.py` → `translate-pdf` (entry_point)

### 移行ガイド

```python
# Before (v1.x)
from modules.translate import pdf_translate
from config import DEEPL_API_KEY

result = await pdf_translate(
    key=DEEPL_API_KEY,
    pdf_data=data,
    source_lang="en",
    to_lang="ja",
)

# After (v2.x)
from index_pdf_translation import pdf_translate, TranslationConfig

config = TranslationConfig(api_key="your-key")
result = await pdf_translate(data, config=config)
```

---

## 8. チェックリスト

### Phase 1 完了チェック
- [ ] `src/index_pdf_translation/__init__.py` 作成
- [ ] `pyproject.toml` src-layout対応
- [ ] `uv pip install -e .` 成功

### Phase 2 完了チェック
- [ ] 全モジュールを `src/` 配下に移動
- [ ] インポートパス更新
- [ ] `modules/` ディレクトリ削除

### Phase 3 完了チェック
- [ ] `importlib.resources` 導入
- [ ] フォント・ロゴのパッケージ同梱
- [ ] `unzip -l dist/*.whl | grep -E '\.(ttf|png)$'` でwheel内リソース確認

### Phase 4 完了チェック
- [ ] `TranslationConfig` dataclass 実装
- [ ] 環境変数 `DEEPL_API_KEY` サポート
- [ ] ルート `config.py` 削除

### Phase 5 完了チェック
- [ ] `cli.py` 実装
- [ ] `[project.scripts]` 設定
- [ ] `translate-pdf` コマンド動作
- [ ] ルート `translate_pdf.py` 削除

### Phase 6 完了チェック
- [ ] テスト追加
- [ ] README 更新
- [ ] docstring 確認

---

## 9. 参考リンク

- [Python Packaging User Guide](https://packaging.python.org/)
- [src layout vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- [importlib.resources](https://docs.python.org/3/library/importlib.resources.html)
- [PEP 561 – Distributing and Packaging Type Information](https://peps.python.org/pep-0561/)
