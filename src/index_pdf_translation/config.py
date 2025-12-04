# SPDX-License-Identifier: AGPL-3.0-only
"""
Index PDF Translation - Configuration

翻訳設定と言語設定を管理します。
"""

import os
from dataclasses import dataclass, field
from typing import TypedDict


class LanguageConfig(TypedDict):
    """言語設定の型定義"""

    deepl: str  # DeepL API用言語コード
    spacy: str  # spaCyモデル名


# 言語設定
SUPPORTED_LANGUAGES: dict[str, LanguageConfig] = {
    "en": {"deepl": "EN", "spacy": "en_core_web_sm"},
    "ja": {"deepl": "JA", "spacy": "ja_core_news_sm"},
}

# デフォルト出力ディレクトリ
DEFAULT_OUTPUT_DIR: str = "./output/"


@dataclass
class TranslationConfig:
    """
    翻訳設定を管理するdataclass。

    環境変数または直接パラメータで設定可能。

    Attributes:
        api_key: DeepL APIキー。環境変数 DEEPL_API_KEY からも取得可能。
        api_url: DeepL API URL。環境変数 DEEPL_API_URL からも取得可能。
        source_lang: 翻訳元言語コード (default: "en")
        target_lang: 翻訳先言語コード (default: "ja")
        add_logo: ロゴウォーターマークを追加 (default: True)
        debug: デバッグモード (default: False)

    Examples:
        >>> # 環境変数から設定
        >>> os.environ["DEEPL_API_KEY"] = "your-key"
        >>> config = TranslationConfig()

        >>> # 直接パラメータで設定
        >>> config = TranslationConfig(api_key="your-key", target_lang="ja")

        >>> # 設定オブジェクトを使用
        >>> result = await pdf_translate(pdf_data, config=config)
    """

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
        if not self.api_key:
            raise ValueError(
                "DeepL API key required. "
                "Set DEEPL_API_KEY environment variable or pass api_key parameter."
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

    @property
    def deepl_target_lang(self) -> str:
        """DeepL API用の言語コードを取得"""
        return SUPPORTED_LANGUAGES[self.target_lang]["deepl"]

    @property
    def deepl_source_lang(self) -> str:
        """DeepL API用のソース言語コードを取得"""
        return SUPPORTED_LANGUAGES[self.source_lang]["deepl"]
