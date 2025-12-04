# SPDX-License-Identifier: AGPL-3.0-only
"""
Index PDF Translation - Configuration

DeepL API設定と言語設定を管理します。
"""

from typing import TypedDict


class LanguageConfig(TypedDict):
    """言語設定の型定義"""

    deepl: str  # DeepL API用言語コード
    spacy: str  # spaCyモデル名


# DeepL API 設定
DEEPL_API_KEY: str = "your-api-key"
DEEPL_API_URL: str = "https://api-free.deepl.com/v2/translate"  # Pro: https://api.deepl.com/v2/translate

# 出力設定
OUTPUT_DIR: str = "./output/"

# 言語設定
SUPPORTED_LANGUAGES: dict[str, LanguageConfig] = {
    "en": {"deepl": "EN", "spacy": "en_core_web_sm"},
    "ja": {"deepl": "JA", "spacy": "ja_core_news_sm"},
}
