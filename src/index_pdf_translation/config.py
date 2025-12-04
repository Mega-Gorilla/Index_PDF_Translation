# SPDX-License-Identifier: AGPL-3.0-only
"""
Index PDF Translation - Configuration

言語設定を管理します。
TranslationConfig dataclass は Phase 4 で追加予定。
"""

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

# Phase 4 で追加予定:
# @dataclass
# class TranslationConfig:
#     api_key: str
#     api_url: str
#     source_lang: str = "en"
#     target_lang: str = "ja"
#     ...
