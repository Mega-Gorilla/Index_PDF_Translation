# SPDX-License-Identifier: AGPL-3.0-only
"""
Index PDF Translation - spaCy Language Processing

spaCyを使用したテキストトークン化機能を提供します。
"""

from typing import Optional

import spacy
from spacy.language import Language

from config import SUPPORTED_LANGUAGES
from modules.logger import get_logger

logger = get_logger("spacy")

# spaCyモデルのキャッシュ
_loaded_models: dict[str, Language] = {}


def load_model(lang_code: str) -> Optional[Language]:
    """
    指定された言語コードのspaCyモデルをロードします。

    一度ロードされたモデルはキャッシュされ、次回以降は再利用されます。

    Args:
        lang_code: 言語コード ('en', 'ja' など)

    Returns:
        spaCy Languageオブジェクト、またはロード失敗時はNone
    """
    # キャッシュ済みモデルがあれば返す
    if lang_code in _loaded_models:
        return _loaded_models[lang_code]

    # サポート対象言語かチェック
    if lang_code not in SUPPORTED_LANGUAGES:
        logger.warning(f"サポートされていない言語コード: '{lang_code}'")
        return None

    model_name = SUPPORTED_LANGUAGES[lang_code]["spacy"]

    try:
        nlp = spacy.load(model_name)
        _loaded_models[lang_code] = nlp
        logger.debug(f"spaCyモデルをロードしました: {model_name}")
        return nlp
    except OSError as e:
        logger.error(f"spaCyモデルのロードに失敗しました ({lang_code}): {e}")
        return None


def tokenize_text(lang_code: str, text: str) -> list[str]:
    """
    テキストをトークン化し、アルファベット文字のみのトークンリストを返します。

    Args:
        lang_code: 言語コード ('en', 'ja' など)
        text: トークン化するテキスト

    Returns:
        トークンのリスト（アルファベット文字のみ）
    """
    nlp = load_model(lang_code)

    if nlp is None:
        return []

    doc = nlp(text)
    # アルファベット文字で構成されるトークンのみを抽出
    tokens = [token.text for token in doc if token.is_alpha]

    return tokens


if __name__ == "__main__":
    # 使用例
    text_en = "This is an English sentence."
    tokens_en = tokenize_text("en", text_en)
    print(f"English text tokens: {tokens_en}")

    text_ja = "これは日本語の文です。"
    tokens_ja = tokenize_text("ja", text_ja)
    print(f"Japanese text tokens: {tokens_ja}")
