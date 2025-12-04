# SPDX-License-Identifier: AGPL-3.0-only
"""
Index PDF Translation - Logging Configuration

アプリケーション全体で使用するロギング設定を提供します。
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "index_pdf_translation",
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    アプリケーション用のロガーをセットアップします。

    Args:
        name: ロガー名
        level: ログレベル (default: INFO)
        format_string: ログフォーマット文字列 (default: None で標準フォーマット使用)

    Returns:
        設定済みのLoggerインスタンス
    """
    logger = logging.getLogger(name)

    # 既にハンドラーが設定されている場合はスキップ
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # デフォルトフォーマット
    if format_string is None:
        format_string = "%(message)s"

    formatter = logging.Formatter(format_string)

    # コンソールハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


# デフォルトロガー
logger = setup_logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    ロガーインスタンスを取得します。

    Args:
        name: ロガー名 (None の場合はデフォルトロガーを返す)

    Returns:
        Loggerインスタンス
    """
    if name is None:
        return logger
    return logging.getLogger(f"index_pdf_translation.{name}")
