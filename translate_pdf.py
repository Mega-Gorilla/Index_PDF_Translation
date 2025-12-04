#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Index PDF Translation - CLI Tool

学術論文PDFを翻訳し、見開きPDF（オリジナル + 翻訳）を生成します。

Usage:
    python translate_pdf.py <input.pdf> [options]
    uv run python translate_pdf.py <input.pdf> [options]

Examples:
    python translate_pdf.py paper.pdf
    python translate_pdf.py paper.pdf -o ./translated.pdf
    python translate_pdf.py paper.pdf --source en --target ja
    python translate_pdf.py paper.pdf --no-logo --debug

Environment Variables:
    DEEPL_API_KEY: DeepL APIキー (必須)
    DEEPL_API_URL: DeepL API URL (オプション、デフォルト: Free API)
"""

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
    """
    コマンドライン引数をパースします。

    Returns:
        パースされた引数のNamespace
    """
    parser = argparse.ArgumentParser(
        prog="translate_pdf",
        description="PDF翻訳ツール - 学術論文PDFを翻訳し見開きPDFを生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s paper.pdf                    # デフォルト設定で翻訳
  %(prog)s paper.pdf -o result.pdf      # 出力ファイル指定
  %(prog)s paper.pdf -s en -t ja        # 英語→日本語
  %(prog)s paper.pdf --no-logo          # ロゴなし
  %(prog)s paper.pdf --debug            # デバッグモード

Environment Variables:
  DEEPL_API_KEY    DeepL APIキー (必須、--api-keyでも指定可)
  DEEPL_API_URL    DeepL API URL (オプション)
""",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="翻訳するPDFファイルのパス",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=f"出力ファイルのパス (デフォルト: {DEFAULT_OUTPUT_DIR}translated_<input>.pdf)",
    )

    parser.add_argument(
        "-s",
        "--source",
        default="en",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help="翻訳元の言語 (デフォルト: en)",
    )

    parser.add_argument(
        "-t",
        "--target",
        default="ja",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help="翻訳先の言語 (デフォルト: ja)",
    )

    parser.add_argument(
        "--api-key",
        help="DeepL APIキー (環境変数 DEEPL_API_KEY でも設定可)",
    )

    parser.add_argument(
        "--api-url",
        help="DeepL API URL (環境変数 DEEPL_API_URL でも設定可)",
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


async def translate(args: argparse.Namespace) -> int:
    """
    PDFを翻訳します。

    Args:
        args: コマンドライン引数

    Returns:
        終了コード（0: 成功, 1: 失敗）
    """
    input_path: Path = args.input

    # 入力ファイルの検証
    if not input_path.exists():
        print(f"エラー: ファイルが見つかりません: {input_path}", file=sys.stderr)
        return 1

    if input_path.suffix.lower() != ".pdf":
        print(f"エラー: PDFファイルではありません: {input_path}", file=sys.stderr)
        return 1

    # APIキーの取得（コマンドライン > 環境変数）
    api_key = args.api_key or os.environ.get("DEEPL_API_KEY", "")
    if not api_key:
        print(
            "エラー: DeepL APIキーが設定されていません。\n"
            "  --api-key オプションまたは環境変数 DEEPL_API_KEY を設定してください。",
            file=sys.stderr,
        )
        return 1

    # API URLの取得
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
    print(f"翻訳: {args.source.upper()} → {args.target.upper()}")
    if args.no_logo:
        print("ロゴ: 無効")
    if args.debug:
        print("デバッグモード: 有効")
    print()

    # TranslationConfig を作成
    try:
        config = TranslationConfig(
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
    """メインエントリーポイント。"""
    args = parse_args()
    exit_code = asyncio.run(translate(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
