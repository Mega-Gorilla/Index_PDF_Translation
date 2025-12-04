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
"""

import argparse
import asyncio
import sys
from pathlib import Path

from config import DEEPL_API_KEY, DEEPL_API_URL, OUTPUT_DIR, SUPPORTED_LANGUAGES
from modules.translate import pdf_translate


def parse_args():
    """コマンドライン引数をパースします。"""
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
        help=f"出力ファイルのパス (デフォルト: {OUTPUT_DIR}translated_<input>.pdf)",
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


async def translate(args):
    """PDFを翻訳します。"""
    input_path = args.input

    # 入力ファイルの検証
    if not input_path.exists():
        print(f"エラー: ファイルが見つかりません: {input_path}", file=sys.stderr)
        return 1

    if not input_path.suffix.lower() == ".pdf":
        print(f"エラー: PDFファイルではありません: {input_path}", file=sys.stderr)
        return 1

    # 出力パスの決定
    if args.output:
        output_path = args.output
    else:
        output_dir = Path(OUTPUT_DIR)
        output_path = output_dir / f"translated_{input_path.name}"

    # 出力ディレクトリの作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 言語コードの取得
    source_deepl = SUPPORTED_LANGUAGES[args.source]["deepl"]
    target_deepl = SUPPORTED_LANGUAGES[args.target]["deepl"]

    # 進捗表示
    print(f"入力: {input_path}")
    print(f"出力: {output_path}")
    print(f"翻訳: {args.source.upper()} → {args.target.upper()}")
    if args.no_logo:
        print("ロゴ: 無効")
    if args.debug:
        print("デバッグモード: 有効")
    print()

    # PDFの読み込み
    with open(input_path, "rb") as f:
        pdf_data = f.read()

    # 翻訳の実行
    result_pdf = await pdf_translate(
        key=DEEPL_API_KEY,
        pdf_data=pdf_data,
        source_lang=args.source,
        to_lang=args.target,
        api_url=DEEPL_API_URL,
        debug=args.debug,
        add_logo=not args.no_logo,
    )

    if result_pdf is None:
        print("エラー: 翻訳に失敗しました", file=sys.stderr)
        return 1

    # 結果の保存
    with open(output_path, "wb") as f:
        f.write(result_pdf)

    print()
    print(f"完了: {output_path}")
    return 0


def main():
    """メインエントリーポイント。"""
    args = parse_args()
    exit_code = asyncio.run(translate(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
