#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Index PDF Translation - CLI Tool

Translates academic paper PDFs and generates side-by-side PDF (original + translated).

Usage:
    translate-pdf <input.pdf> [options]

Examples:
    translate-pdf paper.pdf                      # Google Translate (default)
    translate-pdf paper.pdf --backend deepl      # DeepL (high quality)
    translate-pdf paper.pdf --backend openai     # OpenAI GPT (customizable)
    translate-pdf paper.pdf -o ./translated.pdf
    translate-pdf paper.pdf --source en --target ja

Environment Variables:
    DEEPL_API_KEY: DeepL API key (required for --backend deepl)
    DEEPL_API_URL: DeepL API URL (optional, default: Free API)
    OPENAI_API_KEY: OpenAI API key (required for --backend openai)
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
    Parse command line arguments.

    Returns:
        Parsed argument Namespace
    """
    parser = argparse.ArgumentParser(
        prog="translate-pdf",
        description="PDF Translation Tool - Translates academic paper PDFs and generates side-by-side PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s paper.pdf                        # Google Translate (default)
  %(prog)s paper.pdf --backend deepl        # DeepL (high quality)
  %(prog)s paper.pdf --backend openai       # OpenAI GPT (customizable)
  %(prog)s paper.pdf -o result.pdf          # Specify output file
  %(prog)s paper.pdf -s en -t ja            # English to Japanese
  %(prog)s paper.pdf --no-logo              # Without logo
  %(prog)s paper.pdf --debug                # Debug mode

OpenAI Options:
  %(prog)s paper.pdf --backend openai --openai-model gpt-4o
  %(prog)s paper.pdf --backend openai --openai-prompt "Translate medical terminology..."

Environment Variables:
  DEEPL_API_KEY    DeepL API key (required for --backend deepl)
  DEEPL_API_URL    DeepL API URL (optional)
  OPENAI_API_KEY   OpenAI API key (required for --backend openai)
""",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to PDF file to translate",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=f"Output file path (default: {DEFAULT_OUTPUT_DIR}translated_<input>.pdf)",
    )

    parser.add_argument(
        "-b",
        "--backend",
        default="google",
        choices=["google", "deepl", "openai"],
        help="Translation backend (default: google)",
    )

    parser.add_argument(
        "-s",
        "--source",
        default="en",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help="Source language (default: en)",
    )

    parser.add_argument(
        "-t",
        "--target",
        default="ja",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help="Target language (default: ja)",
    )

    parser.add_argument(
        "--api-key",
        help="DeepL API key (required for --backend deepl)",
    )

    parser.add_argument(
        "--api-url",
        help="DeepL API URL (optional)",
    )

    # OpenAI options
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (required for --backend openai)",
    )

    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model (default: gpt-4o-mini)",
    )

    parser.add_argument(
        "--openai-prompt",
        help="Custom system prompt for OpenAI ({source_lang}, {target_lang} placeholders)",
    )

    parser.add_argument(
        "--openai-prompt-file",
        type=Path,
        help="File containing custom system prompt for OpenAI",
    )

    parser.add_argument(
        "--no-logo",
        action="store_true",
        help="Disable logo watermark",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (generates block classification visualization PDF)",
    )

    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    """
    Translate PDF.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0: success, 1: failure)
    """
    input_path: Path = args.input

    # Validate input file
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    if input_path.suffix.lower() != ".pdf":
        print(f"Error: Not a PDF file: {input_path}", file=sys.stderr)
        return 1

    # Get API key (only for DeepL backend)
    api_key = ""
    api_url = ""
    if args.backend == "deepl":
        api_key = args.api_key or os.environ.get("DEEPL_API_KEY", "")
        if not api_key:
            print(
                "Error: DeepL API key is required for --backend deepl.\n"
                "  Set --api-key option or DEEPL_API_KEY environment variable.\n"
                "  Or use --backend google for API-key-free translation.",
                file=sys.stderr,
            )
            return 1
        api_url = args.api_url or os.environ.get(
            "DEEPL_API_URL", "https://api-free.deepl.com/v2/translate"
        )

    # Get OpenAI options (only for OpenAI backend)
    openai_api_key = ""
    openai_model = "gpt-4o-mini"
    openai_system_prompt = None
    if args.backend == "openai":
        openai_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not openai_api_key:
            print(
                "Error: OpenAI API key is required for --backend openai.\n"
                "  Set --openai-api-key option or OPENAI_API_KEY environment variable.\n"
                "  Or use --backend google for API-key-free translation.",
                file=sys.stderr,
            )
            return 1
        openai_model = args.openai_model

        # Load custom prompt from file or command line
        if args.openai_prompt_file:
            if not args.openai_prompt_file.exists():
                print(f"Error: Prompt file not found: {args.openai_prompt_file}", file=sys.stderr)
                return 1
            openai_system_prompt = args.openai_prompt_file.read_text(encoding="utf-8")
        elif args.openai_prompt:
            openai_system_prompt = args.openai_prompt

    # Determine output path
    if args.output:
        output_path: Path = args.output
    else:
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        output_path = output_dir / f"translated_{input_path.name}"

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Display progress
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Backend: {args.backend}")
    if args.backend == "openai":
        print(f"Model: {openai_model}")
        if openai_system_prompt:
            print("Custom prompt: enabled")
    print(f"Translation: {args.source.upper()} -> {args.target.upper()}")
    if args.no_logo:
        print("Logo: disabled")
    if args.debug:
        print("Debug mode: enabled")
    print()

    # Create TranslationConfig
    try:
        config = TranslationConfig(
            backend=args.backend,
            api_key=api_key,
            api_url=api_url,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            openai_system_prompt=openai_system_prompt,
            source_lang=args.source,
            target_lang=args.target,
            add_logo=not args.no_logo,
            debug=args.debug,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Read PDF
    with open(input_path, "rb") as f:
        pdf_data = f.read()

    # Execute translation
    try:
        result_pdf = await pdf_translate(pdf_data, config=config)
    except Exception as e:
        print(f"Error: Translation failed: {e}", file=sys.stderr)
        return 1

    if result_pdf is None:
        print("Error: Translation failed", file=sys.stderr)
        return 1

    # Save result
    with open(output_path, "wb") as f:
        f.write(result_pdf)

    print()
    print(f"Complete: {output_path}")
    return 0


def main() -> NoReturn:
    """Main entry point."""
    args = parse_args()
    exit_code = asyncio.run(run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
