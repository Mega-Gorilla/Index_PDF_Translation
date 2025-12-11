#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
レイアウト解析ツール評価スクリプト

各ツールのレイアウト解析性能を評価するための共通フレームワーク。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


@dataclass
class BlockInfo:
    """検出されたブロック情報"""
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    text: str
    block_type: str  # "body", "heading", "header", "footer", "caption", "other"
    confidence: float = 1.0
    font_size: float | None = None
    page_num: int = 0


@dataclass
class EvaluationResult:
    """評価結果"""
    tool_name: str
    pdf_path: str
    total_pages: int
    processing_time_seconds: float
    blocks_detected: int
    blocks_by_type: dict[str, int] = field(default_factory=dict)
    memory_usage_mb: float | None = None
    errors: list[str] = field(default_factory=list)
    sample_blocks: list[dict[str, Any]] = field(default_factory=list)


def evaluate_pymupdf_baseline(pdf_path: Path) -> EvaluationResult:
    """
    PyMuPDF標準のテキスト抽出（現行実装のベースライン）
    """
    start_time = time.perf_counter()
    errors: list[str] = []
    blocks: list[BlockInfo] = []

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        for page_num, page in enumerate(doc):
            text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # テキストブロック
                    bbox = block.get("bbox", (0, 0, 0, 0))

                    # ブロック内のテキストを結合
                    text_parts = []
                    font_sizes = []
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_parts.append(span.get("text", ""))
                            font_sizes.append(span.get("size", 0))

                    text = " ".join(text_parts)
                    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0

                    blocks.append(BlockInfo(
                        bbox=bbox,
                        text=text[:200],  # サンプル用に切り詰め
                        block_type="unknown",  # ベースラインでは分類なし
                        font_size=avg_font_size,
                        page_num=page_num,
                    ))

        doc.close()

    except Exception as e:
        errors.append(str(e))
        total_pages = 0

    processing_time = time.perf_counter() - start_time

    # タイプ別集計
    blocks_by_type: dict[str, int] = {}
    for block in blocks:
        blocks_by_type[block.block_type] = blocks_by_type.get(block.block_type, 0) + 1

    return EvaluationResult(
        tool_name="PyMuPDF Baseline",
        pdf_path=str(pdf_path),
        total_pages=total_pages,
        processing_time_seconds=processing_time,
        blocks_detected=len(blocks),
        blocks_by_type=blocks_by_type,
        errors=errors,
        sample_blocks=[asdict(b) for b in blocks[:10]],  # 最初の10ブロックをサンプル
    )


def evaluate_pymupdf4llm_layout(pdf_path: Path) -> EvaluationResult:
    """
    PyMuPDF4LLM + Layout によるレイアウト解析評価
    """
    start_time = time.perf_counter()
    errors: list[str] = []
    blocks: list[BlockInfo] = []
    total_pages = 0

    try:
        # pymupdf.layout を先にインポート（重要）
        try:
            import pymupdf.layout
            has_layout = True
        except ImportError:
            has_layout = False
            errors.append("pymupdf-layout not installed")

        import pymupdf4llm

        # Markdown形式で抽出
        md_result = pymupdf4llm.to_markdown(str(pdf_path))

        # ページ数取得
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        # Markdownからブロック情報を解析（簡易版）
        lines = md_result.split("\n")
        current_type = "body"

        for line in lines:
            if not line.strip():
                continue

            # 見出し検出
            if line.startswith("# "):
                current_type = "heading"
                blocks.append(BlockInfo(
                    bbox=(0, 0, 0, 0),
                    text=line[2:].strip()[:200],
                    block_type="heading",
                    page_num=0,
                ))
            elif line.startswith("## ") or line.startswith("### "):
                blocks.append(BlockInfo(
                    bbox=(0, 0, 0, 0),
                    text=line.lstrip("#").strip()[:200],
                    block_type="heading",
                    page_num=0,
                ))
            elif line.startswith("|"):
                # テーブル
                blocks.append(BlockInfo(
                    bbox=(0, 0, 0, 0),
                    text=line[:200],
                    block_type="table",
                    page_num=0,
                ))
            else:
                blocks.append(BlockInfo(
                    bbox=(0, 0, 0, 0),
                    text=line.strip()[:200],
                    block_type="body",
                    page_num=0,
                ))

        if not has_layout:
            errors.append("Running without pymupdf-layout (reduced accuracy)")

    except Exception as e:
        errors.append(f"Error: {str(e)}")

    processing_time = time.perf_counter() - start_time

    # タイプ別集計
    blocks_by_type: dict[str, int] = {}
    for block in blocks:
        blocks_by_type[block.block_type] = blocks_by_type.get(block.block_type, 0) + 1

    return EvaluationResult(
        tool_name="PyMuPDF4LLM" + (" + Layout" if "pymupdf-layout not installed" not in str(errors) else ""),
        pdf_path=str(pdf_path),
        total_pages=total_pages,
        processing_time_seconds=processing_time,
        blocks_detected=len(blocks),
        blocks_by_type=blocks_by_type,
        errors=errors,
        sample_blocks=[asdict(b) for b in blocks[:10]],
    )


def run_evaluation(pdf_path: Path, tools: list[str] | None = None) -> list[EvaluationResult]:
    """
    指定されたPDFに対して各ツールの評価を実行
    """
    if tools is None:
        tools = ["baseline", "pymupdf4llm"]

    results: list[EvaluationResult] = []

    tool_functions = {
        "baseline": evaluate_pymupdf_baseline,
        "pymupdf4llm": evaluate_pymupdf4llm_layout,
    }

    for tool in tools:
        if tool in tool_functions:
            print(f"Evaluating {tool}...")
            result = tool_functions[tool](pdf_path)
            results.append(result)
            print(f"  - Blocks detected: {result.blocks_detected}")
            print(f"  - Processing time: {result.processing_time_seconds:.3f}s")
            if result.errors:
                print(f"  - Errors: {result.errors}")

    return results


def main():
    parser = argparse.ArgumentParser(description="レイアウト解析ツール評価")
    parser.add_argument("pdf_path", type=Path, help="評価対象のPDFファイルパス")
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=["baseline", "pymupdf4llm"],
        default=["baseline", "pymupdf4llm"],
        help="評価するツール",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="結果をJSONで出力するパス",
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: PDF not found: {args.pdf_path}")
        return 1

    results = run_evaluation(args.pdf_path, args.tools)

    # 結果表示
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for result in results:
        print(f"\n{result.tool_name}:")
        print(f"  Pages: {result.total_pages}")
        print(f"  Blocks: {result.blocks_detected}")
        print(f"  Time: {result.processing_time_seconds:.3f}s")
        print(f"  Types: {result.blocks_by_type}")

    # JSON出力
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
