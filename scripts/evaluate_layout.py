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
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


@dataclass
class BlockInfo:
    """検出されたブロック情報"""
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    text: str
    block_type: str  # "body", "heading", "header", "footer", "caption", "table", "other"
    confidence: float = 1.0
    font_size: float | None = None
    page_num: int = 0


@dataclass
class EvaluationResult:
    """評価結果"""
    tool_name: str
    pdf_path: str
    pdf_name: str
    total_pages: int
    processing_time_seconds: float
    blocks_detected: int
    blocks_by_type: dict[str, int] = field(default_factory=dict)
    memory_usage_mb: float | None = None
    errors: list[str] = field(default_factory=list)
    blocks: list[dict[str, Any]] = field(default_factory=list)
    markdown_output: str | None = None
    json_output: str | None = None  # to_json() の生出力
    evaluation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


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
                        bbox=tuple(bbox),
                        text=text,
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
        tool_name="PyMuPDF_Baseline",
        pdf_path=str(pdf_path),
        pdf_name=pdf_path.stem,
        total_pages=total_pages,
        processing_time_seconds=processing_time,
        blocks_detected=len(blocks),
        blocks_by_type=blocks_by_type,
        errors=errors,
        blocks=[asdict(b) for b in blocks],
    )


def evaluate_pymupdf4llm_layout(pdf_path: Path) -> EvaluationResult:
    """
    PyMuPDF4LLM + Layout によるレイアウト解析評価

    to_json() を使用して正確な座標情報を取得します。
    """
    start_time = time.perf_counter()
    errors: list[str] = []
    blocks: list[BlockInfo] = []
    total_pages = 0
    markdown_output = ""
    json_output = ""

    try:
        # pymupdf.layout を先にインポート（重要）
        has_layout = False
        try:
            import pymupdf.layout
            has_layout = True
        except ImportError:
            errors.append("pymupdf-layout not installed")

        import pymupdf4llm

        # JSON形式で抽出（座標情報を含む）
        json_output = pymupdf4llm.to_json(str(pdf_path))
        json_data = json.loads(json_output)

        # Markdown形式でも抽出（参照用）
        markdown_output = pymupdf4llm.to_markdown(str(pdf_path))

        # ページ数取得
        total_pages = len(json_data.get("pages", []))

        # boxclass から block_type へのマッピング
        boxclass_to_type = {
            "title": "heading_1",
            "section-header": "heading_2",
            "page-header": "header",
            "page-footer": "footer",
            "text": "body",
            "list-item": "list",
            "table": "table",
            "figure": "image",
            "caption": "caption",
        }

        # JSONからブロック情報を解析（正確な座標付き）
        for page_data in json_data.get("pages", []):
            page_num = page_data.get("page_number", 1) - 1  # 0-indexed

            for box in page_data.get("boxes", []):
                # 座標を取得
                x0 = box.get("x0", 0)
                y0 = box.get("y0", 0)
                x1 = box.get("x1", 0)
                y1 = box.get("y1", 0)
                bbox = (x0, y0, x1, y1)

                # boxclass を取得してマッピング
                boxclass = box.get("boxclass", "text")
                block_type = boxclass_to_type.get(boxclass, "body")

                # テキストを抽出 (textlines[].spans[].text の構造)
                text_parts = []
                textlines = box.get("textlines") or []
                for textline in textlines:
                    if isinstance(textline, dict):
                        spans = textline.get("spans") or []
                        for span in spans:
                            if isinstance(span, dict):
                                span_text = span.get("text", "")
                                if span_text:
                                    text_parts.append(span_text)
                text = " ".join(text_parts).strip()

                if text:  # 空でないブロックのみ追加
                    blocks.append(BlockInfo(
                        bbox=bbox,
                        text=text,
                        block_type=block_type,
                        page_num=page_num,
                    ))

        if not has_layout:
            errors.append("Running without pymupdf-layout (reduced accuracy)")

    except Exception as e:
        errors.append(f"Error: {str(e)}")
        import traceback
        errors.append(traceback.format_exc())

    processing_time = time.perf_counter() - start_time

    # タイプ別集計
    blocks_by_type: dict[str, int] = {}
    for block in blocks:
        blocks_by_type[block.block_type] = blocks_by_type.get(block.block_type, 0) + 1

    return EvaluationResult(
        tool_name="PyMuPDF4LLM_Layout",
        pdf_path=str(pdf_path),
        pdf_name=pdf_path.stem,
        total_pages=total_pages,
        processing_time_seconds=processing_time,
        blocks_detected=len(blocks),
        blocks_by_type=blocks_by_type,
        errors=errors,
        blocks=[asdict(b) for b in blocks],
        markdown_output=markdown_output,
        json_output=json_output,
    )


def save_evaluation_outputs(result: EvaluationResult, output_dir: Path) -> dict[str, Path]:
    """
    評価結果を保存
    """
    # 出力ディレクトリ作成
    tool_dir = output_dir / result.tool_name / result.pdf_name
    tool_dir.mkdir(parents=True, exist_ok=True)

    saved_files: dict[str, Path] = {}

    # メタデータ保存 (ブロック詳細を除く)
    metadata = {
        "tool_name": result.tool_name,
        "pdf_path": result.pdf_path,
        "pdf_name": result.pdf_name,
        "total_pages": result.total_pages,
        "processing_time_seconds": result.processing_time_seconds,
        "blocks_detected": result.blocks_detected,
        "blocks_by_type": result.blocks_by_type,
        "errors": result.errors,
        "evaluation_timestamp": result.evaluation_timestamp,
    }
    metadata_path = tool_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    saved_files["metadata"] = metadata_path

    # ブロック詳細保存
    blocks_path = tool_dir / "blocks.json"
    with open(blocks_path, "w", encoding="utf-8") as f:
        json.dump(result.blocks, f, indent=2, ensure_ascii=False)
    saved_files["blocks"] = blocks_path

    # Markdown出力保存 (PyMuPDF4LLMの場合)
    if result.markdown_output:
        md_path = tool_dir / "output.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.markdown_output)
        saved_files["markdown"] = md_path

    # JSON出力保存 (PyMuPDF4LLMの場合 - to_json()の生出力)
    if result.json_output:
        json_raw_path = tool_dir / "output_raw.json"
        with open(json_raw_path, "w", encoding="utf-8") as f:
            # 整形して保存
            json_data = json.loads(result.json_output)
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        saved_files["json_raw"] = json_raw_path

    return saved_files


def run_evaluation(
    pdf_path: Path,
    tools: list[str] | None = None,
    output_dir: Path | None = None,
) -> list[EvaluationResult]:
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
            print(f"Evaluating {tool} on {pdf_path.name}...")
            result = tool_functions[tool](pdf_path)
            results.append(result)
            print(f"  - Blocks detected: {result.blocks_detected}")
            print(f"  - Processing time: {result.processing_time_seconds:.3f}s")
            print(f"  - Types: {result.blocks_by_type}")
            if result.errors:
                print(f"  - Errors: {result.errors}")

            # 出力保存
            if output_dir:
                saved = save_evaluation_outputs(result, output_dir)
                print(f"  - Saved: {list(saved.keys())}")

    return results


def run_batch_evaluation(
    pdf_dir: Path,
    tools: list[str] | None = None,
    output_dir: Path | None = None,
) -> list[EvaluationResult]:
    """
    ディレクトリ内の全PDFを評価
    """
    all_results: list[EvaluationResult] = []

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in sorted(pdf_files):
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")
        print('='*60)
        results = run_evaluation(pdf_path, tools, output_dir)
        all_results.extend(results)

    return all_results


def generate_summary_report(results: list[EvaluationResult], output_path: Path) -> None:
    """
    評価結果のサマリーレポートを生成
    """
    # ツール別・PDF別に整理
    by_tool: dict[str, list[EvaluationResult]] = {}
    for r in results:
        if r.tool_name not in by_tool:
            by_tool[r.tool_name] = []
        by_tool[r.tool_name].append(r)

    report_lines = [
        "# レイアウト解析評価結果サマリー",
        "",
        f"**評価日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 評価概要",
        "",
        f"- 評価ツール数: {len(by_tool)}",
        f"- 評価PDF数: {len(results) // len(by_tool) if by_tool else 0}",
        "",
    ]

    # ツール別サマリー
    report_lines.extend([
        "## 2. ツール別結果",
        "",
    ])

    for tool_name, tool_results in by_tool.items():
        report_lines.extend([
            f"### {tool_name}",
            "",
            "| PDF | ページ数 | ブロック数 | 処理時間 | タイプ分布 |",
            "|-----|---------|-----------|---------|-----------|",
        ])

        total_time = 0
        total_blocks = 0
        for r in tool_results:
            types_str = ", ".join(f"{k}: {v}" for k, v in sorted(r.blocks_by_type.items()))
            report_lines.append(
                f"| {r.pdf_name} | {r.total_pages} | {r.blocks_detected} | "
                f"{r.processing_time_seconds:.3f}s | {types_str} |"
            )
            total_time += r.processing_time_seconds
            total_blocks += r.blocks_detected

        report_lines.extend([
            "",
            f"**合計**: {total_blocks} ブロック, {total_time:.3f}s",
            "",
        ])

    # 比較テーブル
    if len(by_tool) > 1:
        report_lines.extend([
            "## 3. ツール比較",
            "",
        ])

        # PDF名でグループ化
        pdf_names = set(r.pdf_name for r in results)
        for pdf_name in sorted(pdf_names):
            report_lines.extend([
                f"### {pdf_name}",
                "",
                "| ツール | ブロック数 | 処理時間 | 見出し | 本文 | 表 |",
                "|--------|-----------|---------|--------|------|-----|",
            ])

            for tool_name in by_tool:
                r = next((x for x in by_tool[tool_name] if x.pdf_name == pdf_name), None)
                if r:
                    heading_count = sum(v for k, v in r.blocks_by_type.items() if "heading" in k)
                    body_count = r.blocks_by_type.get("body", 0)
                    table_count = r.blocks_by_type.get("table", 0)
                    report_lines.append(
                        f"| {tool_name} | {r.blocks_detected} | {r.processing_time_seconds:.3f}s | "
                        f"{heading_count} | {body_count} | {table_count} |"
                    )

            report_lines.append("")

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nSummary report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="レイアウト解析ツール評価")
    parser.add_argument(
        "input_path",
        type=Path,
        help="評価対象のPDFファイルまたはディレクトリパス",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=["baseline", "pymupdf4llm"],
        default=["baseline", "pymupdf4llm"],
        help="評価するツール",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/evaluation/outputs"),
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="サマリーレポートの出力パス",
    )

    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"Error: Path not found: {args.input_path}")
        return 1

    # 出力ディレクトリ作成
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 評価実行
    if args.input_path.is_file():
        results = run_evaluation(args.input_path, args.tools, args.output_dir)
    else:
        results = run_batch_evaluation(args.input_path, args.tools, args.output_dir)

    # サマリーレポート生成
    if args.report:
        generate_summary_report(results, args.report)
    elif len(results) > 2:
        # 複数PDFの場合は自動的にレポート生成
        report_path = args.output_dir / "evaluation_summary.md"
        generate_summary_report(results, report_path)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
