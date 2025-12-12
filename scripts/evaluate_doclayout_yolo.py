#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
DocLayout-YOLO 評価スクリプト

DocLayout-YOLOを使用したレイアウト解析の評価。
PDFを画像に変換してからYOLOモデルで検出を行う。

検出クラス (10種類):
- Title: タイトル
- Plain Text: 本文
- Abandoned Text: 削除テキスト
- Figure: 図
- Figure Caption: 図のキャプション
- Table: 表
- Table Caption: 表のキャプション
- Table Footnote: 表の脚注
- Isolated Formula: 数式
- Formula Caption: 数式のキャプション
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


# DocLayout-YOLO クラス名マッピング
DOCLAYOUT_CLASSES = {
    0: "title",
    1: "plain_text",
    2: "abandoned_text",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolated_formula",
    9: "formula_caption",
}


@dataclass
class BlockInfo:
    """検出されたブロック情報"""
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    text: str
    block_type: str
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
    device: str = "cpu"
    evaluation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def pdf_to_images(pdf_path: Path, dpi: int = 150) -> list[tuple[int, Any]]:
    """
    PDFを画像に変換

    Returns:
        list of (page_num, PIL.Image)
    """
    from PIL import Image
    import io

    images = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        # 高解像度でレンダリング
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        # PIL Imageに変換
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append((page_num, img))

    doc.close()
    return images


def evaluate_doclayout_yolo(
    pdf_path: Path,
    device: str = "cuda:0",
    conf_threshold: float = 0.2,
    imgsz: int = 1024,
) -> EvaluationResult:
    """
    DocLayout-YOLO によるレイアウト解析評価
    """
    from doclayout_yolo import YOLOv10
    from huggingface_hub import hf_hub_download
    import torch

    start_time = time.perf_counter()
    errors: list[str] = []
    blocks: list[BlockInfo] = []
    total_pages = 0

    # デバイス確認
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        errors.append("CUDA not available, falling back to CPU")

    try:
        # モデルダウンロード
        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )

        # モデルロード
        model = YOLOv10(model_path)

        # PDFを画像に変換
        images = pdf_to_images(pdf_path)
        total_pages = len(images)

        # 各ページで検出
        for page_num, img in images:
            # 推論実行
            results = model.predict(
                img,
                imgsz=imgsz,
                conf=conf_threshold,
                device=device,
            )

            # 結果を解析
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # bbox取得 (x0, y0, x1, y1)
                        xyxy = box.xyxy[0].cpu().numpy()
                        bbox = tuple(float(x) for x in xyxy)

                        # クラスID取得
                        cls_id = int(box.cls[0].cpu().numpy())
                        block_type = DOCLAYOUT_CLASSES.get(cls_id, f"unknown_{cls_id}")

                        # 信頼度
                        confidence = float(box.conf[0].cpu().numpy())

                        blocks.append(BlockInfo(
                            bbox=bbox,
                            text="",  # YOLOはテキスト抽出しない
                            block_type=block_type,
                            confidence=confidence,
                            page_num=page_num,
                        ))

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
        tool_name="DocLayout_YOLO",
        pdf_path=str(pdf_path),
        pdf_name=pdf_path.stem,
        total_pages=total_pages,
        processing_time_seconds=processing_time,
        blocks_detected=len(blocks),
        blocks_by_type=blocks_by_type,
        errors=errors,
        blocks=[asdict(b) for b in blocks],
        device=device,
    )


def save_evaluation_outputs(result: EvaluationResult, output_dir: Path) -> dict[str, Path]:
    """
    評価結果を保存
    """
    tool_dir = output_dir / result.tool_name / result.pdf_name
    tool_dir.mkdir(parents=True, exist_ok=True)

    saved_files: dict[str, Path] = {}

    # メタデータ保存
    metadata = {
        "tool_name": result.tool_name,
        "pdf_path": result.pdf_path,
        "pdf_name": result.pdf_name,
        "total_pages": result.total_pages,
        "processing_time_seconds": result.processing_time_seconds,
        "blocks_detected": result.blocks_detected,
        "blocks_by_type": result.blocks_by_type,
        "device": result.device,
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

    return saved_files


def save_detection_visualization(
    pdf_path: Path,
    result: EvaluationResult,
    output_dir: Path,
) -> Path | None:
    """
    検出結果の可視化画像を保存
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        images = pdf_to_images(pdf_path)

        # 各ページの検出結果を描画
        vis_dir = output_dir / result.tool_name / result.pdf_name / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # ブロックをページごとにグループ化
        blocks_by_page: dict[int, list] = {}
        for block in result.blocks:
            page_num = block["page_num"]
            if page_num not in blocks_by_page:
                blocks_by_page[page_num] = []
            blocks_by_page[page_num].append(block)

        # カラーマップ
        colors = {
            "title": "red",
            "plain_text": "blue",
            "abandoned_text": "gray",
            "figure": "green",
            "figure_caption": "lightgreen",
            "table": "orange",
            "table_caption": "yellow",
            "table_footnote": "gold",
            "isolated_formula": "purple",
            "formula_caption": "violet",
        }

        for page_num, img in images:
            draw = ImageDraw.Draw(img)

            if page_num in blocks_by_page:
                for block in blocks_by_page[page_num]:
                    bbox = block["bbox"]
                    block_type = block["block_type"]
                    confidence = block["confidence"]

                    color = colors.get(block_type, "black")

                    # 矩形を描画
                    draw.rectangle(bbox, outline=color, width=2)

                    # ラベルを描画
                    label = f"{block_type} ({confidence:.2f})"
                    draw.text((bbox[0], bbox[1] - 15), label, fill=color)

            # 保存
            vis_path = vis_dir / f"page_{page_num:03d}.png"
            img.save(vis_path)

        return vis_dir

    except Exception as e:
        print(f"Visualization error: {e}")
        return None


def run_evaluation(
    pdf_path: Path,
    output_dir: Path | None = None,
    device: str = "cuda:0",
    visualize: bool = False,
) -> EvaluationResult:
    """
    評価を実行
    """
    print(f"\nEvaluating DocLayout-YOLO on {pdf_path.name}...")
    print(f"  Device: {device}")

    result = evaluate_doclayout_yolo(pdf_path, device=device)

    print(f"  - Blocks detected: {result.blocks_detected}")
    print(f"  - Processing time: {result.processing_time_seconds:.3f}s")
    print(f"  - Types: {result.blocks_by_type}")
    if result.errors:
        print(f"  - Errors: {result.errors}")

    # 出力保存
    if output_dir:
        saved = save_evaluation_outputs(result, output_dir)
        print(f"  - Saved: {list(saved.keys())}")

        if visualize:
            vis_dir = save_detection_visualization(pdf_path, result, output_dir)
            if vis_dir:
                print(f"  - Visualizations: {vis_dir}")

    return result


def run_batch_evaluation(
    pdf_dir: Path,
    output_dir: Path | None = None,
    device: str = "cuda:0",
    visualize: bool = False,
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
        result = run_evaluation(pdf_path, output_dir, device, visualize)
        all_results.append(result)

    return all_results


def generate_summary_report(results: list[EvaluationResult], output_path: Path) -> None:
    """
    評価結果のサマリーレポートを生成
    """
    report_lines = [
        "# DocLayout-YOLO 評価結果サマリー",
        "",
        f"**評価日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 評価概要",
        "",
        f"- 評価PDF数: {len(results)}",
        f"- デバイス: {results[0].device if results else 'N/A'}",
        "",
        "## 2. 検出クラス",
        "",
        "| クラスID | クラス名 | 説明 |",
        "|----------|----------|------|",
        "| 0 | title | タイトル |",
        "| 1 | plain_text | 本文 |",
        "| 2 | abandoned_text | 削除テキスト |",
        "| 3 | figure | 図 |",
        "| 4 | figure_caption | 図のキャプション |",
        "| 5 | table | 表 |",
        "| 6 | table_caption | 表のキャプション |",
        "| 7 | table_footnote | 表の脚注 |",
        "| 8 | isolated_formula | 数式 |",
        "| 9 | formula_caption | 数式のキャプション |",
        "",
        "## 3. PDF別結果",
        "",
        "| PDF | ページ数 | ブロック数 | 処理時間 | タイプ分布 |",
        "|-----|---------|-----------|---------|-----------|",
    ]

    total_time = 0
    total_blocks = 0

    for r in results:
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
        "## 4. 処理速度",
        "",
        "| PDF | ページ数 | 処理時間 | ページ/秒 |",
        "|-----|---------|---------|----------|",
    ])

    for r in results:
        pages_per_sec = r.total_pages / r.processing_time_seconds if r.processing_time_seconds > 0 else 0
        report_lines.append(
            f"| {r.pdf_name} | {r.total_pages} | {r.processing_time_seconds:.3f}s | {pages_per_sec:.2f} |"
        )

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nSummary report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DocLayout-YOLO 評価")
    parser.add_argument(
        "input_path",
        type=Path,
        help="評価対象のPDFファイルまたはディレクトリパス",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="使用デバイス (cuda:0 or cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/evaluation/outputs"),
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="検出結果の可視化画像を保存",
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
        results = [run_evaluation(args.input_path, args.output_dir, args.device, args.visualize)]
    else:
        results = run_batch_evaluation(args.input_path, args.output_dir, args.device, args.visualize)

    # サマリーレポート生成
    if args.report:
        generate_summary_report(results, args.report)
    elif len(results) > 1:
        report_path = args.output_dir / "doclayout_yolo_summary.md"
        generate_summary_report(results, report_path)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
