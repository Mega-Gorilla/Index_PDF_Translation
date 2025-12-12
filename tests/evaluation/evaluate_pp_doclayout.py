#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""PP-DocLayout evaluation script for document layout analysis."""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
from paddleocr import LayoutDetection


def evaluate_pp_doclayout(
    pdf_paths: list[str],
    output_dir: str,
    model_name: str = "PP-DocLayout-L",
) -> dict:
    """Evaluate PP-DocLayout on given PDFs.

    Args:
        pdf_paths: List of PDF file paths to evaluate
        output_dir: Directory to save evaluation results
        model_name: PP-DocLayout model variant (S, M, L, plus-L)

    Returns:
        Summary statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize model
    print(f"Loading {model_name}...")
    model = LayoutDetection(model_name=model_name)

    results = []

    for pdf_path in pdf_paths:
        pdf_name = Path(pdf_path).stem
        print(f"\nProcessing: {pdf_name}")

        pdf_output_dir = output_path / pdf_name
        pdf_output_dir.mkdir(parents=True, exist_ok=True)
        vis_dir = pdf_output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Open PDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        all_blocks = []
        blocks_by_type = {}
        errors = []

        start_time = time.perf_counter()

        for page_num in range(total_pages):
            try:
                page = doc[page_num]
                # Render page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_path = vis_dir / f"page_{page_num:03d}_input.png"
                pix.save(str(img_path))

                # Run detection
                output = model.predict(str(img_path), batch_size=1)

                for res in output:
                    # Save visualization
                    vis_path = vis_dir / f"page_{page_num:03d}.png"
                    res.save_to_img(str(vis_path))

                    # Extract detection results (res is dict-like with 'boxes' key)
                    boxes = res.get('boxes', [])
                    if boxes:
                        for box in boxes:
                            # Scale bbox back to original PDF coordinates
                            coord = box['coordinate']
                            bbox = [float(coord[0]) / 2, float(coord[1]) / 2,
                                    float(coord[2]) / 2, float(coord[3]) / 2]  # Undo 2x zoom
                            block_type = box.get('label', 'unknown')
                            confidence = float(box.get('score', 0.0))

                            block = {
                                "bbox": bbox,
                                "text": "",
                                "block_type": block_type,
                                "confidence": confidence,
                                "font_size": None,
                                "page_num": page_num
                            }
                            all_blocks.append(block)

                            blocks_by_type[block_type] = blocks_by_type.get(block_type, 0) + 1

                # Remove input image to save space
                img_path.unlink()

            except Exception as e:
                errors.append(f"Page {page_num}: {str(e)}")
                print(f"  Error on page {page_num}: {e}")

        processing_time = time.perf_counter() - start_time
        doc.close()

        # Save blocks
        blocks_path = pdf_output_dir / "blocks.json"
        with open(blocks_path, "w", encoding="utf-8") as f:
            json.dump(all_blocks, f, indent=2, ensure_ascii=False)

        # Save metadata
        metadata = {
            "tool_name": f"PP-DocLayout ({model_name})",
            "pdf_path": pdf_path,
            "pdf_name": pdf_name,
            "total_pages": total_pages,
            "processing_time_seconds": processing_time,
            "blocks_detected": len(all_blocks),
            "blocks_by_type": dict(sorted(blocks_by_type.items())),
            "model_name": model_name,
            "errors": errors,
            "evaluation_timestamp": datetime.now().isoformat()
        }

        metadata_path = pdf_output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        results.append(metadata)
        print(f"  Pages: {total_pages}, Blocks: {len(all_blocks)}, Time: {processing_time:.3f}s")

    return results


def main():
    # PDF samples
    pdf_dir = Path("tests/pdf_sample")
    pdf_paths = [
        str(pdf_dir / "2201.11903v6.pdf"),
        str(pdf_dir / "2302.13971v1.pdf"),
        str(pdf_dir / "2308.08155v2.pdf"),
    ]

    # Filter existing files
    pdf_paths = [p for p in pdf_paths if Path(p).exists()]

    if not pdf_paths:
        print("No PDF files found in tests/pdf_sample/")
        return

    output_dir = "tests/evaluation/outputs/PP_DocLayout"

    print("=" * 60)
    print("PP-DocLayout Evaluation")
    print("=" * 60)

    results = evaluate_pp_doclayout(pdf_paths, output_dir, model_name="PP-DocLayout-L")

    # Generate summary
    summary_path = Path(output_dir) / "evaluation_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# PP-DocLayout 評価結果サマリー\n\n")
        f.write(f"**評価日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. 評価概要\n\n")
        f.write(f"- 評価PDF数: {len(results)}\n")
        f.write(f"- モデル: PP-DocLayout-L\n\n")

        f.write("## 2. 検出クラス\n\n")
        all_types = set()
        for r in results:
            all_types.update(r["blocks_by_type"].keys())

        f.write("| クラス名 | 説明 |\n")
        f.write("|----------|------|\n")
        for t in sorted(all_types):
            f.write(f"| {t} | - |\n")

        f.write("\n## 3. PDF別結果\n\n")
        f.write("| PDF | ページ数 | ブロック数 | 処理時間 | タイプ分布 |\n")
        f.write("|-----|---------|-----------|---------|-----------|\n")

        total_blocks = 0
        total_time = 0
        for r in results:
            types_str = ", ".join([f"{k}: {v}" for k, v in sorted(r["blocks_by_type"].items())])
            f.write(f"| {r['pdf_name']} | {r['total_pages']} | {r['blocks_detected']} | {r['processing_time_seconds']:.3f}s | {types_str} |\n")
            total_blocks += r["blocks_detected"]
            total_time += r["processing_time_seconds"]

        f.write(f"\n**合計**: {total_blocks} ブロック, {total_time:.3f}s\n")

        f.write("\n## 4. 処理速度\n\n")
        f.write("| PDF | ページ数 | 処理時間 | ページ/秒 |\n")
        f.write("|-----|---------|---------|----------|\n")
        for r in results:
            pages_per_sec = r["total_pages"] / r["processing_time_seconds"]
            f.write(f"| {r['pdf_name']} | {r['total_pages']} | {r['processing_time_seconds']:.3f}s | {pages_per_sec:.2f} |\n")

    print(f"\nSummary saved to: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
