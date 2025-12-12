#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""YOLO-DocLayNet evaluation script for document layout analysis."""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# DocLayNet class names (11 categories)
DOCLAYNET_CLASSES = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title"
}


def evaluate_yolo_doclaynet(
    pdf_paths: list[str],
    output_dir: str,
    model_variant: str = "yolov11l",  # yolov8x, yolov10b/l, yolov11l, yolov12l
) -> dict:
    """Evaluate YOLO-DocLayNet on given PDFs.

    Args:
        pdf_paths: List of PDF file paths to evaluate
        output_dir: Directory to save evaluation results
        model_variant: YOLO model variant (yolov8x, yolov10x, yolov11x, yolov12x)

    Returns:
        Summary statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download model from HuggingFace
    print(f"Downloading {model_variant}-doclaynet model...")
    model_filename = f"{model_variant}-doclaynet.pt"

    try:
        model_path = hf_hub_download(
            repo_id="hantian/yolo-doclaynet",
            filename=model_filename
        )
    except Exception as e:
        print(f"Failed to download {model_filename}: {e}")
        print("Falling back to yolov11l-doclaynet.pt")
        model_path = hf_hub_download(
            repo_id="hantian/yolo-doclaynet",
            filename="yolov11l-doclaynet.pt"
        )
        model_variant = "yolov11l"

    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Check device
    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None or True else "cpu"
    print(f"Using device: {device}")

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
                detections = model.predict(
                    str(img_path),
                    device=device,
                    verbose=False,
                    conf=0.25
                )

                for det in detections:
                    # Save visualization
                    vis_path = vis_dir / f"page_{page_num:03d}.png"
                    det_img = det.plot()
                    import cv2
                    cv2.imwrite(str(vis_path), det_img)

                    # Extract boxes
                    if det.boxes is not None and len(det.boxes) > 0:
                        boxes = det.boxes
                        for i in range(len(boxes)):
                            # Get bbox in xyxy format and scale back
                            xyxy = boxes.xyxy[i].cpu().numpy()
                            bbox = [float(coord / 2) for coord in xyxy]  # Undo 2x zoom

                            cls_id = int(boxes.cls[i].cpu().item())
                            confidence = float(boxes.conf[i].cpu().item())
                            block_type = DOCLAYNET_CLASSES.get(cls_id, f"class_{cls_id}")

                            block = {
                                "bbox": bbox,  # Already a list
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
            "tool_name": f"YOLO-DocLayNet ({model_variant})",
            "pdf_path": pdf_path,
            "pdf_name": pdf_name,
            "total_pages": total_pages,
            "processing_time_seconds": processing_time,
            "blocks_detected": len(all_blocks),
            "blocks_by_type": dict(sorted(blocks_by_type.items())),
            "model_variant": model_variant,
            "device": device,
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

    output_dir = "tests/evaluation/outputs/YOLO_DocLayNet"

    print("=" * 60)
    print("YOLO-DocLayNet Evaluation")
    print("=" * 60)

    # Use YOLOv11l (largest YOLOv11 model available)
    results = evaluate_yolo_doclaynet(pdf_paths, output_dir, model_variant="yolov11l")

    # Generate summary
    summary_path = Path(output_dir) / "evaluation_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# YOLO-DocLayNet 評価結果サマリー\n\n")
        f.write(f"**評価日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. 評価概要\n\n")
        f.write(f"- 評価PDF数: {len(results)}\n")
        f.write(f"- モデル: {results[0]['model_variant'] if results else 'N/A'}-doclaynet\n")
        f.write(f"- デバイス: {results[0]['device'] if results else 'N/A'}\n\n")

        f.write("## 2. 検出クラス (DocLayNet 11カテゴリ)\n\n")
        f.write("| クラスID | クラス名 | 説明 |\n")
        f.write("|----------|----------|------|\n")
        for cls_id, cls_name in DOCLAYNET_CLASSES.items():
            f.write(f"| {cls_id} | {cls_name} | - |\n")

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
