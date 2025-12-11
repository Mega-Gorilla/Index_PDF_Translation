#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
DeepSeek-VL2 Evaluation Script for Issue #31

This script evaluates DeepSeek-VL2's OCR and layout detection capabilities
for PDF translation use cases.

Note: Requires transformers patch for newer versions (LlamaFlashAttention2 removed).
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, GenerationConfig

# Import DeepSeek-VL2 components
from deepseek_vl2.models import DeepseekVLV2Processor


def load_model(model_id: str = "deepseek-ai/deepseek-vl2-tiny"):
    """Load DeepSeek-VL2 model and processor."""
    print(f"Loading model: {model_id}")

    processor = DeepseekVLV2Processor.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )

    # Initialize generation_config for compatibility with newer transformers
    if model.generation_config is None:
        model.generation_config = GenerationConfig(
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    return model, processor


def pdf_page_to_image(pdf_path: Path, page_num: int, dpi: int = 150) -> Image.Image:
    """Convert a PDF page to PIL Image."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # Render at specified DPI
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)

    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    return img


def extract_text_with_deepseek(
    model,
    processor,
    image: Image.Image,
    prompt: str = "Extract all text from this image. Preserve the layout and structure."
) -> str:
    """Extract text from an image using DeepSeek-VL2."""

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n{prompt}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # Prepare inputs
    prepare_inputs = processor.__call__(
        conversations=conversation,
        images=[image],
        force_batchify=True,
        system_prompt=""
    ).to(model.device, dtype=torch.bfloat16)

    # Generate
    with torch.no_grad():
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        # Create generation config
        generation_config = GenerationConfig(
            max_new_tokens=2048,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            generation_config=generation_config,
        )

    # Decode
    answer = processor.tokenizer.decode(
        outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
        skip_special_tokens=True
    )

    return answer.strip()


def detect_layout_with_deepseek(
    model,
    processor,
    image: Image.Image,
) -> str:
    """Detect layout elements using DeepSeek-VL2."""

    prompt = """Analyze this academic paper page and identify the following layout elements:
1. Title (main title of the paper)
2. Section headers (numbered or bold headers like "1. Introduction")
3. Body text (main paragraphs)
4. Figures and their captions
5. Tables and their captions
6. Equations
7. Page headers/footers
8. Author information
9. Abstract

For each element found, describe its location (top, middle, bottom, left, right) and provide a brief excerpt of its content."""

    return extract_text_with_deepseek(model, processor, image, prompt)


def evaluate_single_pdf(
    model,
    processor,
    pdf_path: Path,
    output_dir: Path,
    max_pages: int = 3
) -> dict[str, Any]:
    """Evaluate DeepSeek-VL2 on a single PDF."""

    pdf_name = pdf_path.stem
    pdf_output_dir = output_dir / pdf_name
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages_to_process = min(max_pages, total_pages)
    doc.close()

    results = {
        "pdf_name": pdf_name,
        "total_pages": total_pages,
        "pages_processed": pages_to_process,
        "ocr_results": [],
        "layout_results": [],
        "processing_times": [],
    }

    for page_num in range(pages_to_process):
        print(f"  Processing page {page_num + 1}/{pages_to_process}...")

        # Convert page to image
        image = pdf_page_to_image(pdf_path, page_num)

        # Save the image for reference
        image_path = pdf_output_dir / f"page_{page_num:03d}.png"
        image.save(image_path)

        # OCR extraction
        start_time = time.time()
        ocr_text = extract_text_with_deepseek(model, processor, image)
        ocr_time = time.time() - start_time

        # Layout detection
        start_time = time.time()
        layout_text = detect_layout_with_deepseek(model, processor, image)
        layout_time = time.time() - start_time

        results["ocr_results"].append({
            "page": page_num,
            "text": ocr_text,
            "time_seconds": ocr_time
        })

        results["layout_results"].append({
            "page": page_num,
            "analysis": layout_text,
            "time_seconds": layout_time
        })

        results["processing_times"].append({
            "page": page_num,
            "ocr_time": ocr_time,
            "layout_time": layout_time,
            "total_time": ocr_time + layout_time
        })

        # Save individual page results
        with open(pdf_output_dir / f"page_{page_num:03d}_ocr.txt", "w", encoding="utf-8") as f:
            f.write(ocr_text)

        with open(pdf_output_dir / f"page_{page_num:03d}_layout.txt", "w", encoding="utf-8") as f:
            f.write(layout_text)

        print(f"    OCR: {ocr_time:.2f}s, Layout: {layout_time:.2f}s")

    # Calculate summary statistics
    total_ocr_time = sum(r["time_seconds"] for r in results["ocr_results"])
    total_layout_time = sum(r["time_seconds"] for r in results["layout_results"])

    results["summary"] = {
        "total_ocr_time": total_ocr_time,
        "total_layout_time": total_layout_time,
        "total_time": total_ocr_time + total_layout_time,
        "avg_ocr_time_per_page": total_ocr_time / pages_to_process,
        "avg_layout_time_per_page": total_layout_time / pages_to_process,
        "pages_per_second_ocr": pages_to_process / total_ocr_time if total_ocr_time > 0 else 0,
        "pages_per_second_layout": pages_to_process / total_layout_time if total_layout_time > 0 else 0,
    }

    # Save full results
    with open(pdf_output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def main():
    """Main evaluation function."""

    # Paths
    project_root = Path(__file__).parent.parent
    fixtures_dir = project_root / "tests" / "fixtures"
    arxiv_dir = project_root / "tests" / "evaluation" / "fixtures"
    output_dir = project_root / "tests" / "evaluation" / "outputs" / "DeepSeek_VL2"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect PDFs to evaluate
    test_pdfs = []

    # Fixture PDFs
    for pdf_file in fixtures_dir.glob("*.pdf"):
        test_pdfs.append(pdf_file)

    # arXiv PDFs
    for pdf_file in arxiv_dir.glob("*.pdf"):
        test_pdfs.append(pdf_file)

    print(f"Found {len(test_pdfs)} PDFs to evaluate")

    # Load model
    print("\n" + "=" * 60)
    print("Loading DeepSeek-VL2-Tiny model...")
    print("=" * 60)

    model, processor = load_model()

    # Check GPU memory
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory used: {memory_used:.2f} GB")

    # Evaluate each PDF
    all_results = []

    for pdf_path in test_pdfs:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {pdf_path.name}")
        print("=" * 60)

        try:
            result = evaluate_single_pdf(model, processor, pdf_path, output_dir, max_pages=2)
            all_results.append(result)

            print(f"\nSummary for {pdf_path.name}:")
            print(f"  Pages processed: {result['pages_processed']}")
            print(f"  Total OCR time: {result['summary']['total_ocr_time']:.2f}s")
            print(f"  Total Layout time: {result['summary']['total_layout_time']:.2f}s")
            print(f"  OCR speed: {result['summary']['pages_per_second_ocr']:.2f} pages/sec")

        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Generate summary report
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    summary = {
        "tool_name": "DeepSeek-VL2-Tiny",
        "model_id": "deepseek-ai/deepseek-vl2-tiny",
        "evaluation_timestamp": datetime.now().isoformat(),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "torch_dtype": "bfloat16",
        "pdfs_evaluated": len(all_results),
        "results": all_results
    }

    # Calculate overall statistics
    total_pages = sum(r["pages_processed"] for r in all_results)
    total_ocr_time = sum(r["summary"]["total_ocr_time"] for r in all_results)
    total_layout_time = sum(r["summary"]["total_layout_time"] for r in all_results)

    summary["overall_stats"] = {
        "total_pages_processed": total_pages,
        "total_ocr_time_seconds": total_ocr_time,
        "total_layout_time_seconds": total_layout_time,
        "avg_ocr_pages_per_second": total_pages / total_ocr_time if total_ocr_time > 0 else 0,
        "avg_layout_pages_per_second": total_pages / total_layout_time if total_layout_time > 0 else 0,
    }

    # Save summary
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nTotal pages processed: {total_pages}")
    print(f"Total OCR time: {total_ocr_time:.2f}s ({total_pages / total_ocr_time:.2f} pages/sec)")
    print(f"Total Layout time: {total_layout_time:.2f}s ({total_pages / total_layout_time:.2f} pages/sec)")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
