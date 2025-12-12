#!/usr/bin/env python3
"""
pypdfium2 Text Insertion Verification Script

This script verifies whether pypdfium2 can insert text into PDFs,
including Japanese text (important for the translation pipeline).

License: AGPL-3.0-only
"""

import pypdfium2 as pdfium
import json
from pathlib import Path
from datetime import datetime


def verify_text_insertion(output_dir: str) -> dict:
    """
    Verify pypdfium2's text insertion capabilities.

    Args:
        output_dir: Directory to save outputs

    Returns:
        dict: Verification results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "capabilities": {},
        "errors": [],
        "tests": []
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Font paths
    # Use bundled fonts from the project
    japanese_font_path = "src/index_pdf_translation/resources/fonts/ipam.ttf"
    english_font_path = "src/index_pdf_translation/resources/fonts/LiberationSerif-Regular.ttf"

    # Check if fonts exist
    if not Path(japanese_font_path).exists():
        results["errors"].append(f"Japanese font not found: {japanese_font_path}")
        japanese_font_path = None
    if not Path(english_font_path).exists():
        results["errors"].append(f"English font not found: {english_font_path}")
        english_font_path = None

    results["fonts"] = {
        "japanese_font": japanese_font_path,
        "english_font": english_font_path
    }

    # Test 1: Create new PDF with English text
    test1_result = {"name": "english_text_insertion", "success": False}
    try:
        pdf = pdfium.PdfDocument.new()
        width, height = 595, 842  # A4 size

        page = pdf.new_page(width, height)

        if english_font_path:
            # Add font
            pdf_font = pdf.add_font(
                english_font_path,
                type=pdfium.FPDF_FONT_TRUETYPE,
                is_cid=False
            )

            # Insert text using raw API (simpler approach)
            page.insert_text(
                text="Hello, World! This is a test.",
                pos_x=50,
                pos_y=height - 100,
                font_size=14,
                pdf_font=pdf_font,
            )

            page.gen_content()

            # Save
            output_pdf = output_path / "english_text_test.pdf"
            with open(output_pdf, "wb") as f:
                pdf.save(f)

            test1_result["success"] = True
            test1_result["output"] = str(output_pdf)

            # Verify by re-opening
            verify_pdf = pdfium.PdfDocument(str(output_pdf))
            test1_result["pages_created"] = len(verify_pdf)
            verify_pdf.close()

        else:
            test1_result["error"] = "English font not available"

        pdf.close()

    except Exception as e:
        test1_result["error"] = str(e)
        import traceback
        test1_result["traceback"] = traceback.format_exc()

    results["tests"].append(test1_result)
    results["capabilities"]["can_insert_english_text"] = test1_result["success"]

    # Test 2: Create new PDF with Japanese text
    test2_result = {"name": "japanese_text_insertion", "success": False}
    try:
        pdf = pdfium.PdfDocument.new()
        width, height = 595, 842

        page = pdf.new_page(width, height)

        if japanese_font_path:
            # Add Japanese font
            pdf_font = pdf.add_font(
                japanese_font_path,
                type=pdfium.FPDF_FONT_TRUETYPE,
                is_cid=True  # CID font for CJK
            )

            # Try with harfbuzz if available
            try:
                hb_font = pdfium.HarfbuzzFont(japanese_font_path)
                page.insert_text(
                    text="こんにちは、世界！これはテストです。",
                    pos_x=50,
                    pos_y=height - 100,
                    font_size=14,
                    pdf_font=pdf_font,
                    hb_font=hb_font,
                )
                test2_result["harfbuzz_used"] = True
            except (AttributeError, ImportError):
                # Fallback without harfbuzz
                page.insert_text(
                    text="こんにちは、世界！",
                    pos_x=50,
                    pos_y=height - 100,
                    font_size=14,
                    pdf_font=pdf_font,
                )
                test2_result["harfbuzz_used"] = False

            page.gen_content()

            # Save
            output_pdf = output_path / "japanese_text_test.pdf"
            with open(output_pdf, "wb") as f:
                pdf.save(f)

            test2_result["success"] = True
            test2_result["output"] = str(output_pdf)

        else:
            test2_result["error"] = "Japanese font not available"

        pdf.close()

    except Exception as e:
        test2_result["error"] = str(e)
        import traceback
        test2_result["traceback"] = traceback.format_exc()

    results["tests"].append(test2_result)
    results["capabilities"]["can_insert_japanese_text"] = test2_result["success"]

    # Test 3: Modify existing PDF - remove text and insert new text
    test3_result = {"name": "modify_existing_pdf", "success": False}
    try:
        pdf_path = "tests/fixtures/sample_llama.pdf"

        # Get constants
        raw_module = getattr(pdfium, 'raw', None) or getattr(pdfium, 'pdfium_c', pdfium)
        FPDF_PAGEOBJ_TEXT = getattr(raw_module, 'FPDF_PAGEOBJ_TEXT', 1)

        pdf = pdfium.PdfDocument(pdf_path)
        page = pdf[0]
        height = page.get_height()

        # Target region: title area
        target_region = (100.0, 750.0, 500.0, 770.0)

        # Find and remove text in target region
        text_objects = list(page.get_objects(filter=[FPDF_PAGEOBJ_TEXT]))
        initial_count = len(text_objects)

        removed = 0
        for obj in text_objects:
            bounds = obj.get_bounds()
            left, bottom, right, top = bounds
            # Check if object is in target region (title)
            if bottom >= 750 and top <= 780:
                page.remove_obj(obj)
                removed += 1

        test3_result["text_objects_removed"] = removed

        # Add new text (translated title)
        if english_font_path:
            pdf_font = pdf.add_font(
                english_font_path,
                type=pdfium.FPDF_FONT_TRUETYPE,
                is_cid=False
            )

            page.insert_text(
                text="[REPLACED] New Title Text",
                pos_x=120,
                pos_y=height - 80,  # Near top of page
                font_size=14,
                pdf_font=pdf_font,
            )

            test3_result["new_text_inserted"] = True
        else:
            test3_result["new_text_inserted"] = False

        page.gen_content()

        # Save
        output_pdf = output_path / "modified_with_new_text.pdf"
        with open(output_pdf, "wb") as f:
            pdf.save(f)

        test3_result["success"] = True
        test3_result["output"] = str(output_pdf)

        # Verify
        verify_pdf = pdfium.PdfDocument(str(output_pdf))
        verify_page = verify_pdf[0]
        final_count = len(list(verify_page.get_objects(filter=[FPDF_PAGEOBJ_TEXT])))
        test3_result["initial_text_objects"] = initial_count
        test3_result["final_text_objects"] = final_count
        verify_pdf.close()

        pdf.close()

    except Exception as e:
        test3_result["error"] = str(e)
        import traceback
        test3_result["traceback"] = traceback.format_exc()

    results["tests"].append(test3_result)
    results["capabilities"]["can_modify_existing_pdf"] = test3_result["success"]

    # Summary
    results["summary"] = {
        "all_capabilities": results["capabilities"],
        "full_workflow_possible": all([
            results["capabilities"].get("can_insert_english_text", False),
            results["capabilities"].get("can_insert_japanese_text", False),
            results["capabilities"].get("can_modify_existing_pdf", False),
        ])
    }

    return results


def main():
    output_dir = "tests/evaluation/outputs/pypdfium2_verification"

    print("=" * 60)
    print("pypdfium2 Text Insertion Verification")
    print("=" * 60)

    results = verify_text_insertion(output_dir)

    # Save results
    output_path = Path(output_dir)
    with open(output_path / "text_insertion_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print results
    print("\nCapabilities:")
    for cap, value in results.get("capabilities", {}).items():
        status = "✅" if value else "❌"
        print(f"  {status} {cap}: {value}")

    print("\nTest Results:")
    for test in results.get("tests", []):
        status = "✅" if test.get("success") else "❌"
        print(f"  {status} {test['name']}")
        if test.get("output"):
            print(f"      Output: {test['output']}")
        if test.get("error"):
            print(f"      Error: {test['error']}")

    if results.get("errors"):
        print("\nErrors:")
        for err in results["errors"]:
            print(f"  ⚠️ {err}")

    print(f"\nFull Workflow Possible: {'✅' if results.get('summary', {}).get('full_workflow_possible') else '❌'}")
    print(f"\nResults saved to: {output_path}/text_insertion_results.json")

    return results


if __name__ == "__main__":
    main()
