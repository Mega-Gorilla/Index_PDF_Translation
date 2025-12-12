#!/usr/bin/env python3
"""
pypdfium2 Text Insertion Verification Script (Raw API)

This script uses the raw PDFium API to verify text insertion.

License: AGPL-3.0-only
"""

import pypdfium2 as pdfium
import ctypes
import json
from pathlib import Path
from datetime import datetime


def verify_text_insertion_raw(output_dir: str) -> dict:
    """
    Verify pypdfium2's text insertion using raw PDFium API.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "capabilities": {},
        "errors": [],
        "tests": []
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Test 1: Create PDF with standard font (simple ASCII text)
    test1_result = {"name": "standard_font_text", "success": False}
    try:
        # Create new PDF
        pdf = pdfium.PdfDocument.new()
        width, height = 595, 842  # A4

        page = pdf.new_page(width, height)

        # Load standard font (Helvetica)
        doc_handle = pdf.raw  # Get raw document handle
        font_handle = pdfium.raw.FPDFText_LoadStandardFont(doc_handle, b"Helvetica")

        if not font_handle:
            test1_result["error"] = "Failed to load standard font"
        else:
            test1_result["font_loaded"] = True

            # Create text object
            page_handle = page.raw
            text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(
                doc_handle,
                font_handle,
                ctypes.c_float(14.0)
            )

            if not text_obj:
                test1_result["error"] = "Failed to create text object"
            else:
                test1_result["text_obj_created"] = True

                # Set text content
                test_text = "Hello, World! This is pypdfium2 test."
                # Convert to FPDF_WIDESTRING (UTF-16LE)
                text_bytes = test_text.encode('utf-16-le') + b'\x00\x00'

                ok = pdfium.raw.FPDFText_SetText(text_obj, text_bytes)
                test1_result["text_set"] = bool(ok)

                # Set position using transformation matrix
                # FPDFPageObj_Transform(obj, a, b, c, d, e, f) where:
                # a=scale_x, d=scale_y, e=translate_x, f=translate_y
                pdfium.raw.FPDFPageObj_Transform(
                    text_obj,
                    ctypes.c_double(1.0),  # scale x
                    ctypes.c_double(0.0),  # skew x
                    ctypes.c_double(0.0),  # skew y
                    ctypes.c_double(1.0),  # scale y
                    ctypes.c_double(50.0),  # translate x
                    ctypes.c_double(height - 100)  # translate y
                )

                # Insert text object into page
                pdfium.raw.FPDFPage_InsertObject(page_handle, text_obj)

                # Generate content
                ok = pdfium.raw.FPDFPage_GenerateContent(page_handle)
                test1_result["content_generated"] = bool(ok)

                # Save
                output_pdf = output_path / "standard_font_test.pdf"
                with open(output_pdf, "wb") as f:
                    pdf.save(f)

                test1_result["success"] = True
                test1_result["output"] = str(output_pdf)

            # Don't close font - it's owned by the document

        pdf.close()

    except Exception as e:
        test1_result["error"] = str(e)
        import traceback
        test1_result["traceback"] = traceback.format_exc()

    results["tests"].append(test1_result)
    results["capabilities"]["can_insert_standard_font_text"] = test1_result["success"]

    # Test 2: Load custom TrueType font and insert text
    test2_result = {"name": "truetype_font_text", "success": False}
    try:
        font_path = "src/index_pdf_translation/resources/fonts/LiberationSerif-Regular.ttf"

        if not Path(font_path).exists():
            test2_result["error"] = f"Font file not found: {font_path}"
        else:
            # Read font data
            with open(font_path, "rb") as f:
                font_data = f.read()

            # Create PDF
            pdf = pdfium.PdfDocument.new()
            width, height = 595, 842
            page = pdf.new_page(width, height)

            doc_handle = pdf.raw
            page_handle = page.raw

            # Load TrueType font
            font_handle = pdfium.raw.FPDFText_LoadFont(
                doc_handle,
                font_data,
                len(font_data),
                pdfium.raw.FPDF_FONT_TRUETYPE,
                ctypes.c_int(0)  # not CID
            )

            if not font_handle:
                test2_result["error"] = "Failed to load TrueType font"
            else:
                test2_result["font_loaded"] = True

                # Create text object
                text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(
                    doc_handle,
                    font_handle,
                    ctypes.c_float(14.0)
                )

                if not text_obj:
                    test2_result["error"] = "Failed to create text object"
                else:
                    # Set text
                    test_text = "TrueType Font Test - Liberation Serif"
                    text_bytes = test_text.encode('utf-16-le') + b'\x00\x00'
                    ok = pdfium.raw.FPDFText_SetText(text_obj, text_bytes)
                    test2_result["text_set"] = bool(ok)

                    # Position
                    pdfium.raw.FPDFPageObj_Transform(
                        text_obj,
                        ctypes.c_double(1.0), ctypes.c_double(0.0),
                        ctypes.c_double(0.0), ctypes.c_double(1.0),
                        ctypes.c_double(50.0), ctypes.c_double(height - 100)
                    )

                    # Insert
                    pdfium.raw.FPDFPage_InsertObject(page_handle, text_obj)
                    pdfium.raw.FPDFPage_GenerateContent(page_handle)

                    # Save
                    output_pdf = output_path / "truetype_font_test.pdf"
                    with open(output_pdf, "wb") as f:
                        pdf.save(f)

                    test2_result["success"] = True
                    test2_result["output"] = str(output_pdf)

            pdf.close()

    except Exception as e:
        test2_result["error"] = str(e)
        import traceback
        test2_result["traceback"] = traceback.format_exc()

    results["tests"].append(test2_result)
    results["capabilities"]["can_insert_truetype_font_text"] = test2_result["success"]

    # Test 3: Japanese text with CID font
    test3_result = {"name": "japanese_cid_font_text", "success": False}
    try:
        font_path = "src/index_pdf_translation/resources/fonts/ipam.ttf"

        if not Path(font_path).exists():
            test3_result["error"] = f"Font file not found: {font_path}"
        else:
            with open(font_path, "rb") as f:
                font_data = f.read()

            pdf = pdfium.PdfDocument.new()
            width, height = 595, 842
            page = pdf.new_page(width, height)

            doc_handle = pdf.raw
            page_handle = page.raw

            # Load as CID font for CJK
            font_handle = pdfium.raw.FPDFText_LoadFont(
                doc_handle,
                font_data,
                len(font_data),
                pdfium.raw.FPDF_FONT_TRUETYPE,
                ctypes.c_int(1)  # is_cid = True for CJK
            )

            if not font_handle:
                test3_result["error"] = "Failed to load Japanese font"
            else:
                test3_result["font_loaded"] = True

                text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(
                    doc_handle,
                    font_handle,
                    ctypes.c_float(14.0)
                )

                if not text_obj:
                    test3_result["error"] = "Failed to create text object"
                else:
                    # Japanese text
                    test_text = "こんにちは世界"
                    text_bytes = test_text.encode('utf-16-le') + b'\x00\x00'
                    ok = pdfium.raw.FPDFText_SetText(text_obj, text_bytes)
                    test3_result["text_set"] = bool(ok)

                    pdfium.raw.FPDFPageObj_Transform(
                        text_obj,
                        ctypes.c_double(1.0), ctypes.c_double(0.0),
                        ctypes.c_double(0.0), ctypes.c_double(1.0),
                        ctypes.c_double(50.0), ctypes.c_double(height - 100)
                    )

                    pdfium.raw.FPDFPage_InsertObject(page_handle, text_obj)
                    pdfium.raw.FPDFPage_GenerateContent(page_handle)

                    output_pdf = output_path / "japanese_font_test.pdf"
                    with open(output_pdf, "wb") as f:
                        pdf.save(f)

                    test3_result["success"] = True
                    test3_result["output"] = str(output_pdf)

            pdf.close()

    except Exception as e:
        test3_result["error"] = str(e)
        import traceback
        test3_result["traceback"] = traceback.format_exc()

    results["tests"].append(test3_result)
    results["capabilities"]["can_insert_japanese_text"] = test3_result["success"]

    # Test 4: Full workflow - remove text and insert new text in existing PDF
    test4_result = {"name": "full_workflow", "success": False}
    try:
        pdf_path = "tests/fixtures/sample_llama.pdf"

        raw_module = getattr(pdfium, 'raw', None)
        FPDF_PAGEOBJ_TEXT = getattr(raw_module, 'FPDF_PAGEOBJ_TEXT', 1)

        pdf = pdfium.PdfDocument(pdf_path)
        page = pdf[0]
        page_height = page.get_height()

        doc_handle = pdf.raw
        page_handle = page.raw

        # Remove title text objects (y > 750)
        text_objects = list(page.get_objects(filter=[FPDF_PAGEOBJ_TEXT]))
        initial_count = len(text_objects)
        removed = 0

        for obj in text_objects:
            bounds = obj.get_bounds()
            if bounds[1] >= 750:  # bottom >= 750
                page.remove_obj(obj)
                removed += 1

        test4_result["objects_removed"] = removed

        # Load font for replacement text
        font_path = "src/index_pdf_translation/resources/fonts/LiberationSerif-Regular.ttf"
        if Path(font_path).exists():
            with open(font_path, "rb") as f:
                font_data = f.read()

            font_handle = pdfium.raw.FPDFText_LoadFont(
                doc_handle,
                font_data,
                len(font_data),
                pdfium.raw.FPDF_FONT_TRUETYPE,
                ctypes.c_int(0)
            )

            if font_handle:
                text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(
                    doc_handle,
                    font_handle,
                    ctypes.c_float(14.0)
                )

                if text_obj:
                    replacement_text = "[TRANSLATED] New Title Goes Here"
                    text_bytes = replacement_text.encode('utf-16-le') + b'\x00\x00'
                    pdfium.raw.FPDFText_SetText(text_obj, text_bytes)

                    pdfium.raw.FPDFPageObj_Transform(
                        text_obj,
                        ctypes.c_double(1.0), ctypes.c_double(0.0),
                        ctypes.c_double(0.0), ctypes.c_double(1.0),
                        ctypes.c_double(120.0), ctypes.c_double(page_height - 88)
                    )

                    pdfium.raw.FPDFPage_InsertObject(page_handle, text_obj)
                    test4_result["text_inserted"] = True

        page.gen_content()

        output_pdf = output_path / "full_workflow_test.pdf"
        with open(output_pdf, "wb") as f:
            pdf.save(f)

        # Verify
        verify_pdf = pdfium.PdfDocument(str(output_pdf))
        verify_page = verify_pdf[0]
        final_count = len(list(verify_page.get_objects(filter=[FPDF_PAGEOBJ_TEXT])))
        verify_pdf.close()

        test4_result["initial_text_objects"] = initial_count
        test4_result["final_text_objects"] = final_count
        test4_result["success"] = True
        test4_result["output"] = str(output_pdf)

        pdf.close()

    except Exception as e:
        test4_result["error"] = str(e)
        import traceback
        test4_result["traceback"] = traceback.format_exc()

    results["tests"].append(test4_result)
    results["capabilities"]["can_do_full_workflow"] = test4_result["success"]

    # Summary
    results["summary"] = {
        "all_capabilities": results["capabilities"],
        "pymupdf_replacement_viable": all([
            results["capabilities"].get("can_insert_standard_font_text", False),
            results["capabilities"].get("can_insert_truetype_font_text", False),
            results["capabilities"].get("can_insert_japanese_text", False),
            results["capabilities"].get("can_do_full_workflow", False),
        ])
    }

    return results


def main():
    output_dir = "tests/evaluation/outputs/pypdfium2_verification"

    print("=" * 60)
    print("pypdfium2 Text Insertion Verification (Raw API)")
    print("=" * 60)

    results = verify_text_insertion_raw(output_dir)

    # Save results
    output_path = Path(output_dir)
    with open(output_path / "text_insertion_raw_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

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
            print(f"      Error: {test['error'][:100]}...")

    summary = results.get("summary", {})
    print(f"\nPyMuPDF Replacement Viable: {'✅' if summary.get('pymupdf_replacement_viable') else '❌'}")
    print(f"\nResults saved to: {output_path}/text_insertion_raw_results.json")

    return results


if __name__ == "__main__":
    main()
