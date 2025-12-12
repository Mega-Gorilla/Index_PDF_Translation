#!/usr/bin/env python3
"""
pypdfium2 Text Insertion Verification Script v2

Using corrected ctypes handling for PDFium raw API.

License: AGPL-3.0-only
"""

import pypdfium2 as pdfium
import ctypes
import json
from pathlib import Path
from datetime import datetime


def to_widestring(text: str):
    """Convert Python string to FPDF_WIDESTRING (UTF-16LE + null terminator)."""
    # Encode as UTF-16LE and add null terminator
    encoded = text.encode('utf-16-le') + b'\x00\x00'
    # Create ctypes array
    arr = (ctypes.c_ushort * (len(encoded) // 2))()
    for i in range(len(encoded) // 2):
        arr[i] = int.from_bytes(encoded[i*2:i*2+2], 'little')
    return arr


def to_byte_array(data: bytes):
    """Convert bytes to ctypes array."""
    arr = (ctypes.c_ubyte * len(data))()
    for i, b in enumerate(data):
        arr[i] = b
    return arr


def verify_text_insertion(output_dir: str) -> dict:
    """Verify text insertion with corrected ctypes."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "capabilities": {},
        "errors": [],
        "tests": []
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Test 1: Standard font text
    test1 = {"name": "standard_font_text", "success": False}
    try:
        pdf = pdfium.PdfDocument.new()
        width, height = 595, 842
        page = pdf.new_page(width, height)

        doc_handle = pdf.raw
        page_handle = page.raw

        # Load standard font
        font_handle = pdfium.raw.FPDFText_LoadStandardFont(
            doc_handle,
            b"Helvetica"
        )

        if font_handle:
            test1["font_loaded"] = True

            # Create text object
            text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(
                doc_handle,
                font_handle,
                ctypes.c_float(14.0)
            )

            if text_obj:
                test1["text_obj_created"] = True

                # Set text using wide string
                text_ws = to_widestring("Hello World - pypdfium2 Test")
                ok = pdfium.raw.FPDFText_SetText(text_obj, text_ws)
                test1["text_set"] = bool(ok)

                # Transform to position
                pdfium.raw.FPDFPageObj_Transform(
                    text_obj,
                    ctypes.c_double(1.0), ctypes.c_double(0.0),
                    ctypes.c_double(0.0), ctypes.c_double(1.0),
                    ctypes.c_double(50.0), ctypes.c_double(height - 100)
                )

                # Insert into page
                pdfium.raw.FPDFPage_InsertObject(page_handle, text_obj)
                pdfium.raw.FPDFPage_GenerateContent(page_handle)

                # Save
                output_pdf = output_path / "standard_font_v2.pdf"
                with open(output_pdf, "wb") as f:
                    pdf.save(f)

                test1["success"] = True
                test1["output"] = str(output_pdf)
            else:
                test1["error"] = "Failed to create text object"
        else:
            test1["error"] = "Failed to load font"

        pdf.close()

    except Exception as e:
        test1["error"] = str(e)
        import traceback
        test1["traceback"] = traceback.format_exc()

    results["tests"].append(test1)
    results["capabilities"]["standard_font"] = test1["success"]

    # Test 2: TrueType font
    test2 = {"name": "truetype_font_text", "success": False}
    try:
        font_path = "src/index_pdf_translation/resources/fonts/LiberationSerif-Regular.ttf"

        if Path(font_path).exists():
            with open(font_path, "rb") as f:
                font_data = f.read()

            pdf = pdfium.PdfDocument.new()
            width, height = 595, 842
            page = pdf.new_page(width, height)

            doc_handle = pdf.raw
            page_handle = page.raw

            # Convert font data to ctypes array
            font_arr = to_byte_array(font_data)

            font_handle = pdfium.raw.FPDFText_LoadFont(
                doc_handle,
                font_arr,
                ctypes.c_uint(len(font_data)),
                ctypes.c_int(pdfium.raw.FPDF_FONT_TRUETYPE),
                ctypes.c_int(0)  # not CID
            )

            if font_handle:
                test2["font_loaded"] = True

                text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(
                    doc_handle,
                    font_handle,
                    ctypes.c_float(14.0)
                )

                if text_obj:
                    text_ws = to_widestring("TrueType Font Test - Liberation Serif")
                    pdfium.raw.FPDFText_SetText(text_obj, text_ws)

                    pdfium.raw.FPDFPageObj_Transform(
                        text_obj,
                        ctypes.c_double(1.0), ctypes.c_double(0.0),
                        ctypes.c_double(0.0), ctypes.c_double(1.0),
                        ctypes.c_double(50.0), ctypes.c_double(height - 100)
                    )

                    pdfium.raw.FPDFPage_InsertObject(page_handle, text_obj)
                    pdfium.raw.FPDFPage_GenerateContent(page_handle)

                    output_pdf = output_path / "truetype_font_v2.pdf"
                    with open(output_pdf, "wb") as f:
                        pdf.save(f)

                    test2["success"] = True
                    test2["output"] = str(output_pdf)
            else:
                test2["error"] = "Failed to load TrueType font"

            pdf.close()
        else:
            test2["error"] = f"Font not found: {font_path}"

    except Exception as e:
        test2["error"] = str(e)
        import traceback
        test2["traceback"] = traceback.format_exc()

    results["tests"].append(test2)
    results["capabilities"]["truetype_font"] = test2["success"]

    # Test 3: Japanese text with CID font
    test3 = {"name": "japanese_cid_font", "success": False}
    try:
        font_path = "src/index_pdf_translation/resources/fonts/ipam.ttf"

        if Path(font_path).exists():
            with open(font_path, "rb") as f:
                font_data = f.read()

            pdf = pdfium.PdfDocument.new()
            width, height = 595, 842
            page = pdf.new_page(width, height)

            doc_handle = pdf.raw
            page_handle = page.raw

            font_arr = to_byte_array(font_data)

            font_handle = pdfium.raw.FPDFText_LoadFont(
                doc_handle,
                font_arr,
                ctypes.c_uint(len(font_data)),
                ctypes.c_int(pdfium.raw.FPDF_FONT_TRUETYPE),
                ctypes.c_int(1)  # CID font for CJK
            )

            if font_handle:
                test3["font_loaded"] = True

                text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(
                    doc_handle,
                    font_handle,
                    ctypes.c_float(14.0)
                )

                if text_obj:
                    # Japanese text
                    text_ws = to_widestring("こんにちは世界")
                    pdfium.raw.FPDFText_SetText(text_obj, text_ws)

                    pdfium.raw.FPDFPageObj_Transform(
                        text_obj,
                        ctypes.c_double(1.0), ctypes.c_double(0.0),
                        ctypes.c_double(0.0), ctypes.c_double(1.0),
                        ctypes.c_double(50.0), ctypes.c_double(height - 100)
                    )

                    pdfium.raw.FPDFPage_InsertObject(page_handle, text_obj)
                    pdfium.raw.FPDFPage_GenerateContent(page_handle)

                    output_pdf = output_path / "japanese_cid_v2.pdf"
                    with open(output_pdf, "wb") as f:
                        pdf.save(f)

                    test3["success"] = True
                    test3["output"] = str(output_pdf)
            else:
                test3["error"] = "Failed to load Japanese CID font"

            pdf.close()
        else:
            test3["error"] = f"Font not found: {font_path}"

    except Exception as e:
        test3["error"] = str(e)
        import traceback
        test3["traceback"] = traceback.format_exc()

    results["tests"].append(test3)
    results["capabilities"]["japanese_cid"] = test3["success"]

    # Test 4: Full workflow - modify existing PDF
    test4 = {"name": "full_workflow", "success": False}
    try:
        pdf_path = "tests/fixtures/sample_llama.pdf"
        FPDF_PAGEOBJ_TEXT = getattr(pdfium.raw, 'FPDF_PAGEOBJ_TEXT', 1)

        pdf = pdfium.PdfDocument(pdf_path)
        page = pdf[0]
        page_height = page.get_height()

        doc_handle = pdf.raw
        page_handle = page.raw

        # Remove title text
        text_objects = list(page.get_objects(filter=[FPDF_PAGEOBJ_TEXT]))
        test4["initial_objects"] = len(text_objects)

        removed = 0
        for obj in text_objects:
            bounds = obj.get_bounds()
            if bounds[1] >= 750:
                page.remove_obj(obj)
                removed += 1

        test4["objects_removed"] = removed

        # Add replacement text
        font_path = "src/index_pdf_translation/resources/fonts/LiberationSerif-Regular.ttf"
        if Path(font_path).exists():
            with open(font_path, "rb") as f:
                font_data = f.read()

            font_arr = to_byte_array(font_data)
            font_handle = pdfium.raw.FPDFText_LoadFont(
                doc_handle,
                font_arr,
                ctypes.c_uint(len(font_data)),
                ctypes.c_int(pdfium.raw.FPDF_FONT_TRUETYPE),
                ctypes.c_int(0)
            )

            if font_handle:
                text_obj = pdfium.raw.FPDFPageObj_CreateTextObj(
                    doc_handle,
                    font_handle,
                    ctypes.c_float(14.0)
                )

                if text_obj:
                    text_ws = to_widestring("[TRANSLATED] Replacement Title")
                    pdfium.raw.FPDFText_SetText(text_obj, text_ws)

                    pdfium.raw.FPDFPageObj_Transform(
                        text_obj,
                        ctypes.c_double(1.0), ctypes.c_double(0.0),
                        ctypes.c_double(0.0), ctypes.c_double(1.0),
                        ctypes.c_double(120.0), ctypes.c_double(page_height - 88)
                    )

                    pdfium.raw.FPDFPage_InsertObject(page_handle, text_obj)
                    test4["text_inserted"] = True

        page.gen_content()

        output_pdf = output_path / "full_workflow_v2.pdf"
        with open(output_pdf, "wb") as f:
            pdf.save(f)

        # Verify
        verify_pdf = pdfium.PdfDocument(str(output_pdf))
        verify_page = verify_pdf[0]
        final_count = len(list(verify_page.get_objects(filter=[FPDF_PAGEOBJ_TEXT])))
        verify_pdf.close()

        test4["final_objects"] = final_count
        test4["success"] = True
        test4["output"] = str(output_pdf)

        pdf.close()

    except Exception as e:
        test4["error"] = str(e)
        import traceback
        test4["traceback"] = traceback.format_exc()

    results["tests"].append(test4)
    results["capabilities"]["full_workflow"] = test4["success"]

    # Summary
    results["summary"] = {
        "all_capabilities": results["capabilities"],
        "pymupdf_replacement_viable": all(results["capabilities"].values())
    }

    return results


def main():
    output_dir = "tests/evaluation/outputs/pypdfium2_verification"

    print("=" * 60)
    print("pypdfium2 Text Insertion Verification v2")
    print("=" * 60)

    results = verify_text_insertion(output_dir)

    # Save
    output_path = Path(output_dir)
    with open(output_path / "text_insertion_v2_results.json", "w") as f:
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
            err = test["error"]
            print(f"      Error: {err[:80]}...")

    summary = results.get("summary", {})
    print(f"\nPyMuPDF Replacement Viable: {'✅' if summary.get('pymupdf_replacement_viable') else '❌'}")

    return results


if __name__ == "__main__":
    main()
