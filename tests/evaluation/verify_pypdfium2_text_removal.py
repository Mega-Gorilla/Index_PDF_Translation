#!/usr/bin/env python3
"""
pypdfium2 Text Removal Verification Script

This script verifies whether pypdfium2 can:
1. Iterate through page objects (specifically text objects)
2. Get bounding boxes of text objects
3. Remove text objects from a page
4. Save the modified PDF

License: AGPL-3.0-only
"""

import pypdfium2 as pdfium
import json
from pathlib import Path
from datetime import datetime


def verify_text_removal(pdf_path: str, output_dir: str) -> dict:
    """
    Verify pypdfium2's text removal capabilities.

    Args:
        pdf_path: Path to input PDF
        output_dir: Directory to save outputs

    Returns:
        dict: Verification results
    """
    # Get version info safely
    try:
        pypdfium2_ver = getattr(pdfium, 'V_PYPDFIUM2', None) or getattr(pdfium, '__version__', 'unknown')
        pdfium_ver = getattr(pdfium, 'V_LIBPDFIUM', None) or getattr(pdfium, 'V_PDFIUM', 'unknown')
    except Exception:
        pypdfium2_ver = "unknown"
        pdfium_ver = "unknown"

    results = {
        "pdf_path": pdf_path,
        "pypdfium2_version": str(pypdfium2_ver),
        "pdfium_version": str(pdfium_ver),
        "verification_timestamp": datetime.now().isoformat(),
        "capabilities": {},
        "errors": [],
        "page_analyses": []
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Open PDF
        pdf = pdfium.PdfDocument(pdf_path)
        results["total_pages"] = len(pdf)

        # Check page object constants - try different attribute paths
        try:
            # Try pdfium.raw first, then pdfium_c
            raw_module = getattr(pdfium, 'raw', None) or getattr(pdfium, 'pdfium_c', pdfium)
            FPDF_PAGEOBJ_TEXT = getattr(raw_module, 'FPDF_PAGEOBJ_TEXT', 1)
            FPDF_PAGEOBJ_PATH = getattr(raw_module, 'FPDF_PAGEOBJ_PATH', 2)
            FPDF_PAGEOBJ_IMAGE = getattr(raw_module, 'FPDF_PAGEOBJ_IMAGE', 3)
            FPDF_PAGEOBJ_SHADING = getattr(raw_module, 'FPDF_PAGEOBJ_SHADING', 4)
            FPDF_PAGEOBJ_FORM = getattr(raw_module, 'FPDF_PAGEOBJ_FORM', 5)
        except Exception:
            FPDF_PAGEOBJ_TEXT = 1
            FPDF_PAGEOBJ_PATH = 2
            FPDF_PAGEOBJ_IMAGE = 3
            FPDF_PAGEOBJ_SHADING = 4
            FPDF_PAGEOBJ_FORM = 5

        results["object_type_constants"] = {
            "FPDF_PAGEOBJ_TEXT": FPDF_PAGEOBJ_TEXT,
            "FPDF_PAGEOBJ_PATH": FPDF_PAGEOBJ_PATH,
            "FPDF_PAGEOBJ_IMAGE": FPDF_PAGEOBJ_IMAGE,
            "FPDF_PAGEOBJ_SHADING": FPDF_PAGEOBJ_SHADING,
            "FPDF_PAGEOBJ_FORM": FPDF_PAGEOBJ_FORM,
        }

        # Analyze first page
        page = pdf[0]
        page_analysis = {
            "page_num": 0,
            "page_size": (page.get_width(), page.get_height()),
            "objects_by_type": {},
            "text_objects": [],
            "text_objects_with_bounds": []
        }

        # Get all objects
        all_objects = list(page.get_objects())
        page_analysis["total_objects"] = len(all_objects)

        # Count by type
        for obj in all_objects:
            type_name = f"type_{obj.type}"
            if type_name not in page_analysis["objects_by_type"]:
                page_analysis["objects_by_type"][type_name] = 0
            page_analysis["objects_by_type"][type_name] += 1

        # Get text objects specifically
        text_objects = list(page.get_objects(filter=[FPDF_PAGEOBJ_TEXT]))
        page_analysis["text_object_count"] = len(text_objects)
        results["capabilities"]["can_filter_text_objects"] = len(text_objects) > 0

        # Try to get bounds for text objects
        bounds_success = 0
        for i, obj in enumerate(text_objects[:10]):  # First 10 text objects
            try:
                bounds = obj.get_bounds()
                page_analysis["text_objects_with_bounds"].append({
                    "index": i,
                    "bounds": bounds,  # (left, bottom, right, top)
                    "type": obj.type
                })
                bounds_success += 1
            except Exception as e:
                results["errors"].append(f"get_bounds error for object {i}: {str(e)}")

        results["capabilities"]["can_get_text_bounds"] = bounds_success > 0
        page_analysis["bounds_retrieved"] = bounds_success

        results["page_analyses"].append(page_analysis)

        # Now test actual removal
        if len(text_objects) > 0:
            # Create a copy for testing removal
            # Re-open the PDF for modification
            pdf_for_edit = pdfium.PdfDocument(pdf_path)
            page_for_edit = pdf_for_edit[0]

            # Get text objects again
            text_objs_for_removal = list(page_for_edit.get_objects(
                filter=[FPDF_PAGEOBJ_TEXT]
            ))

            initial_count = len(text_objs_for_removal)
            removed_count = 0
            removal_errors = []

            # Try to remove first 3 text objects (if available)
            objects_to_remove = text_objs_for_removal[:min(3, len(text_objs_for_removal))]

            # Close any text page handles first (as documented)
            # Note: We haven't opened any textpage, so this should be fine

            for obj in objects_to_remove:
                try:
                    page_for_edit.remove_obj(obj)
                    removed_count += 1
                except Exception as e:
                    removal_errors.append(str(e))

            results["capabilities"]["can_remove_text_objects"] = removed_count > 0
            results["removal_test"] = {
                "initial_text_objects": initial_count,
                "attempted_removals": len(objects_to_remove),
                "successful_removals": removed_count,
                "errors": removal_errors
            }

            # Generate content and save
            if removed_count > 0:
                try:
                    page_for_edit.gen_content()
                    results["capabilities"]["can_gen_content"] = True

                    # Save modified PDF
                    output_pdf_path = output_path / "modified_output.pdf"
                    with open(output_pdf_path, "wb") as f:
                        pdf_for_edit.save(f)
                    results["capabilities"]["can_save_modified_pdf"] = True
                    results["output_pdf"] = str(output_pdf_path)

                    # Verify the saved PDF
                    verify_pdf = pdfium.PdfDocument(str(output_pdf_path))
                    verify_page = verify_pdf[0]
                    new_text_count = len(list(verify_page.get_objects(
                        filter=[FPDF_PAGEOBJ_TEXT]
                    )))
                    results["verification"] = {
                        "original_text_objects": initial_count,
                        "after_removal_text_objects": new_text_count,
                        "difference": initial_count - new_text_count,
                        "removal_verified": new_text_count < initial_count
                    }
                    verify_pdf.close()

                except Exception as e:
                    results["errors"].append(f"gen_content/save error: {str(e)}")
                    results["capabilities"]["can_gen_content"] = False
                    results["capabilities"]["can_save_modified_pdf"] = False

            pdf_for_edit.close()

        pdf.close()

        # Summary
        results["summary"] = {
            "text_removal_possible": all([
                results["capabilities"].get("can_filter_text_objects", False),
                results["capabilities"].get("can_get_text_bounds", False),
                results["capabilities"].get("can_remove_text_objects", False),
                results["capabilities"].get("can_gen_content", False),
                results["capabilities"].get("can_save_modified_pdf", False),
            ]),
            "all_capabilities": results["capabilities"]
        }

    except Exception as e:
        results["errors"].append(f"Fatal error: {str(e)}")
        import traceback
        results["traceback"] = traceback.format_exc()

    return results


def main():
    # Use sample PDF
    pdf_path = "tests/fixtures/sample_llama.pdf"
    output_dir = "tests/evaluation/outputs/pypdfium2_verification"

    print("=" * 60)
    print("pypdfium2 Text Removal Verification")
    print("=" * 60)

    results = verify_text_removal(pdf_path, output_dir)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "verification_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print(f"\nPDF: {results.get('pdf_path')}")
    print(f"pypdfium2 version: {results.get('pypdfium2_version')}")
    print(f"PDFium version: {results.get('pdfium_version')}")
    print(f"\nCapabilities:")
    for cap, value in results.get("capabilities", {}).items():
        status = "✅" if value else "❌"
        print(f"  {status} {cap}: {value}")

    if results.get("verification"):
        print(f"\nVerification:")
        v = results["verification"]
        print(f"  Original text objects: {v['original_text_objects']}")
        print(f"  After removal: {v['after_removal_text_objects']}")
        print(f"  Removal verified: {'✅' if v['removal_verified'] else '❌'}")

    if results.get("errors"):
        print(f"\nErrors:")
        for err in results["errors"]:
            print(f"  ⚠️ {err}")

    print(f"\nSummary: Text removal {'POSSIBLE' if results.get('summary', {}).get('text_removal_possible') else 'NOT POSSIBLE'}")
    print(f"\nResults saved to: {output_path}/verification_results.json")

    return results


if __name__ == "__main__":
    main()
