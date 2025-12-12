#!/usr/bin/env python3
"""
pypdfium2 Targeted Text Removal Verification Script

This script verifies whether pypdfium2 can:
1. Remove text objects within a specific bounding box region
2. This is the key functionality needed to replace PyMuPDF

License: AGPL-3.0-only
"""

import pypdfium2 as pdfium
import json
from pathlib import Path
from datetime import datetime


def boxes_overlap(box1: tuple, box2: tuple, threshold: float = 0.5) -> bool:
    """
    Check if two bounding boxes overlap significantly.

    Args:
        box1: (left, bottom, right, top) - region to remove
        box2: (left, bottom, right, top) - text object bounds
        threshold: Minimum overlap ratio to consider as match

    Returns:
        bool: True if boxes overlap significantly
    """
    # Unpack boxes
    left1, bottom1, right1, top1 = box1
    left2, bottom2, right2, top2 = box2

    # Calculate intersection
    inter_left = max(left1, left2)
    inter_bottom = max(bottom1, bottom2)
    inter_right = min(right1, right2)
    inter_top = min(top1, top2)

    # Check if there's an intersection
    if inter_left >= inter_right or inter_bottom >= inter_top:
        return False

    # Calculate areas
    inter_area = (inter_right - inter_left) * (inter_top - inter_bottom)
    box2_area = (right2 - left2) * (top2 - bottom2)

    if box2_area == 0:
        return False

    # Check if box2 is mostly within box1
    overlap_ratio = inter_area / box2_area
    return overlap_ratio >= threshold


def targeted_text_removal(
    pdf_path: str,
    output_dir: str,
    target_region: tuple
) -> dict:
    """
    Remove text objects within a specific region.

    Args:
        pdf_path: Path to input PDF
        output_dir: Directory to save outputs
        target_region: (left, bottom, right, top) region to clear

    Returns:
        dict: Removal results
    """
    results = {
        "pdf_path": pdf_path,
        "target_region": target_region,
        "timestamp": datetime.now().isoformat(),
        "removed_objects": [],
        "errors": []
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Get FPDF_PAGEOBJ_TEXT constant
        raw_module = getattr(pdfium, 'raw', None) or getattr(pdfium, 'pdfium_c', pdfium)
        FPDF_PAGEOBJ_TEXT = getattr(raw_module, 'FPDF_PAGEOBJ_TEXT', 1)

        # Open PDF
        pdf = pdfium.PdfDocument(pdf_path)
        page = pdf[0]

        # Get all text objects
        text_objects = list(page.get_objects(filter=[FPDF_PAGEOBJ_TEXT]))
        results["initial_text_objects"] = len(text_objects)

        # Find objects within target region
        objects_to_remove = []
        objects_analyzed = []

        for i, obj in enumerate(text_objects):
            try:
                bounds = obj.get_bounds()
                obj_info = {
                    "index": i,
                    "bounds": bounds,
                    "overlaps_target": boxes_overlap(target_region, bounds)
                }
                objects_analyzed.append(obj_info)

                if obj_info["overlaps_target"]:
                    objects_to_remove.append((obj, bounds))
            except Exception as e:
                results["errors"].append(f"Error analyzing object {i}: {str(e)}")

        results["objects_analyzed"] = len(objects_analyzed)
        results["objects_in_target_region"] = len(objects_to_remove)

        # Remove the objects
        removed_count = 0
        for obj, bounds in objects_to_remove:
            try:
                page.remove_obj(obj)
                results["removed_objects"].append({
                    "bounds": bounds,
                    "status": "removed"
                })
                removed_count += 1
            except Exception as e:
                results["errors"].append(f"Error removing object: {str(e)}")

        results["successfully_removed"] = removed_count

        # Generate content and save
        if removed_count > 0:
            page.gen_content()
            output_pdf_path = output_path / "targeted_removal_output.pdf"
            with open(output_pdf_path, "wb") as f:
                pdf.save(f)
            results["output_pdf"] = str(output_pdf_path)

            # Verify
            verify_pdf = pdfium.PdfDocument(str(output_pdf_path))
            verify_page = verify_pdf[0]
            final_count = len(list(verify_page.get_objects(filter=[FPDF_PAGEOBJ_TEXT])))
            results["final_text_objects"] = final_count
            results["removal_verified"] = (
                results["initial_text_objects"] - final_count == removed_count
            )
            verify_pdf.close()

        pdf.close()

        results["success"] = True

    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Fatal error: {str(e)}")
        import traceback
        results["traceback"] = traceback.format_exc()

    return results


def main():
    pdf_path = "tests/fixtures/sample_llama.pdf"
    output_dir = "tests/evaluation/outputs/pypdfium2_verification"

    print("=" * 60)
    print("pypdfium2 Targeted Text Removal Test")
    print("=" * 60)

    # First, let's see what text objects exist and their bounds
    # We'll target the title area (top of the page)
    # From previous results, title was at [117.54, 753.35, 477.60, 766.20]

    # Target region: Remove text in the title area (top portion of page)
    # PDF coordinates: origin is bottom-left, y increases upward
    # Page height is ~841, so y=700-770 is near the top

    target_region = (100.0, 700.0, 500.0, 780.0)  # Title and author area

    print(f"\nPDF: {pdf_path}")
    print(f"Target region: {target_region}")
    print("  (left, bottom, right, top)")

    results = targeted_text_removal(pdf_path, output_dir, target_region)

    # Save results
    output_path = Path(output_dir)
    with open(output_path / "targeted_removal_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print results
    print(f"\nInitial text objects: {results.get('initial_text_objects', 'N/A')}")
    print(f"Objects in target region: {results.get('objects_in_target_region', 'N/A')}")
    print(f"Successfully removed: {results.get('successfully_removed', 'N/A')}")
    print(f"Final text objects: {results.get('final_text_objects', 'N/A')}")
    print(f"Removal verified: {'✅' if results.get('removal_verified') else '❌'}")

    if results.get("removed_objects"):
        print(f"\nRemoved objects:")
        for i, obj in enumerate(results["removed_objects"][:5]):
            print(f"  {i+1}. bounds: {obj['bounds']}")
        if len(results["removed_objects"]) > 5:
            print(f"  ... and {len(results['removed_objects']) - 5} more")

    if results.get("errors"):
        print(f"\nErrors:")
        for err in results["errors"]:
            print(f"  ⚠️ {err}")

    print(f"\nSuccess: {'✅' if results.get('success') else '❌'}")
    if results.get("output_pdf"):
        print(f"Output PDF: {results['output_pdf']}")

    print(f"\nResults saved to: {output_path}/targeted_removal_results.json")

    return results


if __name__ == "__main__":
    main()
