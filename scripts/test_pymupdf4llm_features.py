#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
PyMuPDF4LLM 追加機能テストスクリプト

座標情報保持とヘッダー/フッター除外機能のテスト
"""

from __future__ import annotations

import json
from pathlib import Path


def test_page_chunks_with_boxes(pdf_path: Path) -> dict:
    """
    page_chunks=True で page_boxes の座標情報を確認
    """
    print(f"\n{'='*60}")
    print(f"Test: page_chunks - page_boxes detail")
    print(f"PDF: {pdf_path.name}")
    print('='*60)

    try:
        import pymupdf.layout
    except ImportError:
        print("WARNING: pymupdf-layout not installed")

    import pymupdf4llm

    # page_chunks=True で抽出
    result = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
    )

    print(f"\nResult type: {type(result)}")
    print(f"Number of pages: {len(result)}")

    if result:
        page_data = result[0]
        print(f"\nPage 0 keys: {list(page_data.keys())}")

        # page_boxes の詳細
        if "page_boxes" in page_data:
            boxes = page_data["page_boxes"]
            print(f"\n--- page_boxes (count: {len(boxes)}) ---")
            for i, box in enumerate(boxes[:5]):
                print(f"\n  Box {i}:")
                if isinstance(box, dict):
                    for key, value in box.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"    {key}: '{value[:100]}...'")
                        else:
                            print(f"    {key}: {value}")
                else:
                    print(f"    Type: {type(box)}, Value: {box}")

    return {"success": True}


def test_to_json_detail(pdf_path: Path) -> dict:
    """
    to_json() の詳細な構造を確認
    """
    print(f"\n{'='*60}")
    print(f"Test: to_json() detail structure")
    print(f"PDF: {pdf_path.name}")
    print('='*60)

    try:
        import pymupdf.layout
    except ImportError:
        print("WARNING: pymupdf-layout not installed")

    import pymupdf4llm

    json_result = pymupdf4llm.to_json(str(pdf_path))
    data = json.loads(json_result)

    print(f"\nTop-level keys: {list(data.keys())}")

    # pages の構造
    if "pages" in data:
        pages = data["pages"]
        print(f"\nPages count: {len(pages)}")

        if pages:
            page = pages[0]
            print(f"\nPage 0 keys: {list(page.keys())}")

            # 各キーの詳細
            for key in page:
                value = page[key]
                if isinstance(value, list):
                    print(f"\n  {key}: list (len={len(value)})")
                    if value:
                        first_item = value[0]
                        if isinstance(first_item, dict):
                            print(f"    First item keys: {list(first_item.keys())}")
                            # bbox があるか確認
                            if "bbox" in first_item:
                                print(f"    bbox: {first_item['bbox']}")
                            # 最初のアイテムの詳細を表示
                            for k, v in first_item.items():
                                if isinstance(v, str) and len(v) > 50:
                                    print(f"      {k}: '{v[:50]}...'")
                                else:
                                    print(f"      {k}: {v}")
                        else:
                            print(f"    First item: {first_item}")
                elif isinstance(value, dict):
                    print(f"\n  {key}: dict (keys={list(value.keys())})")
                elif isinstance(value, str) and len(value) > 100:
                    print(f"\n  {key}: str (len={len(value)})")
                else:
                    print(f"\n  {key}: {value}")

    return {"success": True}


def test_extract_words_parameter(pdf_path: Path) -> dict:
    """
    extract_words=True パラメータをテスト
    """
    print(f"\n{'='*60}")
    print(f"Test: extract_words parameter")
    print(f"PDF: {pdf_path.name}")
    print('='*60)

    try:
        import pymupdf.layout
    except ImportError:
        print("WARNING: pymupdf-layout not installed")

    import pymupdf4llm

    # extract_words=True を試す
    try:
        result = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            extract_words=True,
        )

        if result:
            page_data = result[0]
            print(f"Keys with extract_words=True: {list(page_data.keys())}")

            # words キーがあるか確認
            if "words" in page_data:
                words = page_data["words"]
                print(f"\nWords count: {len(words)}")
                if words:
                    print(f"First 3 words:")
                    for w in words[:3]:
                        print(f"  {w}")
            else:
                print("\n'words' key not found in result")

    except TypeError as e:
        print(f"extract_words parameter not supported: {e}")

    return {"success": True}


def test_tables_with_bbox(pdf_path: Path) -> dict:
    """
    tables の bbox 情報を確認
    """
    print(f"\n{'='*60}")
    print(f"Test: tables bbox")
    print(f"PDF: {pdf_path.name}")
    print('='*60)

    try:
        import pymupdf.layout
    except ImportError:
        print("WARNING: pymupdf-layout not installed")

    import pymupdf4llm

    # tables 情報を取得
    result = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
    )

    if result:
        page_data = result[0]

        # tables キーがあるか確認
        if "tables" in page_data:
            tables = page_data["tables"]
            print(f"\nTables count: {len(tables)}")
            for i, table in enumerate(tables[:3]):
                print(f"\nTable {i}:")
                if isinstance(table, dict):
                    for k, v in table.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"  {table}")
        else:
            print("\n'tables' key not found")

        # images キーがあるか確認
        if "images" in page_data:
            images = page_data["images"]
            print(f"\nImages count: {len(images)}")
            for i, img in enumerate(images[:3]):
                print(f"\nImage {i}:")
                if isinstance(img, dict):
                    for k, v in img.items():
                        print(f"  {k}: {v}")

    return {"success": True}


def test_margins_effect(pdf_path: Path) -> dict:
    """
    margins パラメータの効果をより詳細にテスト
    """
    print(f"\n{'='*60}")
    print(f"Test: margins parameter effect (detailed)")
    print(f"PDF: {pdf_path.name}")
    print('='*60)

    try:
        import pymupdf.layout
    except ImportError:
        print("WARNING: pymupdf-layout not installed")

    import pymupdf4llm
    import fitz

    # ページサイズを取得
    doc = fitz.open(pdf_path)
    page = doc[0]
    page_rect = page.rect
    print(f"Page size: {page_rect.width:.1f} x {page_rect.height:.1f}")
    doc.close()

    # 異なるマージンでテスト
    margin_tests = [
        None,  # デフォルト
        (0, 0, 0, 0),  # マージンなし
        (50, 100, 50, 100),  # 大きめのマージン
        (100, 200, 100, 200),  # さらに大きいマージン
    ]

    for margins in margin_tests:
        try:
            if margins is None:
                result = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
                label = "Default (no margins)"
            else:
                result = pymupdf4llm.to_markdown(
                    str(pdf_path),
                    page_chunks=True,
                    margins=margins,
                )
                label = f"margins={margins}"

            if result:
                text = result[0].get("text", "")
                lines = text.strip().split("\n")
                print(f"\n{label}:")
                print(f"  Text length: {len(text)}")
                print(f"  Line count: {len(lines)}")
                print(f"  First line: {lines[0][:60]}..." if lines else "  (empty)")

        except Exception as e:
            print(f"\n{label if 'label' in dir() else margins}: Error - {e}")

    return {"success": True}


def main():
    # テスト用PDF
    test_pdf = Path("tests/fixtures/sample_llama.pdf")

    if not test_pdf.exists():
        test_pdf = Path("tests/pdf_sample/2201.11903v6.pdf")

    if not test_pdf.exists():
        print(f"Error: Test PDF not found")
        return 1

    print(f"Using test PDF: {test_pdf}")

    # テスト1: page_boxes の詳細
    test_page_chunks_with_boxes(test_pdf)

    # テスト2: to_json() の詳細
    test_to_json_detail(test_pdf)

    # テスト3: extract_words パラメータ
    test_extract_words_parameter(test_pdf)

    # テスト4: tables の bbox
    test_tables_with_bbox(test_pdf)

    # テスト5: margins の効果
    test_margins_effect(test_pdf)

    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETE")
    print('='*60)

    return 0


if __name__ == "__main__":
    exit(main())
