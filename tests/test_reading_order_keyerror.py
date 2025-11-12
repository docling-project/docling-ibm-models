"""
Unit test for KeyError fix in reading order prediction.

This test verifies that the reading order predictor handles edge cases
where l2r_map references may lead to invalid indices.
"""

import pytest
from docling_core.types.doc.base import CoordOrigin, Size
from docling_core.types.doc.labels import DocItemLabel

from docling_ibm_models.reading_order.reading_order_rb import (
    PageElement,
    ReadingOrderPredictor,
)


def test_reading_order_no_keyerror():
    """Test that reading order predictor doesn't raise KeyError on edge cases."""
    # Create a simple page with a few elements
    romodel = ReadingOrderPredictor()

    page_size = Size(width=612.0, height=792.0)

    # Create simple test elements in a column layout
    elements = [
        PageElement(
            cid=0,
            text="Title",
            page_no=0,
            page_size=page_size,
            label=DocItemLabel.TITLE,
            l=100.0,
            r=500.0,
            b=700.0,
            t=750.0,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        PageElement(
            cid=1,
            text="Paragraph 1",
            page_no=0,
            page_size=page_size,
            label=DocItemLabel.TEXT,
            l=100.0,
            r=500.0,
            b=600.0,
            t=680.0,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        PageElement(
            cid=2,
            text="Paragraph 2",
            page_no=0,
            page_size=page_size,
            label=DocItemLabel.TEXT,
            l=100.0,
            r=500.0,
            b=500.0,
            t=580.0,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
    ]

    # This should not raise KeyError
    result = romodel.predict_reading_order(page_elements=elements)

    # Verify we got all elements back
    assert len(result) == len(elements)

    # Verify elements are PageElement instances
    for elem in result:
        assert isinstance(elem, PageElement)


def test_reading_order_multi_column():
    """Test reading order with multi-column layout."""
    romodel = ReadingOrderPredictor()

    page_size = Size(width=612.0, height=792.0)

    # Create elements in two columns
    elements = [
        # Left column
        PageElement(
            cid=0,
            text="Left column top",
            page_no=0,
            page_size=page_size,
            label=DocItemLabel.TEXT,
            l=50.0,
            r=280.0,
            b=700.0,
            t=750.0,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        PageElement(
            cid=1,
            text="Left column middle",
            page_no=0,
            page_size=page_size,
            label=DocItemLabel.TEXT,
            l=50.0,
            r=280.0,
            b=600.0,
            t=680.0,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        # Right column
        PageElement(
            cid=2,
            text="Right column top",
            page_no=0,
            page_size=page_size,
            label=DocItemLabel.TEXT,
            l=332.0,
            r=562.0,
            b=700.0,
            t=750.0,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        PageElement(
            cid=3,
            text="Right column middle",
            page_no=0,
            page_size=page_size,
            label=DocItemLabel.TEXT,
            l=332.0,
            r=562.0,
            b=600.0,
            t=680.0,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
    ]

    # This should not raise KeyError
    result = romodel.predict_reading_order(page_elements=elements)

    # Verify we got all elements back
    assert len(result) == len(elements)


def test_reading_order_with_headers_and_footers():
    """Test reading order with headers and footers."""
    romodel = ReadingOrderPredictor()

    page_size = Size(width=612.0, height=792.0)

    # Create elements with headers and footers
    elements = [
        PageElement(
            cid=0,
            text="Header",
            page_no=0,
            page_size=page_size,
            label=DocItemLabel.PAGE_HEADER,
            l=100.0,
            r=500.0,
            b=750.0,
            t=780.0,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        PageElement(
            cid=1,
            text="Body text",
            page_no=0,
            page_size=page_size,
            label=DocItemLabel.TEXT,
            l=100.0,
            r=500.0,
            b=400.0,
            t=700.0,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        PageElement(
            cid=2,
            text="Footer",
            page_no=0,
            page_size=page_size,
            label=DocItemLabel.PAGE_FOOTER,
            l=100.0,
            r=500.0,
            b=10.0,
            t=30.0,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
    ]

    # This should not raise KeyError
    result = romodel.predict_reading_order(page_elements=elements)

    # Verify we got all elements back
    assert len(result) == len(elements)

    # Verify order: header, body, footer
    assert result[0].label == DocItemLabel.PAGE_HEADER
    assert result[1].label == DocItemLabel.TEXT
    assert result[2].label == DocItemLabel.PAGE_FOOTER
