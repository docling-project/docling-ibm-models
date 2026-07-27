import pytest
from docling_core.types.doc.base import CoordOrigin, Size
from docling_core.types.doc.labels import DocItemLabel

from docling_ibm_models.reading_order.reading_order_rb import (
    PageElement,
    ReadingOrderPredictor,
)


def _list_item(cid: int, text: str, left: float, right: float) -> PageElement:
    return PageElement(
        cid=cid,
        text=text,
        page_no=1,
        page_size=Size(width=100, height=100),
        label=DocItemLabel.LIST_ITEM,
        l=left,
        r=right,
        b=0,
        t=10,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )


@pytest.mark.parametrize("hyphen", ["-", "\u00ad"])
def test_predict_merges_hyphenated_list_item_continuation(hyphen: str) -> None:
    elements = [
        _list_item(0, f"1. Understand the algo{hyphen}", 0, 40),
        _list_item(1, "rithms embedded in clinical imaging devices.", 60, 100),
    ]

    assert ReadingOrderPredictor().predict_merges(elements) == {0: [1]}


@pytest.mark.parametrize(
    ("first_text", "second_text"),
    [
        ("1. First list item", "second list item"),
        ("1. Understand the algo-", "Rithms embedded in clinical imaging devices."),
    ],
)
def test_predict_merges_does_not_join_unrelated_list_items(
    first_text: str, second_text: str
) -> None:
    elements = [
        _list_item(0, first_text, 0, 40),
        _list_item(1, second_text, 60, 100),
    ]

    assert ReadingOrderPredictor().predict_merges(elements) == {}
