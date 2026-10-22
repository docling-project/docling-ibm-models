"""Unit tests for a column whose predicted cells are all empty-class.

TableFormer marks a cell empty when the image shows nothing there. The PDF text
layer can still hold a word in it, for example a Type3 emoji glyph the page
render left out. Such a column has no good cell to align to. It was moved to
x=0, its words fell to the nearest other column, and the column was lost.

See https://github.com/docling-project/docling/issues/4922.
"""

from docling_ibm_models.tableformer.data_management.matching_post_processor import (
    MatchingPostProcessor,
)


def _cell(cell_id, row, col, bbox, cell_class):
    return {
        "cell_id": cell_id,
        "row_id": row,
        "column_id": col,
        "label": "fcel" if cell_class > 1 else "ecel",
        "cell_class": cell_class,
        "bbox": bbox,
    }


def _table_cells(right_x0=170):
    # Two rows of: a name, an empty-class middle cell and a value.
    cells = []
    for row, (y0, y1) in enumerate([(10, 30), (40, 60)]):
        cells.append(_cell(3 * row, row, 0, [10, y0, 90, y1], 2))
        cells.append(_cell(3 * row + 1, row, 1, [110, y0, 150, y1], 1))
        cells.append(_cell(3 * row + 2, row, 2, [right_x0, y0, 250, y1], 2))
    return cells


def _process(table_cells, pdf_cells):
    proc = MatchingPostProcessor({"predict": {"pdf_cell_iou_thres": 0.05}})
    matches, _ = proc._cell_matcher._intersection_over_pdf_match(table_cells, pdf_cells)
    details = proc.process(
        {"table_cells": table_cells, "pdf_cells": pdf_cells, "matches": matches}
    )
    cells_by_id = {cell["cell_id"]: cell for cell in details["table_cells"]}
    columns = {}
    for pdf_cell_id, match_list in details["matches"].items():
        cell = cells_by_id[match_list[0]["table_cell_id"]]
        columns[str(pdf_cell_id)] = (cell["row_id"], cell["column_id"])
    return columns, details["table_cells"]


def test_words_in_an_empty_class_column_keep_their_column():
    pdf_cells = [
        {"id": 0, "bbox": [12, 12, 60, 28], "text": "Alice"},
        {"id": 1, "bbox": [120, 12, 140, 28], "text": "✔"},
        {"id": 2, "bbox": [172, 12, 220, 28], "text": "team"},
        {"id": 3, "bbox": [12, 42, 50, 58], "text": "Bob"},
        {"id": 4, "bbox": [120, 42, 140, 58], "text": "✔"},
        {"id": 5, "bbox": [172, 42, 210, 58], "text": "unit"},
    ]

    columns, _ = _process(_table_cells(), pdf_cells)

    assert columns == {
        "0": (0, 0),
        "1": (0, 1),
        "2": (0, 2),
        "3": (1, 0),
        "4": (1, 1),
        "5": (1, 2),
    }


def test_empty_class_column_without_words_is_still_dropped():
    pdf_cells = [
        {"id": 0, "bbox": [12, 12, 60, 28], "text": "Alice"},
        {"id": 2, "bbox": [172, 12, 220, 28], "text": "team"},
        {"id": 3, "bbox": [12, 42, 50, 58], "text": "Bob"},
        {"id": 5, "bbox": [172, 42, 210, 58], "text": "unit"},
    ]

    columns, table_cells = _process(_table_cells(), pdf_cells)

    assert columns == {"0": (0, 0), "2": (0, 2), "3": (1, 0), "5": (1, 2)}
    assert all(cell["column_id"] != 1 for cell in table_cells)


def test_word_shared_with_a_good_cell_goes_to_the_good_cell():
    # The value column overlaps the empty-class column, so the empty-class cell
    # is not the only match of the value word and does not keep its place.
    pdf_cells = [
        {"id": 0, "bbox": [12, 12, 60, 28], "text": "Alice"},
        {"id": 2, "bbox": [145, 12, 220, 28], "text": "team"},
        {"id": 3, "bbox": [12, 42, 50, 58], "text": "Bob"},
        {"id": 5, "bbox": [145, 42, 210, 58], "text": "unit"},
    ]

    columns, table_cells = _process(_table_cells(right_x0=140), pdf_cells)

    assert columns == {"0": (0, 0), "2": (0, 2), "3": (1, 0), "5": (1, 2)}
    assert all(cell["column_id"] != 1 for cell in table_cells)
