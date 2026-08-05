import pytest

from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor


def _cell(start_col, start_row, col_span=1, row_span=1):
    return {
        "start_col_offset_idx": start_col,
        "start_row_offset_idx": start_row,
        "col_span": col_span,
        "row_span": row_span,
    }


def _issue_123_cells():
    r"""
    The table reported in docling-ibm-models#123.

    TableFormer predicts 6 columns and 7 rows. Column ID 4 is covered by a
    single empty "ched" that starts on row 1 and spans the remaining 6 rows;
    the CellMatcher drops it (and its "ucel" continuations) because it has no
    text, so no cell starts at column ID 4 anymore. The header on row 0 spans
    all 6 predicted columns.
    """
    cells = [_cell(0, 0, col_span=6)]
    for col in (0, 1, 2, 3, 5):
        cells.append(_cell(col, 1))
    for row in range(2, 7):
        for col in (0, 1, 2, 3, 5):
            cells.append(_cell(col, row))
    return cells


def test_compress_row_col_indexes_does_not_count_dropped_columns():
    cells = _issue_123_cells()

    num_cols, num_rows = TFPredictor._compress_row_col_indexes(cells)

    # Column ID 4 held only empty cells, so the table has 5 columns, not 6.
    assert num_cols == 5
    assert num_rows == 7

    header = cells[0]
    assert header["start_col_offset_idx"] == 0
    assert header["end_col_offset_idx"] == 5
    assert header["col_span"] == 5

    # No cell may point past the compressed table.
    for cell in cells:
        assert cell["end_col_offset_idx"] <= num_cols
        assert cell["end_row_offset_idx"] <= num_rows


def test_compress_row_col_indexes_does_not_count_dropped_rows():
    r"""Same defect on the row axis: row ID 2 has no surviving cell."""
    cells = [_cell(0, 0, row_span=4), _cell(1, 0), _cell(1, 1), _cell(1, 3)]

    num_cols, num_rows = TFPredictor._compress_row_col_indexes(cells)

    assert num_cols == 2
    assert num_rows == 3

    spanning = cells[0]
    assert spanning["start_row_offset_idx"] == 0
    assert spanning["end_row_offset_idx"] == 3
    assert spanning["row_span"] == 3


@pytest.mark.parametrize("col_span", [1, 2, 3])
@pytest.mark.parametrize("row_span", [1, 2, 3])
def test_compress_row_col_indexes_keeps_gap_free_tables_intact(col_span, row_span):
    r"""A table without dropped rows/columns must survive unchanged."""
    cells = [_cell(col, row) for col in range(3) for row in range(3)]
    cells[0]["col_span"] = col_span
    cells[0]["row_span"] = row_span

    num_cols, num_rows = TFPredictor._compress_row_col_indexes(cells)

    assert (num_cols, num_rows) == (3, 3)
    for cell in cells:
        assert cell["end_col_offset_idx"] == (
            cell["start_col_offset_idx"] + cell["col_span"]
        )
        assert cell["end_row_offset_idx"] == (
            cell["start_row_offset_idx"] + cell["row_span"]
        )
