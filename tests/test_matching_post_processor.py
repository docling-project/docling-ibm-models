"""Unit tests for `MatchingPostProcessor._pick_orphan_cells`.

Specifically the silent-drop case where a PDF text cell falls inside a row
band but outside every column band: prior to the nearest-column fallback,
such cells were dropped on the floor without warning. Now they are snapped
to the closest column by X-centroid distance.

See https://github.com/docling-project/docling-ibm-models/issues/28.
"""

from docling_ibm_models.tableformer.data_management.matching_post_processor import (
    MatchingPostProcessor,
)


def _make_proc():
    # `_pick_orphan_cells` does not exercise the cell matcher; a stub config
    # is sufficient. CellMatcher init reads `pdf_cell_iou_thres`.
    return MatchingPostProcessor({"predict": {"pdf_cell_iou_thres": 0.05}})


def test_orphan_with_no_col_band_match_is_recovered_to_nearest_column():
    """A pdf cell inside the row band but outside every column band must
    still produce a match — historically it was silently dropped."""
    proc = _make_proc()

    # Two predicted columns whose x-bands lie at x≈10..200, in two rows.
    # The orphan pdf cell sits in row 1's y-band but its x=560 is outside
    # both column x-bands — exactly the silent-drop case.
    table_cells = [
        {"cell_id": 0, "row_id": 0, "column_id": 0, "label": "body",
         "cell_class": 2, "bbox": [10, 10, 90, 25]},
        {"cell_id": 1, "row_id": 0, "column_id": 1, "label": "body",
         "cell_class": 2, "bbox": [110, 10, 200, 25]},
        {"cell_id": 2, "row_id": 1, "column_id": 0, "label": "body",
         "cell_class": 2, "bbox": [10, 30, 90, 45]},
        {"cell_id": 3, "row_id": 1, "column_id": 1, "label": "body",
         "cell_class": 2, "bbox": [110, 30, 200, 45]},
    ]
    pdf_cells = [
        # In row 1's y band (30..45) but x=560 — outside col 0 (10..90)
        # and col 1 (110..200). Pre-fix this gets dropped silently.
        {"id": 99, "bbox": [550, 32, 590, 43], "text": "$4,129.51"},
    ]
    matches: dict = {}  # the orphan is unmatched at entry

    new_matches, new_table_cells, _max_cell_id = proc._pick_orphan_cells(
        tab_rows=2,
        tab_cols=2,
        max_cell_id=3,
        table_cells=table_cells,
        pdf_cells=pdf_cells,
        matches=matches,
    )

    # After the fix: the orphan must appear in new_matches assigned to a
    # column (the nearest-centroid one — column 1 at centroid x=155 vs
    # column 0 at centroid x=50).
    assert "99" in new_matches, (
        "orphan pdf_cell was silently dropped (no match emitted)"
    )
    assigned = new_matches["99"][0]
    target_table_cell = next(
        tc for tc in new_table_cells if tc["cell_id"] == assigned["table_cell_id"]
    )
    assert target_table_cell["column_id"] == 1, (
        f"expected nearest-column fallback to col 1 (centroid 155), "
        f"got col {target_table_cell['column_id']}"
    )
    # confidence is negative for nearest-column fallback.
    assert assigned["post"] < 0, (
        f"nearest-column fallback should mark confidence < 0, got {assigned['post']}"
    )


def test_band_matched_orphans_use_normal_path():
    """The fast-path (orphan inside a column band) must be unchanged."""
    proc = _make_proc()

    # Two rows of body cells. Orphan pdf cell sits in row 1's y-band AND
    # within column 0's x-band — the existing happy path.
    table_cells = [
        {"cell_id": 0, "row_id": 0, "column_id": 0, "label": "body",
         "cell_class": 2, "bbox": [10, 10, 90, 25]},
        {"cell_id": 1, "row_id": 0, "column_id": 1, "label": "body",
         "cell_class": 2, "bbox": [110, 10, 200, 25]},
        {"cell_id": 2, "row_id": 1, "column_id": 0, "label": "body",
         "cell_class": 2, "bbox": [10, 30, 90, 45]},
        {"cell_id": 3, "row_id": 1, "column_id": 1, "label": "body",
         "cell_class": 2, "bbox": [110, 30, 200, 45]},
    ]
    pdf_cells = [
        # In col 0's x-band (10..90) AND row 1's y-band (30..45).
        {"id": 7, "bbox": [20, 32, 80, 43], "text": "in band"},
    ]
    matches: dict = {}
    new_matches, new_table_cells, _ = proc._pick_orphan_cells(
        tab_rows=2, tab_cols=2, max_cell_id=3,
        table_cells=table_cells, pdf_cells=pdf_cells, matches=matches,
    )
    assert "7" in new_matches
    assigned = new_matches["7"][0]
    target = next(tc for tc in new_table_cells if tc["cell_id"] == assigned["table_cell_id"])
    assert target["column_id"] == 0, "in-band orphan should land in col 0"
    # Normal-path confidence is non-negative.
    assert assigned["post"] >= 0, (
        f"in-band match should have non-negative confidence, got {assigned['post']}"
    )
