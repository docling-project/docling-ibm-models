import logging

import pytest

from docling_ibm_models.tableformer.data_management.matching_post_processor import (
    MatchingPostProcessor,
)


def _processor() -> MatchingPostProcessor:
    return MatchingPostProcessor({"predict": {"pdf_cell_iou_thres": 0.05}})


@pytest.fixture
def captured_logs():
    """Collect what MatchingPostProcessor logs, whatever handlers it installs."""
    records: list[tuple[str, str]] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append((record.levelname, record.getMessage()))

    logger = logging.getLogger(MatchingPostProcessor.__name__)
    handler = _Collector()
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def test_orphan_cells_outside_every_band_are_reported(captured_logs):
    """An orphan matching neither a row nor a column band is discarded.

    The layout cluster has already claimed its text, so it vanishes from the
    document. That used to happen with no log, no warning and no counter; the
    two nearest-band fallbacks cannot help because there is no band to snap to
    on either axis.
    """
    table_cells = [
        {
            "cell_id": 0,
            "row_id": 0,
            "column_id": 0,
            "bbox": [0, 0, 50, 20],
            "cell_class": 2,
            "label": "body",
        },
    ]
    # Two of the three pdf cells sit far below the only row band (y in [0, 20])
    # AND to the right of the only column band (x in [0, 50]), so neither the
    # nearest-row nor the nearest-column fallback can rescue them.
    pdf_cells = [
        {"id": 0, "bbox": [0, 0, 50, 20], "text": "inside the grid"},
        {"id": 1, "bbox": [200, 200, 250, 220], "text": "below and right"},
        {"id": 2, "bbox": [200, 300, 250, 320], "text": "also below and right"},
    ]
    matches = {"0": [{"iou": 0.9, "table_cell_id": 0}]}

    new_matches, _, _ = _processor()._pick_orphan_cells(
        1, 1, 0, table_cells, pdf_cells, matches
    )

    # The drop itself is unchanged -- this is about making it observable.
    assert sorted(new_matches) == ["0"]

    warnings = [message for level, message in captured_logs if level == "WARNING"]
    assert warnings == [
        "2 of 3 pdf cells matched neither a row nor a column band of the "
        "1x1 grid and were dropped from the table"
    ]


def test_orphan_with_column_band_no_row_band_recovered_to_nearest_row(captured_logs):
    """Mirror of the nearest-column fallback on the row axis.

    A pdf cell that matches a column band but falls below every row band (e.g.
    a trailing row just under the predicted grid, docling issue #3402) is
    snapped to the nearest row instead of being dropped.
    """
    table_cells = [
        {"cell_id": 0, "row_id": 0, "column_id": 0, "label": "body",
         "cell_class": 2, "bbox": [0, 0, 50, 20]},
        {"cell_id": 1, "row_id": 1, "column_id": 0, "label": "body",
         "cell_class": 2, "bbox": [0, 30, 50, 50]},
    ]
    pdf_cells = [
        {"id": 0, "bbox": [0, 30, 50, 50], "text": "matched"},
        # In col 0's x-band (0..50) but below every row band (y=200..220).
        {"id": 9, "bbox": [0, 200, 50, 220], "text": "trailing row"},
    ]
    matches = {"0": [{"iou": 0.9, "table_cell_id": 1}]}

    new_matches, new_table_cells, _ = _processor()._pick_orphan_cells(
        2, 1, 1, table_cells, pdf_cells, matches
    )

    assert "9" in new_matches, "trailing-row orphan was dropped"
    assigned = new_matches["9"][0]
    target = next(
        tc for tc in new_table_cells if tc["cell_id"] == assigned["table_cell_id"]
    )
    # Nearest row by Y-centroid is row 1 (band 30..50) over row 0 (band 0..20).
    assert target["row_id"] == 1
    assert target["column_id"] == 0
    # confidence < 0 marks the snapped fallback.
    assert assigned["post"] < 0

    warnings = [m for level, m in captured_logs if level == "WARNING"]
    assert any("nearest-row fallback" in m for m in warnings)
    # No cell was left unplaceable, so no drop is reported.
    assert not any("were dropped" in m for m in warnings)


def test_no_warning_when_every_orphan_is_assigned(captured_logs):
    """An orphan inside the band is rescued as before, so nothing is reported."""
    table_cells = [
        {
            "cell_id": 0,
            "row_id": 0,
            "column_id": 0,
            "bbox": [0, 0, 50, 20],
            "cell_class": 2,
            "label": "body",
        },
    ]
    pdf_cells = [
        {"id": 0, "bbox": [0, 0, 50, 20], "text": "matched"},
        {"id": 1, "bbox": [0, 5, 50, 18], "text": "orphan inside the band"},
    ]
    matches = {"0": [{"iou": 0.9, "table_cell_id": 0}]}

    new_matches, _, _ = _processor()._pick_orphan_cells(
        1, 1, 0, table_cells, pdf_cells, matches
    )

    assert sorted(new_matches) == ["0", "1"]
    assert [message for level, message in captured_logs if level == "WARNING"] == []
