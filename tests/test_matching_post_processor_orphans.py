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


def test_orphan_cells_outside_every_row_band_are_reported(captured_logs):
    """Step 9 only rescues an orphan that intersects a row band.

    A pdf cell matching no band at all is discarded, and the layout cluster has
    already claimed its text, so it vanishes from the document. That used to
    happen with no log, no warning and no counter.
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
    # Two of the three pdf cells sit far below the only row band (y in [0, 20]).
    pdf_cells = [
        {"id": 0, "bbox": [0, 0, 50, 20], "text": "inside the grid"},
        {"id": 1, "bbox": [0, 200, 50, 220], "text": "below the grid"},
        {"id": 2, "bbox": [0, 300, 50, 320], "text": "also below the grid"},
    ]
    matches = {"0": [{"iou": 0.9, "table_cell_id": 0}]}

    new_matches, _, _ = _processor()._pick_orphan_cells(
        1, 1, 0, table_cells, pdf_cells, matches
    )

    # The drop itself is unchanged -- this is about making it observable.
    assert sorted(new_matches) == ["0"]

    warnings = [message for level, message in captured_logs if level == "WARNING"]
    assert warnings == [
        "2 of 3 pdf cells matched no row band of the 1x1 grid and "
        "were dropped from the table"
    ]


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
