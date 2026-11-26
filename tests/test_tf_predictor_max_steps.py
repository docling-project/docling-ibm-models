import logging
from unittest.mock import patch

import pytest

import docling_ibm_models.tableformer.data_management.tf_predictor as _tfp
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor

START, END, FCEL, NL = 2, 3, 5, 9
MAX_STEPS = 1024


def _predictor() -> TFPredictor:
    r"""
    A TFPredictor without the model: only what _flag_unterminated reads
    """
    predictor = TFPredictor.__new__(TFPredictor)
    predictor._word_map = {
        "word_map_tag": {"<start>": START, "<end>": END, "fcel": FCEL, "nl": NL}
    }
    predictor._config = {"predict": {"max_steps": MAX_STEPS}}
    return predictor


@pytest.fixture
def captured_logs():
    r"""
    Collect what TFPredictor logs, whatever handlers it installs
    """
    records: list[tuple[str, str]] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append((record.levelname, record.getMessage()))

    logger = logging.getLogger(TFPredictor.__name__)
    handler = _Collector()
    logger.addHandler(handler)
    with (
        patch.object(_tfp, "LOG_LEVEL", logging.WARNING),
        patch.object(TFPredictor, "_logger", None),
    ):
        try:
            yield records
        finally:
            logger.removeHandler(handler)


def test_reached_max_steps_reads_the_last_tag():
    assert not TFPredictor._reached_max_steps([START, FCEL, NL, END], END)
    assert TFPredictor._reached_max_steps([START, FCEL, NL, FCEL], END)
    assert not TFPredictor._reached_max_steps([], END)


def test_unterminated_sequence_is_flagged_and_reported(captured_logs):
    r"""
    The tag decoder stops silently at max_steps when it never emits "<end>".

    Such a sequence is a row pattern repeated until the cap, not a table: see
    docling-project/docling#4255, where it made a table lose its data rows with
    nothing in the logs and nothing in the prediction to tell it apart.
    """
    # <start> followed by max_steps tags, none of them "<end>"
    tag_seq = [START] + [FCEL, FCEL, NL] * (MAX_STEPS // 3) + [FCEL]
    assert len(tag_seq) == MAX_STEPS + 1
    prediction = {"tag_seq": tag_seq}

    assert _predictor()._flag_unterminated(prediction)

    assert prediction["reached_max_steps"] is True
    warnings = [message for level, message in captured_logs if level == "WARNING"]
    assert warnings == [
        "TableFormer reached max_steps (1024) without an end token: "
        "the predicted table structure is likely degenerate"
    ]


def test_terminated_sequence_is_not_reported(captured_logs):
    prediction = {"tag_seq": [START, FCEL, FCEL, NL, FCEL, FCEL, NL, END]}

    assert not _predictor()._flag_unterminated(prediction)

    assert prediction["reached_max_steps"] is False
    assert not [m for level, m in captured_logs if level == "WARNING"]
