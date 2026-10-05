import math
import random

import numpy as np

from docling_ibm_models.tableformer.data_management.tf_cell_matcher import (
    CellMatcher,
    find_intersection,
)


def test_find_intersection_overlapping():
    # Two overlapping bboxes ([x1, y1, x2, y2], y1 <= y2) share a rectangle.
    b1 = [0, 0, 10, 10]
    b2 = [5, 5, 15, 15]
    assert find_intersection(b1, b2) == [5, 5, 10, 10]


def test_find_intersection_disjoint_horizontally():
    # b2 sits entirely to the right of b1.
    assert find_intersection([0, 0, 10, 10], [20, 0, 30, 10]) is None
    # b2 sits entirely to the left of b1.
    assert find_intersection([20, 0, 30, 10], [0, 0, 10, 10]) is None


def test_find_intersection_disjoint_vertically():
    # b1 sits entirely below b2 (in bbox coordinates, b1[1] > b2[3]).
    assert find_intersection([0, 20, 10, 30], [0, 0, 10, 10]) is None
    # b2 sits entirely below b1 (b2[1] > b1[3]).  This is the case the buggy
    # ``b2[1] > b2[3]`` check missed: it never fired for a well-formed box, so
    # find_intersection returned a degenerate, y-inverted rectangle instead of
    # None.
    assert find_intersection([0, 0, 10, 10], [0, 20, 10, 30]) is None


def _double_loop_match(table_cells, pdf_cells):
    # CellMatcher._intersection_over_pdf_match before pruning, as a reference
    pdf_bboxes = np.asarray([p["bbox"] for p in pdf_cells])
    pdf_bboxes_areas = (pdf_bboxes[:, 2] - pdf_bboxes[:, 0]) * (
        pdf_bboxes[:, 3] - pdf_bboxes[:, 1]
    )
    matches = {}
    matches_counter = 0
    for table_cell in table_cells:
        for j, pdf_cell in enumerate(pdf_cells):
            i_bbox = find_intersection(table_cell["bbox"], pdf_cell["bbox"])
            if i_bbox is None:
                continue
            i_bbox_area = (i_bbox[2] - i_bbox[0]) * (i_bbox[3] - i_bbox[1])
            iopdf = 0
            if float(pdf_bboxes_areas[j]) > 0:
                iopdf = i_bbox_area / float(pdf_bboxes_areas[j])
            if iopdf > 0:
                match = {"table_cell_id": table_cell["cell_id"], "iopdf": iopdf}
                if pdf_cell["id"] not in matches:
                    matches[pdf_cell["id"]] = [match]
                    matches_counter += 1
                elif match not in matches[pdf_cell["id"]]:
                    matches[pdf_cell["id"]].append(match)
                    matches_counter += 1
    return matches, matches_counter


def _exact(result):
    # Key order, list order and value types, not just ==
    matches, counter = result
    return counter, [
        (k, [(m["table_cell_id"], m["iopdf"], type(m["iopdf"])) for m in v])
        for k, v in matches.items()
    ]


def test_intersection_over_pdf_match_is_unchanged_by_pruning():
    """Touching, inverted, zero-area, duplicate, integer and non-finite boxes."""
    rng = random.Random(20261005)
    matcher = CellMatcher({"predict": {"pdf_cell_iou_thres": 0.05}})

    def coord(scale):
        x = rng.random()
        if x < 0.02:
            return rng.choice([math.nan, math.inf, -math.inf, 0.0, -0.0])
        return rng.randint(0, scale) if x < 0.3 else rng.uniform(0, scale)

    def bbox(scale):
        b = [coord(scale) for _ in range(4)]
        if rng.random() < 0.85:
            b = [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])]
        return b

    for _ in range(2000):
        scale = rng.choice([5, 20, 100])
        pdf_cells = [{"id": i, "bbox": bbox(scale)} for i in range(rng.randint(1, 20))]
        table_cells = [
            {"cell_id": rng.randint(0, 8), "bbox": bbox(scale)}
            for _ in range(rng.randint(0, 20))
        ]
        if table_cells and rng.random() < 0.3:
            table_cells.append(dict(rng.choice(table_cells)))
        assert _exact(
            matcher._intersection_over_pdf_match(table_cells, pdf_cells)
        ) == _exact(_double_loop_match(table_cells, pdf_cells))
