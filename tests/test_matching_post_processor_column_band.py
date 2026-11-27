"""Unit tests for `MatchingPostProcessor._reassign_by_column_band` (step 8.b).

The final assignment gives every pdf cell to the predicted table cell with
the largest intersection, whatever that intersection is and wherever the
predicted bbox sits. When TableFormer shifts one bbox into the neighbouring
column, or predicts two bboxes on the same spot, a pdf cell lands in a
column that no other row of the table would put it in. Step 8.b checks each
assigned pdf cell against the x band of its column, built from the
unambiguous pdf cells of the other rows, and moves it when another column's
band is where it belongs.

The coordinates come from the two documents in
https://github.com/docling-project/docling/issues/4728 (144 dpi page space).
"""

from docling_ibm_models.tableformer.data_management.matching_post_processor import (
    MatchingPostProcessor,
)


def _processor():
    # Neither step reads the model config beyond the CellMatcher threshold.
    return MatchingPostProcessor({"predict": {"pdf_cell_iou_thres": 0.05}})


def _cell(cell_id, row_id, column_id, bbox, label="body", **extra):
    cell = {
        "cell_id": cell_id,
        "row_id": row_id,
        "column_id": column_id,
        "bbox": list(bbox),
        "label": label,
        "cell_class": 2,
    }
    cell.update(extra)
    return cell


def _pdf(pdf_id, bbox, text):
    return {"id": pdf_id, "bbox": list(bbox), "text": text}


def _column_of(table_cells, matches, pdf_id):
    cell_id = matches[str(pdf_id)][0]["table_cell_id"]
    return next(tc for tc in table_cells if tc["cell_id"] == cell_id)["column_id"]


def test_value_under_a_shifted_bbox_moves_to_the_column_that_holds_it():
    """Belastingdienst table, row 1: the predicted bbox of column 2 is shifted
    left and takes 31 % of `3,05`; the column-1 bbox does not touch it. With
    no minimum, `3,05` goes to column 2 (the `Maximumbedrag per jaar` column)
    and the cell reads `3,05 € 6.000`. The header and row 2 put column 1 at
    x 522..601, which contains `3,05` completely."""
    proc = _processor()
    table_cells = [
        _cell(1, 0, 1, [522, 557, 596, 611], label="ched"),
        _cell(2, 0, 2, [613, 557, 688, 611], label="ched"),
        _cell(6, 1, 1, [522, 663, 561, 681]),
        _cell(7, 1, 2, [592, 664, 675, 682]),
        _cell(11, 2, 1, [517, 729, 593, 747]),
        _cell(12, 2, 2, [612, 730, 694, 748]),
    ]
    pdf_cells = [
        _pdf(92, [522, 557, 570, 571], "Bedrag"),
        _pdf(96, [613, 557, 688, 571], "Maximum-"),
        _pdf(128, [522, 667, 533, 681], "€"),
        _pdf(129, [573, 667, 601, 681], "3,05"),
        _pdf(130, [613, 667, 624, 681], "€"),
        _pdf(131, [655, 667, 692, 681], "6.000"),
        _pdf(179, [522, 732, 533, 746], "€"),
        _pdf(180, [573, 732, 601, 746], "1,01"),
        _pdf(181, [613, 732, 624, 746], "€"),
        _pdf(182, [655, 732, 692, 746], "2.000"),
    ]
    candidate_matches = {
        "92": [{"table_cell_id": 1, "iopdf": 1.0}],
        "96": [{"table_cell_id": 2, "iopdf": 0.891}],
        "128": [{"table_cell_id": 6, "iopdf": 0.992}],
        "129": [{"table_cell_id": 7, "iopdf": 0.308}],
        "130": [{"table_cell_id": 7, "iopdf": 1.0}],
        "131": [{"table_cell_id": 7, "iopdf": 0.534}],
        "179": [{"table_cell_id": 11, "iopdf": 1.0}],
        "180": [{"table_cell_id": 11, "iopdf": 0.696}],
        "181": [{"table_cell_id": 12, "iopdf": 1.0}],
        "182": [{"table_cell_id": 12, "iopdf": 1.0}],
    }
    final_matches = {k: [dict(v[0])] for k, v in candidate_matches.items()}

    new_matches = proc._reassign_by_column_band(
        table_cells, pdf_cells, final_matches, candidate_matches
    )

    assert _column_of(table_cells, new_matches, 129) == 1
    assert new_matches["129"][0]["table_cell_id"] == 6
    # The move is marked, and keeps the original score for the record.
    assert new_matches["129"][0]["band"] > 0.8
    assert new_matches["129"][0]["iopdf"] == 0.308
    # Nothing else moves.
    for pdf_id in (92, 96, 128, 130, 131, 179, 180, 181, 182):
        assert new_matches[str(pdf_id)] == final_matches[str(pdf_id)]


def test_tie_between_duplicate_bboxes_is_resolved_by_the_column_band():
    """Provinciaal blad table, row 1: columns 2 and 3 are both predicted on
    the amount, so `€` overlaps both at 1.0 and the first wins (column 2, the
    date column), while `3.500.000` goes to column 3. The header and the rows
    below put column 2 at x 738..915 and column 3 at x 932..1104."""
    proc = _processor()
    table_cells = [
        _cell(2, 0, 2, [729, 1341, 912, 1384], label="ched"),
        _cell(3, 0, 3, [925, 1341, 1104, 1386], label="ched"),
        _cell(6, 1, 2, [928, 1390, 1029, 1413]),
        _cell(7, 1, 3, [928, 1390, 1029, 1412]),
        _cell(14, 3, 2, [798, 1488, 898, 1510]),
        _cell(15, 3, 3, [928, 1488, 1014, 1510]),
    ]
    pdf_cells = [
        _pdf(201, [739, 1343, 787, 1359], "Einde"),
        _pdf(198, [932, 1343, 995, 1359], "Hoogte"),
        _pdf(208, [932, 1391, 942, 1408], "€"),
        _pdf(209, [947, 1391, 1030, 1408], "3.500.000"),
        _pdf(227, [739, 1489, 831, 1505], "31-12-2026"),
        _pdf(225, [932, 1489, 942, 1505], "€"),
        _pdf(226, [947, 1489, 1014, 1505], "750.000"),
    ]
    candidate_matches = {
        "201": [{"table_cell_id": 2, "iopdf": 1.0}],
        "198": [{"table_cell_id": 3, "iopdf": 1.0}],
        "208": [{"table_cell_id": 6, "iopdf": 1.0}, {"table_cell_id": 7, "iopdf": 1.0}],
        "209": [
            {"table_cell_id": 6, "iopdf": 0.985},
            {"table_cell_id": 7, "iopdf": 0.99},
        ],
        "227": [{"table_cell_id": 14, "iopdf": 0.357}],
        "225": [{"table_cell_id": 15, "iopdf": 1.0}],
        "226": [{"table_cell_id": 15, "iopdf": 0.993}],
    }
    final_matches = {
        "201": [{"table_cell_id": 2, "iopdf": 1.0}],
        "198": [{"table_cell_id": 3, "iopdf": 1.0}],
        "208": [{"table_cell_id": 6, "iopdf": 1.0}],
        "209": [{"table_cell_id": 7, "iopdf": 0.99}],
        "227": [{"table_cell_id": 14, "iopdf": 0.357}],
        "225": [{"table_cell_id": 15, "iopdf": 1.0}],
        "226": [{"table_cell_id": 15, "iopdf": 0.993}],
    }

    new_matches = proc._reassign_by_column_band(
        table_cells, pdf_cells, final_matches, candidate_matches
    )

    assert new_matches["208"][0]["table_cell_id"] == 7
    assert _column_of(table_cells, new_matches, 208) == 3
    # The weak but correct date in row 3 stays: its column band (header) holds it.
    assert new_matches["227"] == final_matches["227"]
    for pdf_id in (201, 198, 209, 225, 226):
        assert new_matches[str(pdf_id)] == final_matches[str(pdf_id)]


def test_a_wide_number_counts_as_inside_a_band_of_narrow_numbers():
    """Figure 1 of the TableFormer paper, rows 3 and 4: the model shifts the
    rows' bboxes one column to the left, so `13` and `17` (two digits) sit in
    column-0 cells although they stand under `4` and `9` (one digit each).
    Column 1's band is narrower than the words; the words still lie in that
    column. And the misplaced `17` must not keep `13` in column 0."""
    proc = _processor()
    table_cells = [
        _cell(3, 1, 0, [634, 895, 649, 918]),
        _cell(4, 1, 1, [690, 895, 705, 919]),
        _cell(9, 2, 1, [691, 920, 707, 943]),
        _cell(13, 3, 0, [691, 942, 713, 967]),
        _cell(14, 3, 1, [693, 942, 713, 967]),
        _cell(18, 4, 0, [691, 966, 713, 990]),
        _cell(19, 4, 1, [783, 966, 808, 990]),
    ]
    pdf_cells = [
        _pdf(364, [637.5, 900.6, 646.3, 916.1], "3"),
        _pdf(365, [694.5, 900.6, 703.3, 916.1], "4"),
        _pdf(370, [694.5, 925.2, 703.3, 940.7], "9"),
        _pdf(374, [694.5, 949.0, 712.1, 964.5], "13"),
        _pdf(378, [694.5, 971.8, 712.1, 987.3], "17"),
    ]
    candidate_matches = {
        "364": [{"table_cell_id": 3, "iopdf": 1.0}],
        "365": [{"table_cell_id": 4, "iopdf": 1.0}],
        "370": [{"table_cell_id": 9, "iopdf": 1.0}],
        # Row 3: two predicted bboxes on `13`, the first (column 0) wins.
        "374": [
            {"table_cell_id": 13, "iopdf": 1.0},
            {"table_cell_id": 14, "iopdf": 1.0},
        ],
        "378": [{"table_cell_id": 18, "iopdf": 1.0}],
    }
    final_matches = {k: [dict(v[0])] for k, v in candidate_matches.items()}

    new_matches = proc._reassign_by_column_band(
        table_cells, pdf_cells, final_matches, candidate_matches
    )

    assert new_matches["378"][0]["table_cell_id"] == 19
    # `13` needs the second pass: in the first, `17` still widened column
    # 0's band to cover it.
    assert new_matches["374"][0]["table_cell_id"] == 14
    assert new_matches["364"] == final_matches["364"]


def test_rows_out_of_column_order_build_no_band_and_stay_as_predicted():
    """PubTabNet PMC3039950_012_00 (val 301): the model emits every body row
    with the row header as the LAST cell, so the baseline reads
    `1.9 | 4.9 | 2.0 | R1` under a correct header `| Bound | Unbound | Model`.
    Those rows are not in column order and must not vouch for the columns,
    or the correct header would be moved to match the wrong majority. With
    the header alone as reference, `R1` fits no column (column 0 has no
    band), so its row is left as predicted."""
    proc = _processor()
    table_cells = [
        _cell(1, 0, 1, [78, 4, 115, 18], label="ched"),
        _cell(2, 0, 2, [133, 4, 184, 18], label="ched"),
        _cell(3, 0, 3, [203, 4, 239, 18], label="ched"),
    ]
    pdf_cells = [
        _pdf(0, [79, 4, 112, 18], "Bound"),
        _pdf(1, [135, 4, 181, 18], "Unbound"),
        _pdf(2, [204, 4, 236, 18], "Model"),
    ]
    candidate_matches = {
        "0": [{"table_cell_id": 1, "iopdf": 1.0}],
        "1": [{"table_cell_id": 2, "iopdf": 1.0}],
        "2": [{"table_cell_id": 3, "iopdf": 1.0}],
    }
    cell_id = 4
    pdf_id = 3
    for r, (label, a, b, c) in enumerate(
        [("R1", "1.9", "4.9", "2.0"), ("R2", "3.0", "4.2", "1.3"), ("R3", "2.5", "3.7", "1.4")],
        start=1,
    ):
        y = 20 + 15 * (r - 1)
        for col, (text, x1, x2) in enumerate(
            [(a, 88, 104), (b, 150, 166), (c, 213, 228), (label, 26, 41)]
        ):
            table_cells.append(_cell(cell_id, r, col, [x1 - 1, y, x2 + 1, y + 10]))
            pdf_cells.append(_pdf(pdf_id, [x1, y, x2, y + 10], text))
            candidate_matches[str(pdf_id)] = [{"table_cell_id": cell_id, "iopdf": 1.0}]
            cell_id += 1
            pdf_id += 1
    final_matches = {k: [dict(v[0])] for k, v in candidate_matches.items()}

    contributions = proc._column_band_contributions(table_cells, pdf_cells, candidate_matches)
    assert set(contributions) == {1, 2, 3}, "only the ordered header row builds bands"

    new_matches = proc._reassign_by_column_band(
        table_cells, pdf_cells, final_matches, candidate_matches
    )

    assert new_matches == final_matches


def test_ordered_header_row_repairs_body_rows_emitted_out_of_order():
    """PubTabNet PMC4038852_003_02 (val 207): same model error, but here
    every body cell has a column it fits, so the whole row is rotated back:
    `9,300 | 13,234 | Clinical` becomes `Clinical | 9,300 | 13,234`."""
    proc = _processor()
    table_cells = [
        _cell(0, 0, 0, [0, 4, 57, 13], label="ched"),
        _cell(1, 0, 1, [62, 4, 133, 13], label="ched"),
        _cell(2, 0, 2, [162, 4, 221, 13], label="ched"),
        _cell(3, 1, 0, [74, 17, 121, 26]),
        _cell(4, 1, 1, [138, 17, 235, 26]),
        _cell(5, 1, 2, [0, 17, 27, 26]),
        _cell(6, 2, 0, [72, 31, 120, 40]),
        _cell(7, 2, 1, [140, 31, 233, 40]),
        _cell(8, 2, 2, [0, 31, 54, 40]),
    ]
    pdf_cells = [
        _pdf(0, [1, 4, 57, 13], "Type of malaria"),
        _pdf(1, [63, 4, 132, 13], "Median WBC count"),
        _pdf(2, [158, 4, 217, 13], "Geometric mean"),
        _pdf(3, [1, 17, 27, 26], "Clinical"),
        _pdf(4, [75, 17, 120, 26], "9,300 cells/uL"),
        _pdf(5, [139, 17, 236, 26], "13, 234.5 ring trophozoites/uL"),
        _pdf(6, [1, 31, 53, 40], "Asymptomatic"),
        _pdf(7, [74, 31, 121, 40], "6, 555 cells/uL"),
        _pdf(8, [140, 31, 234, 40], "7, 834.7 ring trophozoites/uL"),
    ]
    candidate_matches = {
        "0": [{"table_cell_id": 0, "iopdf": 0.99}],
        "1": [{"table_cell_id": 1, "iopdf": 0.99}],
        "2": [{"table_cell_id": 2, "iopdf": 0.86}],
        "3": [{"table_cell_id": 5, "iopdf": 0.88}],
        "4": [{"table_cell_id": 3, "iopdf": 0.94}],
        "5": [{"table_cell_id": 4, "iopdf": 0.93}],
        "6": [{"table_cell_id": 8, "iopdf": 0.88}],
        "7": [{"table_cell_id": 6, "iopdf": 0.95}],
        "8": [{"table_cell_id": 7, "iopdf": 0.98}],
    }
    final_matches = {k: [dict(v[0])] for k, v in candidate_matches.items()}

    new_matches = proc._reassign_by_column_band(
        table_cells, pdf_cells, final_matches, candidate_matches
    )

    def slot(pdf_id):
        tc = next(c for c in table_cells if c["cell_id"] == new_matches[str(pdf_id)][0]["table_cell_id"])
        return tc["row_id"], tc["column_id"]

    assert slot(3) == (1, 0) and slot(4) == (1, 1) and slot(5) == (1, 2)
    assert slot(6) == (2, 0) and slot(7) == (2, 1) and slot(8) == (2, 2)
    for pdf_id in (0, 1, 2):
        assert new_matches[str(pdf_id)] == final_matches[str(pdf_id)]


def test_a_row_with_a_cell_that_fits_no_column_is_left_alone():
    """Moves are applied per row, all or nothing. Row 1 has a cell that is
    outside its band and fits no other column, so even its cell with a good
    destination stays; row 2 has the same movable cell and it moves."""
    proc = _processor()
    table_cells = [
        _cell(0, 0, 0, [10, 10, 90, 25]),
        _cell(1, 0, 1, [110, 10, 200, 25]),
        _cell(2, 1, 0, [10, 30, 90, 45]),
        _cell(3, 1, 1, [110, 30, 200, 45]),
        _cell(4, 2, 0, [10, 50, 90, 65]),
        _cell(5, 2, 1, [110, 50, 200, 65]),
        _cell(6, 3, 0, [10, 70, 90, 85]),
        _cell(7, 3, 1, [110, 70, 200, 85]),
    ]
    pdf_cells = [
        _pdf(0, [12, 12, 80, 23], "a0"),
        _pdf(1, [112, 12, 190, 23], "b0"),
        # row 1: a word of column 1 sitting in column 0's cell (movable) and
        # a word far right of every band, in column 1's cell (stuck)
        _pdf(2, [120, 32, 180, 43], "b1 in a1's cell"),
        _pdf(3, [300, 32, 340, 43], "stray"),
        # row 2: only the movable word
        _pdf(4, [120, 52, 180, 63], "b2 in a2's cell"),
        _pdf(5, [12, 72, 80, 83], "a3"),
        _pdf(6, [112, 72, 190, 83], "b3"),
    ]
    candidate_matches = {
        "0": [{"table_cell_id": 0, "iopdf": 1.0}],
        "1": [{"table_cell_id": 1, "iopdf": 1.0}],
        "2": [{"table_cell_id": 2, "iopdf": 0.3}],
        "3": [{"table_cell_id": 3, "iopdf": 0.3}],
        "4": [{"table_cell_id": 4, "iopdf": 0.3}],
        "5": [{"table_cell_id": 6, "iopdf": 1.0}],
        "6": [{"table_cell_id": 7, "iopdf": 1.0}],
    }
    final_matches = {k: [dict(v[0])] for k, v in candidate_matches.items()}

    new_matches = proc._reassign_by_column_band(
        table_cells, pdf_cells, final_matches, candidate_matches
    )

    assert new_matches["2"] == final_matches["2"], "row 1 is frozen by the stray word"
    assert new_matches["3"] == final_matches["3"]
    assert new_matches["4"][0]["table_cell_id"] == 5, "row 2 moves"


def test_pdf_cell_is_detached_when_the_target_slot_has_no_table_cell():
    """When the row has no table cell in the target column, the match is
    dropped so that step 9 places the pdf cell as an orphan by band."""
    proc = _processor()
    table_cells = [
        _cell(0, 0, 0, [10, 10, 90, 25]),
        _cell(1, 0, 1, [110, 10, 200, 25]),
        _cell(2, 1, 0, [10, 30, 90, 45]),
        # row 1 has no cell in column 1
    ]
    pdf_cells = [
        _pdf(0, [12, 12, 80, 23], "a"),
        _pdf(1, [112, 12, 190, 23], "b"),
        _pdf(2, [120, 32, 180, 43], "misplaced"),
    ]
    candidate_matches = {
        "0": [{"table_cell_id": 0, "iopdf": 1.0}],
        "1": [{"table_cell_id": 1, "iopdf": 1.0}],
        "2": [{"table_cell_id": 2, "iopdf": 0.2}],
    }
    final_matches = {k: [dict(v[0])] for k, v in candidate_matches.items()}

    new_matches = proc._reassign_by_column_band(
        table_cells, pdf_cells, final_matches, candidate_matches
    )

    assert "2" not in new_matches
    assert new_matches["0"] == final_matches["0"]
    assert new_matches["1"] == final_matches["1"]


def test_colspan_cells_neither_build_a_band_nor_move():
    """A header spanning two columns legitimately lies over the band of the
    second column. Its words must stay, and its extent must not widen the
    band of the first column."""
    proc = _processor()
    table_cells = [
        _cell(0, 0, 0, [10, 10, 200, 25], label="ched", colspan_val=2),
        _cell(1, 1, 0, [10, 30, 90, 45]),
        _cell(2, 1, 1, [110, 30, 200, 45]),
        _cell(3, 2, 0, [10, 50, 90, 65]),
        _cell(4, 2, 1, [110, 50, 200, 65]),
    ]
    pdf_cells = [
        _pdf(0, [120, 12, 190, 23], "spanning header word"),
        _pdf(1, [12, 32, 80, 43], "a1"),
        _pdf(2, [112, 32, 190, 43], "b1"),
        _pdf(3, [12, 52, 80, 63], "a2"),
        # a word in column 0's cell that sits under column 1's band
        _pdf(4, [120, 52, 190, 63], "b2 in a2's cell"),
    ]
    candidate_matches = {
        "0": [{"table_cell_id": 0, "iopdf": 1.0}],
        "1": [{"table_cell_id": 1, "iopdf": 1.0}],
        "2": [{"table_cell_id": 2, "iopdf": 1.0}],
        "3": [{"table_cell_id": 3, "iopdf": 1.0}],
        "4": [{"table_cell_id": 3, "iopdf": 0.3}],
    }
    final_matches = {k: [dict(v[0])] for k, v in candidate_matches.items()}

    new_matches = proc._reassign_by_column_band(
        table_cells, pdf_cells, final_matches, candidate_matches
    )

    # The spanning header word stays in the colspan cell.
    assert new_matches["0"] == final_matches["0"]
    # The misplaced word moves to column 1: the colspan did not widen column
    # 0's band to cover it.
    assert new_matches["4"][0]["table_cell_id"] == 4


def test_without_other_rows_nothing_moves():
    """A single-row table gives no band to judge against, so the final
    assignment stands even for a weak match."""
    proc = _processor()
    table_cells = [
        _cell(0, 0, 0, [10, 10, 90, 25]),
        _cell(1, 0, 1, [110, 10, 200, 25]),
    ]
    pdf_cells = [
        _pdf(0, [12, 12, 80, 23], "a"),
        _pdf(1, [95, 12, 130, 23], "weak"),
    ]
    candidate_matches = {
        "0": [{"table_cell_id": 0, "iopdf": 1.0}],
        "1": [{"table_cell_id": 1, "iopdf": 0.3}],
    }
    final_matches = {k: [dict(v[0])] for k, v in candidate_matches.items()}

    new_matches = proc._reassign_by_column_band(
        table_cells, pdf_cells, final_matches, candidate_matches
    )

    assert new_matches == final_matches


def test_process_places_a_row_whose_bboxes_are_shifted_by_one_column():
    """End to end through `process`, Provinciaal blad row 1 against the
    header: the model predicts column 1 on the date, columns 2 and 3 both on
    the amount, and nothing on the name. Before step 8.b the row came out as
    `31-12-2026 Grootschalige maatregelen hoofdfietsnet` | `€` | `3.500.000`.
    Expected: name | date | `€ 3.500.000`, with the name words recovered as
    orphans into the slot the date left."""
    proc = _processor()
    table_cells = [
        _cell(0, 0, 0, [236, 1341, 399, 1363], label="ched"),
        _cell(1, 0, 1, [436, 1341, 640, 1363], label="ched"),
        _cell(2, 0, 2, [729, 1341, 912, 1384], label="ched"),
        _cell(3, 0, 3, [925, 1341, 1104, 1386], label="ched"),
        _cell(4, 1, 0, [236, 1390, 352, 1413]),
        _cell(5, 1, 1, [691, 1390, 858, 1413]),
        _cell(6, 1, 2, [928, 1390, 1029, 1413]),
        _cell(7, 1, 3, [928, 1390, 1029, 1412]),
        _cell(12, 2, 0, [238, 1488, 362, 1510]),
        _cell(13, 2, 1, [438, 1488, 700, 1509]),
        _cell(14, 2, 2, [798, 1488, 898, 1510]),
        _cell(15, 2, 3, [928, 1488, 1014, 1510]),
    ]
    pdf_cells = [
        _pdf(207, [239, 1343, 399, 1359], "Subsidiegrondslag"),
        _pdf(205, [440, 1343, 491, 1359], "Naam"),
        _pdf(206, [496, 1343, 637, 1359], "subsidieregeling"),
        _pdf(201, [739, 1343, 787, 1359], "Einde"),
        _pdf(202, [792, 1343, 858, 1359], "looptijd"),
        _pdf(198, [932, 1343, 995, 1359], "Hoogte"),
        _pdf(214, [239, 1391, 321, 1408], "Paragraaf"),
        _pdf(215, [326, 1391, 351, 1408], "4.5"),
        _pdf(211, [440, 1391, 559, 1408], "Grootschalige"),
        _pdf(212, [564, 1391, 672, 1408], "maatregelen"),
        _pdf(213, [440, 1412, 554, 1429], "hoofdfietsnet"),
        _pdf(210, [739, 1391, 831, 1408], "31-12-2026"),
        _pdf(208, [932, 1391, 942, 1408], "€"),
        _pdf(209, [947, 1391, 1030, 1408], "3.500.000"),
        _pdf(231, [239, 1489, 321, 1505], "Paragraaf"),
        _pdf(232, [326, 1489, 359, 1505], "4.11"),
        _pdf(228, [440, 1489, 533, 1505], "Kruisingen"),
        _pdf(230, [564, 1489, 694, 1505], "fietsoversteken"),
        _pdf(227, [739, 1489, 831, 1505], "31-12-2026"),
        _pdf(225, [932, 1489, 942, 1505], "€"),
        _pdf(226, [947, 1489, 1014, 1505], "750.000"),
    ]
    matching_details = {
        "table_cells": table_cells,
        "pdf_cells": pdf_cells,
        # Empty initial matches: `process` regenerates them by intersection.
        "matches": {},
    }

    out = proc.process(matching_details)

    cells = {tc["cell_id"]: tc for tc in out["table_cells"]}

    def slot(pdf_id):
        tc = cells[out["matches"][str(pdf_id)][0]["table_cell_id"]]
        return tc["row_id"], tc["column_id"]

    assert slot(211) == (1, 1) and slot(212) == (1, 1) and slot(213) == (1, 1)
    assert slot(210) == (1, 2)
    assert slot(208) == (1, 3) and slot(209) == (1, 3)
    # The rows the model predicted well are untouched.
    assert slot(227) == (2, 2)
    assert slot(225) == (2, 3) and slot(226) == (2, 3)
    assert slot(201) == (0, 2) and slot(202) == (0, 2)
    assert slot(205) == (0, 1) and slot(206) == (0, 1)
