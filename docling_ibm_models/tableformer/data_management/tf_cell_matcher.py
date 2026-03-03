import copy
import logging
import re

import numpy as np

import docling_ibm_models.tableformer.otsl as otsl
import docling_ibm_models.tableformer.settings as s

# LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG
LOG_LEVEL = logging.WARN

# Cell labels
BODY = "body"
COL_HEADER = "col_header"
MULTI_COL_HEADER = "multi_col_header"
MULTI_ROW_HEADER = "multi_row_header"
MULTI_ROW = "multi_row"
MULTI_COL = "multi_col"


def validate_bboxes_page(bboxes):
    r"""
    Useful function for Debugging

    Validate that the bboxes have a positive area in the page coordinate system

    Parameters
    ----------
    bboxes : list of 4
        Each element of the list is expected to be a bbox in the page coordinates system

    Returns
    -------
    int
        The number of invalid bboxes.
    """
    invalid_counter = 0
    for i, bbox in enumerate(bboxes):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        if area < 0:
            print("Wrong bbox: {} - {}".format(i, bbox))
            invalid_counter += 1

    if invalid_counter > 0:
        print("Invalid bboxes in total: {}".format(invalid_counter))
    return invalid_counter


def find_intersection(b1, b2):
    r"""
    Compute the intersection between 2 bboxes

    Parameters
    ----------
    b1 : list of 4
        The page x1y1x2y2 coordinates of the bbox
    b2 : list of 4
        The page x1y1x2y2 coordinates of the bbox

    Returns
    -------
    The bbox of the intersection or None if there is no intersection
    """
    # Check when the bboxes do NOT intersect
    if b1[2] < b2[0] or b2[2] < b1[0] or b1[1] > b2[3] or b2[1] > b2[3]:
        return None

    i_bbox = [
        max(b1[0], b2[0]),
        max(b1[1], b2[1]),
        min(b1[2], b2[2]),
        min(b1[3], b2[3]),
    ]
    return i_bbox


class CellMatcher:
    r"""
    Match the table cells to the pdf page cells.

    NOTICE: PDF page coordinate system vs table coordinate system.
    In both systems the bboxes are described in as (x1, y1, x2, y2) with the following meaning:

    Page coordinate system:
    - Origin (0, 0) at the lower-left corner
    - (x1, y1) the lower left corner of the box
    - (x2, y2) the upper right corner of the box

    Table coordinate system:
    - Origin (0, 0) at the upper-left corner
    - (x1, y1) the upper left corner of the box
    - (x2, y2) the lower right corner of the box
    """

    def __init__(self, config):
        self._config = config
        self._iou_thres = config["predict"]["pdf_cell_iou_thres"]

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def match_cells(self, iocr_page, table_bbox, prediction):
        r"""
        Convert the tablemodel prediction into the Docling format

        Parameters
        ----------
        iocr_page : dict
            The original Docling provided table data
        prediction : dict
            The dictionary has the keys:
            "tag_seq": The sequence in indices from the WORDMAP
            "html_seq": The sequence as html tags
            "bboxes": The bounding boxes

        Returns
        -------
        matching_details : dict
            Dictionary with all details about the mathings between the table and pdf cells
        """
        pdf_cells = copy.deepcopy(iocr_page["tokens"])
        if len(pdf_cells) > 0:
            for word in pdf_cells:
                if isinstance(word["bbox"], list):
                    continue
                elif isinstance(word["bbox"], dict):
                    word["bbox"] = [
                        word["bbox"]["l"],
                        word["bbox"]["t"],
                        word["bbox"]["r"],
                        word["bbox"]["b"],
                    ]
        table_bboxes = prediction["bboxes"]
        table_classes = prediction["classes"]
        # BBOXES transformed...
        table_bboxes_page = self._translate_bboxes(table_bbox, table_bboxes)

        # Combine the table tags and bboxes into TableCells
        html_seq = prediction["html_seq"]
        otsl_seq = prediction["rs_seq"]
        table_cells = self._build_table_cells(
            html_seq, otsl_seq, table_bboxes_page, table_classes
        )

        matches = {}
        matches_counter = 0
        if len(pdf_cells) > 0:
            matches, matches_counter = self._intersection_over_pdf_match(
                table_cells, pdf_cells
            )

        self._log().debug("matches_counter: {}".format(matches_counter))

        # Build output
        matching_details = {
            "iou_threshold": self._iou_thres,
            "table_bbox": table_bbox,
            "prediction_bboxes_page": table_bboxes_page,  # Make easier the comparison with c++
            "prediction": prediction,
            "pdf_cells": pdf_cells,
            "page_height": iocr_page["height"],
            "page_width": iocr_page["width"],
            "table_cells": table_cells,
            "pdf_cells": pdf_cells,
            "matches": matches,
        }
        return matching_details

    def match_cells_dummy(self, iocr_page, table_bbox, prediction):
        r"""
        Convert the tablemodel prediction into the Docling format
        DUMMY version doesn't do matching with text cells, but propagates predicted bboxes,
        respecting the rest of the format

        Parameters
        ----------
        iocr_page : dict
            The original Docling provided table data
        prediction : dict
            The dictionary has the keys:
            "tag_seq": The sequence in indices from the WORDMAP
            "html_seq": The sequence as html tags
            "bboxes": The bounding boxes

        Returns
        -------
        matching_details : dict
            Dictionary with all details about the mathings between the table and pdf cells
        """
        pdf_cells = copy.deepcopy(iocr_page["tokens"])
        if len(pdf_cells) > 0:
            for word in pdf_cells:
                word["bbox"] = [
                    word["bbox"]["l"],
                    word["bbox"]["t"],
                    word["bbox"]["r"],
                    word["bbox"]["b"],
                ]

        table_bboxes = prediction["bboxes"]
        table_classes = prediction["classes"]
        # BBOXES transformed...
        table_bboxes_page = self._translate_bboxes(table_bbox, table_bboxes)

        # Combine the table tags and bboxes into TableCells
        html_seq = prediction["html_seq"]
        otsl_seq = prediction["rs_seq"]

        table_cells = self._build_table_cells(
            html_seq, otsl_seq, table_bboxes_page, table_classes
        )

        # Build output
        matching_details = {
            "iou_threshold": self._iou_thres,
            "table_bbox": table_bbox,
            "prediction_bboxes_page": table_bboxes_page,
            "prediction": prediction,
            "pdf_cells": pdf_cells,
            "page_height": iocr_page["height"],
            "page_width": iocr_page["width"],
            "table_cells": table_cells,
            "pdf_cells": pdf_cells,
            "matches": {},
        }
        return matching_details

    def _build_table_cells(self, html_seq, otsl_seq, bboxes, table_classes):
        r"""
        Combine the tags and bboxes of the table into unified TableCell objects.
        Each TableCell takes a row_id, column_id index based on the html structure provided by
        html_seq.
        It is assumed that the bboxes are in sync with the appearence of the closing </td>

        Parameters
        ----------
        html_seq : list
            List of html tags
        bboxes : list of lists of 4
            Bboxes for the table cells at the page origin

        Returns
        -------
        list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        """
        table_html_structure = {
            "html": {"structure": {"tokens": html_seq}},
            "split": "predict",
            "filename": "memory",
        }

        otsl_spans = {}

        # r, o = otsl.html_to_otsl(table, writer, true, extra_debug, include_html)
        r, o = otsl.html_to_otsl(table_html_structure, None, False, False, True, False)
        if not r:
            ermsg = "ERR#: COULD NOT CONVERT TO RS THIS TABLE TO COMPUTE SPANS"
            self._log().debug(ermsg)
        else:
            otsl_spans = o["otsl_spans"]

        table_cells = []

        # It is assumed that the bboxes appear in sync (at the same order) as the TDs
        cell_id = 0

        row_id = -1
        column_id = -1
        in_header = False
        in_body = False
        multicol_tag = ""
        colspan_val = 0
        rowspan_val = 0

        mode = "OTSL"
        if mode == "HTML":
            for tag in html_seq:
                label = None
                if tag == "<thead>":
                    in_header = True
                    multicol_tag = ""
                    colspan_val = 0
                    rowspan_val = 0
                elif tag == "</thead>":
                    in_header = False
                    multicol_tag = ""
                    colspan_val = 0
                    rowspan_val = 0
                elif tag == "<tbody>":
                    in_body = True
                    multicol_tag = ""
                    colspan_val = 0
                    rowspan_val = 0
                elif tag == "</tbody>":
                    in_body = False
                    multicol_tag = ""
                    colspan_val = 0
                    rowspan_val = 0
                elif tag == "<td>" or tag == "<td":
                    column_id += 1
                    multicol_tag = ""
                    colspan_val = 0
                    rowspan_val = 0
                    if tag == "<td":
                        multicol_tag = tag
                elif tag == "<tr>":
                    row_id += 1
                    column_id = -1
                    multicol_tag = ""
                    colspan_val = 0
                    rowspan_val = 0
                elif "colspan" in tag:
                    label = MULTI_COL
                    multicol_tag += tag
                    colspan_val = int(re.findall(r'"([^"]*)"', tag)[0])
                elif "rowspan" in tag:
                    label = MULTI_ROW
                    multicol_tag += tag
                    rowspan_val = int(re.findall(r'"([^"]*)"', tag)[0])
                elif tag == "</td>":  # Create a TableCell on each closing td
                    if len(multicol_tag) > 0:
                        multicol_tag += tag
                    if in_header:
                        if label is None:
                            label = COL_HEADER
                        elif label == MULTI_COL:
                            label = MULTI_COL_HEADER
                        elif label == MULTI_ROW:
                            label = MULTI_ROW_HEADER
                    if label is None and in_body:
                        label = BODY

                    err_mismatch = "Mismatching bboxes with closing TDs {} < {}".format(
                        cell_id, len(bboxes)
                    )
                    assert cell_id < len(bboxes), err_mismatch
                    bbox = bboxes[cell_id]
                    cell_class = table_classes[cell_id]

                    table_cell = {}
                    table_cell["cell_id"] = cell_id
                    table_cell["row_id"] = row_id
                    table_cell["column_id"] = column_id
                    table_cell["bbox"] = bbox
                    table_cell["cell_class"] = cell_class
                    table_cell["label"] = label
                    table_cell["multicol_tag"] = multicol_tag
                    if colspan_val > 0:
                        table_cell["colspan_val"] = colspan_val
                        column_id += (
                            colspan_val - 1
                        )  # Shift column index to account for span
                    if rowspan_val > 0:
                        table_cell["rowspan_val"] = rowspan_val

                    table_cells.append(table_cell)
                    cell_id += 1

        if mode == "OTSL":
            row_id = 0
            column_id = 0
            multicol_tag = ""
            otsl_line = []
            cell_id_line = []

            for tag in otsl_seq:
                otsl_line.append(tag)
                if tag == "nl":
                    row_id += 1
                    column_id = 0
                    otsl_line = []
                    cell_id_line = []
                if tag in ["fcel", "ecel", "xcel", "ched", "rhed", "srow"]:
                    cell_id_line.append(cell_id)
                    bbox = [0.0, 0.0, 0.0, 0.0]
                    if cell_id < len(bboxes):
                        bbox = bboxes[cell_id]

                    cell_class = 2
                    if cell_id < len(table_classes):
                        cell_class = table_classes[cell_id]
                    label = tag

                    table_cell = {}
                    table_cell["cell_id"] = cell_id
                    table_cell["row_id"] = row_id
                    table_cell["column_id"] = column_id
                    table_cell["bbox"] = bbox
                    table_cell["cell_class"] = cell_class
                    table_cell["label"] = label
                    table_cell["multicol_tag"] = multicol_tag

                    colspan_val = 0
                    rowspan_val = 0

                    if cell_id in otsl_spans:
                        colspan_val = otsl_spans[cell_id][0]
                        rowspan_val = otsl_spans[cell_id][1]
                    if colspan_val > 0:
                        table_cell["colspan_val"] = colspan_val
                    if rowspan_val > 0:
                        table_cell["rowspan_val"] = rowspan_val

                    table_cells.append(table_cell)
                    cell_id += 1
                if tag != "nl":
                    column_id += 1

        return table_cells

    def _translate_bboxes(self, table_bbox, cell_bboxes):
        r"""
        Translate table cell bboxes to the lower-left corner of the page.

        The cells of the table are given:
        - Origin at the top left corner
        - Point A: Top left corner
        - Point B: Low right corner
        - Coordinate values are normalized to the table width/height

        Parameters
        ----------
        table_bbox : list of 4
            The whole table bbox page coordinates
        cell_bboxes : list of lists of 4
            The bboxes of the table cells

        Returns
        -------
        list of 4
            The translated bboxes of the table cells
        """
        W = table_bbox[2] - table_bbox[0]
        H = table_bbox[3] - table_bbox[1]
        b = np.asarray(cell_bboxes)
        t_mask = np.asarray(
            [table_bbox[0], table_bbox[3], table_bbox[0], table_bbox[3]]
        )
        m = np.asarray([W, -H, W, -H])
        page_bboxes_y_flipped = t_mask + m * b
        page_bboxes = page_bboxes_y_flipped[:, [0, 3, 2, 1]]  # Flip y1' with y2'
        page_bboxes_list = page_bboxes.tolist()

        t_height = table_bbox[3]
        page_bboxes_list1 = []
        for page_bbox in page_bboxes_list:
            page_bbox1 = [
                page_bbox[0],
                t_height - page_bbox[3] + table_bbox[1],
                page_bbox[2],
                t_height - page_bbox[1] + table_bbox[1],
            ]
            page_bboxes_list1.append(page_bbox1)
        return page_bboxes_list1

    def _intersection_over_pdf_match(self, table_cells, pdf_cells):
        r"""
        Compute Intersection between table cells and pdf cells,
        match 1 pdf cell with highest intersection with only 1 table cell.

        First compute and cache the areas for all involved bboxes.
        Then compute the pairwise intersections

        Parameters
        ----------
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"

        pdf_cells : list of dict
            Each element of the list is a dictionary which should have the keys: "id", "bbox"
        Returns
        -------
        dictionary of lists of table_cells
            Return a dictionary which is indexed by the pdf_cell_id as key and the value is a list
            of the table_cells that fall inside that pdf cell
        int
            Number of total matches
        """
        pdf_bboxes = np.asarray([p["bbox"] for p in pdf_cells])
        pdf_bboxes_areas = (pdf_bboxes[:, 2] - pdf_bboxes[:, 0]) * (
            pdf_bboxes[:, 3] - pdf_bboxes[:, 1]
        )

        # key: pdf_cell_id, value: list of TableCell that fall inside that pdf_cell
        matches = {}
        matches_counter = 0

        # Compute Intersections and build matches
        for i, table_cell in enumerate(table_cells):
            table_cell_id = table_cell["cell_id"]
            t_bbox = table_cell["bbox"]

            for j, pdf_cell in enumerate(pdf_cells):
                pdf_cell_id = pdf_cell["id"]
                p_bbox = pdf_cell["bbox"]

                # Compute intersection
                i_bbox = find_intersection(t_bbox, p_bbox)
                if i_bbox is None:
                    continue

                # Compute IOU and filter on threshold
                i_bbox_area = (i_bbox[2] - i_bbox[0]) * (i_bbox[3] - i_bbox[1])
                iopdf = 0
                if float(pdf_bboxes_areas[j]) > 0:
                    iopdf = i_bbox_area / float(pdf_bboxes_areas[j])

                if iopdf > 0:
                    match = {"table_cell_id": table_cell_id, "iopdf": iopdf}
                    if pdf_cell_id not in matches:
                        matches[pdf_cell_id] = [match]
                        matches_counter += 1
                    else:
                        # Check if the same match was not already counted
                        if match not in matches[pdf_cell_id]:
                            matches[pdf_cell_id].append(match)
                            matches_counter += 1
        return matches, matches_counter

    def _iou_match(self, table_cells, pdf_cells):
        r"""
        Use Intersection over Union to decide the matching between table cells and pdf cells

        First compute and cache the areas for all involved bboxes.
        Then compute the pairwise intersections and IOUs and keep those pairs that exceed the IOU
        threshold

        Parameters
        ----------
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"

        pdf_cells : list of dict
            Each element of the list is a dictionary which should have the keys: "id", "bbox"
        Returns
        -------
        dictionary of lists of table_cells
            Return a dictionary which is indexed by the pdf_cell_id as key and the value is a list
            of the table_cells that fall inside that pdf cell
        int
            Number of total matches
        """
        table_bboxes = np.asarray([t["bbox"] for t in table_cells])
        pdf_bboxes = np.asarray([p["bbox"] for p in pdf_cells])

        # Cache the areas for table bboxes and pdf bboxes
        table_bboxes_areas = (table_bboxes[:, 2] - table_bboxes[:, 0]) * (
            table_bboxes[:, 3] - table_bboxes[:, 1]
        )

        pdf_bboxes_areas = (pdf_bboxes[:, 2] - pdf_bboxes[:, 0]) * (
            pdf_bboxes[:, 3] - pdf_bboxes[:, 1]
        )

        # key: pdf_cell_id, value: list of TableCell that fall inside that pdf_cell
        matches = {}
        matches_counter = 0

        # Compute IOUs and build matches
        for i, table_cell in enumerate(table_cells):
            table_cell_id = table_cell["cell_id"]
            t_bbox = table_cell["bbox"]

            for j, pdf_cell in enumerate(pdf_cells):
                pdf_cell_id = pdf_cell["id"]
                pdf_cell_text = pdf_cell["text"]
                p_bbox = pdf_cell["bbox"]

                # Compute intersection
                i_bbox = find_intersection(t_bbox, p_bbox)
                if i_bbox is None:
                    continue

                # Compute IOU and filter on threshold
                i_bbox_area = (i_bbox[2] - i_bbox[0]) * (i_bbox[3] - i_bbox[1])
                iou = 0
                div_area = float(
                    table_bboxes_areas[i] + pdf_bboxes_areas[j] - i_bbox_area
                )
                if div_area > 0:
                    iou = i_bbox_area / div_area
                if iou < self._iou_thres:
                    continue

                if pdf_cell_id not in matches:
                    matches[pdf_cell_id] = []

                match = {
                    "table_cell_id": table_cell_id,
                    "iou": iou,
                    "text": pdf_cell_text,
                }
                matches[pdf_cell_id].append(match)
                matches_counter += 1

        return matches, matches_counter
