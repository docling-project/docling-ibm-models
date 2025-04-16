#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import json
import logging
import math
import statistics

import docling_ibm_models.tableformer.settings as s
from docling_ibm_models.tableformer.data_management.tf_cell_matcher import CellMatcher

LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


class MatchingPostProcessor:
    r"""
    The MatchingPostProcessor aims to improve the matchings between the predicted table cells and
    the pdf cells
    """

    def __init__(self, config):
        self._config = config
        self._cell_matcher = CellMatcher(config)

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def _get_table_dimension(self, table_cells):
        r"""
        Get dimensions (columns, rows) of a table from table_cells

        Parameters
        ----------
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"

        Returns
        -------
        columns : integer,
        rows : integer,
        max_cell_id : integer,
            highest cell_id in table_cells
        """
        columns = 1
        rows = 1
        max_cell_id = 0

        for cell in table_cells:
            if cell["column_id"] > columns:
                columns = cell["column_id"]
            if cell["row_id"] > rows:
                rows = cell["row_id"]
            if cell["cell_id"] > max_cell_id:
                max_cell_id = cell["cell_id"]

        return columns + 1, rows + 1, max_cell_id

    def _get_good_bad_cells_in_column(self, table_cells, column, matches):
        r"""
        1. step
        Get good/bad IOU predicted cells for each structural column (of minimal grid)

        Parameters
        ----------
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        column : integer
            Index of a column
        matches : dictionary of lists of table_cells
            A dictionary which is indexed by the pdf_cell_id as key and the value is a list
            of the table_cells that fall inside that pdf cell

        Returns
        -------
        good_table_cells : list of dict
            cells in a column that have match
        bad_table_cells : list of dict
            cells in a column that don't have match
        """
        good_table_cells = []
        bad_table_cells = []

        for cell in table_cells:
            if cell["column_id"] == column:
                table_cell_id = cell["cell_id"]

                bad_match = True
                allow_class = True

                for pdf_cell_id in matches:
                    # CHECK IF CELL CLASS TO BE VERIFIED HERE
                    if "cell_class" in cell:
                        if cell["cell_class"] <= 1:
                            allow_class = False
                    else:
                        self._log().debug("***")
                        self._log().debug("no cell_class in...")
                        self._log().debug(cell)
                        self._log().debug("***")
                    if allow_class:
                        match_list = matches[pdf_cell_id]
                        for match in match_list:
                            if match["table_cell_id"] == table_cell_id:
                                good_table_cells.append(cell)
                                bad_match = False
                if bad_match:
                    bad_table_cells.append(cell)

        return good_table_cells, bad_table_cells

    def _delete_column_from_table(self, table_cells, column):
        r"""
        1.a. step
        If all IOU in a column are bad - eliminate column (from bboxes and structure)

        Parameters
        ----------
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        column : integer
            Index of a column

        Returns
        -------
        new_table_cells : list of dict
        """
        new_table_cells = []

        for cell in table_cells:
            if cell["column_id"] < column:
                new_table_cells.append(cell)
            if cell["column_id"] > column:
                new_cell = {
                    "bbox": cell["bbox"],
                    "cell_id": cell["cell_id"],
                    "column_id": cell["column_id"] - 1,
                    "label": cell["label"],
                    "row_id": cell["row_id"],
                    "cell_class": cell["cell_class"],
                }
                new_table_cells.append(new_cell)

        return new_table_cells

    def _find_alignment_in_column(self, cells):
        r"""
        2. step
        Find alignment of good IOU cells per column

        Parameters
        ----------
        cells : list of dict
            Cells in a column
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"

        Returns
        -------
        alignment : string
            column general alignment can be: "left", "right", "center"
        """
        possible_alignments = ["left", "middle", "right"]
        alignment = "left"  # left / right / center

        lefts = []
        rights = []
        middles = []

        for cell in cells:
            x_left = cell["bbox"][0]
            x_right = cell["bbox"][2]
            x_middle = (x_left + x_right) / 2
            lefts.append(x_left)
            rights.append(x_right)
            middles.append(x_middle)

        if len(lefts) > 0:
            delta_left = max(lefts) - min(lefts)
            delta_middle = max(middles) - min(middles)
            delta_right = max(rights) - min(rights)

            deltas = [delta_left, delta_middle, delta_right]
            align_index = deltas.index(min(deltas))
            alignment = possible_alignments[align_index]

        return alignment

    def _get_median_pos_size(self, cells, alignment):
        r"""
        3. step
        Get median* (according to alignment) "bbox left/middle/right X" coord
        for good IOU cells, get median* cell size in a column.

        Parameters
        ----------
        cells : list of dict
            Cells in a column
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        alignment : string
            column general alignment can be: "left", "right", "center"

        Returns
        -------
        median_x : number
            Median X position of a cell (according to alignment)
        median_y : number
            Median Y position of a cell (according to alignment)
        median_width : number
            Median width of a cell
        median_height : number
            Median height of a cell
        """
        median_x = 0
        median_y = 0
        median_width = 1
        median_height = 1

        coords_x = []
        coords_y = []
        widths = []
        heights = []

        for cell in cells:
            if "rowspan_val" not in cell:
                if "colspan_val" not in cell:
                    if cell["cell_class"] > 1:
                        # Use left alignment
                        x_coord = cell["bbox"][0]
                        if alignment == "middle":
                            # Use middle alignment
                            x_coord = (cell["bbox"][2] + cell["bbox"][0]) / 2
                        if alignment == "right":
                            # Use right alignment
                            x_coord = cell["bbox"][2]

                        coords_x.append(x_coord)
                        y_coord = cell["bbox"][1]
                        coords_y.append(y_coord)

                        width = cell["bbox"][2] - cell["bbox"][0]
                        widths.append(width)
                        height = cell["bbox"][3] - cell["bbox"][1]
                        heights.append(height)
                    else:
                        self._log().debug("Empty cells not considered in medians")
                        self._log().debug(cell)
                else:
                    self._log().debug("Colspans not considered in medians")
                    self._log().debug(cell)
            else:
                self._log().debug("Rowspans not considered in medians")
                self._log().debug(cell)

        if len(coords_x) > 0:
            median_x = statistics.median(coords_x)
        if len(coords_y) > 0:
            median_y = statistics.median(coords_y)
        if len(widths) > 0:
            median_width = statistics.median(widths)
        if len(heights) > 0:
            median_height = statistics.median(heights)
        return median_x, median_y, median_width, median_height

    def _move_cells_to_left_pos(
        self, cells, median_x, rescale, median_width, median_height, alignment
    ):
        r"""
        4. step
        Move bad cells to the median* (left/middle/right) good in a column
        (Additionally), re-size cell to median* size of cells in a column

        Parameters
        ----------
        cells : list of dict
            Cells in a column
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        median_x : number
            Median X position of a cell (according to alignment)
        rescale : boolean
            should cells be re-sized to median or not
        median_width : number
            Median width of a cell
        median_height : number
            Median height of a cell
        alignment : string
            column general alignment can be: "left", "right", "center"

        Returns
        -------

        new_table_cells : list of dict
            Cells in a column
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        """
        new_table_cells = []

        for cell in cells:
            new_cell = {
                "bbox": [],
                "cell_id": 0,
                "column_id": 0,
                "label": "",
                "row_id": 0,
                "cell_class": 0,
            }
            x1 = cell["bbox"][0]
            y1 = cell["bbox"][1]
            x2 = cell["bbox"][2]
            y2 = cell["bbox"][3]
            original_width = x2 - x1
            # original_height = y2 - y1

            # Move to left by default
            new_x1 = median_x
            new_y1 = y1
            new_x2 = median_x + original_width
            new_y2 = y2

            if rescale:
                new_x2 = median_x + median_width
                # Next line does vertical resizing of BBOX:
                new_y2 = y1 + median_height

            # Move to middle
            if alignment == "middle":
                # TODO
                new_x1 = median_x - (original_width / 2)
                new_x2 = new_x1 + original_width
                if rescale:
                    new_x1 = median_x - (median_width / 2)
                    new_x2 = median_x + (median_width / 2)

            # Move to right
            if alignment == "right":
                new_x1 = median_x - original_width
                new_x2 = median_x
                if rescale:
                    new_x1 = median_x - median_width

            new_cell["bbox"] = [new_x1, new_y1, new_x2, new_y2]
            new_cell["cell_id"] = cell["cell_id"]
            new_cell["column_id"] = cell["column_id"]
            new_cell["label"] = cell["label"]
            new_cell["row_id"] = cell["row_id"]
            new_cell["cell_class"] = cell["cell_class"]
            # Add spans if present
            if "rowspan_val" in cell:
                new_cell["rowspan_val"] = cell["rowspan_val"]
            if "colspan_val" in cell:
                new_cell["colspan_val"] = cell["colspan_val"]
            new_table_cells.append(new_cell)
        return new_table_cells

    def _run_intersection_match(self, cell_matcher, table_cells, pdf_cells):
        r"""
        5. step
        Generate new matches, run Intersection over cell(pdf) on a table cells

        Parameters
        ----------
        cell_matcher : CellMatcher
            src.data_management.cell_matcher
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        pdf_cells : list of dict
            List of PDF cells as defined by Docling

        Returns
        -------
        clean_matches : dictionary of lists of table_cells
            A dictionary which is indexed by the pdf_cell_id as key and the value is a list
            of the table_cells that fall inside that pdf cell
        """
        new_matches = {}
        clean_matches = {}
        new_matches, matches_counter = cell_matcher._intersection_over_pdf_match(
            table_cells, pdf_cells
        )
        clean_matches = new_matches
        # Convert to JSON and back to have string keys in the dictionary
        clean_matches_string = json.dumps(clean_matches)
        clean_matches = json.loads(clean_matches_string)
        return clean_matches

    def _find_overlapping(self, table_cells):

        def correct_overlap(box1, box2):
            # Extract coordinates from the bounding boxes
            x1_min, y1_min, x1_max, y1_max = box1["bbox"]
            x2_min, y2_min, x2_max, y2_max = box2["bbox"]

            # Calculate the overlap in both x and y directions
            overlap_x = min(x1_max, x2_max) - max(x1_min, x2_min)
            overlap_y = min(y1_max, y2_max) - max(y1_min, y2_min)

            # If there is no overlap, return the original boxes
            if overlap_x <= 0 or overlap_y <= 0:
                return box1, box2

            # Decide how to push the boxes apart
            if overlap_x < overlap_y:
                # Push horizontally
                if x1_min < x2_min:
                    # Move box1 to the left and box2 to the right
                    box1["bbox"][2] -= math.ceil(overlap_x / 2) + 2
                    box2["bbox"][0] += math.floor(overlap_x / 2)
                else:
                    # Move box2 to the left and box1 to the right
                    box2["bbox"][2] -= math.ceil(overlap_x / 2) + 2
                    box1["bbox"][0] += math.floor(overlap_x / 2)
            else:
                # Push vertically
                if y1_min < y2_min:
                    # Move box1 up and box2 down
                    box1["bbox"][3] -= math.ceil(overlap_y / 2) + 2
                    box2["bbox"][1] += math.floor(overlap_y / 2)
                else:
                    # Move box2 up and box1 down
                    box2["bbox"][3] -= math.ceil(overlap_y / 2) + 2
                    box1["bbox"][1] += math.floor(overlap_y / 2)

            # Will flip coordinates in proper order, if previous operations reversed it
            box1["bbox"] = [
                min(box1["bbox"][0], box1["bbox"][2]),
                min(box1["bbox"][1], box1["bbox"][3]),
                max(box1["bbox"][0], box1["bbox"][2]),
                max(box1["bbox"][1], box1["bbox"][3]),
            ]
            box2["bbox"] = [
                min(box2["bbox"][0], box2["bbox"][2]),
                min(box2["bbox"][1], box2["bbox"][3]),
                max(box2["bbox"][0], box2["bbox"][2]),
                max(box2["bbox"][1], box2["bbox"][3]),
            ]

            return box1, box2

        def do_boxes_overlap(box1, box2):
            B1 = box1["bbox"]
            B2 = box2["bbox"]
            if (
                (B1[0] >= B2[2])
                or (B1[2] <= B2[0])
                or (B1[3] <= B2[1])
                or (B1[1] >= B2[3])
            ):
                return False
            else:
                return True

        def find_overlapping_pairs_indexes(bboxes):
            overlapping_indexes = []
            # Compare each box with every other box (combinations)
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    if i != j:
                        if bboxes[i] != bboxes[j]:
                            if do_boxes_overlap(bboxes[i], bboxes[j]):
                                bboxes[i], bboxes[j] = correct_overlap(
                                    bboxes[i], bboxes[j]
                                )

            return overlapping_indexes, bboxes

        overlapping_indexes, table_cells = find_overlapping_pairs_indexes(table_cells)
        return table_cells

    def _align_table_cells_to_pdf(self, table_cells, pdf_cells, matches):
        """
        Align table cell bboxes with good matches to encapsulate matching pdf cells
        """
        pdf_cell_dict = {pdf_cell["id"]: pdf_cell["bbox"] for pdf_cell in pdf_cells}
        table_cell_dict = {cell["cell_id"]: cell for cell in table_cells}

        # Track unique cells we're going to add
        processed_cells = set()

        # First pass - create initial new_table_cells with aligned bboxes
        new_table_cells = []

        for pdf_cell_id, match_list in matches.items():
            # Extract unique table cell ids from match_list
            table_cell_ids = set(int(match["table_cell_id"]) for match in match_list)

            # Get bbox of pdf_cell
            pdf_cell_bbox = pdf_cell_dict.get(int(pdf_cell_id))
            if not pdf_cell_bbox:
                continue

            # Process each unique table cell
            for cell_id in table_cell_ids:
                if cell_id in processed_cells:
                    continue

                table_cell = table_cell_dict.get(cell_id)
                if not table_cell:
                    continue

                # Create new table cell with aligned bbox
                new_table_cell = table_cell.copy()
                new_table_cell["bbox"] = list(pdf_cell_bbox)

                # Set cell class
                if "cell_class" not in new_table_cell:
                    new_table_cell["cell_class"] = "2"

                new_table_cells.append(new_table_cell)
                processed_cells.add(cell_id)

        # Second pass - aggregate bboxes for duplicate cells
        cell_to_bboxes = {}
        for cell in new_table_cells:
            cell_id = cell["cell_id"]
            if cell_id not in cell_to_bboxes:
                cell_to_bboxes[cell_id] = []
            cell_to_bboxes[cell_id].append(cell["bbox"])

        # Create final clean table cells
        clean_table_cells = []
        processed_ids = set()

        for cell in new_table_cells:
            cell_id = cell["cell_id"]
            if cell_id in processed_ids:
                continue

            bboxes = cell_to_bboxes[cell_id]
            if len(bboxes) > 1:
                # Merge bboxes
                x1s = [bbox[0] for bbox in bboxes]
                y1s = [bbox[1] for bbox in bboxes]
                x2s = [bbox[2] for bbox in bboxes]
                y2s = [bbox[3] for bbox in bboxes]

                cell["bbox"] = [min(x1s), min(y1s), max(x2s), max(y2s)]

            clean_table_cells.append(cell)
            processed_ids.add(cell_id)

        return clean_table_cells

    def _deduplicate_cells(self, tab_columns, table_cells, iou_matches, ioc_matches):
        r"""
        7. step

        De-duplicate columns in table_cells according to highest column score
        in: matches + intersection_pdf_matches

        Parameters
        ----------
        tab_columns : integer
            Number of table columns
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        iou_matches : dictionary of lists of table_cells
            Cell matches done using Intersection Over Union (IOU) method
        ioc_matches : dictionary of lists of table_cells
            Cell matches done using Intersection Over (PDF) Cell method

        Returns
        -------
        new_table_cells : list of dict
            New table cells with removed column duplicates
        new_matches : dictionary of lists of table_cells
            Matches that are in sync with new_table_cells
        new_tab_columns : integer
            New number of table columns
        """
        pdf_cells_in_columns = []
        total_score_in_columns = []

        for col in range(tab_columns):
            column_table_cells = []
            column_pdf_cells_iou = []
            column_pdf_cells_ioc = []
            column_pdf_cells = []
            column_iou_score = 0
            column_ioc_score = 0

            for cell in table_cells:
                if cell["column_id"] == col:
                    table_cell_id = cell["cell_id"]
                    column_table_cells.append(table_cell_id)

            # SUM IOU + IOC Scores for column, collect all pdf_cell_id
            for iou_key in iou_matches:
                iou_match_list = iou_matches[iou_key]
                for uk in range(len(iou_match_list)):
                    t_cell_id = iou_match_list[uk]["table_cell_id"]
                    if t_cell_id in column_table_cells:
                        if "iou" in iou_match_list[uk]:
                            # In case initial match was IOU
                            column_iou_score += iou_match_list[uk]["iou"]
                        elif "iopdf" in iou_match_list[uk]:
                            # Otherwise it's intersection over PDF match
                            column_iou_score += iou_match_list[uk]["iopdf"]
                        column_pdf_cells_iou.append(iou_key)

            for ioc_key in ioc_matches:
                ioc_match_list = ioc_matches[ioc_key]
                for k in range(len(ioc_match_list)):
                    t_cell_id = ioc_match_list[k]["table_cell_id"]
                    if t_cell_id in column_table_cells:
                        column_ioc_score += ioc_match_list[k]["iopdf"]
                        column_pdf_cells_ioc.append(ioc_key)

            column_pdf_cells = column_pdf_cells_iou
            column_pdf_cells += list(
                set(column_pdf_cells_ioc) - set(column_pdf_cells_iou)
            )
            column_total_score = column_iou_score + column_ioc_score

            pdf_cells_in_columns.append(column_pdf_cells)
            total_score_in_columns.append(column_total_score)
            self._log().debug(
                "Column: {}, Score:{}, PDF cells: {}".format(
                    col, column_total_score, column_pdf_cells
                )
            )

        # Eliminate duplicates in the pdf_cells_in_columns and ensure int content
        # pdf_cells_in_columns:
        # - initially:  list of lists of str with duplicates in the inner lists
        # - afterwards: list of lists of int (unique)
        pdf_cells_in_columns = [
            list(set([x for x in map(lambda x: int(x), le)]))
            for le in pdf_cells_in_columns
        ]
        cols_to_eliminate = []
        # Pairwise comparison of all columns, finding intersection, and it's length
        for cl in range(tab_columns - 1):
            col_a = pdf_cells_in_columns[cl]
            col_b = pdf_cells_in_columns[cl + 1]
            score_a = total_score_in_columns[cl]
            score_b = total_score_in_columns[cl + 1]
            intsct = list(set(col_a).intersection(col_b))
            int_prc = 0
            if len(col_a) > 0:
                int_prc = len(intsct) / len(col_a)
            logstring = "Col A: {}, Col B: {}, Int: {}, %: {}, Score A: {}, Score B: {}"
            self._log().debug(
                logstring.format(cl, cl + 1, len(intsct), int_prc, score_a, score_b)
            )

            # Consider structural column elimination
            # if 60% of two columns pointing to the same pdf cells
            if int_prc > 0.6:
                if score_a >= score_b:
                    # Elliminate B
                    cols_to_eliminate.append(cl + 1)
                if score_b > score_a:
                    # Elliminate A
                    cols_to_eliminate.append(cl)

        self._log().debug("Columns to eliminate: {}".format(cols_to_eliminate))
        new_table_cells = []
        new_matches = {}

        removed_table_cell_ids = []
        new_tab_columns = tab_columns - len(cols_to_eliminate)

        # Clean table_cells structure
        for tab_cell in table_cells:
            add_cell = True
            for col_del in cols_to_eliminate:
                if tab_cell["column_id"] == col_del:
                    removed_table_cell_ids.append(tab_cell["cell_id"])
                    add_cell = False
            if add_cell:
                new_table_cells.append(tab_cell)
        # Clean ioc_matches structure
        for pdf_cell_id, pdf_cell_matches in ioc_matches.items():
            new_cell_match = []
            for pdf_match in pdf_cell_matches:
                if pdf_match["table_cell_id"] not in removed_table_cell_ids:
                    new_cell_match.append(pdf_match)

            if len(new_cell_match) > 0:
                new_matches[pdf_cell_id] = new_cell_match

        return new_table_cells, new_matches, new_tab_columns

    def _do_final_asignment(self, table_cells, iou_matches, ioc_matches):
        r"""
        8. step

        Do final assignment of table bbox to pdf cell based on saved scores,
        either preferring IOU over PDF Intersection, and higher Intersection over lower,
        or just use PDF Intersection
        Rule: 1 Table cell can contain many PDF cells,
            but each PDF cell has to be asigned to one Table cell
        Rule: Do not discard table bboxes at this point, asign all of them

        Iterate over matches, if PDF cell has more than 1 table cell match:
        Go over all other matches and delete tab_cell match of lower score
        (prefer iou match over ioc match)

        Parameters
        ----------
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        iou_matches : dictionary of lists of table_cells
            Cell matches done using Intersection Over Union (IOU) method
        ioc_matches : dictionary of lists of table_cells
            Cell matches done using Intersection Over (PDF) Cell method

        Returns
        -------
        new_matches : dictionary of lists of table_cells
            New matches with final table cell asignments
        """
        new_matches = {}

        for pdf_cell_id, pdf_cell_matches in ioc_matches.items():
            max_ioc_match = max(pdf_cell_matches, key=lambda x: x["iopdf"])
            new_matches[pdf_cell_id] = [max_ioc_match]

        return new_matches

    def _merge_two_bboxes(self, bbox1, bbox2):
        r"""
        Merge two bboxes into one bboxes that encompasses the two

        Parameters
        ----------
        bbox1 : list of numbers
            bbox to be merged described as two corners [x1, y1, x2, y2]
        bbox1 : list of numbers
            bbox to be merged described as two corners [x1, y1, x2, y2]

        Returns
        -------
        bbox_result : list of numbers
            bbox that encompasses two input bboxes
        """
        bbox_result = [-1, -1, -1, -1]
        bbox_result[0] = min([bbox1[0], bbox2[0]])
        bbox_result[1] = min([bbox1[1], bbox2[1]])
        bbox_result[2] = max([bbox1[2], bbox2[2]])
        bbox_result[3] = max([bbox1[3], bbox2[3]])
        return bbox_result

    def _pick_orphan_cells(
        self, tab_rows, tab_cols, max_cell_id, table_cells, pdf_cells, matches
    ):
        # 9.
        # new_matches, new_table_cells, max_cell_id
        r"""
        9. step

        Pick up remaining orphan cells (pdf cells that don't have any matches or intersections)
            9.a. Determine row banding per row (min/max Y per row)
                match with orphan cells (intersection with band)
            9.b. Determine column banding per column (min/max X per column)
                match with orphan cells (intersection with band)
            9.c. Decide to which column/row each orphan PDF cell belongs
                append match to existing structural cell, or create cell

        Parameters
        ----------
        tab_rows : number
            Total number of rows
        tab_cols : number
            Total number of columns
        max_cell_id : number
            Highest table cell id
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        pdf_cells : list of dict
            List of PDF cells as defined by Docling
        matches : dictionary of lists of table_cells
            A dictionary which is indexed by the pdf_cell_id as key and the value is a list
            of the table_cells that fall inside that pdf cell

        Returns
        -------
        new_matches : dictionary of lists of table_cells
            updated matches
        new_table_cells : list of dict
            updated table cells
        max_cell_id : number
            New highest table cell id, accounting freshly added table cells (if any)
        """

        new_matches = matches
        new_table_cells = table_cells

        # Identify orphan rows (START)
        orphan_rows = []
        orphan_rows_depth = []
        orphan_rows_bbox = []

        # List with pdf_ids which are used in some (any) row
        used_row_pdf_ids = []
        used_row_rowid = []

        for row in range(tab_rows):
            bbox_y1s = []  # y2 > y1
            bbox_y2s = []
            row_y1 = -1
            row_y2 = -1
            row_table_cells = []
            row_table_cell_ids = []
            orphan_cells_in_row = []
            orphan_cells_in_row_depth = []
            orphan_cells_in_row_bbox = []

            for cell in table_cells:
                if cell["row_id"] == row:
                    # Do not consider spanned cells
                    if "rowspan_val" not in cell:
                        # Do not consider empty cells
                        if cell["cell_class"] > 1:
                            table_cell_id = cell["cell_id"]
                            row_table_cells.append(cell)
                            row_table_cell_ids.append(table_cell_id)
                            bbox_y1s.append(cell["bbox"][1])
                            bbox_y2s.append(cell["bbox"][3])

            # Y coordinates that define band of rows
            if len(bbox_y1s) > 0:
                row_y1 = min(bbox_y1s)
            if len(bbox_y2s) > 0:
                row_y2 = max(bbox_y2s)

            # Find "orphan" cells that intersect the band
            for pdf_cell in pdf_cells:
                pdf_str_id = str(pdf_cell["id"])
                if pdf_str_id not in matches:
                    within_band = False
                    depth = -1

                    centroid_band = (row_y2 + row_y1) / 2
                    centroid_cell = (pdf_cell["bbox"][3] + pdf_cell["bbox"][1]) / 2

                    # pdf_cell - Orphan, and don't have any match
                    if pdf_cell["bbox"][1] >= row_y1 and pdf_cell["bbox"][1] <= row_y2:
                        depth = abs(centroid_band - centroid_cell)
                        within_band = True

                    if pdf_cell["bbox"][3] >= row_y1 and pdf_cell["bbox"][3] <= row_y2:
                        depth = abs(centroid_band - centroid_cell)
                        within_band = True

                    if pdf_cell["bbox"][1] <= row_y1 and pdf_cell["bbox"][3] >= row_y2:
                        depth = abs(centroid_band - centroid_cell)
                        within_band = True

                    if within_band:
                        if pdf_str_id not in used_row_pdf_ids:
                            used_row_pdf_ids.append(pdf_str_id)
                            used_row_rowid.append(row)
                            orphan_cells_in_row.append(pdf_str_id)
                            orphan_cells_in_row_depth.append(round(depth))
                            orphan_cells_in_row_bbox.append(pdf_cell["bbox"])
                        else:
                            self._log().debug("Found duplicate: {}".format(pdf_str_id))
                            # Get index of a row where pdf cell was already detected
                            used_ind = used_row_pdf_ids.index(pdf_str_id)
                            duplicate_id_found_in_row = used_row_rowid[used_ind]
                            valid_duplicate = False
                            if len(orphan_rows) > duplicate_id_found_in_row:
                                index_in_row_list = orphan_rows[
                                    duplicate_id_found_in_row
                                ].index(pdf_str_id)
                                valid_duplicate = True

                            if valid_duplicate:
                                i1 = duplicate_id_found_in_row
                                i2 = index_in_row_list
                                score1 = orphan_rows_depth[i1][i2]
                                score2 = round(depth)
                                # If new cell better than the old one
                                if score2 < score1:
                                    # Delete old record about the pdf cell...
                                    orphan_rows[i1].pop(index_in_row_list)
                                    orphan_rows_depth[i1].pop(index_in_row_list)
                                    orphan_rows_bbox[i1].pop(index_in_row_list)

                                    used_row_pdf_ids.pop(used_ind)
                                    used_row_rowid.pop(used_ind)
                                    # Then proceed adding new cell
                                    used_row_pdf_ids.append(pdf_str_id)
                                    used_row_rowid.append(row)
                                    orphan_cells_in_row.append(pdf_str_id)
                                    orphan_cells_in_row_depth.append(round(depth))
                                    orphan_cells_in_row_bbox.append(pdf_cell["bbox"])
                                    msg = "Resolved duplicate: {} in favor of new one"
                                    self._log().debug(msg.format(pdf_str_id))
                                else:
                                    msg = "Resolved duplicate: {} in favor of old one"
                                    self._log().debug(msg.format(pdf_str_id))

            log_msg = "Row: {}, Band: {}/{}, Orphan PDF cells: {}"
            self._log().debug(log_msg.format(row, row_y1, row_y2, orphan_cells_in_row))
            orphan_rows.append(orphan_cells_in_row)
            orphan_rows_depth.append(orphan_cells_in_row_depth)
            orphan_rows_bbox.append(orphan_cells_in_row_bbox)

        # Identify orphan rows (END)
        self._log().debug("...")
        # Identify orphan columns
        orphan_columns = []
        orphan_columns_depth = []
        orphan_columns_bbox = []
        used_col_pdf_ids = []
        used_col_columnid = []

        for col in range(tab_cols):
            bbox_x1s = []  # y2 > y1
            bbox_x2s = []
            col_x1 = -1
            col_x2 = -1
            col_table_cells = []
            col_table_cell_ids = []
            orphan_cells_in_col = []
            orphan_cells_in_col_depth = []
            orphan_cells_in_col_bbox = []

            for cell in table_cells:
                if cell["column_id"] == col:
                    # Do not consider spanned cells
                    if "colspan_val" not in cell:
                        # Do not consider empty cells
                        if cell["cell_class"] > 1:
                            table_cell_id = cell["cell_id"]
                            col_table_cells.append(cell)
                            col_table_cell_ids.append(table_cell_id)
                            bbox_x1s.append(cell["bbox"][0])
                            bbox_x2s.append(cell["bbox"][2])
                    else:
                        wrn_txt = "Orphan matching skipped cell in column {} because of colspan"
                        self._log().debug(wrn_txt.format(col))
                        # self._log().info(cell)

            # X coordinates that define band of columns
            if len(bbox_x1s) > 0:
                col_x1 = min(bbox_x1s)
            if len(bbox_x2s) > 0:
                col_x2 = max(bbox_x2s)

            # Find "orphan" cells that intersect the band
            for pdf_cell in pdf_cells:
                pdf_str_id = str(pdf_cell["id"])
                if pdf_str_id not in matches:
                    within_band = False
                    depth = -1

                    centroid_band = (col_x2 + col_x1) / 2
                    centroid_cell = (pdf_cell["bbox"][2] + pdf_cell["bbox"][0]) / 2

                    if pdf_cell["bbox"][0] >= col_x1 and pdf_cell["bbox"][0] <= col_x2:
                        depth = abs(centroid_band - centroid_cell)
                        within_band = True

                    if pdf_cell["bbox"][2] >= col_x1 and pdf_cell["bbox"][2] <= col_x2:
                        depth = abs(centroid_band - centroid_cell)
                        within_band = True

                    if pdf_cell["bbox"][0] < col_x1 and pdf_cell["bbox"][2] > col_x2:
                        depth = abs(centroid_band - centroid_cell)
                        within_band = True

                    if within_band:
                        if pdf_str_id not in used_col_pdf_ids:
                            used_col_pdf_ids.append(pdf_str_id)
                            used_col_columnid.append(col)
                            orphan_cells_in_col.append(pdf_str_id)
                            orphan_cells_in_col_depth.append(round(depth))
                            orphan_cells_in_col_bbox.append(pdf_cell["bbox"])
                        else:
                            self._log().debug("Found duplicate: {}".format(pdf_str_id))
                            # Get index of a column where pdf cell was already detected
                            used_ind = used_col_pdf_ids.index(pdf_str_id)
                            duplicate_id_found_in_column = used_col_columnid[used_ind]

                            valid_col_duplicate = False
                            if len(orphan_columns) > duplicate_id_found_in_column:
                                index_in_col_list = orphan_columns[
                                    duplicate_id_found_in_column
                                ].index(pdf_str_id)
                                valid_col_duplicate = True

                            if valid_col_duplicate:
                                i1 = duplicate_id_found_in_column
                                i2 = index_in_col_list
                                score1 = orphan_columns_depth[i1][i2]
                                score2 = round(depth)
                                # If new cell better than the old one
                                if score2 < score1:
                                    # Delete old record about the pdf cell...
                                    orphan_columns[i1].pop(index_in_col_list)
                                    orphan_columns_depth[i1].pop(index_in_col_list)
                                    orphan_columns_bbox[i1].pop(index_in_col_list)

                                    used_col_pdf_ids.pop(used_ind)
                                    used_col_columnid.pop(used_ind)
                                    # Then proceed adding new cell
                                    used_col_pdf_ids.append(pdf_str_id)
                                    used_col_columnid.append(col)
                                    orphan_cells_in_col.append(pdf_str_id)
                                    orphan_cells_in_col_depth.append(round(depth))
                                    orphan_cells_in_col_bbox.append(pdf_cell["bbox"])
                                    msg = "Resolved duplicate: {} in favor of new one"
                                    self._log().debug(msg.format(pdf_str_id))
                                else:
                                    msg = "Resolved duplicate: {} in favor of old one"
                                    self._log().debug(msg.format(pdf_str_id))

            orphan_columns.append(orphan_cells_in_col)
            orphan_columns_depth.append(orphan_cells_in_col_depth)
            orphan_columns_bbox.append(orphan_cells_in_col_bbox)

        # Assign to structural cells and/or create new cells when absent

        for col_ind in range(len(orphan_columns)):
            self._log().debug(
                "Col: {}, Orphan PDF cells: {}".format(col_ind, orphan_columns[col_ind])
            )
            self._log().debug(
                "Col: {},     Orphan Depth: {}".format(
                    col_ind, orphan_columns_depth[col_ind]
                )
            )
        self._log().debug("...")

        # Collect the pdf_ids from the orphan_rows and sort them in order to produce the same
        # results with the c++ implementation
        orphan_rows_pdf_ids = []
        row_id_per_pdf_id = {}  # pdf_cell_id -> row_id
        for row_id, row_pdf_ids in enumerate(orphan_rows):
            if len(row_pdf_ids) == 0:
                continue
            # Extend the orphan_rows_pdf_ids with the pdf_ids as int
            orphan_rows_pdf_ids.extend([int(x) for x in row_pdf_ids])
            # Set the row_id for the pdf_ids
            for pdf_cell_id in row_pdf_ids:
                row_id_per_pdf_id[int(pdf_cell_id)] = row_id

        orphan_rows_pdf_ids.sort()

        # Assign Table cell Row ID / Table cell Column ID to orphans,
        # Check if Table cell doesn't exist in the table_cells, create one,
        # add match to new_matches
        for pdf_cell_id_int in orphan_rows_pdf_ids:
            new_row_id = row_id_per_pdf_id[pdf_cell_id_int]
            new_column_id = 0
            pdf_cell_id = str(pdf_cell_id_int)

            if pdf_cell_id in used_col_pdf_ids:
                new_column_id = used_col_columnid[used_col_pdf_ids.index(pdf_cell_id)]

                self._log().debug(
                    "new_column_id {}, pdf_cell_id {}".format(
                        new_column_id, pdf_cell_id
                    )
                )
                self._log().debug(orphan_columns[new_column_id])
                depth_index = orphan_columns[new_column_id].index(pdf_cell_id)
                confidence = orphan_columns_depth[new_column_id][depth_index]
                pdf_bbox = orphan_columns_bbox[new_column_id][depth_index]

                # 1. Find table_cell_id by new_row_id / new_column_id
                new_table_cell_id = -1
                tcell = list(
                    filter(
                        lambda table_cell: table_cell["row_id"] == new_row_id
                        and table_cell["column_id"] == new_column_id,
                        table_cells,
                    )
                )

                if len(tcell) > 0:
                    new_table_cell_id = tcell[0]["cell_id"]
                    self._log().debug(
                        "reusing table_cell_id: {}".format(new_table_cell_id)
                    )

                    for i in range(len(new_table_cells)):
                        if new_table_cells[i]["cell_id"] == new_table_cell_id:
                            bbox_tmp = self._merge_two_bboxes(
                                new_table_cells[i]["bbox"], pdf_bbox
                            )
                            new_table_cells[i]["bbox"] = bbox_tmp

                if new_table_cell_id < 0:
                    max_cell_id += 1
                    new_table_cell_id = max_cell_id

                    new_table_cell = {
                        "bbox": pdf_bbox,
                        "cell_id": new_table_cell_id,
                        "column_id": new_column_id,
                        "label": "body",
                        "row_id": new_row_id,
                        "cell_class": 2,
                    }
                    self._log().debug(
                        "making new table_cell_id: {}".format(new_table_cell_id)
                    )
                    new_table_cells.append(new_table_cell)

                # And then add new match to the new_matches
                new_matches[str(pdf_cell_id)] = [
                    {"post": confidence, "table_cell_id": new_table_cell_id}
                ]
        return new_matches, new_table_cells, max_cell_id

    def _clear_pdf_cells(self, pdf_cells):
        r"""
        Clean PDF cells from cells that have an empty string as text

        Parameters
        ----------
        pdf_cells : list of dict
            List of PDF cells as defined by Docling

        Returns
        -------
        new_pdf_cells : list of dict
            updated, cleaned list of pdf_cells
        """
        new_pdf_cells = []
        for i in range(len(pdf_cells)):
            if pdf_cells[i]["text"] != "":
                new_pdf_cells.append(pdf_cells[i])
        return new_pdf_cells

    def process(self, matching_details, correct_overlapping_cells=False):
        r"""
        Do post processing, see details in the comments below

        Parameters
        ----------
        matching_details : dictionary
            contains all the necessary information for Docling processing
            already has predictions and initial (IOU) matches

        Returns
        -------
        matching_details : dictionary
            matching_details that contain post-processed matches
        """

        # ====================================================================================
        # Start post-processing
        # ====================================================================================

        # General description of post-processing algorithm to improve matching

        # Uses: (IOU) matches, pdf_cells, table_cells
        # Generates: new_table_cells, new_matches

        # +0. Get minimal grid table dimension (cols/rows)
        # +1. Get good/bad IOU predicted cells for each structural column (of minimal grid)
        #   +1.a. If all IOU in a column are bad - eliminate column (from bboxes and structure)
        # +2. Find alignment of good IOU cells per column
        #   +2.a. For this, measure min-max distance for left, min-max distance for centroid,
        #         min-max distance for right side of cell rectangles
        #         smallest distance would determine Left / Center / Right alignment
        # +3. Get median* (according to alignment) "bbox left/middle/right X" coord
        #         for good IOU cells, get median* cell size in a column.
        # +4. Move bad cells to the median* (left/middle/right) good in a column
        #   +4.a. (Additionally), re-size cell to median* size of cells in a column
        # +5. Generate new matches, run Intersection over cell(pdf) on a table cells

        # NOT USED +6. Align table cell bboxes with good matches
        #              to encapsulate matching pdf cells

        # +7. De-duplicate columns in table_cells according to highest column score
        #        in: matches + intersection_pdf_matches
        # +8. Do final assignment of table bbox to pdf cell based on saved scores,
        #     either preferring IOU over PDF Intersection, and higher Intersection over lower,
        #     or just use PDF Intersection
        #     Rule: 1 Table cell can contain many PDF cells,
        #       but each PDF cell has to be asigned to one Table cell
        #     Rule: Do not discard table bboxes at this point, asign all of them
        #   +8.a. Align table cell bboxes with matched pdf cells bboxes (instead of step 6)

        # +9. Pick up remaining orphan cells
        #   +9.a. Determine row banding per row (min/max Y per row)
        #           match with orphan cells (intersection with band)
        #   +9.b. Determine column banding per column (min/max X per column)
        #           match with orphan cells (intersection with band)
        #   +9.c. Decide to which column/row each orphan PDF cell belongs
        #           append match to existing structural cell,
        #           or create cell
        #  ?9.d. For each pdf-cell from the list find closest and MATCH with row-table-cell
        #        (minimal rectangle to-rectangle distance)

        # *Use median instead of average to account for occasional
        # colspans that would span across columns,
        # as they are rare they shouldn't indfluence much a median position
        # of other cells in a minimal-grid column

        self._log().debug("Start prediction post-processing...")
        table_cells = matching_details["table_cells"]
        pdf_cells = self._clear_pdf_cells(matching_details["pdf_cells"])
        matches = matching_details["matches"]

        # ------------------------------------------------------------------------------------------
        # -1. If initial (IOU) matches are empty,
        # generate new ones based on intersection over cell

        if not matches:
            self._log().debug(
                "-----------------------------------------------------------------"
            )
            self._log().debug(
                "-----------------------------------------------------------------"
            )
            self._log().debug(
                "-   NO INITIAL MATCHES TO POST PROCESS, GENERATING NEW ONES...  -"
            )
            self._log().debug(
                "-----------------------------------------------------------------"
            )
            self._log().debug(
                "-----------------------------------------------------------------"
            )
            matches = self._run_intersection_match(
                self._cell_matcher, table_cells, pdf_cells
            )

        # ------------------------------------------------------------------------------------------
        # 0. Get minimal grid table dimension (cols/rows)
        tab_columns, tab_rows, max_cell_id = self._get_table_dimension(table_cells)
        self._log().debug(
            "COLS {}/ ROWS {}/ MAX CELL ID {}".format(
                tab_columns, tab_rows, max_cell_id
            )
        )

        good_table_cells = []
        bad_table_cells = []
        new_bad_table_cells = []
        fixed_table_cells = []

        # 1. Get good/bad IOU predicted cells for each structural column (of minimal grid)
        for col in range(tab_columns):
            g1, g2 = self._get_good_bad_cells_in_column(table_cells, col, matches)
            good_table_cells = g1
            bad_table_cells = g2
            self._log().debug(
                "COLUMN {}, Good table cells: {}".format(col, len(good_table_cells))
            )
            self._log().debug(
                "COLUMN {}, Bad table cells: {}".format(col, len(bad_table_cells))
            )

            # 2. Find alignment of good IOU cells per column
            alignment = self._find_alignment_in_column(good_table_cells)
            self._log().debug("COLUMN {}, Alignment: {}".format(col, alignment))
            # alignment = "left"

            # 3. Get median (according to alignment) "bbox left/middle/right X"
            #    coordinate for good IOU cells, get median* cell size in a column.
            gm1, gm2, gm3, gm4 = self._get_median_pos_size(good_table_cells, alignment)
            median_x = gm1
            # median_y = gm2
            median_width = gm3
            median_height = gm4
            self._log().debug("Median good X = {}".format(median_x))

            # 4. Move bad cells to the median* (left/middle/right) good in a column
            # nc = self._move_cells_to_left_pos(bad_table_cells, median_x, True,
            # TODO:
            nc = self._move_cells_to_left_pos(
                bad_table_cells, median_x, False, median_width, median_height, alignment
            )
            new_bad_table_cells = nc
            fixed_table_cells.extend(good_table_cells)
            fixed_table_cells.extend(new_bad_table_cells)

        # ====================================================================================
        # Sort table_cells by cell_id before running IOU, to have correct indexes on the output
        fixed_table_cells_sorted = sorted(fixed_table_cells, key=lambda k: k["cell_id"])

        # 5. Generate new matches, run Intersection over cell(pdf) on a table cells
        ip = self._run_intersection_match(
            self._cell_matcher, fixed_table_cells_sorted, pdf_cells
        )
        intersection_pdf_matches = ip

        # 6. NOT USED

        # 7. De-duplicate columns in aligned_table_cells
        # according to highest column score in: matches + intersection_pdf_matches
        # (this is easier now, because duplicated cells will have same bboxes)
        dd1, dd2, dd3 = self._deduplicate_cells(
            tab_columns, fixed_table_cells_sorted, matches, intersection_pdf_matches
        )
        dedupl_table_cells = dd1
        dedupl_matches = dd2

        self._log().debug("...")

        # 8. Do final assignment of table bbox to pdf cell based on saved scores,
        # preferring IOU over PDF Intersection, and higher Intersection over lower
        # ! IOU matches currently disabled,
        # and final assigment is done only on IOC matches
        final_matches = self._do_final_asignment(
            dedupl_table_cells, matches, dedupl_matches
        )

        # 8.a. Re-align bboxes / re-run matching
        dedupl_table_cells_sorted = sorted(
            dedupl_table_cells, key=lambda k: k["cell_id"]
        )

        if (
            len(pdf_cells) > 300
        ):  # For performance, skip this step if there are too many pdf_cells
            aligned_table_cells2 = dedupl_table_cells_sorted
        else:
            aligned_table_cells2 = self._align_table_cells_to_pdf(
                dedupl_table_cells_sorted, pdf_cells, final_matches
            )

        # 9. Distance-match orphans
        po1, po2, po3 = self._pick_orphan_cells(
            tab_rows,
            tab_columns,
            max_cell_id,
            aligned_table_cells2,
            pdf_cells,
            final_matches,
        )
        final_matches_wo = po1
        table_cells_wo = po2
        max_cell_id = po3

        if correct_overlapping_cells:
            # As the last step - correct cell bboxes in a way that they don't overlap:
            if len(table_cells_wo) <= 300:  # For performance reasons
                table_cells_wo = self._find_overlapping(table_cells_wo)

        self._log().debug("*** final_matches_wo")
        self._log().debug(final_matches_wo)
        self._log().debug("*** table_cells_wo")
        self._log().debug(table_cells_wo)

        for pdf_cell_id in range(len(final_matches_wo)):
            if str(pdf_cell_id) in final_matches_wo:
                pdf_cell_match = final_matches_wo[str(pdf_cell_id)]
                if len(pdf_cell_match) > 1:
                    l1 = "!!! Multiple - {}x pdf cell match with id: {}"
                    self._log().info(l1.format(len(pdf_cell_match), pdf_cell_id))
                if pdf_cell_match:
                    tcellid = pdf_cell_match[0]["table_cell_id"]
                    for tcell in table_cells_wo:
                        if tcell["cell_id"] == tcellid:
                            mrow = tcell["row_id"]
                            mcol = tcell["column_id"]
                            l2 = "pdf cell: {} -> row: {} | col:{}"
                            self._log().debug(l2.format(pdf_cell_id, mrow, mcol))
            else:
                self._log().debug(
                    "!!! pdf cell doesn't have match: {}".format(pdf_cell_id)
                )

        # Example of an object:
        # matching_details = {
        #    "iou_threshold": self._iou_thres,
        #    "table_bbox": table_bbox,
        #    "prediction": prediction,
        #    "pdf_cells": pdf_cells,
        #    "page_height": docling_table["page_height"],
        #    "page_width": docling_table["page_width"],
        #    "table_cells": table_cells,
        #    "matches": matches
        # }

        matching_details["table_cells"] = table_cells_wo
        matching_details["matches"] = final_matches_wo
        matching_details["pdf_cells"] = pdf_cells

        self._log().debug("Done prediction matching and post-processing!")
        return matching_details
