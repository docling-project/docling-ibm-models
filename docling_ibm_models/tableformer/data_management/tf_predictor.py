#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import glob
import json
import logging
import os
import threading
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_model

import docling_ibm_models.tableformer.common as c
import docling_ibm_models.tableformer.data_management.transforms as T
import docling_ibm_models.tableformer.settings as s
import docling_ibm_models.tableformer.utils.utils as u
from docling_ibm_models.tableformer.data_management.matching_post_processor import (
    MatchingPostProcessor,
)
from docling_ibm_models.tableformer.data_management.tf_cell_matcher import CellMatcher
from docling_ibm_models.tableformer.models.common.base_model import BaseModel
from docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs import (
    TableModel04_rs,
)
from docling_ibm_models.tableformer.otsl import otsl_to_html
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler

# LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG
LOG_LEVEL = logging.WARN

logger = s.get_custom_logger(__name__, LOG_LEVEL)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def otsl_sqr_chk(rs_list, logdebug):
    rs_list_split = [
        list(group) for k, group in groupby(rs_list, lambda x: x == "nl") if not k
    ]
    isSquare = True
    if len(rs_list_split) > 0:
        init_tag_len = len(rs_list_split[0]) + 1

        totcelnum = rs_list.count("fcel") + rs_list.count("ecel")
        if logdebug:
            logger.debug("Total number of cells = {}".format(totcelnum))

        for ind, ln in enumerate(rs_list_split):
            ln.append("nl")
            if logdebug:
                logger.debug("{}".format(ln))
            if len(ln) != init_tag_len:
                isSquare = False
        if isSquare:
            if logdebug:
                logger.debug(
                    "{}*OK* Table is square! *OK*{}".format(
                        bcolors.OKGREEN, bcolors.ENDC
                    )
                )
        else:
            if logdebug:
                err_name = "{}***** ERR ******{}"
                logger.debug(err_name.format(bcolors.FAIL, bcolors.ENDC))
                logger.debug(
                    "{}*ERR* Table is not square! *ERR*{}".format(
                        bcolors.FAIL, bcolors.ENDC
                    )
                )
    return isSquare


class TFPredictor:
    r"""
    Table predictions for the in-memory Docling API
    """

    def __init__(self, config, device: str = "cpu", num_threads: int = 4):
        r"""
        Parameters
        ----------
        config : dict Parameters configuration
        device: (Optional) torch device to run the inference.
        num_threads: (Optional) Number of threads to run the inference if device = 'cpu'

        Raises
        ------
        ValueError
        When the model cannot be found
        """
        # self._device = torch.device(device)
        self._device = device
        self._log().info("Running on device: {}".format(device))

        self._config = config
        self.enable_post_process = True

        self._padding = config["predict"].get("padding", False)
        self._padding_size = config["predict"].get("padding_size", 10)

        self._cell_matcher = CellMatcher(config)
        self._post_processor = MatchingPostProcessor(config)

        self._init_word_map()

        # Set the number of threads
        if device == "cpu":
            self._num_threads = num_threads
            torch.set_num_threads(self._num_threads)

        # Load the model
        self._model = self._load_model()
        self._model.eval()
        self._prof = config["predict"].get("profiling", False)
        self._profiling_agg_window = config["predict"].get("profiling_agg_window", None)
        if self._profiling_agg_window is not None:
            AggProfiler(self._profiling_agg_window)
        else:
            AggProfiler()

    def _init_word_map(self):
        self._prepared_data_dir = c.safe_get_parameter(
            self._config, ["dataset", "prepared_data_dir"], required=False
        )

        if self._prepared_data_dir is None:
            self._word_map = c.safe_get_parameter(
                self._config, ["dataset_wordmap"], required=True
            )
        else:
            data_name = c.safe_get_parameter(
                self._config, ["dataset", "name"], required=True
            )
            word_map_fn = c.get_prepared_data_filename("WORDMAP", data_name)

            # Load word_map
            with open(os.path.join(self._prepared_data_dir, word_map_fn), "r") as f:
                self._log().debug("Load WORDMAP from: {}".format(word_map_fn))
                self._word_map = json.load(f)

        self._init_data = {"word_map": self._word_map}
        # Prepare a reversed index for the word map
        self._rev_word_map = {v: k for k, v in self._word_map["word_map_tag"].items()}

    def get_init_data(self):
        r"""
        Return the initialization data
        """
        return self._init_data

    def get_model(self):
        r"""
        Return the loaded model
        """
        return self._model

    def _load_model(self):
        r"""
        Load the proper model
        """

        self._model_type = self._config["model"]["type"]

        # Use lock to prevent threading issues during model initialization
        with _model_init_lock:
            model = TableModel04_rs(self._config, self._init_data, self._device)

            if model is None:
                err_msg = "Not able to initiate a model for {}".format(self._model_type)
                self._log().error(err_msg)
                raise ValueError(err_msg)

            self._remove_padding = False
            if self._model_type == "TableModel02":
                self._remove_padding = True

            # Load model from safetensors
            save_dir = self._config["model"]["save_dir"]
            models_fn = glob.glob(f"{save_dir}/tableformer_*.safetensors")
            if not models_fn:
                err_msg = "Not able to find a model file for {}".format(
                    self._model_type
                )
                self._log().error(err_msg)
                raise ValueError(err_msg)
            model_fn = models_fn[
                0
            ]  # Take the first tableformer safetensors file inside the save_dir
            missing, unexpected = load_model(model, model_fn, device=self._device)
            if missing or unexpected:
                err_msg = "Not able to load the model weights for {}".format(
                    self._model_type
                )
                self._log().error(err_msg)
                raise ValueError(err_msg)

        return model

    def get_device(self):
        return self._device

    def get_model_type(self):
        return self._model_type

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def _deletebbox(self, listofbboxes, index):
        newlist = []
        for i in range(len(listofbboxes)):
            bbox = listofbboxes[i]
            if i not in index:
                newlist.append(bbox)
        return newlist

    def _remove_bbox_span_desync(self, prediction):
        # Delete 1 extra bbox after span tag
        index_to_delete_from = 0
        indexes_to_delete = []
        newbboxes = []
        for html_elem in prediction["html_seq"]:
            if html_elem == "<td>":
                index_to_delete_from += 1
            if html_elem == ">":
                index_to_delete_from += 1
                # remove element from bboxes
                self._log().debug(
                    "========= DELETE BBOX INDEX: {}".format(index_to_delete_from)
                )
                indexes_to_delete.append(index_to_delete_from)

        newbboxes = self._deletebbox(prediction["bboxes"], indexes_to_delete)
        return newbboxes

    def _check_bbox_sync(self, prediction):
        bboxes = []
        match = False
        # count bboxes
        count_bbox = len(prediction["bboxes"])
        # count td tags
        count_td = 0
        for html_elem in prediction["html_seq"]:
            if html_elem == "<td>" or html_elem == ">":
                count_td += 1
            if html_elem in ["fcel", "ecel", "ched", "rhed", "srow"]:
                count_td += 1
        self._log().debug(
            "======================= PREDICTED BBOXES: {}".format(count_bbox)
        )
        self._log().debug(
            "=======================  PREDICTED CELLS: {}".format(count_td)
        )
        if count_bbox != count_td:
            bboxes = self._remove_bbox_span_desync(prediction)
        else:
            bboxes = prediction["bboxes"]
            match = True
        return match, bboxes

    def page_coords_to_table_coords(self, bbox, table_bbox, im_width, im_height):
        r"""
        Transforms given bbox from page coordinate system into table image coordinate system

        Parameters
        ----------
        bbox : list
            bbox to transform in page coordinates
        table_bbox : list
            table bbox, in page coordinates
        im_width : integer
            width of an image with rendered table (in pixels)
        im_height : integer
            height of an image height rendered table (in pixels)

        Returns
        -------
        bbox: list
            bbox with transformed coordinates
        """
        # Coordinates of given bbox
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        # Coordinates of table bbox
        t_x1 = table_bbox[0]
        t_y1 = table_bbox[1]
        t_x2 = table_bbox[2]
        t_y2 = table_bbox[3]

        # Table width / height
        tw = t_x2 - t_x1
        th = t_y2 - t_y1
        new_bbox = [0, 0, 0, 0]
        # Flip corners, substract table coordinates and rescale to new image size
        new_bbox[0] = im_width * (x1 - t_x1) / tw
        new_bbox[1] = im_height * (t_y2 - y2) / th
        new_bbox[2] = im_width * (x2 - t_x1) / tw
        new_bbox[3] = im_height * (t_y2 - y1) / th

        return new_bbox

    def _depad_bboxes(self, bboxes, new_image_ratio):
        r"""
        Removes padding from predicted bboxes for previously padded image

        Parameters
        ----------
        bboxes : list of lists
            list of bboxes that have to be recalculated to remove implied padding
        new_image_ratio : float
            Ratio of padded image size to the original image size

        Returns
        -------
        new_bboxes: list
            bboxes with transformed coordinates
        """
        new_bboxes = []
        c_x = 0.5
        c_y = 0.5

        self._log().debug("PREDICTED BBOXES: {}".format(bboxes))
        self._log().debug("new_image_ratio: {}".format(new_image_ratio))

        for bbox in bboxes:
            # 1. corner coords -> center coords
            cb_x1 = bbox[0] - c_x
            cb_y1 = bbox[1] - c_y
            cb_x2 = bbox[2] - c_x
            cb_y2 = bbox[3] - c_y

            # 2. center coords * new_image_ratio
            r_cb_x1 = cb_x1 * new_image_ratio
            r_cb_y1 = cb_y1 * new_image_ratio
            r_cb_x2 = cb_x2 * new_image_ratio
            r_cb_y2 = cb_y2 * new_image_ratio

            # 3. center coords -> corner coords
            x1 = r_cb_x1 + c_x
            y1 = r_cb_y1 + c_y
            x2 = r_cb_x2 + c_x
            y2 = r_cb_y2 + c_y

            x1 = np.clip(x1, 0.0, 1.0)
            y1 = np.clip(y1, 0.0, 1.0)
            x2 = np.clip(x2, 0.0, 1.0)
            y2 = np.clip(y2, 0.0, 1.0)

            new_bbox = [x1, y1, x2, y2]
            new_bboxes.append(new_bbox)

        self._log().debug("DEPAD BBOXES: {}".format(new_bboxes))

        return new_bboxes

    def _merge_tf_output(self, docling_output, pdf_cells):
        tf_output = []
        tf_cells_map = {}
        max_row_idx = 0

        for docling_item in docling_output:
            r_idx = str(docling_item["start_row_offset_idx"])
            c_idx = str(docling_item["start_col_offset_idx"])
            cell_key = c_idx + "_" + r_idx
            if cell_key in tf_cells_map:
                for pdf_cell in pdf_cells:
                    if pdf_cell["id"] == docling_item["cell_id"]:
                        text_cell_bbox = {
                            "b": pdf_cell["bbox"][3],
                            "l": pdf_cell["bbox"][0],
                            "r": pdf_cell["bbox"][2],
                            "t": pdf_cell["bbox"][1],
                            "token": pdf_cell["text"],
                        }
                        tf_cells_map[cell_key]["text_cell_bboxes"].append(
                            text_cell_bbox
                        )
            else:
                tf_cells_map[cell_key] = {
                    "bbox": docling_item["bbox"],
                    "row_span": docling_item["row_span"],
                    "col_span": docling_item["col_span"],
                    "start_row_offset_idx": docling_item["start_row_offset_idx"],
                    "end_row_offset_idx": docling_item["end_row_offset_idx"],
                    "start_col_offset_idx": docling_item["start_col_offset_idx"],
                    "end_col_offset_idx": docling_item["end_col_offset_idx"],
                    "indentation_level": docling_item["indentation_level"],
                    "text_cell_bboxes": [],
                    "column_header": docling_item["column_header"],
                    "row_header": docling_item["row_header"],
                    "row_section": docling_item["row_section"],
                }

                if docling_item["start_row_offset_idx"] > max_row_idx:
                    max_row_idx = docling_item["start_row_offset_idx"]

                for pdf_cell in pdf_cells:
                    if pdf_cell["id"] == docling_item["cell_id"]:
                        text_cell_bbox = {
                            "b": pdf_cell["bbox"][3],
                            "l": pdf_cell["bbox"][0],
                            "r": pdf_cell["bbox"][2],
                            "t": pdf_cell["bbox"][1],
                            "token": pdf_cell["text"],
                        }
                        tf_cells_map[cell_key]["text_cell_bboxes"].append(
                            text_cell_bbox
                        )

        for k in tf_cells_map:
            tf_output.append(tf_cells_map[k])
        return tf_output

    def resize_img(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]
        sf = 1.0
        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image, sf
        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            sf = r
            dim = (int(w * r), height)
        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            sf = r
            dim = (width, int(h * r))
        # resize the image
        # TODO(Nikos): Try to remove cv2 dependency
        resized = cv2.resize(image, dim, interpolation=inter)
        # return the resized image
        return resized, sf

    def multi_table_predict(
        self,
        iocr_page,
        table_bboxes,
        do_matching=True,
        correct_overlapping_cells=False,
        sort_row_col_indexes=True,
    ):
        multi_tf_output = []
        page_image = iocr_page["image"]

        # Prevent large image submission, by resizing input
        page_image_resized, scale_factor = self.resize_img(page_image, height=1024)

        for table_bbox in table_bboxes:
            # Downscale table bounding box to the size of new image
            table_bbox[0] = table_bbox[0] * scale_factor
            table_bbox[1] = table_bbox[1] * scale_factor
            table_bbox[2] = table_bbox[2] * scale_factor
            table_bbox[3] = table_bbox[3] * scale_factor

            table_image = page_image_resized[
                round(table_bbox[1]) : round(table_bbox[3]),
                round(table_bbox[0]) : round(table_bbox[2]),
            ]
            # table_image = page_image
            # Predict
            if do_matching:
                tf_responses, predict_details = self.predict(
                    iocr_page,
                    table_bbox,
                    table_image,
                    scale_factor,
                    None,
                    correct_overlapping_cells,
                )
            else:
                tf_responses, predict_details = self.predict_dummy(
                    iocr_page, table_bbox, table_image, scale_factor, None
                )

            # ======================================================================================
            # PROCESS PREDICTED RESULTS, TO TURN PREDICTED COL/ROW IDs into Indexes
            # Indexes should be in increasing order, without gaps

            if sort_row_col_indexes:
                # Fix col/row indexes
                # Arranges all col/row indexes sequentially without gaps using input IDs

                indexing_start_cols = (
                    []
                )  # Index of original start col IDs (not indexes)
                indexing_end_cols = []  # Index of original end col IDs (not indexes)
                indexing_start_rows = (
                    []
                )  # Index of original start row IDs (not indexes)
                indexing_end_rows = []  # Index of original end row IDs (not indexes)

                # First, collect all possible predicted IDs, to be used as indexes
                # ID's returned by Tableformer are sequential, but might contain gaps
                for tf_response_cell in tf_responses:
                    start_col_offset_idx = tf_response_cell["start_col_offset_idx"]
                    end_col_offset_idx = tf_response_cell["end_col_offset_idx"]
                    start_row_offset_idx = tf_response_cell["start_row_offset_idx"]
                    end_row_offset_idx = tf_response_cell["end_row_offset_idx"]

                    # Collect all possible col/row IDs:
                    if start_col_offset_idx not in indexing_start_cols:
                        indexing_start_cols.append(start_col_offset_idx)
                    if end_col_offset_idx not in indexing_end_cols:
                        indexing_end_cols.append(end_col_offset_idx)
                    if start_row_offset_idx not in indexing_start_rows:
                        indexing_start_rows.append(start_row_offset_idx)
                    if end_row_offset_idx not in indexing_end_rows:
                        indexing_end_rows.append(end_row_offset_idx)

                indexing_start_cols.sort()
                indexing_end_cols.sort()
                indexing_start_rows.sort()
                indexing_end_rows.sort()

                # After this - put actual indexes of IDs back into predicted structure...
                for tf_response_cell in tf_responses:
                    tf_response_cell["start_col_offset_idx"] = (
                        indexing_start_cols.index(
                            tf_response_cell["start_col_offset_idx"]
                        )
                    )
                    tf_response_cell["end_col_offset_idx"] = (
                        tf_response_cell["start_col_offset_idx"]
                        + tf_response_cell["col_span"]
                    )
                    tf_response_cell["start_row_offset_idx"] = (
                        indexing_start_rows.index(
                            tf_response_cell["start_row_offset_idx"]
                        )
                    )
                    tf_response_cell["end_row_offset_idx"] = (
                        tf_response_cell["start_row_offset_idx"]
                        + tf_response_cell["row_span"]
                    )
                # Counting matched cols/rows from actual indexes (and not ids)
                predict_details["num_cols"] = len(indexing_end_cols)
                predict_details["num_rows"] = len(indexing_end_rows)
            else:
                otsl_seq = predict_details["prediction"]["rs_seq"]
                predict_details["num_cols"] = otsl_seq.index("nl")
                predict_details["num_rows"] = otsl_seq.count("nl")

            # Put results into multi_tf_output
            multi_tf_output.append(
                {"tf_responses": tf_responses, "predict_details": predict_details}
            )
            # Upscale table bounding box back, for visualization purposes
            table_bbox[0] = table_bbox[0] / scale_factor
            table_bbox[1] = table_bbox[1] / scale_factor
            table_bbox[2] = table_bbox[2] / scale_factor
            table_bbox[3] = table_bbox[3] / scale_factor
        # Return grouped results of predictions
        return multi_tf_output

    def predict_dummy(
        self, iocr_page, table_bbox, table_image, scale_factor, eval_res_preds=None
    ):
        r"""
        Predict the table out of an image in memory

        Parameters
        ----------
        iocr_page : dict
            Docling provided table data
        eval_res_preds : dict
            Ready predictions provided by the evaluation results

        Returns
        -------
        docling_output : string
            json response formatted according to Docling api expectations

        matching_details : string
            json with details about the matching between the pdf cells and the table cells
        """
        AggProfiler().start_agg(self._prof)

        max_steps = self._config["predict"]["max_steps"]
        beam_size = self._config["predict"]["beam_size"]
        image_batch = self._prepare_image(table_image)
        # Make predictions
        prediction = {}

        with torch.no_grad():
            # Compute predictions
            if (
                eval_res_preds is not None
            ):  # Don't run the model, use the provided predictions
                prediction["bboxes"] = eval_res_preds["bboxes"]
                pred_tag_seq = eval_res_preds["tag_seq"]
            elif self._config["predict"]["bbox"]:
                pred_tag_seq, outputs_class, outputs_coord = self._model.predict(
                    image_batch, max_steps, beam_size
                )

                if outputs_coord is not None:
                    if len(outputs_coord) == 0:
                        prediction["bboxes"] = []
                    else:
                        bbox_pred = u.box_cxcywh_to_xyxy(outputs_coord)
                        prediction["bboxes"] = bbox_pred.tolist()
                else:
                    prediction["bboxes"] = []

                if outputs_class is not None:
                    if len(outputs_class) == 0:
                        prediction["classes"] = []
                    else:
                        result_class = torch.argmax(outputs_class, dim=1)
                        prediction["classes"] = result_class.tolist()
                else:
                    prediction["classes"] = []
                if self._remove_padding:
                    pred_tag_seq, _ = u.remove_padding(pred_tag_seq)
            else:
                pred_tag_seq, _, _ = self._model.predict(
                    image_batch, max_steps, beam_size
                )
                # Check if padding should be removed
                if self._remove_padding:
                    pred_tag_seq, _ = u.remove_padding(pred_tag_seq)

            prediction["tag_seq"] = pred_tag_seq
            prediction["rs_seq"] = self._get_html_tags(pred_tag_seq)
            prediction["html_seq"] = otsl_to_html(prediction["rs_seq"], False)
        # Remove implied padding from bbox predictions,
        # that we added on image pre-processing stage
        self._log().debug("----- rs_seq -----")
        self._log().debug(prediction["rs_seq"])
        self._log().debug(len(prediction["rs_seq"]))
        otsl_sqr_chk(prediction["rs_seq"], False)

        # Check that bboxes are in sync with predicted tags
        sync, corrected_bboxes = self._check_bbox_sync(prediction)
        if not sync:
            prediction["bboxes"] = corrected_bboxes

        # Match the cells
        matching_details = {
            "table_cells": [],
            "matches": {},
            "pdf_cells": [],
            "prediction_bboxes_page": [],
        }

        # Table bbox upscaling will scale predicted bboxes too within cell matcher
        scaled_table_bbox = [
            table_bbox[0] / scale_factor,
            table_bbox[1] / scale_factor,
            table_bbox[2] / scale_factor,
            table_bbox[3] / scale_factor,
        ]

        if len(prediction["bboxes"]) > 0:
            matching_details = self._cell_matcher.match_cells_dummy(
                iocr_page, scaled_table_bbox, prediction
            )
            # Generate the expected Docling responses
            AggProfiler().begin("generate_docling_response", self._prof)
            docling_output = self._generate_tf_response_dummy(
                matching_details["table_cells"]
            )

            AggProfiler().end("generate_docling_response", self._prof)
            # Add the docling_output sorted by cell_id into the matching_details
            docling_output.sort(key=lambda item: item["cell_id"])
            matching_details["docling_responses"] = docling_output
            # Merge docling_output and pdf_cells into one TF output,
            # with deduplicated table cells
            # tf_output = self._merge_tf_output_dummy(docling_output)
            tf_output = docling_output

        return tf_output, matching_details

    def predict(
        self,
        iocr_page,
        table_bbox,
        table_image,
        scale_factor,
        eval_res_preds=None,
        correct_overlapping_cells=False,
    ):
        r"""
        Predict the table out of an image in memory

        Parameters
        ----------
        iocr_page : dict
            Docling provided table data
        eval_res_preds : dict
            Ready predictions provided by the evaluation results
        correct_overlapping_cells : boolean
            Enables or disables last post-processing step, that fixes cell bboxes to remove overlap

        Returns
        -------
        docling_output : string
            json response formatted according to Docling api expectations

        matching_details : string
            json with details about the matching between the pdf cells and the table cells
        """
        AggProfiler().start_agg(self._prof)

        max_steps = self._config["predict"]["max_steps"]
        beam_size = self._config["predict"]["beam_size"]
        image_batch = self._prepare_image(table_image)
        # Make predictions
        prediction = {}

        with torch.no_grad():
            # Compute predictions
            if (
                eval_res_preds is not None
            ):  # Don't run the model, use the provided predictions
                prediction["bboxes"] = eval_res_preds["bboxes"]
                pred_tag_seq = eval_res_preds["tag_seq"]
            elif self._config["predict"]["bbox"]:
                pred_tag_seq, outputs_class, outputs_coord = self._model.predict(
                    image_batch, max_steps, beam_size
                )

                if outputs_coord is not None:
                    if len(outputs_coord) == 0:
                        prediction["bboxes"] = []
                    else:
                        bbox_pred = u.box_cxcywh_to_xyxy(outputs_coord)
                        prediction["bboxes"] = bbox_pred.tolist()
                else:
                    prediction["bboxes"] = []

                if outputs_class is not None:
                    if len(outputs_class) == 0:
                        prediction["classes"] = []
                    else:
                        result_class = torch.argmax(outputs_class, dim=1)
                        prediction["classes"] = result_class.tolist()
                else:
                    prediction["classes"] = []
                if self._remove_padding:
                    pred_tag_seq, _ = u.remove_padding(pred_tag_seq)
            else:
                pred_tag_seq, _, _ = self._model.predict(
                    image_batch, max_steps, beam_size
                )
                # Check if padding should be removed
                if self._remove_padding:
                    pred_tag_seq, _ = u.remove_padding(pred_tag_seq)

            prediction["tag_seq"] = pred_tag_seq
            prediction["rs_seq"] = self._get_html_tags(pred_tag_seq)
            prediction["html_seq"] = otsl_to_html(prediction["rs_seq"], False)
        # Remove implied padding from bbox predictions,
        # that we added on image pre-processing stage
        self._log().debug("----- rs_seq -----")
        self._log().debug(prediction["rs_seq"])
        self._log().debug(len(prediction["rs_seq"]))
        otsl_sqr_chk(prediction["rs_seq"], False)

        sync, corrected_bboxes = self._check_bbox_sync(prediction)
        if not sync:
            prediction["bboxes"] = corrected_bboxes

        # Match the cells
        matching_details = {
            "table_cells": [],
            "matches": {},
            "pdf_cells": [],
            "prediction_bboxes_page": [],
        }

        # Table bbox upscaling will scale predicted bboxes too within cell matcher
        scaled_table_bbox = [
            table_bbox[0] / scale_factor,
            table_bbox[1] / scale_factor,
            table_bbox[2] / scale_factor,
            table_bbox[3] / scale_factor,
        ]

        if len(prediction["bboxes"]) > 0:
            matching_details = self._cell_matcher.match_cells(
                iocr_page, scaled_table_bbox, prediction
            )
        # Post-processing
        if len(prediction["bboxes"]) > 0:
            if (
                len(iocr_page["tokens"]) > 0
            ):  # There are at least some pdf cells to match with
                if self.enable_post_process:
                    AggProfiler().begin("post_process", self._prof)
                    matching_details = self._post_processor.process(
                        matching_details, correct_overlapping_cells
                    )
                    AggProfiler().end("post_process", self._prof)

        # Generate the expected Docling responses
        AggProfiler().begin("generate_docling_response", self._prof)
        docling_output = self._generate_tf_response(
            matching_details["table_cells"],
            matching_details["matches"],
        )

        AggProfiler().end("generate_docling_response", self._prof)
        # Add the docling_output sorted by cell_id into the matching_details
        docling_output.sort(key=lambda item: item["cell_id"])
        matching_details["docling_responses"] = docling_output

        # Merge docling_output and pdf_cells into one TF output,
        # with deduplicated table cells
        tf_output = self._merge_tf_output(docling_output, matching_details["pdf_cells"])

        return tf_output, matching_details

    def _generate_tf_response_dummy(self, table_cells):
        tf_cell_list = []

        for table_cell in table_cells:
            colspan_val = 1
            if "colspan_val" in table_cell:
                colspan_val = table_cell["colspan_val"]
            rowspan_val = 1
            if "rowspan_val" in table_cell:
                rowspan_val = table_cell["rowspan_val"]

            column_header = False
            if table_cell["label"] == "ched":
                column_header = True

            row_header = False
            if table_cell["label"] == "rhed":
                row_header = True

            row_section = False
            if table_cell["label"] == "srow":
                row_section = True

            row_id = table_cell["row_id"]
            column_id = table_cell["column_id"]

            cell_bbox = {
                "b": table_cell["bbox"][3],
                "l": table_cell["bbox"][0],
                "r": table_cell["bbox"][2],
                "t": table_cell["bbox"][1],
                "token": "",
            }

            tf_cell = {
                "cell_id": table_cell["cell_id"],
                "bbox": cell_bbox,  # b,l,r,t,token
                "row_span": rowspan_val,
                "col_span": colspan_val,
                "start_row_offset_idx": row_id,
                "end_row_offset_idx": row_id + rowspan_val,
                "start_col_offset_idx": column_id,
                "end_col_offset_idx": column_id + colspan_val,
                "indentation_level": 0,
                # No text cell bboxes, because no matching was done
                "text_cell_bboxes": [],
                "column_header": column_header,
                "row_header": row_header,
                "row_section": row_section,
            }
            tf_cell_list.append(tf_cell)
        return tf_cell_list

    def _generate_tf_response(self, table_cells, matches):
        r"""
        Convert the matching details to the expected output for Docling

        Parameters
        ----------
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id",
                                                  "bbox", "label", "class"
        matches : dictionary of lists of table_cells
            A dictionary which is indexed by the pdf_cell_id as key and the value is a list
            of the table_cells that fall inside that pdf cell

        Returns
        -------
        docling_output : string
            json response formatted according to Docling api expectations
        """

        # format output to look similar to tests/examples/tf_gte_output_2.json
        tf_cell_list = []
        for pdf_cell_id, pdf_cell_matches in matches.items():
            tf_cell = {
                "bbox": {},  # b,l,r,t,token
                "row_span": 1,
                "col_span": 1,
                "start_row_offset_idx": -1,
                "end_row_offset_idx": -1,
                "start_col_offset_idx": -1,
                "end_col_offset_idx": -1,
                "indentation_level": 0,
                # return text cell bboxes additionally to the matched index
                "text_cell_bboxes": [{}],  # b,l,r,t,token
                "column_header": False,
                "row_header": False,
                "row_section": False,
            }
            tf_cell["cell_id"] = int(pdf_cell_id)

            row_ids = set()
            column_ids = set()
            labels = set()

            for match in pdf_cell_matches:
                tm = match["table_cell_id"]
                tcl = list(
                    filter(lambda table_cell: table_cell["cell_id"] == tm, table_cells)
                )
                if len(tcl) > 0:
                    table_cell = tcl[0]
                    row_ids.add(table_cell["row_id"])
                    column_ids.add(table_cell["column_id"])
                    labels.add(table_cell["label"])

                    if table_cell["label"] is not None:
                        if table_cell["label"] in ["ched"]:
                            tf_cell["column_header"] = True
                        if table_cell["label"] in ["rhed"]:
                            tf_cell["row_header"] = True
                        if table_cell["label"] in ["srow"]:
                            tf_cell["row_section"] = True

                    tf_cell["start_col_offset_idx"] = table_cell["column_id"]
                    tf_cell["end_col_offset_idx"] = table_cell["column_id"] + 1
                    tf_cell["start_row_offset_idx"] = table_cell["row_id"]
                    tf_cell["end_row_offset_idx"] = table_cell["row_id"] + 1

                    if "colspan_val" in table_cell:
                        tf_cell["col_span"] = table_cell["colspan_val"]
                        tf_cell["start_col_offset_idx"] = table_cell["column_id"]
                        off_idx = table_cell["column_id"] + tf_cell["col_span"]
                        tf_cell["end_col_offset_idx"] = off_idx
                    if "rowspan_val" in table_cell:
                        tf_cell["row_span"] = table_cell["rowspan_val"]
                        tf_cell["start_row_offset_idx"] = table_cell["row_id"]
                        tf_cell["end_row_offset_idx"] = (
                            table_cell["row_id"] + tf_cell["row_span"]
                        )
                    if "bbox" in table_cell:
                        table_match_bbox = table_cell["bbox"]
                        tf_bbox = {
                            "b": table_match_bbox[3],
                            "l": table_match_bbox[0],
                            "r": table_match_bbox[2],
                            "t": table_match_bbox[1],
                        }
                        tf_cell["bbox"] = tf_bbox

            tf_cell["row_ids"] = list(row_ids)
            tf_cell["column_ids"] = list(column_ids)
            tf_cell["label"] = "None"
            l_labels = list(labels)
            if len(l_labels) > 0:
                tf_cell["label"] = l_labels[0]
            tf_cell_list.append(tf_cell)
        return tf_cell_list

    def _prepare_image(self, mat_image):
        r"""
        Rescale the image and prepare a batch of 1 with the image as as tensor

        Parameters
        ----------
        mat_image: cv2.Mat
            The image as an openCV Mat object

        Returns
        -------
        tensor (batch_size, image_channels, resized_image, resized_image)
        """
        normalize = T.Normalize(
            mean=self._config["dataset"]["image_normalization"]["mean"],
            std=self._config["dataset"]["image_normalization"]["std"],
        )
        resized_size = self._config["dataset"]["resized_image"]
        resize = T.Resize([resized_size, resized_size])

        img, _ = normalize(mat_image, None)
        img, _ = resize(img, None)

        img = img.transpose(2, 1, 0)  # (channels, width, height)
        img = torch.FloatTensor(img / 255.0)
        image_batch = img.unsqueeze(dim=0)
        image_batch = image_batch.to(device=self._device)
        return image_batch

    def _get_html_tags(self, seq):
        r"""
        Convert indices to actual html tags

        """
        # Map the tag indices back to actual tags (without start, end)
        html_tags = [self._rev_word_map[ind] for ind in seq[1:-1]]

        return html_tags
