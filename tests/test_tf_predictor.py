import glob
import json
import os
from pathlib import Path

import torch
import pytest
import cv2
from PIL import Image, ImageDraw
from huggingface_hub import snapshot_download

from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler
import docling_ibm_models.tableformer.data_management.tf_predictor as tf_predictor
from docling_ibm_models.tableformer.data_management.tf_predictor import \
    TFPredictor

"""
- Implements TF predictor to accept the input format from IOCR, e.g.
  "./tests/test_data/samples/tf_table_example_0.json" (trivial table crop)
- Shape the output format like GTE would: e.g.
  "tests/test_data/samples/tf_gte_output_2.json" (Note: full form image)
"""

docling_api_data = {
    "table_jsons": [
        "./tests/test_data/samples/ADS.2007.page_123.png_iocr.parse_format.json",
        "./tests/test_data/samples/PHM.2013.page_30.png_iocr.parse_format.json",
        "./tests/test_data/samples/empty_iocr.png.json"
    ],
    "png_images": [
        "./tests/test_data/samples/ADS.2007.page_123.png",
        "./tests/test_data/samples/PHM.2013.page_30.png",
        "./tests/test_data/samples/empty_iocr.png"
    ],
    "table_bboxes": [
        [[178, 748, 1061, 976], [177, 1163, 1062, 1329]],
        [[100, 186, 1135, 525]],
        [[178, 748, 1061, 976], [177, 1163, 1062, 1329]]
    ],
}

# The config is missing the keys: "model.save_dir"
test_config = {
    "dataset": {
        "type": "TF_prepared",
        "name": "TF",
        "raw_data_dir": "./tests/test_data/model_artifacts/",
        "load_cells": True,
        "bbox_format": "5plet",
        "resized_image": 448,
        "keep_AR": False,
        "up_scaling_enabled": True,
        "down_scaling_enabled": True,
        "padding_mode": "null",
        "padding_color": [0, 0, 0],
        "image_normalization": {
            "state": True,
            "mean": [0.94247851, 0.94254675, 0.94292611],
            "std": [0.17910956, 0.17940403, 0.17931663],
        },
        "color_jitter": True,
        "rand_crop": True,
        "rand_pad": True,
        "image_grayscale": False,
    },
    "model": {
        "type": "TableModel04_rs",
        "name": "14_128_256_4_true",
        # "save_dir": "./tests/test_data/model_artifacts/",
        "backbone": "resnet18",
        "enc_image_size": 28,
        "tag_embed_dim": 16,
        "hidden_dim": 512,
        "tag_decoder_dim": 512,
        "bbox_embed_dim": 256,
        "tag_attention_dim": 256,
        "bbox_attention_dim": 512,
        "enc_layers": 4,  # 6
        "dec_layers": 2,  # 6
        "nheads": 8,
        "dropout": 0.1,
        "bbox_classes": 2,
    },
    "train": {
        "save_periodicity": 1,
        "disable_cuda": False,
        "epochs": 23,
        "batch_size": 50,
        "clip_gradient": 0.1,
        "clip_max_norm": 0.1,
        "bbox": True,
        "validation": False,
    },
    "predict": {
        "max_steps": 1024,
        "beam_size": 5,
        "bbox": True,
        "predict_dir": "./tests/test_data/samples",
        "pdf_cell_iou_thres": 0.05,
        "padding": False,
        "padding_size": 50,
        "disable_post_process": False,
        "profiling": True
    },
    "dataset_wordmap": {
        "word_map_tag": {
            "<pad>": 0,
            "<unk>": 1,
            "<start>": 2,
            "<end>": 3,
            "ecel": 4,
            "fcel": 5,
            "lcel": 6,
            "ucel": 7,
            "xcel": 8,
            "nl": 9,
            "ched": 10,
            "rhed": 11,
            "srow": 12,
        },
        "word_map_cell": {
            " ": 13,
            "!": 179,
            '"': 126,
            "#": 101,
            "$": 119,
            "%": 18,
            "&": 114,
            "'": 108,
            "(": 29,
            ")": 32,
            "*": 26,
            "+": 97,
            ",": 71,
            "-": 63,
            ".": 34,
            "/": 66,
            "0": 33,
            "1": 36,
            "2": 43,
            "3": 41,
            "4": 45,
            "5": 17,
            "6": 37,
            "7": 35,
            "8": 40,
            "9": 16,
            ":": 88,
            ";": 92,
            "<": 73,
            "</b>": 9,
            "</i>": 23,
            "</overline>": 219,
            "</strike>": 233,
            "</sub>": 94,
            "</sup>": 77,
            "</underline>": 151,
            "<b>": 1,
            "<end>": 280,
            "<i>": 21,
            "<overline>": 218,
            "<pad>": 0,
            "<start>": 279,
            "<strike>": 232,
            "<sub>": 93,
            "<sup>": 75,
            "<underline>": 150,
            "<unk>": 278,
            "=": 99,
            ">": 39,
            "?": 96,
            "@": 125,
            "A": 27,
            "B": 86,
            "C": 19,
            "D": 57,
            "E": 64,
            "F": 47,
            "G": 44,
            "H": 10,
            "I": 20,
            "J": 80,
            "K": 81,
            "L": 52,
            "M": 46,
            "N": 69,
            "O": 65,
            "P": 62,
            "Q": 59,
            "R": 60,
            "S": 58,
            "T": 48,
            "U": 55,
            "V": 2,
            "W": 83,
            "X": 104,
            "Y": 89,
            "Z": 113,
            "[": 70,
            "\\": 165,
            "]": 72,
            "^": 132,
            "_": 84,
            "`": 196,
            "a": 3,
            "b": 6,
            "c": 54,
            "d": 12,
            "e": 8,
            "f": 50,
            "g": 28,
            "h": 56,
            "i": 5,
            "j": 82,
            "k": 95,
            "l": 7,
            "m": 30,
            "n": 31,
            "o": 15,
            "p": 22,
            "q": 67,
            "r": 4,
            "s": 51,
            "t": 14,
            "u": 25,
            "v": 24,
            "w": 53,
            "x": 61,
            "y": 49,
            "z": 11,
            "{": 158,
            "|": 139,
            "}": 159,
            "~": 147,
            "\u00a2": 203,
            "\u00a3": 162,
            "\u00a4": 220,
            "\u00a5": 176,
            "\u00a7": 142,
            "\u00a9": 268,
            "\u00ab": 239,
            "\u00ad": 275,
            "\u00ae": 130,
            "\u00b0": 100,
            "\u00b1": 79,
            "\u00b6": 171,
            "\u00b7": 137,
            "\u00bb": 240,
            "\u00d7": 118,
            "\u00d8": 192,
            "\u00df": 197,
            "\u00e6": 261,
            "\u00f7": 225,
            "\u00f8": 163,
            "\u0131": 242,
            "\u0142": 267,
            "\u01c2": 211,
            "\u025b": 223,
            "\u02b9": 248,
            "\u02c2": 195,
            "\u02c3": 208,
            "\u02c6": 253,
            "\u0300": 209,
            "\u0301": 131,
            "\u0302": 138,
            "\u0303": 156,
            "\u0304": 152,
            "\u0306": 222,
            "\u0307": 247,
            "\u0308": 103,
            "\u030a": 102,
            "\u030c": 254,
            "\u0327": 155,
            "\u0328": 269,
            "\u0338": 170,
            "\u0391": 173,
            "\u0392": 169,
            "\u0393": 180,
            "\u0394": 85,
            "\u0398": 243,
            "\u0399": 271,
            "\u039b": 272,
            "\u03a0": 213,
            "\u03a3": 185,
            "\u03a6": 148,
            "\u03a7": 212,
            "\u03a8": 141,
            "\u03a9": 161,
            "\u03b1": 90,
            "\u03b2": 107,
            "\u03b3": 110,
            "\u03b4": 153,
            "\u03b5": 166,
            "\u03b6": 178,
            "\u03b7": 146,
            "\u03b8": 186,
            "\u03b9": 229,
            "\u03ba": 164,
            "\u03bb": 91,
            "\u03bc": 78,
            "\u03bd": 230,
            "\u03be": 244,
            "\u03c0": 127,
            "\u03c1": 149,
            "\u03c3": 116,
            "\u03c4": 198,
            "\u03c5": 189,
            "\u03c6": 140,
            "\u03c7": 124,
            "\u03c8": 216,
            "\u03c9": 167,
            "\u0410": 273,
            "\u0421": 194,
            "\u115f": 217,
            "\u200b": 265,
            "\u2010": 117,
            "\u2012": 135,
            "\u2013": 42,
            "\u2014": 106,
            "\u2015": 228,
            "\u2016": 259,
            "\u2018": 123,
            "\u2019": 121,
            "\u201c": 87,
            "\u201d": 115,
            "\u201e": 245,
            "\u2020": 109,
            "\u2021": 129,
            "\u2022": 128,
            "\u2028": 190,
            "\u2030": 154,
            "\u2032": 68,
            "\u203b": 224,
            "\u2044": 188,
            "\u204e": 199,
            "\u2061": 200,
            "\u20ac": 184,
            "\u2190": 202,
            "\u2191": 112,
            "\u2192": 120,
            "\u2193": 111,
            "\u2194": 183,
            "\u21d1": 266,
            "\u21d2": 264,
            "\u21d3": 255,
            "\u2205": 215,
            "\u2206": 175,
            "\u2208": 262,
            "\u2211": 160,
            "\u2212": 76,
            "\u2216": 206,
            "\u2217": 105,
            "\u2218": 246,
            "\u2219": 236,
            "\u221a": 187,
            "\u221e": 207,
            "\u2223": 260,
            "\u2225": 193,
            "\u2227": 182,
            "\u2229": 256,
            "\u222b": 258,
            "\u223c": 98,
            "\u2248": 210,
            "\u2264": 38,
            "\u2265": 74,
            "\u2266": 214,
            "\u2267": 181,
            "\u2295": 263,
            "\u22c5": 174,
            "\u22c6": 191,
            "\u22ee": 277,
            "\u22ef": 270,
            "\u2500": 205,
            "\u2551": 231,
            "\u25a0": 250,
            "\u25a1": 177,
            "\u25aa": 145,
            "\u25b2": 136,
            "\u25b3": 143,
            "\u25bc": 251,
            "\u25c6": 226,
            "\u25ca": 235,
            "\u25cb": 227,
            "\u25cf": 172,
            "\u25e6": 274,
            "\u2605": 204,
            "\u2606": 144,
            "\u2640": 133,
            "\u2642": 134,
            "\u2663": 252,
            "\u2666": 157,
            "\u266f": 221,
            "\u2713": 122,
            "\u2714": 249,
            "\u2717": 201,
            "\u2794": 168,
            "\u27a2": 276,
            "\u2a7d": 234,
            "\u2a7e": 241,
            "\u3008": 237,
            "\u3009": 238,
            "\ufeff": 257,
        },
    },
}

# ==================================================================================================
configs = [test_config]


@pytest.fixture(scope="module")
def init() -> list[dict]:
    r"""
    Initialize the testing environment
    """
    # Download models from HF
    download_path = snapshot_download(repo_id="ds4sd/docling-models", revision="v2.1.0")
    save_dir = os.path.join(download_path, "model_artifacts/tableformer/fast")

    # Add the missing config keys
    for config in configs:
        config["model"]["save_dir"] = save_dir
    return configs

def test_tf_predictor(init):
    r"""
    Test the TFPredictor
    """
    viz = True
    device = "cpu"
    num_threads = 2

    # Load the docling_api_data
    iocr_pages = []
    for table_json_fn, png_image_fn, table_bboxes_b in zip(
        docling_api_data["table_jsons"],
        docling_api_data["png_images"],
        docling_api_data["table_bboxes"],
    ):
        with open(table_json_fn, "r") as fp:
            iocr_page_raw = json.load(fp)
            iocr_page = iocr_page_raw["pages"][0]
        # TODO(Nikos): Try to remove the opencv dependency
        iocr_page["image"] = cv2.imread(png_image_fn)
        iocr_page["png_image_fn"] = png_image_fn
        iocr_page["table_bboxes"] = table_bboxes_b
        iocr_pages.append(iocr_page)

    # Loop over the test configs
    for test_config in init:
        # Check if the checkpoint file should be combined
        # assert (
        #     combine_checkpoint(test_config["model"]["save_dir"]) >= 0
        # ), "Model checkpoint is missing"

        # Loop over the iocr_pages
        predictor = TFPredictor(test_config, device=device, num_threads=num_threads)
        for iocr_page in iocr_pages:
            # Prepare "Predict" parameters
            # iw = iocr_page["width"]
            # ih = iocr_page["height"]
            # table_bboxes = [[0, 0, iw, ih]]  # just one table per page in our examples
            table_bboxes = iocr_page["table_bboxes"]

            # for t, table_bbox in enumerate(table_bboxes):
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            png_img_bfn = os.path.basename(iocr_page["png_image_fn"])
            print("Predicting image: {}".format(png_img_bfn))

            # Run prediction, post-processing, and cell matching
            # PARAMETERS:
            # iocr_page - json received from iocr, augmented with iocr_page["image"]
            # table_bboxes - list of detected bboxes on page: [[x1, y1, x2, y2], [...]...]
            # do_matching - Boolean, when True - will match with text cells provided,
            #               when False - returns original cell prediction BBOXes in the same format
            # OUTPUT:
            # List of dicts per table: [{"tf_responses":[...], "predict_details": {}}]

            multi_tf_output = predictor.multi_table_predict(
                iocr_page,
                table_bboxes, 
                do_matching=True,
                correct_overlapping_cells=False,
                sort_row_col_indexes=True
            )

            # Test output for validity, create visualizations...
            for t, tf_output in enumerate(multi_tf_output):
                tf_responses = tf_output["tf_responses"]
                predict_details = tf_output["predict_details"]
                assert tf_responses is not None, "Empty prediction response"
                assert isinstance(
                    tf_responses, list
                ), " Wrong response type. It should be a list"

                img = Image.open(iocr_page["png_image_fn"])
                img1 = ImageDraw.Draw(img)

                xt0 = table_bboxes[t][0]
                yt0 = table_bboxes[t][1]
                xt1 = max(xt0, table_bboxes[t][2])
                yt1 = max(yt0, table_bboxes[t][3])
                img1.rectangle(((xt0, yt0), (xt1, yt1)), outline="pink", width=5)

                if viz:
                    # Visualize original OCR words:
                    for iocr_word in iocr_page["tokens"]:
                        xi0 = iocr_word["bbox"]["l"]
                        yi0 = iocr_word["bbox"]["t"]
                        xi1 = max(xi0, iocr_word["bbox"]["r"])
                        yi1 = max(yi0, iocr_word["bbox"]["b"])
                        img1.rectangle(((xi0, yi0), (xi1, yi1)), outline="gray")
                    # Visualize original docling_ibm_models.tableformer predictions:
                    for predicted_bbox in predict_details["prediction_bboxes_page"]:
                        xp0 = predicted_bbox[0] - 1
                        yp0 = predicted_bbox[1] - 1
                        xp1 = max(xp0, predicted_bbox[2] + 1)
                        yp1 = max(yp0, predicted_bbox[3] + 1)
                        img1.rectangle(((xp0, yp0), (xp1, yp1)), outline="green")

                # Check the structure of the list items
                for i, response in enumerate(tf_responses):
                    assert (
                        "bbox" in response
                    ), "bbox field is missing from response: " + str(i)
                    assert (
                        "text_cell_bboxes" in response
                    ), "text_cell_bboxes is missing: " + str(i)
                    assert (
                        "row_span" in response
                    ), "row_span is missing from resp: " + str(i)
                    assert (
                        "col_span" in response
                    ), "col_span is missing from response: " + str(i)
                    # print("*********** column_header: {}".format(response["column_header"]))
                    if viz:
                        # Visualization:
                        for text_cell in response["text_cell_bboxes"]:
                            xc0 = text_cell["l"]
                            yc0 = text_cell["t"]
                            xc1 = max(xc0, text_cell["r"])
                            yc1 = max(yc0, text_cell["b"])
                            img1.rectangle(((xc0, yc0), (xc1, yc1)), outline="red")

                        x0 = response["bbox"]["l"] - 2
                        y0 = response["bbox"]["t"] - 2
                        x1 = max(x0, response["bbox"]["r"] + 2)
                        y1 = max(y0, response["bbox"]["b"] + 2)

                        if response["column_header"]:
                            img1.rectangle(
                                ((x0, y0), (x1, y1)), outline="blue", width=2
                            )
                        elif response["row_header"]:
                            img1.rectangle(
                                ((x0, y0), (x1, y1)), outline="magenta", width=2
                            )
                        elif response["row_section"]:
                            img1.rectangle(
                                ((x0, y0), (x1, y1)), outline="brown", width=2
                            )
                        else:
                            img1.rectangle(
                                ((x0, y0), (x1, y1)), outline="black", width=1
                            )
                if viz:
                    viz_root = "./tests/test_data/viz/"
                    Path(viz_root).mkdir(parents=True, exist_ok=True)
                    png_img_bfn1 = png_img_bfn.replace(".png", "." + str(t) + ".png")
                    viz_fn = os.path.join(viz_root, png_img_bfn1)
                    img.save(viz_fn)

    # Get profiling data
    profiling_data = AggProfiler().get_data()
    print("Profiling data:")
    print(json.dumps(profiling_data, indent=2, sort_keys=True))
