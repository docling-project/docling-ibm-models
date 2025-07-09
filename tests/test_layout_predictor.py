#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import os
import json

import numpy as np
import pytest
from huggingface_hub import snapshot_download
from PIL import Image

from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor


@pytest.fixture(scope="module")
def init() -> dict:
    r"""
    Initialize the testing environment
    """
    # This config is missing the keys: "artifact_path", "info1.torch_file", "info2.torch_file"
    init = {
        "num_threads": 1,
        "test_imgs": [
            "tests/test_data/samples/ADS.2007.page_123.png",
        ],
        "info1": {
            "device": "cpu",
            "image_size": 640,
            "threshold": 0.6,
        },
        "pred_bboxes": 9,
    }

    # Download models from HF
    artifact_path = snapshot_download(repo_id="ds4sd/docling-layout-old")

    # Add the missing config keys
    init["artifact_path"] = artifact_path

    return init


def test_layoutpredictor(init: dict):
    r"""
    Unit test for the LayoutPredictor
    """
    device = "cpu"
    num_threads = 2

    # Initialize LayoutPredictor
    lpredictor = LayoutPredictor(
        init["artifact_path"], device=device, num_threads=num_threads
    )

    # Check info
    info = lpredictor.info()
    assert info["device"] == device, "Wronly set device"
    assert info["num_threads"] == num_threads, "Wronly set number of threads"

    # Unsupported input image
    is_exception = False
    try:
        for pred in lpredictor.predict("wrong"):
            pass
    except TypeError:
        is_exception = True
    assert is_exception

    # Predict on the test image
    for img_fn in init["test_imgs"]:
        
        true_layout_fn = img_fn+".json"
        with Image.open(img_fn) as img:

            w, h = img.size

            # Load images as PIL objects
            for i, pred in enumerate(lpredictor.predict(img)):
                print("PIL pred: {}".format(pred))
                assert pred["l"] >= 0 and pred["l"] <= w
                assert pred["t"] >= 0 and pred["t"] <= h
                assert pred["r"] >= 0 and pred["r"] <= w
                assert pred["b"] >= 0 and pred["b"] <= h

            assert i + 1 == init["pred_bboxes"]

            if os.path.exists(true_layout_fn):
                with open(true_layout_fn, "r") as fr:
                    true_layout = json.load(fr)

            """
                # FIXME: write a simple test to check all objects are found
            else:
                with open(true_layout_fn, "w") as fw:
                    fw.write(json.dumps(pred_layout, indent=4))
            """
            
            # Load images as numpy arrays
            np_arr = np.asarray(img)
            for i, pred in enumerate(lpredictor.predict(np_arr)):
                print("numpy pred: {}".format(pred))
                assert pred["l"] >= 0 and pred["l"] <= w
                assert pred["t"] >= 0 and pred["t"] <= h
                assert pred["r"] >= 0 and pred["r"] <= w
                assert pred["b"] >= 0 and pred["b"] <= h
            assert i + 1 == init["pred_bboxes"]
