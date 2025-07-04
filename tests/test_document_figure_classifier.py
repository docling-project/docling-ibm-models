#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import os
import numpy as np
import pytest
from PIL import Image

from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor import (
    DocumentFigureClassifierPredictor,
)

from huggingface_hub import snapshot_download


@pytest.fixture(scope="module")
def init() -> dict:
    r"""
    Initialize the testing environment
    """
    init = {
        "num_threads": 1,
        "test_imgs": [
            {
                "label": "bar_chart",
                "image_path": "tests/test_data/figure_classifier/images/bar_chart.jpg",
            },
            {
                "label": "map",
                "image_path": "tests/test_data/figure_classifier/images/map.jpg",
            },
        ],
        "info": {
            "device": "auto",
        },
    }

    # Download models from HF 
    init["artifact_path"] = snapshot_download(
        repo_id="ds4sd/DocumentFigureClassifier", revision="v1.0.1"
    )

    return init


def test_figure_classifier(init: dict):
    r"""
    Unit test for the CodeFormulaPredictor
    """
    device = "cpu"
    num_threads = 2

    # Initialize LayoutPredictor
    figure_classifier = DocumentFigureClassifierPredictor(
        init["artifact_path"], device=device, num_threads=num_threads
    )

    # Check info
    info = figure_classifier.info()
    assert info["device"] == device, "Wronly set device"
    assert info["num_threads"] == num_threads, "Wronly set number of threads"

    # Unsupported input image
    is_exception = False
    try:
        for _ in figure_classifier.predict(["wrong"]):
            pass
    except TypeError:
        is_exception = True
    assert is_exception

    # Predict on test images, not batched
    for d in init["test_imgs"]:
        label = d["label"]
        img_path = d["image_path"]

        with Image.open(img_path) as img:

            output = figure_classifier.predict([img])
            predicted_class = output[0][0][0]

            assert predicted_class == label

            # Load images as numpy arrays
            np_arr = np.asarray(img)
            output = figure_classifier.predict([np_arr])
            predicted_class = output[0][0][0]

            assert predicted_class == label

    # Predict on test images, batched
    labels = [d['label'] for d in init["test_imgs"]]
    images = [Image.open(d["image_path"]) for d in init["test_imgs"]]

    outputs = figure_classifier.predict(images)
    outputs = [output[0][0] for output in outputs]
    assert outputs == labels
