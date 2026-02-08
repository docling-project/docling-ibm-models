#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import os
import numpy as np
import pytest
from PIL import Image

from docling_ibm_models.code_formula_model.code_formula_predictor import CodeFormulaPredictor

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
                "label": "code",
                "image_path": "tests/test_data/code_formula/images/code.png",
                "gt_path": "tests/test_data/code_formula/gt/code.txt",
            },
            {
                "label": "formula",
                "image_path": "tests/test_data/code_formula/images/formula.png",
                "gt_path": "tests/test_data/code_formula/gt/formula.txt",
            },
        ],
        "info": {
            "device": "auto",
            "temperature": 0,
        },
    }

    # Download models from HF
    artifact_path = snapshot_download(repo_id="ds4sd/CodeFormula", revision="v1.0.2")
    
    init["artifact_path"] = artifact_path

    return init


def test_code_formula_predictor(init: dict):
    r"""
    Unit test for the CodeFormulaPredictor
    """
    device = "cpu"
    num_threads = 2

    # Initialize LayoutPredictor
    code_formula_predictor = CodeFormulaPredictor(
        init["artifact_path"], device=device, num_threads=num_threads
    )

    # Check info
    info = code_formula_predictor.info()
    assert info["device"] == device, "Wronly set device"
    assert info["num_threads"] == num_threads, "Wronly set number of threads"

    # Unsupported input image
    is_exception = False
    try:
        for _ in code_formula_predictor.predict(["wrong"], ['label']):
            pass
    except TypeError:
        is_exception = True
    assert is_exception

    # wrong type for temperature
    is_exception = False
    try:
        dummy_image = Image.new(mode="RGB", size=(100, 100), color=(255, 255, 255))
        for _ in code_formula_predictor.predict([dummy_image], ['label'], "0.1"):
            pass
    except Exception:
        is_exception = True
    assert is_exception

    # wrong value for temperature
    is_exception = False
    try:
        dummy_image = Image.new(mode="RGB", size=(100, 100), color=(255, 255, 255))
        for _ in code_formula_predictor.predict([dummy_image], ['label'], -0.1):
            pass
    except Exception:
        is_exception = True
    assert is_exception

    # wrong value for temperature
    is_exception = False
    try:
        dummy_image = Image.new(mode="RGB", size=(100, 100), color=(255, 255, 255))
        for _ in code_formula_predictor.predict([dummy_image], ["label"], None):
            pass
    except Exception:
        is_exception = True
    assert is_exception

    # mistmatched number of images and labels
    is_exception = False
    try:
        dummy_image = Image.new(mode="RGB", size=(100, 100), color=(255, 255, 255))
        for _ in code_formula_predictor.predict([dummy_image], ['label', 'label']):
            pass
    except Exception:
        is_exception = True
    assert is_exception

    # Predict on test images, not batched
    temperature = init['info']['temperature']
    for d in init["test_imgs"]:
        label = d['label']
        img_path = d['image_path']
        gt_path = d['gt_path']

        with Image.open(img_path) as img, open(gt_path, 'r') as gt_fp:
            gt = gt_fp.read()

            output = code_formula_predictor.predict([img], [label], temperature)
            output = output[0]

            assert output == gt

            # Load images as numpy arrays
            np_arr = np.asarray(img)
            output = code_formula_predictor.predict([np_arr], [label], temperature)
            output = output[0]

            assert output == gt

    # Predict on test images, batched
    labels = [d['label'] for d in init["test_imgs"]]
    images = [Image.open(d['image_path']) for d in init["test_imgs"]]
    gts = [open(d['gt_path'], 'r').read() for d in init["test_imgs"]]

    outputs = code_formula_predictor.predict(images, labels, temperature)
    assert outputs == gts
