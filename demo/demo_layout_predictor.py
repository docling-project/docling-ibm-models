#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont

from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor


def save_predictions(
    prefix: str, viz_dir: str, img_fn: Path, img, predictions: List[Dict[str, Any]]
):
    img_path = Path(img_fn)

    image = img.copy()
    draw = ImageDraw.Draw(image)

    predictions_filename = f"{prefix}_{img_path.stem}.txt"
    predictions_fn = os.path.join(viz_dir, predictions_filename)
    with open(predictions_fn, "w") as fd:
        for pred in predictions:
            bbox = [
                round(pred["l"], 2),
                round(pred["t"], 2),
                round(pred["r"], 2),
                round(pred["b"], 2),
            ]
            label = pred["label"]
            confidence = round(pred["confidence"], 3)

            # Save the predictions in txt file
            pred_txt = f"{prefix} {str(img_fn)}: {label} - {bbox} - {confidence}\n"
            fd.write(pred_txt)

            # Draw the bbox and label
            draw.rectangle(bbox, outline="orange")
            txt = f"{label}: {confidence}"
            draw.text(
                (bbox[0], bbox[1]), text=txt, font=ImageFont.load_default(), fill="blue"
            )

    draw_filename = f"{prefix}_{img_path.name}"
    draw_fn = os.path.join(viz_dir, draw_filename)
    image.save(draw_fn)


def demo(
    logger: logging.Logger,
    artifact_path: str,
    device: str,
    num_threads: int,
    img_dir: str,
    viz_dir: str,
    threshold: float,
):
    r"""
    Apply LayoutPredictor on the input image directory

    If you want to load from PDF:
    pdf_image = pyvips.Image.new_from_file("test_data/ADS.2007.page_123.pdf", page=0)
    """
    # Create the layout predictor
    predictor = LayoutPredictor(artifact_path, device=device, num_threads=num_threads, base_threshold=threshold)

    # Predict all test png images
    t0 = time.perf_counter()
    img_counter = 0
    for img_fn in Path(img_dir).rglob("*.png"):
        img_counter += 1
        logger.info("Predicting '%s'...", img_fn)

        with Image.open(img_fn) as image:
            # Predict layout
            img_t0 = time.perf_counter()
            preds: List[Dict[str, Any]] = list(predictor.predict(image))
            img_ms = 1000 * (time.perf_counter() - img_t0)
            logger.debug("Prediction(ms): {:.2f}".format(img_ms))

            # Save predictions
            logger.info("Saving prediction visualization in: '%s'", viz_dir)
            save_predictions("ST", viz_dir, img_fn, image, preds)
    total_ms = 1000 * (time.perf_counter() - t0)
    avg_ms = (total_ms / img_counter) if img_counter > 0 else 0
    logger.info(
        "For {} images(ms): [total|avg] = [{:.1f}|{:.1f}]".format(
            img_counter, total_ms, avg_ms
        )
    )


def main(args):
    r""" """
    num_threads = int(args.num_threads) if args.num_threads is not None else 4
    device = args.device.lower()
    img_dir = args.img_dir
    viz_dir = args.viz_dir
    hugging_face_repo = args.hugging_face_repo
    threshold = float(args.threshold)

    # Initialize logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("LayoutPredictor")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Ensure the viz dir
    Path(viz_dir).mkdir(parents=True, exist_ok=True)

    # Download models from HF
    download_path = snapshot_download(repo_id=hugging_face_repo)

    # Test the LayoutPredictor
    demo(logger, download_path, device, num_threads, img_dir, viz_dir, threshold)


if __name__ == "__main__":
    r"""
    python -m demo.demo_layout_predictor -i <images_dir>
    """
    parser = argparse.ArgumentParser(description="Test the LayoutPredictor")

    supported_hf_repos = [
        "ds4sd/docling-layout-old",
        "ds4sd/docling-layout-heron",
        "ds4sd/docling-layout-heron-101",
        "ds4sd/docling-layout-egret-medium",
        "ds4sd/docling-layout-egret-large",
        "ds4sd/docling-layout-egret-xlarge",
    ]
    parser.add_argument(
        "-r",
        "--hugging-face-repo",
        required=False,
        default="ds4sd/docling-layout-old",
        help=f"The hugging face repo id: [{', '.join(supported_hf_repos)}]",
    )
    parser.add_argument(
        "-t", "--threshold", required=False, default=0.3, help="Threshold for the LayoutPredictor"
    )
    parser.add_argument(
        "-d", "--device", required=False, default="cpu", help="One of [cpu, cuda, mps]"
    )
    parser.add_argument(
        "-n", "--num_threads", required=False, default=4, help="Number of threads"
    )
    parser.add_argument(
        "-i",
        "--img_dir",
        required=True,
        help="PNG images input directory",
    )
    parser.add_argument(
        "-v",
        "--viz_dir",
        required=False,
        default="viz/",
        help="Directory to save prediction visualizations",
    )

    args = parser.parse_args()
    main(args)
