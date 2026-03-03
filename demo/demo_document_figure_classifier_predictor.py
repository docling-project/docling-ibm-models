import argparse
import logging
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image

from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor import DocumentFigureClassifierPredictor


def demo(
    logger: logging.Logger,
    artifact_path: str,
    device: str,
    num_threads: int,
    image_dir: str,
    viz_dir: str,
):
    r"""
    Apply DocumentFigureClassifierPredictor on the input image directory
    """
    # Create the layout predictor
    document_figure_classifier_predictor = DocumentFigureClassifierPredictor(artifact_path, device=device, num_threads=num_threads)

    image_dir = Path(image_dir)
    images = []
    image_names = os.listdir(image_dir)
    image_names.sort()
    for image_name in image_names:
        image = Image.open(image_dir / image_name)
        images.append(image)

    t0 = time.perf_counter()
    outputs = document_figure_classifier_predictor.predict(images)
    total_ms = 1000 * (time.perf_counter() - t0)
    avg_ms = (total_ms / len(image_names)) if len(image_names) > 0 else 0
    logger.info(
        "For {} images(ms): [total|avg] = [{:.1f}|{:.1f}]".format(
            len(image_names), total_ms, avg_ms
        )
    )

    for i, output in enumerate(outputs):
        image_name = image_names[i]
        logger.info(f"Predictions for: '{image_name}':")
        for pred in output:
            logger.info(f" Class '{pred[0]}' has probability {pred[1]}")


def main(args):
    num_threads = int(args.num_threads) if args.num_threads is not None else None
    device = args.device.lower()
    image_dir = args.image_dir
    viz_dir = args.viz_dir

    # Initialize logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("DocumentFigureClassifierPredictor")
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
    download_path = snapshot_download(repo_id="ds4sd/DocumentFigureClassifier", revision="v1.0.0")

    # Test the figure classifier model
    demo(logger, download_path, device, num_threads, image_dir, viz_dir)


if __name__ == "__main__":
    r"""
    python -m demo.demo_document_figure_classifier_predictor -i <images_dir>
    """
    parser = argparse.ArgumentParser(description="Test the DocumentFigureClassifierPredictor")
    parser.add_argument(
        "-d", "--device", required=False, default="cpu", help="One of [cpu, cuda, mps]"
    )
    parser.add_argument(
        "-n", "--num_threads", required=False, default=4, help="Number of threads"
    )
    parser.add_argument(
        "-i",
        "--image_dir",
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