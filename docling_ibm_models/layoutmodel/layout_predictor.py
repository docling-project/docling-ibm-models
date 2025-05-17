#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging
import os
from collections.abc import Iterable
from typing import Set, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import DFineForObjectDetection, RTDetrImageProcessor

_log = logging.getLogger(__name__)


class LayoutPredictor:
    """
    Document layout prediction using safe tensors
    """

    def __init__(
        self,
        artifact_path: str,
        device: str = "cpu",
        num_threads: int = 4,
        base_threshold: float = 0.3,
        blacklist_classes: Set[str] = set(),
    ):
        """
        Provide the artifact path that contains the LayoutModel file

        Parameters
        ----------
        artifact_path: Path for the model torch file.
        device: (Optional) device to run the inference.
        num_threads: (Optional) Number of threads to run the inference if device = 'cpu'

        Raises
        ------
        FileNotFoundError when the model's torch file is missing
        """
        # Initialize classes map:
        self._classes_map = {
            0: "Caption",
            1: "Footnote",
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            8: "Table",
            9: "Text",
            10: "Title",
            11: "Document Index",
            12: "Code",
            13: "Checkbox-Selected",
            14: "Checkbox-Unselected",
            15: "Form",
            16: "Key-Value Region",
        }

        # Blacklisted classes
        self._black_classes = blacklist_classes  # set(["Form", "Key-Value Region"])

        # Set basic params
        self._threshold = base_threshold  # Score threshold
        self._image_size = 640
        self._size = np.asarray([[self._image_size, self._image_size]], dtype=np.int64)

        # Set number of threads for CPU
        self._device = torch.device(device)
        self._num_threads = num_threads
        if device == "cpu":
            torch.set_num_threads(self._num_threads)

        # Model file and configurations
        self._st_fn = os.path.join(artifact_path, "model.safetensors")
        if not os.path.isfile(self._st_fn):
            raise FileNotFoundError("Missing safe tensors file: {}".format(self._st_fn))

        # Load model and move to device
        processor_config = os.path.join(artifact_path, "preprocessor_config.json")
        model_config = os.path.join(artifact_path, "config.json")
        self._image_processor = RTDetrImageProcessor.from_json_file(processor_config)
        self._model = DFineForObjectDetection.from_pretrained(
            artifact_path, config=model_config
        ).to(self._device)
        self._model.eval()

        _log.debug("LayoutPredictor settings: {}".format(self.info()))

    def info(self) -> dict:
        """
        Get information about the configuration of LayoutPredictor
        """
        info = {
            "safe_tensors_file": self._st_fn,
            "device": self._device.type,
            "num_threads": self._num_threads,
            "image_size": self._image_size,
            "threshold": self._threshold,
        }
        return info

    @torch.inference_mode()
    def predict(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        """
        Predict bounding boxes for a given image.
        The origin (0, 0) is the top-left corner and the predicted bbox coords are provided as:
        [left, top, right, bottom]

        Parameter
        ---------
        origin_img: Image to be predicted as a PIL Image object or numpy array.

        Yield
        -----
        Bounding box as a dict with the keys: "label", "confidence", "l", "t", "r", "b"

        Raises
        ------
        TypeError when the input image is not supported
        """
        # Convert image format
        if isinstance(orig_img, Image.Image):
            page_img = orig_img.convert("RGB")
        elif isinstance(orig_img, np.ndarray):
            page_img = Image.fromarray(orig_img).convert("RGB")
        else:
            raise TypeError("Not supported input image format")

        resize = {"height": self._image_size, "width": self._image_size}
        inputs = self._image_processor(
            images=page_img,
            return_tensors="pt",
            size=resize,
        ).to(self._device)
        outputs = self._model(**inputs)
        results = self._image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([page_img.size[::-1]]),
            threshold=self._threshold,
        )

        w, h = page_img.size

        result = results[0]
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            score = float(score.item())

            label_id = int(label_id.item())
            label_str = self._classes_map[label_id]

            # Filter out blacklisted classes
            if label_str in self._black_classes:
                continue

            bbox_float = [float(b.item()) for b in box]
            l = min(w, max(0, bbox_float[0]))
            t = min(h, max(0, bbox_float[1]))
            r = min(w, max(0, bbox_float[2]))
            b = min(h, max(0, bbox_float[3]))
            yield {
                "l": l,
                "t": t,
                "r": r,
                "b": b,
                "label": label_str,
                "confidence": score,
            }
