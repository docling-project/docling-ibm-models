import numbers
from collections.abc import Iterable, Sequence

import cv2
import numpy as np
import torch
from torchvision.transforms import functional

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

INTER_MODE = {
    "NEAREST": cv2.INTER_NEAREST,
    "BILINEAR": cv2.INTER_LINEAR,
    "BICUBIC": cv2.INTER_CUBIC,
}

PAD_MOD = {
    "constant": cv2.BORDER_CONSTANT,
    "edge": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_DEFAULT,
    "symmetric": cv2.BORDER_REFLECT,
}


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.
    See ``Normalize`` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if _is_tensor_image(tensor):
        for t, m, s in zip(tensor, mean, std, strict=False):
            t.sub_(m).div_(s)
        return tensor
    elif _is_numpy_image(tensor):
        return (tensor.astype(np.float32) - 255.0 * np.array(mean)) / np.array(std)
    else:
        raise RuntimeError("Undefined type")


def resize(img, size, interpolation="BILINEAR"):
    """Resize the input CV Image to the given size.
    Args:
        img (np.ndarray): Image to be resized.
        size (tuple or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (str, optional): Desired interpolation. Default is ``BILINEAR``
    Returns:
        cv Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError("img should be CV Image. Got {}".format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError("Got inappropriate size arg: {}".format(size))

    # TODO(Nikos): Try to remove the opencv dependency
    if isinstance(size, int):
        h, w, c = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(
                img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation]
            )
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(
                img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation]
            )
    else:
        oh, ow = size
        return cv2.resize(
            img, dsize=(int(ow), int(oh)), interpolation=INTER_MODE[interpolation]
        )
