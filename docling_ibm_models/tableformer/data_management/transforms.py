from __future__ import division

import collections
import numbers
import random

import torch

from docling_ibm_models.tableformer.data_management import functional as F


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, target=None):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std), target

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
        (h, w), output size will be matched to this. If size is an int,
        smaller edge of the image will be matched to this number.
        i.e, if height > width, then image will be rescaled to
        (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
        ``BILINEAR``
    """

    def __init__(self, size, interpolation="BILINEAR"):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, target=None):
        """
        Args:
        img (np.ndarray): Image to be scaled.
        Returns:
        np.ndarray: Rescaled image.
        """
        # Resize bboxes (in pixels)
        x_scale = 0
        y_scale = 0

        if img.shape[1] > 0:
            x_scale = self.size[0] / img.shape[1]
        if img.shape[0] > 0:
            y_scale = self.size[1] / img.shape[0]

        # loop over bboxes
        if target is not None:
            if target["boxes"] is not None:
                target_ = target.copy()
                target_["boxes"][:, 0] = x_scale * target_["boxes"][:, 0]
                target_["boxes"][:, 1] = y_scale * target_["boxes"][:, 1]
                target_["boxes"][:, 2] = x_scale * target_["boxes"][:, 2]
                target_["boxes"][:, 3] = y_scale * target_["boxes"][:, 3]
        return F.resize(img, self.size, self.interpolation), target

    def __repr__(self):
        interpolate_str = self.interpolation
        return self.__class__.__name__ + "(size={0}, interpolation={1})".format(
            self.size, interpolate_str
        )
