import logging

import torch.nn as nn
import torchvision

import docling_ibm_models.tableformer.settings as s

LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


class Encoder04(nn.Module):
    """
    Encoder based on resnet-18
    """

    def __init__(self, enc_image_size, enc_dim=512):
        r"""
        Parameters
        ----------
        enc_image_size : int
                Assuming that the encoded image is a square, this is the length of the image side
        """

        super(Encoder04, self).__init__()
        self.enc_image_size = enc_image_size
        self._encoder_dim = enc_dim

        resnet = torchvision.models.resnet18()
        modules = list(resnet.children())[:-3]

        self._resnet = nn.Sequential(*modules)
        self._adaptive_pool = nn.AdaptiveAvgPool2d(
            (self.enc_image_size, self.enc_image_size)
        )

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def get_encoder_dim(self):
        return self._encoder_dim

    def forward(self, images):
        """
        Forward propagation
        The encoder_dim 512 is decided by the structure of the image network (modified resnet-19)

        Parameters
        ----------
        images : tensor (batch_size, image_channels, resized_image, resized_image)
                images input

        Returns
        -------
        tensor : (batch_size, enc_image_size, enc_image_size, 256)
                encoded images
        """
        out = self._resnet(images)  # (batch_size, 256, 28, 28)
        self._log().debug("forward: resnet out: {}".format(out.size()))
        out = self._adaptive_pool(out)
        out = out.permute(
            0, 2, 3, 1
        )  # (batch_size, enc_image_size, enc_image_size, 256)

        self._log().debug("enc forward: final out: {}".format(out.size()))

        return out
