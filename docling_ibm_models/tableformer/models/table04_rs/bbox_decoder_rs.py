import logging

import torch
import torch.nn as nn

import docling_ibm_models.tableformer.settings as s
import docling_ibm_models.tableformer.utils.utils as u

# from scipy.optimize import linear_sum_assignment

LOG_LEVEL = logging.INFO


class CellAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, tag_decoder_dim, language_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param tag_decoder_dim: size of tag decoder's RNN
        :param language_dim: size of language model's RNN
        :param attention_dim: size of the attention network
        """
        super(CellAttention, self).__init__()
        # linear layer to transform encoded image
        self._encoder_att = nn.Linear(encoder_dim, attention_dim)
        # linear layer to transform tag decoder output
        self._tag_decoder_att = nn.Linear(tag_decoder_dim, attention_dim)
        # linear layer to transform language models output
        self._language_att = nn.Linear(language_dim, attention_dim)
        # linear layer to calculate values to be softmax-ed
        self._full_att = nn.Linear(attention_dim, 1)
        self._relu = nn.ReLU()
        self._softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def forward(self, encoder_out, decoder_hidden, language_out):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (1, num_pixels, encoder_dim)
        :param decoder_hidden: tag decoder output, a tensor of dimension [(num_cells,
                               tag_decoder_dim)]
        :param language_out: language model output, a tensor of dimension (num_cells,
                               language_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self._encoder_att(encoder_out)  # (1, num_pixels, attention_dim)
        att2 = self._tag_decoder_att(decoder_hidden)  # (num_cells, tag_decoder_dim)
        att3 = self._language_att(language_out)  # (num_cells, attention_dim)
        att = self._full_att(
            self._relu(att1 + att2.unsqueeze(1) + att3.unsqueeze(1))
        ).squeeze(2)
        alpha = self._softmax(att)  # (num_cells, num_pixels)
        # (num_cells, encoder_dim)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


class BBoxDecoder(nn.Module):
    """
    CellDecoder generates cell content
    """

    def __init__(
        self,
        device,
        attention_dim,
        embed_dim,
        tag_decoder_dim,
        decoder_dim,
        num_classes,
        encoder_dim=512,
        dropout=0.5,
        cnn_layer_stride=1,
    ):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param tag_decoder_dim: size of tag decoder's RNN
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        :param mini_batch_size: batch size of cells to reduce GPU memory usage
        """
        super(BBoxDecoder, self).__init__()
        self._device = device
        self._encoder_dim = encoder_dim
        self._attention_dim = attention_dim
        self._embed_dim = embed_dim
        self._decoder_dim = decoder_dim
        self._dropout = dropout
        self._num_classes = num_classes

        if cnn_layer_stride is not None:
            self._input_filter = u.resnet_block(stride=cnn_layer_stride)
        # attention network
        self._attention = CellAttention(
            encoder_dim, tag_decoder_dim, decoder_dim, attention_dim
        )
        # decoder LSTMCell
        self._init_h = nn.Linear(encoder_dim, decoder_dim)

        # linear layer to create a sigmoid-activated gate
        self._f_beta = nn.Linear(decoder_dim, encoder_dim)
        self._sigmoid = nn.Sigmoid()
        self._dropout = nn.Dropout(p=self._dropout)
        self._class_embed = nn.Linear(512, self._num_classes + 1)
        self._bbox_embed = u.MLP(512, 256, 4, 3)

    def _init_hidden_state(self, encoder_out, batch_size):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self._init_h(mean_encoder_out).expand(batch_size, -1)
        return h

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def inference(self, encoder_out, tag_H):
        """
        Inference on test images with beam search
        """
        if hasattr(self, "_input_filter"):
            encoder_out = self._input_filter(encoder_out.permute(0, 3, 1, 2)).permute(
                0, 2, 3, 1
            )

        encoder_dim = encoder_out.size(3)

        # Flatten encoding (1, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(1, -1, encoder_dim)

        num_cells = len(tag_H)
        predictions_bboxes = []
        predictions_classes = []

        for c_id in range(num_cells):
            # Start decoding
            h = self._init_hidden_state(encoder_out, 1)
            cell_tag_H = tag_H[c_id]
            awe, _ = self._attention(encoder_out, cell_tag_H, h)
            gate = self._sigmoid(self._f_beta(h))
            awe = gate * awe
            h = awe * h

            predictions_bboxes.append(self._bbox_embed(h).sigmoid())
            predictions_classes.append(self._class_embed(h))
        if len(predictions_bboxes) > 0:
            predictions_bboxes = torch.stack([x[0] for x in predictions_bboxes])
        else:
            predictions_bboxes = torch.empty(0)

        if len(predictions_classes) > 0:
            predictions_classes = torch.stack([x[0] for x in predictions_classes])
        else:
            predictions_classes = torch.empty(0)

        return predictions_classes, predictions_bboxes
