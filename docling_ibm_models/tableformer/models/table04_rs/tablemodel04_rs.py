import logging

import torch
import torch.nn as nn

import docling_ibm_models.tableformer.settings as s
from docling_ibm_models.tableformer.models.common.base_model import BaseModel
from docling_ibm_models.tableformer.models.table04_rs.bbox_decoder_rs import BBoxDecoder
from docling_ibm_models.tableformer.models.table04_rs.encoder04_rs import Encoder04
from docling_ibm_models.tableformer.models.table04_rs.tag_decoder_rs import (
    TagDecoders,
)
from docling_ibm_models.tableformer.models.table04_rs.transformer_rs import (
    Tag_Transformer,
)
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler

LOG_LEVEL = logging.WARN
# LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


class TableModel04_rs(BaseModel, nn.Module):
    r"""
    TableNet04Model encoder, dual-decoder model with OTSL+ support
    """

    def __init__(self, config, init_data, device):
        super(TableModel04_rs, self).__init__(config, init_data, device)

        self._prof = config["predict"].get("profiling", False)
        self._device = device
        # Extract the word_map from the init_data
        word_map = init_data["word_map"]

        # Encoder
        self._enc_image_size = config["model"]["enc_image_size"]
        self._encoder_dim = config["model"]["hidden_dim"]
        self._encoder = Encoder04(self._enc_image_size, self._encoder_dim).to(device)

        tag_vocab_size = len(word_map["word_map_tag"])

        td_encode = []
        for t in ["ecel", "fcel", "ched", "rhed", "srow"]:
            if t in word_map["word_map_tag"]:
                td_encode.append(word_map["word_map_tag"][t])
        self._log().debug("td_encode length: {}".format(len(td_encode)))
        self._log().debug("td_encode: {}".format(td_encode))

        self._tag_attention_dim = config["model"]["tag_attention_dim"]
        self._tag_embed_dim = config["model"]["tag_embed_dim"]
        self._tag_decoder_dim = config["model"]["tag_decoder_dim"]
        self._decoder_dim = config["model"]["hidden_dim"]
        self._dropout = config["model"]["dropout"]

        self._bbox = config["train"]["bbox"]
        self._bbox_attention_dim = config["model"]["bbox_attention_dim"]
        self._bbox_embed_dim = config["model"]["bbox_embed_dim"]
        self._bbox_decoder_dim = config["model"]["hidden_dim"]

        self._enc_layers = config["model"]["enc_layers"]
        self._dec_layers = config["model"]["dec_layers"]
        self._n_heads = config["model"]["nheads"]

        self._num_classes = config["model"]["bbox_classes"]
        self._enc_image_size = config["model"]["enc_image_size"]

        self._max_pred_len = config["predict"]["max_steps"]

        self._tag_transformer = Tag_Transformer(
            device,
            tag_vocab_size,
            td_encode,
            self._decoder_dim,
            self._enc_layers,
            self._dec_layers,
            self._enc_image_size,
            n_heads=self._n_heads,
        ).to(device)

        self._bbox_decoder = BBoxDecoder(
            device,
            self._bbox_attention_dim,
            self._bbox_embed_dim,
            self._tag_decoder_dim,
            self._bbox_decoder_dim,
            self._num_classes,
            self._encoder_dim,
            self._dropout,
        ).to(device)

        self._tag_decoders = None

    _logger = None

    def _log(self):
        # Setup a custom logger, built once per instance
        if self._logger is None:
            self._logger = s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)
        return self._logger

    def mergebboxes(self, bbox1, bbox2):
        new_w = (bbox2[0] + bbox2[2] / 2) - (bbox1[0] - bbox1[2] / 2)
        new_h = (bbox2[1] + bbox2[3] / 2) - (bbox1[1] - bbox1[3] / 2)

        new_left = bbox1[0] - bbox1[2] / 2
        new_top = min((bbox2[1] - bbox2[3] / 2), (bbox1[1] - bbox1[3] / 2))

        new_cx = new_left + new_w / 2
        new_cy = new_top + new_h / 2

        bboxm = torch.tensor([new_cx, new_cy, new_w, new_h])
        return bboxm

    def predict(self, imgs, max_steps, k, return_attention=False):
        r"""
        Inference.
        The input image must be preprocessed and transformed.

        Parameters
        ----------
        img : tensor FloatTensor - torch.Size([1, 3, 448, 448])
            Input image for the inference

        Returns
        -------
        seq : list
            Predictions for the tags as indices over the word_map
        outputs_class : tensor(x, 3)
            Classes of predicted bboxes. x is the number of bboxes. There are 3 bbox classes

        outputs_coord : tensor(x, 4)
            Coords of predicted bboxes. x is the number of bboxes. Each bbox is in [cxcywh] format
        """
        return self.predict_batch(imgs, max_steps)[0]

    @torch.no_grad()
    def predict_batch(self, imgs, max_steps):
        r"""
        Inference for several tables at once.

        Parameters
        ----------
        imgs : tensor FloatTensor - torch.Size([n, 3, 448, 448])
            Preprocessed table images
        max_steps : int
            Maximum number of tags per table

        Returns
        -------
        list of (seq, outputs_class, outputs_coord), one per table, as predict()
        """
        AggProfiler().begin("predict_total", self._prof)
        self._tag_transformer.eval()
        max_steps = min(max_steps, self._max_pred_len)
        if self._tag_decoders is None:
            self._tag_decoders = TagDecoders(self, self._max_pred_len)
        decoders = self._tag_decoders

        results = []
        with decoders.lock:
            for s in range(0, imgs.size(0), decoders.max_batch_size):
                chunk = imgs[s : s + decoders.max_batch_size]
                # The encoders run per table: batching the CNN changes the
                # results by more than float noise (different conv kernels)
                encoded = [self._encode(chunk[i : i + 1]) for i in range(chunk.size(0))]
                memory = torch.cat([m for _, m in encoded], dim=1)
                AggProfiler().begin("model_tag_transformer_decoder", self._prof)
                seqs, hidden = decoders.get(chunk.size(0)).decode(memory, max_steps)
                AggProfiler().end("model_tag_transformer_decoder", self._prof)
                for i, seq in enumerate(seqs):
                    results.append(self._table_output(seq, hidden[:, i], encoded[i][0]))
        AggProfiler().end("predict_total", self._prof)
        return results

    def _encode(self, img):
        r"""
        CNN encoder and tag transformer encoder of one table.

        Returns the CNN output (1, 28, 28, C) and the encoder memory (784, 1, 512).
        """
        enc_out = self._encoder(img)
        AggProfiler().end("model_encoder", self._prof)
        n_heads = self._tag_transformer._n_heads
        encoder_out = self._tag_transformer._input_filter(
            enc_out.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        enc_inputs = encoder_out.view(batch_size, -1, encoder_dim).to(self._device)
        enc_inputs = enc_inputs.permute(1, 0, 2)
        positions = enc_inputs.shape[0]
        encoder_mask = torch.zeros(
            (batch_size * n_heads, positions, positions), device=self._device
        ) == torch.ones(
            (batch_size * n_heads, positions, positions), device=self._device
        )
        AggProfiler().begin("model_tag_transformer_encoder", self._prof)
        memory = self._tag_transformer._encoder(enc_inputs, mask=encoder_mask)
        AggProfiler().end("model_tag_transformer_encoder", self._prof)
        return enc_out, memory

    def _table_output(self, seq, hidden, enc_out):
        r"""
        Bboxes of one decoded table.

        Parameters
        ----------
        seq : list of int
            <start>, the generated tags and <end> (if reached)
        hidden : tensor (steps, 512)
            Last decoder layer output of every step
        enc_out : tensor (1, 28, 28, C)
            CNN output of the table
        """
        word_map = self._init_data["word_map"]["word_map_tag"]
        cell_steps, bboxes_to_merge = self._cell_steps(seq[1:], word_map)

        if self._bbox:
            AggProfiler().begin("model_bbox_decoder", self._prof)
            outputs_class, outputs_coord = self._bbox_decoder.inference(
                enc_out, hidden[cell_steps]
            )
            AggProfiler().end("model_bbox_decoder", self._prof)
        else:
            return seq, None, None

        ########################################################################################
        # Merge First and Last predicted BBOX for each span, according to bboxes_to_merge
        ########################################################################################

        outputs_class1 = []
        outputs_coord1 = []
        boxes_to_skip = []

        for box_ind in range(len(outputs_coord)):
            box1 = outputs_coord[box_ind].to(self._device)
            cls1 = outputs_class[box_ind].to(self._device)
            if box_ind in bboxes_to_merge:
                box2 = outputs_coord[bboxes_to_merge[box_ind]].to(self._device)
                boxes_to_skip.append(bboxes_to_merge[box_ind])
                boxm = self.mergebboxes(box1, box2).to(self._device)
                outputs_coord1.append(boxm)
                outputs_class1.append(cls1)
            else:
                if box_ind not in boxes_to_skip:
                    outputs_coord1.append(box1)
                    outputs_class1.append(cls1)

        if len(outputs_coord1) > 0:
            outputs_coord1 = torch.stack(outputs_coord1)
        else:
            outputs_coord1 = torch.empty(0)
        if len(outputs_class1) > 0:
            outputs_class1 = torch.stack(outputs_class1)
        else:
            outputs_class1 = torch.empty(0)

        num_tab_cells = seq.count(4) + seq.count(5)
        num_rows = seq.count(9)
        self._log().info(
            "OTSL predicted table cells#: {}; rows#: {}".format(num_tab_cells, num_rows)
        )
        return seq, outputs_class1, outputs_coord1

    @staticmethod
    def _cell_steps(tags, word_map):
        r"""
        The steps whose decoder output feeds the bbox decoder, and the first and last
        bbox of every horizontal span, from the generated tags.

        Returns
        -------
        cell_steps : list of int
        bboxes_to_merge : dict
            Index of the first bbox of a span -> index of its last bbox (-1: none)
        """
        cell_tags = {
            word_map[t] for t in ["fcel", "ecel", "ched", "rhed", "srow", "nl", "ucel"]
        }
        skip_after = {word_map["nl"], word_map["ucel"], word_map["xcel"]}
        lcel = word_map["lcel"]

        cell_steps = []
        bboxes_to_merge = {}
        skip_next_tag = True
        first_lcel = True
        cur_bbox_ind = -1
        bbox_ind = 0
        for step, tag in enumerate(tags):
            if tag == word_map["<end>"]:
                break
            # MAKE SURE TO SYNC NUMBER OF CELLS WITH NUMBER OF BBOXes
            if not skip_next_tag and tag in cell_tags:
                cell_steps.append(step)
                if first_lcel is not True:
                    # Mark end index for horizontal cell bbox merge
                    bboxes_to_merge[cur_bbox_ind] = bbox_ind
                bbox_ind += 1
            # Treat horizontal span bboxes...
            if tag != lcel:
                first_lcel = True
            elif first_lcel:
                # Beginning of horizontal span
                cell_steps.append(step)
                first_lcel = False
                cur_bbox_ind = bbox_ind
                bboxes_to_merge[cur_bbox_ind] = -1
                bbox_ind += 1
            skip_next_tag = tag in skip_after
        return cell_steps, bboxes_to_merge
