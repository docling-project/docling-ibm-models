# Copyright © Docling a Series of LF Projects, LLC
# For web site terms of use, trademark policy and other project policies please see https://lfprojects.org.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from docling_ibm_models.tableformer_v2.model import (
    BboxHead,
    CachedTransformerDecoder,
    TableFormerV2,
)


@pytest.fixture
def model():
    torch.manual_seed(30)
    model = TableFormerV2.__new__(TableFormerV2)
    nn.Module.__init__(model)
    model.input_embedding = nn.Embedding(4, 16)
    model.positional_encoding = nn.Parameter(torch.randn(1, 32, 16))
    model.transformer_decoder = CachedTransformerDecoder(16, 4, 32, 2)
    model.output_projection = nn.Linear(16, 4)
    model.bbox_head = BboxHead(16, 4, num_layers=1, dropout=0.0)
    model.data_cells = [2]
    model.encode_images = lambda images: {"last_hidden_state": images}
    return model.eval()


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("token", [1, 2])
def test_generate_skips_discarded_boxes(model, batch, token):
    # Force either immediate EOS or a cell at every step.
    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.zero_()
        model.output_projection.bias[token] = 10
    images = torch.randn(batch, 9, 16)
    tokenizer = SimpleNamespace(bos_token_id=0, eos_token_id=1)
    forward = model.forward

    def with_discarded_boxes(*args, **kwargs):
        kwargs["predict_bboxes"] = True
        return forward(*args, **kwargs)

    with patch.object(model, "forward", side_effect=with_discarded_boxes):
        expected = model.generate(images, tokenizer, max_length=8)
    with patch.object(
        model.bbox_head, "forward", wraps=model.bbox_head.forward
    ) as bbox:
        actual = model.generate(images, tokenizer, max_length=8)
        assert bbox.call_count == (batch if token == 2 else 0)
    torch.testing.assert_close(
        actual["generated_ids"], expected["generated_ids"], rtol=0, atol=0
    )
    if token == 2:
        torch.testing.assert_close(
            actual["predicted_bboxes"], expected["predicted_bboxes"], rtol=0, atol=0
        )
    else:
        assert actual["predicted_bboxes"] is expected["predicted_bboxes"] is None


@pytest.mark.parametrize("return_dict", [False, True])
def test_forward_keeps_boxes_by_default(model, return_dict):
    inputs = dict(
        input_ids=torch.tensor([[0, 2, 2]]),
        encoder_outputs={"last_hidden_state": torch.randn(1, 9, 16)},
        use_cache=False,
        return_dict=return_dict,
    )
    with torch.no_grad():
        expected = model(**inputs)
        with patch.object(
            model.bbox_head, "forward", wraps=model.bbox_head.forward
        ) as bbox:
            actual = model(**inputs, predict_bboxes=False)
            bbox.assert_not_called()
    for index in [0, 1]:
        torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)
    if return_dict:
        assert expected.predicted_bboxes is not None
        assert actual.predicted_bboxes is None
    else:
        assert expected[2] is not None
        assert actual[2] is None
