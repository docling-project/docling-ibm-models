# Copyright © Docling a Series of LF Projects, LLC
# For web site terms of use, trademark policy and other project policies please see https://lfprojects.org.

from unittest.mock import patch

import pytest
import torch

from docling_ibm_models.tableformer_v2.model import BboxHead, _cxcywh_to_xyxy


def per_cell_boxes(head, cells, memory, indices):
    if not cells.numel():
        return cells.new_empty(0, 4)
    x = head.input_norm(head.input_proj(cells))
    memory = head.kv_norm(head.kv_proj(memory[indices]))
    mask = indices[:, None] == indices[None, :]
    # Stock attention treats each cell as a separate attention batch.
    for layer in head.layers:
        queries = x.unsqueeze(0)
        output, _ = layer.self_attn(queries, queries, queries, attn_mask=~mask)
        x = layer.self_attn_norm(x + output.squeeze(0))
        output, _ = layer.cross_attn(x.unsqueeze(1), memory, memory)
        x = layer.cross_attn_norm(x + output.squeeze(1))
        x = layer.ffn_norm(x + layer.ffn(x))
    return _cxcywh_to_xyxy(torch.sigmoid(head.bbox_mlp(x)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "images,indices",
    [(1, []), (1, [0]), (1, [0] * 50), (3, [2, 0, 2, 0, 2]), (3, [2] * 7)],
)
def test_box_head_matches_per_cell_attention(dtype, images, indices):
    torch.manual_seed(21)
    head = BboxHead(16, 4, num_layers=2, dropout=0).to(dtype=dtype).eval()
    cells = torch.randn(len(indices), 16, dtype=dtype)
    memory = torch.randn(images, 11, 16, dtype=dtype)
    indices = torch.tensor(indices, dtype=torch.long)
    with torch.no_grad():
        expected = per_cell_boxes(head, cells, memory, indices)
        actual = head(cells, memory, indices)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.shape == (len(indices), 4)
    assert actual.dtype == dtype


def test_box_head_projects_each_image_once():
    head = BboxHead(16, 4, num_layers=1, dropout=0).eval()
    indices = torch.tensor([2, 0, 2, 0, 2])
    cells = torch.randn(5, 16)
    memory = torch.randn(3, 11, 16)
    layer = head.layers[0]
    with (
        torch.no_grad(),
        patch.object(head.kv_proj, "forward", wraps=head.kv_proj.forward) as project,
        patch.object(
            layer.cross_attn, "forward", wraps=layer.cross_attn.forward
        ) as attend,
    ):
        head(cells, memory, indices)
    assert project.call_count == 1
    assert project.call_args.args[0].shape == (3, 11, 16)
    assert [call.args[0].shape for call in attend.call_args_list] == [
        (1, 2, 16),
        (1, 3, 16),
    ]
    assert all(call.args[1].shape == (1, 11, 16) for call in attend.call_args_list)
