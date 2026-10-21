"""Regressions for deferring table boxes until token generation is complete."""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from docling_ibm_models.tableformer_v2 import TableFormerV2, TableFormerV2Config


@pytest.fixture
def model() -> Iterator[TableFormerV2]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    config = TableFormerV2Config(
        vocab_size=13,
        embed_dim=16,
        num_heads=2,
        ff_dim=32,
        num_decoder_layers=1,
        data_cells=[4, 5, 10, 11, 12],
    )
    subject = TableFormerV2(config).eval()
    try:
        yield subject
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("return_dict", [True, False])
def test_forward_can_skip_boxes_without_changing_decoder_outputs(
    model: TableFormerV2, return_dict: bool
) -> None:
    input_ids = torch.tensor([[2, 5, 9, 5, 3]])
    encoder_outputs = {"last_hidden_state": torch.randn(1, 4, 16)}
    with torch.no_grad():
        default = model.forward(
            input_ids=input_ids,
            encoder_outputs=encoder_outputs,
            return_dict=return_dict,
        )
        skipped = model.forward(
            input_ids=input_ids,
            encoder_outputs=encoder_outputs,
            return_dict=return_dict,
            predict_bboxes=False,
        )

    if return_dict:
        assert default.predicted_bboxes.shape == (2, 4)
        assert skipped.predicted_bboxes is None
        torch.testing.assert_close(default.logits, skipped.logits, rtol=0, atol=0)
        torch.testing.assert_close(
            default.hidden_states, skipped.hidden_states, rtol=0, atol=0
        )
        torch.testing.assert_close(
            default.past_key_values, skipped.past_key_values, rtol=0, atol=0
        )
    else:
        assert len(default) == len(skipped) == 4
        assert default[2].shape == (2, 4)
        assert skipped[2] is None
        for index in (0, 1, 3):
            torch.testing.assert_close(default[index], skipped[index], rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "next_token", [3, 5, 9], ids=["early-eos", "cells", "no-cells"]
)
def test_generate_matches_final_forward_boxes_without_intermediate_box_calls(
    model: TableFormerV2, batch_size: int, next_token: int
) -> None:
    # Control the greedy token choice while retaining the real encoder,
    # decoder, cache, and box head. This covers EOS, length limits, and no cells.
    tokenizer = SimpleNamespace(bos_token_id=2, eos_token_id=3)
    max_length = 5
    images = torch.randn(batch_size, 3, 32, 32)
    generated_steps = 1 if next_token == tokenizer.eos_token_id else max_length
    expected_ids = torch.full((batch_size, generated_steps + 1), next_token)
    expected_ids[:, 0] = tokenizer.bos_token_id

    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.fill_(-10)
        model.output_projection.bias[next_token] = 10
        encoder_outputs = model.encode_images(images)
        final = model.forward(
            input_ids=expected_ids,
            encoder_outputs=encoder_outputs,
            use_cache=False,
        )
        expected_boxes = None
        if next_token == 5:
            # This is the original generate() final per-table box calculation.
            expected_boxes = torch.stack(
                [
                    model.bbox_head(
                        final.hidden_states[index, 1:],
                        encoder_outputs["last_hidden_state"][index : index + 1],
                        torch.zeros(generated_steps, dtype=torch.long),
                        spatial_size=encoder_outputs.get("spatial_size"),
                    )
                    for index in range(batch_size)
                ]
            )

    calls = []
    handle = model.bbox_head.register_forward_hook(
        lambda _module, arguments, _output: calls.append(arguments[0].shape[0])
    )
    try:
        output = model.generate(images, tokenizer, max_length=max_length)
    finally:
        handle.remove()

    torch.testing.assert_close(output["generated_ids"], expected_ids, rtol=0, atol=0)
    if expected_boxes is None:
        assert output["predicted_bboxes"] is None
        assert calls == []
    else:
        torch.testing.assert_close(
            output["predicted_bboxes"], expected_boxes, rtol=0, atol=0
        )
        assert calls == [generated_steps] * batch_size
