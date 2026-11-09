r"""
The batched TableFormer v1 decoder (TagDecoder) against the previous token-by-token
decoding loop of TableModel04_rs.predict, kept here as the reference.
"""

import os

import cv2
import pytest
import torch
from huggingface_hub import snapshot_download

import docling_ibm_models.tableformer.common as c
from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
from docling_ibm_models.tableformer.models.table04_rs.tablemodel04_rs import (
    TableModel04_rs,
)
from docling_ibm_models.tableformer.models.table04_rs.tag_decoder_rs import TagDecoder

# (page image, table bbox on the page)
TABLES = [
    ("./tests/test_data/samples/ADS.2007.page_123.png", [178, 748, 1061, 976]),
    ("./tests/test_data/samples/ADS.2007.page_123.png", [177, 1163, 1062, 1329]),
    ("./tests/test_data/samples/PHM.2013.page_30.png", [100, 186, 1135, 525]),
]


@pytest.fixture(scope="module")
def predictor() -> TFPredictor:
    download_path = snapshot_download(repo_id="ds4sd/docling-models", revision="v2.1.0")
    save_dir = os.path.join(download_path, "model_artifacts/tableformer/accurate")
    config = c.read_config(os.path.join(save_dir, "tm_config.json"))
    config["model"]["save_dir"] = save_dir
    return TFPredictor(config, device="cpu", num_threads=2)


@pytest.fixture(scope="module")
def images(predictor) -> torch.Tensor:
    crops = []
    for png, bbox in TABLES:
        # as TFPredictor.multi_table_predict crops them
        page, scale = predictor.resize_img(cv2.imread(png), height=1024)
        x1, y1, x2, y2 = (round(v * scale) for v in bbox)
        crop = page[y1:y2, x1:x2]
        crops.append(predictor._prepare_image(crop))
    return torch.cat(crops)


@torch.no_grad()
def reference_predict(model: TableModel04_rs, img: torch.Tensor):
    r"""
    The decoding loop TableModel04_rs.predict used before TagDecoder: one token at a
    time through TMTransformerDecoder, bboxes one cell at a time.
    """
    word_map = model._init_data["word_map"]["word_map_tag"]
    tt = model._tag_transformer
    enc_out, encoder_out = model._encode(img)

    decoded_tags = torch.LongTensor([word_map["<start>"]]).unsqueeze(1)
    output_tags, tag_H_buf, cache = [], [], None
    skip_next_tag, prev_tag_ucel, first_lcel = True, False, True
    bboxes_to_merge, cur_bbox_ind, bbox_ind = {}, -1, 0
    cell_tags = [word_map[t] for t in ["fcel", "ecel", "ched", "rhed", "srow"]]
    cell_tags += [word_map["nl"], word_map["ucel"]]
    while len(output_tags) < model._max_pred_len:
        emb = tt._positional_encoding(tt._embedding(decoded_tags))
        decoded, cache = tt._decoder(emb, encoder_out, cache)
        new_tag = tt._fc(decoded[-1, :, :]).argmax(1).item()
        if new_tag == word_map["xcel"]:
            new_tag = word_map["lcel"]
        if prev_tag_ucel and new_tag == word_map["lcel"]:
            new_tag = word_map["fcel"]
        output_tags.append(new_tag)
        decoded_tags = torch.cat([decoded_tags, torch.LongTensor([[new_tag]])])
        if new_tag == word_map["<end>"]:
            break
        if not skip_next_tag and new_tag in cell_tags:
            tag_H_buf.append(decoded[-1, :, :])
            if first_lcel is not True:
                bboxes_to_merge[cur_bbox_ind] = bbox_ind
            bbox_ind += 1
        if new_tag != word_map["lcel"]:
            first_lcel = True
        elif first_lcel:
            tag_H_buf.append(decoded[-1, :, :])
            first_lcel = False
            cur_bbox_ind = bbox_ind
            bboxes_to_merge[cur_bbox_ind] = -1
            bbox_ind += 1
        skip_next_tag = new_tag in [word_map[t] for t in ["nl", "ucel", "xcel"]]
        prev_tag_ucel = new_tag == word_map["ucel"]

    outputs = [model._bbox_decoder.inference(enc_out, [h]) for h in tag_H_buf]
    classes, coords, skip = [], [], []
    for i, (cls, box) in enumerate(outputs):
        if i in bboxes_to_merge:
            skip.append(bboxes_to_merge[i])
            coords.append(model.mergebboxes(box[0], outputs[bboxes_to_merge[i]][1][0]))
            classes.append(cls[0])
        elif i not in skip:
            coords.append(box[0])
            classes.append(cls[0])
    return decoded_tags.squeeze().tolist(), torch.stack(classes), torch.stack(coords)


def test_predict_matches_reference(predictor, images):
    model = predictor._model
    for i in range(images.size(0)):
        seq, classes, coords = model.predict(images[i : i + 1], 1024, 5)
        ref_seq, ref_classes, ref_coords = reference_predict(model, images[i : i + 1])
        assert seq == ref_seq
        assert torch.equal(classes.argmax(1), ref_classes.argmax(1))
        assert torch.allclose(coords, ref_coords, atol=1e-4)


def test_predict_batch_matches_predict(predictor, images):
    model = predictor._model
    batch = model.predict_batch(images, 1024)
    assert len(batch) == images.size(0)
    for i, (seq, classes, coords) in enumerate(batch):
        one_seq, one_classes, one_coords = model.predict(images[i : i + 1], 1024, 5)
        assert seq == one_seq
        assert torch.equal(classes.argmax(1), one_classes.argmax(1))
        assert torch.allclose(coords, one_coords, atol=1e-4)


def test_max_steps_without_end(predictor, images):
    model = predictor._model
    seq, _, _ = model.predict_batch(images[:1], 5)[0]
    word_map = model._init_data["word_map"]["word_map_tag"]
    assert len(seq) == 6  # <start> and 5 tags, no <end>
    assert word_map["<end>"] not in seq


def test_cell_steps():
    word_map = {
        "<start>": 2,
        "<end>": 3,
        "ecel": 4,
        "fcel": 5,
        "lcel": 6,
        "ucel": 7,
        "xcel": 8,
        "nl": 9,
        "ched": 10,
        "rhed": 11,
        "srow": 12,
    }
    names = "ched ched nl fcel lcel lcel fcel nl ucel fcel nl <end> fcel".split()
    steps, merges = TableModel04_rs._cell_steps([word_map[n] for n in names], word_map)
    # The first tag of the table and the tag after nl/ucel get no bbox, a horizontal
    # span gets one at its first lcel, merged with the next cell's
    assert steps == [1, 2, 4, 6, 7, 10]
    assert merges == {2: 3}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_matches_eager(predictor, images):
    model = TFPredictor(predictor._config, device="cuda", num_threads=2)._model
    with torch.no_grad():
        memory = torch.cat(
            [model._encode(images[i : i + 1].cuda())[1] for i in range(images.size(0))],
            dim=1,
        )
    graph = TagDecoder(model, 4, 1024, use_cuda_graph=True)
    eager = TagDecoder(model, 4, 1024, use_cuda_graph=False)
    assert graph._graph is not None
    seqs, hidden = graph.decode(memory, 1024)
    hidden = hidden.clone()
    eager_seqs, eager_hidden = eager.decode(memory, 1024)
    assert seqs == eager_seqs
    assert torch.allclose(hidden, eager_hidden, atol=1e-5)
