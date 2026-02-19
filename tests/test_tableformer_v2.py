#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw
from transformers import AutoTokenizer
import torchvision.transforms as transforms

from docling_ibm_models.tableformer_v2 import TableFormerV2
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler


def load_tokenizer(path: str):
    """Load tokenizer from local path or HuggingFace repo."""
    import os
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    
    # Check if it's a local path with tokenizer.json
    tokenizer_file = os.path.join(path, "tokenizer.json")
    if os.path.exists(tokenizer_file):
        # Load directly from tokenizer.json
        backend_tokenizer = Tokenizer.from_file(tokenizer_file)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend_tokenizer,
            bos_token="<start>",
            eos_token="<end>",
            pad_token="<pad>",
            unk_token="[UNK]",
        )
        return tokenizer
    else:
        # Try loading from HuggingFace repo using AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            # Set special tokens if not already set
            if tokenizer.bos_token is None:
                tokenizer.bos_token = "<start>"
            if tokenizer.eos_token is None:
                tokenizer.eos_token = "<end>"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = "<pad>"
            return tokenizer
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load tokenizer from {path}. "
                f"Tried local tokenizer.json and HuggingFace AutoTokenizer. Error: {e}"
            )


# Test data matching test_tf_predictor.py structure exactly
test_data = {
    "png_images": [
        "tests/test_data/samples/ADS.2007.page_123.png",
        "tests/test_data/samples/PHM.2013.page_30.png",
        "tests/test_data/samples/empty_iocr.png",
    ],
    "table_bboxes": [
        [[178, 748, 1061, 976], [177, 1163, 1062, 1329]],  # ADS.2007 has 2 tables
        [[100, 186, 1135, 525]],  # PHM.2013 has 1 table
        [[178, 748, 1061, 976], [177, 1163, 1062, 1329]],  # empty_iocr has 2 tables
    ],
}

# Test configuration
test_config = {
    "num_threads": 1,
    "image_size": 448,
    "max_length": 512,
}


@pytest.fixture(scope="module")
def init() -> dict:
    r"""
    Initialize the testing environment
    """
    init = test_config.copy()
    init["test_data"] = test_data

    # Download model and tokenizer from HuggingFace Hub
    artifact_path = snapshot_download(repo_id="docling-project/TableFormerV2")

    # Use local checkpoint with tokenizer files
    init["artifact_path"] = "docling-project/TableFormerV2"

    return init


def test_tableformer_v2_model_loading(init: dict):
    r"""
    Test that the TableFormerV2 model loads correctly
    """
    device = "cpu"

    # Load the model
    model = TableFormerV2.from_pretrained(init["artifact_path"])
    model = model.to(device)
    model.eval()

    # Check model attributes
    assert hasattr(model, "config"), "Model missing config attribute"
    assert hasattr(model, "generate"), "Model missing generate method"
    assert hasattr(model, "forward"), "Model missing forward method"
    assert hasattr(model, "encode_images"), "Model missing encode_images method"
    assert hasattr(model, "bbox_head"), "Model missing bbox_head"

    # Check config values
    config = model.config
    assert config.model_type == "TableFormerV2", "Wrong model type"
    assert config.vocab_size > 0, "Invalid vocab size"
    assert config.embed_dim > 0, "Invalid embed_dim"
    assert len(config.data_cells) > 0, "data_cells should not be empty"


def test_tableformer_v2_tokenizer_loading(init: dict):
    r"""
    Test that the tokenizer loads correctly
    """
    tokenizer = load_tokenizer(init["artifact_path"])

    # Check tokenizer attributes
    assert tokenizer.bos_token_id is not None, "Missing bos_token_id"
    assert tokenizer.eos_token_id is not None, "Missing eos_token_id"
    assert tokenizer.pad_token_id is not None, "Missing pad_token_id"

    # Check OTSL tokens exist
    vocab = tokenizer.get_vocab()
    expected_tokens = ["<fcel>", "<ecel>", "<nl>", "<start>", "<end>", "<pad>"]
    for token in expected_tokens:
        assert token in vocab, f"Missing expected token: {token}"


def test_tableformer_v2_image_encoding(init: dict):
    r"""
    Test image encoding functionality on cropped table
    """
    device = "cpu"

    model = TableFormerV2.from_pretrained(init["artifact_path"])
    model = model.to(device)
    model.eval()

    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((init["image_size"], init["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Test with first table crop from first image
    img_fn = init["test_data"]["png_images"][0]
    table_bbox = init["test_data"]["table_bboxes"][0][0]  # First table
    
    with Image.open(img_fn) as img:
        img_rgb = img.convert("RGB")
        # Crop to table region
        x1, y1, x2, y2 = table_bbox
        table_crop = img_rgb.crop((x1, y1, x2, y2))
        image_tensor = transform(table_crop).unsqueeze(0).to(device)

        with torch.no_grad():
            encoder_outputs = model.encode_images(image_tensor)

        assert "last_hidden_state" in encoder_outputs, "Missing last_hidden_state"
        assert "spatial_size" in encoder_outputs, "Missing spatial_size"

        hidden_state = encoder_outputs["last_hidden_state"]
        assert hidden_state.dim() == 3, "Hidden state should be 3D (B, S, D)"
        assert hidden_state.size(0) == 1, "Batch size should be 1"
        assert hidden_state.size(2) == model.config.embed_dim, "Wrong embed dim"


def test_tableformer_v2_forward_pass(init: dict):
    r"""
    Test forward pass functionality on cropped table
    """
    device = "cpu"

    model = TableFormerV2.from_pretrained(init["artifact_path"])
    tokenizer = load_tokenizer(init["artifact_path"])
    model = model.to(device)
    model.eval()

    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((init["image_size"], init["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Test with first table crop
    img_fn = init["test_data"]["png_images"][0]
    table_bbox = init["test_data"]["table_bboxes"][0][0]
    
    with Image.open(img_fn) as img:
        img_rgb = img.convert("RGB")
        x1, y1, x2, y2 = table_bbox
        table_crop = img_rgb.crop((x1, y1, x2, y2))
        image_tensor = transform(table_crop).unsqueeze(0).to(device)

        # Create dummy input_ids (start token)
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)

        with torch.no_grad():
            outputs = model.forward(
                images=image_tensor,
                input_ids=input_ids,
                return_dict=True
            )

        assert outputs.logits is not None, "Missing logits"
        assert outputs.hidden_states is not None, "Missing hidden_states"
        assert outputs.logits.size(-1) == model.config.vocab_size, "Wrong vocab size in logits"


def test_tableformer_v2_predict(init: dict):
    r"""
    Test TableFormerV2 prediction on cropped table images.
    Matches the pattern from test_tf_predictor.py.
    Includes profiling similar to test_tf_predictor.py.
    """
    device = "cpu"
    viz = True  # Save visualizations
    enable_profiling = True

    # Initialize profiler (cycles started per-table for fair comparison with V1)
    profiler = AggProfiler()

    model = TableFormerV2.from_pretrained(init["artifact_path"])
    tokenizer = load_tokenizer(init["artifact_path"])
    model = model.to(device)
    model.eval()

    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((init["image_size"], init["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process each page and its tables
    for img_idx, (img_fn, table_bboxes) in enumerate(zip(
        init["test_data"]["png_images"],
        init["test_data"]["table_bboxes"]
    )):
        print(f"\n{'>'*40}")
        img_basename = os.path.basename(img_fn)
        print(f"Processing image: {img_basename}")
        print(f"Number of tables: {len(table_bboxes)}")

        with Image.open(img_fn) as page_img:
            page_rgb = page_img.convert("RGB")
            page_w, page_h = page_rgb.size

            # Process each table on the page
            for t_idx, table_bbox in enumerate(table_bboxes):
                x1, y1, x2, y2 = table_bbox
                print(f"\n  Table {t_idx}: bbox=[{x1}, {y1}, {x2}, {y2}]")

                # Crop table from page
                table_crop = page_rgb.crop((x1, y1, x2, y2))
                crop_w, crop_h = table_crop.size
                print(f"  Crop size: {crop_w}x{crop_h}")

                # Transform for model
                image_tensor = transform(table_crop).unsqueeze(0).to(device)

                # Start new profiling cycle for this table (matches V1 behavior)
                profiler.start_agg(enable=enable_profiling)

                # Run prediction (profiling is handled inside model.generate)
                with torch.no_grad():
                    output = model.generate(
                        images=image_tensor,
                        tokenizer=tokenizer,
                        max_length=init["max_length"]
                    )

                # Check output structure
                assert "generated_ids" in output, "Missing generated_ids"
                assert output["generated_ids"] is not None, "generated_ids is None"

                generated_ids = output["generated_ids"]
                assert generated_ids.dim() == 2, "generated_ids should be 2D"

                # Decode OTSL output
                decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                print(f"  Generated OTSL: {decoded_text[:80]}...")

                # Check bounding boxes
                pred_bboxes = output.get("predicted_bboxes")
                num_cells = 0
                if pred_bboxes is not None and pred_bboxes.numel() > 0:
                    # Handle (B, num_cells, 4) format
                    if pred_bboxes.dim() == 3:
                        pred_bboxes = pred_bboxes[0]
                    # Filter zero-padded boxes
                    valid_mask = pred_bboxes.sum(dim=-1) > 0
                    pred_bboxes = pred_bboxes[valid_mask]
                    num_cells = pred_bboxes.size(0)
                    print(f"  Predicted cells: {num_cells}")

                    # Validate bbox ranges
                    if num_cells > 0:
                        assert pred_bboxes.min() >= 0.0, "Bboxes should be >= 0"
                        assert pred_bboxes.max() <= 1.0, "Bboxes should be <= 1"

                # Visualization
                if viz and pred_bboxes is not None and num_cells > 0:
                    viz_root = "./tests/test_data/viz/"
                    Path(viz_root).mkdir(parents=True, exist_ok=True)

                    # Draw on the table crop
                    draw_img = table_crop.copy()
                    draw = ImageDraw.Draw(draw_img)

                    for i, bbox in enumerate(pred_bboxes.cpu().tolist()):
                        # Unnormalize to crop coordinates
                        bx1 = bbox[0] * crop_w
                        by1 = bbox[1] * crop_h
                        bx2 = bbox[2] * crop_w
                        by2 = bbox[3] * crop_h
                        draw.rectangle([bx1, by1, bx2, by2], outline="red", width=2)

                    # Also draw table bbox on full page for context
                    draw.rectangle([0, 0, crop_w-1, crop_h-1], outline="blue", width=3)

                    viz_fn = os.path.join(
                        viz_root,
                        f"tableformer_v2_{img_basename.replace('.png', '')}_table{t_idx}.png"
                    )
                    draw_img.save(viz_fn)
                    print(f"  Saved visualization: {viz_fn}")

    # Get and print profiling data
    if enable_profiling:
        profiling_data = profiler.get_data()
        print("\n" + "="*60)
        print("PROFILING DATA")
        print("="*60)
        print(json.dumps(profiling_data, indent=2, sort_keys=True))


def test_tableformer_v2_numpy_input(init: dict):
    r"""
    Test that model works with numpy array input (converted to tensor)
    """
    device = "cpu"

    model = TableFormerV2.from_pretrained(init["artifact_path"])
    tokenizer = load_tokenizer(init["artifact_path"])
    model = model.to(device)
    model.eval()

    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((init["image_size"], init["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_fn = init["test_data"]["png_images"][0]
    table_bbox = init["test_data"]["table_bboxes"][0][0]
    
    with Image.open(img_fn) as img:
        img_rgb = img.convert("RGB")
        x1, y1, x2, y2 = table_bbox
        table_crop = img_rgb.crop((x1, y1, x2, y2))

        # Convert to numpy array first, then back to PIL for transform
        np_arr = np.asarray(table_crop)
        img_from_np = Image.fromarray(np_arr)
        image_tensor = transform(img_from_np).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model.generate(
                images=image_tensor,
                tokenizer=tokenizer,
                max_length=init["max_length"]
            )

        assert "generated_ids" in output, "Missing generated_ids"
        assert output["generated_ids"] is not None, "generated_ids is None"


def test_tableformer_v2_batch_inference(init: dict):
    r"""
    Test batch inference with multiple table crops.
    Runs on GPU if available, otherwise CPU.
    """
    # Detect device: use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"BATCH INFERENCE TEST - Device: {device}")
    print(f"{'='*60}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    model = TableFormerV2.from_pretrained(init["artifact_path"])
    tokenizer = load_tokenizer(init["artifact_path"])
    model = model.to(device)
    model.eval()

    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((init["image_size"], init["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Collect table crops from all pages
    image_tensors = []
    for img_fn, table_bboxes in zip(
        init["test_data"]["png_images"],
        init["test_data"]["table_bboxes"]
    ):
        with Image.open(img_fn) as img:
            img_rgb = img.convert("RGB")
            # Take first table from each page
            x1, y1, x2, y2 = table_bboxes[0]
            table_crop = img_rgb.crop((x1, y1, x2, y2))
            image_tensors.append(transform(table_crop))

    # Stack into batch
    batch_tensor = torch.stack(image_tensors).to(device)
    batch_size = batch_tensor.size(0)
    print(f"\nBatch size: {batch_size}")
    print(f"Input tensor shape: {batch_tensor.shape}")

    # Warmup run (especially important for GPU)
    if device == "cuda":
        print("\nRunning warmup...")
        with torch.no_grad():
            _ = model.generate(
                images=batch_tensor,
                tokenizer=tokenizer,
                max_length=min(init["max_length"], 100)  # Shorter for warmup
            )
        torch.cuda.synchronize() if device == "cuda" else None

    # Actual inference with timing
    import time
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        output = model.generate(
            images=batch_tensor,
            tokenizer=tokenizer,
            max_length=init["max_length"]
        )

    if device == "cuda":
        torch.cuda.synchronize()
    inference_time = time.time() - start_time

    assert "generated_ids" in output, "Missing generated_ids"
    generated_ids = output["generated_ids"]
    assert generated_ids.size(0) == batch_size, f"Expected batch size {batch_size}"

    print(f"\nBatch inference results ({batch_size} tables):")
    for i in range(batch_size):
        decoded = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
        seq_len = generated_ids[i].size(0)
        print(f"  Table {i}: seq_len={seq_len}, {decoded[:50]}...")

    print(f"\nInference time: {inference_time:.3f} seconds")
    print(f"Time per table: {inference_time/batch_size:.3f} seconds")
    print(f"Throughput: {batch_size/inference_time:.2f} tables/second")

    # Check bounding boxes
    pred_bboxes = output.get("predicted_bboxes")
    if pred_bboxes is not None and pred_bboxes.numel() > 0:
        print(f"\nBounding boxes shape: {pred_bboxes.shape}")
        if pred_bboxes.dim() == 3:
            for b in range(batch_size):
                valid_mask = pred_bboxes[b].sum(dim=-1) > 0
                num_cells = valid_mask.sum().item()
                print(f"  Batch {b}: {num_cells} cells")


def test_tableformer_v2_unsupported_input(init: dict):
    r"""
    Test that model raises appropriate errors for unsupported inputs
    """
    device = "cpu"

    model = TableFormerV2.from_pretrained(init["artifact_path"])
    tokenizer = load_tokenizer(init["artifact_path"])
    model = model.to(device)
    model.eval()

    # Test with wrong tensor shape
    is_exception = False
    try:
        wrong_shape = torch.randn(1, 1, 224, 224).to(device)  # Wrong channels
        with torch.no_grad():
            model.generate(images=wrong_shape, tokenizer=tokenizer)
    except Exception:
        is_exception = True
    assert is_exception, "Should raise exception for wrong input shape"
