# tests/test_integration_sentinel.py

import os
import pytest
from pathlib import Path
import rasterio
import numpy as np
import tempfile
import shutil

# Import your functions
from src.data.preprocess import tile_scene
from src.models.unet import UNet
from src.models.infer import infer_folder
from src.evaluation.metrics import iou_score

TEST_SCENE_URL = "https://storage.googleapis.com/sen1floods11/sen1floods11/chips/Bolivia_0/Bolivia_0_S1_raw.tif"
# and ground truth label URL (water mask)
TEST_LABEL_URL = "https://storage.googleapis.com/sen1floods11/sen1floods11/chips/Bolivia_0/Bolivia_0_S1_label.tif"
# These URLs might need adjustment to a chip you know exists and is small.

@pytest.mark.integration
def test_sentinel_patch_inference(tmp_path):
    # Setup
    scene_path = tmp_path / "scene.tif"
    label_path = tmp_path / "label.tif"
    processed_dir = tmp_path / "processed"
    outputs_dir = tmp_path / "outputs"
    processed_dir.mkdir()
    outputs_dir.mkdir()

    # Download scene and label
    import requests
    r = requests.get(TEST_SCENE_URL, stream=True)
    if r.status_code != 200:
        pytest.skip("Cannot download test scene; skipping integration test")
    with open(scene_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    r2 = requests.get(TEST_LABEL_URL, stream=True)
    if r2.status_code != 200:
        pytest.skip("Cannot download test label; skipping integration test")
    with open(label_path, "wb") as f:
        for chunk in r2.iter_content(8192):
            f.write(chunk)

    # Preprocess (tile)
    nt = tile_scene(str(scene_path), str(processed_dir), tile=256, overlap=0, speckle=False, normalize="zscore")

    assert nt > 0, "No tiles produced"

    # For inference, use a minimal UNet (random weights) to produce outputs
    # Save a dummy checkpoint
    import torch
    model = UNet(in_channels=1, out_channels=1)
    # We won't train; we just test shape
    ckpt_path = tmp_path / "dummy_best.pt"
    torch.save({"model_state": model.state_dict()}, str(ckpt_path))

    # Now run inference on processed tiles
    infer_folder(str(ckpt_path), str(processed_dir), str(outputs_dir), threshold=0.5)

    # Check one mask output exists
    mask_files = list(Path(outputs_dir).glob("*.pred.npy")) + list(Path(outputs_dir).glob("*.pred.tif"))
    assert len(mask_files) > 0, "No predicted mask files produced"

    # Compare one mask with label: load matching tile of label
    # For simplicity, take first pred, then find the corresponding area in label image
    pred = None
    for pf in mask_files:
        pred = pf
        break
    # Load prediction and ground truth tile
    # If prediction is .npy:
    if str(pred).endswith(".npy"):
        pred_arr = np.load(str(pred))
    else:
        with rasterio.open(str(pred)) as ds:
            pred_arr = ds.read(1).astype(np.uint8)

    # Crop ground truth label to match prediction tile spatially
    # Simplest: read first tile bounds from its JSON (if exists), else assume aligned and read first window
    # Here we assume exact alignment: load full label and resize / crop to pred_arr shape
    with rasterio.open(str(label_path)) as ds2:
        gt_full = ds2.read(1).astype(np.uint8)
    # if gt_full bigger, crop center
    y0 = (gt_full.shape[0] - pred_arr.shape[0]) // 2
    x0 = (gt_full.shape[1] - pred_arr.shape[1]) // 2
    gt_crop = gt_full[y0:y0+pred_arr.shape[0], x0:x0+pred_arr.shape[1]]

    iou = iou_score(gt_crop, pred_arr)
    # since model is untrained, expect low iou, but greater than -inf; we assert it runs
    assert 0.0 <= iou <= 1.0, f"IoU out of bounds: {iou}"

    # Optionally assert mask shape matches tile shape
    assert pred_arr.shape == gt_crop.shape

    # Clean up
    # shutil.rmtree(tmp_path)  # Let pytest handle temp
