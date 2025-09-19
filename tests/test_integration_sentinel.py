# tests/test_integration_sentinel.py

import os
import pytest
from pathlib import Path
from src.data.preprocess import tile_scene
from src.models.unet import UNet
from src.models.infer import infer_folder
from src.evaluation.metrics import iou_score
import numpy as np
import torch
import rasterio


@pytest.mark.integration
def test_small_sentinel_patch(tmp_path):
    sample_scene = "tests/data/sample_s1_patch.tif"
    gt_label = "tests/data/sample_s1_patch_label.tif"
    if not Path(sample_scene).exists():
        pytest.skip("Sample Sentinel-1 patch not present")

    processed = tmp_path / "proc"
    outputs = tmp_path / "out"
    processed.mkdir()
    outputs.mkdir()

    n_tiles = tile_scene(sample_scene, str(processed), tile=256, overlap=0)
    assert n_tiles > 0

    model = UNet(in_channels=1, out_channels=1)
    ckpt = tmp_path / "dummy.pt"
    torch.save({"model_state": model.state_dict()}, ckpt)

    infer_folder(str(ckpt), str(processed), str(outputs), threshold=0.5)
    preds = list(outputs.glob("*.tif"))
    assert len(preds) > 0

    with rasterio.open(preds[0]) as ds:
        pred = ds.read(1).astype(np.uint8)
    with rasterio.open(gt_label) as ds2:
        gt = ds2.read(1).astype(np.uint8)

    gt_crop = gt[: pred.shape[0], : pred.shape[1]]
    iou = iou_score(gt_crop, pred)
    assert 0 <= iou <= 1
