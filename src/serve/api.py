# src/serve/api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import shutil
import os
from pathlib import Path
import uuid
import json
import uvicorn
import logging

logger = logging.getLogger(__name__)
app = FastAPI(title="FloodMapper Inference API", version="0.1")

class SegmentRequest(BaseModel):
    domain: str = "flood_sar"
    image_uri: Optional[str] = None  # path or vsis3 uri
    options: Optional[Dict[str,Any]] = {"tile":512, "overlap":64, "explain": True}

@app.post("/v1/segment")
async def segment(req: SegmentRequest = None, file: UploadFile = File(None)):
    run_id = uuid.uuid4().hex[:8]
    workdir = Path("data") / "serve_tmp" / run_id
    workdir.mkdir(parents=True, exist_ok=True)
    input_uri = None
    try:
        if file is not None:
            # save upload to workdir
            dest = workdir / file.filename
            with open(dest, "wb") as fh:
                shutil.copyfileobj(file.file, fh)
            input_uri = str(dest)
        elif req and req.image_uri:
            input_uri = req.image_uri
        else:
            raise HTTPException(status_code=400, detail="Provide file upload or image_uri")
        # tile scene
        from src.data.preprocess import tile_scene
        tiles_out = workdir / "tiles"
        tiles_out.mkdir(exist_ok=True)
        tile_size = req.options.get("tile", 512) if req else 512
        overlap = req.options.get("overlap", 64) if req else 64
        tile_count = tile_scene(input_uri, str(tiles_out), tile=tile_size, overlap=overlap, speckle=True, normalize="zscore")
        # run inference on tiles
        from src.models.infer import infer_folder
        ckpt = os.environ.get("FLOODMAPPER_CKPT", "artifacts/checkpoints/best.pt")
        out_tiles = workdir / "out_tiles"
        out_tiles.mkdir(exist_ok=True)
        infer_folder(ckpt, str(tiles_out), str(out_tiles), threshold=0.5)
        # build outputs list
        outputs = {"tiles": []}
        for p in sorted(out_tiles.glob("*")):
            outputs["tiles"].append(str(p.resolve()))
        # explainability: pick first tile and make CAM
        explain_uri = None
        if req and req.options.get("explain", True):
            try:
                from src.explainability.gradcam import simple_cam
                import numpy as np
                model_device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
                # load model
                from src.models.unet import UNet
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = UNet(in_channels=1, out_channels=1).to(device)
                # load ckpt if exists
                ckpt_path = Path(ckpt)
                if ckpt_path.exists():
                    d = torch.load(str(ckpt_path), map_location=device)
                    model.load_state_dict(d.get("model_state", d))
                # choose first tile file (npy)
                first_proba = next(out_tiles.glob("*.proba.npy"), None)
                if first_proba is not None:
                    tile_name = first_proba.stem.replace(".proba","")
                    tile_tif = tiles_out / (tile_name + ".tif")
                    if tile_tif.exists():
                        arr = np.load(str(tile_tif.with_suffix(".npy"))) if tile_tif.with_suffix(".npy").exists() else None
                    else:
                        # try reading tif using rasterio
                        import rasterio
                        with rasterio.open(str(tiles_out / (tile_name + ".tif"))) as src:
                            arr = src.read(1)
                    if arr is not None:
                        x = torch.from_numpy((arr - arr.mean())/(arr.std()+1e-6))[None,None,:,:].float().to(device)
                        cam = simple_cam(model, x)
                        cam_path = workdir / "cam.png"
                        import matplotlib.pyplot as plt
                        plt.imsave(str(cam_path), cam, cmap="viridis")
                        explain_uri = str(cam_path.resolve())
            except Exception as e:
                logger.exception("Explainability generation failed: %s", e)
        # caption: simple heuristic
        caption = f"Processed {tile_count} tiles from input. Example outputs in {str(out_tiles.resolve())}"
        provenance = {"model": str(ckpt), "threshold": 0.5, "calibration": None}
        response = {
            "scene_id": Path(input_uri).stem,
            "outputs": {"tile_outputs_dir": str(out_tiles.resolve()), "explain_png": explain_uri},
            "caption": caption,
            "provenance": provenance,
            "policy": {"crs_kept": True, "geojson_exported": False}
        }
        return response
    finally:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
