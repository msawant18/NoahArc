# src/serve/api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
import shutil
from src.data.preprocess import tile_scene
from src.models.infer import infer_folder

app = FastAPI(title="NoahArc API", version="0.1.0")


@app.post("/v1/segment")
async def segment(file: UploadFile = File(...)):
    tmp_dir = Path(tempfile.mkdtemp())
    scene_path = tmp_dir / file.filename

    with open(scene_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    processed = tmp_dir / "processed"
    outputs = tmp_dir / "outputs"
    processed.mkdir()
    outputs.mkdir()

    n_tiles = tile_scene(str(scene_path), str(processed), tile=512)

    ckpt = Path("checkpoints/best_model.pt")
    infer_folder(str(ckpt), str(processed), str(outputs), threshold=0.5)

    return {
        "tiles_processed": n_tiles,
        "output_dir": str(outputs),
        "provenance": {"source": file.filename}
    }
