"""
Inference script.
"""
import argparse
import torch
import numpy as np
from src.models.unet import UNet
from pathlib import Path
from tqdm import tqdm
import os

def load_model(ckpt_path, device):
    model = UNet(in_channels=1, out_channels=1).to(device)
    d = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(d["model_state"] if "model_state" in d else d)
    model.eval()
    return model

def infer_folder(ckpt, in_dir, out_dir, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt, device)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in Path(in_dir).glob("*.npy") if "mask" not in p.name])
    for p in tqdm(files, desc="inference"):
        arr = np.load(str(p)).astype(np.float32)
        tile = (arr - np.nanmean(arr))/ (np.nanstd(arr)+1e-6)
        x = torch.from_numpy(tile[None,None,:,:]).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0,0]
        pred = (probs >= threshold).astype(np.uint8)
        np.save(str(Path(out_dir)/ (p.stem + ".proba.npy")), probs)
        np.save(str(Path(out_dir)/ (p.stem + ".pred.npy")), pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--in_dir", default="data/processed")
    parser.add_argument("--out_dir", default="data/outputs")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    infer_folder(args.ckpt, args.in_dir, args.out_dir, threshold=args.threshold)
