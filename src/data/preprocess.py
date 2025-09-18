# src/data/preprocess.py
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import trange
from src.data.utils import ensure_dir
import scipy.ndimage as ndi

def _make_sar_like_tile(h=512, w=512, water_frac=0.1, speckle_sigma=0.6, seed=None):
    """
    Generate a synthetic SAR-like tile and binary flood mask.
    Water: low backscatter (centered near -2). Land: higher values.
    """
    if seed is not None:
        np.random.seed(seed)
    base = np.random.normal(loc=0.0, scale=1.0, size=(h,w))
    # add low-intensity water patches
    mask = np.zeros((h,w), dtype=np.uint8)
    n_patches = max(1, int(w*h*water_frac / (50*50)))
    for _ in range(n_patches):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        rx = np.random.randint(20, 120)
        ry = np.random.randint(20, 120)
        y,x = np.ogrid[-cy:h-cy, -cx:w-cx]
        circle = (x*x)/(rx*rx) + (y*y)/(ry*ry) <= 1
        mask[circle] = 1
        base[circle] += -3.0  # depress water backscatter
    # add speckle
    speckle = np.random.normal(1.0, speckle_sigma, size=(h,w))
    sar = base * speckle
    # optional smoothing
    sar = ndi.gaussian_filter(sar, sigma=1.0)
    return sar.astype(np.float32), mask

def write_tile(tile, mask, out_dir, idx):
    p = Path(out_dir)
    tfile = p / f"tile_{idx:05d}.npy"
    mfile = p / f"tile_{idx:05d}.mask.npy"
    np.save(str(tfile), tile)
    np.save(str(mfile), mask)

def generate_synthetic_tiles(out_dir: str, n_tiles: int = 200, tile_size=512, seed=42):
    ensure_dir(out_dir)
    for i in trange(n_tiles, desc="generate tiles"):
        sar, mask = _make_sar_like_tile(h=tile_size, w=tile_size, water_frac=0.08, seed=seed+i)
        write_tile(sar, mask, out_dir, i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--n_tiles", type=int, default=200)
    parser.add_argument("--tile_size", type=int, default=512)
    args = parser.parse_args()
    generate_synthetic_tiles(args.out, n_tiles=args.n_tiles, tile_size=args.tile_size)
