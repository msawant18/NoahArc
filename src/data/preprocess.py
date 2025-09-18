# src/data/preprocess.py
import os
import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import json
import math
from src.data.utils import ensure_dir
import logging

logger = logging.getLogger(__name__)

def lee_filter(img: np.ndarray, size=3):
    """
    Basic Lee speckle filter implementation for single-band images.
    img: 2D float array (linear scale). Returns filtered array.
    Reference: basic local statistics approach.
    """
    from scipy.ndimage import uniform_filter
    img = img.astype(np.float32)
    mean = uniform_filter(img, size=size)
    mean_sq = uniform_filter(img*img, size=size)
    var = mean_sq - mean*mean
    overall_var = np.nanmean(var)
    # avoid division by zero
    weight = var / (var + overall_var + 1e-12)
    result = mean + weight * (img - mean)
    return result

def normalize_zscore(arr: np.ndarray):
    valid = ~np.isnan(arr)
    if not valid.any():
        return arr
    mu = np.nanmean(arr)
    sigma = np.nanstd(arr)
    sigma = sigma if sigma > 1e-6 else 1.0
    out = (arr - mu) / sigma
    out[~valid] = np.nan
    return out

def sliding_windows(width: int, height: int, tile: int, overlap: int):
    step = tile - overlap
    if step <= 0:
        raise ValueError("tile must be larger than overlap")
    xs = list(range(0, max(1, width - overlap), step))
    ys = list(range(0, max(1, height - overlap), step))
    for y in ys:
        for x in xs:
            w = min(tile, width - x)
            h = min(tile, height - y)
            yield x, y, w, h

def bounds_from_transform(transform: Affine, w: int, h: int):
    minx, maxy = transform * (0, 0)
    maxx, miny = transform * (w, h)
    return [float(minx), float(miny), float(maxx), float(maxy)]

def make_record(image_id, modality, crs, pixel_spacing, tile_size, bounds, provenance):
    return {
        "image_id": image_id,
        "modality": modality,
        "crs": crs,
        "pixel_spacing": pixel_spacing,
        "tile_size": tile_size,
        "bounds": bounds,
        "provenance": provenance,
        "label_set": ["flooded", "non_flooded"]
    }

def tile_scene(scene_uri: str, out_dir: str, tile: int = 512, overlap: int = 64, band: int = 1, speckle=True, normalize="zscore"):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    logger.info(f"Opening scene: {scene_uri}")
    with rasterio.open(scene_uri) as src:
        height = src.height
        width = src.width
        crs = src.crs.to_string() if src.crs else None
        profile = src.profile
        transform = src.transform
        pixel_x = abs(transform.a)
        pixel_y = abs(transform.e) if transform.e != 0 else pixel_x
        # iterate windows
        idx = 0
        for x, y, w, h in sliding_windows(width, height, tile, overlap):
            window = Window(xoff=x, yoff=y, width=w, height=h)
            arr = src.read(band, window=window).astype(np.float32)
            # set nodata to NaN if src has nodata
            if src.nodata is not None:
                arr[arr==src.nodata] = np.nan
            if speckle:
                try:
                    arr = lee_filter(arr, size=3)
                except Exception as e:
                    logger.warning("Lee filter failed: %s", e)
            if normalize == "zscore":
                arr = normalize_zscore(arr)
            # write tile GeoTIFF
            tile_transform = src.window_transform(window)
            tile_profile = profile.copy()
            tile_profile.update({
                "height": arr.shape[0],
                "width": arr.shape[1],
                "count": 1,
                "dtype": "float32",
                "transform": tile_transform
            })
            scene_name = Path(scene_uri).stem
            tile_id = f"{scene_name}_tile_{idx:05d}"
            out_tif = out_dir / f"{tile_id}.tif"
            with rasterio.open(out_tif, "w", **tile_profile) as dst:
                dst.write(arr, 1)
            # write record JSON
            bounds = bounds_from_transform(tile_transform, arr.shape[1], arr.shape[0])
            record = make_record(tile_id, "sentinel1_sar_vv", crs, [pixel_x, pixel_y], [w, h], bounds, {"source_uri": str(scene_uri), "processing": ["lee_filter" if speckle else "none", f"normalize:{normalize}"]})
            json_path = out_dir / f"{tile_id}.json"
            with open(json_path, "w") as fh:
                json.dump(record, fh, indent=2)
            idx += 1
    logger.info("Tiling complete: wrote %d tiles", idx)
    return idx

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_uri", required=True, help="Path or URI to a Sentinel-1-derived GeoTIFF (VV)")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--band", type=int, default=1)
    parser.add_argument("--speckle", action="store_true")
    parser.add_argument("--normalize", default="zscore")
    args = parser.parse_args()
    tile_scene(args.scene_uri, args.out_dir, tile=args.tile, overlap=args.overlap, band=args.band, speckle=args.speckle, normalize=args.normalize)
