# src/data/preprocess.py
import os
import rasterio
from rasterio.windows import Window
import numpy as np
import json
from pathlib import Path
from src.data.utils import ensure_dir, make_record, bounds_from_transform


def lee_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
    """Apply Lee filter for speckle noise reduction."""
    from scipy.ndimage import uniform_filter, variance
    img_mean = uniform_filter(img, size)
    img_sqr_mean = uniform_filter(img ** 2, size)
    img_variance = img_sqr_mean - img_mean ** 2
    overall_variance = np.var(img)
    img_weights = img_variance / (img_variance + overall_variance)
    return img_mean + img_weights * (img - img_mean)


def normalize_zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score normalize, ignoring NaNs."""
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    return (arr - mean) / (std + 1e-6)


def sliding_windows(width, height, tile, overlap):
    """Yield (x, y, w, h) windows with overlap."""
    step = tile - overlap
    for y in range(0, height, step):
        for x in range(0, width, step):
            w = min(tile, width - x)
            h = min(tile, height - y)
            yield x, y, w, h


def tile_scene(scene_uri: str, out_dir: str, tile: int = 512,
               overlap: int = 64, band: int = 1, speckle: bool = True,
               normalize: str = "zscore"):
    """Tile a Sentinel-1 scene into GeoTIFF chips and JSON records."""
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    idx = 0
    with rasterio.open(scene_uri) as src:
        width, height = src.width, src.height

        for x, y, w, h in sliding_windows(width, height, tile, overlap):
            window = Window(x, y, w, h)
            arr = src.read(band, window=window).astype(np.float32)

            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan

            if speckle:
                arr = lee_filter(arr)

            if normalize == "zscore":
                arr = normalize_zscore(arr)

            # Save tile
            profile = src.profile.copy()
            profile.update({
                "height": h,
                "width": w,
                "count": 1,
                "dtype": "float32",
                "transform": src.window_transform(window)
            })

            tile_id = f"{Path(scene_uri).stem}_tile_{idx:05d}"
            out_tif = out_dir / f"{tile_id}.tif"
            with rasterio.open(out_tif, "w", **profile) as dst:
                dst.write(arr, 1)

            # Save record
            bounds = bounds_from_transform(profile["transform"], w, h)
            record = make_record(
                id=tile_id,
                source="sentinel1_sar_vv",
                crs=src.crs.to_string(),
                pixel_spacing=[abs(profile["transform"].a), abs(profile["transform"].e)],
                shape=[w, h],
                bounds=bounds,
                processing=[
                    "lee_filter" if speckle else "none",
                    f"normalize:{normalize}"
                ]
            )
            with open(out_dir / f"{tile_id}.json", "w") as f:
                json.dump(record, f, indent=2)

            idx += 1

    return idx
