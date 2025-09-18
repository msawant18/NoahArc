"""
Export predictions to GeoTIFF.
"""
import numpy as np
from pathlib import Path
try:
    import rasterio
    from rasterio.shutil import copy as rio_copy
    HAS_RIO = True
except Exception:
    HAS_RIO = False

def save_mask_geotiff(mask: np.ndarray, out_path: str, reference_raster: str = None, dtype="uint8", compress="deflate"):
    """
    Save a mask/proba array as GeoTIFF; if reference_raster provided, copy its CRS & transform.
    """
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if not HAS_RIO:
        # fallback
        np.save(str(outp.with_suffix(".npy")), mask)
        return

    if reference_raster:
        with rasterio.open(reference_raster) as ref:
            profile = ref.profile.copy()
            profile.update({
                "height": mask.shape[0],
                "width": mask.shape[1],
                "count": 1,
                "dtype": dtype,
                "compress": compress
            })
            with rasterio.open(str(outp), "w", **profile) as dst:
                dst.write(mask.astype(dtype), 1)
    else:
        # no reference – create a simple profile with WGS84 (WARNING: user should supply reference)
        h,w = mask.shape
        profile = {
            "driver":"GTiff",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": dtype,
            "crs": "EPSG:4326",
            "transform": rasterio.transform.from_origin(0,0,1,1),
            "compress": compress
        }
        with rasterio.open(str(outp), "w", **profile) as dst:
            dst.write(mask.astype(dtype), 1)
