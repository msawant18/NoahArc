"""
Export predictions to GeoTIFF.
"""
import numpy as np
from pathlib import Path
try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RIO = True
except Exception:
    HAS_RIO = False

def save_mask_geotiff(mask, out_path, transform=(0,1,0,0), crs="EPSG:4326"):
    """
    mask: 2D numpy array (0/1) or float prob
    transform: (left, xres, 0, top) rough; for synthetic runs it's placeholder
    """
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if HAS_RIO:
        h,w = mask.shape
        # simple placeholder transform (left, xres, 0, top, 0, -yres)
        left, xres, _, top = transform
        tf = from_origin(left, top, xres, xres)
        dtype = "float32" if mask.dtype.kind == "f" else "uint8"
        with rasterio.open(str(outp), 'w', driver='GTiff', height=h, width=w, count=1, dtype=dtype, crs=crs, transform=tf) as dst:
            dst.write(mask.astype(dtype), 1)
    else:
        # fallback: save as numpy
        np.save(str(outp.with_suffix(".npy")), mask)
