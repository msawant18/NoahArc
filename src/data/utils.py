"""
Utility functions for data.
"""
import os
from typing import Tuple
from affine import Affine

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path
def make_record(tile_id: str, metadata: dict, label: str = None):
    """
    Construct a record dictionary for a Sentinel-1 tile.
    Args:
        tile_id (str): Unique tile identifier
        metadata (dict): Metadata info (bounds, timestamp, etc.)
        label (str, optional): Flood label if available
    
    Returns:
        dict: A structured record for downstream processing
    """
    record = {
        "id": tile_id,
        "metadata": metadata,
    }
    if label is not None:
        record["label"] = label
    return record
def bounds_from_transform(transform: Affine, width: int, height: int) -> Tuple[float, float, float, float]:
    """
    Compute bounding box (left, bottom, right, top) from a raster transform and shape.
    
    Args:
        transform (Affine): Affine transform of the raster
        width (int): Number of columns (x direction)
        height (int): Number of rows (y direction)

    Returns:
        Tuple[float, float, float, float]: (left, bottom, right, top)
    """
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    return (left, bottom, right, top)
