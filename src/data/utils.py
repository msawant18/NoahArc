"""
Utility functions for data.
"""
import os

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
