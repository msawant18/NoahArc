"""
Download Sentinel-1 SAR tiles (placeholder).
"""
Behavior:
- If environment variables COPERNICUS_USER and COPERNICUS_PASS are set, uses sentinelsat
  to query and download products by area/time.
- Otherwise, expects user to place scene(s) in `data/raw/` and will act as a no-op.
- Optionally supports reading S3 URIs via rasterio (rasterio will handle vsis3 if AWS creds present).

Note: sentinelsat is the recommended route for programmatic download from Copernicus.
"""
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

def download_with_sentinelsat(footprint_wkt: str, start_date: str, end_date: str, out_dir: str, product_type="GRD"):
    """
    Query Copernicus Open Access Hub for Sentinel-1 products covering the footprint and date range.
    Requires COPERNICUS_USER and COPERNICUS_PASS env vars to be set.
    Returns list of local file paths downloaded.
    """
    try:
        from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
    except Exception as e:
        raise RuntimeError("sentinelsat not installed. pip install sentinelsat") from e

    user = os.environ.get("COPERNICUS_USER")
    pwd = os.environ.get("COPERNICUS_PASS")
    if not user or not pwd:
        raise RuntimeError("COPERNICUS_USER and COPERNICUS_PASS must be set to use sentinelsat")

    api = SentinelAPI(user, pwd, 'https://apihub.copernicus.eu/apihub')
    logger.info("Querying Copernicus for products...")
    products = api.query(footprint_wkt,
                         date=(start_date, end_date),
                         platformname='Sentinel-1',
                         producttype=product_type,
                         processinglevel='GRD')
    if len(products) == 0:
        logger.warning("No products found for the provided footprint/date range.")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for prod_id, prod_info in products.items():
        title = prod_info.get('title') or prod_id
        logger.info(f"Downloading {title} ...")
        try:
            local_path = api.download(prod_id, directory_path=str(out_dir))['path']
            downloaded.append(local_path)
        except Exception as e:
            logger.exception(f"Failed to download {title}: {e}")
    return downloaded

def ensure_local_scenes(out_dir="data/raw"):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    # No-op — user should place Sentinel-1 SAFE/.zip/.tif files here if not using sentinelsat
    files = list(p.glob("*"))
    return [str(x) for x in files]

def fetch_sentinel1(footprint_wkt=None, start_date=None, end_date=None, out_dir="data/raw", prefer_sent_api=False):
    """
    Top-level convenience function.
    - If credentials present and prefer_sent_api True, attempt sentinelsat.
    - Otherwise, return local files found under out_dir.
    """
    if prefer_sent_api and os.environ.get("COPERNICUS_USER") and os.environ.get("COPERNICUS_PASS") and footprint_wkt and start_date and end_date:
        try:
            return download_with_sentinelsat(footprint_wkt, start_date, end_date, out_dir)
        except Exception as e:
            logger.warning("sentinelsat download failed, falling back to local/raw folder: %s", e)
    # fallback
    return ensure_local_scenes(out_dir)
