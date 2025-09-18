from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from pathlib import Path
import logging


def fetch_sentinel1(user, password, geojson_path, out_dir, product_type="GRD", orbit="DESCENDING"):
    """
    Download Sentinel-1 scenes for AOI described by geojson.
    """
    api = SentinelAPI(user, password, "https://scihub.copernicus.eu/dhus")
    footprint = geojson_to_wkt(read_geojson(geojson_path))
    products = api.query(
        footprint,
        producttype=product_type,
        orbitdirection=orbit
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    api.download_all(products, directory_path=out_dir)
    logging.info(f"Downloaded {len(products)} Sentinel-1 products to {out_dir}")
