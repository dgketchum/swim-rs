"""STAC search against the Planetary Computer, and EE-style scene IDs.

The catalog is opened WITHOUT a signing modifier; asset hrefs are signed at
read time (grid.read_window) so SAS tokens stay fresh on multi-hour runs.
"""

from datetime import datetime

from pystac_client import Client

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
LANDSAT_COLLECTION = "landsat-c2-l2"
SENTINEL_COLLECTION = "sentinel-2-l2a"

# EE system:index sensor prefixes (uppercase — parity with the EE exports)
PLATFORM_TO_EE_SENSOR = {
    "landsat-4": "LT04",
    "landsat-5": "LT05",
    "landsat-7": "LE07",
    "landsat-8": "LC08",
    "landsat-9": "LC09",
}


def open_catalog():
    """STAC client without signing modifier (sign per-read instead)."""
    return Client.open(STAC_URL)


def search_landsat(catalog, bbox, year, tier="T1"):
    """All Landsat C2 L2 items over `bbox` (EPSG:4326) for one year.

    Matches the EE footing: Tier 1 only, all platforms, no cloud-cover
    metadata filter (per-pixel QA masking handles clouds downstream).
    Returns item dicts sorted by acquisition time.
    """
    query = {}
    if tier:
        query["landsat:collection_category"] = {"eq": tier}
    search = catalog.search(
        collections=[LANDSAT_COLLECTION],
        bbox=list(bbox),
        datetime=f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
        query=query or None,
    )
    items = [item.to_dict() for item in search.items()]
    items.sort(key=lambda d: d["properties"]["datetime"])
    return items


def search_sentinel2(catalog, bbox, year):
    """All Sentinel-2 L2A items over `bbox` (EPSG:4326) for one year."""
    search = catalog.search(
        collections=[SENTINEL_COLLECTION],
        bbox=list(bbox),
        datetime=f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
    )
    items = [item.to_dict() for item in search.items()]
    items.sort(key=lambda d: d["properties"]["datetime"])
    return items


def landsat_scene_id(item_dict):
    """EE system:index for a Landsat item: e.g. LC08_040030_20160401."""
    props = item_dict["properties"]
    sensor = PLATFORM_TO_EE_SENSOR[props["platform"].lower()]
    path = str(props["landsat:wrs_path"]).zfill(3)
    row = str(props["landsat:wrs_row"]).zfill(3)
    date = _dt(props["datetime"]).strftime("%Y%m%d")
    return f"{sensor}_{path}{row}_{date}"


def sentinel_scene_id(item_dict):
    """EE-style S2 index: {datatake_start}_{granule_sensing}_T{mgrs_tile}.

    The first token is taken from the product URI (datatake sensing start,
    identical to EE's); the middle token is the granule sensing time, which
    can differ from EE's by a few seconds — only the leading YYYYMMDD is
    parsed downstream.
    """
    props = item_dict["properties"]
    tile = props["s2:mgrs_tile"]
    sensing = _dt(props["datetime"]).strftime("%Y%m%dT%H%M%S")
    uri = props.get("s2:product_uri", "")
    datatake = uri.split("_")[2] if uri.count("_") >= 2 else sensing
    return f"{datatake}_{sensing}_T{tile}"


def item_epsg(item_dict):
    """Native EPSG of an item's rasters (proj:epsg or proj:code)."""
    props = item_dict["properties"]
    if "proj:epsg" in props:
        return int(props["proj:epsg"])
    code = props.get("proj:code", "")
    if code.upper().startswith("EPSG:"):
        return int(code.split(":")[1])
    raise ValueError(f"no projection code on item {item_dict.get('id')}")


def asset_href(item_dict, key):
    assets = item_dict["assets"]
    if key not in assets:
        raise KeyError(f"asset '{key}' not in item {item_dict.get('id')}: {sorted(assets)}")
    return assets[key]["href"]


def _dt(iso_string):
    return datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
