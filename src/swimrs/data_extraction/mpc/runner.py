"""Scene-major worker for process-pool extraction.

The driver writes three files into the records dir before launching the
pool: fields.parquet (EPSG:4326 geometries with _row and partition
columns), targets.json (per-year field ids + mask types), and config.json
(feature id, masks dir, mask convention). Each worker loads them once via
the pool initializer; each task then processes ONE scene: find intersecting
target fields with a spatial index, read only that window, reduce every
field in one exactextract pass per mask, and write an idempotent per-scene
parquet. A scene whose parquet exists is never resubmitted by the driver.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape

from swimrs.data_extraction.mpc import landsat, sentinel, stac, zonal
from swimrs.data_extraction.mpc.grid import NODATA
from swimrs.data_extraction.mpc.masks import IrrMapperMasks

BOUNDS_PAD_M = 60.0

_W = {}


def records_path(records_dir, instrument, year, scene_id):
    return Path(records_dir) / instrument / str(year) / f"{scene_id}.parquet"


def write_worker_inputs(records_dir, fields_gdf, per_year, config):
    """Persist the shared worker inputs (driver-side, before the pool)."""
    records_dir = Path(records_dir)
    records_dir.mkdir(parents=True, exist_ok=True)
    fields_gdf.to_parquet(records_dir / "fields.parquet")
    targets = {
        str(year): {"masks": sorted(entry["masks"]), "fields": sorted(entry["fields"])}
        for year, entry in per_year.items()
    }
    (records_dir / "targets.json").write_text(json.dumps(targets))
    (records_dir / "config.json").write_text(json.dumps(config))


def init_worker(records_dir):
    records_dir = Path(records_dir)
    config = json.loads((records_dir / "config.json").read_text())
    fields = gpd.read_parquet(records_dir / "fields.parquet")
    _W.update(
        records_dir=records_dir,
        config=config,
        fields=fields,
        sindex=fields.sindex,
        targets=json.loads((records_dir / "targets.json").read_text()),
        masks=None,
        utm_cache={},
    )
    mask_types = {m for entry in _W["targets"].values() for m in entry["masks"]}
    if mask_types - {"no_mask"}:
        _W["masks"] = IrrMapperMasks(
            config["masks_dir"],
            irrigated_value=config.get("irrigated_value", 0),
            min_years=config.get("min_years", 5),
            irr_max_year=config.get("irr_max_year", 2023),
        )


def process_scene(task):
    """Extract one scene. task = {item, instrument, year}. Returns a status tuple."""
    item = task["item"]
    instrument = task["instrument"]
    year = task["year"]
    scene_id = (
        stac.landsat_scene_id(item) if instrument == "landsat" else stac.sentinel_scene_id(item)
    )
    out = records_path(_W["records_dir"], instrument, year, scene_id)
    if out.exists():
        return scene_id, 0, "skip"

    entry = _W["targets"][str(year)]
    target_ids = set(entry["fields"])
    feature_id = _W["config"]["feature_id"]
    fields = _W["fields"]

    footprint = shape(item["geometry"])
    idx = _W["sindex"].query(footprint, predicate="intersects")
    sub = fields.iloc[idx]
    sub = sub[sub[feature_id].isin(target_ids)]
    if sub.empty:
        _write_atomic(out, _empty_records(feature_id))
        return scene_id, 0, "empty"

    epsg = stac.item_epsg(item)
    sub_utm = sub.to_crs(epsg)
    xmin, ymin, xmax, ymax = sub_utm.total_bounds
    bounds = (xmin - BOUNDS_PAD_M, ymin - BOUNDS_PAD_M, xmax + BOUNDS_PAD_M, ymax + BOUNDS_PAD_M)

    module = landsat if instrument == "landsat" else sentinel
    ndvi, grid = module.scene_ndvi(item, bounds)

    frames = []
    for mask_type in entry["masks"]:
        include = (
            np.ones(grid.shape, dtype=bool)
            if mask_type == "no_mask"
            else _W["masks"].mask_for(mask_type, year, grid)
        )
        values = np.where(include, ndvi, NODATA).astype(np.float32)
        df = zonal.field_means(
            values,
            grid,
            sub_utm[[feature_id, "_row", "geometry"]],
            feature_id,
            include_cols=[feature_id, "_row"],
        )
        df["mask"] = mask_type
        df["scene_id"] = scene_id
        frames.append(df)

    _write_atomic(out, pd.concat(frames, ignore_index=True))
    return scene_id, len(sub), "ok"


def _empty_records(feature_id):
    return pd.DataFrame(
        {feature_id: [], "_row": [], "mean": [], "count": [], "mask": [], "scene_id": []}
    )


def _write_atomic(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    tmp.rename(path)
