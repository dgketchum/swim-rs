"""All Earth Engine / GridMET data extraction for Example 5 (Flux Ensemble).

Steps (run all by default, or select with --steps). All EE steps use the
synchronous .getInfo() approach (no bucket export / gsutil sync):
  snodas     - SNODAS SWE via year-stack reduceRegions (local CSV)
  properties - CDL, irrigation, SSURGO, landcover via reduceRegions (local CSVs)
  ndvi       - Landsat + Sentinel NDVI via per-site stack+reduceRegion (local CSVs)
  gridmet    - GridMET meteorology download (local; not EE)
  etf_v21    - ETf from OpenET v2.1 collections via .getInfo() (local CSVs)
  refet      - OpenET bias-corrected ETo/ETr via .getInfo() (local CSVs)

Legacy bucket-export steps (kept for speed/value comparison; never in "all"):
  snodas_bucket, properties_bucket, ndvi_bucket

ETf extraction uses the per-site stack+reduceRegion approach (~0.7s per
site-year for fraction models, ~2s for ET models with OpenET ETo join).
ET-denominated models (ptjpl, geesebal, disalexi) are divided by the
OpenET corrected ETo (not raw GridMET), matching what OpenET uses
internally. Extraction starts from each model's earliest available date.

getInfo outputs (never written to the legacy extracts/ trees):
  ETf:        data/etf_v21_openet_eto/{model}_etf_no_mask.csv
  RefET:      data/openet_refet/openet_{eto,etr}.csv
  NDVI:       data/remote_sensing/{instrument}/getinfo/ndvi/{mask}/ndvi_{site}_{mask}.csv
  SNODAS:     data/snow/snodas/getinfo/swe.csv (meters; ingest converts to mm)
  Properties: data/properties/getinfo/{project}_{cdl,irr,ssurgo,landcover}.csv
ETf/RefET are read by ``build_container.py``; NDVI/SNODAS/properties are read
by ``container_prep.py --getinfo``.
"""

import json
import os
import time
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from swimrs.data_extraction.ee.ee_utils import is_authorized, landsat_masked, sentinel2_masked
from swimrs.swim.config import ProjectConfig

# Official OpenET v2.1 source collections
OPENET_V21 = {
    "ssebop": "projects/openet/assets/ssebop/conus/gridmet/landsat/v2_1",
    "sims": "projects/openet/assets/sims/conus/gridmet/landsat/v2_1",
    "eemetric": "projects/openet/assets/eemetric/conus/gridmet/landsat/v2_1",
    "geesebal": "projects/openet/assets/geesebal/conus/gridmet/landsat/v2_1",
    "ptjpl": "projects/openet/assets/ptjpl/conus/nldas2/landsat/v2_1",
    "disalexi": "projects/openet/assets/disalexi/conus/cfsr/landsat/v2_1",
}

# Models that store raw ET (mm × 1000) and need division by reference ET
ET_MODELS = {"geesebal", "ptjpl", "disalexi"}

# Models that store et_fraction (integer × 10000)
FRACTION_MODELS = {"ssebop", "sims", "eemetric"}

# Earliest available data per model (from EE collection metadata)
MODEL_START_YEAR = {
    "ssebop": 1999,
    "sims": 1999,
    "eemetric": 1999,
    "geesebal": 1999,
    "ptjpl": 1999,
    "disalexi": 2001,
}

OPENET_REFET = "projects/openet/assets/reference_et/conus/gridmet/daily/v1"
REFET_VARIABLES = ("eto", "etr")
REFET_START_YEAR = 1999

SNODAS = "projects/earthengine-legacy/assets/projects/climate-engine/snodas/daily"
SNODAS_START_YEAR = 2004

SENTINEL_START_YEAR = 2017

MAX_VALID_ETF = 2.0
MAX_RETRIES = 3


def _load_config() -> ProjectConfig:
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "5_Flux_Ensemble.toml"

    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent))
    return cfg


def _sentinel_dir(cfg: ProjectConfig) -> str:
    """Sentinel data dir: TOML ``paths.sentinel`` if set, else derived from landsat dir."""
    if cfg.sentinel_dir:
        return cfg.sentinel_dir
    return cfg.landsat_dir.replace("landsat", "sentinel")


def extract_snodas_bucket(cfg: ProjectConfig) -> None:
    """Legacy SNODAS SWE export to bucket (ee.batch tasks + gsutil sync)."""
    is_authorized()
    from swimrs.data_extraction.ee.snodas_export import sample_snodas_swe

    sample_snodas_swe(
        feature_coll=cfg.fields_shapefile,
        bucket=cfg.ee_bucket,
        dest="bucket",
        debug=False,
        check_dir=None,
        feature_id=cfg.feature_id_col,
        file_prefix=cfg.project_name,
    )


def extract_properties_bucket(cfg: ProjectConfig) -> None:
    """Legacy properties export to bucket (ee.batch tasks + gsutil sync)."""
    is_authorized()
    from swimrs.data_extraction.ee.ee_props import (
        get_cdl,
        get_irrigation,
        get_landcover,
        get_ssurgo,
    )

    project = cfg.project_name
    get_cdl(
        cfg.fields_shapefile,
        f"{project}_cdl",
        selector=cfg.feature_id_col,
        dest="bucket",
        bucket=cfg.ee_bucket,
        file_prefix=project,
    )
    get_irrigation(
        cfg.fields_shapefile,
        f"{project}_irr",
        debug=True,
        selector=cfg.feature_id_col,
        lanid=True,
        dest="bucket",
        bucket=cfg.ee_bucket,
        file_prefix=project,
    )
    get_ssurgo(
        cfg.fields_shapefile,
        f"{project}_ssurgo",
        debug=False,
        selector=cfg.feature_id_col,
        dest="bucket",
        bucket=cfg.ee_bucket,
        file_prefix=project,
    )
    get_landcover(
        cfg.fields_shapefile,
        f"{project}_landcover",
        debug=False,
        selector=cfg.feature_id_col,
        out_fmt="CSV",
        dest="bucket",
        bucket=cfg.ee_bucket,
        file_prefix=project,
    )


def extract_ndvi_bucket(cfg: ProjectConfig, sites=None, get_sentinel: bool = True) -> None:
    """Legacy NDVI export to bucket (ee.batch tasks + gsutil sync)."""
    is_authorized()
    from swimrs.data_extraction.ee.ndvi_export import sparse_sample_ndvi

    for mask in ["irr", "inv_irr", "no_mask"]:
        dst = os.path.join(cfg.landsat_dir, "extracts", "ndvi", mask)
        sparse_sample_ndvi(
            cfg.fields_shapefile,
            bucket=cfg.ee_bucket,
            dest="bucket",
            debug=False,
            mask_type=mask,
            check_dir=dst,
            start_yr=cfg.start_dt.year,
            end_yr=cfg.end_dt.year,
            feature_id=cfg.feature_id_col,
            satellite="landsat",
            state_col=cfg.state_col,
            select=sites,
            file_prefix=cfg.project_name,
        )

        if get_sentinel:
            dst = os.path.join(_sentinel_dir(cfg), "extracts", "ndvi", mask)
            sparse_sample_ndvi(
                cfg.fields_shapefile,
                bucket=cfg.ee_bucket,
                dest="bucket",
                debug=False,
                mask_type=mask,
                check_dir=dst,
                start_yr=max(2017, cfg.start_dt.year),
                end_yr=cfg.end_dt.year,
                feature_id=cfg.feature_id_col,
                satellite="sentinel",
                state_col=cfg.state_col,
                select=sites,
                file_prefix=cfg.project_name,
            )


def extract_gridmet(cfg: ProjectConfig, sites=None) -> None:
    from swimrs.data_extraction.gridmet.gridmet import (
        assign_gridmet_ids,
        download_gridmet,
        sample_gridmet_corrections,
    )

    nldas_needed = getattr(cfg, "runoff_process", "cn") == "ier"
    join_path = cfg.gridmet_mapping_shp
    factors_path = cfg.gridmet_factors

    # Reuse the existing GFID mapping; build only if missing (mirrors
    # container_prep.build_gridmet_mapping). GFIDs are keyed to the parquet
    # filenames in the met store, and the on-disk gridmet_centroids.shp does
    # not cover the field extent — regenerating would orphan the met store.
    if not os.path.exists(join_path):
        assign_gridmet_ids(
            fields=cfg.fields_shapefile,
            fields_join=join_path,
            gridmet_points=cfg.gridmet_centroids,
            field_select=sites,
            feature_id=cfg.feature_id_col,
            gridmet_id_col=cfg.gridmet_id_col,
        )
    else:
        print(f"GridMET mapping already exists: {join_path}")

    if cfg.correction_tifs:
        sample_gridmet_corrections(
            fields_join=join_path,
            gridmet_ras=cfg.correction_tifs,
            factors_js=factors_path,
            gridmet_id_col=cfg.gridmet_id_col,
        )

    download_gridmet(
        join_path,
        factors_path,
        cfg.met_dir,
        start=str(cfg.start_dt.date()),
        end=str(cfg.end_dt.date()),
        overwrite=False,
        append=True,
        use_nldas=nldas_needed,
        feature_id=cfg.gridmet_mapping_index_col,
        target_fields=sites,
        gridmet_id_col=cfg.gridmet_id_col,
    )


def _load_polygons(shapefile, feature_id="site_id", select=None):
    """Load field polygons and return {fid: ee.Geometry} + ordered fid list."""
    gdf = gpd.read_file(shapefile, engine="fiona").to_crs(4326)
    geometries = {}
    fids = []
    for _, row in gdf.iterrows():
        fid = row[feature_id]
        if select is not None and str(fid) not in {str(s) for s in select}:
            continue
        fids.append(fid)
        geometries[fid] = ee.Geometry(row.geometry.__geo_interface__)
    return geometries, fids


def _extract_fraction_model(coll_path, geometry, year):
    """Extract ETf for a fraction model (ssebop/sims/eemetric) at one site-year.

    Returns dict {YYYYMMDD: etf_value}.
    """
    coll = (
        ee.ImageCollection(coll_path)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(geometry)
    )

    def _tag(img):
        d = img.date().format("yyyyMMdd")
        return img.select("et_fraction").divide(10000).rename(ee.String("etf_").cat(d))

    stacked = coll.map(_tag).toBands()
    result = stacked.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=30, maxPixels=1e11
    ).getInfo()

    if not result:
        return {}

    values = {}
    for k, v in result.items():
        if v is None:
            continue
        parts = k.split("_")
        date_str = parts[-1]
        values[date_str] = v
    return values


def _extract_et_model(coll_path, geometry, year):
    """Extract ETf for an ET model (ptjpl/geesebal/disalexi) at one site-year.

    Divides raw ET by same-day OpenET reference ETo (server-side join).
    Returns dict {YYYYMMDD: etf_value}.
    """
    openet_ref = ee.ImageCollection(OPENET_REFET)
    coll = (
        ee.ImageCollection(coll_path)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(geometry)
    )

    def _compute_etf(img):
        d = img.date()
        eto = openet_ref.filterDate(d, d.advance(1, "day")).first().select("eto")
        et_mm = img.select("et").divide(1000)
        return et_mm.divide(eto).rename(ee.String("etf_").cat(d.format("yyyyMMdd")))

    stacked = coll.map(_compute_etf).toBands()
    result = stacked.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=30, maxPixels=1e11
    ).getInfo()

    if not result:
        return {}

    values = {}
    for k, v in result.items():
        if v is None:
            continue
        parts = k.split("_")
        date_str = parts[-1]
        values[date_str] = v
    return values


def _with_retries(fn, *args, label=""):
    """Call fn(*args) with exponential-backoff retries; {} after final failure."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                print(f"    retry {attempt + 1}/{MAX_RETRIES} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"    {label} FAILED after {MAX_RETRIES} attempts: {e}")
                return {}


def _extract_site_year(model, coll_path, geometry, year):
    """Extract ETf for one model at one site-year, with retries."""
    extract_fn = _extract_et_model if model in ET_MODELS else _extract_fraction_model
    return _with_retries(extract_fn, coll_path, geometry, year, label=f"{model} {year}")


def _resolve_duplicates(values):
    """Resolve duplicate dates (overlapping path/rows) by taking max."""
    if not values:
        return values
    series = pd.Series(values, dtype=float)
    if series.index.duplicated().any():
        series = series.groupby(series.index).max()
    return series.to_dict()


def _merge_checkpoint(df, checkpoint_path):
    """Merge newly extracted rows into an existing checkpoint CSV.

    Preserves rows for sites that were not re-extracted (e.g. when running
    with a --sites subset), so a partial run never truncates the checkpoint.
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        old = pd.read_csv(checkpoint_path, index_col=0)
        old.index = old.index.astype(str)
        keep = old.loc[~old.index.isin(df.index.astype(str))]
        if not keep.empty:
            df = pd.concat([df, keep])
            df = df[sorted(df.columns)]
    return df


def extract_model(model, geometries, fids, start_yr, end_yr, checkpoint_path=None):
    """Extract ETf for one model across all sites and years.

    Returns DataFrame with rows=sites, columns=YYYYMMDD.
    """
    coll_path = OPENET_V21[model]
    model_start = max(MODEL_START_YEAR[model], start_yr)

    # Load checkpoint if exists
    existing = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        df_ckpt = pd.read_csv(checkpoint_path, index_col=0)
        for fid in df_ckpt.index:
            existing[fid] = df_ckpt.loc[fid].dropna().to_dict()
        print(f"  Loaded checkpoint: {len(existing)} sites with data")

    years = list(range(model_start, end_yr + 1))
    total_ops = len(fids) * len(years)

    print(f"\n{'=' * 60}")
    print(f"{model}: {len(fids)} sites x {len(years)} years = {total_ops} site-years")
    print(f"  Collection: {coll_path}")
    print(f"  {'ET / OpenET_ETo' if model in ET_MODELS else 'et_fraction / 10000'}")
    print(f"{'=' * 60}")

    all_data = {}
    t_start = time.time()

    for fid in tqdm(fids, desc=model):
        site_values = dict(existing.get(fid, {}))
        geometry = geometries[fid]

        for year in years:
            values = _extract_site_year(model, coll_path, geometry, year)
            values = _resolve_duplicates(values)
            site_values.update(values)

        # Filter invalid ETf
        site_values = {
            k: v for k, v in site_values.items() if v is not None and 0 < v <= MAX_VALID_ETF
        }
        all_data[fid] = site_values

    elapsed = time.time() - t_start
    print(f"  Completed in {elapsed:.0f}s ({elapsed / total_ops:.2f}s per site-year)")

    # Build DataFrame
    df = pd.DataFrame.from_dict(all_data, orient="index")
    df.index.name = "site_id"
    df = df.reindex(fids)
    df = df[sorted(df.columns)]
    df = _merge_checkpoint(df, checkpoint_path)

    total_values = df.notna().sum().sum()
    per_site = df.notna().sum(axis=1)
    print(f"  Total values: {total_values:,}")
    print(
        f"  Per-site: min={per_site.min()}, max={per_site.max()}, "
        f"mean={per_site.mean():.0f}, median={per_site.median():.0f}"
    )

    return df


def extract_etf_v21(cfg: ProjectConfig, sites=None, models=None) -> None:
    """Extract ETf for all OpenET v2.1 models via .getInfo().

    Writes one CSV per model to data/etf_v21_openet_eto/{model}_etf_no_mask.csv
    (rows=sites, columns=YYYYMMDD), compatible with ``container.ingest.etf()``
    via the ``csv_dir`` parameter.
    """
    is_authorized()

    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "data" / "etf_v21_openet_eto"
    os.makedirs(output_dir, exist_ok=True)

    geometries, fids = _load_polygons(
        cfg.fields_shapefile, feature_id=cfg.feature_id_col, select=sites
    )

    if models is None:
        models = list(OPENET_V21.keys())

    for model in models:
        if model not in OPENET_V21:
            print(f"Unknown model: {model}, skipping")
            continue

        start_yr = MODEL_START_YEAR[model]
        checkpoint = str(output_dir / f"{model}_etf_no_mask.csv")

        df = extract_model(
            model,
            geometries,
            fids,
            start_yr=start_yr,
            end_yr=cfg.end_dt.year,
            checkpoint_path=checkpoint,
        )

        df.to_csv(checkpoint)
        print(f"  Saved: {checkpoint}")

        summary = {
            "model": model,
            "n_sites": len(fids),
            "n_dates": df.shape[1],
            "n_values": int(df.notna().sum().sum()),
            "date_range": [df.columns.min(), df.columns.max()] if df.shape[1] > 0 else [],
            "start_yr": start_yr,
            "end_yr": cfg.end_dt.year,
            "et_denominated": model in ET_MODELS,
            "eto_source": OPENET_REFET if model in ET_MODELS else "N/A (et_fraction native)",
        }
        summary_path = str(output_dir / f"{model}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


def _build_centroid_fc(shapefile, feature_id="site_id", select=None):
    """Build an EE FeatureCollection of field centroids (4km grid is fine)."""
    gdf = gpd.read_file(shapefile, engine="fiona").to_crs(4326)
    if select is not None:
        gdf = gdf[gdf[feature_id].astype(str).isin({str(s) for s in select})]
    features = []
    for _, row in gdf.iterrows():
        c = row.geometry.centroid
        features.append(ee.Feature(ee.Geometry.Point(c.x, c.y), {feature_id: row[feature_id]}))
    return ee.FeatureCollection(features), list(gdf[feature_id])


def _build_polygon_fc(shapefile, feature_id="site_id", select=None):
    """Build an EE FeatureCollection of field polygons."""
    gdf = gpd.read_file(shapefile, engine="fiona").to_crs(4326)
    if select is not None:
        gdf = gdf[gdf[feature_id].astype(str).isin({str(s) for s in select})]
    features = [
        ee.Feature(ee.Geometry(row.geometry.__geo_interface__), {feature_id: row[feature_id]})
        for _, row in gdf.iterrows()
    ]
    return ee.FeatureCollection(features), list(gdf[feature_id])


def _extract_year(fc, fids, year, variable, feature_id="site_id"):
    """Extract one year of daily reference ET for all sites. Returns DataFrame."""
    coll = (
        ee.ImageCollection(OPENET_REFET)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select(variable)
    )

    def _tag(img):
        d = img.date().format("yyyyMMdd")
        return img.rename(ee.String(variable).cat("_").cat(d))

    stacked = coll.map(_tag).toBands()
    result = stacked.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=4000).getInfo()

    records = {}
    for feat in result["features"]:
        props = feat["properties"]
        fid = props.pop(feature_id)
        row = {}
        for k, v in props.items():
            # band names: "0_eto_20200101" -> extract date part
            parts = k.split("_")
            date_str = parts[-1]  # YYYYMMDD
            row[date_str] = v
        records[fid] = row

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "site_id"
    df = df.reindex(fids)
    df = df[sorted(df.columns)]
    return df


def extract_openet_refet(cfg: ProjectConfig, sites=None) -> None:
    """Extract daily OpenET bias-corrected reference ET (ETo and ETr) via .getInfo().

    This is the same ETo/ETr used internally by all OpenET v2.1 models.
    Stacks one year of daily images into a single multi-band image, then
    reduceRegions over all site centroids at once (~5s per year). Writes
    data/openet_refet/openet_{eto,etr}.csv (rows=sites, columns=YYYYMMDD).
    """
    is_authorized()

    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "data" / "openet_refet"
    os.makedirs(output_dir, exist_ok=True)

    fc, fids = _build_centroid_fc(cfg.fields_shapefile, feature_id=cfg.feature_id_col, select=sites)
    years = list(range(max(REFET_START_YEAR, cfg.start_dt.year), cfg.end_dt.year + 1))

    for variable in REFET_VARIABLES:
        print(f"\n{'=' * 60}")
        print(f"Extracting OpenET {variable}: {len(fids)} sites, {years[0]}-{years[-1]}")
        print(f"{'=' * 60}")

        frames = []
        for year in years:
            t0 = time.time()
            try:
                df = _extract_year(fc, fids, year, variable, feature_id=cfg.feature_id_col)
                frames.append(df)
                elapsed = time.time() - t0
                print(
                    f"  {year}: {df.shape[1]} days, {df.notna().sum().sum():,} values ({elapsed:.1f}s)"
                )
            except Exception as e:
                print(f"  {year}: FAILED — {e}")
                continue

        if not frames:
            print(f"No data extracted for {variable}")
            continue

        combined = pd.concat(frames, axis=1)
        combined = combined[sorted(combined.columns)]
        out_path = str(output_dir / f"openet_{variable}.csv")
        combined.to_csv(out_path)
        print(f"\nSaved: {out_path}")
        print(f"  Shape: {combined.shape} ({combined.notna().sum().sum():,} values)")


def _extract_ndvi_site_year(geometry, year, instrument):
    """Extract NDVI for one site-year. Returns dict {scene_id: ndvi}.

    Uses the same masked/harmonized collections as the bucket exporter
    (cloud mask + SBAF harmonization to OLI), NDVI from NIR_H/RED_H.
    """
    if instrument == "landsat":
        coll = landsat_masked(year, geometry, harmonize=True)
    elif instrument == "sentinel":
        coll = sentinel2_masked(year, geometry, harmonize=True)
    else:
        raise ValueError("instrument must be one of {'landsat','sentinel'}")

    def _tag(img):
        # shorten merged-collection index to scene id, e.g. LC08_035026_20200115
        short = ee.String(img.get("system:index")).split("_").slice(-3).join("_")
        return img.normalizedDifference(["NIR_H", "RED_H"]).rename(short)

    stacked = coll.map(_tag).toBands()
    result = stacked.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=30, maxPixels=1e11
    ).getInfo()

    if not result:
        return {}

    values = {}
    for k, v in result.items():
        if v is None:
            continue
        # toBands key: "{image_index}_{scene_id}" -> recover scene id
        scene = "_".join(k.split("_")[-3:])
        values[scene] = v
    return values


def extract_ndvi(
    cfg: ProjectConfig, sites=None, masks=("no_mask",), get_sentinel: bool = True
) -> None:
    """Extract Landsat (and Sentinel-2) NDVI via .getInfo().

    Per site-year stack+reduceRegion (mean, 30 m), matching the bucket
    exporter's collections, cloud masking, and SBAF harmonization. Writes one
    CSV per site to {instrument_dir}/getinfo/ndvi/{mask}/ndvi_{site}_{mask}.csv
    with rows=site, columns=scene IDs — the layout ``container.ingest.ndvi()``
    parses. Existing per-site CSVs are merged (re-extracted values win).

    Only no_mask is implemented; for irr/inv_irr use the ndvi_bucket step
    (IrrMapper/LANID masking in swimrs.data_extraction.ee.ndvi_export).
    """
    unsupported = [m for m in masks if m != "no_mask"]
    if unsupported:
        raise NotImplementedError(
            f"getInfo NDVI supports no_mask only (got {unsupported}); "
            "use the ndvi_bucket step for IrrMapper/LANID-masked extraction"
        )

    is_authorized()

    geometries, fids = _load_polygons(
        cfg.fields_shapefile, feature_id=cfg.feature_id_col, select=sites
    )

    instruments = ["landsat"] + (["sentinel"] if get_sentinel else [])

    for instrument in instruments:
        base_dir = cfg.landsat_dir if instrument == "landsat" else _sentinel_dir(cfg)
        start_yr = (
            cfg.start_dt.year
            if instrument == "landsat"
            else max(SENTINEL_START_YEAR, cfg.start_dt.year)
        )
        years = list(range(start_yr, cfg.end_dt.year + 1))

        for mask in masks:
            out_dir = os.path.join(base_dir, "getinfo", "ndvi", mask)
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n{'=' * 60}")
            print(f"{instrument} NDVI ({mask}): {len(fids)} sites x {len(years)} years")
            print(f"{'=' * 60}")

            t_start = time.time()
            n_values = 0

            for fid in tqdm(fids, desc=f"{instrument} ndvi {mask}"):
                out_path = os.path.join(out_dir, f"ndvi_{fid}_{mask}.csv")

                site_values = {}
                if os.path.exists(out_path):
                    existing = pd.read_csv(out_path, index_col=0)
                    if len(existing) > 0:
                        site_values = existing.iloc[0].dropna().to_dict()

                for year in years:
                    values = _with_retries(
                        _extract_ndvi_site_year,
                        geometries[fid],
                        year,
                        instrument,
                        label=f"{instrument} ndvi {fid} {year}",
                    )
                    site_values.update(values)

                df = pd.DataFrame([site_values], index=pd.Index([fid], name=cfg.feature_id_col))
                df = df[sorted(df.columns)]
                df.to_csv(out_path)
                n_values += int(df.notna().sum().sum())

            elapsed = time.time() - t_start
            total_ops = len(fids) * len(years)
            print(
                f"  {n_values:,} values in {elapsed:.0f}s "
                f"({elapsed / total_ops:.2f}s per site-year)"
            )


def _extract_snodas_year(fc, fids, year, feature_id="site_id"):
    """Extract one year of daily SNODAS SWE for all sites. Returns DataFrame (meters)."""
    coll = ee.ImageCollection(SNODAS).filterDate(f"{year}-01-01", f"{year + 1}-01-01").select("SWE")

    def _tag(img):
        d = img.date().format("yyyyMMdd")
        return img.rename(ee.String("swe_").cat(d))

    stacked = coll.map(_tag).toBands()
    result = stacked.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30).getInfo()

    records = {}
    for feat in result["features"]:
        props = feat["properties"]
        fid = props.pop(feature_id)
        row = {}
        for k, v in props.items():
            date_str = k.split("_")[-1]  # "{idx}_swe_{YYYYMMDD}"
            row[date_str] = v
        records[fid] = row

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = feature_id
    df = df.reindex(fids)
    df = df[sorted(df.columns)]
    return df


def _snodas_year_complete(existing, fids, year):
    """True if the checkpoint has data for every requested site in this year."""
    if existing is None:
        return False
    cols = [c for c in existing.columns if c.startswith(str(year))]
    if not cols:
        return False
    fids = [str(f) for f in fids]
    if not set(fids).issubset(set(existing.index)):
        return False
    return bool(existing.loc[fids, cols].notna().any(axis=1).all())


def extract_snodas(cfg: ProjectConfig, sites=None) -> None:
    """Extract daily SNODAS SWE via .getInfo() (year-stack reduceRegions).

    One reduceRegions call per year over all site polygons (mean, 30 m,
    matching the bucket exporter). Writes data/snow/snodas/getinfo/swe.csv
    with rows=sites, columns=YYYYMMDD, values in meters (``ingest.snodas``
    converts to mm). Years already complete for all requested sites are
    skipped; partial runs merge into the existing CSV.
    """
    is_authorized()

    out_dir = os.path.join(cfg.data_dir, "snow", "snodas", "getinfo")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "swe.csv")

    fc, fids = _build_polygon_fc(cfg.fields_shapefile, feature_id=cfg.feature_id_col, select=sites)

    existing = None
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, index_col=0)
        existing.index = existing.index.astype(str)
        print(f"  Loaded checkpoint: {existing.shape}")

    years = list(range(max(SNODAS_START_YEAR, cfg.start_dt.year), cfg.end_dt.year + 1))

    print(f"\n{'=' * 60}")
    print(f"SNODAS SWE: {len(fids)} sites, {years[0]}-{years[-1]}")
    print(f"{'=' * 60}")

    frames = []
    for year in years:
        if _snodas_year_complete(existing, fids, year):
            continue
        t0 = time.time()
        try:
            df = _extract_snodas_year(fc, fids, year, feature_id=cfg.feature_id_col)
        except Exception as e:
            print(f"  {year}: FAILED — {e}")
            continue
        frames.append(df)
        print(
            f"  {year}: {df.shape[1]} days, {df.notna().sum().sum():,} values "
            f"({time.time() - t0:.1f}s)"
        )

    if not frames:
        print("  Nothing to extract (checkpoint complete)")
        return

    new = pd.concat(frames, axis=1)
    new.index = new.index.astype(str)
    combined = new.combine_first(existing) if existing is not None else new
    combined = combined[sorted(combined.columns)]
    combined.index.name = cfg.feature_id_col
    combined.to_csv(out_path)
    print(f"\nSaved: {out_path}")
    print(f"  Shape: {combined.shape} ({combined.notna().sum().sum():,} values)")


def extract_properties(cfg: ProjectConfig) -> None:
    """Extract CDL, irrigation, SSURGO, and landcover tables via .getInfo().

    Same server-side images and reducers as the bucket exporters in
    ``ee_props.py``, written locally (dest="local") to data/properties/getinfo/.
    Always extracts all features in the shapefile (one-off small tables).
    """
    is_authorized()
    from swimrs.data_extraction.ee.ee_props import (
        get_cdl,
        get_irrigation,
        get_landcover,
        get_ssurgo,
    )

    project = cfg.project_name
    out_dir = os.path.join(cfg.data_dir, "properties", "getinfo")

    get_cdl(
        cfg.fields_shapefile,
        f"{project}_cdl",
        selector=cfg.feature_id_col,
        dest="local",
        out_dir=out_dir,
    )
    get_irrigation(
        cfg.fields_shapefile,
        f"{project}_irr",
        selector=cfg.feature_id_col,
        lanid=True,
        dest="local",
        out_dir=out_dir,
    )
    get_ssurgo(
        cfg.fields_shapefile,
        f"{project}_ssurgo",
        selector=cfg.feature_id_col,
        dest="local",
        out_dir=out_dir,
    )
    get_landcover(
        cfg.fields_shapefile,
        f"{project}_landcover",
        selector=cfg.feature_id_col,
        dest="local",
        out_dir=out_dir,
    )


ALL_STEPS = ["snodas", "properties", "ndvi", "gridmet", "etf_v21", "refet"]
BUCKET_STEPS = ["snodas_bucket", "properties_bucket", "ndvi_bucket"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help=f"Comma-separated steps to run: {', '.join(ALL_STEPS + BUCKET_STEPS)} "
        f"(default: all = {', '.join(ALL_STEPS)})",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated OpenET v2.1 models for the etf_v21 step (default: all 6)",
    )
    parser.add_argument(
        "--sites",
        type=str,
        default=None,
        help="Comma-separated site IDs (default: all sites in the fields shapefile)",
    )
    args = parser.parse_args()

    valid_steps = ALL_STEPS + BUCKET_STEPS
    steps = ALL_STEPS if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    unknown = [s for s in steps if s not in valid_steps]
    if unknown:
        raise SystemExit(f"Unknown steps: {unknown}. Choose from: {valid_steps}")

    etf_models = [m.strip() for m in args.models.split(",")] if args.models else None

    config = _load_config()
    gdf = gpd.read_file(config.fields_shapefile, engine="fiona")
    if config.feature_id_col not in gdf.columns:
        raise ValueError(
            f"Feature ID column {config.feature_id_col!r} not found in {config.fields_shapefile}"
        )
    all_sites = gdf[config.feature_id_col].astype(str).to_list()

    if args.sites:
        select_sites = [s.strip() for s in args.sites.split(",")]
        missing = [s for s in select_sites if s not in all_sites]
        if missing:
            raise SystemExit(f"Sites not in shapefile: {missing}")
    else:
        select_sites = all_sites

    if "snodas" in steps:
        extract_snodas(config, select_sites)
    if "properties" in steps:
        extract_properties(config)
    if "ndvi" in steps:
        extract_ndvi(config, select_sites, get_sentinel=True)
    if "gridmet" in steps:
        extract_gridmet(config, select_sites)
    if "etf_v21" in steps:
        extract_etf_v21(config, select_sites, models=etf_models)
    if "refet" in steps:
        extract_openet_refet(config, select_sites)

    if "snodas_bucket" in steps:
        extract_snodas_bucket(config)
    if "properties_bucket" in steps:
        extract_properties_bucket(config)
    if "ndvi_bucket" in steps:
        extract_ndvi_bucket(config, select_sites, get_sentinel=True)
