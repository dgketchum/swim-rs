"""Container coordinate helpers, the configured E1/E2/E3 scopes, and the E2 closure-pool gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from swimrs.swim import reproduce

from .cohorts import EXPECTED
from .manifest import BuildError, require_count, require_unique
from .paths import _CFG6_ROOT, E1_CONTAINER, E2_CONTAINER, E2_MAPPING_META, E3_CONTAINER, REPO

# --------------------------------------------------------------------------
# shared cohort helpers
# --------------------------------------------------------------------------

_CONTAINER_CRS_CACHE: dict[str, str] = {}


def container_coord_crs(container: Path) -> str:
    """CRS of the centroid coordinates stored in a ``.swim`` container.

    ``SwimContainer.create`` writes ``geometry/lon`` and ``geometry/lat`` as the
    raw centroid x / y of the source fields shapefile in that file's *native*
    CRS; it never reprojects (see ``src/swimrs/container/container.py``).  E2 and
    E3 are built from EPSG:4326 shapefiles so their stored values really are
    degrees, but E1 is built from an EPSG:5071 (CONUS Albers) shapefile, so its
    stored "lon"/"lat" are eastings / northings in metres.  Resolve the native
    CRS from the shapefile the container records in its provenance attributes.
    """
    import geopandas as gpd
    import zarr

    key = str(container)
    if key in _CONTAINER_CRS_CACHE:
        return _CONTAINER_CRS_CACHE[key]
    z = zarr.open(key, mode="r")
    shp = z.attrs.get("source_shapefile")
    shp = str(reproduce.rebase(shp, root=_CFG6_ROOT, repo=REPO)) if shp else shp
    if not shp or not Path(shp).exists():
        raise BuildError(
            f"container coordinate CRS: {container} records source_shapefile={shp!r}, "
            "which is missing; cannot establish whether geometry/lon,lat are degrees"
        )
    crs = gpd.read_file(shp, engine="fiona", rows=1).crs
    if crs is None:
        raise BuildError(f"container coordinate CRS: {shp} carries no CRS definition")
    out = crs.to_string()
    _CONTAINER_CRS_CACHE[key] = out
    return out


def container_lonlat(container: Path, lon, lat, label: str):
    """Container centroids as true WGS84 degrees, whatever the container stores.

    Guards the D3 failure mode: projected metres silently published under an
    EPSG:4326 tag.  Returns ``(lon_deg, lat_deg)`` numpy arrays.
    """
    import geopandas as gpd

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    crs = container_coord_crs(container)
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=crs)
    if pts.crs.to_epsg() != 4326:
        pts = pts.to_crs("EPSG:4326")
    lon_d = pts.x.to_numpy()
    lat_d = pts.y.to_numpy()
    if not (np.abs(lon_d) <= 180.0).all() or not (np.abs(lat_d) <= 90.0).all():
        raise BuildError(
            f"{label}: centroids are not valid WGS84 degrees after reprojection from {crs}"
        )
    return lon_d, lat_d


def require_layer_crs_consistent(gdf, label: str) -> None:
    """Fail if a layer's stored coordinates cannot live in its declared CRS.

    A geographic CRS bounds coordinates to +/-180 / +/-90; anything larger means
    the tag and the numbers disagree.
    """
    if gdf.crs is None:
        raise BuildError(f"{label}: layer has no CRS tag")
    minx, miny, maxx, maxy = (float(v) for v in gdf.total_bounds)
    tol = 1e-6  # Natural Earth clips land exactly at +/-180, with float slop
    if gdf.crs.is_geographic:
        if not (
            minx >= -180.0 - tol
            and maxx <= 180.0 + tol
            and miny >= -90.0 - tol
            and maxy <= 90.0 + tol
        ):
            raise BuildError(
                f"{label}: tagged {gdf.crs.to_string()} (geographic) but bounds are "
                f"({minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}) -- coordinates are "
                "not degrees"
            )
    elif abs(maxx) <= 180.0 and abs(maxy) <= 90.0:
        raise BuildError(
            f"{label}: tagged {gdf.crs.to_string()} (projected) but bounds look like "
            "degrees -- coordinates and CRS tag disagree"
        )


def e1_configured() -> pd.DataFrame:
    import zarr

    z = zarr.open(str(E1_CONTAINER), mode="r")
    lon, lat = container_lonlat(
        E1_CONTAINER,
        z["geometry/lon"][:],
        z["geometry/lat"][:],
        "E1 configured cohort",
    )
    df = pd.DataFrame(
        {
            "site_id": [str(x) for x in z["geometry/uid"][:]],
            "lat": lat,
            "lon": lon,
            "irr_source_fraction": np.asarray(z["properties/irrigation/irr"][:], dtype=float),
        }
    )
    df["irrigation_class"] = np.where(df["irr_source_fraction"] > 0.5, "irrigated", "rainfed")
    require_count(len(df), EXPECTED["E1_configured"], "E1 configured cohort")
    if "MB_Pch" not in set(df["site_id"]):
        raise BuildError("E1 configured scope: MB_Pch absent")
    return df


def e2_configured() -> pd.DataFrame:
    import zarr

    mapping = json.loads(E2_MAPPING_META.read_text())
    assignments = mapping["assignments"]
    z = zarr.open(str(E2_CONTAINER), mode="r")
    uid = [str(x) for x in z["geometry/uid"][:]]
    e2_lon, e2_lat = container_lonlat(
        E2_CONTAINER,
        z["geometry/lon"][:],
        z["geometry/lat"][:],
        "E2 configured cohort",
    )
    props = pd.DataFrame(
        {
            "site_id": uid,
            "lat": e2_lat,
            "lon": e2_lon,
            "country": [str(x) for x in z["geometry/properties/country"][:]],
            "network": [str(x) for x in z["geometry/properties/network"][:]],
        }
    )
    df = props[props["site_id"].isin(assignments)].copy()
    df["equipped_for_irrigation"] = df["site_id"].map(lambda s: bool(assignments[s]["equipped"]))
    df["irrigation_class"] = df["site_id"].map(lambda s: assignments[s]["irr_class"])
    require_count(len(df), EXPECTED["E2_configured"], "E2 configured cohort")
    return df.reset_index(drop=True)


def e3_configured() -> pd.DataFrame:
    import zarr

    z = zarr.open(str(E3_CONTAINER), mode="r")
    e3_lon, e3_lat = container_lonlat(
        E3_CONTAINER,
        z["geometry/lon"][:],
        z["geometry/lat"][:],
        "E3 configured cohort",
    )
    df = pd.DataFrame(
        {
            "site_id": [str(x) for x in z["geometry/uid"][:]],
            "lat": e3_lat,
            "lon": e3_lon,
            "src_id": [str(x) for x in z["geometry/properties/src_id"][:]],
            "crop": [str(x) for x in z["geometry/properties/crop"][:]],
            "basin": [str(x) for x in z["geometry/properties/basin"][:]],
            "acres": np.asarray(z["geometry/properties/acres"][:], dtype=float),
        }
    )
    df = df[df["site_id"].str.startswith("SLV_")].reset_index(drop=True)
    require_count(len(df), EXPECTED["E3_fields"], "E3 SLV cohort")
    return df


E2_CONFIGS = {
    "e3_uncal": ("generic_defaults", "E2 uncalibrated default parameters"),
    "ex5_transfer_strat": (
        "e1_irrigation_class_transfer",
        "fixed E1-derived irrigated/rainfed parameter sets assigned by the E2 equipped-for-irrigation classification",
    ),
    "e3_cal": (
        "local_calibration",
        "local E2 satellite calibration (LSEnsemble_GrassBasis_POR_annual2yr, HWSD aw priors)",
    ),
    "ls_ensemble": (
        "landsat_benchmark",
        "interpolated coincident Landsat SSEBop + PT-JPL benchmark",
    ),
}

E2_POOL_SUMMARY_LABELS = {
    "E3 uncalibrated/default": "e3_uncal",
    "Ex5 stratified transfer": "ex5_transfer_strat",
    "E3 calibrated": "e3_cal",
    "LS ensemble": "ls_ensemble",
}


def gate_closure_pool(pool: pd.DataFrame, label: str) -> set[str]:
    """Gate the frozen E2 closure pool and return its site ids.

    The pool must hold exactly the paper's 47 closure-corrected towers, each
    once, with no raw-ET tier present.
    """
    require_count(len(pool), EXPECTED["E2_pool_daily"], f"{label} closure pool")
    require_unique(pool, ["fid"], f"{label} closure pool")
    if (pool["closure_tier"] != "closure_corrected").any():
        raise BuildError(f"{label}: closure pool contains a non-closure-corrected site")
    return set(pool["fid"].astype(str))


def restrict_to_pool(df: pd.DataFrame, pool_ids: set[str], expected: int, label: str):
    """Restrict a per-site table indexed by site id to the closure pool.

    Every pool site must be present in the source table; the restricted table
    must have exactly ``expected`` rows.
    """
    ids = set(df.index.astype(str))
    if not pool_ids <= ids:
        raise BuildError(f"{label}: closure pool site absent from the per-site table")
    out = df.loc[df.index.astype(str).isin(pool_ids)].copy()
    require_count(len(out), expected, label)
    return out


def gate_pool_summary(summ: pd.DataFrame, label: str) -> pd.DataFrame:
    """Gate the frozen closure-pool cohort summary (``transfer_refresh_summary.csv``).

    Keeps the four reported configurations (seven rows: four daily, three
    monthly) and requires the daily rows to be on the 47-site pool and the
    monthly rows on the 39 finite-metric pool sites.
    """
    summ = summ[summ["config"].isin(E2_POOL_SUMMARY_LABELS)].copy()
    summ["legacy_config"] = summ["config"].map(E2_POOL_SUMMARY_LABELS)
    require_count(len(summ), 7, f"{label} frozen cohort summaries")
    if (summ.loc[summ["basis"] == "daily", "n_sites"] != EXPECTED["E2_pool_daily"]).any():
        raise BuildError(f"{label}: frozen daily summary is not on the 47-site pool")
    if (
        summ.loc[summ["basis"] == "monthly", "n_sites"] != EXPECTED["E2_pool_monthly_finite"]
    ).any():
        raise BuildError(
            f"{label}: frozen monthly summary is not on the 39 finite-metric pool sites"
        )
    return summ


def check_pool_medians(
    ps: pd.DataFrame, summ: pd.DataFrame, cfg: str, label: str, atol: float = 1e-6
) -> None:
    """The frozen 47-pool daily medians must be the medians of the restricted per-site table."""
    frozen = summ[(summ["legacy_config"] == cfg) & (summ["basis"] == "daily")]
    require_count(len(frozen), 1, f"{label} frozen daily summary for {cfg}")
    for col, fcol in (("kge", "kge_med"), ("rmse", "rmse_med"), ("bias", "bias_med")):
        got = float(ps[f"{cfg}_{col}"].median())
        want = float(frozen[fcol].iloc[0])
        if not np.isclose(got, want, rtol=0, atol=atol):
            raise BuildError(f"{label}: pool median {cfg} {col} {got!r} != frozen summary {want!r}")


def flag_evaluation_pool(
    e2: pd.DataFrame, e2_pool: pd.DataFrame, label: str
) -> tuple[pd.DataFrame, int, int]:
    """Flag the configured E2 scope with the paper's evaluation pool (Fig. 1).

    ``e2`` is the configured 66-site scope with ``site_id`` and ``in_e1``;
    ``e2_pool`` is the frozen closure pool with ``fid``, ``country`` and
    ``continent``.  Every pool site must be in scope; the flag count, the E1
    overlap within the pool, and the pool's country and continent counts must
    match ``EXPECTED``.  Returns the flagged frame and the two pool counts.
    """
    pool_ids = gate_closure_pool(e2_pool, label)
    if not pool_ids <= set(e2["site_id"].astype(str)):
        raise BuildError(f"{label}: closure-pool site not in the configured E2 scope")
    e2 = e2.copy()
    e2["in_evaluation_pool"] = e2["site_id"].astype(str).isin(pool_ids)
    require_count(
        int(e2["in_evaluation_pool"].sum()), EXPECTED["E2_pool_daily"], f"{label} E2 pool flag"
    )
    require_count(
        int((e2["in_evaluation_pool"] & e2["in_e1"]).sum()),
        EXPECTED["E1_E2_overlap_pool"],
        f"{label} E1 overlap within the E2 pool",
    )
    n_pool_countries = int(e2_pool["country"].nunique())
    n_pool_continents = int(e2_pool["continent"].nunique())
    require_count(n_pool_countries, EXPECTED["E2_pool_countries"], f"{label} E2 pool countries")
    require_count(n_pool_continents, EXPECTED["E2_pool_continents"], f"{label} E2 pool continents")
    return e2, n_pool_countries, n_pool_continents
