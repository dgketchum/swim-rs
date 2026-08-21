"""Freeze the display-ready figure-data package for the six-figure RSE package.

Phase 1 of ``paper/notes/six_figure_plan.md`` section 13: create
``paper/data/final/figures/`` and populate the frozen display tables plus
``fig_manifest.json``.

This module is a read-only transformation of already-archived evaluator
artifacts, plus one newly specified whole-field bootstrap (Fig. 6 panel d).
It never opens a calibration, never reruns a forward model, and never issues an
Earth Engine request.  The only containers it touches are opened read-only.

Legacy -> current experiment mapping (recorded once in the manifest and written
into every table):

    legacy ``e2_*`` / repository ``examples/5_Flux_Ensemble``  -> current E1
    legacy ``e3_*`` / repository ``examples/6_Flux_International`` -> current E2
    legacy ``e4_*`` / repository ``examples/7_Applied_Water``  -> current E3

The legacy ``e1_*`` artifacts belong to the removed broad-land-cover experiment
and are never read here.

Usage::

    uv run python scripts/figures/build_figure_data.py --all
    uv run python scripts/figures/build_figure_data.py --only fig06_bootstrap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0"

# Figure 1 display package only.  Kept separate from SCRIPT_VERSION so that
# revising the fig01 data contract does not perturb the byte content of the
# fig02-fig06 artifacts (two of which embed the package generator version).
FIG01_BUILDER_VERSION = "2.0.0"

REPO = Path(__file__).resolve().parents[2]
FINAL = REPO / "paper" / "data" / "final"
OUT = FINAL / "figures"

E1_RUN22 = Path("/data/ssd1/swim/5_Flux_Ensemble/results/run22")
E1_ARCHIVE = E1_RUN22 / "archive"
E1_CONTAINER = Path("/data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble_run22.swim")
E1_WITHIN_STRAT = Path(
    "/data/ssd1/swim/5_Flux_Ensemble/results/within_e2_transfer_irrigation_stratified"
)

E2_RESULTS = Path(
    "/data/ssd1/swim/6_Flux_International/results/6_Flux_International_LSEnsemble_POR_annual2yr"
)
E2_TRANSFER = Path(
    "/data/ssd1/swim/6_Flux_International/results/e2_run22_transfer_by_irrigation_to_e3"
)
E2_CONTAINER = Path(
    "/data/ssd1/swim/6_Flux_International/data/6_Flux_International_ls_ensemble_por_annual2yr.swim"
)

E3_LOCAL = Path("/data/ssd1/swim/7_Applied_Water/results/applied_calibrated")
E3_TRANSFER = Path("/data/ssd1/swim/7_Applied_Water/results/applied_transfer_run22_by_irrigation")
E3_CONTAINER = Path("/data/ssd1/swim/7_Applied_Water/data/7_Applied_Water_e7cal.swim")

NE_COUNTRIES = REPO / "data" / "cartographic" / "ne_110m_admin_0_countries.shp"

EXPERIMENT_MAP = {
    "E1": {
        "legacy_prefix": "e2_",
        "repository": "examples/5_Flux_Ensemble",
        "reader_facing_role": "CONUS cropland ET evaluation, reconstruction, ensemble reliability, held-out transfer",
        "configured_n": 60,
        "configured_unit": "cropland flux sites",
    },
    "E2": {
        "legacy_prefix": "e3_",
        "repository": "examples/6_Flux_International",
        "reader_facing_role": "Ten-country cropland evaluation and E1-derived parameter transfer under changed inputs",
        "configured_n": 66,
        "configured_unit": "cropland flux sites",
    },
    "E3": {
        "legacy_prefix": "e4_",
        "repository": "examples/7_Applied_Water",
        "reader_facing_role": "San Luis Valley metered applied-water evaluation",
        "configured_n": 50,
        "configured_unit": "metered fields",
    },
}

# Cohort counts asserted everywhere.  A mismatch stops the affected table.
EXPECTED = {
    "E1_configured": 60,
    "E1_daily": 45,
    "E1_monthly_finite": 29,
    "E1_transfer_daily": 45,
    "E1_transfer_monthly": 31,
    "E1_split_common": 43,
    "E2_configured": 66,
    "E2_daily": 63,
    "E2_monthly_support": 56,
    "E2_monthly_finite": 50,
    "E3_fields": 50,
    "E3_field_years": 408,
    "E1_E2_overlap": 13,
}

METRIC_DEFS = {
    "nse": {
        "definition": "Nash-Sutcliffe efficiency, 1 - SSE/SST, computed as sklearn.metrics.r2_score(flux_obs, model) with the observation vector as y_true",
        "direction": "higher is better",
        "units": "dimensionless",
        "legacy_column": "r2_*",
        "verification": "examples/5_Flux_Ensemble/evaluate.py::calc_metrics L228 and examples/6_Flux_International/evaluate.py::calc_metrics L357 both call r2_score(obs, mod) with obs=flux truth; that is 1 - SS_res/SS_tot about the observed mean, i.e. NSE, not a squared Pearson correlation. Pearson r is emitted separately as 'r'. Verified 2026-08-19.",
    },
    "kge": {
        "definition": "Kling-Gupta efficiency, Gupta et al. (2009): 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2), alpha = sd(mod)/sd(obs), beta = mean(mod)/mean(obs)",
        "direction": "higher is better",
        "units": "dimensionless",
        "legacy_column": "kge_*",
        "verification": "examples/5_Flux_Ensemble/evaluate.py L231-233; identical form in examples/6_Flux_International/evaluate.py L360-362.",
    },
    "rmse": {
        "definition": "root mean square error of model against flux ET",
        "direction": "lower is better",
        "units": "mm d-1 daily; mm month-1 monthly",
        "legacy_column": "rmse_*",
        "verification": "sqrt(sklearn.metrics.mean_squared_error(obs, mod))",
    },
    "mbe": {
        "definition": "mean bias error, mean(model - flux)",
        "direction": "zero is best; sign retained",
        "units": "mm d-1 daily; mm month-1 monthly",
        "legacy_column": "bias_*",
        "verification": "float((mod - obs).mean())",
    },
    "r": {
        "definition": "Pearson correlation coefficient between model and flux ET",
        "direction": "higher is better",
        "units": "dimensionless",
        "legacy_column": "r_*",
        "verification": "scipy.stats.pearsonr",
    },
    "mae": {
        "definition": "mean absolute error of model against flux ET",
        "direction": "lower is better",
        "units": "mm d-1 daily; mm month-1 monthly",
        "legacy_column": "mae_*",
        "verification": "mean(|mod - obs|) in transfer_ex5_params.py",
    },
}


class BuildError(RuntimeError):
    """Raised when a frozen table cannot be built to specification."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_sha256(path: Path, pattern: str = "*") -> str:
    """Stable hash over a directory's file names + contents (sorted)."""
    h = hashlib.sha256()
    for p in sorted(path.glob(pattern)):
        if p.is_file():
            h.update(p.name.encode())
            h.update(sha256(p).encode())
    return h.hexdigest()


def require_columns(df: pd.DataFrame, cols, label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise BuildError(f"{label}: missing required columns {missing}")


def require_unique(df: pd.DataFrame, keys, label: str) -> None:
    dup = df.duplicated(subset=list(keys)).sum()
    if dup:
        raise BuildError(f"{label}: {dup} duplicate rows on key {list(keys)}")


def require_count(actual: int, expected: int, label: str) -> None:
    if actual != expected:
        raise BuildError(f"{label}: expected {expected}, got {actual}")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "-C", str(REPO), "status", "--porcelain"], text=True
            ).strip()
        )
    except Exception:  # pragma: no cover
        return True


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


class Manifest:
    def __init__(self) -> None:
        self.tables: dict[str, dict] = {}
        self.status: list[dict] = []

    def add(self, name: str, **meta) -> None:
        meta.setdefault("generator_script", "scripts/figures/build_figure_data.py")
        meta.setdefault("generator_version", SCRIPT_VERSION)
        meta.setdefault("frozen_utc", datetime.now(UTC).isoformat())
        self.tables[name] = meta

    def block(self, item: str, reason: str, detail: str = "") -> None:
        self.status.append({"item": item, "reason": reason, "detail": detail})

    # Files the manifest self-hashes (everything the builders emit into OUT).
    OUTPUT_GLOBS = ("fig*.csv", "fig*.json", "fig*.gpkg")
    MANIFEST_NAME = "fig_manifest.json"

    def _output_hashes(self) -> dict[str, dict]:
        """SHA256 + size for every file produced into ``OUT``, computed after write.

        The manifest cannot hash itself, so ``fig_manifest.json`` is excluded;
        every other emitted artifact -- including ``fig01_scope.gpkg`` -- gets an
        independently verifiable digest (bug D4).
        """
        found: dict[str, Path] = {}
        for pattern in self.OUTPUT_GLOBS:
            for p in OUT.glob(pattern):
                if p.is_file() and p.name != self.MANIFEST_NAME:
                    found[p.name] = p
        return {
            name: {"sha256": sha256(found[name]), "bytes": found[name].stat().st_size}
            for name in sorted(found)
        }

    def write(self) -> Path:
        outputs = self._output_hashes()
        for name, meta in self.tables.items():
            if name in outputs:
                meta["output_sha256"] = outputs[name]["sha256"]
                meta["output_bytes"] = outputs[name]["bytes"]
            else:
                meta["output_sha256"] = None
                meta["output_bytes"] = None
        payload = {
            "package": "SWIM-RS six-figure display package",
            "phase": "Phase 1 - frozen figure data contract",
            "generator_script": "scripts/figures/build_figure_data.py",
            "generator_version": SCRIPT_VERSION,
            "frozen_utc": datetime.now(UTC).isoformat(),
            "repo_git_sha": git_sha(),
            "repo_worktree_dirty": git_dirty(),
            "governing_documents": [
                "paper/notes/figure_production_handoff.md",
                "paper/notes/six_figure_plan.md",
                "paper/notes/fig01_production_handoff.md",
                "paper/text/main.md",
                "paper/text/supp.md",
            ],
            "legacy_to_current_experiment_map": EXPERIMENT_MAP,
            "removed_experiment_note": (
                "The legacy e1_* artifacts under paper/data/final/ describe the removed "
                "broad-land-cover experiment. They are never read by this builder and must "
                "not appear in any reader-facing figure."
            ),
            "metric_definitions": METRIC_DEFS,
            "nse_vs_r2_guard": (
                "Every legacy r2_* column mapped to NSE in this package was verified "
                "against the evaluator implementation before renaming; see "
                "metric_definitions.nse.verification. No column was renamed on the basis "
                "of its name alone."
            ),
            "expected_cohort_counts": EXPECTED,
            "output_files": outputs,
            "output_hash_note": (
                "sha256 of every file this builder emitted into the package directory, "
                "computed after the file was written. fig_manifest.json itself is excluded "
                "because it cannot contain its own digest; hash it directly to verify. Each "
                "tables[*] entry repeats its own digest as output_sha256."
            ),
            "tables": self.tables,
            "blocked_or_incomplete": self.status,
        }
        p = OUT / "fig_manifest.json"
        p.write_text(json.dumps(payload, indent=2, sort_keys=False))
        return p


MANIFEST = Manifest()


def write_table(df: pd.DataFrame, name: str) -> int:
    path = OUT / name
    df.to_csv(path, index=False)
    return len(df)


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
    the tag and the numbers disagree (bug D3).
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

    mapping = json.loads(
        (FINAL / "e3_irrigation_stratified_param_mapping_metadata.json").read_text()
    )
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


# --------------------------------------------------------------------------
# Figure 2 -- external ET agreement
# --------------------------------------------------------------------------

_LONG_COLS = [
    "experiment",
    "legacy_prefix",
    "scale",
    "site_id",
    "treatment",
    "treatment_provenance",
    "n_paired",
    "nse",
    "kge",
    "rmse",
    "mbe",
    "r",
]


def _long_from_wide(df, suffix, experiment, scale, treatment, provenance):
    out = pd.DataFrame(
        {
            "experiment": experiment,
            "legacy_prefix": EXPERIMENT_MAP[experiment]["legacy_prefix"],
            "scale": scale,
            "site_id": df["fid"].astype(str),
            "treatment": treatment,
            "treatment_provenance": provenance,
            "n_paired": df["n"].astype(int),
            "nse": df[f"r2_{suffix}"].astype(float),
            "kge": df[f"kge_{suffix}"].astype(float),
            "rmse": df[f"rmse_{suffix}"].astype(float),
            "mbe": df[f"bias_{suffix}"].astype(float),
            "r": df[f"r_{suffix}"].astype(float),
        }
    )
    return out[_LONG_COLS]


def build_fig02() -> None:
    srcs = {
        "E1_daily": FINAL / "e2_primary_daily_site_metrics.csv",
        "E1_monthly": FINAL / "e2_primary_monthly_site_metrics.csv",
        "E1_daily_ledger": FINAL / "e2_primary_daily_exclusion_ledger.csv",
        "E1_monthly_ledger": FINAL / "e2_primary_monthly_exclusion_ledger.csv",
        "E2_daily": E2_RESULTS / "evaluation_metrics.csv",
        "E2_monthly": E2_RESULTS / "evaluation_monthly_metrics.csv",
    }
    for k, p in srcs.items():
        if not p.exists():
            raise BuildError(f"fig02 source missing: {k} -> {p}")

    e1d = pd.read_csv(srcs["E1_daily"])
    e1m = pd.read_csv(srcs["E1_monthly"])
    e2d = pd.read_csv(srcs["E2_daily"])
    e2m = pd.read_csv(srcs["E2_monthly"])

    for df, lbl in [(e1d, "E1 daily"), (e1m, "E1 monthly")]:
        require_columns(
            df,
            ["fid", "n", "r2_swim", "kge_swim", "rmse_swim", "bias_swim", "r_swim"],
            lbl,
        )
        require_unique(df, ["fid"], lbl)
    for df, lbl in [(e2d, "E2 daily"), (e2m, "E2 monthly")]:
        require_columns(df, ["fid", "n", "r2_swim", "r2_rs", "kge_swim", "kge_rs"], lbl)
        require_unique(df, ["fid"], lbl)

    # Finite-metric filtering: a site contributes only when both paired treatments
    # have a finite metric on identical support.
    e1m = e1m[e1m["kge_swim"].notna() & e1m["kge_ensemble"].notna()].copy()
    e2m = e2m[e2m["kge_swim"].notna() & e2m["kge_rs"].notna()].copy()

    require_count(len(e1d), EXPECTED["E1_daily"], "fig02 E1 daily cohort")
    require_count(len(e1m), EXPECTED["E1_monthly_finite"], "fig02 E1 monthly cohort")
    require_count(len(e2d), EXPECTED["E2_daily"], "fig02 E2 daily cohort")
    require_count(len(e2m), EXPECTED["E2_monthly_finite"], "fig02 E2 monthly cohort")

    daily = pd.concat(
        [
            _long_from_wide(
                e1d,
                "swim",
                "E1",
                "daily",
                "swim_rs_local_calibration",
                "run22 canonical daily evaluation",
            ),
            _long_from_wide(
                e1d,
                "ensemble",
                "E1",
                "daily",
                "openet_ensemble_benchmark",
                "separately extracted 3x3 OpenET v2.1 ensemble, linearly interpolated",
            ),
            _long_from_wide(
                e2d,
                "swim",
                "E2",
                "daily",
                "swim_rs_local_calibration",
                "ls_ensemble_por_annual2yr canonical daily evaluation",
            ),
            _long_from_wide(
                e2d,
                "rs",
                "E2",
                "daily",
                "landsat_benchmark",
                "interpolated coincident Landsat SSEBop + PT-JPL ensemble",
            ),
        ],
        ignore_index=True,
    )
    monthly = pd.concat(
        [
            _long_from_wide(
                e1m,
                "swim",
                "E1",
                "monthly",
                "swim_rs_local_calibration",
                "run22 canonical monthly evaluation (complete calendar months)",
            ),
            _long_from_wide(
                e1m,
                "ensemble",
                "E1",
                "monthly",
                "openet_ensemble_benchmark",
                "separately extracted 3x3 OpenET v2.1 ensemble, linearly interpolated",
            ),
            _long_from_wide(
                e2m,
                "swim",
                "E2",
                "monthly",
                "swim_rs_local_calibration",
                "ls_ensemble_por_annual2yr canonical monthly evaluation (paired-day aggregates)",
            ),
            _long_from_wide(
                e2m,
                "rs",
                "E2",
                "monthly",
                "landsat_benchmark",
                "interpolated coincident Landsat SSEBop + PT-JPL ensemble",
            ),
        ],
        ignore_index=True,
    )

    if not np.isfinite(daily[["nse", "kge", "rmse", "mbe"]].to_numpy()).all():
        raise BuildError("fig02 daily: non-finite plotted metric")
    if not np.isfinite(monthly[["nse", "kge", "rmse", "mbe"]].to_numpy()).all():
        raise BuildError("fig02 monthly: non-finite plotted metric")

    # paired site effects
    def effects(df, experiment, scale, bench):
        s = df[df["treatment"] == "swim_rs_local_calibration"].set_index("site_id")
        b = df[df["treatment"] == bench].set_index("site_id")
        common = s.index.intersection(b.index)
        if len(common) != len(s) or len(common) != len(b):
            raise BuildError(f"fig02 effects {experiment}/{scale}: support mismatch")
        return pd.DataFrame(
            {
                "experiment": experiment,
                "legacy_prefix": EXPERIMENT_MAP[experiment]["legacy_prefix"],
                "scale": scale,
                "site_id": common,
                "benchmark": bench,
                "n_paired": s.loc[common, "n_paired"].values,
                "d_nse": (s.loc[common, "nse"] - b.loc[common, "nse"]).values,
                "d_kge": (s.loc[common, "kge"] - b.loc[common, "kge"]).values,
                "d_rmse": (s.loc[common, "rmse"] - b.loc[common, "rmse"]).values,
                "d_abs_mbe": (s.loc[common, "mbe"].abs() - b.loc[common, "mbe"].abs()).values,
            }
        )

    eff = pd.concat(
        [
            effects(daily[daily.experiment == "E1"], "E1", "daily", "openet_ensemble_benchmark"),
            effects(daily[daily.experiment == "E2"], "E2", "daily", "landsat_benchmark"),
            effects(
                monthly[monthly.experiment == "E1"],
                "E1",
                "monthly",
                "openet_ensemble_benchmark",
            ),
            effects(monthly[monthly.experiment == "E2"], "E2", "monthly", "landsat_benchmark"),
        ],
        ignore_index=True,
    )

    nd = write_table(daily, "fig02_daily_site_metrics.csv")
    nm = write_table(monthly, "fig02_monthly_site_metrics.csv")
    ne = write_table(eff, "fig02_site_effects.csv")

    common_meta = {
        "sources": {k: {"path": str(p), "sha256": sha256(p)} for k, p in srcs.items()},
        "experiment_mapping": {"E1": "legacy e2_*", "E2": "legacy e3_*"},
        "cohort_key": "site_id",
        "inclusion_rule": (
            "E1 daily: the 45 sites surviving the run22 VALIDATION_POLICY site minimum "
            "(>=90 valid flux days, >=3 qualifying months) with MB_Pch excluded for flux "
            "provenance. E1 monthly: complete calendar months, >=10 paired months for a "
            "finite site-level metric (29 sites). E2 daily: the 63 of 66 configured sites "
            "with >=10 paired days. E2 monthly: paired-day aggregates, >=10 paired months "
            "(50 of 56 supported sites). Every retained site carries a finite metric for "
            "both paired treatments on identical support."
        ),
        "temporal_support_rule": (
            "Daily = common-support days on which flux ET, SWIM-RS ET, and the benchmark "
            "are all finite. E1 monthly = complete calendar months; E2 monthly = sums of "
            "paired days within a month (>=20 paired days per month, >=30 daily overlap). "
            "E1 and E2 monthly values are therefore NOT cross-comparable."
        ),
        "units": {
            "nse": "dimensionless",
            "kge": "dimensionless",
            "rmse": "mm d-1 (daily) / mm month-1 (monthly)",
            "mbe": "mm d-1 (daily) / mm month-1 (monthly)",
        },
        "display_transformations": [
            "legacy column r2_* renamed to nse after evaluator verification",
            "legacy column bias_* renamed to mbe",
            "legacy suffix _ensemble renamed to treatment openet_ensemble_benchmark",
            "legacy suffix _rs renamed to treatment landsat_benchmark",
            "wide per-site table reshaped to long (one row per site x treatment)",
        ],
        "deterministic_seed": None,
        "configured_counts": {"E1": 60, "E2": 66},
        "evaluated_counts": {
            "E1_daily": EXPECTED["E1_daily"],
            "E1_monthly_finite": EXPECTED["E1_monthly_finite"],
            "E2_daily": EXPECTED["E2_daily"],
            "E2_monthly_finite": EXPECTED["E2_monthly_finite"],
        },
    }
    MANIFEST.add("fig02_daily_site_metrics.csv", rows=nd, **common_meta)
    MANIFEST.add("fig02_monthly_site_metrics.csv", rows=nm, **common_meta)
    MANIFEST.add(
        "fig02_site_effects.csv",
        rows=ne,
        note=(
            "Paired SWIM-minus-benchmark site effects. Natural signs retained: positive "
            "d_nse/d_kge favour SWIM-RS; negative d_rmse/d_abs_mbe favour SWIM-RS. "
            "Descriptive distributions only -- no bootstrap interval is attached."
        ),
        **common_meta,
    )
    print(
        f"  fig02: daily {nd} rows ({EXPECTED['E1_daily']} E1 + {EXPECTED['E2_daily']} E2 sites x 2 treatments), "
        f"monthly {nm} rows, effects {ne} rows"
    )


# --------------------------------------------------------------------------
# Figure 3 -- temporal reconstruction
# --------------------------------------------------------------------------


def build_fig03_deltas() -> pd.DataFrame:
    src_metrics = E1_ARCHIVE / "6_evaluation" / "overpass_split_metrics.csv"
    src_summary = FINAL / "e2_temporal_summary.csv"
    src_deltas = FINAL / "e2_temporal_paired_deltas.csv"
    src_audit = E1_ARCHIVE / "6_evaluation" / "overpass_date_audit.csv"
    for p in (src_metrics, src_summary, src_deltas, src_audit):
        if not p.exists():
            raise BuildError(f"fig03 source missing: {p}")

    m = pd.read_csv(src_metrics)
    require_columns(
        m,
        [
            "fid",
            "subset",
            "n_paired",
            "eligible",
            "nse_swim",
            "kge_swim",
            "rmse_swim",
            "mbe_swim",
            "nse_openet",
            "kge_openet",
            "rmse_openet",
            "mbe_openet",
        ],
        "overpass_split_metrics",
    )
    require_unique(m, ["fid", "subset"], "overpass_split_metrics")

    wide = m.pivot_table(index="fid", columns="subset", values="eligible", aggfunc="first")
    common = sorted(
        wide.index[
            wide.get("overpass", False).astype(bool) & wide.get("non_overpass", False).astype(bool)
        ]
    )
    require_count(len(common), EXPECTED["E1_split_common"], "fig03 common-split cohort")

    audit = pd.read_csv(src_audit).set_index("fid")

    df = m[m["fid"].isin(common)].copy()
    label = {
        "overpass": "direct_benchmark",
        "non_overpass": "benchmark_interpolated",
        "all_days": "all_days",
    }
    df["subset_display"] = df["subset"].map(label)
    if df["subset_display"].isna().any():
        raise BuildError("fig03: unmapped legacy subset label")
    df["experiment"] = "E1"
    df["legacy_prefix"] = "e2_"
    df["legacy_subset"] = df["subset"]
    df["d_nse"] = df["nse_swim"] - df["nse_openet"]
    df["d_kge"] = df["kge_swim"] - df["kge_openet"]
    df["d_rmse"] = df["rmse_swim"] - df["rmse_openet"]
    df["d_abs_mbe"] = df["mbe_swim"].abs() - df["mbe_openet"].abs()
    df["n_calibration_captures"] = df["fid"].map(audit["n_calibration_captures"])
    df["n_raw_benchmark_captures"] = df["fid"].map(audit["n_benchmark_captures"])

    all_days = df[df["subset"] == "all_days"].set_index("fid")["n_paired"]
    non = df[df["subset"] == "non_overpass"].set_index("fid")["n_paired"]
    df["site_fraction_benchmark_interpolated"] = df["fid"].map(non / all_days)

    cols = [
        "experiment",
        "legacy_prefix",
        "site_id",
        "subset_display",
        "legacy_subset",
        "n_paired",
        "first_date",
        "last_date",
        "nse_swim",
        "kge_swim",
        "r_swim",
        "rmse_swim",
        "mbe_swim",
        "nse_openet",
        "kge_openet",
        "r_openet",
        "rmse_openet",
        "mbe_openet",
        "d_nse",
        "d_kge",
        "d_rmse",
        "d_abs_mbe",
        "n_calibration_captures",
        "n_raw_benchmark_captures",
        "site_fraction_benchmark_interpolated",
    ]
    df = df.rename(columns={"fid": "site_id"})[cols].sort_values(["site_id", "subset_display"])
    n = write_table(df, "fig03_temporal_site_deltas.csv")
    require_count(n, EXPECTED["E1_split_common"] * 3, "fig03 site-delta rows")

    # cohort-level frozen effects + summary (companion table)
    deltas = pd.read_csv(src_deltas)
    deltas = deltas[deltas["cohort"] == "common_split"].copy()
    deltas["subset_display"] = deltas["subset"].map(label)
    summ = pd.read_csv(src_summary)
    summ = summ[summ["cohort"] == "common_split"].copy()
    summ["subset_display"] = summ["subset"].map(label)
    eff = deltas.rename(
        columns={
            "median_delta_swim_minus_openet": "median_delta",
            "ci95_low": "ci95_lo",
            "ci95_high": "ci95_hi",
        }
    )
    eff["experiment"] = "E1"
    eff["legacy_prefix"] = "e2_"
    eff = eff[
        [
            "experiment",
            "legacy_prefix",
            "metric",
            "subset_display",
            "subset",
            "n_sites",
            "median_delta",
            "ci95_lo",
            "ci95_hi",
            "seed",
            "n_resamples",
        ]
    ]
    keep = [
        "subset_display",
        "subset",
        "n_sites",
        "total_paired_site_days",
        "median_paired_days_per_site",
        "iqr25_paired_days",
        "iqr75_paired_days",
        "median_nse_swim",
        "median_kge_swim",
        "median_rmse_swim",
        "median_mbe_swim",
        "median_nse_openet",
        "median_kge_openet",
        "median_rmse_openet",
        "median_mbe_openet",
        "site_median_non_overpass_fraction",
        "pooled_non_overpass_fraction",
    ]
    summ_out = summ[keep].copy()
    summ_out.insert(0, "experiment", "E1")
    both = pd.concat(
        [
            eff.assign(record_type="paired_effect"),
            summ_out.assign(record_type="cohort_summary", legacy_prefix="e2_"),
        ],
        ignore_index=True,
    )
    ne = write_table(both, "fig03_temporal_cohort_effects.csv")

    meta = {
        "sources": {
            "overpass_split_metrics": {
                "path": str(src_metrics),
                "sha256": sha256(src_metrics),
            },
            "overpass_date_audit": {"path": str(src_audit), "sha256": sha256(src_audit)},
            "e2_temporal_summary": {
                "path": str(src_summary),
                "sha256": sha256(src_summary),
            },
            "e2_temporal_paired_deltas": {
                "path": str(src_deltas),
                "sha256": sha256(src_deltas),
            },
        },
        "experiment_mapping": {"E1": "legacy e2_*"},
        "cohort_key": "site_id",
        "inclusion_rule": (
            "43-site common-split cohort: canonical run22 45-site daily cohort restricted "
            "to sites with >=10 paired days in BOTH the direct-benchmark and "
            "benchmark-interpolated subsets. JPL1_Smith5 (8 direct days) and US-OF1 (7) "
            "are excluded from the split comparison."
        ),
        "temporal_support_rule": (
            "A date is direct_benchmark when the raw Volk v2.1 ensemble_mean_3x3 value is "
            "finite BEFORE interpolation; benchmark_interpolated when it lacks a raw value "
            "but falls inside the site's first-to-last raw benchmark support and receives a "
            "finite linearly interpolated value. No extrapolation outside raw support. The "
            "archived is_overpass column is a calibration-capture flag and was NOT used to "
            "classify dates; it is retained as n_calibration_captures for cross-tabulation."
        ),
        "display_transformations": [
            "legacy subset label 'overpass' -> reader-facing 'direct_benchmark'",
            "legacy subset label 'non_overpass' -> reader-facing 'benchmark_interpolated'",
            "no numeric transformation applied",
        ],
        "units": {"rmse": "mm d-1", "mbe": "mm d-1"},
        "deterministic_seed": 42,
        "bootstrap": "site-level, 10,000 resamples, seed 42 (frozen upstream)",
        "configured_counts": {"E1": 60},
        "evaluated_counts": {
            "daily_cohort": 45,
            "common_split_cohort": 43,
            "direct_benchmark_site_days": 4751,
            "benchmark_interpolated_site_days": 55584,
        },
    }
    MANIFEST.add("fig03_temporal_site_deltas.csv", rows=n, **meta)
    MANIFEST.add(
        "fig03_temporal_cohort_effects.csv",
        rows=ne,
        note=(
            "Companion to fig03_temporal_site_deltas.csv holding the frozen cohort medians "
            "and the 10,000-resample site-bootstrap 95% intervals used in panel (b). Added "
            "beyond the recommended package file list so that panel (b) never recomputes an "
            "interval."
        ),
        **meta,
    )
    print(f"  fig03: site deltas {n} rows (43 sites x 3 subsets), cohort effects {ne} rows")
    return df


def build_fig03_example(deltas: pd.DataFrame) -> None:
    """Select the representative site/window by the frozen Section 7 rule."""
    interp = deltas[deltas["subset_display"] == "benchmark_interpolated"].copy()
    cohort_median = float(interp["d_kge"].median())
    interp["rank_value"] = (interp["d_kge"] - cohort_median).abs()
    interp = interp.sort_values(["rank_value", "site_id"]).reset_index(drop=True)

    window_days = 120
    rule = {
        "step_1_cohort": "43-site common split cohort (both subsets eligible, >=10 paired days each)",
        "step_2_support": (
            "A window qualifies when it is a contiguous 120-day span whose start date "
            "falls between 1 April and 31 July, in which (a) at least 90% of days have "
            "finite flux ET, SWIM-RS ET and the interpolated benchmark, (b) at least 3 "
            "direct-benchmark dates occur, (c) at least 50% of paired days are "
            "benchmark-interpolated, and (d) at least 1 SWIM-RS calibration capture occurs."
        ),
        "step_3_ranking": "abs(site d_kge on benchmark-interpolated dates - cohort median d_kge on benchmark-interpolated dates)",
        "step_4_selection": "nearest qualifying site",
        "step_5_tiebreak": "earliest qualifying window start date within the selected site; site ties broken by ascending site_id",
        "step_6_freeze": "eligible list, ranking values, chosen site and dates frozen in this file",
        "window_length_days": window_days,
        "cohort_median_d_kge_benchmark_interpolated": cohort_median,
    }

    chosen = None
    for _, row in interp.iterrows():
        fid = row["site_id"]
        series = _load_e1_site_series(fid)
        if series is None:
            continue
        win = _first_qualifying_window(series, window_days)
        if win is not None:
            chosen = (fid, win, row)
            break
    if chosen is None:
        raise BuildError("fig03 example: no site produced a qualifying window")

    fid, win, row = chosen
    start, end, diag = win
    series = _load_e1_site_series(fid)
    sub = series.loc[start:end].copy()

    ndvi = _e1_raw_ndvi(fid)
    sub = sub.join(ndvi, how="left")

    om = pd.read_csv(E1_ARCHIVE / "3_problem_definition" / "observation_metadata.csv")
    om = om[(om["site"] == fid) & (om["model"] == "ensemble")].copy()
    om["date"] = pd.to_datetime(om["date"])
    om = om.set_index("date")
    sub["calibration_target_etf"] = om["target_etf"].reindex(sub.index)
    sub["calibration_target_member_count"] = om["member_count"].reindex(sub.index)
    sub["calibration_target_ensemble_std"] = om["ensemble_std"].reindex(sub.index)
    sub["calibration_target_final_weight"] = om["final_weight"].reindex(sub.index)
    sub["is_calibration_capture"] = sub["calibration_target_etf"].notna()

    out = sub.reset_index().rename(columns={"index": "date"})
    out.insert(0, "site_id", fid)
    out.insert(0, "legacy_prefix", "e2_")
    out.insert(0, "experiment", "E1")
    n = write_table(out, "fig03_example_timeseries.csv")

    payload = {
        "figure": "fig03",
        "panel": "a",
        "experiment": "E1",
        "legacy_prefix": "e2_",
        "selection_rule": rule,
        "eligible_sites": interp["site_id"].tolist(),
        "ranking": interp[["site_id", "d_kge", "rank_value"]].to_dict("records"),
        "selected_site": fid,
        "selected_window_start": str(start.date()),
        "selected_window_end": str(end.date()),
        "selected_window_diagnostics": diag,
        "selected_site_rank_value": float(row["rank_value"]),
        "selected_site_d_kge_benchmark_interpolated": float(row["d_kge"]),
        "sources": {
            "site_daily_series": {
                "path": str(E1_ARCHIVE / "6_evaluation" / "site_daily_timeseries" / f"{fid}.csv"),
                "sha256": sha256(
                    E1_ARCHIVE / "6_evaluation" / "site_daily_timeseries" / f"{fid}.csv"
                ),
            },
            "raw_openet_benchmark": {
                "path": str(
                    Path("/data/ssd1/swim/5_Flux_Ensemble/data/openet_flux/daily_data")
                    / f"{fid}.csv"
                ),
                "sha256": sha256(
                    Path("/data/ssd1/swim/5_Flux_Ensemble/data/openet_flux/daily_data")
                    / f"{fid}.csv"
                ),
            },
            "observation_metadata": {
                "path": str(E1_ARCHIVE / "3_problem_definition" / "observation_metadata.csv"),
                "sha256": sha256(E1_ARCHIVE / "3_problem_definition" / "observation_metadata.csv"),
            },
            "container_raw_ndvi": {
                "path": str(E1_CONTAINER),
                "arrays": [
                    "remote_sensing/ndvi/landsat/no_mask",
                    "remote_sensing/ndvi/sentinel/no_mask",
                ],
                "note": "zarr store; per-array hash not computed (directory store)",
            },
        },
        "generator_script": "scripts/figures/build_figure_data.py",
        "generator_version": SCRIPT_VERSION,
    }
    (OUT / "fig03_example_selection.json").write_text(json.dumps(payload, indent=2))

    MANIFEST.add(
        "fig03_example_selection.json",
        rows=None,
        note="Frozen Section 7 six-step representative-window selection; no visual inspection entered the choice.",
        selected_site=fid,
        selected_window=[str(start.date()), str(end.date())],
        cohort_key="site_id",
        inclusion_rule=rule["step_1_cohort"],
        deterministic_seed=None,
    )
    MANIFEST.add(
        "fig03_example_timeseries.csv",
        rows=n,
        sources=payload["sources"],
        experiment_mapping={"E1": "legacy e2_*"},
        cohort_key="site_id + date",
        inclusion_rule=f"single selected site {fid}, {start.date()} to {end.date()} inclusive",
        temporal_support_rule=(
            "daily; benchmark_raw is the pre-interpolation Volk v2.1 ensemble_mean_3x3, "
            "benchmark_interpolated is that series linearly interpolated between its first "
            "and last finite value with no extrapolation"
        ),
        units={
            "swim_ET": "mm d-1",
            "flux_ET": "mm d-1",
            "benchmark_raw": "mm d-1",
            "benchmark_interpolated": "mm d-1",
            "precip": "mm d-1",
            "irr_applied": "mm d-1",
            "rz_depletion": "mm",
            "ndvi_kcb": "dimensionless",
            "ndvi_landsat_raw": "dimensionless",
            "ndvi_sentinel_raw": "dimensionless",
        },
        display_transformations=[
            "raw Landsat and Sentinel-2 NDVI joined as separate observation columns; no filled/interpolated NDVI trace is frozen because the Section 4.1 NDVI-filling gate is unresolved",
            "calibration-target ETf rows joined from archived observation metadata and flagged separately from benchmark support",
        ],
        deterministic_seed=None,
        configured_counts={"E1": 60},
        evaluated_counts={"days_in_window": n},
    )
    print(f"  fig03 example: site {fid}, {start.date()}..{end.date()}, {n} daily rows")


def _load_e1_site_series(fid: str):
    """Archived run22 daily series joined to the raw + interpolated benchmark."""
    p = E1_ARCHIVE / "6_evaluation" / "site_daily_timeseries" / f"{fid}.csv"
    b = Path("/data/ssd1/swim/5_Flux_Ensemble/data/openet_flux/daily_data") / f"{fid}.csv"
    if not p.exists() or not b.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    raw = pd.read_csv(b)
    date_col = "DATE" if "DATE" in raw.columns else "date"
    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = raw.set_index(date_col)
    if "ensemble_mean_3x3" not in raw.columns:
        return None
    ens = pd.to_numeric(raw["ensemble_mean_3x3"], errors="coerce")
    finite = ens.dropna()
    if finite.empty:
        return None
    idx = pd.date_range(finite.index.min(), finite.index.max(), freq="D")
    ens_daily = ens.reindex(idx).interpolate(method="linear", limit_area="inside")
    df = df.rename(columns={"is_overpass": "is_calibration_capture_archived"})
    df["benchmark_raw"] = ens.reindex(df.index)
    df["benchmark_interpolated"] = ens_daily.reindex(df.index)
    df["is_direct_benchmark"] = df["benchmark_raw"].notna()
    return df


def _first_qualifying_window(series: pd.DataFrame, window_days: int):
    need = ["flux_ET", "swim_ET", "benchmark_interpolated"]
    for c in need:
        if c not in series.columns:
            return None
    years = sorted(set(series.index.year))
    for yr in years:
        for month, day in [(4, 1), (5, 1), (6, 1), (7, 1)]:
            start = pd.Timestamp(year=yr, month=month, day=day)
            end = start + pd.Timedelta(days=window_days - 1)
            if end > series.index.max():
                continue
            w = series.loc[start:end]
            if len(w) < window_days:
                continue
            complete = w[need].notna().all(axis=1)
            frac = float(complete.mean())
            n_direct = int((w["is_direct_benchmark"] & complete).sum())
            n_paired = int(complete.sum())
            n_interp = n_paired - n_direct
            n_cal = int(w["is_calibration_capture_archived"].fillna(False).astype(bool).sum())
            if (
                frac >= 0.90
                and n_direct >= 3
                and n_paired > 0
                and (n_interp / n_paired) >= 0.50
                and n_cal >= 1
            ):
                return (
                    start,
                    end,
                    {
                        "complete_fraction": frac,
                        "n_paired_days": n_paired,
                        "n_direct_benchmark_days": n_direct,
                        "n_benchmark_interpolated_days": n_interp,
                        "n_calibration_captures": n_cal,
                    },
                )
    return None


def _e1_raw_ndvi(fid: str) -> pd.DataFrame:
    import zarr

    z = zarr.open(str(E1_CONTAINER), mode="r")
    uid = [str(x) for x in z["geometry/uid"][:]]
    j = uid.index(fid)
    t = pd.to_datetime(z["time/daily"][:])
    return pd.DataFrame(
        {
            "ndvi_landsat_raw": np.asarray(
                z["remote_sensing/ndvi/landsat/no_mask"][:, j], dtype=float
            ),
            "ndvi_sentinel_raw": np.asarray(
                z["remote_sensing/ndvi/sentinel/no_mask"][:, j], dtype=float
            ),
        },
        index=t,
    )


# --------------------------------------------------------------------------
# Figure 4 -- ensemble spread as reliability information
# --------------------------------------------------------------------------

E1_MEMBERS = ["ssebop", "ptjpl", "sims", "geesebal", "eemetric", "disalexi"]


def build_fig04() -> None:
    src_obs = E1_RUN22 / "spread_error" / "spread_error_observations.csv"
    src_q = FINAL / "e2_spread_error_quintiles.csv"
    src_ps = FINAL / "e2_spread_error_persite.csv"
    src_sum = FINAL / "e2_spread_error_summary.csv"
    src_unc_ps = E1_RUN22 / "conditioned_ensemble_uncertainty" / "uncertainty_persite.csv"
    src_unc_obs = E1_RUN22 / "conditioned_ensemble_uncertainty" / "uncertainty_observations.csv"
    src_unc_sum = E1_RUN22 / "conditioned_ensemble_uncertainty" / "uncertainty_summary.csv"
    src_wd = FINAL / "e2_weighting_ablation_paired_deltas.csv"
    src_wdd = FINAL / "e2_weighting_ablation_daily_site_deltas.csv"
    src_wdm = FINAL / "e2_weighting_ablation_monthly_site_deltas.csv"
    for p in (
        src_obs,
        src_q,
        src_ps,
        src_sum,
        src_unc_ps,
        src_unc_obs,
        src_unc_sum,
        src_wd,
        src_wdd,
        src_wdm,
    ):
        if not p.exists():
            raise BuildError(f"fig04 source missing: {p}")

    obs = pd.read_csv(src_obs, parse_dates=["date"])
    require_columns(
        obs,
        ["site", "date", "member_count", "spread", "target", "flux_etf", "weight"],
        "spread_error_observations",
    )
    require_unique(obs, ["site", "date"], "spread_error_observations")
    if len(obs) != 2131:
        raise BuildError(f"fig04: expected 2131 paired captures, got {len(obs)}")
    if obs["site"].nunique() != 33:
        raise BuildError(f"fig04: expected 33 sites, got {obs['site'].nunique()}")

    members = _e1_member_etf(obs)
    cap = obs.merge(members, on=["site", "date"], how="left", validate="one_to_one")

    unc_obs = pd.read_csv(src_unc_obs, parse_dates=["date"])
    keep = [
        "site",
        "date",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "spread_conditioned_iqr",
        "spread_conditioned_std",
        "width90",
        "etf_effective",
        "abs_error_effective",
        "covered_90",
    ]
    cap = cap.merge(unc_obs[keep], on=["site", "date"], how="left", validate="one_to_one")

    recomputed = cap[[f"etf_{m}" for m in E1_MEMBERS]].notna().sum(axis=1)
    mismatch = int((recomputed != cap["member_count"]).sum())
    if mismatch:
        raise BuildError(
            f"fig04: member-count mismatch between container arrays and archived metadata at {mismatch} captures"
        )

    cap.insert(0, "legacy_prefix", "e2_")
    cap.insert(0, "experiment", "E1")
    cap = cap.rename(
        columns={
            "site": "site_id",
            "spread": "spread_retrieval",
            "target": "ensemble_mean_etf",
            "flux_etf": "flux_derived_etf",
            "err_etf": "error_etf",
            "abs_err_etf": "abs_error_etf",
            "etf_effective": "swim_etf_effective",
            "abs_error_effective": "swim_abs_error_effective",
        }
    )
    n_cap = write_table(cap, "fig04_spread_capture_values.csv")

    q = pd.read_csv(src_q)
    q.insert(0, "legacy_prefix", "e2_")
    q.insert(0, "experiment", "E1")
    n_q = write_table(q, "fig04_spread_quintiles.csv")

    ps = pd.read_csv(src_ps).rename(
        columns={"site": "site_id", "spearman_rho": "rho_retrieval_vs_abs_ensemble_error"}
    )
    unc_ps = pd.read_csv(src_unc_ps).rename(columns={"site": "site_id"})
    assoc = ps.merge(
        unc_ps[
            [
                "site_id",
                "n",
                "eligible",
                "rho_retrieval",
                "rho_conditioned_iqr",
                "rho_conditioned_std",
                "rho_conditioned_width90",
                "delta_rho",
                "coverage_90",
                "median_width90",
                "median_iqr",
                "median_spread_retrieval",
            ]
        ],
        on="site_id",
        how="outer",
        suffixes=("_spread_analysis", "_conditioned"),
    )
    assoc.insert(0, "legacy_prefix", "e2_")
    assoc.insert(0, "experiment", "E1")
    n_assoc = write_table(assoc, "fig04_site_associations.csv")

    elig = unc_ps[unc_ps["eligible"].astype(bool)]
    if len(elig) != 27:
        raise BuildError(f"fig04: expected 27 eligible sites, got {len(elig)}")
    if len(ps) != 27:
        raise BuildError(f"fig04: expected 27 spread-analysis sites, got {len(ps)}")
    n_pos = int((ps["rho_retrieval_vs_abs_ensemble_error"] > 0).sum())
    if n_pos != 26:
        raise BuildError(
            f"fig04: expected 26 of 27 positive within-site spread-error associations, got {n_pos}"
        )

    cond = unc_ps.copy()
    cond.insert(0, "legacy_prefix", "e2_")
    cond.insert(0, "experiment", "E1")
    summary = pd.read_csv(src_unc_sum)
    n_cond = write_table(cond, "fig04_conditioned_spread.csv")

    wd = pd.read_csv(src_wd)
    require_columns(
        wd,
        ["scale", "metric", "n_sites", "median_delta", "ci_lower", "ci_upper"],
        "weighting_ablation_paired_deltas",
    )
    wdd = pd.read_csv(src_wdd)
    wdm = pd.read_csv(src_wdm)

    def _site_deltas(df, scale):
        return pd.DataFrame(
            {
                "experiment": "E1",
                "legacy_prefix": "e2_",
                "record_type": "site_delta",
                "scale": scale,
                "site_id": df["fid"],
                "metric": None,
                "n_paired": df["n_paired"],
                "d_nse": df["delta_r2_swim"],
                "d_kge": df["delta_kge_swim"],
                "d_rmse": df["delta_rmse_swim"],
                "d_abs_mbe": df["e1_bias_swim"].abs() - df["e2_bias_swim"].abs(),
            }
        )

    site_rows = pd.concat(
        [_site_deltas(wdd, "daily"), _site_deltas(wdm, "monthly")], ignore_index=True
    )
    metric_rename = {
        "nse": "nse",
        "r2": "nse",
        "kge": "kge",
        "rmse": "rmse",
        "abs_bias": "abs_mbe",
        "abs_mbe": "abs_mbe",
    }
    eff_rows = wd.copy()
    eff_rows["metric_display"] = eff_rows["metric"].map(metric_rename)
    if eff_rows["metric_display"].isna().any():
        raise BuildError("fig04 weighting: unmapped metric label")
    eff_rows = pd.DataFrame(
        {
            "experiment": "E1",
            "legacy_prefix": "e2_",
            "record_type": "cohort_effect",
            "scale": eff_rows["scale"],
            "site_id": None,
            "metric": eff_rows["metric_display"],
            "n_paired": eff_rows["n_sites"],
            "median_delta": eff_rows["median_delta"],
            "ci95_lo": eff_rows["ci_lower"],
            "ci95_hi": eff_rows["ci_upper"],
            "delta_definition": eff_rows["delta_definition"],
            "favorable_direction": eff_rows["favorable_direction"],
            "bootstrap_seed": eff_rows["bootstrap_seed"],
            "bootstrap_reps": eff_rows["bootstrap_reps"],
        }
    )
    weighting = pd.concat([eff_rows, site_rows], ignore_index=True)
    n_w = write_table(weighting, "fig04_weighting_effects.csv")

    # panel (a) example selection
    eligible = ps.merge(unc_ps[["site_id", "eligible"]], on="site_id", how="left")
    eligible = eligible[eligible["n"] >= 20].copy()
    med = float(eligible["rho_retrieval_vs_abs_ensemble_error"].median())
    eligible["rank_value"] = (eligible["rho_retrieval_vs_abs_ensemble_error"] - med).abs()
    eligible = eligible.sort_values(["rank_value", "site_id"]).reset_index(drop=True)
    chosen_site = str(eligible.loc[0, "site_id"])
    caps = cap[cap["site_id"] == chosen_site].copy()
    payload = {
        "figure": "fig04",
        "panel": "a",
        "experiment": "E1",
        "legacy_prefix": "e2_",
        "selection_rule": {
            "step_1": "sites with at least 20 paired capture-level observations in the E1 spread-error analysis",
            "step_2": "rank by abs(within-site Spearman rho(retrieval spread, |ensemble-mean ETf error|) - median rho among eligible sites)",
            "step_3": "select the nearest site; ties broken by ascending site_id",
            "step_4": "freeze the eligible list, ranking values, chosen site, and the exact capture dates",
        },
        "n_eligible_sites": int(len(eligible)),
        "median_rho_among_eligible": med,
        "ranking": eligible[
            ["site_id", "n", "rho_retrieval_vs_abs_ensemble_error", "rank_value"]
        ].to_dict("records"),
        "selected_site": chosen_site,
        "selected_site_rho": float(eligible.loc[0, "rho_retrieval_vs_abs_ensemble_error"]),
        "selected_captures": [
            {
                "date": str(pd.Timestamp(r["date"]).date()),
                "member_count": int(r["member_count"]),
                "ensemble_mean_etf": float(r["ensemble_mean_etf"]),
                "spread_retrieval": float(r["spread_retrieval"]),
                "flux_derived_etf": float(r["flux_derived_etf"]),
                **{
                    f"etf_{m}": (None if pd.isna(r[f"etf_{m}"]) else float(r[f"etf_{m}"]))
                    for m in E1_MEMBERS
                },
            }
            for _, r in caps.sort_values("date").iterrows()
        ],
        "generator_script": "scripts/figures/build_figure_data.py",
        "generator_version": SCRIPT_VERSION,
    }
    (OUT / "fig04_example_selection.json").write_text(json.dumps(payload, indent=2))

    srcs = {
        "spread_error_observations": src_obs,
        "e2_spread_error_quintiles": src_q,
        "e2_spread_error_persite": src_ps,
        "e2_spread_error_summary": src_sum,
        "uncertainty_persite": src_unc_ps,
        "uncertainty_observations": src_unc_obs,
        "uncertainty_summary": src_unc_sum,
        "e2_weighting_ablation_paired_deltas": src_wd,
        "e2_weighting_ablation_daily_site_deltas": src_wdd,
        "e2_weighting_ablation_monthly_site_deltas": src_wdm,
    }
    base = {
        "sources": {k: {"path": str(p), "sha256": sha256(p)} for k, p in srcs.items()},
        "container_source": {
            "path": str(E1_CONTAINER),
            "arrays": [f"remote_sensing/etf/landsat/{m}/no_mask" for m in E1_MEMBERS],
        },
        "experiment_mapping": {"E1": "legacy e2_*"},
        "cohort_key": "site_id (+ date for capture-level rows)",
        "inclusion_rule": (
            "Capture-level: 2,131 ETf ensemble calibration targets across 33 sites that "
            "have a finite same-day flux ET_corr and a reference ETo >= 0.5 mm d-1; sites "
            "must pass the VALIDATION_POLICY minimum and MB_Pch is excluded. Site-level "
            "associations: the 27 sites with at least 20 paired captures."
        ),
        "temporal_support_rule": "Landsat acquisition dates retained as ETf calibration targets, 2016-2025, paired to same-day flux observations (Volk v2.1 record ends mid-2022).",
        "units": {
            "etf": "dimensionless ET fraction",
            "spread_retrieval": "dimensionless ETf standard deviation across valid members (ddof=1)",
            "weight": "dimensionless PEST observation weight, ETf_obs/(sigma+0.1)",
            "rmse_daily": "mm d-1",
            "rmse_monthly": "mm month-1",
        },
        "deterministic_seed": 42,
        "configured_counts": {"E1": 60},
        "evaluated_counts": {
            "captures": 2131,
            "capture_sites": 33,
            "association_sites": 27,
            "weighting_daily_sites": 45,
            "weighting_monthly_sites": 31,
        },
    }
    MANIFEST.add(
        "fig04_spread_capture_values.csv",
        rows=n_cap,
        display_transformations=[
            "six OpenET member ETf values joined from the run22 container and cross-checked against the archived member_count (exact match required)",
            "conditioned-ensemble quantiles joined from uncertainty_observations.csv on (site, date)",
            "legacy column 'spread' renamed spread_retrieval; 'target' renamed ensemble_mean_etf",
        ],
        **base,
    )
    MANIFEST.add(
        "fig04_spread_quintiles.csv",
        rows=n_q,
        display_transformations=["experiment label columns prepended; values unchanged"],
        note="Quintiles of retrieval spread over the 2,131 pooled captures; MAE/RMSE in ETf units. Descriptive, not a calibrated uncertainty function.",
        **base,
    )
    MANIFEST.add(
        "fig04_site_associations.csv",
        rows=n_assoc,
        display_transformations=[
            "spread-analysis per-site Spearman rho joined to the conditioned-ensemble per-site rho on site_id",
            "legacy column spearman_rho renamed rho_retrieval_vs_abs_ensemble_error",
        ],
        **base,
    )
    MANIFEST.add(
        "fig04_conditioned_spread.csv",
        rows=n_cond,
        display_transformations=["experiment label columns prepended; values unchanged"],
        note=(
            "Conditioned-ensemble diagnostic. delta_rho = rho(retrieval spread, |err|) - "
            "rho(conditioned IQR, |err|). Cohort estimate +0.216 [+0.128, +0.334] over 27 "
            "eligible sites (10,000 site-bootstrap, seed 42). Pooled 90% conditioned "
            "envelope covers 25.7% of flux-derived ETf; this is an empirical coverage "
            "diagnostic and must never be shown as a predictive interval."
        ),
        cohort_summary=summary.to_dict("records"),
        **base,
    )
    MANIFEST.add(
        "fig04_weighting_effects.csv",
        rows=n_w,
        display_transformations=[
            "legacy arm labels e1_*/e2_* mapped to spread-weighted / fixed-scale (0.33 ETf) arms",
            "legacy metric label r2 renamed nse after evaluator verification; abs_bias renamed abs_mbe",
            "delta_abs_mbe recomputed at site level as |MBE_spread| - |MBE_fixed|",
        ],
        note=(
            "record_type=cohort_effect rows carry the frozen 10,000-resample site-bootstrap "
            "95% intervals (seed 42); record_type=site_delta rows carry the raw paired site "
            "values. Arms differ only in the weight denominator (sigma_ensemble+0.1 versus "
            "fixed 0.33 ETf). Objective-function values are not comparable across arms."
        ),
        **base,
    )
    MANIFEST.add(
        "fig04_example_selection.json",
        rows=None,
        selected_site=chosen_site,
        n_selected_captures=len(caps),
        note="Frozen Section 8 panel (a) selection rule; no visual inspection entered the choice.",
        cohort_key="site_id",
        inclusion_rule=base["inclusion_rule"],
        deterministic_seed=None,
    )
    print(
        f"  fig04: captures {n_cap}, quintiles {n_q}, associations {n_assoc}, "
        f"conditioned {n_cond}, weighting {n_w}; example site {chosen_site} ({len(caps)} captures)"
    )


def _e1_member_etf(obs: pd.DataFrame) -> pd.DataFrame:
    import zarr

    z = zarr.open(str(E1_CONTAINER), mode="r")
    uid = [str(x) for x in z["geometry/uid"][:]]
    t = pd.to_datetime(z["time/daily"][:])
    tpos = pd.Series(np.arange(len(t)), index=t)
    arrays = {
        m: np.asarray(z[f"remote_sensing/etf/landsat/{m}/no_mask"][:], dtype=float)
        for m in E1_MEMBERS
    }
    rows = []
    for site, grp in obs.groupby("site"):
        j = uid.index(site)
        idx = tpos.reindex(grp["date"]).to_numpy()
        if np.isnan(idx).any():
            raise BuildError(f"fig04: capture date outside container time axis at {site}")
        idx = idx.astype(int)
        rec = {"site": site, "date": grp["date"].values}
        for m in E1_MEMBERS:
            rec[f"etf_{m}"] = arrays[m][idx, j]
        rows.append(pd.DataFrame(rec))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# Figure 5 -- parameter transfer
# --------------------------------------------------------------------------

E1_ARMS = {
    "default": ("generic_defaults", "uncalibrated generic prior parameter set"),
    "loro_strat": (
        "heldout_transfer_class_specific_loro",
        "irrigation-class-specific leave-region-out transfer (primary)",
    ),
    "loso_strat": (
        "heldout_transfer_class_specific_loso",
        "irrigation-class-specific leave-one-site-out transfer (fold-granularity sensitivity)",
    ),
    "loro": (
        "heldout_transfer_pooled_loro",
        "superseded pooled leave-region-out transfer (provenance only)",
    ),
    "loso": (
        "heldout_transfer_pooled_loso",
        "superseded pooled leave-one-site-out transfer (provenance only)",
    ),
    "local": ("local_calibration", "Run 22 local site calibration"),
}


def build_fig05_e1() -> None:
    src_d = E1_WITHIN_STRAT / "persite_daily.csv"
    src_m = E1_WITHIN_STRAT / "persite_monthly.csv"
    src_sum = FINAL / "e2_irrigation_stratified_transfer_summary.csv"
    src_mad = FINAL / "e2_irrigation_stratified_fold_mad_domain.csv"
    src_fold = E1_WITHIN_STRAT / "class_fold_support.csv"
    for p in (src_d, src_m, src_sum, src_mad, src_fold):
        if not p.exists():
            raise BuildError(f"fig05 E1 source missing: {p}")

    d = pd.read_csv(src_d)
    m = pd.read_csv(src_m)
    require_unique(d, ["fid"], "E1 transfer persite_daily")
    require_unique(m, ["fid"], "E1 transfer persite_monthly")
    require_count(len(d), EXPECTED["E1_transfer_daily"], "fig05 E1 daily cohort")
    m = m[m["local_kge"].notna()].copy()
    require_count(len(m), EXPECTED["E1_transfer_monthly"], "fig05 E1 monthly cohort")

    mad = pd.read_csv(src_mad)
    fold = pd.read_csv(src_fold)

    rows = []
    for scale, df in (("daily", d), ("monthly", m)):
        for arm, (name, prov) in E1_ARMS.items():
            need = [f"{arm}_{k}" for k in ("n", "r2", "kge", "rmse", "bias")]
            require_columns(df, need, f"E1 transfer {scale} arm {arm}")
            block = pd.DataFrame(
                {
                    "experiment": "E1",
                    "legacy_prefix": "e2_",
                    "scale": scale,
                    "site_id": df["fid"],
                    "region": df["region"],
                    "irrigation_class": df["irr_class"],
                    "treatment": name,
                    "legacy_arm": arm,
                    "treatment_provenance": prov,
                    "n_paired": df[f"{arm}_n"],
                    "nse": df[f"{arm}_r2"],
                    "kge": df[f"{arm}_kge"],
                    "rmse": df[f"{arm}_rmse"],
                    "mbe": df[f"{arm}_bias"],
                    "r": df[f"{arm}_r"],
                }
            )
            for metric, col in (
                ("nse", "r2"),
                ("kge", "kge"),
                ("rmse", "rmse"),
            ):
                block[f"d_{metric}_vs_local"] = df[f"{arm}_{col}"] - df[f"local_{col}"]
                block[f"d_{metric}_vs_default"] = df[f"{arm}_{col}"] - df[f"default_{col}"]
            block["d_abs_mbe_vs_local"] = df[f"{arm}_bias"].abs() - df["local_bias"].abs()
            block["d_abs_mbe_vs_default"] = df[f"{arm}_bias"].abs() - df["default_bias"].abs()
            rows.append(block)
    site = pd.concat(rows, ignore_index=True)

    madm = mad.set_index(["arm", "fid"])
    site["fold_mad"] = [
        madm["mad"].get((a, f), np.nan) for a, f in zip(site["legacy_arm"], site["site_id"])
    ]
    site["fold_mad_in_class_prior"] = [
        madm["mad_in_class_prior"].get((a, f), np.nan)
        for a, f in zip(site["legacy_arm"], site["site_id"])
    ]
    foldm = fold.set_index("fid")
    site["n_train_loro"] = site["site_id"].map(foldm["n_train_loro"])
    site["n_train_loro_strat"] = site["site_id"].map(foldm["n_train_loro_strat"])

    summ = pd.read_csv(src_sum)
    summ = summ[summ["experiment"] == "E2_within_held_out"].copy()
    summ["treatment"] = summ["config"].map(lambda c: E1_ARMS.get(c, (c, ""))[0])
    summ_out = pd.DataFrame(
        {
            "experiment": "E1",
            "legacy_prefix": "e2_",
            "scale": summ["basis"],
            "site_id": None,
            "region": None,
            "irrigation_class": summ["stratum"],
            "treatment": summ["treatment"],
            "legacy_arm": summ["config"],
            "treatment_provenance": summ["label"],
            "record_type": "cohort_summary",
            "metric": summ["metric"].map(
                {"r2": "nse", "kge": "kge", "rmse": "rmse", "bias": "mbe", "abs_bias": "abs_mbe"}
            ),
            "median_common": summ["median_common"],
            "ci95_lo": summ["ci_lo"],
            "ci95_hi": summ["ci_hi"],
            "n_common": summ["n_common"],
        }
    )
    if summ_out["metric"].isna().any():
        raise BuildError("fig05 E1: unmapped summary metric label")
    site["record_type"] = "site_metric"
    out = pd.concat([site, summ_out], ignore_index=True)
    n = write_table(out, "fig05_e1_heldout_transfer.csv")

    MANIFEST.add(
        "fig05_e1_heldout_transfer.csv",
        rows=n,
        sources={
            k: {"path": str(p), "sha256": sha256(p)}
            for k, p in {
                "within_e1_stratified_persite_daily": src_d,
                "within_e1_stratified_persite_monthly": src_m,
                "e2_irrigation_stratified_transfer_summary": src_sum,
                "e2_irrigation_stratified_fold_mad_domain": src_mad,
                "class_fold_support": src_fold,
            }.items()
        },
        experiment_mapping={"E1": "legacy e2_*; legacy experiment label E2_within_held_out"},
        cohort_key="site_id",
        inclusion_rule=(
            "Common paired cohort of the Run 22 45-site daily and 31-site monthly "
            "evaluations, evaluated identically under all six arms. 29 irrigated / 16 "
            "rainfed daily sites. Held-out folds exclude each evaluated site from the "
            "parameter set applied to it; the class-specific vector for each fold is the "
            "median of per-site posterior medians within the site's irrigation class."
        ),
        temporal_support_rule="Daily common-support days; monthly complete calendar months with a 10-month finite-metric floor.",
        units={"rmse_daily": "mm d-1", "rmse_monthly": "mm month-1", "mbe": "mm d-1 / mm month-1"},
        display_transformations=[
            "legacy arm labels mapped to reader-facing treatment names (see treatment/legacy_arm columns)",
            "legacy r2 -> nse; legacy bias -> mbe",
            "paired site deltas computed against the local-calibration and generic-default arms",
            "fold-level mad prior legality joined from the frozen fold_mad_domain table",
        ],
        deterministic_seed=1234,
        bootstrap="paired site bootstrap, 2,000 draws, seed 1234 (frozen upstream harness default)",
        configured_counts={"E1": 60},
        evaluated_counts={"daily_sites": 45, "monthly_sites": 31, "irrigated": 29, "rainfed": 16},
        note=(
            "Primary reader-facing treatment is heldout_transfer_class_specific_loro. The "
            "pooled LORO/LOSO arms are retained with treatment_provenance marking them "
            "superseded; they must not be plotted as coequal main treatments."
        ),
    )
    print(f"  fig05 E1: {n} rows (45 daily + 31 monthly sites x 6 arms + cohort summaries)")


E2_CONFIGS = {
    "e3_uncal": ("generic_defaults", "E2 uncalibrated default parameters"),
    "ex5_transfer_strat": (
        "e1_irrigation_class_transfer",
        "fixed E1-derived irrigated/rainfed parameter sets assigned by the E2 equipped-for-irrigation classification",
    ),
    "e3_cal": ("local_calibration", "local E2 satellite calibration (ls_ensemble_por_annual2yr)"),
    "ls_ensemble": (
        "landsat_benchmark",
        "interpolated coincident Landsat SSEBop + PT-JPL benchmark",
    ),
}


def build_fig05_e2() -> None:
    src_ps = E2_TRANSFER / "transfer_comparison_persite.csv"
    src_sum = E2_TRANSFER / "transfer_comparison_summary.csv"
    src_mstrat = E2_TRANSFER / "evaluation_monthly_metrics_strat.csv"
    src_mcal = E2_RESULTS / "evaluation_monthly_metrics.csv"
    src_persite_strat = FINAL / "e3_irrigation_stratified_transfer_persite_daily.csv"
    src_boot = FINAL / "e3_irrigation_stratified_transfer_bootstrap.csv"
    src_e3sum = FINAL / "e3_irrigation_stratified_transfer_summary.csv"
    for p in (src_ps, src_sum, src_mstrat, src_mcal, src_persite_strat, src_boot, src_e3sum):
        if not p.exists():
            raise BuildError(f"fig05 E2 source missing: {p}")

    ps = pd.read_csv(src_ps, index_col=0)
    ps.index.name = "site_id"
    require_count(len(ps), EXPECTED["E2_daily"], "fig05 E2 daily cohort")

    rows = []
    for cfg, (name, prov) in E2_CONFIGS.items():
        need = [f"{cfg}_{k}" for k in ("kge", "r2", "rmse", "bias", "mae")]
        require_columns(ps.reset_index(), need, f"E2 persite config {cfg}")
        rows.append(
            pd.DataFrame(
                {
                    "experiment": "E2",
                    "legacy_prefix": "e3_",
                    "record_type": "site_metric",
                    "scale": "daily",
                    "site_id": ps.index,
                    "treatment": name,
                    "legacy_config": cfg,
                    "treatment_provenance": prov,
                    "nse": ps[f"{cfg}_r2"].values,
                    "kge": ps[f"{cfg}_kge"].values,
                    "rmse": ps[f"{cfg}_rmse"].values,
                    "mbe": ps[f"{cfg}_bias"].values,
                    "mae": ps[f"{cfg}_mae"].values,
                }
            )
        )
    daily = pd.concat(rows, ignore_index=True)

    # monthly: per-site available for three of four treatments
    mstrat = pd.read_csv(src_mstrat)
    mcal = pd.read_csv(src_mcal)
    require_count(len(mstrat), EXPECTED["E2_monthly_support"], "fig05 E2 monthly strat")
    require_count(len(mcal), EXPECTED["E2_monthly_support"], "fig05 E2 monthly cal")
    if set(mstrat["fid"]) != set(mcal["fid"]):
        raise BuildError("fig05 E2 monthly: site sets differ between transfer and canonical runs")

    def _m(df, suffix, name, prov):
        return pd.DataFrame(
            {
                "experiment": "E2",
                "legacy_prefix": "e3_",
                "record_type": "site_metric",
                "scale": "monthly",
                "site_id": df["fid"],
                "treatment": name,
                "legacy_config": suffix,
                "treatment_provenance": prov,
                "nse": df[f"r2_{suffix}"],
                "kge": df[f"kge_{suffix}"],
                "rmse": df[f"rmse_{suffix}"],
                "mbe": df[f"bias_{suffix}"],
                "mae": np.nan,
                "n_paired": df["n"],
                "finite_metric": df[f"kge_{suffix}"].notna(),
            }
        )

    monthly = pd.concat(
        [
            _m(mstrat, "swim", *E2_CONFIGS["ex5_transfer_strat"]),
            _m(mcal, "swim", *E2_CONFIGS["e3_cal"]),
            _m(mcal, "rs", *E2_CONFIGS["ls_ensemble"]),
        ],
        ignore_index=True,
    )
    n_finite = int(monthly[monthly["treatment"] == "local_calibration"]["finite_metric"].sum())
    require_count(n_finite, EXPECTED["E2_monthly_finite"], "fig05 E2 monthly finite-metric")

    summ = pd.read_csv(src_sum)
    label_to_cfg = {
        "E3 uncalibrated/default": "e3_uncal",
        "Ex5 stratified transfer": "ex5_transfer_strat",
        "E3 calibrated": "e3_cal",
        "LS ensemble": "ls_ensemble",
    }
    summ = summ[summ["config"].isin(label_to_cfg)].copy()
    summ["legacy_config"] = summ["config"].map(label_to_cfg)
    summ_out = pd.DataFrame(
        {
            "experiment": "E2",
            "legacy_prefix": "e3_",
            "record_type": "cohort_summary",
            "scale": summ["basis"],
            "site_id": None,
            "treatment": summ["legacy_config"].map(lambda c: E2_CONFIGS[c][0]),
            "legacy_config": summ["legacy_config"],
            "treatment_provenance": summ["config"],
            "nse": summ["r2_med"],
            "kge": summ["kge_med"],
            "rmse": summ["rmse_med"],
            "mbe": summ["bias_med"],
            "mae": summ["mae_med"],
            "n_paired": summ["n_sites_common"],
            "alpha": summ["alpha_med"],
            "beta": summ["beta_med"],
        }
    )

    out = pd.concat([daily, monthly, summ_out], ignore_index=True)
    n = write_table(out, "fig05_e2_cross_environment_transfer.csv")

    # geography / class lookup, over the full configured 66
    cfg66 = e2_configured()
    strat = pd.read_csv(src_persite_strat).set_index("fid")
    cfg66["conus"] = cfg66["site_id"].map(strat["conus"])
    cfg66["conus"] = np.where(
        cfg66["conus"].notna(),
        cfg66["conus"],
        (cfg66["lon"].between(-125, -66)) & (cfg66["lat"].between(24, 50)),
    ).astype(bool)
    cfg66["in_daily_cohort"] = cfg66["site_id"].isin(ps.index)
    cfg66["in_monthly_cohort"] = cfg66["site_id"].isin(set(mcal["fid"]))
    e1_sites = set(e1_configured()["site_id"])
    cfg66["also_in_e1"] = cfg66["site_id"].isin(e1_sites)
    require_count(int(cfg66["also_in_e1"].sum()), EXPECTED["E1_E2_overlap"], "E1/E2 overlap")
    require_count(int(cfg66["in_daily_cohort"].sum()), EXPECTED["E2_daily"], "E2 daily cohort flag")
    n_conus = int(cfg66.loc[cfg66["in_daily_cohort"], "conus"].sum())
    n_ex = int((~cfg66.loc[cfg66["in_daily_cohort"], "conus"]).sum())
    require_count(n_conus, 42, "E2 daily CONUS split")
    require_count(n_ex, 21, "E2 daily ex-CONUS split")
    cfg66.insert(0, "legacy_prefix", "e3_")
    cfg66.insert(0, "experiment", "E2")
    n_lu = write_table(cfg66, "fig05_geography_lookup.csv")

    MANIFEST.add(
        "fig05_e2_cross_environment_transfer.csv",
        rows=n,
        sources={
            k: {"path": str(p), "sha256": sha256(p)}
            for k, p in {
                "transfer_comparison_persite": src_ps,
                "transfer_comparison_summary": src_sum,
                "evaluation_monthly_metrics_strat": src_mstrat,
                "evaluation_monthly_metrics_canonical": src_mcal,
                "e3_irrigation_stratified_transfer_bootstrap": src_boot,
                "e3_irrigation_stratified_transfer_summary": src_e3sum,
            }.items()
        },
        experiment_mapping={"E2": "legacy e3_*"},
        cohort_key="site_id",
        inclusion_rule=(
            "66 configured E2 sites; 63 with >=10 paired daily observations form the daily "
            "common cohort; 56 have paired monthly support and 50 of those meet the "
            "10-month finite-metric criterion. All four treatments are scored on identical "
            "common-support days."
        ),
        temporal_support_rule="Daily common-support days 2013-2025; monthly are sums of paired days (>=20 paired days per month, >=6 paired months).",
        units={"rmse_daily": "mm d-1", "rmse_monthly": "mm month-1", "mbe": "mm d-1 / mm month-1"},
        display_transformations=[
            "legacy config e3_uncal -> generic_defaults",
            "legacy config ex5_transfer_strat -> e1_irrigation_class_transfer",
            "legacy config e3_cal -> local_calibration",
            "legacy config ls_ensemble -> landsat_benchmark",
            "legacy r2 -> nse; legacy bias -> mbe",
            "superseded pooled transfer (ex5_transfer) deliberately excluded from the display package",
        ],
        deterministic_seed=1234,
        configured_counts={"E2": 66},
        evaluated_counts={
            "daily_sites": 63,
            "monthly_support_sites": 56,
            "monthly_finite_metric_sites": 50,
            "conus_daily": 42,
            "ex_conus_daily": 21,
            "also_in_e1": 13,
        },
        known_gap=(
            "Per-site MONTHLY metrics for the generic-defaults treatment are not persisted "
            "by transfer_ex5_params.py (the uncalibrated arm is a fresh in-process forward "
            "run). Only the cohort median is available and is frozen as a record_type="
            "cohort_summary row. Site-level monthly distributions can therefore be drawn "
            "for three of four treatments."
        ),
    )
    MANIFEST.add(
        "fig05_geography_lookup.csv",
        rows=n_lu,
        sources={
            "e2_container_geometry": {"path": str(E2_CONTAINER), "arrays": ["geometry/*"]},
            "e3_irrigation_stratified_param_mapping_metadata": {
                "path": str(FINAL / "e3_irrigation_stratified_param_mapping_metadata.json"),
                "sha256": sha256(FINAL / "e3_irrigation_stratified_param_mapping_metadata.json"),
            },
            "e3_irrigation_stratified_transfer_persite_daily": {
                "path": str(src_persite_strat),
                "sha256": sha256(src_persite_strat),
            },
        },
        experiment_mapping={"E2": "legacy e3_*"},
        cohort_key="site_id",
        inclusion_rule="All 66 configured E2 sites, flagged for daily/monthly cohort membership.",
        display_transformations=[
            "CONUS flag taken from the frozen stratified per-site table where present; otherwise derived from the container lon/lat bounding box (-125..-66 E, 24..50 N)",
            "equipped_for_irrigation is stage 1 of the two-stage classifier, recovered per the frozen param-mapping metadata",
        ],
        units={"lat": "degrees north", "lon": "degrees east"},
        deterministic_seed=None,
        configured_counts={"E2": 66},
        evaluated_counts={"daily": 63, "conus_daily": 42, "ex_conus_daily": 21, "equipped": 13},
    )
    print(
        f"  fig05 E2: {n} rows (63 daily x 4 treatments + 56 monthly x 3 + 8 cohort summaries), "
        f"lookup {n_lu} rows (42 CONUS / 21 ex-CONUS daily; 13 also in E1)"
    )


# --------------------------------------------------------------------------
# Figure 6 -- metered applied water
# --------------------------------------------------------------------------

E3_TREATMENTS = {
    "local_calibration": (E3_LOCAL, "E3 local satellite calibration (e7cal)"),
    "e1_irrigated_transfer": (
        E3_TRANSFER,
        "fixed E1-derived irrigated parameter set, forward run, no local calibration",
    ),
}


def _load_e3_paths() -> dict[str, pd.DataFrame]:
    frames = {}
    for name, (d, _) in E3_TREATMENTS.items():
        p = d / "per_field_year.csv"
        if not p.exists():
            raise BuildError(f"fig06 source missing: {p}")
        df = pd.read_csv(p)
        require_columns(
            df,
            [
                "site_id",
                "year",
                "metered_depth_mm",
                "sim_applied_mm",
                "sim_et_mm",
                "acres",
                "crop",
                "basin",
            ],
            f"E3 {name}",
        )
        df = df[df["site_id"].astype(str).str.startswith("SLV_")].copy()
        require_unique(df, ["site_id", "year"], f"E3 {name} SLV")
        df = df.sort_values(["site_id", "year"]).reset_index(drop=True)
        frames[name] = df
    keys = [set(zip(f["site_id"], f["year"])) for f in frames.values()]
    if keys[0] != keys[1]:
        raise BuildError("fig06: (site_id, year) keys differ between local and transfer paths")
    for name, f in frames.items():
        require_count(len(f), EXPECTED["E3_field_years"], f"fig06 {name} field-years")
        require_count(f["site_id"].nunique(), EXPECTED["E3_fields"], f"fig06 {name} fields")
    a, b = frames["local_calibration"], frames["e1_irrigated_transfer"]
    if not np.allclose(a["metered_depth_mm"].values, b["metered_depth_mm"].values):
        raise BuildError("fig06: metered truth differs between the two evaluator outputs")
    return frames


def build_fig06() -> pd.DataFrame:
    frames = _load_e3_paths()
    blocks = []
    for name, df in frames.items():
        d = df.copy()
        d["field_mean_metered_mm"] = d.groupby("site_id")["metered_depth_mm"].transform("mean")
        d["field_mean_sim_mm"] = d.groupby("site_id")["sim_applied_mm"].transform("mean")
        d["anomaly_metered_mm"] = d["metered_depth_mm"] - d["field_mean_metered_mm"]
        d["anomaly_sim_mm"] = d["sim_applied_mm"] - d["field_mean_sim_mm"]
        d["treatment"] = name
        d["treatment_provenance"] = E3_TREATMENTS[name][1]
        d["experiment"] = "E3"
        d["legacy_prefix"] = "e4_"
        blocks.append(d)
    fy = pd.concat(blocks, ignore_index=True)
    cols = [
        "experiment",
        "legacy_prefix",
        "site_id",
        "year",
        "treatment",
        "treatment_provenance",
        "crop",
        "basin",
        "acres",
        "metered_depth_mm",
        "sim_applied_mm",
        "sim_et_mm",
        "field_mean_metered_mm",
        "field_mean_sim_mm",
        "anomaly_metered_mm",
        "anomaly_sim_mm",
    ]
    fy = fy[cols].sort_values(["treatment", "site_id", "year"])
    n_fy = write_table(fy, "fig06_field_years.csv")
    require_count(n_fy, EXPECTED["E3_field_years"] * 2, "fig06 field-year rows")

    summaries = []
    for name, df in frames.items():
        g = df.groupby("site_id")
        s = pd.DataFrame(
            {
                "n_years": g.size(),
                "mean_metered_mm": g["metered_depth_mm"].mean(),
                "mean_sim_mm": g["sim_applied_mm"].mean(),
                "total_metered_mm": g["metered_depth_mm"].sum(),
                "total_sim_mm": g["sim_applied_mm"].sum(),
                "acres": g["acres"].first(),
                "crop": g["crop"].first(),
            }
        )
        s["field_bias_pct"] = (
            (s["total_sim_mm"] - s["total_metered_mm"]) / s["total_metered_mm"] * 100.0
        )
        s["within_20pct"] = s["field_bias_pct"].abs() <= 20.0
        s["field_temporal_r"] = g.apply(
            lambda x: np.corrcoef(x["metered_depth_mm"], x["sim_applied_mm"])[0, 1]
            if len(x) > 2
            else np.nan,
            include_groups=False,
        )
        s = s.reset_index()
        s["treatment"] = name
        s["treatment_provenance"] = E3_TREATMENTS[name][1]
        s["experiment"] = "E3"
        s["legacy_prefix"] = "e4_"
        summaries.append(s)
    fs = pd.concat(summaries, ignore_index=True)
    n_fs = write_table(fs, "fig06_field_summaries.csv")
    require_count(n_fs, EXPECTED["E3_fields"] * 2, "fig06 field-summary rows")

    srcs = {
        "applied_calibrated_per_field_year": E3_LOCAL / "per_field_year.csv",
        "applied_transfer_run22_by_irrigation_per_field_year": E3_TRANSFER / "per_field_year.csv",
    }
    base = {
        "sources": {k: {"path": str(p), "sha256": sha256(p)} for k, p in srcs.items()},
        "experiment_mapping": {"E3": "legacy e4_*"},
        "cohort_key": "(site_id, year)",
        "inclusion_rule": (
            "50 San Luis Valley metered fields (site_id prefix SLV_) and their 408 paired "
            "field-years with a positive recorded pumping volume, 2011-2021. The local and "
            "transfer evaluator outputs were asserted to carry identical (site_id, year) "
            "keys and identical metered truth before joining. metered_truth.csv is never "
            "read directly. The excluded ESPA basin and the 10 rainfed control fields are "
            "not part of this package."
        ),
        "temporal_support_rule": "Annual: model-generated daily irrigation summed by calendar year and paired with positive meter observations on identical field-year keys.",
        "units": {
            "metered_depth_mm": "mm year-1",
            "sim_applied_mm": "mm year-1 gross applied water",
            "sim_et_mm": "mm year-1",
            "acres": "acres",
            "field_bias_pct": "percent",
        },
        "display_transformations": [
            "restricted to SLV_ fields",
            "within-field anomalies computed by removing each field's record mean from both the metered and simulated annual depths",
            "field_bias_pct is a record-total bias, (sum(sim) - sum(metered)) / sum(metered) * 100",
        ],
        "deterministic_seed": None,
        "configured_counts": {"E3": 50},
        "evaluated_counts": {"fields": 50, "field_years": 408},
        "independent_unit": "field (not field-year)",
    }
    MANIFEST.add("fig06_field_years.csv", rows=n_fy, **base)
    MANIFEST.add("fig06_field_summaries.csv", rows=n_fs, **base)
    print(
        f"  fig06: field-years {n_fy} rows, field summaries {n_fs} rows (50 fields, 408 field-years)"
    )
    return fy


# ---- panel (d): new prespecified whole-field bootstrap ---------------------

BOOTSTRAP_STATS = {
    "between_field_mean_depth_r": {
        "definition": "Pearson r between per-field record-mean metered depth and per-field record-mean simulated depth across the resampled fields",
        "group": "cross-field magnitude",
        "favorable_direction": "positive favours the transfer path",
    },
    "pooled_annual_depth_r": {
        "definition": "Pearson r between metered and simulated annual depths pooled over all field-years in the resample",
        "group": "cross-field magnitude",
        "favorable_direction": "positive favours the transfer path",
    },
    "pooled_rmse_mm": {
        "definition": "sqrt(mean((sim - metered)^2)) over pooled field-years, mm year-1",
        "group": "cross-field magnitude",
        "favorable_direction": "negative favours the transfer path",
    },
    "within_field_anomaly_r": {
        "definition": "Pearson r between metered and simulated within-field anomalies (each field's record mean removed within the resample) pooled over field-years",
        "group": "within-field response",
        "favorable_direction": "positive favours the transfer path",
    },
    "within_field_anomaly_slope": {
        "definition": "ordinary least-squares slope of simulated anomaly on metered anomaly, pooled over field-years",
        "group": "within-field response",
        "favorable_direction": "positive moves the slope toward one; sign retained",
    },
    "pooled_abs_bias_pct": {
        "definition": "absolute value of pooled percent bias, |sum(sim) - sum(metered)| / sum(metered) * 100",
        "group": "bias and coverage",
        "favorable_direction": "negative favours the transfer path",
    },
    "median_abs_field_bias_pct": {
        "definition": "median across resampled fields of |record-total field bias| in percent",
        "group": "bias and coverage",
        "favorable_direction": "negative favours the transfer path",
    },
    "fraction_within_20pct": {
        "definition": "fraction of resampled fields whose record-total percent bias lies within +/-20%",
        "group": "bias and coverage",
        "favorable_direction": "positive favours the transfer path",
    },
    "median_field_temporal_r": {
        "definition": "median across resampled fields of the within-field Pearson r between annual metered and simulated depths (fields with >2 years only)",
        "group": "within-field response",
        "favorable_direction": "positive favours the transfer path",
    },
}


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    return float((xc * yc).sum() / den) if den > 0 else np.nan


def _e3_statistics(obs: np.ndarray, sim: np.ndarray, field: np.ndarray) -> dict:
    """Nine prespecified applied-water statistics.

    ``field`` may be labels of any dtype; it is factorised to contiguous
    integers so that a bootstrap resample can relabel a repeated field as two
    distinct inferential units.
    """
    lab, _ = pd.factorize(field, sort=False)
    k = lab.max() + 1
    n = np.bincount(lab, minlength=k).astype(float)
    so = np.bincount(lab, weights=obs, minlength=k)
    ss = np.bincount(lab, weights=sim, minlength=k)
    soo = np.bincount(lab, weights=obs * obs, minlength=k)
    sss = np.bincount(lab, weights=sim * sim, minlength=k)
    sos = np.bincount(lab, weights=obs * sim, minlength=k)
    mo, ms = so / n, ss / n

    out = {
        "between_field_mean_depth_r": _pearson(mo, ms),
        "pooled_annual_depth_r": _pearson(obs, sim),
        "pooled_rmse_mm": float(np.sqrt(np.mean((sim - obs) ** 2))),
    }
    ao = obs - mo[lab]
    as_ = sim - ms[lab]
    out["within_field_anomaly_r"] = _pearson(ao, as_)
    den = (ao * ao).sum()
    out["within_field_anomaly_slope"] = float((ao * as_).sum() / den) if den > 0 else np.nan

    fb = (ss - so) / so * 100.0
    out["pooled_abs_bias_pct"] = float(abs((sim.sum() - obs.sum()) / obs.sum() * 100.0))
    out["median_abs_field_bias_pct"] = float(np.median(np.abs(fb)))
    out["fraction_within_20pct"] = float(np.mean(np.abs(fb) <= 20.0))

    num = n * sos - so * ss
    dsq = (n * soo - so * so) * (n * sss - ss * ss)
    with np.errstate(invalid="ignore", divide="ignore"):
        tr = np.where(dsq > 0, num / np.sqrt(dsq), np.nan)
    tr = np.where(n > 2, tr, np.nan)
    out["median_field_temporal_r"] = float(np.nanmedian(tr))
    return out


def build_fig06_bootstrap() -> None:
    frames = _load_e3_paths()
    a = frames["local_calibration"].set_index(["site_id", "year"]).sort_index()
    b = frames["e1_irrigated_transfer"].set_index(["site_id", "year"]).sort_index()
    if not a.index.equals(b.index):
        raise BuildError("fig06 bootstrap: index mismatch after alignment")

    obs = a["metered_depth_mm"].to_numpy(float)
    sim_local = a["sim_applied_mm"].to_numpy(float)
    sim_transfer = b["sim_applied_mm"].to_numpy(float)
    fields = a.index.get_level_values(0).to_numpy()
    uniq = np.array(sorted(set(fields)))
    require_count(len(uniq), EXPECTED["E3_fields"], "fig06 bootstrap fields")
    require_count(len(obs), EXPECTED["E3_field_years"], "fig06 bootstrap field-years")

    idx_by_field = {f: np.flatnonzero(fields == f) for f in uniq}

    point_local = _e3_statistics(obs, sim_local, fields)
    point_transfer = _e3_statistics(obs, sim_transfer, fields)
    point_delta = {k: point_transfer[k] - point_local[k] for k in BOOTSTRAP_STATS}

    n_boot, seed = 10000, 42
    rng = np.random.default_rng(seed)
    reps = {k: np.empty(n_boot) for k in BOOTSTRAP_STATS}
    for i in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        take = np.concatenate([idx_by_field[f] for f in draw])
        # relabel so a field drawn twice contributes as two distinct fields
        lab = np.concatenate([np.full(len(idx_by_field[f]), j) for j, f in enumerate(draw)])
        sl = _e3_statistics(obs[take], sim_local[take], lab)
        st = _e3_statistics(obs[take], sim_transfer[take], lab)
        for k in BOOTSTRAP_STATS:
            reps[k][i] = st[k] - sl[k]

    rows = []
    for k, meta in BOOTSTRAP_STATS.items():
        r = reps[k]
        rows.append(
            {
                "experiment": "E3",
                "legacy_prefix": "e4_",
                "statistic": k,
                "statistic_group": meta["group"],
                "statistic_definition": meta["definition"],
                "favorable_direction": meta["favorable_direction"],
                "value_local_calibration": point_local[k],
                "value_e1_irrigated_transfer": point_transfer[k],
                "delta_transfer_minus_local": point_delta[k],
                "bootstrap_median_delta": float(np.median(r)),
                "ci95_lo": float(np.percentile(r, 2.5)),
                "ci95_hi": float(np.percentile(r, 97.5)),
                "excludes_zero": bool(np.percentile(r, 2.5) > 0 or np.percentile(r, 97.5) < 0),
                "n_resamples": n_boot,
                "seed": seed,
                "resample_unit": "whole field (all years retained per sampled field)",
                "n_fields": len(uniq),
                "n_field_years": len(obs),
            }
        )
    out = pd.DataFrame(rows)
    n = write_table(out, "fig06_bootstrap_effects.csv")

    MANIFEST.add(
        "fig06_bootstrap_effects.csv",
        rows=n,
        sources={
            "applied_calibrated_per_field_year": {
                "path": str(E3_LOCAL / "per_field_year.csv"),
                "sha256": sha256(E3_LOCAL / "per_field_year.csv"),
            },
            "applied_transfer_run22_by_irrigation_per_field_year": {
                "path": str(E3_TRANSFER / "per_field_year.csv"),
                "sha256": sha256(E3_TRANSFER / "per_field_year.csv"),
            },
        },
        experiment_mapping={"E3": "legacy e4_*"},
        cohort_key="(site_id, year); resample unit = site_id",
        inclusion_rule=(
            "Identical 50-field / 408-field-year SLV keys under both treatments. This is a "
            "NEW comparison of local satellite calibration against the current E1-derived "
            "IRRIGATED parameter set. The archived applied_local_vs_transfer_run22 "
            "bootstrap compares local calibration with the SUPERSEDED pooled transfer "
            "vector and is deliberately not used."
        ),
        temporal_support_rule="Annual gross applied water, 2011-2021, positive meter records only.",
        units={
            "pooled_rmse_mm": "mm year-1",
            "pooled_abs_bias_pct": "percent",
            "median_abs_field_bias_pct": "percent",
            "fraction_within_20pct": "fraction",
            "correlations_and_slope": "dimensionless",
        },
        display_transformations=[
            "all statistics reported as transfer minus local with natural signs retained",
            "no metric was sign-flipped to make favourable results point the same way",
        ],
        deterministic_seed=seed,
        bootstrap=(
            "Whole-field resampling with replacement, all years retained for each sampled "
            "field, 10,000 resamples, numpy default_rng(42). A field drawn more than once "
            "is relabelled so that it contributes as separate fields to the field-level "
            "statistics. Design frozen before any interval was inspected."
        ),
        configured_counts={"E3": 50},
        evaluated_counts={"fields": 50, "field_years": 408},
        independent_unit="field",
    )
    print(f"  fig06 bootstrap: {n} statistics, 10,000 whole-field resamples, seed 42")
    for _, r in out.iterrows():
        print(
            f"    {r['statistic']:<32} delta {r['delta_transfer_minus_local']:+.4f} "
            f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}]"
            f"{'  *' if r['excludes_zero'] else ''}"
        )


# --------------------------------------------------------------------------
# Supplementary observation-support diagnostics (Section 4.1)
# --------------------------------------------------------------------------


def _capture_dates_e1() -> pd.DataFrame:
    om = pd.read_csv(E1_ARCHIVE / "3_problem_definition" / "observation_metadata.csv")
    om = om[om["model"] == "ensemble"].copy()
    om["date"] = pd.to_datetime(om["date"])
    return om[["site", "date", "member_count"]].rename(columns={"site": "site_id"})


def _capture_dates_e2() -> pd.DataFrame:
    import zarr

    z = zarr.open(str(E2_CONTAINER), mode="r")
    uid = [str(x) for x in z["geometry/uid"][:]]
    t = pd.to_datetime(z["time/daily"][:])
    ss = np.asarray(z["remote_sensing/etf/landsat/ssebop/no_mask"][:], dtype=float)
    pt = np.asarray(z["remote_sensing/etf/landsat/ptjpl/no_mask"][:], dtype=float)
    cohort = set(e2_configured()["site_id"])
    rows = []
    for j, s in enumerate(uid):
        if s not in cohort:
            continue
        ok = np.isfinite(ss[:, j]) & np.isfinite(pt[:, j])
        if not ok.any():
            continue
        rows.append(pd.DataFrame({"site_id": s, "date": t[ok], "member_count": 2}))
    return pd.concat(rows, ignore_index=True)


def _capture_dates_e3() -> pd.DataFrame:
    import zarr

    z = zarr.open(str(E3_CONTAINER), mode="r")
    uid = [str(x) for x in z["geometry/uid"][:]]
    t = pd.to_datetime(z["time/daily"][:])
    arrays = {
        m: np.asarray(z[f"remote_sensing/etf/landsat/{m}/no_mask"][:], dtype=float)
        for m in E1_MEMBERS
    }
    rows = []
    for j, s in enumerate(uid):
        if not s.startswith("SLV_"):
            continue
        cnt = np.sum(np.vstack([np.isfinite(arrays[m][:, j]) for m in E1_MEMBERS]), axis=0)
        ok = cnt >= 2
        rows.append(pd.DataFrame({"site_id": s, "date": t[ok], "member_count": cnt[ok]}))
    return pd.concat(rows, ignore_index=True)


def build_obs_support() -> None:
    caps = {
        "E1": _capture_dates_e1(),
        "E2": _capture_dates_e2(),
        "E3": _capture_dates_e3(),
    }
    expected_sites = {"E1": 60, "E2": 66, "E3": 50}
    etf_rows = []
    for exp, df in caps.items():
        require_count(df["site_id"].nunique(), expected_sites[exp], f"{exp} ETf support sites")
        df = df.sort_values(["site_id", "date"])
        df["year"] = df["date"].dt.year
        per_sy = df.groupby(["site_id", "year"]).size().rename("n_captures").reset_index()
        # site-year denominator: first through last retained target year per site
        spans = df.groupby("site_id")["year"].agg(["min", "max"])
        full = []
        for s, r in spans.iterrows():
            for y in range(int(r["min"]), int(r["max"]) + 1):
                full.append((s, y))
        full = pd.DataFrame(full, columns=["site_id", "year"])
        per_sy = full.merge(per_sy, on=["site_id", "year"], how="left").fillna({"n_captures": 0})
        gaps = (
            df.groupby("site_id")["date"]
            .apply(lambda x: x.sort_values().diff().dt.days.dropna())
            .reset_index(level=0)
        )
        gap_stats = (
            gaps.groupby("site_id")["date"]
            .agg(
                gap_median="median",
                gap_q25=lambda x: x.quantile(0.25),
                gap_q75=lambda x: x.quantile(0.75),
                gap_p90=lambda x: x.quantile(0.90),
                gap_frac_gt_16=lambda x: float((x > 16).mean()),
                gap_frac_gt_32=lambda x: float((x > 32).mean()),
                n_intervals="size",
            )
            .reset_index()
        )
        per_sy["experiment"] = exp
        per_sy["legacy_prefix"] = EXPERIMENT_MAP[exp]["legacy_prefix"]
        per_sy = per_sy.merge(gap_stats, on="site_id", how="left")
        site_tot = df.groupby("site_id").size().rename("n_captures_site").reset_index()
        per_sy = per_sy.merge(site_tot, on="site_id", how="left")
        etf_rows.append(per_sy)
    etf = pd.concat(etf_rows, ignore_index=True)
    etf["n_captures"] = etf["n_captures"].astype(int)
    n_etf = write_table(
        etf[
            [
                "experiment",
                "legacy_prefix",
                "site_id",
                "year",
                "n_captures",
                "n_captures_site",
                "gap_median",
                "gap_q25",
                "gap_q75",
                "gap_p90",
                "gap_frac_gt_16",
                "gap_frac_gt_32",
                "n_intervals",
            ]
        ],
        "figs02_etf_support.csv",
    )

    # member availability
    mem_rows = []
    for exp, df in caps.items():
        vc = df["member_count"].value_counts().sort_index()
        for k, v in vc.items():
            mem_rows.append(
                {
                    "experiment": exp,
                    "legacy_prefix": EXPERIMENT_MAP[exp]["legacy_prefix"],
                    "record_type": "valid_member_count",
                    "key": int(k),
                    "n_captures": int(v),
                    "fraction_of_captures": float(v / len(df)),
                    "n_captures_total": int(len(df)),
                }
            )
    for exp, container, members in (
        ("E1", E1_CONTAINER, E1_MEMBERS),
        ("E3", E3_CONTAINER, E1_MEMBERS),
        ("E2", E2_CONTAINER, ["ssebop", "ptjpl"]),
    ):
        contrib = _member_contribution(exp, container, members, caps[exp])
        mem_rows.extend(contrib)
    mem = pd.DataFrame(mem_rows)
    n_mem = write_table(mem, "figs02_member_availability.csv")

    # NDVI sensor contribution (Landsat-first merge rule)
    ndvi_rows = []
    for exp, container in (
        ("E1", E1_CONTAINER),
        ("E2", E2_CONTAINER),
        ("E3", E3_CONTAINER),
    ):
        ndvi_rows.append(_ndvi_support(exp, container, caps[exp]))
    ndvi = pd.concat(ndvi_rows, ignore_index=True)
    n_ndvi = write_table(ndvi, "figs02_ndvi_support.csv")

    # Pooled sanity checks against six_figure_plan.md section 4.2, which quotes
    # POOLED distributions.  The frozen tables are site-level by instruction, so
    # the pooled reconciliation is recorded here instead of in a table.
    pooled = {}
    for exp, df in caps.items():
        g = (
            df.sort_values(["site_id", "date"])
            .groupby("site_id")["date"]
            .apply(lambda x: x.diff().dt.days.dropna())
        )
        v = g.to_numpy(dtype=float)
        sy = df.assign(year=df["date"].dt.year).groupby(["site_id", "year"]).size()
        sub = ndvi[ndvi["experiment"] == exp]
        pooled[exp] = {
            "median_captures_per_site_year": float(sy.median()),
            "pooled_median_inter_capture_gap_days": float(np.median(v)),
            "pooled_p90_inter_capture_gap_days": float(np.percentile(v, 90)),
            "pooled_frac_landsat_selected_ndvi_full_record": float(
                sub["n_landsat_selected"].sum() / sub["n_raw_observation_dates"].sum()
            ),
            "pooled_frac_landsat_selected_ndvi_etf_window": float(
                sub["n_landsat_selected_etf_window"].sum()
                / sub["n_raw_observation_dates_etf_window"].sum()
            ),
            "site_median_raw_ndvi_interval_days_full_record": float(
                sub["raw_interval_median_days"].median()
            ),
            "site_median_raw_ndvi_interval_days_etf_window": float(
                sub["raw_interval_median_days_etf_window"].median()
            ),
        }
    pooled["_note"] = (
        "six_figure_plan.md section 4.2 quotes NDVI Landsat-selected fractions of "
        "40 / 49 / 66 percent and median raw NDVI intervals of 3 / 5 / 5 days for E1 / E2 / "
        "E3. Those three values were not computed on a single window: E1 reproduces exactly "
        "on the ETf target-support window (0.405, 3 d) while E2 and E3 reproduce exactly on "
        "the full container record (0.491 / 5 d and 0.658 / 5 d). Both windows are therefore "
        "frozen side by side in figs02_ndvi_support.csv so no downstream panel has to guess. "
        "The capture-density and inter-capture-gap feasibility numbers (31 / 18 / 55 captures "
        "per site-year; 8/23, 8/32, 7/15 day median/p90 gaps) reproduce exactly."
    )

    base = {
        "experiment_mapping": {
            "E1": "legacy e2_* (examples/5_Flux_Ensemble run22)",
            "E2": "legacy e3_* (examples/6_Flux_International ls_ensemble_por_annual2yr)",
            "E3": "legacy e4_* (examples/7_Applied_Water e7cal)",
        },
        "sources": {
            "E1_observation_metadata": {
                "path": str(E1_ARCHIVE / "3_problem_definition" / "observation_metadata.csv"),
                "sha256": sha256(E1_ARCHIVE / "3_problem_definition" / "observation_metadata.csv"),
            },
            "E1_container": {"path": str(E1_CONTAINER), "note": "zarr store, read-only"},
            "E2_container": {"path": str(E2_CONTAINER), "note": "zarr store, read-only"},
            "E3_container": {"path": str(E3_CONTAINER), "note": "zarr store, read-only"},
        },
        "cohort_key": "site_id (+ year for site-year rows)",
        "inclusion_rule": (
            "E1: the 20,328 retained ETf ensemble calibration targets in the run22 archived "
            "observation metadata, 60 configured sites. E2: dates where BOTH coincident "
            "Landsat SSEBop and PT-JPL ETf are finite, restricted to the 66-site "
            "publication cohort. E3: dates with at least two valid OpenET members, "
            "restricted to the 50 SLV_ fields."
        ),
        "temporal_support_rule": (
            "Captures per site-year are counted between each site's first and last retained "
            "target year (missing years inside that span are retained with zero captures). "
            "Inter-capture intervals are gaps in days between consecutive retained targets "
            "within a site. These are observation-density diagnostics, not a "
            "performance-by-gap-length analysis."
        ),
        "units": {"n_captures": "count", "gap_*": "days", "fraction_*": "fraction"},
        "deterministic_seed": None,
        "configured_counts": {"E1": 60, "E2": 66, "E3": 50},
    }
    MANIFEST.add(
        "figs02_etf_support.csv",
        rows=n_etf,
        display_transformations=[
            "site-year and site-level values frozen, not only pooled distributions"
        ],
        evaluated_counts={
            "E1_sites": 60,
            "E2_sites": 66,
            "E3_fields": 50,
            "E1_captures": int(len(caps["E1"])),
            "E2_captures": int(len(caps["E2"])),
            "E3_captures": int(len(caps["E3"])),
        },
        **base,
    )
    MANIFEST.add(
        "figs02_member_availability.csv",
        rows=n_mem,
        display_transformations=[
            "record_type=valid_member_count gives the distribution of valid members per retained target",
            "record_type=member_contribution gives the fraction of retained targets at which each named retrieval member was valid",
        ],
        note=(
            "Member counts differ by experiment (E1/E3 six-member OpenET, E2 two coincident "
            "Landsat members). Spread must never be compared across unequal member counts as "
            "though it were calibrated uncertainty."
        ),
        **base,
    )
    MANIFEST.add(
        "figs02_ndvi_support.csv",
        rows=n_ndvi,
        display_transformations=[
            "reproduces the implemented Landsat-first chronological merge "
            "(SwimContainer.compute.merged_ndvi, preference_order=(landsat, sentinel))",
            "Landsat-selected, Sentinel-2-only, and same-day overlap are reported separately; "
            "the same-day overlap count is a subset of the Landsat-selected count",
            "counted over the full container daily record, which is the period the NDVI series actually forces; the ETf target-support window is carried as separate columns",
        ],
        sanity_checks=pooled,
        note=(
            "BLOCKED AND OMITTED: no NDVI interpolation fraction, fill classification, or "
            "endpoint-extension label is computed or frozen. Section 4.1 records that the "
            "implemented input path applies a 100-step interpolation limit followed by "
            "unrestricted backward/forward filling, while main.md and supp.md describe "
            "nearest-value extension only at record endpoints. That reconciliation is a "
            "separate work item and gates the NDVI-support panel."
        ),
        **base,
    )
    print(
        f"  figs02: etf_support {n_etf} rows, member_availability {n_mem} rows, ndvi_support {n_ndvi} rows"
    )


def _member_contribution(exp, container, members, caps) -> list[dict]:
    import zarr

    z = zarr.open(str(container), mode="r")
    uid = [str(x) for x in z["geometry/uid"][:]]
    t = pd.to_datetime(z["time/daily"][:])
    tpos = pd.Series(np.arange(len(t)), index=t)
    arrays = {
        m: np.asarray(z[f"remote_sensing/etf/landsat/{m}/no_mask"][:], dtype=float) for m in members
    }
    counts = {m: 0 for m in members}
    total = 0
    for site, grp in caps.groupby("site_id"):
        if site not in uid:
            raise BuildError(f"{exp}: capture site {site} absent from container")
        j = uid.index(site)
        idx = tpos.reindex(grp["date"]).to_numpy()
        if np.isnan(idx).any():
            raise BuildError(f"{exp}: capture date outside container time axis at {site}")
        idx = idx.astype(int)
        total += len(idx)
        for m in members:
            counts[m] += int(np.isfinite(arrays[m][idx, j]).sum())
    return [
        {
            "experiment": exp,
            "legacy_prefix": EXPERIMENT_MAP[exp]["legacy_prefix"],
            "record_type": "member_contribution",
            "key": m,
            "n_captures": counts[m],
            "fraction_of_captures": counts[m] / total if total else np.nan,
            "n_captures_total": total,
        }
        for m in members
    ]


def _ndvi_support(exp, container, caps) -> pd.DataFrame:
    import zarr

    z = zarr.open(str(container), mode="r")
    uid = [str(x) for x in z["geometry/uid"][:]]
    t = pd.to_datetime(z["time/daily"][:])
    ls = np.asarray(z["remote_sensing/ndvi/landsat/no_mask"][:], dtype=float)
    s2 = np.asarray(z["remote_sensing/ndvi/sentinel/no_mask"][:], dtype=float)
    span = caps.groupby("site_id")["date"].agg(["min", "max"])
    rows = []
    for site, r in span.iterrows():
        j = uid.index(site)
        L = np.isfinite(ls[:, j])
        S = np.isfinite(s2[:, j])
        any_obs = L | S
        dates = t[any_obs]
        gaps = pd.Series(dates).diff().dt.days.dropna()
        w = (t >= r["min"]) & (t <= r["max"])
        Lw, Sw = L & w, S & w
        anyw = Lw | Sw
        gapsw = pd.Series(t[anyw]).diff().dt.days.dropna()
        rows.append(
            {
                "n_raw_observation_dates_etf_window": int(anyw.sum()),
                "n_landsat_selected_etf_window": int(Lw.sum()),
                "frac_landsat_selected_etf_window": float(Lw.sum() / anyw.sum())
                if anyw.sum()
                else np.nan,
                "raw_interval_median_days_etf_window": float(gapsw.median())
                if len(gapsw)
                else np.nan,
                "experiment": exp,
                "legacy_prefix": EXPERIMENT_MAP[exp]["legacy_prefix"],
                "site_id": site,
                "record_start": str(pd.Timestamp(t.min()).date()),
                "record_end": str(pd.Timestamp(t.max()).date()),
                "etf_support_start": str(pd.Timestamp(r["min"]).date()),
                "etf_support_end": str(pd.Timestamp(r["max"]).date()),
                "n_raw_observation_dates": int(any_obs.sum()),
                "n_landsat_selected": int(L.sum()),
                "n_sentinel_only": int((S & ~L).sum()),
                "n_same_day_overlap": int((L & S).sum()),
                "frac_landsat_selected": float(L.sum() / any_obs.sum())
                if any_obs.sum()
                else np.nan,
                "frac_sentinel_only": float((S & ~L).sum() / any_obs.sum())
                if any_obs.sum()
                else np.nan,
                "frac_same_day_overlap": float((L & S).sum() / any_obs.sum())
                if any_obs.sum()
                else np.nan,
                "raw_interval_median_days": float(gaps.median()) if len(gaps) else np.nan,
                "raw_interval_p90_days": float(gaps.quantile(0.90)) if len(gaps) else np.nan,
                "raw_interval_max_days": float(gaps.max()) if len(gaps) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figure 1 -- scope, evidence matrix, architecture
# --------------------------------------------------------------------------

# Strings that belong to the caption or Methods and must never reach a string
# classified title / direct_label / annotation.  Derived by hand from
# fig01_production_handoff.md section 5 ("Caption- or manuscript-owned") so the
# audit is explicit rather than an accidental substring match: the configured
# counts 60 / 66 / 50 ARE allowed in the artwork and are deliberately absent.
CAPTION_ONLY_VISIBLE_PATTERNS = [
    r"Φ",  # objective symbol
    r"σ",  # sigma
    r"\bsigma",
    r"realization",
    r"iteration",
    r"\bSSEBop\b",
    r"\bPT-?JPL\b",
    r"\bSIMS\b",
    r"\bgeeSEBAL\b",
    r"\beeMETRIC\b",
    r"\bDisALEXI\b",
    r"\bECOSTRESS\b",
    r"\bVolk\b",
    r"\bGridMET\b",
    r"\bSNODAS\b",
    r"\b45\b",
    r"\b29\b",
    r"\b63\b",
    r"\b56\b",
    r"\b31\b",
    r"\b408\b",
    r"\b200\b",
    r"field[- ]year",
    r"\bpaired\b",
    r"withheld",
    r"\bweight",
]

# Exactly the strings a *current* handoff section authorizes as visible copy even
# though a caption-owned pattern matches them.  Matching is by whole-string
# equality, never substring, so an exemption cannot widen to a longer phrase.
CAPTION_PATTERN_EXEMPTIONS = {
    "spread-weighted": (
        "fig01_production_handoff.md section 5.3 explicitly permits this one short label on "
        "the inverse-estimation loop to identify why retrieval-member dispersion is drawn. "
        "The weighting rule, the sigma term, and the objective remain caption/Methods-owned "
        "and are still blocked by the r'\\bweight' and r'sigma' patterns for every other "
        "string."
    ),
}

VISIBLE_STRING_CLASSES = {"title", "direct_label", "annotation", "proof_only"}

# Nodes on the held-out side of the firewall, and nodes that participate in
# parameter fitting or transfer-vector construction.  No edge may run from the
# first set into the second.  Revised for architecture 3.0.0 (handoff sections
# 5.4 and 6.3): the daily balance and both class-specific parameter tokens are
# now protected alongside inverse estimation and the transfer destinations.
EVALUATION_NODES = {"flux_et", "meters"}
FITTING_NODES = {
    "inverse_estimation",
    "daily_balance",
    "e1_map",
    "irrigated_params",
    "rainfed_params",
    "e2_map",
    "e3_map",
    "e0_tag",
}

# Panel (b) transfer topology, asserted before the architecture is written.
FIG01_TRANSFER_SOURCE = "e1_map"
FIG01_PARAM_TOKENS = ("irrigated_params", "rainfed_params")
FIG01_TRANSFER_DESTINATIONS = {
    "e2_map": ("irrigated_params", "rainfed_params"),
    "e3_map": ("irrigated_params",),
}

# ---- Figure 1 example record (handoff sections 5.1, 10.2, 11) --------------
# The frozen selection.  The build compares the reconstructed record against
# these values and stops if the audited site or window has moved.
FIG01_EXAMPLE_SITE = "US-Bi1"
FIG01_EXAMPLE_START = "2017-04-01"
FIG01_EXAMPLE_END = "2017-07-29"
FIG01_EXAMPLE_DAYS = 120
FIG01_EXAMPLE_CAPTURES = 15

# Frozen E1/E2 source-class split used to construct the transfer parameter sets
# (handoff section 6.3).  These are the assignment used to build the vectors,
# not annual irrigation activation.
FIG01_CLASS_COUNTS = {
    "e1_irrigated": 39,
    "e1_rainfed": 21,
    "e2_irrigated": 13,
    "e2_rainfed": 53,
}

# Column-name patterns that may never reach the Figure 1 example display table
# (handoff sections 8, 10.2 and 11).  Figure 3 owns the benchmark comparison,
# Figures 2-6 own every performance metric, and the daily filled NDVI trajectory
# is refused outright until its fill provenance is reconciled.
FIG01_FORBIDDEN_EXAMPLE_COLUMNS = [
    (r"benchmark", "OpenET benchmark series and flags are owned by Figure 3"),
    (r"interpolat", "benchmark-interpolation classification is owned by Figure 3"),
    (r"is_direct", "direct-versus-interpolated benchmark flag is owned by Figure 3"),
    (
        r"ndvi_(kcb|fill|filled|daily|interp|smooth|model)",
        "a daily filled NDVI trajectory is refused until the section 4.1 NDVI fill "
        "provenance is reconciled (handoff sections 5.2 and 11)",
    ),
    (
        r"(^|_)(kge|nse|rmse|mbe|mae|bias|r2|d_kge|rho)($|_)",
        "performance metrics belong to Figures 2-6 and the text",
    ),
    (r"(error|resid)", "evaluation residuals must not reach Figure 1"),
    (r"weight", "observation weights are caption/Methods-owned"),
]


def _audit_key(src: str) -> str:
    """Non-identifying, deterministic audit key for a restricted source record.

    A salted SHA256 truncated to 16 hex characters.  It is stable across
    rebuilds (so an auditor holding the restricted source list can verify the
    linkage) but carries no source-agency identifier into the public layer.
    """
    salt = "swim-rs/fig01/e3-display/v1"
    return hashlib.sha256(f"{salt}|{src}".encode()).hexdigest()[:16]


def _assert_no_caption_facts_visible(classification: dict[str, str]) -> None:
    visible = [s for s, c in classification.items() if c in {"title", "direct_label", "annotation"}]
    hits = []
    for s in visible:
        if s in CAPTION_PATTERN_EXEMPTIONS:
            continue
        for pat in CAPTION_ONLY_VISIBLE_PATTERNS:
            if re.search(pat, s, flags=re.IGNORECASE):
                hits.append((s, pat))
    if hits:
        raise BuildError(
            "fig01: caption/Methods-owned content reached a visible string: "
            + "; ".join(f"{s!r} matches {p!r}" for s, p in hits)
        )


def _build_fig01_example() -> tuple[pd.DataFrame, dict, dict]:
    """Freeze the US-Bi1 example record for Figure 1 panel (a).

    Reads only the two already-audited frozen sibling artifacts -- the Figure 3
    example series and the Figure 4 per-capture member values -- so Figure 1
    reuses the audited record rather than re-deriving it from a container.

    Returns ``(display_frame, selection_payload, column_provenance)``.
    """
    src_series = OUT / "fig03_example_timeseries.csv"
    src_members = OUT / "fig04_spread_capture_values.csv"
    for p in (src_series, src_members):
        if not p.exists():
            raise BuildError(
                f"fig01 example: required frozen source {p.name} is missing; build it first "
                "(fig01 is built after fig03/fig04 by design)"
            )

    ser = pd.read_csv(src_series, parse_dates=["date"])
    ser = ser[(ser["experiment"] == "E1") & (ser["site_id"] == FIG01_EXAMPLE_SITE)].copy()
    start = pd.Timestamp(FIG01_EXAMPLE_START)
    end = pd.Timestamp(FIG01_EXAMPLE_END)
    ser = ser[(ser["date"] >= start) & (ser["date"] <= end)].sort_values("date")

    # ---- section 11: the example must match the frozen selection exactly ----
    sites = sorted(set(ser["site_id"]))
    if sites != [FIG01_EXAMPLE_SITE]:
        raise BuildError(
            f"fig01 example: expected the frozen site {FIG01_EXAMPLE_SITE!r}, got {sites}"
        )
    require_count(len(ser), FIG01_EXAMPLE_DAYS, "fig01 example daily rows")
    if str(ser["date"].min().date()) != FIG01_EXAMPLE_START or (
        str(ser["date"].max().date()) != FIG01_EXAMPLE_END
    ):
        raise BuildError(
            "fig01 example: date window differs from the frozen selection "
            f"({ser['date'].min().date()}..{ser['date'].max().date()} vs "
            f"{FIG01_EXAMPLE_START}..{FIG01_EXAMPLE_END})"
        )
    expected_index = pd.date_range(start, end, freq="D")
    if not ser["date"].reset_index(drop=True).equals(pd.Series(expected_index)):
        raise BuildError("fig01 example: the 120-day window is not a contiguous daily index")
    require_columns(
        ser,
        [
            "eto",
            "precip",
            "rz_depletion",
            "irr_applied",
            "swim_ET",
            "flux_ET",
            "swe",
            "sensor",
            "ndvi_landsat_raw",
            "ndvi_sentinel_raw",
            "is_calibration_capture",
            "calibration_target_etf",
            "calibration_target_member_count",
            "calibration_target_ensemble_std",
        ],
        "fig01 example source series",
    )
    caps_mask = ser["is_calibration_capture"].astype(bool)
    require_count(
        int(caps_mask.sum()), FIG01_EXAMPLE_CAPTURES, "fig01 example calibration captures"
    )
    for col in ("eto", "precip", "rz_depletion", "irr_applied", "swim_ET", "flux_ET"):
        n_null = int(ser[col].isna().sum())
        if n_null:
            raise BuildError(
                f"fig01 example: daily column {col!r} has {n_null} missing value(s) in the frozen "
                "window; investigate upstream rather than filling or dropping them"
            )

    mem = pd.read_csv(src_members, parse_dates=["date"])
    mem = mem[(mem["experiment"] == "E1") & (mem["site_id"] == FIG01_EXAMPLE_SITE)].copy()
    mem = mem[(mem["date"] >= start) & (mem["date"] <= end)].sort_values("date")
    require_count(len(mem), FIG01_EXAMPLE_CAPTURES, "fig01 example member-value captures")
    member_cols = [f"etf_{m}" for m in E1_MEMBERS]
    require_columns(
        mem,
        ["member_count", "spread_retrieval", "ensemble_mean_etf", *member_cols],
        "fig01 example member source",
    )
    if set(ser.loc[caps_mask, "date"]) != set(mem["date"]):
        raise BuildError(
            "fig01 example: the Figure 3 calibration-capture dates and the Figure 4 member "
            "capture dates disagree; the two frozen artifacts are not describing one record"
        )

    joined = ser.loc[caps_mask, ["date"]].merge(mem, on="date", how="left", validate="one_to_one")

    # ---- section 11: plotted member marks must reconcile to the frozen target ----
    n_members = joined[member_cols].notna().sum(axis=1)
    if not bool((n_members == joined["member_count"]).all()):
        bad = joined.loc[n_members != joined["member_count"], "date"].dt.date.tolist()
        raise BuildError(
            f"fig01 example: non-null member marks disagree with the frozen member_count at {bad}"
        )
    plain_mean = joined[member_cols].mean(axis=1, skipna=True)
    d_mean = float((plain_mean - joined["ensemble_mean_etf"]).abs().max())
    if not np.isfinite(d_mean) or d_mean > 1e-6:
        raise BuildError(
            "fig01 example: the frozen target mean is not the plain mean of the available "
            f"members (max abs difference {d_mean:.3e}); investigate the target composition "
            "before plotting member marks against it"
        )
    plain_spread = joined[member_cols].std(axis=1, skipna=True, ddof=1)
    d_spread = float((plain_spread - joined["spread_retrieval"]).abs().max())
    if not np.isfinite(d_spread) or d_spread > 1e-6:
        raise BuildError(
            "fig01 example: the frozen ensemble spread is not the sample standard deviation "
            f"of the available members (max abs difference {d_spread:.3e})"
        )
    # The Figure 3 series and the Figure 4 capture table must carry the same
    # target, member count and spread, or the panel would mix two vintages.
    x = ser.loc[
        caps_mask,
        [
            "date",
            "calibration_target_etf",
            "calibration_target_member_count",
            "calibration_target_ensemble_std",
        ],
    ]
    x = x.merge(mem, on="date", how="left", validate="one_to_one")
    for a, b in (
        ("calibration_target_etf", "ensemble_mean_etf"),
        ("calibration_target_member_count", "member_count"),
        ("calibration_target_ensemble_std", "spread_retrieval"),
    ):
        d = float((x[a] - x[b]).abs().max())
        if d != 0.0:
            raise BuildError(
                f"fig01 example: fig03 {a!r} and fig04 {b!r} disagree by {d:.3e}; the two frozen "
                "artifacts must describe the identical calibration targets"
            )

    out = ser[["experiment", "legacy_prefix", "site_id", "date"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["ndvi_landsat_raw"] = ser["ndvi_landsat_raw"].to_numpy()
    out["ndvi_sentinel_raw"] = ser["ndvi_sentinel_raw"].to_numpy()
    out["is_calibration_capture"] = caps_mask.to_numpy()
    out["capture_sensor"] = ser["sensor"].to_numpy()
    cap_index = ser.index[caps_mask]
    for col in member_cols:
        s = pd.Series(np.nan, index=ser.index, dtype=float)
        s.loc[cap_index] = joined[col].to_numpy()
        out[col] = s.to_numpy()
    for name, src in (
        ("etf_target_mean", "ensemble_mean_etf"),
        ("etf_member_count", "member_count"),
        ("etf_ensemble_spread", "spread_retrieval"),
    ):
        s = pd.Series(np.nan, index=ser.index, dtype=float)
        s.loc[cap_index] = joined[src].to_numpy()
        out[name] = s.to_numpy()
    out["eto"] = ser["eto"].to_numpy()
    out["precip"] = ser["precip"].to_numpy()
    out["rz_depletion"] = ser["rz_depletion"].to_numpy()
    out["irr_applied"] = ser["irr_applied"].to_numpy()
    out["swim_ET"] = ser["swim_ET"].to_numpy()
    out["flux_ET"] = ser["flux_ET"].to_numpy()
    out["swe_audit"] = ser["swe"].to_numpy()
    out = out.reset_index(drop=True)

    fig03_src = "paper/data/final/figures/fig03_example_timeseries.csv"
    fig04_src = "paper/data/final/figures/fig04_spread_capture_values.csv"
    prov: dict[str, dict] = {
        "experiment": {
            "source": fig03_src,
            "source_column": "experiment",
            "units": None,
            "display_role": "key",
        },
        "legacy_prefix": {
            "source": fig03_src,
            "source_column": "legacy_prefix",
            "units": None,
            "display_role": "audit_only",
        },
        "site_id": {
            "source": fig03_src,
            "source_column": "site_id",
            "units": None,
            "display_role": "key",
            "note": "public AmeriFlux site identifier",
        },
        "date": {
            "source": fig03_src,
            "source_column": "date",
            "units": "date (YYYY-MM-DD)",
            "display_role": "shared_date_axis",
        },
        "ndvi_landsat_raw": {
            "source": fig03_src,
            "source_column": "ndvi_landsat_raw",
            "units": "dimensionless",
            "display_role": "evidence_row_ndvi",
            "note": "raw Landsat capture; never connected or filled",
        },
        "ndvi_sentinel_raw": {
            "source": fig03_src,
            "source_column": "ndvi_sentinel_raw",
            "units": "dimensionless",
            "display_role": "evidence_row_ndvi",
            "note": "raw Sentinel-2 capture; never connected or filled",
        },
        "is_calibration_capture": {
            "source": fig03_src,
            "source_column": "is_calibration_capture",
            "units": None,
            "display_role": "structure",
            "note": "marks the 15 acquisition dates that carry an ETf calibration target",
        },
        "capture_sensor": {
            "source": fig03_src,
            "source_column": "sensor",
            "units": None,
            "display_role": "audit_only",
            "note": (
                "instrument of the ETf calibration capture (all 15 are landsat here); it is NOT "
                "the NDVI capture sensor -- NDVI sensor identity is carried by the two separate "
                "raw NDVI columns"
            ),
        },
    }
    for m, col in zip(E1_MEMBERS, member_cols, strict=True):
        prov[col] = {
            "source": fig04_src,
            "source_column": col,
            "units": "dimensionless ETf",
            "display_role": "evidence_row_etf_member",
            "note": (
                f"one OpenET v2.1 retrieval member at the capture; plotted as a small neutral "
                f"mark without a member-name legend (handoff section 5.2). Member key {m!r} is "
                "audit provenance, not reader-facing copy."
            ),
        }
    prov["etf_target_mean"] = {
        "source": fig04_src,
        "source_column": "ensemble_mean_etf",
        "units": "dimensionless ETf",
        "display_role": "evidence_row_etf_target",
        "note": (
            "the calibration target; verified equal to the plain mean of the available members "
            f"to {d_mean:.3e} and equal to fig03 calibration_target_etf exactly"
        ),
    }
    prov["etf_member_count"] = {
        "source": fig04_src,
        "source_column": "member_count",
        "units": "count",
        "display_role": "audit_only",
        "note": "verified equal to the number of non-null member columns at every capture",
    }
    prov["etf_ensemble_spread"] = {
        "source": fig04_src,
        "source_column": "spread_retrieval",
        "units": "dimensionless ETf",
        "display_role": "evidence_row_etf_target",
        "note": (
            "retrieval-member dispersion; verified equal to the ddof=1 sample standard deviation "
            f"of the available members to {d_spread:.3e}"
        ),
    }
    prov["eto"] = {
        "source": fig03_src,
        "source_column": "eto",
        "units": "mm d-1",
        "display_role": "evidence_row_forcing",
    }
    prov["precip"] = {
        "source": fig03_src,
        "source_column": "precip",
        "units": "mm d-1",
        "display_role": "evidence_row_forcing",
    }
    prov["rz_depletion"] = {
        "source": fig03_src,
        "source_column": "rz_depletion",
        "units": "mm",
        "display_role": "evidence_row_state",
        "note": "root-zone depletion; the primary visual evidence of state propagation",
    }
    prov["irr_applied"] = {
        "source": fig03_src,
        "source_column": "irr_applied",
        "units": "mm d-1",
        "display_role": "evidence_row_state_and_output",
        "note": "simulated gross applied water; aggregated annually for the E3 meter comparison",
    }
    prov["swim_ET"] = {
        "source": fig03_src,
        "source_column": "swim_ET",
        "units": "mm d-1",
        "display_role": "evidence_row_output",
    }
    prov["flux_ET"] = {
        "source": fig03_src,
        "source_column": "flux_ET",
        "units": "mm d-1",
        "display_role": "held_out_observation",
        "note": (
            "rendered in the held-out region as a thin near-black reference trace; no metric, "
            "residual or agreement claim may accompany it (handoff section 5.4)"
        ),
    }
    prov["swe_audit"] = {
        "source": fig03_src,
        "source_column": "swe",
        "units": "mm",
        "display_role": "audit_only_not_plotted",
        "note": (
            "gridded SWE is identically zero across this growing-season window. Handoff section "
            "5.3 requires SWE to be represented symbolically as an auxiliary calibration "
            "constraint; a zero-valued trace must not be forced into the panel. Frozen here only "
            "so the absence is auditable."
        ),
    }

    # ---- section 11: refused and unprovenanced columns ----
    for col in out.columns:
        for pat, why in FIG01_FORBIDDEN_EXAMPLE_COLUMNS:
            if re.search(pat, col, flags=re.IGNORECASE):
                raise BuildError(f"fig01 example: column {col!r} is refused -- {why}")
    missing_prov = [c for c in out.columns if c not in prov]
    if missing_prov:
        raise BuildError(
            f"fig01 example: emitted column(s) {missing_prov} have no recorded provenance"
        )
    stale_prov = [c for c in prov if c not in out.columns]
    if stale_prov:
        raise BuildError(f"fig01 example: provenance recorded for absent column(s) {stale_prov}")

    selection = {
        "figure": "fig01",
        "panel": "a",
        "experiment": "E1",
        "legacy_prefix": "e2_",
        "role": (
            "Illustrative cross-figure visual motif. Figure 1 uses this record to show data "
            "form, acquisition cadence and state propagation only. Figure 3 retains ownership "
            "of the SWIM-RS-versus-benchmark reconstruction comparison, the direct-versus-"
            "interpolated benchmark distinction, and every performance metric."
        ),
        "selected_site": FIG01_EXAMPLE_SITE,
        "full_window": {
            "start": FIG01_EXAMPLE_START,
            "end": FIG01_EXAMPLE_END,
            "n_days": int(len(out)),
            "n_calibration_captures": int(caps_mask.sum()),
        },
        "displayed_window": {
            "start": FIG01_EXAMPLE_START,
            "end": FIG01_EXAMPLE_END,
            "n_days": int(len(out)),
            "n_calibration_captures": int(caps_mask.sum()),
            "cropped": False,
            "crop_rationale": None,
            "note": (
                "The full 120-day record is displayed; no crop is applied. If a later "
                "composition crops it, the exact dates and an input/state-based rationale must "
                "be recorded here before the crop is rendered (handoff section 5.1)."
            ),
        },
        "rationale": (
            "The record was inherited whole from the already-audited Figure 3 example, which "
            "was chosen by the frozen six-step rule in fig03_example_selection.json. Figure 1 "
            "adopts it unchanged so a reader meeting the same site in Figure 3 recognizes it, "
            "and so no second, independently motivated example enters the package. It is a "
            "growing-season window containing sparse Landsat and Sentinel-2 NDVI captures, 15 "
            "ETf calibration captures with visible retrieval-member disagreement, precipitation "
            "and simulated irrigation events, and an uninterrupted daily state trajectory -- "
            "the data forms panel (a) must show."
        ),
        "selection_independence": (
            "Neither the choice of this example nor its displayed window used evaluation "
            "residuals, flux or meter errors, benchmark errors, or any performance metric. "
            "Figure 1 applied no ranking of its own: it adopted the Figure 3 selection whole "
            "and cropped nothing. The Figure 3 rule ranked candidate sites toward the cohort "
            "MEDIAN benchmark-interpolated delta rather than toward favorable performance, and "
            "no metric value is carried into, or plotted from, this Figure 1 file."
        ),
        "excluded_from_this_file": {
            "benchmark_raw / benchmark_interpolated / is_direct_benchmark": (
                "Figure 3 owns the benchmark reconstruction comparison (handoff section 8)"
            ),
            "ndvi_kcb": (
                "the daily filled/model NDVI trajectory; refused outright because its fill "
                "provenance is unresolved (handoff sections 5.2 and 11). It is not carried even "
                "as an audit column."
            ),
            "observed_etf / obs_weight / calibration_target_final_weight": (
                "observation weights and the fitted objective are caption/Methods-owned"
            ),
            "etf_model / ks": "not required by any panel (a) evidence row",
            "all performance metrics": "owned by Figures 2-6 and the text",
        },
        "reconciliation": {
            "member_count_matches_non_null_member_marks": True,
            "target_mean_is_plain_mean_of_available_members": True,
            "target_mean_max_abs_difference": d_mean,
            "spread_is_ddof1_sample_sd_of_available_members": True,
            "spread_max_abs_difference": d_spread,
            "fig03_targets_equal_fig04_targets": True,
            "tolerance_note": (
                "The residual ~1e-7 difference on the target mean is a storage precision "
                "artifact, not a method difference: fig04 member values are read from the "
                "float32 container arrays while the target and spread come from the archived "
                "float64 observation analysis. Member counts, and the fig03/fig04 target, "
                "member-count and spread columns, agree exactly (0.0)."
            ),
            "member_count_range": [
                int(joined["member_count"].min()),
                int(joined["member_count"].max()),
            ],
        },
        "sources": {
            "example_series": {
                "path": fig03_src,
                "sha256": sha256(src_series),
                "note": "frozen Figure 3 example record; read, not re-derived from a container",
            },
            "member_values": {
                "path": fig04_src,
                "sha256": sha256(src_members),
                "note": "frozen Figure 4 per-capture retrieval-member ETf values",
            },
            "upstream_selection_record": {
                "path": "paper/data/final/figures/fig03_example_selection.json",
                "sha256": sha256(OUT / "fig03_example_selection.json"),
                "note": "the six-step rule that chose the site and window Figure 1 inherits",
            },
        },
        "column_provenance": prov,
        "generator_script": "scripts/figures/build_figure_data.py",
        "generator_version": FIG01_BUILDER_VERSION,
        "contract": "paper/notes/fig01_production_handoff.md sections 5.1, 10.2 and 11",
    }
    return out, selection, prov


def build_fig01() -> None:
    import geopandas as gpd
    from shapely.geometry import box

    e1 = e1_configured()
    e2 = e2_configured()
    e3 = e3_configured()
    e2_sites = set(e2["site_id"])
    e1["in_e2"] = e1["site_id"].isin(e2_sites)
    e2["in_e1"] = e2["site_id"].isin(set(e1["site_id"]))
    require_count(int(e1["in_e2"].sum()), EXPECTED["E1_E2_overlap"], "fig01 E1->E2 overlap")
    require_count(int(e2["in_e1"].sum()), EXPECTED["E1_E2_overlap"], "fig01 E2->E1 overlap")

    # ---- handoff section 11: configured counts, source classes, MB_Pch ----
    require_count(len(e1), EXPECTED["E1_configured"], "fig01 E1 configured sites")
    require_count(len(e2), EXPECTED["E2_configured"], "fig01 E2 configured sites")
    require_count(len(e3), EXPECTED["E3_fields"], "fig01 E3 configured fields")
    if "MB_Pch" not in set(e1["site_id"]):
        raise BuildError("fig01: MB_Pch is missing from the configured E1 scope")
    class_counts: dict[str, int] = {}
    for exp, frame in (("e1", e1), ("e2", e2)):
        if "irrigation_class" not in frame.columns:
            raise BuildError(f"fig01: {exp} carries no frozen irrigation_class assignment")
        counts = frame["irrigation_class"].value_counts().to_dict()
        seen = set(counts)
        if seen != {"irrigated", "rainfed"}:
            raise BuildError(
                f"fig01 {exp}: irrigation_class must be exactly irrigated/rainfed, got "
                f"{sorted(seen)}"
            )
        for cls_name in ("irrigated", "rainfed"):
            key = f"{exp}_{cls_name}"
            got = int(counts[cls_name])
            require_count(got, FIG01_CLASS_COUNTS[key], f"fig01 {exp} {cls_name} source class")
            class_counts[key] = got

    e1_daily = set(pd.read_csv(FINAL / "e2_primary_daily_site_metrics.csv")["fid"])
    e1["configured"] = True
    e1["in_daily_evaluation"] = e1["site_id"].isin(e1_daily)
    e1["source_key"] = e1["site_id"]
    e1["display_id"] = e1["site_id"]

    e2["configured"] = True
    e2["source_key"] = e2["site_id"]
    e2["display_id"] = e2["site_id"]

    # ---- E3 generalization: 1 km snapped centroids, no source identifiers ----
    grid_m = 1000.0
    e3_gdf = gpd.GeoDataFrame(
        e3.copy(),
        geometry=gpd.points_from_xy(e3["lon"], e3["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5070")
    snapped = gpd.points_from_xy(
        np.round(e3_gdf.geometry.x / grid_m) * grid_m,
        np.round(e3_gdf.geometry.y / grid_m) * grid_m,
    )
    e3_gdf = e3_gdf.set_geometry(
        gpd.GeoSeries(snapped, index=e3_gdf.index, crs="EPSG:5070")
    ).to_crs("EPSG:4326")
    e3_method = (
        f"field centroid snapped to a {int(grid_m)} m EPSG:5070 grid "
        "(approved 2026-08-19); no exact polygon, no source-agency identifier"
    )
    e3_disp = gpd.GeoDataFrame(
        {
            "display_id": [f"E3_{i:03d}" for i in range(len(e3_gdf))],
            "geometry_method": e3_method,
            "source_group": "SLV metered field",
            "source_key": [_audit_key(s) for s in e3_gdf["src_id"]],
            "crop_group": np.where(e3_gdf["crop"].str.upper() == "ALFALFA", "alfalfa", "other"),
        },
        geometry=e3_gdf.geometry,
        crs="EPSG:4326",
    )
    require_count(len(e3_disp), EXPECTED["E3_fields"], "fig01 E3 display rows")
    banned = {"src_id", "source", "site_id", "acres", "lat", "lon", "basin", "state"}
    if banned & set(e3_disp.columns):
        raise BuildError("fig01: restricted E3 attribute leaked into the public layer")
    # The public audit key must not be, contain, or be contained by any source
    # identifier, and the published geometry must be generalized points only.
    restricted = set(e3["src_id"].astype(str)) | set(e3["site_id"].astype(str))
    for col in e3_disp.columns:
        if col == "geometry":
            continue
        vals = {str(v) for v in e3_disp[col]}
        if vals & restricted:
            raise BuildError(f"fig01 e3_display: restricted identifier leaked into column {col!r}")
    # A substring scan is unsound here (source ids are 3-7 digit numbers that
    # occur inside hex digests by chance), so the public per-record fields are
    # instead constrained to formats that cannot carry a source identifier.
    if not all(re.fullmatch(r"E3_\d{3}", str(v)) for v in e3_disp["display_id"]):
        raise BuildError("fig01 e3_display: display_id must be a sequential E3_NNN display label")
    if not all(re.fullmatch(r"[0-9a-f]{16}", str(v)) for v in e3_disp["source_key"]):
        raise BuildError(
            "fig01 e3_display: source_key must be a 16-hex-character non-identifying audit key"
        )
    if set(e3_disp.geom_type) != {"Point"}:
        raise BuildError(
            f"fig01 e3_display: only generalized points may be published, got "
            f"{sorted(set(e3_disp.geom_type))}"
        )
    if len(set(e3_disp["source_key"])) != len(e3_disp):
        raise BuildError("fig01 e3_display: source_key is not one-to-one with the display records")

    e1_gdf = gpd.GeoDataFrame(
        e1[
            [
                "display_id",
                "configured",
                "irrigation_class",
                "in_e2",
                "in_daily_evaluation",
                "source_key",
            ]
        ],
        geometry=gpd.points_from_xy(e1["lon"], e1["lat"]),
        crs="EPSG:4326",
    )
    if not NE_COUNTRIES.exists():
        raise BuildError(f"fig01: boundary source missing {NE_COUNTRIES}")
    world = gpd.read_file(NE_COUNTRIES, engine="fiona")
    world_ctx = world[["ADMIN", "ISO_A3", "CONTINENT", "geometry"]].copy()
    world_ctx["boundary_source"] = "Natural Earth 110m admin 0 countries"
    world_ctx["boundary_version"] = _ne_version()

    e2_gdf = gpd.GeoDataFrame(
        e2[
            [
                "display_id",
                "configured",
                "network",
                "country",
                "in_e1",
                "equipped_for_irrigation",
                "irrigation_class",
                "source_key",
            ]
        ].rename(columns={"country": "country_container_raw"}),
        geometry=gpd.points_from_xy(e2["lon"], e2["lat"]),
        crs="EPSG:4326",
    )
    # The container's country attribute is null for 13 sites and mixes ISO codes
    # with full names, so it cannot support the ten-country claim. Derive country
    # and continent from the archived boundary layer instead, falling back to the
    # nearest polygon for coastal sites the 110m coastline misses.
    poly = world_ctx[["ADMIN", "CONTINENT", "geometry"]].reset_index(drop=True)
    j = gpd.sjoin(e2_gdf[["geometry"]].reset_index(drop=True), poly, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    admin = list(j["ADMIN"])
    cont = list(j["CONTINENT"])
    assign = ["within_polygon"] * len(admin)
    proj_poly = poly.to_crs("EPSG:6933")
    for i, a in enumerate(admin):
        if isinstance(a, str):
            continue
        pt = e2_gdf.geometry.iloc[i : i + 1].to_crs("EPSG:6933").iloc[0]
        k = int(proj_poly.geometry.distance(pt).idxmin())
        admin[i] = str(proj_poly.loc[k, "ADMIN"])
        cont[i] = str(proj_poly.loc[k, "CONTINENT"])
        assign[i] = "nearest_polygon"
    if any(not isinstance(a, str) or a in ("nan", "None") for a in admin):
        raise BuildError("fig01: a configured E2 site could not be assigned a country")
    e2_gdf["country"] = admin
    e2_gdf["continent"] = cont
    e2_gdf["country_assignment"] = assign

    # Handoff section 10.1 required-attribute contract.
    required_attrs = {
        "e1_sites": ["display_id", "configured", "irrigation_class", "in_e2", "source_key"],
        "e2_sites": ["display_id", "network", "country", "in_e1", "source_key"],
        "e3_display": ["display_id", "geometry_method", "source_group"],
    }
    for lyr, cols in required_attrs.items():
        have = set({"e1_sites": e1_gdf, "e2_sites": e2_gdf, "e3_display": e3_disp}[lyr].columns)
        missing = [c for c in cols if c not in have]
        if missing:
            raise BuildError(f"fig01 {lyr}: missing required attributes {missing}")

    conus_bbox = box(-125.0, 24.0, -66.5, 49.5)
    conus_ctx = gpd.clip(world_ctx[world_ctx["ADMIN"] == "United States of America"], conus_bbox)
    conus_ctx = gpd.GeoDataFrame(conus_ctx, geometry="geometry", crs=world_ctx.crs)

    slv_hull = e3_disp.union_all().convex_hull.buffer(0.15)
    slv_ctx = gpd.GeoDataFrame(
        {
            "name": ["San Luis Valley generalized study extent"],
            "geometry_method": [
                "convex hull of the 1 km snapped display centroids, buffered 0.15 degrees"
            ],
            "boundary_source": ["derived from generalized E3 display geometry"],
        },
        geometry=gpd.GeoSeries([slv_hull], crs="EPSG:4326"),
        crs="EPSG:4326",
    )

    gpkg = OUT / "fig01_scope.gpkg"
    gpkg_tmp = OUT / "fig01_scope.gpkg.tmp"
    if gpkg_tmp.exists():
        gpkg_tmp.unlink()
    gpkg_layers = {
        "e1_sites": e1_gdf,
        "e2_sites": e2_gdf,
        "e3_display": e3_disp,
        "conus_context": conus_ctx,
        "world_context": world_ctx,
        "slv_context": slv_ctx,
    }
    # Every published layer is EPSG:4326; verify the stored coordinates can
    # actually be degrees before writing, so a projected-metres layer can never
    # ship under a 4326 tag again (bug D3).
    for layer_name, layer_gdf in gpkg_layers.items():
        if layer_gdf.crs is None or layer_gdf.crs.to_epsg() != 4326:
            raise BuildError(
                f"fig01 {layer_name}: published layers must be EPSG:4326, got "
                f"{layer_gdf.crs.to_string() if layer_gdf.crs is not None else None}"
            )
        require_layer_crs_consistent(layer_gdf, f"fig01 {layer_name}")
    # Written to a sibling temp path and moved into place, so a concurrent
    # reader never observes a half-written GeoPackage.
    for layer_name, layer_gdf in gpkg_layers.items():
        layer_gdf.to_file(gpkg_tmp, layer=layer_name, driver="GPKG", engine="fiona")
    gpkg_tmp.replace(gpkg)

    countries = sorted(set(e2_gdf["country"]))
    n_countries = len(countries)
    continents = sorted(set(e2_gdf["continent"]))
    n_continents = len(continents)
    if n_countries != 10 or n_continents != 4:
        raise BuildError(
            "fig01: manuscript states ten E2 countries on four continents; "
            f"derived {n_countries} countries on {n_continents} continents"
        )

    evid = pd.DataFrame(
        [
            {
                "experiment": "E0",
                "evidence_role": "model_development",
                "configured_n": 60,
                "configured_unit": "CONUS cropland flux sites (shared with E1)",
                "domain": "CONUS cropland",
                "primary_etf_target": "SSEBop ETf (matched across formulations)",
                "primary_weighting": "spread-based",
                "daily_evaluation_n": 45,
                "monthly_supported_n": 31,
                "monthly_finite_metric_n": 31,
                "field_year_n": None,
                "external_evaluation": "flux ET used AFTER satellite calibration to select the cover-scaled sigmoid formulation",
                "parameter_source": "locally calibrated per formulation",
                "scientific_roles": "vegetation-formulation selection",
                "independence_statement": "model-development evidence, not independent validation",
                "source_artifact": "paper/text/main.md Table 3; paper/text/supp.md S3",
                "source_sha256": sha256(REPO / "paper" / "text" / "main.md"),
            },
            {
                "experiment": "E1",
                "evidence_role": "evaluation",
                "configured_n": 60,
                "configured_unit": "CONUS cropland flux sites",
                "domain": "CONUS cropland",
                "primary_etf_target": "per-capture mean of six OpenET v2.1 ETf members",
                "primary_weighting": "spread-based (sigma_ensemble + 0.1)",
                "daily_evaluation_n": 45,
                "monthly_supported_n": 31,
                "monthly_finite_metric_n": 29,
                "field_year_n": None,
                "external_evaluation": "Volk et al. (2024) v2.1 closure-corrected flux ET; separately extracted OpenET ensemble benchmark",
                "parameter_source": "local calibration; source cohort for the fixed irrigation-class parameter sets",
                "scientific_roles": "ET agreement; temporal reconstruction; ensemble reliability; held-out transfer",
                "independence_statement": "external to parameter estimation but not fully independent of model development (E0 shares this cohort)",
                "source_artifact": "paper/data/final/e2_primary_daily_site_metrics.csv",
                "source_sha256": sha256(FINAL / "e2_primary_daily_site_metrics.csv"),
            },
            {
                "experiment": "E2",
                "evidence_role": "evaluation",
                "configured_n": 66,
                "configured_unit": "cropland flux sites",
                "domain": f"{n_countries} countries on {n_continents} continents; CONUS and ex-CONUS",
                "primary_etf_target": "per-capture mean of coincident Landsat SSEBop and PT-JPL ETf",
                "primary_weighting": "spread-based (sigma_ensemble + 0.1); fixed 0.33 scale only for the ECOSTRESS-only sensitivity rows",
                "daily_evaluation_n": 63,
                "monthly_supported_n": 56,
                "monthly_finite_metric_n": 50,
                "field_year_n": None,
                "external_evaluation": "AmeriFlux, FLUXNET, ICOS and OzFlux ET",
                "parameter_source": "local calibration arm; fixed E1-derived irrigated/rainfed sets for the transfer arm",
                "scientific_roles": "international evaluation; E1-to-E2 transfer under changed geography and inputs",
                "independence_statement": "13 of 66 sites also occur in E1 and test changed inputs; 53 are unseen fields",
                "source_artifact": str(E2_RESULTS / "evaluation_metrics.csv"),
                "source_sha256": sha256(E2_RESULTS / "evaluation_metrics.csv"),
            },
            {
                "experiment": "E3",
                "evidence_role": "evaluation",
                "configured_n": 50,
                "configured_unit": "metered San Luis Valley fields",
                "domain": "San Luis Valley, Colorado",
                "primary_etf_target": "per-capture mean of six OpenET v2.1 ETf members, at least two valid members",
                "primary_weighting": "spread-based (sigma_ensemble + 0.1)",
                "daily_evaluation_n": None,
                "monthly_supported_n": None,
                "monthly_finite_metric_n": None,
                "field_year_n": 408,
                "external_evaluation": "state-agency groundwater pumping records",
                "parameter_source": "local calibration arm; fixed E1-derived irrigated set for the transfer arm",
                "scientific_roles": "applied-water consistency under local and transferred parameters",
                "independence_statement": "no overlap with the E1 source cohort; meters withheld from parameter estimation",
                "source_artifact": str(E3_LOCAL / "per_field_year.csv"),
                "source_sha256": sha256(E3_LOCAL / "per_field_year.csv"),
            },
        ]
    )
    # ---- handoff section 11 assertions on the evidence matrix ----
    require_columns(
        evid,
        [
            "experiment",
            "evidence_role",
            "configured_n",
            "configured_unit",
            "domain",
            "primary_etf_target",
            "primary_weighting",
            "daily_evaluation_n",
            "monthly_supported_n",
            "monthly_finite_metric_n",
            "field_year_n",
            "external_evaluation",
            "parameter_source",
            "scientific_roles",
            "source_artifact",
            "source_sha256",
        ],
        "fig01 evidence matrix",
    )
    require_unique(evid, ["experiment"], "fig01 evidence matrix")
    roles = dict(zip(evid["experiment"], evid["evidence_role"], strict=True))
    if roles.get("E0") != "model_development":
        raise BuildError("fig01: the E0 row must be typed evidence_role=model_development")
    for exp in ("E1", "E2", "E3"):
        if roles.get(exp) != "evaluation":
            raise BuildError(f"fig01: {exp} must be typed evidence_role=evaluation")
    conf = dict(zip(evid["experiment"], evid["configured_n"], strict=True))
    for exp, want in (("E1", 60), ("E2", 66), ("E3", 50)):
        require_count(int(conf[exp]), want, f"fig01 evidence matrix {exp} configured_n")
    tgt = dict(zip(evid["experiment"], evid["primary_etf_target"], strict=True))
    for exp in ("E1", "E3"):
        if "six" not in tgt[exp].lower():
            raise BuildError(
                f"fig01: {exp} primary ETf target must be the six-member OpenET mean, "
                f"not a two-member target; got {tgt[exp]!r}"
            )
    if "six" in tgt["E2"].lower() or "6-member" in tgt["E2"].lower():
        raise BuildError("fig01: E2 must not be labelled with a six-member primary ETf target")
    if not ("ssebop" in tgt["E2"].lower() and "pt-jpl" in tgt["E2"].lower()):
        raise BuildError("fig01: E2 primary ETf target must name the coincident Landsat pair")
    for exp, w in zip(evid["experiment"], evid["primary_weighting"], strict=True):
        head = w.split(";")[0].lower()
        if "fixed" in head or "spread" not in head:
            raise BuildError(f"fig01: {exp} primary weighting must be spread-weighted, got {w!r}")
    e0_ind = evid.loc[evid["experiment"] == "E0", "independence_statement"].iloc[0]
    if "independent validation" in e0_ind.lower().replace("not independent validation", ""):
        raise BuildError("fig01: E0 must not be labelled independent validation")

    n_ev = write_table(evid, "fig01_evidence_matrix.csv")

    # ---- panel (a) example record (handoff sections 5.1, 10.2, 11) ----
    example, example_selection, example_prov = _build_fig01_example()
    n_ex = write_table(example, "fig01_example_timeseries.csv")
    (OUT / "fig01_example_selection.json").write_text(
        json.dumps(example_selection, indent=2, ensure_ascii=False)
    )
    n_ex_caps = int(example["is_calibration_capture"].sum())

    arch = {
        "schema_version": "3.0.0",
        "supersedes": {
            "schema_version": "2.1.0",
            "superseded_composition": (
                "map-plus-framework: panel (a) 'Study Domains', panel (b) 'SWIM-RS Framework' "
                "with a generic crop-soil cross-section and three grouped input cards, and one "
                "unlettered full-width transfer ribbon beneath both panels"
            ),
            "reason": (
                "fig01_production_handoff.md was rewritten 2026-08-20. The gray-box composition "
                "read as a graphical abstract: large labels, generic crop artwork, conceptual "
                "regions and unused space carried more weight than data. Figure 1 is now an "
                "evidence-first analytical figure whose density comes from real marks."
            ),
            "removed_requirements": [
                "the unlettered full-width transfer ribbon and its three ribbon nodes",
                "the generic crop-soil-water process cross-section",
                "the grouped input cards 'NDVI · Vegetation State', "
                "'ETf + SWE · Calibration Targets' and 'Gridded Forcing · Model Drivers'",
                "the 'Satellite and Gridded Inputs' input-region heading",
                "the 'Daily Water Balance' titled process node",
                "the inverse_estimation -> e1_source reading-path connector added 2026-08-20",
            ],
            "removed_strings": [
                "Study Domains",
                "SWIM-RS Framework",
                "Satellite and Gridded Inputs",
                "NDVI · Vegetation State",
                "ETf + SWE · Calibration Targets",
                "Gridded Forcing · Model Drivers",
                "Daily Water Balance",
                "E1 · Source Parameters",
                "E2 · Geographic and Input Transfer",
                "E3 · Applied-Water Transfer",
                "E0 · Formulation Selection",
                "E2 · International",
            ],
            "note": (
                "Version 2.1.0, the proofs under paper/figures/proofs/fig01_graybox_110/ and "
                "fig01_graybox/, the assets under paper/figures/fig1_handoff/, and "
                "scripts/figures/fig1_scope_architecture.py are design provenance only. They "
                "must not be polished, relabelled or reinserted."
            ),
        },
        "contract": (
            "paper/notes/fig01_production_handoff.md sections 4-9, rewritten 2026-08-20. Every "
            "string under panels / evidence_rows / inverse_estimation / outputs / held_out / "
            "map_nodes / parameter_tokens / development_tag / axes is frozen reader-facing copy "
            "and must be drawn verbatim. Nothing under caption_facts may be drawn."
        ),
        "canvas_mm": [190, 120],
        "outer_margin_mm": 3,
        "panel_gutter_mm": [3, 4],
        "canvas_note": (
            "Handoff section 4 (2026-08-20) sets 190 x 120 mm. six_figure_plan.md section 3.1 "
            "was reconciled to 120 mm on 2026-08-20; both notes now record the same decision."
        ),
        "figure_thesis": (
            "Sparse, disagreeing satellite observations condition a state-carrying daily water "
            "balance; its ET and applied-water outputs are evaluated with data that remain "
            "outside parameter estimation, and E1-derived parameter sets are tested through "
            "parallel transfer to E2 and E3."
        ),
        "reading_path": (
            "panel (a), top to bottom on one shared date axis: sparse ETf-member and NDVI "
            "captures, then daily forcing, then the conditioned daily state, then the two daily "
            "outputs, which point right across a vertical dashed rule into a narrow held-out "
            "evaluation region; one compact inverse-estimation loop takes the acquisition-date "
            "ETf targets and the auxiliary SWE constraint and returns conditioned parameters to "
            "the daily balance. Panel (b), left to right: the E1 CONUS source map, two "
            "class-specific parameter tokens, then the E2 world map and the E3 San Luis Valley "
            "map as the two parallel transfer endpoints."
        ),
        "panels": [
            {
                "id": "panel_a",
                "letter": "(a)",
                "title": "Sparse Satellite Constraints to Daily State",
                "purpose": (
                    "make the model's temporal logic visible with actual data: intermittent "
                    "satellite information conditions the model while daily drivers and stored "
                    "soil water generate an uninterrupted trajectory and two outputs"
                ),
            },
            {
                "id": "panel_b",
                "letter": "(b)",
                "title": "E1 Source Cohort and Parallel Transfer",
                "purpose": (
                    "combine empirical geography and transfer topology in three coordinated "
                    "maps joined by class-specific parameter paths"
                ),
            },
        ],
        "example_record": {
            "file": "fig01_example_timeseries.csv",
            "selection_record": "fig01_example_selection.json",
            "site_id": FIG01_EXAMPLE_SITE,
            "window": [FIG01_EXAMPLE_START, FIG01_EXAMPLE_END],
            "n_days": FIG01_EXAMPLE_DAYS,
            "n_calibration_captures": FIG01_EXAMPLE_CAPTURES,
            "role": (
                "cross-figure visual motif shared with Figure 3. Figure 1 shows data form, "
                "cadence and state propagation only; it makes no performance claim and duplicates "
                "no part of Figure 3's benchmark comparison."
            ),
            "site_id_is_visible": False,
            "site_id_note": (
                "the site identifier is caption/audit content, not artwork copy; the panel is "
                "labelled by evidence row, not by site"
            ),
        },
        "shared_time_grammar": (
            "All panel (a) rows share one date axis. Sparse observations stay visibly "
            "discontinuous and are never connected or smoothed to look complete; daily "
            "quantities stay visibly continuous. Sparse ticks and units establish scale."
        ),
        "evidence_rows": [
            {
                "id": "etf_ensemble",
                "order": 1,
                "heading": "ETf Ensemble",
                "heading_string_class": "title",
                "axis_label": "ETf",
                "axis_label_string_class": "direct_label",
                "columns": [f"etf_{m}" for m in E1_MEMBERS]
                + ["etf_target_mean", "etf_ensemble_spread", "etf_member_count"],
                "mark": (
                    "at each of the 15 calibration captures, plot every available retrieval "
                    "member as a small neutral mark and emphasize the target mean in "
                    "satellite-benchmark orange with a redundant symbol; member dispersion must "
                    "be visible without a member-name legend"
                ),
                "color_role": "satellite_et_target",
                "member_names_visible": False,
            },
            {
                "id": "ndvi_captures",
                "order": 2,
                "heading": "NDVI Captures",
                "heading_string_class": "title",
                "axis_label": "NDVI",
                "axis_label_string_class": "direct_label",
                "columns": ["ndvi_landsat_raw", "ndvi_sentinel_raw"],
                "mark": (
                    "raw Landsat and Sentinel-2 observations as distinct symbols, unconnected. "
                    "No daily filled NDVI trajectory is plotted; its fill provenance is "
                    "unresolved and the display file does not contain one."
                ),
                "filled_trace": False,
                "sensor_encoding": (
                    "shape first; the package sensor colors are used only if the two instruments "
                    "cannot be separated cleanly by shape alone"
                ),
            },
            {
                "id": "daily_forcing",
                "order": 3,
                "heading": "Daily Forcing",
                "heading_string_class": "title",
                "axis_label": "mm d⁻¹",
                "axis_label_string_class": "direct_label",
                "columns": ["eto", "precip"],
                "direct_labels": [
                    {"label": "ETo", "column": "eto", "string_class": "direct_label"},
                    {"label": "precipitation", "column": "precip", "string_class": "direct_label"},
                ],
                "mark": (
                    "a fine ETo line and compact precipitation bars on the shared date axis; "
                    "not a product inventory"
                ),
            },
            {
                "id": "daily_state",
                "order": 4,
                "heading": "Daily State",
                "heading_string_class": "title",
                "axis_label": "mm",
                "axis_label_string_class": "direct_label",
                "columns": ["rz_depletion", "irr_applied"],
                "direct_labels": [
                    {
                        "label": "root-zone depletion",
                        "column": "rz_depletion",
                        "string_class": "direct_label",
                    },
                    {
                        "label": "irrigation",
                        "column": "irr_applied",
                        "string_class": "direct_label",
                    },
                ],
                "mark": (
                    "root-zone depletion as one continuous line or band, aligned with the "
                    "precipitation bars above and with simulated irrigation events. This "
                    "replaces the removed generic plant-soil block and is the primary visual "
                    "evidence of state propagation."
                ),
                "color_role": "swim_state_and_output",
            },
            {
                "id": "daily_outputs",
                "order": 5,
                "heading": "Daily Outputs",
                "heading_string_class": "title",
                "axis_label": "mm d⁻¹",
                "axis_label_string_class": "direct_label",
                "columns": ["swim_ET", "irr_applied"],
                "direct_labels": [
                    {"label": "Daily ET", "column": "swim_ET", "string_class": "direct_label"},
                    {
                        "label": "Gross Applied Water",
                        "column": "irr_applied",
                        "string_class": "direct_label",
                    },
                ],
                "aggregation_mark": {
                    "label": "annual sum",
                    "string_class": "direct_label",
                    "treatment": (
                        "one compact tie or bracket glyph gathering the gross applied-water "
                        "events before the held-out divider, showing that daily applied water is "
                        "aggregated annually for the E3 meter comparison. Do not draw a sigma "
                        "glyph: the sigma character is reserved for the caption-owned objective "
                        "and weighting notation."
                    ),
                },
                "color_role": "swim_state_and_output",
            },
        ],
        "date_axis": {
            "label": "2017",
            "string_class": "direct_label",
            "columns": ["date"],
            "treatment": (
                "one shared date axis for all five evidence rows, drawn once at the bottom of "
                "panel (a) with sparse month ticks; tick text is generated from the data and is "
                "not frozen copy"
            ),
        },
        "inverse_estimation": {
            "id": "inverse_estimation",
            "label": "Inverse Estimation",
            "string_class": "title",
            "subtitle": "PEST++ IES",
            "subtitle_string_class": "direct_label",
            "subtitle_condition": (
                "secondary label; keep only if it remains legible and useful at final size "
                "(handoff section 5.3)"
            ),
            "spread_label": {
                "label": "spread-weighted",
                "string_class": "direct_label",
                "purpose": (
                    "the one short label permitted by handoff section 5.3 to identify why "
                    "retrieval-member dispersion is drawn"
                ),
                "caption_pattern_exemption": CAPTION_PATTERN_EXEMPTIONS["spread-weighted"],
            },
            "loop": (
                "acquisition-date ETf targets and the auxiliary SWE constraint route into one "
                "compact loop that returns conditioned parameters to the daily balance"
            ),
            "routing_rule": (
                "the inverse path and the driver path must be distinguishable by routing and "
                "stroke treatment. ETf and SWE constrain parameters; NDVI and daily forcing "
                "drive the forward balance. The figure must not imply that all inputs share the "
                "same dates or enter through the same mechanism."
            ),
            "color_role": "inverse_estimation",
            "forbidden_content": [
                "the objective equation",
                "the parameter inventory",
                "the realization count",
                "the iteration count",
                "the weighting formula",
            ],
        },
        "swe_constraint": {
            "id": "swe_constraint",
            "label": "SWE",
            "string_class": "direct_label",
            "column": "swe_audit",
            "treatment": (
                "gridded SWE is shown symbolically as an auxiliary calibration constraint "
                "entering the inverse-estimation loop. It is identically zero across this "
                "growing-season example, so no zero-valued SWE trace may be forced into the "
                "panel (handoff section 5.3)."
            ),
            "plotted_as_trace": False,
        },
        "daily_balance": {
            "id": "daily_balance",
            "represented_by": ["daily_state", "daily_outputs"],
            "label": None,
            "note": (
                "the daily balance has no titled box in 3.0.0. It is represented by the actual "
                "Daily State and Daily Outputs evidence rows; the removed generic crop-soil "
                "cross-section must not be restored."
            ),
        },
        "outputs": [
            {
                "id": "daily_et",
                "label": "Daily ET",
                "string_class": "direct_label",
                "column": "swim_ET",
                "compared_with": "flux_et",
            },
            {
                "id": "applied_water",
                "label": "Gross Applied Water",
                "string_class": "direct_label",
                "column": "irr_applied",
                "compared_with": "meters",
                "aggregation": "annually summed before the E3 meter comparison",
            },
        ],
        "held_out": {
            "heading": "Held-Out Evaluation",
            "heading_string_class": "title",
            "divider": (
                "vertical dashed rule on the far right of panel (a); one-way, with no return "
                "path and no evaluation-to-fitting arrow"
            ),
            "region_treatment": (
                "narrow and quiet; a light neutral gray may identify the region, but no tinted "
                "card may be placed around any other concept"
            ),
            "observations": [
                {
                    "id": "flux_et",
                    "label": "Flux ET · E1–E2",
                    "string_class": "direct_label",
                    "aligned_with": "daily_et",
                    "column": "flux_ET",
                    "treatment": (
                        "the US-Bi1 flux record may be drawn as a thin near-black trace or "
                        "observation rug to make the daily reference data concrete. No example "
                        "metric, residual, or emphasis on agreement is permitted."
                    ),
                },
                {
                    "id": "meters",
                    "label": "Metered Water · E3",
                    "string_class": "direct_label",
                    "aligned_with": "applied_water",
                    "column": None,
                    "treatment": (
                        "a compact data-form glyph with its direct label. No E3 meter values are "
                        "frozen in the Figure 1 package, so no meter mark may be drawn as data "
                        "unless actual frozen observations selected by an observation-support "
                        "rule independent of simulated performance are added first."
                    ),
                },
            ],
            "color_role": "held_out_observation",
        },
        "map_nodes": [
            {
                "id": "e1_map",
                "heading": "E1 · CONUS",
                "heading_string_class": "title",
                "count_line": "60 Cropland Sites",
                "count_line_string_class": "direct_label",
                "layer": "e1_sites",
                "rows": 60,
                "context_layer": "conus_context",
                "width_mm": [48, 52],
                "color_role": "e1",
                "treatment": (
                    "plot all 60 configured sites including MB_Pch on a quiet CONUS base with "
                    "enough internal geographic context to orient the reader; do not label "
                    "individual towers"
                ),
            },
            {
                "id": "e2_map",
                "heading": "E2 · 10 Countries",
                "heading_string_class": "title",
                "count_line": "66 Cropland Sites",
                "count_line_string_class": "direct_label",
                "layer": "e2_sites",
                "rows": 66,
                "context_layer": "world_context",
                "width_mm": [65, 72],
                "color_role": "e2",
                "treatment": (
                    "plot all 66 configured sites across ten countries and four continents; crop "
                    "empty polar latitudes and use a projection that gives the occupied "
                    "continents adequate area; one quiet boundary hierarchy, no choropleth. The "
                    "reader should perceive reach and clustering, not count sites."
                ),
            },
            {
                "id": "e3_map",
                "heading": "E3 · San Luis Valley",
                "heading_string_class": "title",
                "count_line": "50 Metered Fields",
                "count_line_string_class": "direct_label",
                "layer": "e3_display",
                "rows": 50,
                "context_layer": "slv_context",
                "width_mm": [30, 36],
                "color_role": "e3",
                "keep_on_one_line": "San Luis Valley",
                "treatment": (
                    "plot the 50 primary metered fields as the approved generalized display "
                    "(centroids snapped to a 1 km EPSG:5070 grid, reprojected for display). No "
                    "exact field polygons, source-agency identifiers, acreage, or "
                    "source-to-display linkage. If linked to the E1 locator, use a fine magenta "
                    "leader that survives grayscale."
                ),
            },
        ],
        "map_overlap_note": (
            "13 of the 66 E2 sites also occur in E1. Handoff section 6.2 keeps that fact in the "
            "caption: no additional overlap map symbol is drawn unless a final-size proof shows "
            "it improves rather than obscures the transfer story."
        ),
        "parameter_tokens": [
            {
                "id": "irrigated_params",
                "label": "Irrigated Parameters",
                "string_class": "direct_label",
                "source": "e1_map",
                "destinations": ["e2_map", "e3_map"],
                "symbol": "triangle",
            },
            {
                "id": "rainfed_params",
                "label": "Rainfed Parameters",
                "string_class": "direct_label",
                "source": "e1_map",
                "destinations": ["e2_map"],
                "symbol": "circle",
            },
        ],
        "parameter_token_treatment": (
            "two compact tokens or parallel paths occupying roughly 15-20 mm between the E1 map "
            "and the destination maps, visually subordinate to the maps. They encode direction "
            "and class only. No parameter names, values, priors, or internal run identifiers."
        ),
        "symbol_encoding": {
            "attribute": "irrigation_class",
            "source": "frozen E1/E2 source-class assignment used to construct the parameter "
            "sets, not annual irrigation activation",
            "shapes": {"irrigated": "triangle", "rainfed": "circle"},
            "all_e3_fields_use": "triangle",
            "all_e3_fields_reason": (
                "E3 evaluates only the irrigated transfer set, so every E3 field carries the "
                "irrigated shape"
            ),
            "color_encodes": "experiment, not class",
            "legend": (
                "none; the two parameter-token labels carry the class names and the shapes are "
                "read from them"
            ),
            "frozen_counts": {
                "e1_irrigated": FIG01_CLASS_COUNTS["e1_irrigated"],
                "e1_rainfed": FIG01_CLASS_COUNTS["e1_rainfed"],
                "e2_irrigated": FIG01_CLASS_COUNTS["e2_irrigated"],
                "e2_rainfed": FIG01_CLASS_COUNTS["e2_rainfed"],
            },
        },
        "development_tag": {
            "id": "e0_tag",
            "label": "E0 · Model-Form Selection",
            "string_class": "direct_label",
            "attached_to": "e1_map",
            "rendering": (
                "one small, subordinate tag adjacent to the E1 map. E0 is not a fourth map, not "
                "a coequal evaluation branch, and carries no flux-to-parameter arrow."
            ),
        },
        "cartography": {
            "shared_language": [
                "neutral land and restrained boundaries",
                "consistent point halo, opacity and apparent size",
                "comparable label hierarchy",
                "no heavy colored frames",
                "sparse graticules or scale bars only when they improve interpretation",
                "no north arrows on plainly north-up locator maps",
            ],
            "disclaimer_ownership": "caption",
        },
        "typography": {
            "family": "Source Sans 3, embedded, editable vector text",
            "panel_label_pt": [10, 11],
            "panel_heading_pt": [8.5, 9],
            "structural_label_pt": [8, 8.5],
            "direct_label_and_axis_pt": [7.5, 8],
            "minimum_reader_facing_pt": 7.5,
            "case_rule": (
                "title case for structural headings, sentence case for axis text and short "
                "explanatory phrases; conventional acronym capitalization preserved"
            ),
            "forbidden": ["overall title", "all-caps headings", "paragraph-like annotations"],
        },
        "color_roles": {
            "satellite_et_target": {
                "hex": "#E69F00",
                "redundant_encoding": "distinct symbol required",
            },
            "swim_state_and_output": {"hex": "#0072B2"},
            "inverse_estimation": {"hex": "#7B3294"},
            "held_out_observation": {"hex": "#202124"},
            "e1": {"hex": "#4477AA"},
            "e2": {"hex": "#228833"},
            "e3": {"hex": "#AA3377"},
            "background": "white",
            "held_out_region": "quiet gray permitted for this region only",
        },
        "line_weights_pt": {
            "axes_and_reference_rules": [0.5, 0.7],
            "data_strokes": [0.8, 1.2],
            "arrows": "consistent; no single arrow may become the dominant mark",
        },
        "forbidden_visual_treatments": [
            "gradients",
            "drop shadows",
            "glossy icons",
            "thick rounded boxes",
            "decorative satellite art",
            "generic crop artwork or a plant-soil cross-section",
            "a transfer ribbon or process-card matrix",
            "connecting or smoothing raw acquisition points",
            "a zero-valued SWE trace",
        ],
        "annotation_budget": {
            "maximum": 3,
            "used": 0,
            "panel_a": None,
            "panel_b": None,
            "rule": "seven words or fewer, at most two lines, sentence case, no full sentences",
            "note": (
                "Handoff section 9 targets almost no explanatory annotation. Zero are frozen. "
                "Density comes from real marks, shared axes, direct labels and symbol encodings."
            ),
        },
        "edges": [
            ["etf_ensemble", "inverse_estimation"],
            ["swe_constraint", "inverse_estimation"],
            ["inverse_estimation", "daily_balance"],
            ["ndvi_captures", "daily_balance"],
            ["daily_forcing", "daily_balance"],
            ["daily_balance", "daily_et"],
            ["daily_balance", "applied_water"],
            ["daily_et", "flux_et"],
            ["applied_water", "meters"],
            ["e0_tag", "e1_map"],
            ["e1_map", "irrigated_params"],
            ["e1_map", "rainfed_params"],
            ["irrigated_params", "e2_map"],
            ["irrigated_params", "e3_map"],
            ["rainfed_params", "e2_map"],
        ],
        "forbidden_edges": [
            ["flux_et", "inverse_estimation"],
            ["meters", "inverse_estimation"],
            ["flux_et", "daily_balance"],
            ["meters", "daily_balance"],
            ["flux_et", "irrigated_params"],
            ["meters", "irrigated_params"],
            ["flux_et", "rainfed_params"],
            ["meters", "rainfed_params"],
            ["flux_et", "e1_map"],
            ["meters", "e1_map"],
            ["flux_et", "e2_map"],
            ["meters", "e3_map"],
            ["e2_map", "e3_map"],
            ["e3_map", "e2_map"],
            ["e2_map", "irrigated_params"],
            ["e2_map", "rainfed_params"],
            ["rainfed_params", "e3_map"],
        ],
        "edge_rules": [
            "no arrow may run from a held-out observation into inverse estimation, the daily "
            "balance, or either class-specific parameter set",
            "no edge may originate at a held-out observation at all; model outputs point across "
            "the divider, never back",
            "both transfer paths originate at the E1 map; there is no E2-to-E3 edge",
            "E2 receives both parameter classes; E3 receives only the irrigated class",
            "the ETf/SWE constraint path and the NDVI/forcing driver path must be visually "
            "distinguishable by routing and stroke treatment",
            "panels (a) and (b) are not joined by an arrow in 3.0.0; the removed "
            "inverse_estimation -> e1_source connector belonged to the transfer-ribbon "
            "composition",
        ],
        "open_decisions": [
            "The calibration loop is frozen as the single parameter-update arrow "
            "inverse_estimation -> daily_balance. A literal closed cycle would require adding a "
            "daily_balance -> inverse_estimation edge here first; that decision is carried "
            "forward and is not yet made.",
            "No E3 meter observation values are frozen in the Figure 1 package, so the "
            "'Metered Water · E3' node is a data-form glyph. Drawing real meter marks requires "
            "freezing observations chosen by a support rule independent of simulated "
            "performance (handoff section 5.4).",
        ],
        "string_classification": {},
        "caption_facts": {
            "_ownership": "caption/manuscript-owned",
            "_rule": (
                "Recorded here for caption drafting and audit only. The builder never promotes "
                "any of these strings into visible copy, and an assertion fails the build if one "
                "appears among strings classified title, direct_label, or annotation."
            ),
            "contract": "paper/notes/fig01_production_handoff.md section 12, eight required items",
            "item_1_example_independence": (
                "Panel (a) shows one illustrative E1 growing-season record, selected without "
                "reference to evaluation performance; it supports no performance claim."
            ),
            "item_2_sparse_versus_daily": (
                "Acquisition-date satellite ETf members are calibration targets at 15 dates; raw "
                "Landsat and Sentinel-2 NDVI and daily meteorological forcing drive the forward "
                "balance, which carries soil-water state between acquisitions."
            ),
            "item_3_spread_weighting": (
                "Retrieval-member dispersion sets the relative weight of each acquisition-date "
                "target: captures on which the members disagree constrain the parameters less. "
                "This is relative reliability information, not calibrated uncertainty. The "
                "objective and the weight expression stay in the Methods."
            ),
            "item_4_outputs": (
                "The model produces daily ET and daily gross applied water; applied water is "
                "aggregated annually for comparison with the E3 meter records."
            ),
            "item_5_held_out": (
                "Flux ET and meter records were withheld from parameter estimation and from "
                "transfer-vector construction in every experiment. Arrows from model outputs to "
                "these observations denote post-hoc comparison only."
            ),
            "item_6_scope_and_classes": (
                "Configured scope is 60 CONUS cropland sites (E1), 66 cropland sites in ten "
                "countries on four continents (E2), and 50 metered San Luis Valley fields (E3). "
                "E1 supplies two class-specific parameter sets: irrigated (39 source sites) and "
                "rainfed (21 source sites). Both are applied to E2, which is assigned 13 "
                "irrigated and 53 rainfed; only the irrigated set is applied to E3. E2 comprises "
                "13 sites shared with E1 under changed inputs plus 53 new fields."
            ),
            "item_7_e0_disclosure": (
                "E0 and E1 share the 60-site CONUS cropland cohort. Flux ET did not enter "
                "parameter estimation, but E0 used flux ET after satellite calibration to select "
                "the cover-scaled sigmoid formulation carried into E1-E3. E1 flux evaluation is "
                "therefore external to parameter estimation but not fully independent of model "
                "development. E0 is model-development evidence, not independent validation, and "
                "not a fourth geography."
            ),
            "item_8_map_disclaimer": (
                "Map lines delineate study areas and do not necessarily depict accepted national "
                "boundaries."
            ),
            "working_caption": (
                "Fig. 1. Observation, state-propagation, evaluation, and transfer design of "
                "SWIM-RS. (a) An illustrative E1 growing-season record, selected without "
                "reference to evaluation performance, shows the temporal information entering "
                "and leaving the framework. Raw Landsat and Sentinel-2 NDVI observations "
                "represent vegetation dynamics; acquisition-date satellite ETf members and "
                "gridded SWE constrain time-invariant parameters through spread-weighted inverse "
                "estimation; and daily meteorological forcing drives a mass-conserving balance "
                "that carries soil-water state between acquisitions. The resulting daily ET and "
                "gross applied-water series are compared with flux ET in E1-E2 and annually "
                "aggregated meter records in E3, respectively. Flux and meter observations were "
                "withheld from parameter estimation and transferred-parameter construction. (b) "
                "The 60-site E1 CONUS cohort supplies separate irrigated and rainfed parameter "
                "sets. Both are applied without field-specific calibration across the 66-site, "
                "ten-country E2 experiment, whereas the irrigated set is applied to 50 metered "
                "fields in the San Luis Valley. E0 used the E1 cohort's flux observations after "
                "satellite calibration to select the vegetation formulation, so E1 flux "
                "evaluation is external to parameter estimation but not fully independent of "
                "model development. Map lines delineate study areas and do not necessarily "
                "depict accepted national boundaries."
            ),
            "working_caption_status": (
                "to be reconciled with the finished render (handoff section 12 and section 15 "
                "item 3)"
            ),
            "objective_notation": "Φ = Σ_i [w_i(ETf_sim,i − ETf_obs,i)]² + Σ_j [w_j(SWE_sim,j − SWE_obs,j)]²",
            "weighting_treatment": {
                "primary_experiments": "spread-weighted (all of E1, E2, E3 primaries)",
                "weight_rule": "w_i = ETf_obs,i / (σ_ensemble,i + 0.1); relative reliability information, not calibrated uncertainty",
                "fixed_scale_exception": "a fixed 0.33 scale is used only for the E2 ECOSTRESS-only sensitivity rows",
                "localization": "SWE observations update only the two snow parameters",
            },
            "retrieval_members": {
                "openet_v21_members": E1_MEMBERS,
                "E0": "SSEBop ETf, matched across candidate formulations",
                "E1": "per-capture mean of the six OpenET v2.1 ETf members",
                "E2": "per-capture mean of coincident Landsat SSEBop and PT-JPL ETf",
                "E2_sensitivity": "ECOSTRESS ETf on Landsat-gap dates at a fixed 0.33 scale",
                "E3": "per-capture mean of the six OpenET v2.1 ETf members, at least two valid members",
            },
            "calibration_settings": {
                "engine": "PEST++ IES",
                "prior_realizations": 200,
                "iterations": "three or four, by experiment",
                "n_parameters": 8,
            },
            "paired_support": {
                "E1": "45 daily sites; 29 finite-metric monthly sites (31 supported)",
                "E2": "63 daily sites; 50 finite-metric monthly sites (56 supported)",
                "E3": "408 metered field-years across 50 fields",
                "note": "configured scope (60 / 66 / 50) is distinct from paired evaluation support",
            },
            "example_site": (
                f"{FIG01_EXAMPLE_SITE}, {FIG01_EXAMPLE_START} to {FIG01_EXAMPLE_END}; "
                f"{FIG01_EXAMPLE_DAYS} days and {FIG01_EXAMPLE_CAPTURES} calibration captures"
            ),
            "ecostress_sensitivity": (
                "The E2 ECOSTRESS-only ablation adds capture dates on Landsat gaps at a fixed "
                "0.33 scale. The ECOSTRESS-to-Landsat ETf ratio is 0.756; the daily result is a "
                "wash and the monthly medians sit near the noise floor, so the ablation is "
                "reported as a sensitivity, not as the canonical target."
            ),
            "firewall_qualification": (
                "Flux ET and meter records were withheld from parameter estimation and from "
                "transfer-vector construction in every experiment. Arrows from model outputs to "
                "these observations denote post-hoc comparison only."
            ),
            "e0_qualification": (
                "E0 is a vegetation-formulation experiment on the same 60-site CONUS cropland "
                "cohort as E1. Flux ET did not enter parameter estimation, but E0 used flux ET "
                "after satellite calibration to select the cover-scaled sigmoid formulation "
                "carried by E1-E3. E1 flux evaluation is therefore external to fitting but not "
                "fully independent of model development. E0 is model-development evidence, not "
                "independent validation, and is not a fourth geography."
            ),
            "transfer_definition": (
                "The panel (b) tokens denote fixed E1-derived irrigation-class parameter sets: "
                "the irrigated and rainfed sets are applied to E2, and the irrigated set to E3. "
                "E2 does not supply parameters to E3."
            ),
            "map_disclaimer": (
                "Map lines delineate study areas and do not necessarily depict accepted national "
                "boundaries."
            ),
        },
    }

    # ---- frozen visible-string classification (handoff sections 9 and 10.3) ----
    cls: dict[str, str] = {}

    def _classify(s, k):
        if s is None:
            return
        if s in cls and cls[s] != k:
            raise BuildError(f"fig01: string {s!r} classified both {cls[s]} and {k}")
        cls[s] = k

    for p in arch["panels"]:
        _classify(p["letter"], "title")
        _classify(p["title"], "title")
    for row in arch["evidence_rows"]:
        _classify(row["heading"], row["heading_string_class"])
        _classify(row["axis_label"], row["axis_label_string_class"])
        for dl in row.get("direct_labels", []):
            _classify(dl["label"], dl["string_class"])
        agg = row.get("aggregation_mark")
        if agg:
            _classify(agg["label"], agg["string_class"])
    _classify(arch["date_axis"]["label"], arch["date_axis"]["string_class"])
    _classify(arch["inverse_estimation"]["label"], arch["inverse_estimation"]["string_class"])
    _classify(
        arch["inverse_estimation"]["subtitle"],
        arch["inverse_estimation"]["subtitle_string_class"],
    )
    _classify(
        arch["inverse_estimation"]["spread_label"]["label"],
        arch["inverse_estimation"]["spread_label"]["string_class"],
    )
    _classify(arch["swe_constraint"]["label"], arch["swe_constraint"]["string_class"])
    for o in arch["outputs"]:
        _classify(o["label"], o["string_class"])
    _classify(arch["held_out"]["heading"], arch["held_out"]["heading_string_class"])
    for o in arch["held_out"]["observations"]:
        _classify(o["label"], o["string_class"])
    for m in arch["map_nodes"]:
        _classify(m["heading"], m["heading_string_class"])
        _classify(m["count_line"], m["count_line_string_class"])
    for t in arch["parameter_tokens"]:
        _classify(t["label"], t["string_class"])
    _classify(arch["development_tag"]["label"], arch["development_tag"]["string_class"])
    arch["string_classification"] = dict(sorted(cls.items()))
    arch["visible_string_count"] = len(cls)
    arch["visible_strings_by_class"] = {
        k: sorted(s for s, c in cls.items() if c == k) for k in sorted(set(cls.values()))
    }

    if not set(cls.values()) <= VISIBLE_STRING_CLASSES:
        raise BuildError(f"fig01: unknown string class in {sorted(set(cls.values()))}")
    n_annot = sum(1 for v in cls.values() if v == "annotation")
    if n_annot > 3:
        raise BuildError(f"fig01: {n_annot} explanatory annotations exceed the budget of three")
    if arch["annotation_budget"]["used"] != n_annot:
        raise BuildError(
            f"fig01: annotation_budget.used={arch['annotation_budget']['used']} disagrees with "
            f"the {n_annot} string(s) classified as annotation"
        )
    for s, k in cls.items():
        if k == "annotation" and len(s.split()) > 7:
            raise BuildError(f"fig01: annotation {s!r} exceeds seven words")
    if len(cls) > 50:
        raise BuildError(f"fig01: {len(cls)} reader-facing strings exceed the limit of 50")
    _assert_no_caption_facts_visible(cls)
    # No superseded 2.1.0 string may survive into the 3.0.0 visible copy.
    revived = sorted(set(arch["supersedes"]["removed_strings"]) & set(cls))
    if revived:
        raise BuildError(f"fig01: superseded 2.1.0 string(s) {revived} reached visible copy")

    # ---- frozen edge assertions (handoff sections 5.4, 6.3, 11) ----
    edges = [tuple(e) for e in arch["edges"]]
    if len(set(edges)) != len(edges):
        raise BuildError("fig01: the frozen edge list contains duplicates")
    for src, dst in edges:
        if src in EVALUATION_NODES and dst in FITTING_NODES:
            raise BuildError(
                f"fig01: evaluation observation {src!r} has a directed edge into {dst!r}"
            )
        if src in EVALUATION_NODES:
            raise BuildError(f"fig01: no edge may originate at held-out observation {src!r}")
    for e in arch["forbidden_edges"]:
        if tuple(e) in edges:
            raise BuildError(f"fig01: forbidden edge {e} is present in the frozen edge list")
    for dest, tokens in FIG01_TRANSFER_DESTINATIONS.items():
        for tok in tokens:
            if (tok, dest) not in edges:
                raise BuildError(f"fig01: missing transfer path {tok!r} -> {dest!r}")
        inbound = {s for s, d in edges if d == dest}
        if inbound != set(tokens):
            raise BuildError(
                f"fig01: {dest!r} must be reached only from {sorted(tokens)}, got {sorted(inbound)}"
            )
    for tok in FIG01_PARAM_TOKENS:
        inbound = {s for s, d in edges if d == tok}
        if inbound != {FIG01_TRANSFER_SOURCE}:
            raise BuildError(
                f"fig01: parameter token {tok!r} must originate only at "
                f"{FIG01_TRANSFER_SOURCE!r}, got {sorted(inbound)}"
            )
    if ("e2_map", "e3_map") in edges or ("e3_map", "e2_map") in edges:
        raise BuildError("fig01: transfer paths must both originate at E1; no E2-to-E3 edge")
    if ("rainfed_params", "e3_map") in edges:
        raise BuildError("fig01: E3 evaluates the irrigated transfer set only")
    # Every classified string must belong to a node the edge list or the frozen
    # display package actually knows about; nothing may be drawn from nowhere.
    if (
        arch["example_record"]["n_days"] != n_ex
        or arch["example_record"]["n_calibration_captures"] != n_ex_caps
    ):
        raise BuildError(
            "fig01: the architecture's example-record shape disagrees with the frozen "
            "fig01_example_timeseries.csv"
        )

    (OUT / "fig01_architecture.json").write_text(json.dumps(arch, indent=2, ensure_ascii=False))

    meta = {
        "figure": "fig01",
        "architecture_schema_version": arch["schema_version"],
        "canvas_mm": [190, 120],
        "canvas_note": (
            "Target for the new evidence-bearing Gate A proof per fig01_production_handoff.md "
            "section 4 (rewritten 2026-08-20): 190 mm x 120 mm with 3 mm outer margins "
            "(usable 184 x 114 mm), two labelled horizontal panels separated by a 3-4 mm "
            "gutter, no overall title inside the artwork. The rejected 145 mm composition and "
            "the superseded 110 mm map-plus-framework composition must not be restored. "
            "six_figure_plan.md section 3.1 was reconciled to 120 mm on 2026-08-20."
        ),
        "crs": {
            "published_layers": "EPSG:4326",
            "e1_container_native": container_coord_crs(E1_CONTAINER),
            "e2_container_native": container_coord_crs(E2_CONTAINER),
            "e3_container_native": container_coord_crs(E3_CONTAINER),
            "e3_snapping_crs": "EPSG:5070",
            "note": (
                "Container geometry/lon,lat are stored in the native CRS of each container's "
                "source fields shapefile and are reprojected to EPSG:4326 here; every published "
                "layer is checked for coordinate/CRS-tag consistency before writing."
            ),
        },
        "boundary_dataset": {
            "name": "Natural Earth 110m admin 0 countries",
            "version": _ne_version(),
            "path": str(NE_COUNTRIES),
            "sha256": sha256(NE_COUNTRIES),
            "license": "public domain (Natural Earth)",
            "used_for": ["world_context", "conus_context", "E2 country and continent assignment"],
            "disclaimer": (
                "Map lines delineate study areas and do not necessarily depict accepted "
                "national boundaries."
            ),
        },
        "e3_generalization": {
            "method": "field centroid, reprojected to EPSG:5070, snapped to a 1000 m grid, reprojected to EPSG:4326",
            "grid_m": 1000,
            "deterministic": True,
            "seed": None,
            "approved": {
                "date": "2026-08-19",
                "outcome": "approved",
                "authority": "paper/notes/fig01_production_handoff.md section 6.4",
                "statement": (
                    "The approved generalized display is field centroids snapped to a 1 km grid, "
                    "frozen in the e3_display layer of fig01_scope.gpkg with the method recorded "
                    "here. This satisfies the section 15 item 2 approval requirement."
                ),
            },
            "removed_attributes": [
                "src_id (source-agency field identifier)",
                "source",
                "exact field polygon",
                "acres",
                "basin",
                "state",
                "original site_id",
            ],
            "retained_attributes": [
                "display_id",
                "geometry_method",
                "source_group",
                "source_key",
                "crop_group",
            ],
            "public_audit_key": {
                "attribute": "source_key",
                "construction": "sha256('swim-rs/fig01/e3-display/v1|' + src_id), first 16 hex characters",
                "property": (
                    "deterministic and one-to-one with the display records, but carries no "
                    "source-agency meter identifier; the source-to-display linkage is retained "
                    "only in restricted metadata outside this public package"
                ),
            },
        },
        "layer_attributes": {
            "e1_sites": list(e1_gdf.columns),
            "e2_sites": list(e2_gdf.columns),
            "e3_display": list(e3_disp.columns),
            "conus_context": list(conus_ctx.columns),
            "world_context": list(world_ctx.columns),
            "slv_context": list(slv_ctx.columns),
        },
        "cohort_assertions": {
            "e1_configured": int(len(e1)),
            "e2_configured": int(len(e2)),
            "e3_display": int(len(e3_disp)),
            "e1_e2_overlap": int(e1["in_e2"].sum()),
            "mb_pch_present_in_e1_scope": bool("MB_Pch" in set(e1["site_id"])),
            "e1_daily_evaluation": int(e1["in_daily_evaluation"].sum()),
            "e2_countries": n_countries,
            "e2_continents": n_continents,
            "e2_continent_names": continents,
            "e2_countries_names": countries,
            "e3_display_geometry_types": sorted(set(e3_disp.geom_type)),
            "evidence_matrix_rows": int(n_ev),
            "e0_evidence_role": roles.get("E0"),
            "visible_string_count": arch["visible_string_count"],
            "explanatory_annotation_count": n_annot,
            "example_rows": int(n_ex),
            "example_calibration_captures": int(n_ex_caps),
        },
        "class_assertions": {
            "attribute": "irrigation_class",
            "definition": (
                "the frozen source-class assignment used to construct the E1-derived transfer "
                "parameter sets, not annual irrigation activation (handoff section 6.3)"
            ),
            "e1_irrigated": class_counts["e1_irrigated"],
            "e1_rainfed": class_counts["e1_rainfed"],
            "e2_irrigated": class_counts["e2_irrigated"],
            "e2_rainfed": class_counts["e2_rainfed"],
            "e3_all_fields_class": "irrigated",
            "e3_class_reason": (
                "E3 evaluates only the irrigated transfer set, so every E3 field is drawn with "
                "the irrigated symbol"
            ),
            "symbol_encoding": {"irrigated": "triangle", "rainfed": "circle"},
            "layers": {"e1": "fig01_scope.gpkg::e1_sites", "e2": "fig01_scope.gpkg::e2_sites"},
        },
        "example_selection": {
            "record": "fig01_example_selection.json",
            "display_file": "fig01_example_timeseries.csv",
            "site_id": FIG01_EXAMPLE_SITE,
            "full_window": [FIG01_EXAMPLE_START, FIG01_EXAMPLE_END],
            "displayed_window": [FIG01_EXAMPLE_START, FIG01_EXAMPLE_END],
            "display_crop": None,
            "n_days": int(n_ex),
            "n_calibration_captures": int(n_ex_caps),
            "inherited_from": "fig03_example_selection.json",
            "selection_independence": example_selection["selection_independence"],
            "reconciliation": example_selection["reconciliation"],
            "refused_columns": sorted(example_selection["excluded_from_this_file"]),
            "column_provenance_recorded_for": sorted(example_prov),
        },
        "legacy_to_current_map": EXPERIMENT_MAP,
        "legacy_to_current_label_map": {
            "experiments": {
                "legacy e2_* / examples/5_Flux_Ensemble": "E1",
                "legacy e3_* / examples/6_Flux_International": "E2",
                "legacy e4_* / examples/7_Applied_Water": "E3",
                "legacy e1_* (removed broad-land-cover experiment)": None,
            },
            "reader_facing_strings": {
                "Landsat + Sentinel-2 NDVI": "NDVI · Vegetation State",
                "satellite ETf at retained captures": "ETf + SWE · Calibration Targets",
                "gridded SWE": "ETf + SWE · Calibration Targets",
                "meteorology · soils · land cover · irrigation status": "Gridded Forcing · Model Drivers",
                "surface evaporation layer / active root zone / below-root reservoir": "Daily Water Balance",
                "daily ET / ETf": "Daily ET",
                "gross applied water": "Gross Applied Water",
                "flux ET — E1 and E2": "Flux ET · E1–E2",
                "metered applied water — E3": "Metered Water · E3",
                "PEST++ IES calibration": "Inverse Estimation (subtitle PEST++ IES)",
                "withheld from parameter estimation and transfer-vector construction": "Held-Out Evaluation (full qualification moved to the caption)",
                "E0 model development": "E0 · Formulation Selection",
                "panel (c) evidence cards": "removed; replaced by the unlettered transfer ribbon",
            },
            "reader_facing_strings_note": (
                "The two maps above describe the SUPERSEDED architecture 2.1.0 copy. They are "
                "retained as provenance so an earlier proof can be traced; none of those strings "
                "is visible in 3.0.0."
            ),
            "reader_facing_strings_2026_08_20": {
                "Configured Geography": "Study Domains",
                "E1 · 60 CONUS Cropland Sites": "E1 · CONUS / 60 Cropland Sites",
                "E2 · 66 International Cropland Sites": "E2 · International / 66 Cropland Sites",
                "E3 · 50 San Luis Valley Fields": "E3 · San Luis Valley / 50 Metered Fields",
                "E2 · Geographic/Input Transfer": "E2 · Geographic and Input Transfer",
                "daily state propagation": None,
            },
            "reader_facing_strings_architecture_3_0_0": {
                "Study Domains": "E1 Source Cohort and Parallel Transfer",
                "SWIM-RS Framework": "Sparse Satellite Constraints to Daily State",
                "Satellite and Gridded Inputs": None,
                "NDVI · Vegetation State": "NDVI Captures",
                "ETf + SWE · Calibration Targets": "ETf Ensemble (+ the separate SWE constraint)",
                "Gridded Forcing · Model Drivers": "Daily Forcing",
                "Daily Water Balance": "Daily State (an actual root-zone depletion trace)",
                "E2 · International": "E2 · 10 Countries",
                "E1 · Source Parameters": "the E1 · CONUS map itself",
                "E2 · Geographic and Input Transfer": "the E2 · 10 Countries map itself",
                "E3 · Applied-Water Transfer": "the E3 · San Luis Valley map itself",
                "E0 · Formulation Selection": "E0 · Model-Form Selection",
                "the unlettered transfer ribbon": (
                    "removed; replaced by 'Irrigated Parameters' and 'Rainfed Parameters' tokens "
                    "drawn between the maps"
                ),
            },
            "metrics": {"legacy r2_*": "nse", "legacy bias_*": "mbe"},
        },
        "sources": {
            "e1_container": str(E1_CONTAINER),
            "e2_container": str(E2_CONTAINER),
            "e3_container": str(E3_CONTAINER),
            "e2_cohort_mapping": str(
                FINAL / "e3_irrigation_stratified_param_mapping_metadata.json"
            ),
            "e1_daily_cohort": str(FINAL / "e2_primary_daily_site_metrics.csv"),
            "boundary_dataset": str(NE_COUNTRIES),
            "example_series": {
                "path": "paper/data/final/figures/fig03_example_timeseries.csv",
                "sha256": sha256(OUT / "fig03_example_timeseries.csv"),
            },
            "example_member_values": {
                "path": "paper/data/final/figures/fig04_spread_capture_values.csv",
                "sha256": sha256(OUT / "fig04_spread_capture_values.csv"),
            },
            "example_upstream_selection": {
                "path": "paper/data/final/figures/fig03_example_selection.json",
                "sha256": sha256(OUT / "fig03_example_selection.json"),
            },
            "example_source_note": (
                "The Figure 1 example is built by joining two already-frozen, already-audited "
                "sibling artifacts in this package, not by re-deriving anything from a "
                "container. fig01 is therefore built AFTER fig03 and fig04 in a --all run."
            ),
        },
        "package_files": [
            "fig01_scope.gpkg",
            "fig01_evidence_matrix.csv",
            "fig01_example_timeseries.csv",
            "fig01_example_selection.json",
            "fig01_architecture.json",
            "fig01_metadata.json",
        ],
        "generator_script": "scripts/figures/build_figure_data.py",
        "generator_version": SCRIPT_VERSION,
        "fig01_builder_version": FIG01_BUILDER_VERSION,
        "data_contract": "paper/notes/fig01_production_handoff.md section 10, revised 2026-08-20",
        "review": {
            "scientific_review": {
                "date": "2026-08-20",
                "outcome": "data contract rebuilt for the evidence-first composition",
                "detail": (
                    "E0 typed as model_development; configured 60/66/50, the 13-site E1/E2 "
                    "overlap, MB_Pch presence, and the frozen source classes (E1 39 irrigated / "
                    "21 rainfed, E2 13 irrigated / 53 rainfed) asserted; spread-weighted "
                    "primaries and per-experiment ETf target composition asserted against the "
                    "section 11 label rules; the US-Bi1 example frozen from the audited fig03 "
                    "and fig04 artifacts with member marks reconciled to the frozen member "
                    "count and target mean; both transfer paths asserted to originate at E1 "
                    "with no E2-to-E3 edge and no evaluation-to-fitting edge."
                ),
                "supersedes": "2026-08-19 review of architecture 2.1.0",
            },
            "privacy_review": {
                "date": "2026-08-20",
                "outcome": "approved (unchanged)",
                "detail": (
                    "E3 display remains 1 km-snapped centroids; no exact polygons, no "
                    "source-agency identifiers, public source_key is a salted truncated hash. "
                    "The example record adds no restricted content: it is one public "
                    "AmeriFlux site. Open item carried forward: the E3 audit-key salt is "
                    "published in this public metadata file; the mechanism is left as-is."
                ),
            },
            "visual_review": {
                "date": "2026-08-20",
                "outcome": "composition superseded; no proof yet exists for 3.0.0",
                "detail": (
                    "The 190 x 110 mm map-plus-framework gray-box passed its budgets but read "
                    "as a graphical abstract rather than an analytical figure. The handoff was "
                    "rewritten and architecture 2.1.0 is superseded by 3.0.0: a 190 x 120 mm "
                    "two-panel evidence-first composition whose panel (a) plots the audited "
                    "US-Bi1 record on five aligned evidence rows and whose panel (b) carries "
                    "three coordinated maps joined by two class-specific parameter tokens. The "
                    "transfer ribbon, the grouped input cards and the crop-soil cross-section "
                    "are removed. A fresh, evidence-bearing Gate A proof is required; "
                    "fig01_graybox_110.png must not be revised into it."
                ),
                "gate_a_status": "not yet produced against architecture 3.0.0",
                "gate_b_status": "not started",
            },
        },
        "output_dimensions": {
            "canvas_mm": [190, 120],
            "outer_margin_mm": 3,
            "usable_mm": [184, 114],
            "panel_gutter_mm": [3, 4],
            "rendered": None,
        },
        "font_inventory": {
            "planned_family": "Source Sans 3, embedded, with editable vector text",
            "minimum_reader_facing_pt": 7.5,
            "normal_body_pt": 8,
            "panel_label_pt": [10, 11],
            "panel_heading_pt": [8.5, 9],
            "structural_label_pt": [8, 8.5],
            "direct_label_and_axis_pt": [7.5, 8],
            "math_face": "STIX Two Math or another embedded math face, only if notation is drawn",
            "measured": None,
            "note": "render-level font auditing belongs to the figure builder, not this data builder",
        },
    }
    (OUT / "fig01_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    MANIFEST.add(
        "fig01_scope.gpkg",
        rows={
            "e1_sites": len(e1_gdf),
            "e2_sites": len(e2_gdf),
            "e3_display": len(e3_disp),
            "conus_context": len(conus_ctx),
            "world_context": len(world_ctx),
            "slv_context": len(slv_ctx),
        },
        sources=meta["sources"],
        experiment_mapping=EXPERIMENT_MAP,
        cohort_key="display_id",
        inclusion_rule="Configured scope: all 60 E1 sites (MB_Pch retained), all 66 E2 sites, all 50 E3 SLV fields.",
        layer_attributes=meta["layer_attributes"],
        display_transformations=[
            "E1/E2 sites shown as container centroids, reprojected to EPSG:4326 from the "
            "native CRS of each container's source fields shapefile (E1 EPSG:5071, E2 EPSG:4326)",
            "E3 shown as 1 km snapped generalized centroids (approved 2026-08-19); no exact "
            "polygons or source-agency identifiers; source_key is a salted truncated SHA256",
            "E2 country/continent derived from the archived Natural Earth layer, not from the "
            "container's partly null country attribute, and published as 'country'",
            "conus_context is the Natural Earth USA polygon clipped to a -125..-66.5 E, 24..49.5 N box",
            "slv_context is a convex hull of the generalized E3 centroids buffered 0.15 degrees",
        ],
        units={"coordinates": "decimal degrees, EPSG:4326"},
        deterministic_seed=None,
        configured_counts={"E1": 60, "E2": 66, "E3": 50},
        evaluated_counts={"E1_daily": 45, "E2_daily": 63, "E3_field_years": 408},
        generator_version=FIG01_BUILDER_VERSION,
    )
    MANIFEST.add(
        "fig01_evidence_matrix.csv",
        rows=n_ev,
        note=(
            "E0 is typed evidence_role=model_development so generic plotting code cannot count "
            "it as a fourth evaluation geography. Configured, target-composition, weighting and "
            "independence labels are asserted against handoff section 11 before writing."
        ),
        cohort_key="experiment",
        inclusion_rule="One row per current evaluation experiment plus one E0 model-development record.",
        experiment_mapping=EXPERIMENT_MAP,
        deterministic_seed=None,
        generator_version=FIG01_BUILDER_VERSION,
    )
    MANIFEST.add(
        "fig01_example_timeseries.csv",
        rows=n_ex,
        note=(
            "The panel (a) example record: the audited US-Bi1 window inherited whole from the "
            "frozen Figure 3 example and joined to the frozen Figure 4 per-capture retrieval "
            "member values. Benchmark series and flags, every performance metric, and the daily "
            "filled NDVI trajectory are refused by assertion; every emitted column carries "
            "recorded provenance in fig01_example_selection.json."
        ),
        sources=example_selection["sources"],
        experiment_mapping={"E1": "legacy e2_*"},
        cohort_key="site_id + date",
        inclusion_rule=(
            f"single frozen site {FIG01_EXAMPLE_SITE}, {FIG01_EXAMPLE_START} to "
            f"{FIG01_EXAMPLE_END} inclusive, no display crop"
        ),
        temporal_support_rule=(
            "daily rows throughout; the six member ETf columns, the target mean, the member "
            "count and the ensemble spread are populated only on the 15 calibration captures "
            "and are null elsewhere by design -- sparse observations must remain visibly "
            "discontinuous"
        ),
        units={c: v["units"] for c, v in example_prov.items() if v.get("units")},
        display_transformations=[
            "read from the frozen fig03/fig04 package artifacts; nothing is re-derived from a "
            "container",
            "member marks reconciled to the frozen member count (exact) and to the target mean "
            f"and ensemble spread (max abs difference "
            f"{example_selection['reconciliation']['target_mean_max_abs_difference']:.3e} and "
            f"{example_selection['reconciliation']['spread_max_abs_difference']:.3e}, a float32 "
            "container-storage artifact)",
            "swe_audit is frozen as a non-plotted audit column: SWE is identically zero across "
            "this growing-season window and is represented symbolically, never as a zero trace",
        ],
        deterministic_seed=None,
        configured_counts={"E1": 60},
        evaluated_counts={
            "days_in_window": int(n_ex),
            "calibration_captures": int(n_ex_caps),
        },
        generator_version=FIG01_BUILDER_VERSION,
    )
    MANIFEST.add(
        "fig01_example_selection.json",
        rows=None,
        note=(
            "Site, full and displayed windows, rationale, source hashes, the member-mark "
            "reconciliation record, the per-column provenance map, and the explicit statement "
            "that neither the selection nor any crop used evaluation residuals or performance "
            "metrics."
        ),
        selected_site=FIG01_EXAMPLE_SITE,
        selected_window=[FIG01_EXAMPLE_START, FIG01_EXAMPLE_END],
        cohort_key="site_id",
        inclusion_rule="inherited whole from fig03_example_selection.json; no Figure 1 ranking",
        deterministic_seed=None,
        generator_version=FIG01_BUILDER_VERSION,
    )
    MANIFEST.add(
        "fig01_architecture.json",
        rows=None,
        note=(
            "Architecture 3.0.0 (supersedes 2.1.0): the two-panel evidence-first composition. "
            "Frozen reader-facing strings, the five panel (a) evidence rows and their source "
            "columns, the held-out boundary, the three map nodes, the two class-specific "
            "parameter tokens, the allowed and forbidden directed-edge lists, and a "
            "title/direct_label/annotation/proof_only classification of every visible string. "
            "Objective notation, retrieval members, weighting, realization and parameter counts, "
            "paired support, the ECOSTRESS sensitivity, the eight caption-contract items and the "
            "working caption live in a separate caption_facts block that the builder asserts "
            "never reaches visible copy."
        ),
        cohort_key=None,
        inclusion_rule=None,
        deterministic_seed=None,
        schema_version=arch["schema_version"],
        supersedes_schema_version=arch["supersedes"]["schema_version"],
        visible_string_count=arch["visible_string_count"],
        explanatory_annotations=n_annot,
        generator_version=FIG01_BUILDER_VERSION,
    )
    MANIFEST.add(
        "fig01_metadata.json",
        rows=None,
        note=(
            "Boundary dataset version and hash, CRS provenance, legacy-to-current label map, "
            "E3 generalization method with its 2026-08-19 approval, cohort and irrigation-class "
            "assertions, the example-selection reference, font inventory, 190 x 120 mm output "
            "dimensions, and the scientific / privacy / visual review record."
        ),
        cohort_key=None,
        inclusion_rule=None,
        deterministic_seed=None,
        generator_version=FIG01_BUILDER_VERSION,
    )
    print(
        f"  fig01: gpkg layers e1={len(e1_gdf)} e2={len(e2_gdf)} e3={len(e3_disp)} "
        f"(overlap {int(e1['in_e2'].sum())}); classes E1 "
        f"{class_counts['e1_irrigated']}i/{class_counts['e1_rainfed']}r, E2 "
        f"{class_counts['e2_irrigated']}i/{class_counts['e2_rainfed']}r; "
        f"evidence matrix {n_ev} rows; example {n_ex} rows / {n_ex_caps} captures; "
        f"architecture {arch['schema_version']} with {arch['visible_string_count']} visible "
        f"strings, {n_annot} annotation(s), {len(arch['edges'])} edges"
    )


def _ne_version() -> str:
    p = NE_COUNTRIES.with_suffix("").parent / "ne_110m_admin_0_countries.VERSION.txt"
    if p.exists():
        return p.read_text().strip()
    return "unknown"


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

BUILDERS = {
    "fig01": lambda: build_fig01(),
    "fig02": lambda: build_fig02(),
    "fig03": lambda: build_fig03_example(build_fig03_deltas()),
    "fig04": lambda: build_fig04(),
    "fig05_e1": lambda: build_fig05_e1(),
    "fig05_e2": lambda: build_fig05_e2(),
    "fig06": lambda: build_fig06(),
    "fig06_bootstrap": lambda: build_fig06_bootstrap(),
    "obs_support": lambda: build_obs_support(),
}

# fig01 consumes the frozen fig03 example series and fig04 capture values, so a
# --all run must build it last.  Every other builder is independent.
BUILD_ORDER = [
    "fig02",
    "fig03",
    "fig04",
    "fig05_e1",
    "fig05_e2",
    "fig06",
    "fig06_bootstrap",
    "obs_support",
    "fig01",
]
if set(BUILD_ORDER) != set(BUILDERS):
    raise RuntimeError("BUILD_ORDER and BUILDERS disagree")


def main(argv=None) -> int:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="build every table")
    ap.add_argument("--only", action="append", choices=sorted(BUILDERS), default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.only:
        targets = sorted(set(args.only), key=BUILD_ORDER.index)
    elif args.all:
        targets = list(BUILD_ORDER)
    else:
        targets = None
    if targets is None:
        ap.error("pass --all or --only NAME")

    failures = []
    for name in targets:
        print(f"[{name}]")
        try:
            BUILDERS[name]()
        except Exception as exc:  # noqa: BLE001 - reported, then recorded
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            MANIFEST.block(name, f"{type(exc).__name__}: {exc}")

    MANIFEST.block(
        "figs02 NDVI interpolation / filling classification",
        "blocked pending Section 4.1 reconciliation",
        "The implemented input path interpolates NDVI across gaps up to 100 steps and then "
        "back/forward fills without limit, while main.md and supp.md describe nearest-value "
        "extension only at record endpoints. No interpolation fraction or fill "
        "classification is computed or frozen, and fig03_example_timeseries.csv carries no "
        "filled-NDVI trace.",
    )
    p = MANIFEST.write()
    print(f"\nmanifest: {p}")
    if failures:
        print(f"{len(failures)} builder(s) failed; see BUILD_STATUS.md", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
