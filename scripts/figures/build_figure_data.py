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
FIG01_BUILDER_VERSION = "2.2.0"

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
# Archived 2026-08-24 for the section 6.4 / 6.7 CONUS state-boundary context test.
NE_STATES = REPO / "data" / "cartographic" / "ne_50m_admin_1_states_provinces_lakes.shp"
# Archived 2026-08-24 for the section 6.6 / 6.7 privacy-safe E3 basin context test.
SLV_BASIN = REPO / "data" / "cartographic" / "wbd_huc8_rio_grande_headwaters.fgb"
SLV_BASIN_SOURCE = REPO / "data" / "cartographic" / "wbd_huc8_rio_grande_headwaters.SOURCE.json"

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
    "E0_configured": 60,
    "E0_pooled_sites": 45,
    "E0_pooled_daily": 63681,
    "E0_pooled_monthly": 1435,
    "E0_effect_daily_sites": 45,
    "E0_effect_monthly_sites": 31,
    "E0_iso_daily_wins": 43,
    "E0_iso_monthly_wins": 27,
    "E1_configured": 60,
    "E1_daily": 45,
    "E1_monthly_finite": 29,
    "E1_transfer_daily": 45,
    "E1_transfer_monthly": 31,
    "E1_split_common": 43,
    "E1_pool_acquisition": 4751,
    "E1_pool_between": 55584,
    "E1_pool_total": 60335,
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
        # Set by main() on an --only run: the builder names that ran. A partial
        # run merges the prior on-disk manifest so records for builders that did
        # not run are carried forward instead of dropped.
        self.partial_targets: list[str] | None = None

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
        tables = self.tables
        status = self.status
        if self.partial_targets is not None:
            prior_path = OUT / self.MANIFEST_NAME
            if not prior_path.exists():
                raise RuntimeError(
                    "--only run requires an existing fig_manifest.json to merge into; "
                    "run --all first to establish the full package manifest"
                )
            prior = json.loads(prior_path.read_text())
            # Carry forward table records for builders that did not run, but only
            # while their output file is still present on disk.
            tables = {
                name: meta
                for name, meta in prior.get("tables", {}).items()
                if name not in self.tables and name in outputs
            }
            tables.update(self.tables)
            # Carry forward blocked/incomplete records unless this run replaced
            # them or successfully re-ran the builder they describe.
            current_items = {s["item"] for s in self.status}
            ran = set(self.partial_targets)
            status = [
                s
                for s in prior.get("blocked_or_incomplete", [])
                if s["item"] not in current_items and s["item"] not in ran
            ] + self.status
        for name, meta in tables.items():
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
            "last_run": {
                "partial": self.partial_targets is not None,
                "targets": self.partial_targets or "all",
            },
            "tables": tables,
            "blocked_or_incomplete": status,
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
# Figure 2 -- cover scaling makes the vegetation formulation coherent (E0)
# --------------------------------------------------------------------------

# E0 vegetation-formulation trio (six_figure_plan.md section 6, redesigned
# 2026-08-27; provenance in
# examples/5_Flux_Ensemble/notes/e0_ex5_native_results.md).  Three arms
# calibrated on the identical E1 container, ETf-ensemble target, IES budget,
# and prior families; only the vegetation formulation (and, for the linear
# arm, its formulation-specific slope/intercept priors) differs.  Internal
# run labels are provenance only and never appear in a reader-facing field.
E0_RESULTS = Path("/data/ssd1/swim/5_Flux_Ensemble/results")

E0_ARMS = {
    "cover_scaled_sigmoid": {
        "display_label": "Cover-scaled sigmoid",
        "run_label": "run22",
        "par_csv": E0_RESULTS / "run22" / "5_Flux_Ensemble.3.par.csv",
        "kcb_ndvi_mode": "sigmoid",
        "transpiration_cover_mode": "kcb",
        "veg_params": ("ndvi_k", "ndvi_0"),
        "config": "examples/5_Flux_Ensemble/5_Flux_Ensemble.toml",
    },
    "unscaled_linear": {
        "display_label": "Unscaled linear",
        "run_label": "RunFAO56",
        "par_csv": E0_RESULTS / "RunFAO56" / "5_Flux_Ensemble.3.par.csv",
        "kcb_ndvi_mode": "linear",
        "transpiration_cover_mode": "none",
        "veg_params": ("ndvi_alpha", "ndvi_beta"),
        "config": "examples/5_Flux_Ensemble/5_Flux_Ensemble_fao56.toml",
    },
    "unscaled_sigmoid": {
        "display_label": "Unscaled sigmoid",
        "run_label": "fao56_sig",
        "par_csv": E0_RESULTS / "fao56_sig" / "5_Flux_Ensemble.3.par.csv",
        "kcb_ndvi_mode": "sigmoid",
        "transpiration_cover_mode": "none",
        "veg_params": ("ndvi_k", "ndvi_0"),
        "config": "examples/5_Flux_Ensemble/5_Flux_Ensemble_fao56_sig.toml",
    },
}

# Paired-arm comparison archives (examples/5_Flux_Ensemble/pooled_arm_compare
# output).  The cover-scaled arm is arm A in the two reader-facing contrasts;
# the shape-only pair closes the triangle for cross-file consistency checks
# and never reaches a display table on its own.
E0_COMPARISONS = {
    "isolated_cover": {
        "dir": E0_RESULTS / "fao56_sig" / "comparison",
        "arm_a": "cover_scaled_sigmoid",
        "arm_b": "unscaled_sigmoid",
    },
    "whole_formulation": {
        "dir": E0_RESULTS / "RunFAO56" / "comparison",
        "arm_a": "cover_scaled_sigmoid",
        "arm_b": "unscaled_linear",
    },
    "shape_only": {
        "dir": E0_RESULTS / "fao56_sig" / "comparison_vs_fao56std",
        "arm_a": "unscaled_sigmoid",
        "arm_b": "unscaled_linear",
    },
}

# Fixed coefficients shared by every E0 arm, verified against
# src/swimrs/process/input.py (kc_max = np.full(n, 1.35) with empirical_kc_max
# unset in all three configs; kc_min = np.full(n, 0.15)) and the clip
# semantics in src/swimrs/process/loop_fast.py / cover_modes.py.
E0_KC_MAX = 1.35
E0_KC_MIN = 0.15
E0_FC_MAX = 0.99
E0_SIGMOID_EXP_CLIP = 20.0

# Non-vegetation parameters carried identically (same priors) by every arm.
E0_SHARED_PARAMS = ("aw", "ks_alpha", "kr_alpha", "mad", "swe_alpha", "swe_beta")

E0_NDVI_GRID = np.round(np.linspace(0.0, 1.0, 101), 2)
E0_SUPPORT_BINS = np.round(np.linspace(0.0, 1.0, 51), 2)
E0_MASK_ID = "e0_arm_paired_flux_mask_45sites_63681d_1435mo"

E0_SUPPORT_RULE = (
    "site-equal weighting: every finite merged-NDVI observation carries weight 1/n_site so "
    "each of the 60 E0 sites contributes equal total weight; density integrates to 1 over NDVI"
)

# Table 3 (paper/text/main.md) at manuscript precision.  The build fails when
# a frozen pooled value does not reproduce its manuscript string exactly.
E0_TABLE3 = {
    ("cover_scaled_sigmoid", "daily", "rmse"): "1.097",
    ("cover_scaled_sigmoid", "daily", "mbe"): "0.000",
    ("cover_scaled_sigmoid", "daily", "kge"): "0.860",
    ("cover_scaled_sigmoid", "monthly", "rmse"): "19.21",
    ("cover_scaled_sigmoid", "monthly", "mbe"): "0.31",
    ("cover_scaled_sigmoid", "monthly", "kge"): "0.945",
    ("unscaled_linear", "daily", "rmse"): "1.212",
    ("unscaled_linear", "daily", "mbe"): "0.075",
    ("unscaled_linear", "daily", "kge"): "0.846",
    ("unscaled_linear", "monthly", "rmse"): "22.10",
    ("unscaled_linear", "monthly", "mbe"): "2.39",
    ("unscaled_linear", "monthly", "kge"): "0.934",
    ("unscaled_sigmoid", "daily", "rmse"): "1.270",
    ("unscaled_sigmoid", "daily", "mbe"): "0.209",
    ("unscaled_sigmoid", "daily", "kge"): "0.818",
    ("unscaled_sigmoid", "monthly", "rmse"): "25.12",
    ("unscaled_sigmoid", "monthly", "mbe"): "7.24",
    ("unscaled_sigmoid", "monthly", "kge"): "0.862",
}

# calc_metrics (examples/5_Flux_Ensemble/evaluate.py) returns NaN below ten
# paired values, so per-site monthly metrics exist only for sites with at
# least ten qualifying months even though pooled monthly totals admit sites
# from six months up.
E0_SITE_METRIC_MIN_MONTHS = 10


def _e0_table3_format(scale: str, metric: str, value: float) -> str:
    """Format a pooled value exactly as Table 3 prints it (sign retained)."""
    nd = 3 if (scale == "daily" or metric == "kge") else 2
    s = f"{value:.{nd}f}"
    if s == "-" + f"{0.0:.{nd}f}":
        s = s[1:]
    return s


def _e0_par_medians(arm_key: str, uids: list[str]):
    """Posterior-median parameter vector per site for one E0 arm.

    Mirrors examples/5_Flux_Ensemble/evaluate.py::parse_pest_params: the median
    is taken across IES realizations excluding the ``base`` row, and parameter
    columns are matched to sites by lowercase suffix.  Returns
    ``({uid: {param: median}}, n_realizations)``.
    """
    arm = E0_ARMS[arm_key]
    df = pd.read_csv(arm["par_csv"], index_col=0)
    reals = df.loc[df.index.astype(str) != "base"]
    n_real = len(reals)
    if n_real < 50:
        raise BuildError(f"fig02 {arm_key}: only {n_real} IES realizations in {arm['par_csv']}")
    med = reals.median()
    by_len = sorted(uids, key=len, reverse=True)
    by_site: dict[str, dict[str, float]] = {u: {} for u in uids}
    for col in df.columns:
        core = col.split("_ptype:")[0].replace("pname:p_", "").rsplit("_:0", 1)[0]
        site = next((u for u in by_len if core.lower().endswith("_" + u.lower())), None)
        if site is None:
            raise BuildError(f"fig02 {arm_key}: column {col!r} matches no container site")
        pname = core[: -(len(site) + 1)]
        if pname in by_site[site]:
            raise BuildError(f"fig02 {arm_key}: duplicate parameter {pname} for {site}")
        by_site[site][pname] = float(med[col])
    expected = set(E0_SHARED_PARAMS) | set(arm["veg_params"])
    for u in uids:
        if set(by_site[u]) != expected:
            raise BuildError(
                f"fig02 {arm_key}: site {u} carries {sorted(by_site[u])}, "
                f"expected {sorted(expected)}"
            )
    return by_site, n_real


def _e0_kt(arm_key: str, ndvi: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """Potential canopy-transpiration multiplier K_T on an NDVI grid.

    Replicates the Kcb and fc arithmetic of src/swimrs/process/loop_fast.py
    (sigmoid exponent clip +-20, linear Kcb clip [0, Kc_max], fc clip
    [0, 0.99]) with Ks = 1 and the soil-evaporation term omitted.
    """
    arm = E0_ARMS[arm_key]
    if arm["kcb_ndvi_mode"] == "linear":
        kcb = p["ndvi_beta"] * ndvi + p["ndvi_alpha"]
    else:
        e = np.clip(-p["ndvi_k"] * (ndvi - p["ndvi_0"]), -E0_SIGMOID_EXP_CLIP, E0_SIGMOID_EXP_CLIP)
        kcb = E0_KC_MAX / (1.0 + np.exp(e))
    kcb = np.clip(kcb, 0.0, E0_KC_MAX)
    if arm["transpiration_cover_mode"] == "kcb":
        fc = np.clip((kcb - E0_KC_MIN) / (E0_KC_MAX - E0_KC_MIN), 0.0, E0_FC_MAX)
        return fc * kcb
    if arm["transpiration_cover_mode"] != "none":
        raise BuildError(f"fig02 {arm_key}: unhandled cover mode")
    return kcb


def build_fig02() -> None:
    import zarr

    srcs: dict[str, Path] = {"container": E1_CONTAINER}
    for key, arm in E0_ARMS.items():
        srcs[f"par:{key}"] = arm["par_csv"]
        srcs[f"config:{key}"] = REPO / arm["config"]
    for key, comp in E0_COMPARISONS.items():
        srcs[f"gate:{key}"] = comp["dir"] / "pooled_gate.json"
        srcs[f"per_site:{key}"] = comp["dir"] / "pooled_per_site.csv"
    for k, p in srcs.items():
        if not p.exists():
            raise BuildError(f"fig02 source missing: {k} -> {p}")

    # ---- cohort and observed-NDVI support (panel a underlay) ----
    z = zarr.open(str(E1_CONTAINER), mode="r")
    uids = [str(x) for x in z["geometry/uid"][:]]
    require_count(len(uids), EXPECTED["E0_configured"], "fig02 E0 configured cohort")
    ndvi = np.asarray(z["derived/merged_ndvi/no_mask"])
    if ndvi.ndim != 2 or ndvi.shape[1] != len(uids):
        raise BuildError(f"fig02 support: merged NDVI shape {ndvi.shape} != (time, {len(uids)})")
    ndvi_hash = hashlib.sha256(
        np.ascontiguousarray(ndvi).tobytes() + "|".join(uids).encode()
    ).hexdigest()
    finite = np.isfinite(ndvi)
    per_site_obs = finite.sum(axis=0)
    if int(per_site_obs.min()) < 100:
        raise BuildError("fig02 support: a site has fewer than 100 NDVI observations")
    lo, hi = float(np.nanmin(ndvi)), float(np.nanmax(ndvi))
    if lo < 0.0 or hi > 1.0:
        raise BuildError(f"fig02 support: merged NDVI outside [0, 1] ({lo:.3f}, {hi:.3f})")

    width = float(E0_SUPPORT_BINS[1] - E0_SUPPORT_BINS[0])
    dens = np.zeros(len(E0_SUPPORT_BINS) - 1)
    n_obs = np.zeros(len(E0_SUPPORT_BINS) - 1, dtype=int)
    n_sites_bin = np.zeros(len(E0_SUPPORT_BINS) - 1, dtype=int)
    for j in range(len(uids)):
        h, _ = np.histogram(ndvi[finite[:, j], j], bins=E0_SUPPORT_BINS)
        dens += h / float(per_site_obs[j])
        n_obs += h
        n_sites_bin += (h > 0).astype(int)
    dens /= len(uids) * width
    if abs(float(dens.sum()) * width - 1.0) > 1e-9:
        raise BuildError("fig02 support: site-equal density does not integrate to 1")
    if int(n_obs.sum()) != int(finite.sum()):
        raise BuildError("fig02 support: binned observation count mismatch")
    support = pd.DataFrame(
        {
            "bin_left": E0_SUPPORT_BINS[:-1],
            "bin_right": E0_SUPPORT_BINS[1:],
            "density_site_equal": dens,
            "n_sites": n_sites_bin,
            "n_obs": n_obs,
            "support_rule": E0_SUPPORT_RULE,
            "source_sha256": ndvi_hash,
        }
    )

    # ---- fitted vegetation response distributions (panel a) ----
    par_hashes: dict[str, str] = {}
    veg_medians: dict[str, dict[str, dict[str, float]]] = {}
    n_reals: dict[str, int] = {}
    resp_frames = []
    for key, arm in E0_ARMS.items():
        by_site, n_real = _e0_par_medians(key, uids)
        par_hashes[key] = sha256(arm["par_csv"])
        n_reals[key] = n_real
        veg_medians[key] = {u: {p: by_site[u][p] for p in arm["veg_params"]} for u in uids}
        src_label = (
            f"posterior median over {n_real} IES realizations (iteration 3) of {arm['par_csv']}"
        )
        for u in uids:
            kt = _e0_kt(key, E0_NDVI_GRID, by_site[u])
            if not np.isfinite(kt).all() or kt.min() < 0.0 or kt.max() > E0_KC_MAX:
                raise BuildError(f"fig02 response: K_T out of [0, kc_max] for {key}/{u}")
            resp_frames.append(
                pd.DataFrame(
                    {
                        "formulation": key,
                        "site_id": u,
                        "ndvi": E0_NDVI_GRID,
                        "k_t": kt,
                        "parameter_source": src_label,
                        "source_sha256": par_hashes[key],
                    }
                )
            )
    resp = pd.concat(resp_frames, ignore_index=True)
    require_count(len(resp), 3 * len(uids) * len(E0_NDVI_GRID), "fig02 response rows")
    require_unique(resp, ["formulation", "site_id", "ndvi"], "fig02 response")

    # ---- pooled held-out agreement on the Table 3 mask (panel b) ----
    gates = {
        key: json.loads((comp["dir"] / "pooled_gate.json").read_text())
        for key, comp in E0_COMPARISONS.items()
    }
    for key, g in gates.items():
        comp = E0_COMPARISONS[key]
        for side in ("a", "b"):
            arm = E0_ARMS[comp[f"arm_{side}"]]
            if g[f"arm_{side}"] != arm["run_label"]:
                raise BuildError(f"fig02 gate {key}: arm_{side} is not {arm['run_label']}")
            want_phys = {
                "kcb_ndvi_mode": arm["kcb_ndvi_mode"],
                "transpiration_cover_mode": arm["transpiration_cover_mode"],
            }
            if g[f"{side}_physics"] != want_phys:
                raise BuildError(f"fig02 gate {key}: {side}_physics != stated formulation")
            if Path(g[f"par_{side}"]) != arm["par_csv"]:
                raise BuildError(f"fig02 gate {key}: par_{side} != panel-(a) parameter source")
            if Path(g[f"{side}_config"]).name != Path(arm["config"]).name:
                raise BuildError(f"fig02 gate {key}: {side}_config != stated arm config")
        require_count(g["n_sites"], EXPECTED["E0_pooled_sites"], f"fig02 gate {key} sites")
        require_count(g["n_daily"], EXPECTED["E0_pooled_daily"], f"fig02 gate {key} site-days")
        require_count(g["n_monthly"], EXPECTED["E0_pooled_monthly"], f"fig02 gate {key} months")

    pooled_vals: dict[tuple[str, str, str], list[float]] = {}
    for key, g in gates.items():
        comp = E0_COMPARISONS[key]
        for m in g["metrics"]:
            n_want = EXPECTED["E0_pooled_daily" if m["scale"] == "daily" else "E0_pooled_monthly"]
            require_count(m["n"], n_want, f"fig02 gate {key} {m['scale']} {m['metric']} support")
            for side in ("a", "b"):
                form = comp[f"arm_{side}"]
                run = E0_ARMS[form]["run_label"]
                pooled_vals.setdefault((form, m["scale"], m["metric"].lower()), []).append(
                    float(m[run])
                )

    gates_hash = hashlib.sha256(
        "".join(
            sha256(E0_COMPARISONS[k]["dir"] / "pooled_gate.json") for k in sorted(gates)
        ).encode()
    ).hexdigest()
    pooled_rows = []
    table3_record = {}
    for form in E0_ARMS:
        for scale in ("daily", "monthly"):
            for metric in ("kge", "rmse", "mbe"):
                vals = pooled_vals.pop((form, scale, metric))
                if len(vals) != 2 or vals[0] != vals[1]:
                    raise BuildError(
                        f"fig02 pooled: {form}/{scale}/{metric} inconsistent across "
                        f"comparison files: {vals}"
                    )
                v = vals[0]
                want = E0_TABLE3[(form, scale, metric)]
                got = _e0_table3_format(scale, metric, v)
                if got != want:
                    raise BuildError(
                        f"fig02 pooled: {form}/{scale}/{metric} = {v!r} formats to {got}, "
                        f"Table 3 says {want}"
                    )
                if want != "0.000" and v <= 0.0:
                    raise BuildError(f"fig02 pooled: {form}/{scale}/{metric} sign flip")
                table3_record[f"{form}|{scale}|{metric}"] = {"value": v, "manuscript": want}
                unit = (
                    "dimensionless"
                    if metric == "kge"
                    else ("mm d-1" if scale == "daily" else "mm month-1")
                )
                pooled_rows.append(
                    {
                        "formulation": form,
                        "scale": scale,
                        "metric": metric,
                        "value": v,
                        "unit": unit,
                        "n_sites": EXPECTED["E0_pooled_sites"],
                        "n_paired": EXPECTED[
                            "E0_pooled_daily" if scale == "daily" else "E0_pooled_monthly"
                        ],
                        "evaluation_mask_id": E0_MASK_ID,
                        "manuscript_value": want,
                        "source_sha256": gates_hash,
                    }
                )
    if pooled_vals:
        raise BuildError(f"fig02 pooled: unexpected extra entries {sorted(pooled_vals)}")
    pooled = pd.DataFrame(pooled_rows)
    require_count(len(pooled), 18, "fig02 pooled rows")

    # ---- paired site RMSE effects (panel c) ----
    ps_cols = [
        "fid",
        "n_daily",
        "rmse_a_daily",
        "rmse_b_daily",
        "n_monthly",
        "rmse_a_monthly",
        "rmse_b_monthly",
    ]
    eff_frames = []
    win_counts: dict[tuple[str, str], int] = {}
    scale_sets: dict[tuple[str, str], frozenset] = {}
    per_site_hashes: dict[str, str] = {}
    for comp_key in ("isolated_cover", "whole_formulation"):
        comp = E0_COMPARISONS[comp_key]
        ps_path = comp["dir"] / "pooled_per_site.csv"
        ps = pd.read_csv(ps_path)
        label = f"fig02 per-site {comp_key}"
        require_columns(ps, ps_cols, label)
        require_unique(ps, ["fid"], label)
        require_count(len(ps), EXPECTED["E0_effect_daily_sites"], f"{label} daily sites")
        unknown = sorted(set(ps["fid"].astype(str)) - set(uids))
        if unknown:
            raise BuildError(f"{label}: fids not in the E0 container cohort: {unknown}")
        if ps["rmse_a_daily"].isna().any() or ps["rmse_b_daily"].isna().any():
            raise BuildError(f"{label}: missing daily RMSE")
        per_site_hashes[comp_key] = sha256(ps_path)
        mo_fin = ps["rmse_a_monthly"].notna() & ps["rmse_b_monthly"].notna()
        if (ps.loc[mo_fin, "n_monthly"] < E0_SITE_METRIC_MIN_MONTHS).any() or (
            ps.loc[~mo_fin, "n_monthly"] >= E0_SITE_METRIC_MIN_MONTHS
        ).any():
            raise BuildError(
                f"{label}: monthly-metric availability violates the "
                f">= {E0_SITE_METRIC_MIN_MONTHS} qualifying-months rule"
            )
        for scale, sub in (("daily", ps), ("monthly", ps[mo_fin])):
            if scale == "monthly":
                require_count(
                    len(sub), EXPECTED["E0_effect_monthly_sites"], f"{label} monthly sites"
                )
            scale_sets[(comp_key, scale)] = frozenset(sub["fid"].astype(str))
            a = sub[f"rmse_a_{scale}"].to_numpy(dtype=float)
            b = sub[f"rmse_b_{scale}"].to_numpy(dtype=float)
            win = a < b
            win_counts[(comp_key, scale)] = int(win.sum())
            eff_frames.append(
                pd.DataFrame(
                    {
                        "site_id": sub["fid"].astype(str).to_numpy(),
                        "scale": scale,
                        "comparator": comp_key,
                        "rmse_cover_scaled": a,
                        "rmse_unscaled": b,
                        "d_rmse": a - b,
                        "n_paired": sub[f"n_{scale}"].to_numpy(dtype=int),
                        "win_cover_scaled": win,
                        "source_sha256": per_site_hashes[comp_key],
                    }
                )
            )
    for scale in ("daily", "monthly"):
        if scale_sets[("isolated_cover", scale)] != scale_sets[("whole_formulation", scale)]:
            raise BuildError(f"fig02 effects: {scale} site sets differ between comparators")
    require_count(
        win_counts[("isolated_cover", "daily")],
        EXPECTED["E0_iso_daily_wins"],
        "fig02 isolated-cover daily wins",
    )
    require_count(
        win_counts[("isolated_cover", "monthly")],
        EXPECTED["E0_iso_monthly_wins"],
        "fig02 isolated-cover monthly wins",
    )
    eff = pd.DataFrame(pd.concat(eff_frames, ignore_index=True))
    require_count(
        len(eff),
        2 * (EXPECTED["E0_effect_daily_sites"] + EXPECTED["E0_effect_monthly_sites"]),
        "fig02 effect rows",
    )
    if not eff["d_rmse"].notna().all():
        raise BuildError("fig02 effects: non-finite d_rmse")

    # ---- quarantine the superseded External-ET-agreement package ----
    quarantined = []
    qdir = OUT / "superseded_fig02_et_agreement"
    for name in (
        "fig02_daily_site_metrics.csv",
        "fig02_monthly_site_metrics.csv",
        "fig02_site_effects.csv",
    ):
        p = OUT / name
        if p.exists():
            qdir.mkdir(exist_ok=True)
            p.rename(qdir / name)
            quarantined.append(name)

    # ---- write tables, metadata, and manifest records ----
    nr = write_table(resp, "fig02_formulation_response.csv")
    ns = write_table(support, "fig02_ndvi_support.csv")
    np_ = write_table(pooled, "fig02_pooled_metrics.csv")
    ne = write_table(eff, "fig02_site_rmse_effects.csv")

    meta_json = {
        "figure": "Figure 2 -- cover scaling makes the vegetation formulation coherent",
        "role": (
            "E0 model-development evidence: flux ET was excluded from calibration but used "
            "to select model form; E0 is not an independent validation experiment and its "
            "flux cohort is the E1 cohort"
        ),
        "formulations": {
            key: {
                "display_label": arm["display_label"],
                "kcb_ndvi_mode": arm["kcb_ndvi_mode"],
                "transpiration_cover_mode": arm["transpiration_cover_mode"],
                "veg_params": list(arm["veg_params"]),
                "run_label_provenance_only": arm["run_label"],
                "par_csv": str(arm["par_csv"]),
                "par_sha256": par_hashes[key],
                "config": arm["config"],
                "n_ies_realizations": n_reals[key],
            }
            for key, arm in E0_ARMS.items()
        },
        "equations": {
            "kcb_sigmoid": "Kcb = Kc_max / (1 + exp(-ndvi_k (NDVI - ndvi_0)))",
            "kcb_linear": "Kcb = clip(ndvi_beta NDVI + ndvi_alpha, 0, Kc_max)",
            "fc": "fc = clip((Kcb - Kc_min) / (Kc_max - Kc_min), 0, 0.99)",
            "k_t": (
                "K_T = fc Kcb for the cover-scaled formulation and K_T = Kcb for the "
                "unscaled formulations, with Ks = 1 and the common soil-evaporation term "
                "omitted; K_T is the potential canopy-transpiration multiplier, not total Kc"
            ),
        },
        "constants": {
            "kc_max": E0_KC_MAX,
            "kc_min": E0_KC_MIN,
            "fc_max": E0_FC_MAX,
            "sigmoid_exp_clip": E0_SIGMOID_EXP_CLIP,
            "provenance": (
                "src/swimrs/process/input.py (fixed kc_max 1.35 / kc_min 0.15 in every E0 "
                "arm; empirical_kc_max unset) and src/swimrs/process/loop_fast.py clip "
                "semantics"
            ),
        },
        "priors_note": (
            "the unscaled-linear arm carries formulation-specific slope/intercept priors "
            "(ndvi_alpha, ndvi_beta) in place of the logistic (ndvi_k, ndvi_0), so its "
            "ordering against the unscaled sigmoid is not an isolated curve-shape ablation; "
            "all non-vegetation priors are identical across arms"
        ),
        "ndvi_grid": {"start": 0.0, "stop": 1.0, "step": 0.01, "points": len(E0_NDVI_GRID)},
        "ndvi_support": {
            "rule": E0_SUPPORT_RULE,
            "source": "derived/merged_ndvi/no_mask of " + str(E1_CONTAINER),
            "sha256": ndvi_hash,
            "n_obs_total": int(finite.sum()),
            "per_site_obs_min": int(per_site_obs.min()),
            "per_site_obs_max": int(per_site_obs.max()),
            "bin_width": width,
        },
        "evaluation_mask": {
            "id": E0_MASK_ID,
            "n_sites": EXPECTED["E0_pooled_sites"],
            "n_daily": EXPECTED["E0_pooled_daily"],
            "n_monthly": EXPECTED["E0_pooled_monthly"],
            "pooled_month_rule": (
                "full calendar months with >= 28 paired flux days; a site contributes "
                "monthly totals when it has >= 6 such months"
            ),
            "site_metric_month_rule": (
                f">= {E0_SITE_METRIC_MIN_MONTHS} qualifying months (calc_metrics minimum "
                f"n = 10); {EXPECTED['E0_effect_monthly_sites']} of "
                f"{EXPECTED['E0_effect_daily_sites']} evaluation sites qualify"
            ),
        },
        "site_effects": {
            "d_rmse_sign": (
                "d_rmse = rmse_cover_scaled - rmse_unscaled; negative favours the "
                "cover-scaled formulation"
            ),
            "comparators": {
                "isolated_cover": "cover_scaled_sigmoid minus unscaled_sigmoid",
                "whole_formulation": "cover_scaled_sigmoid minus unscaled_linear",
            },
            "win_counts": {
                "isolated_cover": {
                    "daily": f"{win_counts[('isolated_cover', 'daily')]}/45",
                    "monthly": f"{win_counts[('isolated_cover', 'monthly')]}/31",
                },
                "whole_formulation": {
                    "daily": f"{win_counts[('whole_formulation', 'daily')]}/45",
                    "monthly": f"{win_counts[('whole_formulation', 'monthly')]}/31",
                },
            },
        },
        "comparison_sources": {
            key: {
                "dir": str(comp["dir"]),
                "arm_a": comp["arm_a"],
                "arm_b": comp["arm_b"],
                "gate_sha256": sha256(comp["dir"] / "pooled_gate.json"),
                "per_site_sha256": sha256(comp["dir"] / "pooled_per_site.csv"),
            }
            for key, comp in E0_COMPARISONS.items()
        },
        "table3_reproduction": table3_record,
        "veg_param_site_medians": veg_medians,
        "builder_version": SCRIPT_VERSION,
        "superseded_package": {
            "moved_to": str(qdir),
            "files": quarantined,
            "note": (
                "former External-ET-agreement Figure 2 package (E1/E2 metric dashboard); "
                "retained as design provenance only, removed from the active manifest"
            ),
        },
    }
    (OUT / "fig02_metadata.json").write_text(json.dumps(meta_json, indent=2))

    common_meta = dict(
        experiment=(
            "E0 (vegetation-formulation model development on the E1 cohort; "
            "legacy e2_ / examples/5_Flux_Ensemble)"
        ),
        evaluation_mask_id=E0_MASK_ID,
        sources={k: str(p) for k, p in srcs.items()},
        source_hashes={
            **{f"par:{k}": v for k, v in par_hashes.items()},
            **{f"per_site:{k}": v for k, v in per_site_hashes.items()},
            "gates_combined": gates_hash,
            "merged_ndvi": ndvi_hash,
        },
    )
    MANIFEST.add(
        "fig02_formulation_response.csv",
        rows=nr,
        note=(
            "Fitted K_T response per formulation and site on the frozen NDVI grid, from "
            "posterior-median vegetation parameters. Display transformation only; no model "
            "rerun."
        ),
        **common_meta,
    )
    MANIFEST.add(
        "fig02_ndvi_support.csv",
        rows=ns,
        note="Site-equal observed merged-NDVI density underlay for panel (a).",
        **common_meta,
    )
    MANIFEST.add(
        "fig02_pooled_metrics.csv",
        rows=np_,
        note=(
            "Pooled KGE/RMSE/signed-MBE per formulation and scale on the Table 3 arm-paired "
            "flux mask; values asserted to reproduce Table 3 at manuscript precision."
        ),
        **common_meta,
    )
    MANIFEST.add(
        "fig02_site_rmse_effects.csv",
        rows=ne,
        note=(
            "Paired per-site RMSE effects; d_rmse = cover-scaled minus unscaled (negative "
            "favours cover scaling). Isolated-cover wins 43/45 daily and 27/31 monthly."
        ),
        **common_meta,
    )
    MANIFEST.add(
        "fig02_metadata.json",
        rows=None,
        note="Reader-facing labels, equations, provenance, rules, and assertion record.",
        **common_meta,
    )
    print(
        f"  fig02: response {nr} rows, support {ns} bins, pooled {np_} rows, "
        f"effects {ne} rows; quarantined {quarantined or 'nothing'}"
    )


# --------------------------------------------------------------------------
# Figure 3 -- pooled daily ET agreement and temporal-support effects
# --------------------------------------------------------------------------
#
# Contract: paper/notes/fig03_production_handoff.md (2026-08-27). Supersedes
# the seasonal-example Figure 3 and the direct-ET-interpolation display
# package. The daily OpenET benchmark is reconstructed through ETf, never by
# interpolating ET directly: capture ETf = raw ensemble_mean_3x3 / same-day
# bias-corrected ETo; ETf is linearly interpolated in time strictly inside the
# first-to-last raw capture support; daily benchmark ET = interpolated ETf x
# daily ETo. Temporal classes come only from raw OpenET availability -- the
# archived is_overpass calibration-capture flag never classifies dates.
#
# The frozen fig03_example_* files are retained as Figure 1 provenance only
# (build_fig01 consumes the example series); this builder re-registers their
# manifest records but never regenerates them. Their generator was retired
# with the seasonal design (git history at commit 782ca3c and earlier).

E1_OPENET_DAILY = Path("/data/ssd1/swim/5_Flux_Ensemble/data/openet_flux/daily_data")

FIG03_SUPPORTS = {"overpass": "acquisition", "non_overpass": "between_acquisitions"}
FIG03_SUPPORT_ORDER = ["acquisition", "between_acquisitions", "all_dates"]
FIG03_METHODS = [("openet", "OpenET"), ("swim", "SWIM-RS")]
FIG03_EFFECT_METRICS = ["kge", "rmse", "mbe"]
FIG03_IDENTITY_TOL = 1e-10
FIG03_MIN_PAIRED = 10
FIG03_BOOTSTRAP_SEED = 42
FIG03_BOOTSTRAP_REPS = 10_000
FIG03_AXIS_LO = -2.0
FIG03_AXIS_HI = 16.0
FIG03_AXIS_TICKS = [0, 4, 8, 12, 16]
FIG03_DRAW_ORDER_SEED = 27082026  # panel (a) deterministic point-shuffle seed
FIG03_COMPOSITION_ID = "fig03_rewrite_concept_v3 (2026-08-27)"

# Section 5.3 audit anchors from the accepted v3 concept: (n_site_days,
# pearson_r, bias, rmse, display_r, display_bias, display_rmse). The builder
# hard-fails if the frozen package does not reproduce them.
FIG03_ANCHORS = {
    ("OpenET", "acquisition"): (
        4751,
        0.90491714,
        -0.31883530,
        1.09618034,
        "0.90",
        "−0.32",
        "1.10",
    ),
    ("OpenET", "between_acquisitions"): (
        55584,
        0.86666289,
        -0.23775368,
        1.12664810,
        "0.87",
        "−0.24",
        "1.13",
    ),
    ("SWIM-RS", "acquisition"): (
        4751,
        0.87677143,
        -0.04587095,
        1.18738151,
        "0.88",
        "−0.05",
        "1.19",
    ),
    ("SWIM-RS", "between_acquisitions"): (
        55584,
        0.87015031,
        -0.00953528,
        1.09528709,
        "0.87",
        "−0.01",
        "1.10",
    ),
}


def _fig03_signed_display(value: float) -> str:
    """Two-decimal display with an explicit sign and a true minus (U+2212)."""
    if value >= 0:
        return f"+{value:.2f}"
    return f"−{abs(value):.2f}"


def _fig03_metrics(obs: np.ndarray, mod: np.ndarray, label: str) -> dict[str, float]:
    """Pearson r, KGE (Gupta 2009), RMSE, and signed MBE on paired vectors.

    Same arithmetic as the archived evaluator (np.std with ddof=0); hard-fails
    on degenerate support instead of returning NaN.
    """
    if len(obs) != len(mod) or len(obs) < FIG03_MIN_PAIRED:
        raise BuildError(f"fig03 metrics {label}: n={len(obs)} below minimum {FIG03_MIN_PAIRED}")
    if not (np.isfinite(obs).all() and np.isfinite(mod).all()):
        raise BuildError(f"fig03 metrics {label}: nonfinite paired value")
    if np.std(obs) <= 0 or np.mean(obs) <= 0:
        raise BuildError(f"fig03 metrics {label}: degenerate flux vector")
    r = float(np.corrcoef(obs, mod)[0, 1])
    rmse = float(np.sqrt(np.mean((mod - obs) ** 2)))
    mbe = float(np.mean(mod - obs))
    alpha = float(np.std(mod) / np.std(obs))
    beta = float(np.mean(mod) / np.mean(obs))
    kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    return {"pearson_r": r, "kge": kge, "rmse": rmse, "mbe": mbe}


def _fig03_bootstrap_ci(values: np.ndarray) -> tuple[float, float, float]:
    """Median and whole-site bootstrap 95% CI (10,000 resamples, seed 42).

    default_rng is re-seeded per call, so the resample index matrix is
    identical for every metric and support at fixed n -- the same convention
    as the archived run22 decomposition.
    """
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise BuildError("fig03 bootstrap: nonfinite site effect")
    rng = np.random.default_rng(FIG03_BOOTSTRAP_SEED)
    idx = rng.integers(0, len(values), size=(FIG03_BOOTSTRAP_REPS, len(values)))
    medians = np.median(values[idx], axis=1)
    return (
        float(np.median(values)),
        float(np.percentile(medians, 2.5)),
        float(np.percentile(medians, 97.5)),
    )


def _fig03_reconstruct_site(fid: str) -> tuple[pd.DataFrame, float]:
    """ETf-based daily benchmark reconstruction for one site.

    Returns the paired daily frame (index=date; flux/swim/openet/eto/etf/raw
    columns plus temporal_support) and the max acquisition-date absolute
    difference between raw and reconstructed OpenET ET.
    """
    frozen_path = E1_ARCHIVE / "6_evaluation" / "site_daily_timeseries" / f"{fid}.csv"
    raw_path = E1_OPENET_DAILY / f"{fid}.csv"
    for p in (frozen_path, raw_path):
        if not p.exists():
            raise BuildError(f"fig03 source missing: {p}")
    frozen = pd.read_csv(frozen_path, index_col="date", parse_dates=True)
    raw = pd.read_csv(raw_path, index_col="DATE", parse_dates=True)
    if frozen.index.duplicated().any() or raw.index.duplicated().any():
        raise BuildError(f"fig03 {fid}: duplicate daily dates in a source series")
    require_columns(frozen.reset_index(), ["flux_ET", "swim_ET", "eto"], f"fig03 frozen {fid}")
    require_columns(raw.reset_index(), ["ensemble_mean_3x3"], f"fig03 raw benchmark {fid}")

    raw_et = pd.to_numeric(raw["ensemble_mean_3x3"], errors="coerce").dropna()
    if raw_et.empty:
        raise BuildError(f"fig03 {fid}: no finite raw benchmark value")
    eto = pd.to_numeric(frozen["eto"], errors="coerce")
    capture_eto = eto.reindex(raw_et.index)
    bad_eto = ~np.isfinite(capture_eto.values) | (capture_eto.values <= 0)
    if bad_eto.any():
        dates = ", ".join(d.date().isoformat() for d in capture_eto.index[bad_eto][:5])
        raise BuildError(
            f"fig03 {fid}: ETo missing or nonpositive on OpenET captures ({dates}); "
            "investigate the frozen daily record -- do not fill"
        )

    capture_etf = raw_et / capture_eto
    daily_index = pd.date_range(capture_etf.index.min(), capture_etf.index.max(), freq="D")
    daily_etf = capture_etf.reindex(daily_index).interpolate(method="time", limit_area="inside")
    daily_openet = daily_etf.reindex(frozen.index) * eto

    site = pd.DataFrame(
        {
            "flux_et": pd.to_numeric(frozen["flux_ET"], errors="coerce"),
            "swim_et": pd.to_numeric(frozen["swim_ET"], errors="coerce"),
            "openet_et": daily_openet,
            "eto": eto,
            "openet_etf_daily": daily_etf.reindex(frozen.index),
            "openet_et_raw": raw_et.reindex(frozen.index),
        },
        index=frozen.index,
    )
    paired = site[np.isfinite(site[["flux_et", "swim_et", "openet_et"]].values).all(axis=1)].copy()
    paired["temporal_support"] = np.where(
        paired.index.isin(capture_etf.index), "acquisition", "between_acquisitions"
    )

    # Gate: everything plotted is finite and inside raw benchmark support.
    if not np.isfinite(paired[["eto", "openet_etf_daily"]].values).all():
        raise BuildError(f"fig03 {fid}: nonfinite ETo or daily ETf on a paired row")
    if (paired.index < capture_etf.index.min()).any() or (
        paired.index > capture_etf.index.max()
    ).any():
        raise BuildError(f"fig03 {fid}: paired date outside raw benchmark support")

    acq = paired[paired["temporal_support"] == "acquisition"]
    btw = paired[paired["temporal_support"] == "between_acquisitions"]
    if not np.isfinite(acq["openet_et_raw"].values).all():
        raise BuildError(f"fig03 {fid}: acquisition row without a finite raw benchmark value")
    if btw["openet_et_raw"].notna().any():
        raise BuildError(f"fig03 {fid}: between-acquisition row carries a raw benchmark value")
    if len(acq) + len(btw) != len(paired):
        raise BuildError(f"fig03 {fid}: temporal classes do not partition the paired days")

    identity_err = float(np.max(np.abs(acq["openet_et"].values - acq["openet_et_raw"].values)))
    return paired, identity_err


def build_fig03() -> None:
    """Frozen Figure 3 display package: pooled agreement + temporal effects."""
    src_cohort = E1_ARCHIVE / "6_evaluation" / "overpass_split_metrics.csv"
    ts_dir = E1_ARCHIVE / "6_evaluation" / "site_daily_timeseries"
    for p in (src_cohort, ts_dir, E1_OPENET_DAILY):
        if not p.exists():
            raise BuildError(f"fig03 source missing: {p}")

    # ---- cohort: 43 sites eligible (>=10 paired days) in BOTH classes ----
    # The archived split-metrics file is used only for the cohort definition
    # and per-site paired-count cross-checks (sanctioned audit inputs); its
    # direct-ET-interpolation metric values are never read.
    cohort_rec = pd.read_csv(src_cohort)
    require_columns(cohort_rec, ["fid", "subset", "n_paired", "eligible"], "fig03 cohort record")
    require_unique(cohort_rec, ["fid", "subset"], "fig03 cohort record")
    elig = cohort_rec.pivot_table(index="fid", columns="subset", values="eligible", aggfunc="first")
    sites = sorted(
        elig.index[
            elig.get("overpass", False).astype(bool) & elig.get("non_overpass", False).astype(bool)
        ]
    )
    require_count(len(sites), EXPECTED["E1_split_common"], "fig03 common-cohort sites")
    counts_rec = cohort_rec.pivot_table(
        index="fid", columns="subset", values="n_paired", aggfunc="first"
    )

    # ---- per-site ETf-based reconstruction ----
    frames = []
    max_identity_err = 0.0
    src_hashes = {"overpass_split_metrics.csv": sha256(src_cohort)}
    for fid in sites:
        paired, err = _fig03_reconstruct_site(fid)
        max_identity_err = max(max_identity_err, err)
        for legacy, support in FIG03_SUPPORTS.items():
            n_class = int((paired["temporal_support"] == support).sum())
            n_want = int(counts_rec.loc[fid, legacy])
            if n_class != n_want:
                raise BuildError(
                    f"fig03 {fid}: {support} count {n_class} != archived cohort record {n_want}"
                )
            if n_class < FIG03_MIN_PAIRED:
                raise BuildError(f"fig03 {fid}: {support} support below {FIG03_MIN_PAIRED} days")
        paired = paired.reset_index().rename(columns={"index": "date"})
        paired.insert(0, "site_id", fid)
        frames.append(paired)
        src_hashes[f"site_daily_timeseries/{fid}.csv"] = sha256(ts_dir / f"{fid}.csv")
        src_hashes[f"openet_daily/{fid}.csv"] = sha256(E1_OPENET_DAILY / f"{fid}.csv")
    if max_identity_err > FIG03_IDENTITY_TOL:
        raise BuildError(
            f"fig03: acquisition-date raw/reconstructed identity {max_identity_err:.3e} "
            f"exceeds {FIG03_IDENTITY_TOL:.0e}"
        )

    pooled = pd.concat(frames, ignore_index=True)
    pooled.insert(0, "experiment", "E1")
    pooled["date"] = pd.to_datetime(pooled["date"]).dt.strftime("%Y-%m-%d")
    pooled["is_raw_openet_capture"] = pooled["temporal_support"] == "acquisition"
    pooled = pooled[
        [
            "experiment",
            "site_id",
            "date",
            "temporal_support",
            "flux_et",
            "swim_et",
            "openet_et",
            "eto",
            "openet_etf_daily",
            "openet_et_raw",
            "is_raw_openet_capture",
        ]
    ].sort_values(["site_id", "date"])
    require_unique(pooled, ["site_id", "date"], "fig03 pooled daily agreement")
    n_acq = int((pooled["temporal_support"] == "acquisition").sum())
    n_btw = int((pooled["temporal_support"] == "between_acquisitions").sum())
    require_count(n_acq, EXPECTED["E1_pool_acquisition"], "fig03 acquisition site-days")
    require_count(n_btw, EXPECTED["E1_pool_between"], "fig03 between-acquisition site-days")
    require_count(len(pooled), EXPECTED["E1_pool_total"], "fig03 total paired site-days")

    # ---- range gate: every plotted value inside the fixed -2..16 axes ----
    plotted = pooled[["flux_et", "swim_et", "openet_et"]].values
    v_lo, v_hi = float(np.min(plotted)), float(np.max(plotted))
    if v_lo < FIG03_AXIS_LO or v_hi > FIG03_AXIS_HI:
        raise BuildError(
            f"fig03: plotted extrema [{v_lo:.3f}, {v_hi:.3f}] exceed fixed axes "
            f"[{FIG03_AXIS_LO}, {FIG03_AXIS_HI}]"
        )

    # ---- panel (a) pooled scatter metrics: assert the Section 5.3 anchors ----
    scatter_rows = []
    for col, method in FIG03_METHODS:
        for support in ["acquisition", "between_acquisitions"]:
            sub = pooled[pooled["temporal_support"] == support]
            obs = sub["flux_et"].to_numpy()
            est = sub[f"{col}_et"].to_numpy()
            resid = est - obs
            r = float(np.corrcoef(obs, est)[0, 1])
            bias = float(np.mean(resid))
            rmse = float(np.sqrt(np.mean(resid**2)))
            row = {
                "temporal_support": support,
                "method": method,
                "n_sites": int(sub["site_id"].nunique()),
                "n_site_days": len(sub),
                "pearson_r": r,
                "bias": bias,
                "rmse": rmse,
                "display_r": f"{r:.2f}",
                "display_bias": _fig03_signed_display(bias),
                "display_rmse": f"{rmse:.2f}",
            }
            want_n, want_r, want_b, want_rm, disp_r, disp_b, disp_rm = FIG03_ANCHORS[
                (method, support)
            ]
            require_count(row["n_site_days"], want_n, f"fig03 scatter n {method}/{support}")
            require_count(
                row["n_sites"],
                EXPECTED["E1_split_common"],
                f"fig03 scatter sites {method}/{support}",
            )
            for got, want, name in (
                (r, want_r, "r"),
                (bias, want_b, "bias"),
                (rmse, want_rm, "rmse"),
            ):
                if abs(got - want) > 1e-8:
                    raise BuildError(
                        f"fig03 scatter {method}/{support}: {name} {got:.8f} does not "
                        f"reproduce the v3 audit anchor {want:.8f}"
                    )
            for got, want, name in (
                (row["display_r"], disp_r, "display_r"),
                (row["display_bias"], disp_b, "display_bias"),
                (row["display_rmse"], disp_rm, "display_rmse"),
            ):
                if got != want:
                    raise BuildError(
                        f"fig03 scatter {method}/{support}: {name} {got!r} != frozen {want!r}"
                    )
            scatter_rows.append(row)
    scatter = pd.DataFrame(scatter_rows)

    # ---- per-site metrics on all three supports, both methods ----
    metric_rows = []
    pooled_dt = pooled.assign(date=pd.to_datetime(pooled["date"]))
    for fid in sites:
        sdf = pooled_dt[pooled_dt["site_id"] == fid]
        for support in FIG03_SUPPORT_ORDER:
            sub = sdf if support == "all_dates" else sdf[sdf["temporal_support"] == support]
            obs = sub["flux_et"].to_numpy()
            for col, method in FIG03_METHODS:
                m = _fig03_metrics(obs, sub[f"{col}_et"].to_numpy(), f"{fid}/{support}/{method}")
                metric_rows.append(
                    {
                        "experiment": "E1",
                        "site_id": fid,
                        "temporal_support": support,
                        "method": method,
                        "n_paired": len(sub),
                        "first_date": sub["date"].min().date().isoformat(),
                        "last_date": sub["date"].max().date().isoformat(),
                        "kge": m["kge"],
                        "rmse": m["rmse"],
                        "mbe": m["mbe"],
                        "pearson_r": m["pearson_r"],
                    }
                )
    site_metrics = pd.DataFrame(metric_rows)
    require_unique(site_metrics, ["site_id", "temporal_support", "method"], "fig03 site metrics")
    require_count(len(site_metrics), EXPECTED["E1_split_common"] * 3 * 2, "fig03 site-metric rows")

    # ---- per-site paired effects (SWIM-RS minus OpenET; signed MBE) ----
    wide = site_metrics.pivot_table(
        index=["site_id", "temporal_support"],
        columns="method",
        values=["kge", "rmse", "mbe", "n_paired"],
        aggfunc="first",
    )
    if (wide[("n_paired", "SWIM-RS")] != wide[("n_paired", "OpenET")]).any():
        raise BuildError("fig03 effects: methods disagree on paired support within a stratum")
    eff = pd.DataFrame(
        {
            "n_paired": wide[("n_paired", "SWIM-RS")].astype(int),
            "kge_swim": wide[("kge", "SWIM-RS")],
            "kge_openet": wide[("kge", "OpenET")],
            "rmse_swim": wide[("rmse", "SWIM-RS")],
            "rmse_openet": wide[("rmse", "OpenET")],
            "mbe_swim": wide[("mbe", "SWIM-RS")],
            "mbe_openet": wide[("mbe", "OpenET")],
        }
    ).reset_index()
    eff["d_kge"] = eff["kge_swim"] - eff["kge_openet"]
    eff["d_rmse"] = eff["rmse_swim"] - eff["rmse_openet"]
    eff["d_mbe"] = eff["mbe_swim"] - eff["mbe_openet"]
    order = (
        eff[eff["temporal_support"] == "between_acquisitions"]
        .sort_values(["d_kge", "site_id"])
        .reset_index(drop=True)
    )
    rank = {fid: i + 1 for i, fid in enumerate(order["site_id"])}
    eff["site_order_between_kge"] = eff["site_id"].map(rank)
    eff.insert(0, "experiment", "E1")
    eff = eff.sort_values(["site_id", "temporal_support"])
    require_count(len(eff), EXPECTED["E1_split_common"] * 3, "fig03 site-effect rows")
    if eff["site_order_between_kge"].isna().any():
        raise BuildError("fig03 effects: a site lacks the between-acquisition ordering key")

    # ---- cohort effects: median of 43 site effects + frozen bootstrap CI ----
    cohort_rows = []
    for support in FIG03_SUPPORT_ORDER:
        sub = eff[eff["temporal_support"] == support]
        require_count(len(sub), EXPECTED["E1_split_common"], f"fig03 cohort effects {support}")
        for metric in FIG03_EFFECT_METRICS:
            med, lo, hi = _fig03_bootstrap_ci(sub[f"d_{metric}"].to_numpy())
            cohort_rows.append(
                {
                    "experiment": "E1",
                    "temporal_support": support,
                    "metric": metric,
                    "n_sites": len(sub),
                    "total_paired_site_days": int(sub["n_paired"].sum()),
                    "median_delta": med,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "seed": FIG03_BOOTSTRAP_SEED,
                    "n_resamples": FIG03_BOOTSTRAP_REPS,
                }
            )
    cohort_eff = pd.DataFrame(cohort_rows)
    day_totals = dict(
        cohort_eff.drop_duplicates("temporal_support")[
            ["temporal_support", "total_paired_site_days"]
        ].values
    )
    if (
        day_totals["acquisition"] != n_acq
        or day_totals["between_acquisitions"] != n_btw
        or day_totals["all_dates"] != len(pooled)
    ):
        raise BuildError("fig03 cohort effects: support day totals do not reconcile with pooled")

    # ---- quarantine the superseded direct-ET-interpolation package ----
    quarantined = []
    qdir = OUT / "superseded_fig03_direct_interpolation"
    old_deltas = OUT / "fig03_temporal_site_deltas.csv"
    if old_deltas.exists():
        qdir.mkdir(exist_ok=True)
        old_deltas.rename(qdir / old_deltas.name)
        quarantined.append(old_deltas.name)
    old_cohort = OUT / "fig03_temporal_cohort_effects.csv"
    if old_cohort.exists() and "record_type" in pd.read_csv(old_cohort, nrows=0).columns:
        qdir.mkdir(exist_ok=True)
        old_cohort.rename(qdir / old_cohort.name)
        quarantined.append(old_cohort.name)

    # ---- write tables ----
    n_pool = write_table(pooled, "fig03_pooled_daily_agreement.csv")
    n_scat = write_table(scatter, "fig03_scatter_metrics.csv")
    n_sm = write_table(site_metrics, "fig03_temporal_site_metrics.csv")
    n_eff = write_table(eff, "fig03_temporal_site_effects.csv")
    n_ce = write_table(cohort_eff, "fig03_temporal_cohort_effects.csv")

    meta_json = {
        "figure": "Figure 3 -- daily ET agreement and temporal reconstruction",
        "contract": "paper/notes/fig03_production_handoff.md (2026-08-27)",
        "composition_id": FIG03_COMPOSITION_ID,
        "experiment_mapping": {"E1": "legacy e2_* / examples/5_Flux_Ensemble"},
        "cohort": {
            "rule": (
                "43-site common temporal-support cohort: canonical run22 45-site daily "
                "cohort restricted to sites with >=10 paired days in BOTH temporal "
                "classes (JPL1_Smith5 and US-OF1 excluded); cohort membership and "
                "per-site paired counts cross-checked against the archived "
                "overpass_split_metrics.csv record"
            ),
            "sites": sites,
            "n_sites": len(sites),
            "acquisition_site_days": n_acq,
            "between_acquisition_site_days": n_btw,
            "total_site_days": len(pooled),
        },
        "benchmark_construction": {
            "steps": [
                "read raw finite OpenET v2.1 ensemble_mean_3x3 ET per site",
                "read same-day bias-corrected GridMET ETo from the frozen E1 daily record",
                "require finite, strictly positive ETo on every retained capture",
                "capture ETf = raw ET / ETo",
                "reindex ETf to a daily calendar spanning first-to-last finite capture",
                "linear-in-time interpolation strictly inside that support (no extrapolation)",
                "daily benchmark ET = interpolated ETf x daily ETo",
                "pair flux ET, SWIM-RS ET, and reconstructed OpenET ET on identical dates",
            ],
            "never": "direct linear interpolation of ET",
            "identity_tolerance_mm_day": FIG03_IDENTITY_TOL,
            "max_acquisition_identity_error_mm_day": max_identity_err,
        },
        "temporal_support_rule": (
            "acquisition = paired date with a finite raw ensemble_mean_3x3 value before "
            "interpolation; between_acquisitions = paired date inside first-to-last raw "
            "support without a raw value. Classes derive only from the separately "
            "extracted benchmark; the archived is_overpass calibration-capture flag is "
            "never used."
        ),
        "metrics": {
            "pearson_r": "np.corrcoef on the exact plotted facet rows",
            "bias": "mean(estimate - flux), mm d-1, sign retained",
            "rmse": "sqrt(mean((estimate - flux)^2)), mm d-1",
            "kge": "Gupta 2009, alpha = std ratio (ddof=0), beta = mean ratio",
            "mbe": "mean(model - flux), mm d-1, signed; no absolute-value transform",
            "effects": "SWIM-RS minus OpenET per site and support",
        },
        "panel_a": {
            "axes_mm_day": [FIG03_AXIS_LO, FIG03_AXIS_HI],
            "ticks": FIG03_AXIS_TICKS,
            "plotted_extrema_mm_day": [v_lo, v_hi],
            "draw_order_seed": FIG03_DRAW_ORDER_SEED,
            "display_rounding": "two decimals; explicit sign on Bias with true minus",
        },
        "bootstrap": {
            "kind": "whole-site resampling with replacement",
            "n_resamples": FIG03_BOOTSTRAP_REPS,
            "seed": FIG03_BOOTSTRAP_SEED,
            "note": (
                "default_rng re-seeded per call, so the resample index matrix is shared "
                "across metrics and supports at fixed n=43"
            ),
        },
        "site_order_between_kge": (
            "panel (c) frozen ordering: rank 1..43 by between-acquisition d_kge ascending, "
            "ties broken by site_id; identical across all metric facets"
        ),
        "figure1_provenance_note": (
            "fig03_example_timeseries.csv and fig03_example_selection.json are retained "
            "as Figure 1 provenance only; they are not Figure 3 inputs and are no longer "
            "regenerated (superseded seasonal design)"
        ),
        "superseded": {
            "directory": str(qdir),
            "files": quarantined,
            "reason": (
                "built from direct ET interpolation with d_abs_mbe; replaced by the "
                "ETf x ETo reconstruction with signed MBE"
            ),
        },
        "sources": {
            "site_daily_timeseries_dir": str(ts_dir),
            "openet_daily_dir": str(E1_OPENET_DAILY),
            "cohort_record": str(src_cohort),
            "sha256": src_hashes,
        },
        "generator_script": "scripts/figures/build_figure_data.py",
        "generator_version": SCRIPT_VERSION,
        "frozen_utc": datetime.now(UTC).isoformat(),
    }
    (OUT / "fig03_metadata.json").write_text(json.dumps(meta_json, indent=2))

    common_meta = {
        "figure": "fig03",
        "contract": "paper/notes/fig03_production_handoff.md (2026-08-27)",
        "experiment_mapping": {"E1": "legacy e2_*"},
        "cohort_key": "site_id",
        "inclusion_rule": meta_json["cohort"]["rule"],
        "temporal_support_rule": meta_json["temporal_support_rule"],
        "units": {"et": "mm d-1", "eto": "mm d-1", "etf": "dimensionless"},
        "deterministic_seed": FIG03_BOOTSTRAP_SEED,
        "configured_counts": {"E1": EXPECTED["E1_configured"]},
        "evaluated_counts": {
            "sites": len(sites),
            "acquisition_site_days": n_acq,
            "between_acquisition_site_days": n_btw,
            "total_site_days": len(pooled),
        },
    }
    MANIFEST.add(
        "fig03_pooled_daily_agreement.csv",
        rows=n_pool,
        note=(
            "Panel (a) pooled paired site-days; OpenET reconstructed through interpolated "
            "ETf x daily ETo (acquisition identity max "
            f"{max_identity_err:.3e} mm/d vs tolerance {FIG03_IDENTITY_TOL:.0e})."
        ),
        **common_meta,
    )
    MANIFEST.add(
        "fig03_scatter_metrics.csv",
        rows=n_scat,
        note=(
            "Frozen panel (a) facet statistics with display strings; asserted to "
            "reproduce the v3 audit anchors at 8 decimals and 2-decimal display."
        ),
        **common_meta,
    )
    MANIFEST.add(
        "fig03_temporal_site_metrics.csv",
        rows=n_sm,
        note="Per-site KGE/RMSE/signed-MBE on identical paired support for both methods.",
        **common_meta,
    )
    MANIFEST.add(
        "fig03_temporal_site_effects.csv",
        rows=n_eff,
        note=(
            "Paired site effects (SWIM-RS minus OpenET) with component metrics and the "
            "frozen panel (c) ordering key; signed MBE only, no d_abs_mbe."
        ),
        **common_meta,
    )
    MANIFEST.add(
        "fig03_temporal_cohort_effects.csv",
        rows=n_ce,
        note=(
            "Panel (b) medians of 43 site effects with 10,000-resample whole-site "
            "bootstrap 95% intervals (seed 42), rebuilt from the corrected benchmark."
        ),
        **common_meta,
    )
    MANIFEST.add(
        "fig03_metadata.json",
        rows=None,
        note="Reader-facing rules, construction record, anchors context, and provenance.",
        **common_meta,
    )

    # ---- re-register the retained Figure 1 provenance files ----
    prior_manifest = OUT / Manifest.MANIFEST_NAME
    if not prior_manifest.exists():
        raise BuildError("fig03: fig_manifest.json missing; cannot re-register example provenance")
    prior_tables = json.loads(prior_manifest.read_text()).get("tables", {})
    for name in ("fig03_example_timeseries.csv", "fig03_example_selection.json"):
        if not (OUT / name).exists():
            raise BuildError(f"fig03: retained Figure 1 provenance file missing: {name}")
        rec = dict(prior_tables.get(name) or MANIFEST.tables.get(name) or {})
        if not rec:
            raise BuildError(f"fig03: no prior manifest record for {name}")
        rec.pop("output_sha256", None)
        rec.pop("output_bytes", None)
        rec["role"] = (
            "retained as Figure 1 provenance only; frozen under the superseded seasonal "
            "Figure 3 design and no longer regenerated; not an active Figure 3 input"
        )
        MANIFEST.add(name, **rec)

    print(
        f"  fig03: pooled {n_pool} rows ({n_acq} acquisition + {n_btw} between), "
        f"scatter {n_scat}, site metrics {n_sm}, effects {n_eff}, cohort {n_ce}; "
        f"identity {max_identity_err:.3e}; quarantined {quarantined or 'nothing'}"
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
CAPTION_PATTERN_EXEMPTIONS: dict[str, str] = {}
# Architecture 3.1.1 (user review, 2026-08-25) retired the 'Σ year' operation in
# favour of the spelled-out 'Annual Total', so the sigma exemption that 3.1.0
# needed is gone and the r'σ' objective-notation pattern is back at full
# strength: every sigma-bearing string, upper or lower case, now fails the
# caption guard.
#
# Architecture 3.2.0 (handoff section 5.3, revised 2026-08-25) then removed
# 'spread-weighted' from the artwork -- "Put `PEST++ IES` and spread-dependent
# weighting in the caption by default. Do not retain `spread-weighted` in the
# artwork merely to fill the inverse region." -- so the last exemption is gone
# and r'\bweight' is at full strength too.  The exemption table is now empty and
# must stay empty: do not add an entry without a handoff section that authorizes
# the exact string.  Section 5.3 does allow 'spread-weighted' to RETURN on the
# incoming target route if a final-size proof shows it materially clarifies the
# member-spread encoding; that return would need this exemption restored and is
# recorded as a conditional in the architecture, not as a visible string.

VISIBLE_STRING_CLASSES = {"title", "direct_label", "annotation", "proof_only"}

# Nodes on the held-out side of the firewall, and nodes that participate in
# parameter fitting or transfer-vector construction.  No edge may run from the
# first set into the second.  Revised for architecture 3.0.0 (handoff sections
# 5.4 and 6.3): the daily balance and both class-specific parameter tokens are
# now protected alongside inverse estimation and the transfer destinations.
#
# Architecture 3.2.0 (handoff sections 5.3, 10.3 and 11) replaces the single
# 'inverse_estimation' node with the three-stage closed cycle, so every stage of
# that cycle joins the protected set: "Held-out flux and meter observations
# remain forbidden as cycle inputs" (section 10.3).  The retired node id is kept
# in the set so an accidental revival is still caught.
EVALUATION_NODES = {"flux_et", "meters"}
FITTING_NODES = {
    "inverse_estimation",
    "run_balance",
    "compare",
    "update_parameters",
    "daily_balance",
    "e1_map",
    "irrigated_params",
    "rainfed_params",
    "e2_map",
    "e3_map",
    "e0_tag",
}

# Architecture 3.2.0: the literal directed cycle required by handoff sections
# 4, 5.3, 10.3, 11 and 13.  Arrowheads must close it in this order -- the
# balance produces simulated calibration quantities, comparison with targets
# produces mismatches, and the parameter update feeds the next balance run.
FIG01_CYCLE_STAGES = ("run_balance", "compare", "update_parameters")
FIG01_CYCLE_EDGES = (
    ("run_balance", "compare"),
    ("compare", "update_parameters"),
    ("update_parameters", "run_balance"),
)
# The exit that leaves the update stage after convergence and feeds the
# displayed daily state and ET (handoff section 5.3).
FIG01_CYCLE_EXIT_EDGE = ("update_parameters", "daily_balance")
FIG01_CYCLE_EXIT_LABEL = "Conditioned Parameters"
# The acquisition-date targets enter the comparison stage, never the balance.
FIG01_CONSTRAINT_INTO = "compare"
# NDVI and daily forcing drive the forward balance.
FIG01_DRIVER_INTO = "run_balance"

# Panel (b) transfer topology, asserted before the architecture is written.
FIG01_TRANSFER_SOURCE = "e1_map"
FIG01_PARAM_TOKENS = ("irrigated_params", "rainfed_params")
FIG01_TRANSFER_DESTINATIONS = {
    "e2_map": ("irrigated_params", "rainfed_params"),
    "e3_map": ("irrigated_params",),
}
# The E3 transfer path must visibly branch here, before the E2 frame, so it can
# never be read as E2 -> E3 (handoff sections 6.1, 6.3 and 11).
FIG01_TRANSFER_BRANCH_AT = "irrigated_params"

# Architecture 3.1.0 (handoff sections 5.2, 5.4, 10.3 and 11).  The E1 example's
# simulated irrigation series is display evidence for the US-Bi1 record ONLY.  E3
# evaluates applied water simulated for E3 fields, so no drawn relationship --
# directed edge or neutral comparison tie -- may connect the E1 nodes below to
# the E3 meter key, directly or through any intermediate node.
FIG01_E1_APPLIED_WATER_NODES = {"applied_water"}
FIG01_E3_METER_NODE = "meters"

# The two ET traces the reader is invited to compare must be drawn against one
# date mapping at comparable horizontal extent (handoff sections 5.2, 5.4, 10.3
# and 11).  Neither may be squeezed into a side strip.
FIG01_COMMON_DATE_MAPPING_COLUMNS = ("swim_ET", "flux_ET")

# Required visible record identification (handoff sections 5.1, 11 and 13): the
# reader must not have to infer the site and year from the caption.  Architecture
# 3.2.0 sets the parenthesized form and near-black colour required by handoff
# section 5.1 -- "Identify the record directly in the artwork as `US-Bi1 (2017)`
# in near-black text ... Avoid a muted gray subtitle and the middle-dot
# construction used in revisions 1-3."
FIG01_RECORD_ID = "US-Bi1 (2017)"

# Handoff section 9: near-black for every reader-facing identification, title,
# row name, unit, map count line and inverse-stage label.  Muted gray is reserved
# for axes, reference rules and quiet geographic context.  Section 11 stops the
# build if the record identification is rendered as muted gray secondary text.
FIG01_IDENTIFICATION_COLOR = "#202124"

# Handoff sections 4, 5.2 and 11: panel (a) carries FIVE quantitative plotting
# regions, not seven independent sparklines.  Root-zone depletion and irrigation
# share one coordinated region; daily SWIM-RS ET and held-out flux ET share
# another.
FIG01_MAX_PLOTTING_REGIONS = 5

# Recorded display limits for the five regions (handoff section 5.2: "The proof
# script must record the final limits and assert that no plotted value is
# clipped").  These are the CONTRACT domains the section 11 clipping check runs
# against; the proof may refine ticks and padding as a Level 1 render change so
# long as no plotted value falls outside what is recorded here.
#
# Section 5.2's candidate ranges are ETf 0.8-1.6, NDVI 0.2-1.0, forcing 0 to a
# rounded maximum, depletion and irrigation 0 to their respective rounded
# maxima, and ET 0-10 mm d-1.  The frozen US-Bi1 record is adopted for every one
# of them EXCEPT the ETf lower bound: retrieval members reach 0.152 at this site,
# so a 0.8 floor would clip nine member marks and hide exactly the disagreement
# the region exists to show.  The floor is therefore 0.0 and the candidate upper
# bound 1.6 is kept.
FIG01_DISPLAY_DOMAINS = {
    "etf_ensemble": (0.0, 1.6),
    "ndvi_captures": (0.2, 1.0),
    "daily_forcing": (0.0, 16.0),
    "state_and_irrigation": (0.0, 14.0),
    "state_and_irrigation_secondary": (0.0, 22.0),
    "et_comparison": (0.0, 10.0),
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
        "display_role": "e1_example_irrigation_stems_only",
        "note": (
            "simulated gross applied water for the US-Bi1 E1 example. It is drawn ONCE, as the "
            "magnitude-bearing applied-water stems in the daily-outputs row, and nowhere else: "
            "the triangle event lane that repeated these events in the daily-state row is "
            "removed (handoff section 5.2). This column supports only the E1 example display. It "
            "must never be treated as an E3 modeled series, aggregated to an annual total for "
            "display, or connected -- by arrow, bracket, tie or shared glyph -- to the separate "
            "'Metered Water · E3' key. E3 evaluates applied water simulated for E3 fields, not "
            "this record (handoff sections 5.2, 5.4 and 10.2)."
        ),
        "forbidden_uses": [
            "an E3 modeled series",
            "any seasonal or annual aggregation drawn in the artwork",
            "any drawn connection to the E3 meter key or its Annual Total operation",
            "a second irrigation-event encoding in the daily-state row",
        ],
    }
    prov["swim_ET"] = {
        "source": fig03_src,
        "source_column": "swim_ET",
        "units": "mm d-1",
        "display_role": "evidence_row_output",
        "note": (
            "drawn full width on the shared date axis; it and flux_ET must use one common date "
            "mapping at comparable horizontal extent (handoff sections 5.2, 5.4 and 11)"
        ),
    }
    prov["flux_ET"] = {
        "source": fig03_src,
        "source_column": "flux_ET",
        "units": "mm d-1",
        "display_role": "held_out_observation",
        "note": (
            "the actual US-Bi1 flux record, drawn as a thin near-black trace at FULL panel width "
            "on the same date mapping as swim_ET -- never compressed into a right-hand side "
            "strip. A quiet held-out rule or neutral comparison bracket separates the two; no "
            "arrowhead may point into it, and no metric, residual or agreement claim may "
            "accompany it (handoff sections 5.4 and 11)."
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
        "display_role_constraints": {
            "_contract": (
                "paper/notes/fig01_production_handoff.md sections 5.2, 5.4, 10.2 and 11, revised "
                "2026-08-25. Re-audited for architecture 3.1.0 and carried through 3.1.1 into "
                "3.2.0: the frozen column values are unchanged in every revision, the permitted "
                "display roles are not."
            ),
            "irr_applied": (
                "Supports ONLY the E1 example display, as one magnitude-bearing applied-water "
                "stem series in the daily-outputs row. It is not an E3 modeled series. It must "
                "not be summed for display, and it must not be connected by arrow, bracket, tie "
                "or shared glyph to the 'Daily Gross Applied Water -> Annual Total <-> Metered "
                "Water · E3' key, which is an experiment-level schematic describing E3 "
                "simulations. The "
                "builder additionally refuses any drawn path from the E1 applied-water node to "
                "the E3 meter node. If a future proof replaces the schematic key with data "
                "marks, a separately frozen E3 modeled-and-observed record with an independently "
                "justified selection record must be added; this file's display role must not be "
                "extended to cover it."
            ),
            "irrigation_encoding": (
                "One encoding only. The applied-water stems already carry timing and magnitude, "
                "so the same events must not be repeated as triangles or another symbol lane. "
                "For architecture 3.2.0 the stems are integrated into the coordinated "
                "state-and-irrigation region with rz_depletion, on their own explicit magnitude "
                "scale; they must NOT be normalized to the depletion scale (handoff section 5.2)."
            ),
            "swim_ET_and_flux_ET": (
                "Both ET traces are drawn in ONE merged plotting region (architecture 3.2.0, "
                "handoff section 5.2 item 5) at full panel width, on identical date support and "
                "an identical vertical scale. Neither may be compressed into a side strip or "
                "separated into an independent full-height lane, and the relationship between "
                "them is shared alignment alone -- no bracket, no tie glyph, and never an "
                "arrowhead pointing into the flux observations."
            ),
            "swe_audit": (
                "Never plotted. SWE is represented as an inline '+ SWE' label on the constraint "
                "route entering the 'Compare' stage of the inverse cycle; the separate SWE chip "
                "of architecture 3.0.0 is removed."
            ),
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
        "contract": (
            "paper/notes/fig01_production_handoff.md sections 5.1, 10.2 and 11, revised 2026-08-25"
        ),
        "record_identification": {
            "visible_string": FIG01_RECORD_ID,
            "color": FIG01_IDENTIFICATION_COLOR,
            "requirement": (
                "handoff sections 5.1, 11 and 13 require the record to be identified in the "
                "artwork; the reader must not have to infer the site and year from the caption. "
                "Architecture 3.2.0 sets the parenthesized form and near-black colour: 'Identify "
                "the record directly in the artwork as US-Bi1 (2017) in near-black text ... "
                "Avoid a muted gray subtitle and the middle-dot construction used in revisions "
                "1-3' (section 5.1). Section 11 stops the build if it is rendered as muted gray "
                "secondary text."
            ),
        },
        "re_audit": {
            "date": "2026-08-24",
            "against": "fig01_production_handoff.md sections 10.2 and 15.2 (revised 2026-08-24)",
            "column_values_changed": False,
            "column_set_changed": False,
            "display_roles_changed": True,
            "detail": (
                "The frozen US-Bi1 record is unchanged. The re-audit narrows irr_applied to the "
                "E1 example display only, records the single-irrigation-encoding rule, states "
                "the common date mapping required of swim_ET and flux_ET, and notes that SWE is "
                "now an inline label rather than a separate node."
            ),
        },
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

    # ---- handoff sections 6.4 and 6.7: faint CONUS state boundaries ----
    # Archived Natural Earth 1:50m admin-1 units, the same NE release (5.1.1) as
    # the admin-0 source already used for world_context.  Public domain.  Alaska
    # and Hawaii are dropped so the layer matches the CONUS locator extent, and
    # the clip is the same box conus_context uses.
    if not NE_STATES.exists():
        raise BuildError(f"fig01: state-boundary source missing {NE_STATES}")
    states = gpd.read_file(NE_STATES, engine="fiona")
    states = states[(states["iso_a2"] == "US") & (~states["postal"].isin(["AK", "HI"]))]
    states_ctx = gpd.clip(states[["name", "postal", "geometry"]], conus_bbox)
    states_ctx = gpd.GeoDataFrame(states_ctx, geometry="geometry", crs=world_ctx.crs)
    states_ctx = states_ctx.sort_values("postal").reset_index(drop=True)
    states_ctx["boundary_source"] = "Natural Earth 50m admin 1 states provinces lakes"
    states_ctx["boundary_version"] = _ne_states_version()
    if len(states_ctx) != 49:
        raise BuildError(
            "fig01 conus_states_context: expected the 48 contiguous states plus the District of "
            f"Columbia, got {len(states_ctx)}"
        )

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

    # ---- handoff sections 6.6 and 6.7: privacy-safe E3 basin context ----
    # The five HUC8 hydrologic units of the Rio Grande headwaters accounting unit
    # (HUC6 130100), extracted from the USGS Watershed Boundary Dataset and
    # archived in-repo.  Each unit covers 1,987-6,576 km2, so the layer orients
    # the reader without localizing any field more precisely than the approved
    # 1 km centroid snap already does.
    if not SLV_BASIN.exists():
        raise BuildError(f"fig01: E3 basin-context source missing {SLV_BASIN}")
    basin_ctx = gpd.read_file(SLV_BASIN, engine="fiona")
    basin_ctx = basin_ctx.sort_values("huc8").reset_index(drop=True)
    basin_ctx["boundary_source"] = "USGS Watershed Boundary Dataset, WBDHU8"
    if len(basin_ctx) != 5 or set(basin_ctx["huc8"]) != {
        "13010001",
        "13010002",
        "13010003",
        "13010004",
        "13010005",
    }:
        raise BuildError(
            "fig01 slv_basin_context: expected the five HUC8 units of the Rio Grande headwaters "
            f"accounting unit, got {sorted(basin_ctx['huc8'])}"
        )
    # The basin layer must stay coarser than the approved display generalization:
    # it may not be a field boundary, and every generalized display point must
    # fall inside it, so it can add no location information the points lack.
    if float(basin_ctx["areasqkm"].min()) < 1000.0:
        raise BuildError(
            "fig01 slv_basin_context: a context unit smaller than 1000 km2 is not a coarse "
            "watershed boundary; refusing to publish it alongside the generalized field display"
        )
    inside = e3_disp.geometry.apply(lambda p: bool(basin_ctx.contains(p).any()))
    if not bool(inside.all()):
        raise BuildError(
            "fig01 slv_basin_context: the basin context does not contain every generalized E3 "
            "display point; a context layer that crops or excludes points would carry location "
            "information the approved display withholds"
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
        "conus_states_context": states_ctx,
        "world_context": world_ctx,
        "slv_context": slv_ctx,
        "slv_basin_context": basin_ctx,
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

    # ---- panel (a) plotting-region domains (handoff sections 5.2, 11 and 13) ----
    # Section 5.2: "The proof script must record the final limits and assert that
    # no plotted value is clipped."  The contract records the display limits and
    # the observed range of every plotted column behind each region; the section
    # 11 check below stops the build if a value falls outside its region's
    # recorded limits.  These are the same frozen columns in every revision, so
    # later tick or padding refinements inside these limits stay Level 1.
    def _region_data_range(columns: list[str]) -> list[float]:
        vals = pd.concat([example[c].astype(float) for c in columns], ignore_index=True)
        vals = vals[vals.notna()]
        if vals.empty:
            raise BuildError(
                f"fig01: plotting-region columns {columns} carry no values in the frozen example"
            )
        return [float(vals.min()), float(vals.max())]

    _etf_cols = [f"etf_{m}" for m in E1_MEMBERS] + ["etf_target_mean"]
    _region_ranges = {
        "etf_ensemble": _region_data_range(_etf_cols),
        "ndvi_captures": _region_data_range(["ndvi_landsat_raw", "ndvi_sentinel_raw"]),
        "daily_forcing": _region_data_range(["eto", "precip"]),
        "state_and_irrigation": _region_data_range(["rz_depletion"]),
        "state_and_irrigation_secondary": _region_data_range(["irr_applied"]),
        "et_comparison": _region_data_range(["swim_ET", "flux_ET"]),
    }

    def _y_axis(key: str, units: str, candidate, note: str | None = None) -> dict:
        lo, hi = FIG01_DISPLAY_DOMAINS[key]
        block = {
            "visible_spine": True,
            "display_domain": [lo, hi],
            "labeled_bounds": [lo, hi],
            "n_labeled_bounds": 2,
            "units": units,
            "units_placement": "adjacent to the axis or to the region label (handoff section 5.2)",
            "data_range": _region_ranges[key],
            "clipping_forbidden": True,
            "candidate_domain_handoff_5_2": candidate,
        }
        if note:
            block["domain_note"] = note
        return block

    arch = {
        "schema_version": "3.2.0",
        "supersedes": {
            "schema_version": "3.1.1",
            "status": "review provenance, not an accepted composition",
            "superseded_composition": (
                "the r3 proof's panel (a): seven stacked full-height lanes (five numbered "
                "evidence rows, the last carrying three lanes), a 'Daily State' row separate "
                "from the irrigation stems, a 'Daily Outputs' row in which the SWIM-RS and flux "
                "traces sat in separate lanes, a one-way 'Inverse Estimation' element whose "
                "iteration was implied by a single parameter-update arrow, a 'Held-Out "
                "Evaluation' label, a 'spread-weighted' label in the inverse region, and the "
                "record identified as a middle-dot 'US-Bi1 · 2017' subtitle."
            ),
            "reason": (
                "fig01_production_handoff.md was rewritten 2026-08-25 after review of revision "
                "3: 'Revision 3 reviewed; panel (a) redesign and a new Gate A proof are "
                "required'. Section 16 declares it a Level 2 contract change because it changes "
                "visible copy, panel grouping, and inverse-estimation relationships. Panel (b) "
                "is explicitly NOT superseded: section 6.1 accepts its revision-3 core topology."
            ),
            "removed_requirements": [
                "the seven-lane panel (a) structure -- five numbered evidence rows with three "
                "lanes on the last; section 4 now requires five quantitative plotting regions "
                "and section 11 stops the build above five",
                "the separate 'Daily State' region; root-zone depletion and the irrigation stems "
                "are now one coordinated region with explicit scales for both (section 5.2 "
                "item 4)",
                "the separate 'Daily Outputs' region with daily_et and flux_et in distinct "
                "lanes; the two ET traces now share one axis, one date support and one vertical "
                "scale (section 5.2 item 5)",
                "the one-way inverse representation -- a titled element plus a single "
                "parameter-update arrow; section 5.3 requires a literal closed cycle and forbids "
                "'two disconnected L-shaped legs or a generic one-way transformation'",
                "the 'Inverse Estimation' title; sections 4, 5.3 and 13 name only the three "
                "cycle verbs and section 9 speaks of 'inverse-stage labels'",
                "the 'Held-Out Evaluation' label; section 5.4 requires the direct label 'Flux ET "
                "(Held Out)' 'rather than a separate Held-Out Evaluation heading or rule "
                "competing for space'",
                "the 'spread-weighted' label in the artwork; section 5.3 puts spread-dependent "
                "weighting in the caption by default",
                "the middle-dot record identification and its muted-gray subtitle treatment "
                "(section 5.1)",
                "the improvised '⊢—⊣' relation glyph in the E3 key; sections 4 and 5.4 write the "
                "relation with a plain em dash",
            ],
            "removed_strings": [
                "US-Bi1 · 2017",
                "Flux ET",
                "Held-Out Evaluation",
                "Inverse Estimation",
                "spread-weighted",
                "Daily State",
                "Daily Outputs",
                "Daily Gross Applied Water → Annual Total ⊢—⊣ Metered Water · E3",
            ],
            "removed_strings_note": (
                "'US-Bi1 · 2017' -> 'US-Bi1 (2017)', near-black, no middle dot (section 5.1). "
                "'Flux ET' -> 'Flux ET (Held Out)': the merged ET region carries the held-out "
                "qualification on the trace itself, which is what lets the separate 'Held-Out "
                "Evaluation' label go (sections 5.2 item 5 and 5.4). 'Inverse Estimation' is "
                "retired outright: the closed cycle is named by its three stage labels 'Run "
                "Balance', 'Compare' and 'Update Parameters', and no section of the 2026-08-25 "
                "handoff asks for a title above them. 'spread-weighted' is caption-owned by "
                "section 5.3 and may return only on the incoming target route if a final-size "
                "proof shows it materially clarifies the member-spread encoding; that condition "
                "is recorded under inverse_cycle.spread_label_condition, not as a visible "
                "string. 'Daily State' -> 'State + Irrigation' and 'Daily Outputs' -> 'Daily ET' "
                "follow the section 4 wireframe row names for the two merged regions. The E3 "
                "relation string loses the improvised '⊢—⊣' glyph for the plain em dash written "
                "in sections 4 and 5.4. Retiring 'spread-weighted' also emptied "
                "CAPTION_PATTERN_EXEMPTIONS, so the r'\\bweight' caption pattern is back at full "
                "strength alongside r'σ'."
            ),
            "not_superseded": (
                "panel (b). Handoff section 6.1: 'Revision 3's core topology is accepted: the "
                "irrigated triangle is the fork, one horizontal leg reaches E2, and one visibly "
                "divergent curved leg reaches E3. Preserve that relationship.' Subsequent "
                "changes to arc curvature, map positions, clearances, or stroke weights are "
                "render-only unless they change the source or destination of a path."
            ),
        },
        "panel_a_redesign_2026_08_25": {
            "source": "paper/notes/fig01_production_handoff.md, rewritten 2026-08-25",
            "status_line": (
                "Revision 3 reviewed; panel (a) redesign and a new Gate A proof are required"
            ),
            "revision_level": (
                "Level 2 contract change (section 16): it changes visible copy, panel grouping, "
                "and inverse-estimation relationships"
            ),
            "changes": [
                {
                    "id": "r1_five_regions",
                    "sections": ["4", "5.2", "11", "13"],
                    "change": (
                        "panel (a) carries five quantitative plotting regions, not seven "
                        "independent sparklines: ETf Ensemble, NDVI Captures, Daily Forcing, "
                        "State + Irrigation, Daily ET"
                    ),
                },
                {
                    "id": "r2_merged_state_irrigation",
                    "sections": ["4", "5.2"],
                    "change": (
                        "root-zone depletion is the principal continuous trace and the "
                        "magnitude-bearing irrigation stems are integrated into the same "
                        "coordinated region, each with an explicit scale; a compact secondary "
                        "scale or clearly separated internal sub-band is preferable to "
                        "normalizing the stems"
                    ),
                },
                {
                    "id": "r3_merged_et",
                    "sections": ["4", "5.2", "5.4", "11"],
                    "change": (
                        "daily SWIM-RS ET and actual US-Bi1 flux ET are plotted together on ONE "
                        "axis with identical date support and vertical scale, directly labelled "
                        "'Daily ET' and 'Flux ET (Held Out)'"
                    ),
                },
                {
                    "id": "r4_closed_cycle",
                    "sections": ["4", "5.3", "10.3", "11", "13"],
                    "change": (
                        "inverse estimation becomes a literal compact closed cycle 'Run Balance "
                        "-> Compare -> Update Parameters -> Run Balance' with ETf targets and "
                        "'+ SWE' entering Compare and a 'Conditioned Parameters' exit leaving "
                        "the update stage toward the displayed daily trajectory"
                    ),
                },
                {
                    "id": "r5_y_axis_contract",
                    "sections": ["5.2", "9", "11", "13"],
                    "change": (
                        "every region reads as a quantitative small multiple: a short visible "
                        "left y-spine, at least two labelled bound values, units adjacent to the "
                        "axis or row label, and no clipped plotted value at the recorded display "
                        "limits"
                    ),
                },
                {
                    "id": "r6_near_black_identification",
                    "sections": ["5.1", "9", "11", "13"],
                    "change": (
                        "near-black #202124 for all reader-facing identification, titles, row "
                        "names, units, map count lines and inverse-stage labels; muted gray is "
                        "confined to axes, reference rules and quiet geographic context"
                    ),
                },
                {
                    "id": "r7_visible_copy",
                    "sections": ["5.1", "5.2", "5.3", "5.4"],
                    "change": (
                        "'US-Bi1 · 2017' -> 'US-Bi1 (2017)'; 'Flux ET' -> 'Flux ET (Held Out)'; "
                        "'Held-Out Evaluation', 'Inverse Estimation' and 'spread-weighted' "
                        "retired; 'Run Balance', 'Compare', 'Update Parameters' and 'Conditioned "
                        "Parameters' added"
                    ),
                },
                {
                    "id": "r8_panel_b_accepted",
                    "sections": ["6.1", "6.7"],
                    "change": (
                        "panel (b) is accepted as built in revision 3 and is not redesigned; the "
                        "faint CONUS state boundaries are retained and the HUC8 subdivisions are "
                        "kept out of the E3 map"
                    ),
                },
            ],
            "ambiguities_resolved": [
                {
                    "question": "does the 'Inverse Estimation' title survive?",
                    "resolution": "retired",
                    "basis": (
                        "the section 4 wireframe draws the cycle with no title above it; section "
                        "5.3 names only 'Run Balance', 'Compare' and 'Update Parameters'; "
                        "section 13 asks that 'inverse estimation is unmistakably a closed, "
                        "directional cycle through Run Balance, Compare, and Update Parameters'; "
                        "and section 9 speaks of 'inverse-stage labels', not an inverse title. "
                        "No section of the 2026-08-25 handoff requires the string, and section "
                        "5.3 asks the element to stay compact and unboxed, so a title would "
                        "spend space the redesign is trying to recover."
                    ),
                },
                {
                    "question": "what are the five region names?",
                    "resolution": (
                        "'ETf Ensemble', 'NDVI Captures', 'Daily Forcing', 'State + Irrigation', "
                        "'Daily ET'"
                    ),
                    "basis": (
                        "the section 4 wireframe row names are the only literal copy the handoff "
                        "gives; the section 5.2 item headings ('Water State and Irrigation', "
                        "'Daily ET and Held-Out Flux ET') are prose descriptions of the same "
                        "regions, and section 5.2 asks for compact facet labels rather than "
                        "prominent headings."
                    ),
                },
                {
                    "question": "where does the 'Conditioned Parameters' exit originate?",
                    "resolution": "the update stage",
                    "basis": (
                        "section 5.3: 'A separate exit labelled Conditioned Parameters leaves "
                        "the update stage after convergence and feeds the displayed daily state "
                        "and ET.' The section 4 wireframe hangs the exit under 'Run Balance'; "
                        "the prose is normative and is followed here."
                    ),
                },
                {
                    "question": "may 'spread-weighted' return?",
                    "resolution": (
                        "not as frozen copy; recorded as a conditional under "
                        "inverse_cycle.spread_label_condition"
                    ),
                    "basis": (
                        "section 5.3: 'It may return on the incoming target route only if a "
                        "final-size proof shows that it materially clarifies the member-spread "
                        "encoding.' No such proof exists, so the string is retired and its "
                        "caption-guard exemption removed."
                    ),
                },
                {
                    "question": "does the section 5.2 candidate ETf domain 0.8-1.6 hold?",
                    "resolution": "the upper bound holds; the lower bound is set to 0.0",
                    "basis": (
                        "the frozen US-Bi1 record's retrieval members span 0.152-1.452, so a 0.8 "
                        "floor would clip nine member marks. Section 11 stops the build if a "
                        "plotted value is clipped at the recorded display limits, and section "
                        "5.2 calls its ranges 'candidate ranges for the next proof', so the "
                        "recorded domain widens to cover the data."
                    ),
                },
            ],
        },
        "also_supersedes_3_1_0": {
            "schema_version": "3.1.0",
            "status": "review provenance, not an accepted composition",
            "superseded_composition": (
                "the second pass of the evidence-first redesign, rendered as the r2 proof. The "
                "user review of 2026-08-25 accepted the composition as a strong improvement and "
                "returned four must-address items plus refinements before Gate B: the held-out "
                "flux trace was labelled with the E1-E2 evaluation scope although the drawn "
                "record is a single US-Bi1 E1 series; the E3 key used an improvised 'Σ year' "
                "operator; 'Held-Out Evaluation' sat as a left-column secondary heading that "
                "opened a large empty region; the daily-ET-to-flux-ET comparison was drawn as a "
                "right-margin bracket that read as an empty box at thumbnail scale; the E3 "
                "transfer leg ran as a long horizontal corridor along the top of the E2 frame "
                "and read as an upper border on it, with the fork 2.8 mm ahead of the frame "
                "reading as part of the E2 entrance; and '+ SWE' sat beside the 'Inverse "
                "Estimation' title, where it read as part of the method name."
            ),
            "reason": (
                "User review of the r2 proof, 2026-08-25. Four must-address items and a set of "
                "refinements; the subset that changes frozen copy or frozen topology guidance is "
                "recorded here and in review_directives_2026_08_25."
            ),
            "removed_requirements": [
                "the 'Flux ET · E1–E2' label on the drawn US-Bi1 flux trace; the E1-E2 "
                "flux-evaluation scope is caption-owned",
                "the improvised 'Σ year' summation operator in the E3 evaluation key",
                "'Held-Out Evaluation' as a left-column secondary heading",
                "the drawn bracket or tie glyph on the daily-ET / flux-ET comparison",
                "the long horizontal E3 corridor paralleling the top of the E2 frame",
                "the '+ SWE' label placed adjacent to the 'Inverse Estimation' title",
            ],
            "removed_strings": [
                "Flux ET · E1–E2",
                "Σ year",
            ],
            "removed_strings_note": (
                "'Flux ET · E1–E2' became 'Flux ET': the drawn trace is an actual US-Bi1 record "
                "on the E1 cohort, so attaching the broader E1-E2 flux-evaluation role to that "
                "one series was a mislabel. The caption owns that scope, and caption_facts "
                "item_5_heldout still carries it. 'Σ year' became 'Annual Total': a true "
                "subscripted sigma would drop the 'year' subscript below the 7.5 pt floor, so "
                "the spelled-out operation wins. Retiring it also let the sigma exemption be "
                "removed from CAPTION_PATTERN_EXEMPTIONS, restoring the caption guard to full "
                "strength over every sigma-bearing string."
            ),
        },
        "review_directives_2026_08_25": {
            "source": "user review of the r2 proof, relayed 2026-08-25",
            "authority": (
                "paper/notes/fig01_production_handoff.md (revised 2026-08-24) remains the "
                "composition authority. These directives are layered on it and are cited here so "
                "the proof builder can trace each change to its origin."
            ),
            "gate": "one revision before Gate B",
            "directives": [
                {
                    "id": "d1_flux_label",
                    "kind": "frozen copy",
                    "change": "'Flux ET · E1–E2' -> 'Flux ET' (direct_label)",
                    "rationale": (
                        "the drawn trace is an actual US-Bi1 / E1 record, so the broader E1-E2 "
                        "flux-evaluation role was a mislabel on that series; the caption owns "
                        "the scope"
                    ),
                },
                {
                    "id": "d2_annual_total",
                    "kind": "frozen copy",
                    "change": (
                        "'Σ year' -> 'Annual Total' (direct_label); the E3 key now reads "
                        "'Daily Gross Applied Water → Annual Total ⊢—⊣ Metered Water · E3'"
                    ),
                    "rationale": (
                        "'Σ year' was improvised; a subscripted sigma would put 'year' below the "
                        "7.5 pt minimum, so the spelled-out operation is used instead"
                    ),
                },
                {
                    "id": "d3_heldout_label",
                    "kind": "frozen class and placement",
                    "change": "'Held-Out Evaluation' reclassified title -> direct_label",
                    "rationale": (
                        "as a left-column secondary heading it opened a large empty region; it "
                        "is now a compact label on the held-out rule itself"
                    ),
                },
                {
                    "id": "d4_comparison_alignment",
                    "kind": "frozen treatment",
                    "change": (
                        "the daily_et / flux_et comparison is shared alignment ONLY -- no drawn "
                        "bracket or tie glyph"
                    ),
                    "rationale": (
                        "the right-margin bracket read as an empty box at thumbnail scale; the "
                        "comparison entry itself is retained because the section 11 "
                        "common-date-mapping check depends on it, and the no-arrowhead rule "
                        "stands"
                    ),
                },
                {
                    "id": "d5_branch_geometry",
                    "kind": "frozen geometry guidance",
                    "change": (
                        "the irrigated-class triangle sits AT the fork with 'Irrigated "
                        "Parameters' set to its left; the E3 leg leaves the fork as a distinct "
                        "diagonal or shallow curve"
                    ),
                    "rationale": (
                        "the r2 horizontal corridor read as an upper border around the E2 frame, "
                        "and a junction 2.8 mm before the frame read as part of the E2 entrance; "
                        "the section 11 check is unchanged, this tightens the geometry it guards"
                    ),
                },
                {
                    "id": "d6_swe_placement",
                    "kind": "frozen placement",
                    "change": (
                        "'+ SWE' belongs on the incoming constraint route into the inverse "
                        "element, not beside the 'Inverse Estimation' title"
                    ),
                    "rationale": "beside the title it read as part of the method name",
                },
                {
                    "id": "d7_hillshade",
                    "kind": "open decision",
                    "change": (
                        "a faint privacy-safe E3 hillshade stays a user-optional future test; "
                        "not required for Gate B"
                    ),
                    "rationale": "it would need an archived, licensed DEM derivation first",
                },
            ],
        },
        "also_supersedes_3_0_0": {
            "schema_version": "3.0.0",
            "status": "review provenance, not an accepted composition",
            "superseded_composition": (
                "the first pass of the evidence-first redesign, rendered as "
                "paper/figures/proofs/fig01_evidence_190/. Panel (a) carried a narrow right-hand "
                "held-out column with the flux record compressed into a side strip beside the "
                "full-width model trace, an 'annual sum' bracket spanning the E1 example's "
                "seasonal applied-water record and reading as an E1-to-E3 data linkage, a "
                "three-circle data-like meter glyph with no frozen observations behind it, a "
                "separate SWE chip node, a 'PEST++ IES' subtitle on the inverse node, a faint "
                "driver lane that read as an axis spine, a second triangle irrigation lane in "
                "the daily-state row, and an E3 transfer path routed along the bottom of the E2 "
                "frame."
            ),
            "reason": (
                "fig01_production_handoff.md was revised again on 2026-08-24 after review of "
                "that proof. The evidence-first thesis is retained; the false E1-to-E3 data "
                "linkage, the held-out comparison grammar, the transfer routing, and the visual "
                "density are corrected."
            ),
            "removed_requirements": [
                "the narrow right-hand held-out region and the compressed flux side strip",
                "the 'annual sum' aggregation bracket over the E1 example's applied-water record",
                "the applied_water -> meters edge, which asserted a linkage the data do not "
                "support: E3 evaluates applied water simulated for E3 fields, not the US-Bi1 "
                "example",
                "the data-like three-circle meter glyph",
                "the separate swe_constraint chip node and its edge into inverse estimation",
                "the 'PEST++ IES' subtitle on the inverse-estimation node",
                "the second, triangle-based irrigation-event lane in the daily-state row",
                "the daily_et -> flux_et arrow; comparison is now a neutral tie",
                "row headings classified as titles competing with the panel heading",
            ],
            "removed_strings": [
                "PEST++ IES",
                "SWE",
                "annual sum",
                "Gross Applied Water",
                "irrigation",
                "2017",
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
            "removed_strings_note": (
                "The first six entries were visible in 3.0.0 and are removed by the 2026-08-24 "
                "revision; the rest were already removed when 3.0.0 superseded 2.1.0 and are "
                "carried forward so the revival guard still covers them. 'PEST++ IES' moved to "
                "the caption, 'SWE' became the inline '+ SWE' label, 'annual sum' became the "
                "'Σ year' operation in the separate E3 key, 'Gross Applied Water' became "
                "'Irrigation' on the E1 lane so it cannot be confused with the E3 key's 'Daily "
                "Gross Applied Water', 'irrigation' was the removed duplicate event lane's "
                "label, and '2017' was folded into 'US-Bi1 · 2017', which 3.2.0 in turn replaces "
                "with 'US-Bi1 (2017)'."
            ),
            "note": (
                "Architecture 3.0.0 and the proof under paper/figures/proofs/fig01_evidence_190/ "
                "are review provenance. Do not revise fig01_evidence_190.png into the next "
                "proof; write a fresh Gate A revision to a new proof directory (handoff sections "
                "1 and 15)."
            ),
        },
        "also_supersedes": {
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
            "paper/notes/fig01_production_handoff.md sections 4-9, rewritten 2026-08-25. Every "
            "string under panels / example_record / plotting_regions / driver_routing / "
            "inverse_cycle / outputs / held_out / e3_evaluation_key / map_nodes / "
            "parameter_tokens / development_tag / axes is frozen reader-facing copy and must be "
            "drawn verbatim. Nothing under caption_facts may be drawn."
        ),
        "canvas_mm": [190, 120],
        "outer_margin_mm": 3,
        "panel_gutter_mm": [3, 4],
        "canvas_note": (
            "Handoff section 4 (2026-08-20, unchanged 2026-08-24) sets 190 x 120 mm. "
            "six_figure_plan.md section 3.1 was reconciled to 120 mm on 2026-08-20; both notes "
            "now record the same decision."
        ),
        "figure_thesis": (
            "Sparse, disagreeing satellite observations condition a state-carrying daily water "
            "balance; its ET and applied-water outputs are evaluated with data that remain "
            "outside parameter estimation, and E1-derived parameter sets are tested through "
            "parallel transfer to E2 and E3."
        ),
        "reading_path": (
            "panel (a), top to bottom on one shared date axis, five quantitative plotting "
            "regions: the record identification, then the ETf ensemble captures, the NDVI "
            "captures, the daily forcing, the coordinated state-and-irrigation region in which "
            "the magnitude-bearing stems sit against the root-zone depletion trace, and finally "
            "the merged ET region in which the model trace and the held-out flux record share "
            "one axis, one date support and one vertical scale. Beneath them, one compact, "
            "unboxed closed cycle -- Run Balance to Compare to Update Parameters and back -- "
            "takes the acquisition-date ETf targets and the inline '+ SWE' constraint into "
            "Compare, is driven at Run Balance by NDVI and daily forcing, and leaves the update "
            "stage as a labelled 'Conditioned Parameters' exit into the displayed daily "
            "trajectory. A separate, typographic E3 evaluation key sits apart from the example "
            "record. Panel (b), left to right: the E1 CONUS source map, two class-specific "
            "parameter tokens, a visible branch junction at the irrigated token, then the E2 "
            "world map and the E3 San Luis Valley map as the two parallel transfer endpoints."
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
            "site_id_is_visible": True,
            "record_label": FIG01_RECORD_ID,
            "record_label_string_class": "direct_label",
            "record_label_placement": (
                "directly beneath the panel (a) heading, above the plotting regions; subordinate "
                "to the heading in weight but NOT in colour"
            ),
            "record_label_color": FIG01_IDENTIFICATION_COLOR,
            "record_label_rendered_as_muted_gray": False,
            "record_label_treatment": (
                "near-black, set as identification rather than as a subtitle. Handoff section "
                "5.1: 'Identify the record directly in the artwork as US-Bi1 (2017) in "
                "near-black text; do not make the reader infer the site and year from the "
                "caption. Avoid a muted gray subtitle and the middle-dot construction used in "
                "revisions 1-3.' Hierarchy comes from size, weight and placement, never from "
                "washing the identification into gray (section 9)."
            ),
            "record_label_forbidden_treatments": [
                "muted gray secondary text",
                "the middle-dot 'US-Bi1 · 2017' construction of revisions 1-3",
                "supplying the site and year only in the caption",
            ],
            "site_id_note": (
                "handoff sections 5.1, 11 and 13 require the record to be identified in the "
                "artwork. The reader must not have to infer the site and year from the caption. "
                "Architecture 3.0.0 left this to the caption; that decision is reversed."
            ),
        },
        "shared_time_grammar": (
            "All five panel (a) plotting regions share one date axis. Sparse observations stay "
            "visibly discontinuous and are never connected or smoothed to look complete; daily "
            "quantities stay visibly continuous. Sparse ticks and units establish scale."
        ),
        "common_date_mapping": {
            "columns": list(FIG01_COMMON_DATE_MAPPING_COLUMNS),
            "axis": "date_axis",
            "region": "et_comparison",
            "lanes": ["daily_et", "flux_et"],
            "full_width": True,
            "comparable_horizontal_extent": True,
            "single_region": True,
            "shared_vertical_scale": True,
            "rule": (
                "the SWIM-RS ET trace and the held-out flux ET trace are drawn in ONE plotting "
                "region against one date mapping and one vertical scale: identical x limits, "
                "identical days-per-millimetre scale, identical horizontal extent spanning the "
                "full panel-(a) plotting width, and identical y limits. Handoff section 5.2 item "
                "5: 'Plot daily SWIM-RS ET and actual US-Bi1 flux ET together on one axis with "
                "identical date support and vertical scale.' They are presented as a direct "
                "visual comparison, so any difference in date or y mapping would misrepresent it "
                "(sections 5.2, 5.4, 10.3, 11 and 13)."
            ),
            "forbidden": [
                "compressing the flux record into a narrow right-hand side strip while the model "
                "trace runs full width",
                "a second, independent date axis for either trace",
                "a second, independent vertical scale for either trace",
                "separating the two traces into independent full-height lanes",
                "pairing an actual trace and a symbolic glyph inside one apparently equivalent "
                "evidence region",
                "an arrowhead directed into the flux observations",
                "an example metric, residual annotation, causal arrow, or separate comparison "
                "bracket",
            ],
            "comparison_treatment": (
                "shared alignment on one axis: one date mapping, one vertical scale, two direct "
                "labels. No bracket, tie glyph, or connector is drawn, and the relationship "
                "reads as a comparison, never as an arrow implying the model produced the "
                "observations."
            ),
        },
        "plotting_regions_note": (
            "Handoff section 4: 'Panel (a) contains five quantitative plotting regions, not "
            "seven independent sparklines. Root-zone depletion and irrigation share one "
            "coordinated region; daily SWIM-RS ET and held-out flux ET share another. Each "
            "region has an explicit vertical domain.' The seven-lane structure of architecture "
            "3.1.1 -- five numbered evidence rows with three lanes on the last -- is superseded, "
            "and the key is renamed from evidence_rows to plotting_regions so no downstream "
            "script can read the old grouping."
        ),
        "plotting_regions": [
            {
                "id": "etf_ensemble",
                "order": 1,
                "heading": "ETf Ensemble",
                "heading_string_class": "direct_label",
                "axis_label": "ETf",
                "axis_label_string_class": "direct_label",
                "columns": [f"etf_{m}" for m in E1_MEMBERS]
                + ["etf_target_mean", "etf_ensemble_spread", "etf_member_count"],
                "plotted_columns": _etf_cols,
                "mark": (
                    "at each of the 15 calibration captures, plot every available retrieval "
                    "member as a small neutral mark, a VISIBLE min-max line spanning the "
                    "members, and an OPEN target-mean diamond in satellite-benchmark orange. "
                    "Member disagreement must remain legible at final size and no member name is "
                    "drawn (handoff section 5.2 item 1)."
                ),
                "member_marks": "neutral",
                "min_max_line": True,
                "target_mean_marker": "open diamond",
                "color_role": "satellite_et_target",
                "member_names_visible": False,
                "y_axis": _y_axis(
                    "etf_ensemble",
                    "ETf (dimensionless)",
                    [0.8, 1.6],
                    note=(
                        "the section 5.2 candidate range is 0.8-1.6. The frozen record's "
                        "retrieval members reach 0.152, so a 0.8 floor would clip the member "
                        "marks that carry the disagreement this region exists to show; the floor "
                        "is set to 0.0 and the candidate upper bound is kept."
                    ),
                ),
            },
            {
                "id": "ndvi_captures",
                "order": 2,
                "heading": "NDVI Captures",
                "heading_string_class": "direct_label",
                "axis_label": "NDVI",
                "axis_label_string_class": "direct_label",
                "columns": ["ndvi_landsat_raw", "ndvi_sentinel_raw"],
                "plotted_columns": ["ndvi_landsat_raw", "ndvi_sentinel_raw"],
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
                "y_axis": _y_axis("ndvi_captures", "NDVI (dimensionless)", [0.2, 1.0]),
            },
            {
                "id": "daily_forcing",
                "order": 3,
                "heading": "Daily Forcing",
                "heading_string_class": "direct_label",
                "axis_label": "mm d⁻¹",
                "axis_label_string_class": "direct_label",
                "columns": ["eto", "precip"],
                "plotted_columns": ["eto", "precip"],
                "direct_labels": [
                    {"label": "ETo", "column": "eto", "string_class": "direct_label"},
                    {"label": "precipitation", "column": "precip", "string_class": "direct_label"},
                ],
                "mark": (
                    "compact precipitation bars and a fine ETo line on the shared date axis. "
                    "Handoff section 5.2 item 3: 'This is one quantitative plot, not a product "
                    "inventory.'"
                ),
                "single_quantitative_plot": True,
                "y_axis": _y_axis("daily_forcing", "mm d⁻¹", [0, "rounded maximum"]),
            },
            {
                "id": "state_and_irrigation",
                "order": 4,
                "heading": "State + Irrigation",
                "heading_string_class": "direct_label",
                "axis_label": "mm",
                "axis_label_string_class": "direct_label",
                "columns": ["rz_depletion", "irr_applied"],
                "plotted_columns": ["rz_depletion", "irr_applied"],
                "merged": True,
                "merge_note": (
                    "handoff section 5.2 item 4, 'Water State and Irrigation': root-zone "
                    "depletion is the principal continuous trace and the magnitude-bearing "
                    "simulated irrigation stems are integrated into the SAME coordinated region. "
                    "'This replaces the generic plant-soil block and makes event-to-state "
                    "response visible without consuming a separate full-height lane.' "
                    "Architecture 3.1.1 kept them in two separate regions; that split is "
                    "superseded."
                ),
                "direct_labels": [
                    {
                        "label": "root-zone depletion",
                        "column": "rz_depletion",
                        "string_class": "direct_label",
                    },
                    {
                        "label": "Irrigation",
                        "column": "irr_applied",
                        "node": "applied_water",
                        "string_class": "direct_label",
                    },
                ],
                "principal_trace": {
                    "id": "rz_depletion",
                    "column": "rz_depletion",
                    "mark": (
                        "root-zone depletion as one continuous line or band, vertically aligned "
                        "with the precipitation bars above. This is the primary visual evidence "
                        "of state propagation and replaces the removed generic plant-soil block."
                    ),
                    "color_role": "swim_state_and_output",
                },
                "irrigation_stems": {
                    "id": "applied_water",
                    "column": "irr_applied",
                    "mark": (
                        "magnitude-bearing applied-water stems: one stem per simulated "
                        "irrigation event, its height carrying the daily depth on its own "
                        "explicit scale. This is the figure's only irrigation encoding."
                    ),
                    "color_role": "swim_state_and_output",
                    "scope": "the E1 US-Bi1 example only",
                    "not_an_e3_series": True,
                    "normalized_to_state_scale": False,
                    "scale_treatment": (
                        "a compact secondary scale or a clearly separated internal sub-band "
                        "inside this region. Handoff section 5.2 item 4 prefers either to "
                        "normalizing the stems, so the stem heights must stay readable as "
                        "millimetres of applied water rather than as a unitless fraction of the "
                        "depletion axis."
                    ),
                    "forbidden": [
                        "normalizing the stems to the depletion scale",
                        "any aggregation bracket, tie, or sum drawn over these stems",
                        "any drawn connection to the E3 evaluation key or its Annual Total "
                        "operation",
                        "repeating these events as triangles or another symbol lane anywhere in "
                        "panel (a)",
                    ],
                },
                "irrigation_events_repeated_here": False,
                "irrigation_note": (
                    "one irrigation encoding only (handoff section 5.2). The stems carry both "
                    "timing and magnitude, and now sit in the same region as the state trace, so "
                    "event-to-state response is read directly rather than across two lanes."
                ),
                "color_role": "swim_state_and_output",
                "y_axis": _y_axis(
                    "state_and_irrigation",
                    "mm (root-zone depletion)",
                    [0, "rounded maximum"],
                ),
                "secondary_y_axis": _y_axis(
                    "state_and_irrigation_secondary",
                    "mm (applied water)",
                    [0, "rounded maximum"],
                    note=(
                        "the explicit second scale handoff section 5.2 item 4 requires for the "
                        "irrigation magnitudes; without it the stems could only be drawn "
                        "normalized, which the section rules out"
                    ),
                ),
            },
            {
                "id": "et_comparison",
                "order": 5,
                "heading": "Daily ET",
                "heading_string_class": "direct_label",
                "axis_label": "mm d⁻¹",
                "axis_label_string_class": "direct_label",
                "columns": ["swim_ET", "flux_ET"],
                "plotted_columns": ["swim_ET", "flux_ET"],
                "merged": True,
                "merge_note": (
                    "handoff section 5.2 item 5, 'Daily ET and Held-Out Flux ET': 'Plot daily "
                    "SWIM-RS ET and actual US-Bi1 flux ET together on one axis with identical "
                    "date support and vertical scale. Directly label the lines Daily ET and Flux "
                    "ET (Held Out). Do not add an example metric, residual annotation, causal "
                    "arrow, or a separate comparison bracket.' The two traces are lanes of ONE "
                    "region, not two full-height lanes."
                ),
                "shared_axis": True,
                "shared_vertical_scale": True,
                "lanes": [
                    {
                        "id": "daily_et",
                        "label": "Daily ET",
                        "string_class": "direct_label",
                        "column": "swim_ET",
                        "mark": "a continuous daily model trace at full panel width",
                        "color_role": "swim_state_and_output",
                        "horizontal_extent": "full panel-(a) plotting width",
                    },
                    {
                        "id": "flux_et",
                        "label": "Flux ET (Held Out)",
                        "string_class": "direct_label",
                        "column": "flux_ET",
                        "label_note": (
                            "3.2.0: handoff section 5.4 requires 'the direct label Flux ET (Held "
                            "Out) rather than a separate Held-Out Evaluation heading or rule "
                            "competing for space'. The qualification now travels with the trace, "
                            "which is what lets the 'Held-Out Evaluation' label be retired. The "
                            "broader E1-E2 flux-evaluation scope stays caption-owned "
                            "(caption_facts item_5_heldout)."
                        ),
                        "mark": (
                            "the actual US-Bi1 flux record as a thin near-black trace, drawn on "
                            "the SAME axis, the SAME date mapping and the SAME vertical scale as "
                            "the model trace"
                        ),
                        "color_role": "held_out_observation",
                        "horizontal_extent": "full panel-(a) plotting width, identical to daily_et",
                        "held_out": True,
                        "comparison_treatment": (
                            "shared alignment only -- one axis, one date mapping, one vertical "
                            "scale, two direct labels. No bracket, tie glyph, or connector is "
                            "drawn, no arrowhead may point into this lane, and no example "
                            "metric, residual, or emphasis on agreement may accompany it"
                        ),
                    },
                ],
                "aggregation_mark": None,
                "aggregation_note": (
                    "annual aggregation appears only in the separate, experiment-level "
                    "e3_evaluation_key. No bracket, tie, or sum may span this region or the "
                    "irrigation stems above it (handoff sections 4, 5.2, 5.4 and 9)."
                ),
                "color_role": "swim_state_and_output",
                "y_axis": _y_axis("et_comparison", "mm d⁻¹", [0, 10]),
            },
        ],
        "plotting_region_rules": [
            "exactly five regions; handoff section 4 forbids seven independent sparklines and "
            "section 11 stops the build above five",
            "every region reads as a quantitative small multiple rather than a sparkline: a "
            "short visible left y-spine, at least two labelled values defining the displayed "
            "lower and upper bounds, and units adjacent to the axis or the region label",
            "no plotted value may be clipped at the recorded display limits",
            "the five region names are compact facet or y-axis labels, not prominent display "
            "headings; one shared date axis serves all five",
            "rounded, stable domains are preferred; axis-range changes derived from the same "
            "frozen columns are render-level settings, not a reason to rebuild data",
        ],
        "y_axis_contract": {
            "source": "handoff sections 5.2, 9, 11 and 13",
            "required_per_region": [
                "a short visible left y-spine",
                "at least two labelled bound values",
                "units adjacent to the axis or the region label",
                "no clipped plotted value at the recorded display limits",
            ],
            "spine_weight_pt": [0.45, 0.7],
            "final_limits_are_render_level": (
                "the display limits recorded here are the contract the section 11 clipping check "
                "runs against. A proof may refine ticks, padding and label placement inside them "
                "as a Level 1 render change; widening or narrowing them past a plotted value is "
                "not permitted"
            ),
            "candidate_domains_2026_08_25": {
                "etf": [0.8, 1.6],
                "ndvi": [0.2, 1.0],
                "forcing": [0, "rounded maximum"],
                "depletion": [0, "rounded maximum"],
                "irrigation": [0, "rounded maximum"],
                "et": [0, 10],
            },
            "recorded_display_domains": {
                k: list(v) for k, v in sorted(FIG01_DISPLAY_DOMAINS.items())
            },
            "observed_data_ranges": {k: v for k, v in sorted(_region_ranges.items())},
        },
        "date_axis": {
            "id": "date_axis",
            "label": None,
            "string_class": "direct_label",
            "columns": ["date"],
            "treatment": (
                "one shared date axis for all five plotting regions, drawn once at the bottom of "
                "panel (a) with sparse month ticks; tick text is generated from the data and is "
                "not frozen copy"
            ),
            "label_note": (
                "the standalone '2017' axis label of architecture 3.0.0 is dropped: the year is "
                "now carried by the required record identification 'US-Bi1 (2017)', and "
                "repeating it on the axis would spend a visible string on a fact already stated"
            ),
        },
        "inverse_cycle": {
            "id": "inverse_cycle",
            "title": None,
            "title_string_class": "direct_label",
            "title_note": (
                "3.2.0: the 'Inverse Estimation' title of 3.1.1 is RETIRED. The section 4 "
                "wireframe draws the cycle with no title above it, section 5.3 names only the "
                "three stages, section 13 asks that 'inverse estimation is unmistakably a "
                "closed, directional cycle through Run Balance, Compare, and Update Parameters', "
                "and section 9 speaks of 'inverse-stage labels'. The three stage labels name the "
                "element; a separate title would spend space the redesign is recovering and is "
                "held under the revival guard."
            ),
            "representation": "literal directed cycle",
            "compact_treatment": (
                "keep the cycle compact and unboxed: a small triangular or circular routing of "
                "the three verbs is sufficient. No process card, no fill, no thick rounded box, "
                "no drop shadow (handoff sections 5.3, 9 and 13)."
            ),
            "boxed": False,
            "stages": [
                {
                    "id": "run_balance",
                    "order": 1,
                    "label": "Run Balance",
                    "string_class": "direct_label",
                    "role": "the balance run produces simulated calibration quantities",
                    "color": FIG01_IDENTIFICATION_COLOR,
                },
                {
                    "id": "compare",
                    "order": 2,
                    "label": "Compare",
                    "string_class": "direct_label",
                    "role": (
                        "comparison of simulated calibration quantities with the acquisition-date "
                        "ETf targets and the auxiliary SWE constraint produces mismatches"
                    ),
                    "color": FIG01_IDENTIFICATION_COLOR,
                },
                {
                    "id": "update_parameters",
                    "order": 3,
                    "label": "Update Parameters",
                    "string_class": "direct_label",
                    "role": "the parameter update feeds the next balance run",
                    "color": FIG01_IDENTIFICATION_COLOR,
                },
            ],
            "cycle_edges": [list(e) for e in FIG01_CYCLE_EDGES],
            "cycle_rule": (
                "handoff section 5.3: 'Arrowheads must close the cycle in that order: the "
                "balance produces simulated calibration quantities, comparison with targets "
                "produces mismatches, and the parameter update feeds the next balance run.' The "
                "internal feedback edge belongs only to simulated calibration quantities."
            ),
            "constraint_inputs": [
                {
                    "from": "etf_ensemble",
                    "into": FIG01_CONSTRAINT_INTO,
                    "drawn_as_edge": True,
                    "note": "acquisition-date ETf targets enter the comparison stage",
                },
                {
                    "from": "swe_inline",
                    "into": FIG01_CONSTRAINT_INTO,
                    "drawn_as_edge": False,
                    "note": (
                        "'+ SWE' rides the same incoming constraint route as an inline label; it "
                        "is not a separate node, card, or arrow, and it is not a zero-valued "
                        "trace in this growing-season example"
                    ),
                },
            ],
            "driver_inputs": [
                {"from": "ndvi_captures", "into": FIG01_DRIVER_INTO, "drawn_as_edge": True},
                {"from": "daily_forcing", "into": FIG01_DRIVER_INTO, "drawn_as_edge": True},
            ],
            "exit": {
                "id": "conditioned_parameters",
                "label": FIG01_CYCLE_EXIT_LABEL,
                "string_class": "direct_label",
                "edge": list(FIG01_CYCLE_EXIT_EDGE),
                "drawn_as": "a labelled exit arrow, not a fourth stage box",
                "color": FIG01_IDENTIFICATION_COLOR,
                "rule": (
                    "handoff section 5.3: 'A separate exit labelled Conditioned Parameters "
                    "leaves the update stage after convergence and feeds the displayed daily "
                    "state and ET.' The section 4 wireframe hangs the exit under Run Balance; "
                    "the prose is normative and is followed here."
                ),
            },
            "routing_rule": (
                "the inverse path and the driver path must be distinguishable through routing "
                "and stroke treatment. ETf and SWE constrain parameters and enter Compare; NDVI "
                "and daily forcing drive the forward balance and enter Run Balance. Route the "
                "driver connector away from y-axis spines and tick labels so it cannot be "
                "mistaken for an axis. The figure must not imply that all inputs share the same "
                "dates or enter through the same mechanism."
            ),
            "held_out_rule": (
                "flux ET and meter observations remain outside the cycle and have no return "
                "path. Handoff section 10.3: 'Held-out flux and meter observations remain "
                "forbidden as cycle inputs.'"
            ),
            "spread_label_condition": {
                "string": "spread-weighted",
                "status": "retired from the artwork; caption-owned",
                "may_return": True,
                "return_condition": (
                    "handoff section 5.3: 'It may return on the incoming target route only if a "
                    "final-size proof shows that it materially clarifies the member-spread "
                    "encoding.' No such proof exists."
                ),
                "return_placement": "the incoming target route only",
                "return_requires": (
                    "restoring a CAPTION_PATTERN_EXEMPTIONS entry for the exact string, since "
                    "r'\\bweight' otherwise blocks it as caption-owned copy"
                ),
                "recorded_as": "a conditional, not a visible string",
            },
            "engine_label_ownership": "caption",
            "engine_label_condition": (
                "'PEST++ IES' stays in the caption by default (handoff section 5.3). The engine "
                "name is carried by caption_facts.calibration_settings and by the working "
                "caption."
            ),
            "color_role": "inverse_estimation",
            "label_color_rule": (
                "structural cycle labels remain near-black even when their arrows are purple "
                "(handoff section 9)"
            ),
            "forbidden_content": [
                "a title above the cycle",
                "the objective equation",
                "the parameter inventory",
                "the realization count",
                "the iteration count",
                "the weighting formula",
                "an engine name that forces the element to grow",
            ],
            "forbidden_representations": [
                "two disconnected L-shaped legs sharing a heading",
                "a generic one-way transformation",
                "a large process card, a boxed node, or a tinted region",
                "a separate SWE card, chip, or arrow",
                "a zero-valued SWE trace",
                "any edge from flux ET or meter observations into any stage",
            ],
            "supersedes": (
                "the one-way inverse representation of architectures 3.0.0-3.1.1: a titled "
                "'Inverse Estimation' element whose iteration was implied by a single "
                "inverse_estimation -> daily_balance parameter-update arrow. Handoff sections "
                "5.3, 10.3 and 11 now require the literal closed cycle and the labelled exit."
            ),
        },
        "swe_constraint": {
            "id": "swe_inline",
            "label": "+ SWE",
            "string_class": "direct_label",
            "column": "swe_audit",
            "rendered_as": (
                "inline label on the incoming constraint route into the 'Compare' stage (3.2.0)"
            ),
            "enters_stage": FIG01_CONSTRAINT_INTO,
            "drawn_as_separate_node": False,
            "drawn_as_edge": False,
            "placement": (
                "on the constraint route entering the 'Compare' stage of the inverse cycle, "
                "alongside the acquisition-date ETf targets. Handoff section 5.3: "
                "'Acquisition-date ETf targets and the inline + SWE auxiliary constraint enter "
                "the Compare stage.' The r2 proof's placement beside the retired 'Inverse "
                "Estimation' title made it read as part of the method name."
            ),
            "treatment": (
                "gridded SWE is an auxiliary calibration constraint written inline on the route "
                "into the compare stage, not a separate card, chip, or arrow. It is identically "
                "zero across this growing-season example, so no zero-valued SWE trace may be "
                "forced into the panel (handoff section 5.3)."
            ),
            "plotted_as_trace": False,
            "supersedes": (
                "the 'SWE' chip node and the swe_constraint -> inverse_estimation edge of "
                "architecture 3.0.0, and the 3.1.1 placement on the route into the titled "
                "inverse element"
            ),
        },
        "driver_routing": {
            "id": "daily_drivers",
            "label": "daily drivers",
            "string_class": "direct_label",
            "applies_to": ["ndvi_captures", "daily_forcing"],
            "destination": FIG01_DRIVER_INTO,
            "optional": True,
            "treatment": (
                "a short direct label on the driver connector, in place of the faint full-height "
                "lane drawn by architecture 3.0.0. Handoff section 5.3: 'If retained, the daily "
                "drivers bracket should span only the NDVI and forcing rows and use one short "
                "connector into Run Balance; it must not become a long non-data stroke.'"
            ),
            "spans_regions": ["ndvi_captures", "daily_forcing"],
            "layout_constraint": (
                "route the driver connector clear of the y-axis spines, the ticks and the tick "
                "labels so it cannot be mistaken for an axis; it must not overlap or run "
                "parallel and adjacent to any axis line (handoff sections 5.3 and 13)"
            ),
            "forbidden": [
                "a faint vertical lane running alongside the region spines",
                "any routing that overlaps axis tick text",
                "a stroke weight or colour that matches the axes",
                "a bracket spanning regions other than NDVI Captures and Daily Forcing",
                "a long non-data stroke",
            ],
        },
        "daily_balance": {
            "id": "daily_balance",
            "represented_by": ["state_and_irrigation", "et_comparison"],
            "label": None,
            "note": (
                "the displayed daily trajectory has no titled box. It is represented by the "
                "actual State + Irrigation and Daily ET plotting regions, and it is reached by "
                "the labelled 'Conditioned Parameters' exit from the update stage. The removed "
                "generic crop-soil cross-section must not be restored."
            ),
            "relation_to_run_balance": (
                "'Run Balance' inside the cycle and this displayed trajectory are the same "
                "forward water balance in two roles: iterated while parameters are conditioned, "
                "then displayed once conditioned. The drivers therefore enter at Run Balance and "
                "are not drawn twice."
            ),
        },
        "outputs": [
            {
                "id": "daily_et",
                "label": "Daily ET",
                "string_class": "direct_label",
                "column": "swim_ET",
                "region": "et_comparison",
                "compared_with": "flux_et",
                "comparison_kind": "shared alignment on one axis; not a directed edge",
            },
            {
                "id": "applied_water",
                "label": "Irrigation",
                "string_class": "direct_label",
                "column": "irr_applied",
                "region": "state_and_irrigation",
                "compared_with": None,
                "aggregation": None,
                "scope": "the E1 US-Bi1 example only",
                "note": (
                    "renamed from 'Gross Applied Water' so it cannot be confused with the "
                    "separate E3 key's 'Daily Gross Applied Water'. This series has NO drawn "
                    "comparison and NO drawn aggregation: E3 evaluates applied water simulated "
                    "for E3 fields, not this record. The applied_water -> meters edge of "
                    "architecture 3.0.0 is removed and is now a forbidden edge (handoff sections "
                    "5.2, 5.4 and 11)."
                ),
            },
        ],
        "held_out": {
            "heading": None,
            "heading_string_class": "direct_label",
            "heading_treatment": (
                "3.2.0: there is NO held-out heading. Handoff section 5.4 requires 'the direct "
                "label Flux ET (Held Out) rather than a separate Held-Out Evaluation heading or "
                "rule competing for space'. The qualification now travels with the trace inside "
                "the merged ET region, so the 3.1.1 label is retired and held under the revival "
                "guard."
            ),
            "heading_pt": None,
            "divider": (
                "no rule separates the two ET traces -- they share one axis. A quiet rule may "
                "still set off the separate E3 evaluation key. Nothing about the held-out "
                "boundary is two-way: no return path, no evaluation-to-fitting arrow, and no "
                "arrowhead directed into either observation."
            ),
            "region_treatment": (
                "the two held-out treatments stay separated rather than squeezed into one narrow "
                "right-hand column: the flux record is a lane of the merged ET region at full "
                "panel width, and the E3 meter term is a typographic node in the separate key. "
                "No tinted card may be placed around either."
            ),
            "narrow_right_hand_column": False,
            "supersedes": (
                "architecture 3.0.0 placed both treatments in a narrow right-hand region with "
                "the flux record compressed into a side strip beside the full-width model trace, "
                "and paired that actual trace with a symbolic meter glyph inside one apparently "
                "equivalent evidence region. Handoff section 5.4 forbids both."
            ),
            "observations": [
                {
                    "id": "flux_et",
                    "label": "Flux ET (Held Out)",
                    "string_class": "direct_label",
                    "aligned_with": "daily_et",
                    "region": "et_comparison",
                    "lane": "flux_et",
                    "column": "flux_ET",
                    "is_actual_data": True,
                    "treatment": (
                        "the actual US-Bi1 flux record drawn as a thin near-black trace at full "
                        "panel width, on the SAME axis, date mapping and vertical scale as the "
                        "Daily ET trace. Comparison is by shared alignment ONLY: no bracket, no "
                        "tie glyph, no connector, no arrow into the trace, no example metric, no "
                        "residual, and no emphasis on agreement."
                    ),
                    "scope_note": (
                        "the label is 'Flux ET (Held Out)' (3.2.0, handoff sections 5.2 item 5 "
                        "and 5.4): the held-out qualification travels with the trace instead of "
                        "occupying a separate heading. This is one E1 site-year; the broader "
                        "E1-E2 flux-evaluation scope is caption-owned."
                    ),
                },
                {
                    "id": "meters",
                    "label": "Metered Water · E3",
                    "string_class": "direct_label",
                    "aligned_with": "e3_annual_sum",
                    "lane": None,
                    "column": None,
                    "is_actual_data": False,
                    "data_like_glyph": False,
                    "frozen_observation_source": None,
                    "treatment": (
                        "a typographic term in the separate, experiment-level E3 evaluation key "
                        "-- not a mark inside the US-Bi1 example. No E3 meter values are frozen "
                        "in the Figure 1 package, so nothing here may be drawn as data. The "
                        "three-circle meter glyph of the reviewed fig01_evidence_190 proof read "
                        "as data and is disallowed. Real meter marks require frozen observations "
                        "selected by an observation-support rule independent of simulated "
                        "performance, plus the corresponding E3 model output on an honest common "
                        "support, plus its own selection record (handoff section 5.4)."
                    ),
                },
            ],
            "color_role": "held_out_observation",
        },
        "e3_evaluation_key": {
            "id": "e3_key",
            "scope": (
                "experiment-level. It describes the E3 evaluation operation on applied water "
                "simulated for E3 fields. It is NOT an extension of the US-Bi1 record and shares "
                "no mark, tie, bracket, or connector with it."
            ),
            "treatment_class": "typographic_schematic",
            "placement": (
                "below the panel (a) plotting regions, visibly separated from the shared date "
                "axis and from the irrigation stems, so no reader can trace a path from the E1 "
                "stems into it"
            ),
            "relation_string": "Daily Gross Applied Water → Annual Total — Metered Water · E3",
            "relation_string_note": (
                "3.1.1 (review directive d2) replaced the improvised 'Σ year' with the "
                "spelled-out 'Annual Total'. 3.2.0 also drops the improvised '⊢—⊣' tie glyph for "
                "the plain em dash written in handoff sections 4 and 5.4: 'Daily Gross Applied "
                "Water → Annual Total — Metered Water · E3'. The comparison to the meters stays "
                "a neutral, undirected relation -- no arrowhead into the observation."
            ),
            "nodes": [
                {
                    "id": "e3_applied_water",
                    "label": "Daily Gross Applied Water",
                    "string_class": "direct_label",
                    "column": None,
                    "source": "E3 simulations; no values are frozen in the Figure 1 package",
                    "is_actual_data": False,
                },
                {
                    "id": "e3_annual_sum",
                    "label": "Annual Total",
                    "string_class": "direct_label",
                    "column": None,
                    "is_actual_data": False,
                    "purpose": (
                        "the compact annual-aggregation operation required by handoff sections 9 "
                        "and 13, in place of a bracket spanning an entire seasonal record"
                    ),
                    "label_note": (
                        "3.1.1 (review directive d2): replaces the improvised 'Σ year'. Because "
                        "no sigma is drawn, no caption-guard exemption is needed and the sigma "
                        "pattern is back at full strength."
                    ),
                },
                {
                    "id": "meters",
                    "label": "Metered Water · E3",
                    "string_class": "direct_label",
                    "column": None,
                    "is_actual_data": False,
                },
            ],
            "internal_edges": [["e3_applied_water", "e3_annual_sum"]],
            "internal_comparisons": [["e3_annual_sum", "meters"]],
            "data_like_glyph": False,
            "frozen_observation_source": None,
            "forbidden": [
                "the three-circle meter glyph, or any other data-like meter mark, while no E3 "
                "observations are frozen",
                "any connection to the E1 example's Irrigation stems in the State + Irrigation "
                "region, or to a sum of them",
                "an arrowhead directed into 'Metered Water · E3'",
                "drawing this key inside the example's plotting region or on its date axis",
            ],
            "upgrade_path": (
                "to draw real E3 marks, first freeze the E3 modelled and observed values with an "
                "independently justified selection record and display them on an honest common "
                "support; do not extend fig01_example_timeseries.csv to cover them"
            ),
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
                "optional_context_layer": "conus_states_context",
                "optional_context_layer_status": (
                    "ACCEPTED for 3.2.0. Handoff section 6.7: 'Retain the faint CONUS state "
                    "boundaries accepted in revision 3.' Draw them as hairline neutral "
                    "boundaries, strictly subordinate to the site marks; no state labels, no "
                    "fills."
                ),
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
                "optional_context_layer": None,
                "optional_context_layer_status": (
                    "REJECTED for 3.2.0. Handoff section 6.7: 'Keep the HUC8 subdivisions out of "
                    "the E3 map; they competed with the field marks and implied an analytical "
                    "spatial grouping that E3 does not use.' The slv_basin_context layer stays "
                    "in fig01_scope.gpkg as archived, licensed, hashed provenance -- removing it "
                    "would be a Level 3 geography change for no gain -- but it must not be "
                    "drawn. The generalized San Luis Valley boundary (slv_context) is retained."
                ),
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
                "symbol_placement": (
                    "3.1.1 (review directive d5): the triangle sits AT the fork itself -- the "
                    "point where the E2 and E3 legs separate -- with the 'Irrigated Parameters' "
                    "label set to its LEFT. The symbol marks the junction, so the split is read "
                    "as originating at the class token and not at either destination frame."
                ),
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
        "transfer_branch": {
            "id": "irrigated_branch",
            "at": FIG01_TRANSFER_BRANCH_AT,
            "visible": True,
            "outgoing": ["e2_map", "e3_map"],
            "position": (
                "AT the Irrigated Parameters token, marked by its triangle, and well BEFORE the "
                "E2 map frame, so both irrigated paths are seen to originate at the token. A "
                "junction placed a couple of millimetres ahead of the frame reads as part of the "
                "E2 entrance and is not acceptable (3.1.1, review directive d5: the r2 fork sat "
                "2.8 mm before the frame and read that way)."
            ),
            "e3_route": (
                "a clearly separate leg leaving the fork as a diagonal or shallow curve to the "
                "E3 map, visibly diverging from the E2 leg at the junction. It must NOT run as a "
                "long horizontal corridor paralleling the top or bottom of the E2 frame -- in "
                "the r2 proof that corridor read as an upper border drawn around E2. It may "
                "stagger vertically or pass above or below the E2 map, but it must never run "
                "along, beneath, or through the E2 frame in a way that can be read as E2->E3."
            ),
            "forbidden_routes": [
                "the reviewed fig01_evidence_190 route, which ran along the bottom of the E2 "
                "frame and read as E2 -> E3",
                "the r2 route, a long horizontal corridor along the top of the E2 frame that "
                "read as an upper border on the E2 map",
                "any route whose visible origin is the E2 map or its frame",
                "any route that enters and leaves the E2 map's plotting region",
                "any fork placed close enough to the E2 frame to read as its entrance",
            ],
            "rule": (
                "handoff sections 6.1 and 6.3: make the irrigated split explicit with one branch "
                "point immediately after its token; both outgoing paths originate there and "
                "neither originates at the E2 map"
            ),
        },
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
            "placement": "adjacent_to_e1_heading_or_parameter_relay_origin",
            "placement_detail": (
                "set the tag next to the 'E1 · CONUS' heading, or at the origin of the E1 "
                "parameter relay, so its scope is immediate (handoff sections 6.4 and 7). "
                "Architecture 3.0.0 allowed it to drift below the map, where it read as a "
                "detached footnote."
            ),
            "forbidden_placements": [
                "below the E1 map as a detached footnote",
                "in the lower page margin",
                "as a fourth map or another geography",
                "as a coequal evaluation branch",
            ],
            "rendering": (
                "one small, subordinate tag beside the E1 heading or relay origin. E0 is not a "
                "fourth map, not a coequal evaluation branch, and carries no flux-to-parameter "
                "arrow."
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
            "context_decision": {
                "decided": "2026-08-25, handoff section 6.7",
                "supersedes": (
                    "the 3.1.0/3.1.1 'context_test' block, which left both archived layers for "
                    "the proof to decide. Revision 3 tested them and the handoff records the "
                    "outcome, so the decision is frozen here."
                ),
                "conus_states": {
                    "layer": "conus_states_context",
                    "status": "ACCEPTED and drawn",
                    "basis": (
                        "'Retain the faint CONUS state boundaries accepted in revision 3' "
                        "(section 6.7)"
                    ),
                    "treatment": (
                        "hairline neutral boundaries at 0.3-0.4 pt and low opacity, strictly "
                        "below the site marks and the coastline in visual weight; no state "
                        "labels, no fills"
                    ),
                },
                "e3_basin": {
                    "layer": "slv_basin_context",
                    "status": "REJECTED; archived but not drawn",
                    "basis": (
                        "'Keep the HUC8 subdivisions out of the E3 map; they competed with the "
                        "field marks and implied an analytical spatial grouping that E3 does not "
                        "use' (section 6.7)"
                    ),
                    "retained_in_package": True,
                    "retention_reason": (
                        "the layer stays in fig01_scope.gpkg as archived, licensed and hashed "
                        "provenance; deleting it would be a Level 3 geography change that alters "
                        "frozen values for no scientific gain. It simply must not be rendered."
                    ),
                    "privacy": (
                        "HUC8 units span 1,987-6,576 km2 and every generalized display point "
                        "falls inside the layer, so publishing it adds no location precision "
                        "beyond the approved 1 km centroid snap"
                    ),
                },
                "e3_boundary_retained": "slv_context, the generalized San Luis Valley boundary",
                "hillshade": (
                    "optional. Section 6.7: 'A subtle privacy-safe hillshade remains optional and "
                    "requires an archived, licensed, hashed source. Its absence does not block "
                    "Gate B.'"
                ),
            },
        },
        "typography": {
            "family": "Source Sans 3, embedded, editable vector text",
            "panel_label_pt": [10, 11],
            "panel_heading_pt": [8.5, 9],
            "structural_label_pt": [8, 8.5],
            "direct_label_and_axis_pt": [7.5, 8],
            "minimum_reader_facing_pt": 7.5,
            "row_label_pt": [7.5, 8],
            "row_label_treatment": (
                "the five panel (a) region names are conventional compact facet or y-axis "
                "labels, set at direct-label weight and size. Their typographic weight must stay "
                "below the panel heading and below the data. They are not five prominent display "
                "headings competing with the marks: architecture 3.0.0 classified them as titles "
                "and the 2026-08-24 revision demoted them to direct labels (handoff section 9)."
            ),
            "case_rule": (
                "title case for structural headings, sentence case for axis text and short "
                "explanatory phrases; conventional acronym capitalization preserved"
            ),
            "identification_color": FIG01_IDENTIFICATION_COLOR,
            "identification_color_rule": (
                "handoff section 9 (2026-08-25): 'Use near-black (#202124 or equivalent) for "
                "reader-facing identification, titles, row names, units, map count lines, "
                "inverse-stage labels, and the US-Bi1 (2017) identifier. Establish hierarchy "
                "with size, weight, and placement -- not by washing secondary information into "
                "gray.' Structural cycle labels stay near-black even when their arrows are "
                "purple."
            ),
            "near_black_required_for": [
                "the record identification",
                "panel and map headings",
                "plotting-region names",
                "units",
                "map count lines",
                "inverse-stage labels",
                "the Conditioned Parameters exit label",
            ],
            "muted_gray_scope": [
                "axes",
                "reference rules",
                "quiet geographic context",
                "other nonverbal scaffolding",
            ],
            "muted_gray_forbidden_for": [
                "the record identification",
                "plotting-region names",
                "units",
                "map count lines",
                "inverse-stage labels",
                "any generic secondary-label style",
            ],
            "e0_subordination": (
                "E0 may remain subordinate through smaller type and placement, but it must still "
                "be readily legible at final size; it is not washed into gray (section 9)"
            ),
            "forbidden": [
                "overall title",
                "all-caps headings",
                "paragraph-like annotations",
                "region names set at or above the panel-heading weight",
                "gray text used as a generic secondary-label style",
                "concepts placed inside tinted cards",
            ],
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
            "reader_facing_identification": {
                "hex": FIG01_IDENTIFICATION_COLOR,
                "applies_to": (
                    "reader-facing identification, titles, region names, units, map count lines "
                    "and inverse-stage labels (handoff section 9)"
                ),
            },
            "muted_gray": (
                "reserved for axes, reference rules, quiet geographic context and other "
                "nonverbal scaffolding; never a generic secondary-label style for text"
            ),
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
            "a data-like meter glyph -- the reviewed proof's three-circle mark included -- while "
            "no E3 observations are frozen",
            "an actual trace compressed into a side strip beside a full-width trace it is "
            "compared with",
            "an actual trace and a symbolic glyph paired inside one apparently equivalent "
            "evidence region",
            "an arrowhead directed into flux or meter observations",
            "a bracket or tie spanning an entire seasonal record",
            "a second irrigation-event encoding alongside the applied-water stems",
            "a card or box around the inverse cycle",
            "a faint driver lane that reads as an axis or y-tick spine",
            "an E3 transfer path routed along or beneath the E2 frame",
            "more than five full-height plotting regions in panel (a)",
            "a plotting region without a visible y-spine, two labelled bound values and units",
            "irrigation stems normalized to the depletion scale instead of carrying their own",
            "the SWIM-RS and flux ET traces separated into independent full-height lanes, or "
            "drawn on different date or vertical mappings",
            "two disconnected L-shaped legs, or a generic one-way transformation, in place of "
            "the closed inverse cycle",
            "the record identification set as muted gray secondary text or as a middle-dot "
            "subtitle",
            "gray text used as a generic secondary-label style anywhere in the figure",
            "the HUC8 subdivisions drawn on the E3 map",
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
            ["etf_ensemble", "compare"],
            ["run_balance", "compare"],
            ["compare", "update_parameters"],
            ["update_parameters", "run_balance"],
            ["update_parameters", "daily_balance"],
            ["ndvi_captures", "run_balance"],
            ["daily_forcing", "run_balance"],
            ["daily_balance", "daily_et"],
            ["daily_balance", "applied_water"],
            ["e3_applied_water", "e3_annual_sum"],
            ["e0_tag", "e1_map"],
            ["e1_map", "irrigated_params"],
            ["e1_map", "rainfed_params"],
            ["irrigated_params", "e2_map"],
            ["irrigated_params", "e3_map"],
            ["rainfed_params", "e2_map"],
        ],
        "edge_labels": {
            "update_parameters -> daily_balance": FIG01_CYCLE_EXIT_LABEL,
        },
        "cycle_edges": [list(e) for e in FIG01_CYCLE_EDGES],
        "cycle_exit_edge": list(FIG01_CYCLE_EXIT_EDGE),
        "comparisons": [
            {
                "a": "daily_et",
                "b": "flux_et",
                "kind": "shared alignment",
                "directed": False,
                "treatment": (
                    "SHARED ALIGNMENT ONLY. In 3.2.0 the two traces are lanes of ONE plotting "
                    "region (handoff section 5.2 item 5), sharing one axis, one date mapping and "
                    "one vertical scale, distinguished by two direct labels. NO bracket, tie "
                    "glyph, or connector of any kind is drawn between them, and no arrowhead may "
                    "point into the flux lane. Section 5.2 also forbids an example metric, a "
                    "residual annotation, a causal arrow, and a separate comparison bracket."
                ),
                "drawn_glyph": None,
                "region": "et_comparison",
                "scope": "E1 example, US-Bi1",
            },
            {
                "a": "e3_annual_sum",
                "b": "meters",
                "kind": "neutral tie",
                "directed": False,
                "treatment": (
                    "a plain em dash between the two terms inside the separate E3 evaluation "
                    "key, as written in handoff sections 4 and 5.4. No arrowhead into the meter "
                    "term, and no data marks on either side while no E3 observations are frozen."
                ),
                "drawn_glyph": "—",
                "scope": "E3 experiment-level key; not the US-Bi1 record",
            },
        ],
        "comparison_rules": [
            "a comparison is carried by shared alignment or a neutral typographic relation, "
            "never by an arrow that could imply the model produced the observation",
            "the daily ET comparison carries NO drawn glyph at all: one shared axis, one date "
            "mapping, one vertical scale and two direct labels are the entire treatment",
            "the two sides of a comparison must be on an honest common support: one common date "
            "and y mapping for the daily ET pair, one common annual aggregation for the E3 pair",
            "no comparison may join a node of the E1 example to a node of the E3 evaluation key",
        ],
        "forbidden_edges": [
            ["flux_et", "run_balance"],
            ["meters", "run_balance"],
            ["flux_et", "compare"],
            ["meters", "compare"],
            ["flux_et", "update_parameters"],
            ["meters", "update_parameters"],
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
            ["applied_water", "meters"],
            ["applied_water", "e3_annual_sum"],
            ["applied_water", "e3_applied_water"],
            ["e3_annual_sum", "applied_water"],
            ["e3_annual_sum", "meters"],
            ["daily_et", "flux_et"],
            ["swe_constraint", "compare"],
            ["swe_inline", "compare"],
            ["swe_constraint", "inverse_estimation"],
            ["swe_inline", "inverse_estimation"],
            ["etf_ensemble", "run_balance"],
            ["ndvi_captures", "compare"],
            ["daily_forcing", "compare"],
        ],
        "forbidden_edge_reasons": {
            "applied_water -> meters": (
                "the linkage architecture 3.0.0 asserted and this revision removes. The E1 "
                "example's simulated irrigation is not an E3 meter pair; E3 evaluates applied "
                "water simulated for E3 fields."
            ),
            "applied_water -> e3_annual_sum / e3_applied_water": (
                "the same false linkage routed through the E3 key's own nodes"
            ),
            "e3_annual_sum -> applied_water": "the same linkage in reverse",
            "e3_annual_sum -> meters": (
                "the E3 relation is a neutral comparison tie, not an arrow into a held-out "
                "observation"
            ),
            "daily_et -> flux_et": (
                "the two traces share one axis; their relationship is alignment, not an arrow "
                "implying the model produced the observation"
            ),
            "flux_et / meters -> run_balance, compare, update_parameters": (
                "handoff section 10.3: 'Held-out flux and meter observations remain forbidden as "
                "cycle inputs.' Every stage of the closed cycle is protected, not just the "
                "retired single inverse node."
            ),
            "swe_constraint / swe_inline -> compare or inverse_estimation": (
                "SWE is an inline label on the incoming ETf constraint route, not a separate "
                "node with its own arrow"
            ),
            "etf_ensemble -> run_balance": (
                "the acquisition-date targets enter the COMPARE stage. Routing them into the "
                "balance would say the satellite retrievals drive the forward model rather than "
                "constrain its parameters (handoff section 5.3)."
            ),
            "ndvi_captures / daily_forcing -> compare": (
                "NDVI and daily forcing drive the forward balance; they are not calibration "
                "targets and must not enter the comparison stage (handoff section 5.3)"
            ),
        },
        "edge_rules": [
            "no arrow may run from a held-out observation into any stage of the inverse cycle, "
            "the displayed daily balance, or either class-specific parameter set",
            "no edge may originate at a held-out observation at all",
            "comparison with a held-out observation is a neutral tie recorded under comparisons, "
            "never a directed edge; no arrowhead points into flux or meter observations",
            "no drawn relationship of any kind -- edge, comparison, bracket, tie or shared glyph "
            "-- may connect the E1 example's applied-water node to the E3 meter node, directly "
            "or through any intermediate node",
            "both transfer paths originate at the E1 map; there is no E2-to-E3 edge",
            "E2 receives both parameter classes; E3 receives only the irrigated class",
            "the E3 path branches visibly at the irrigated parameter token, before the E2 frame",
            "the ETf constraint path and the NDVI/forcing driver path must be visually "
            "distinguishable by routing and stroke treatment; SWE rides the constraint path as "
            "an inline label rather than its own arrow",
            "the constraint route enters 'Compare'; the driver route enters 'Run Balance'; the "
            "three cycle edges close the loop in the order Run Balance -> Compare -> Update "
            "Parameters -> Run Balance",
            "the 'Conditioned Parameters' exit leaves the update stage for the displayed daily "
            "trajectory and is the only edge out of the cycle",
            "panels (a) and (b) are not joined by an arrow; the removed inverse_estimation -> "
            "e1_source connector belonged to the superseded transfer-ribbon composition",
        ],
        "open_decisions": [
            "No E3 meter observation values are frozen in the Figure 1 package, so the E3 "
            "evaluation key stays typographic and schematic. Drawing real meter marks requires "
            "freezing observations chosen by a support rule independent of simulated "
            "performance, the corresponding E3 model output on an honest common support, and a "
            "separate selection record (handoff section 5.4).",
            "'PEST++ IES' is removed from the artwork and carried by the caption. Handoff "
            "section 5.3 permits its return only if a final-size proof shows it adds useful "
            "specificity without enlarging the inverse element; that proof has not been made.",
            "'spread-weighted' is removed from the artwork and carried by the caption. Handoff "
            "section 5.3 permits its return on the incoming target route only if a final-size "
            "proof shows it materially clarifies the member-spread encoding; that proof has not "
            "been made, and a return would also require restoring its caption-guard exemption.",
            "The 'daily drivers' bracket is conditional: handoff section 5.3 says 'if retained' "
            "it must span only the NDVI and forcing regions and use one short connector into "
            "Run Balance. Whether it survives at final size is a proof-level decision; it stays "
            "in the frozen copy so a proof that keeps it draws the agreed string.",
            "A faint privacy-safe E3 hillshade is optional. Handoff section 6.7: 'A subtle "
            "privacy-safe hillshade remains optional and requires an archived, licensed, hashed "
            "source. Its absence does not block Gate B.' It may only be attempted once a "
            "licensed DEM and its derivation are archived and hashed the way "
            "conus_states_context and slv_basin_context are, and it must not resolve terrain "
            "finely enough to localize a field beyond the approved 1 km centroid snap.",
        ],
        "decisions_closed_2026_08_25": [
            "The closed inverse cycle, carried as an open decision since 3.0.0, is now REQUIRED. "
            "Handoff sections 5.3, 10.3 and 11 mandate the directed cycle 'Run Balance -> "
            "Compare -> Update Parameters -> Run Balance' and the 'Conditioned Parameters' exit; "
            "the single one-way parameter-update arrow is superseded.",
            "conus_states_context is ACCEPTED and drawn: 'Retain the faint CONUS state "
            "boundaries accepted in revision 3' (section 6.7).",
            "slv_basin_context is REJECTED as artwork: 'Keep the HUC8 subdivisions out of the E3 "
            "map' (section 6.7). The layer stays in fig01_scope.gpkg as archived provenance and "
            "is simply not rendered.",
            "Panel (b) is ACCEPTED as built in revision 3 (section 6.1); subsequent changes to "
            "arc curvature, map positions, clearances, or stroke weights are render-only unless "
            "they change the source or destination of a path.",
        ],
        "panel_b_status": {
            "revision": "r3",
            "accepted": True,
            "source": "handoff section 6.1",
            "accepted_topology": (
                "the irrigated triangle is the fork, one horizontal leg reaches E2, and one "
                "visibly divergent curved leg reaches E3"
            ),
            "subsequent_changes": (
                "arc curvature, map positions, clearances and stroke weights are render-only "
                "(Level 1) adjustments unless they change the source or destination of a path, "
                "which would be a Level 2 contract change"
            ),
            "redesign_scope_2026_08_25": "panel (a) only",
        },
        "revision_protocol": {
            "source": "handoff section 15, 'Fast revision and rebuild protocol'",
            "rule": (
                "classify every requested change before editing and use the lowest level that "
                "preserves scientific meaning and provenance. Contract review, background agents "
                "and 'build_figure_data.py --all' are not gates for geometry or style work."
            ),
            "levels": {
                "0_svg_markup": (
                    "a reviewer moves, resizes or restyles existing elements in Inkscape to "
                    "communicate a preference; edit a copy named '*_markup.svg' only, rebuild "
                    "nothing"
                ),
                "1_render_only": (
                    "positions, margins, panel heights, axis limits, generated tick placement, "
                    "font sizes or weights, colors, line weights, map extents using the same "
                    "geometries, connector routing; active proof script only, no figure-data "
                    "build"
                ),
                "2_contract": (
                    "any visible string; adding or removing a label; node, edge, comparison or "
                    "held-out semantics; panel grouping; data role; or relationship change. "
                    "Rebuild through 'uv run python scripts/figures/build_figure_data.py --all' "
                    "and regenerate the architecture, metadata and manifest records."
                ),
                "3_display_data": (
                    "new or changed plotted values, columns, transformations, site/window "
                    "selection, aggregation, cohort/class assignment, geography, or privacy "
                    "generalization; full '--all' build plus a full scientific, provenance, "
                    "selection, cohort, privacy and render audit"
                ),
            },
            "this_change": (
                "Level 2. Handoff section 16: 'This 2026-08-25 redesign is a Level 2 contract "
                "change because it changes visible copy, panel grouping, and "
                "inverse-estimation relationships.'"
            ),
            "level_2_value_equality_required_for": [
                "fig01_example_timeseries.csv",
                "fig01_example_selection.json",
                "fig01_scope.gpkg",
                "fig01_evidence_matrix.csv",
            ],
            "level_2_value_equality_rule": (
                "'A changed value in those files without a declared Level 3 reason is a failure.' "
                "fig01_architecture.json, contract-bearing metadata and manifest entries are the "
                "expected changes. fig01_scope.gpkg differs by its gpkg_contents.last_change "
                "timestamp on every build, so layer content -- not the file hash -- is the valid "
                "equality test."
            ),
            "axis_domain_note": (
                "axis-domain and tick changes stay Level 1 when they use the same frozen "
                "columns, do not transform values, and do not clip marks"
            ),
            "svg_markup_channel": {
                "source": "handoff section 15.2",
                "requirement": (
                    "every proof builder must write an editable SVG alongside the PDF, PNG, "
                    "string ledger and notes, on the same 190 x 120 mm canvas, assigning stable "
                    "IDs to panels, axes, labels, routes, map groups and data-mark groups so "
                    "markup deltas can be identified reliably"
                ),
                "cycle": [
                    "render the deterministic script to SVG/PDF/PNG",
                    "copy the SVG to '*_markup.svg' and edit that copy in Inkscape",
                    "read the moved elements and transforms from the markup copy",
                    "transcribe accepted deltas into named constants or layout logic in the "
                    "proof script",
                    "rerender cleanly and compare the generated SVG with the markup intent",
                ],
                "rule": (
                    "the markup SVG is a communication artifact, not a publication artifact and "
                    "not the source of record. Never use it to hand-edit data positions, visible "
                    "copy, or scientific relationships; only clean scripted outputs may advance "
                    "to Gate B."
                ),
                "stable_id_groups": [
                    "panels",
                    "axes",
                    "labels",
                    "routes",
                    "map groups",
                    "data-mark groups",
                ],
            },
            "proof_directory_practice": (
                "handoff section 15.3: preserve revisions 1-3 as review provenance, create one "
                "active revision-4 working directory, and start a new numbered directory after a "
                "Level 2 or Level 3 change. Architecture 3.2.0 IS a Level 2 change, so revision "
                "4 must be written to a new proof directory; fig01_graybox_110.png, "
                "fig01_evidence_190.png and every accepted r2/r3 artifact stay untouched."
            ),
        },
        "superseded_requirements_removed": {
            "source": "handoff section 10.3",
            "instruction": (
                "'Remove superseded seven-lane, one-way inverse, transfer-ribbon, and "
                "crop-cross-section requirements.'"
            ),
            "seven_lane": (
                "REMOVED. Replaced by the five plotting_regions; the evidence_rows key itself is "
                "gone so no downstream script can read the old grouping."
            ),
            "one_way_inverse": (
                "REMOVED. Replaced by inverse_cycle with the three cycle edges and the labelled "
                "Conditioned Parameters exit; the etf_ensemble -> inverse_estimation and "
                "inverse_estimation -> daily_balance edges are gone from the edge list."
            ),
            "transfer_ribbon": (
                "REMOVED in 3.0.0 and kept out. The unlettered full-width ribbon and its three "
                "ribbon nodes remain in also_supersedes (2.1.0) and in "
                "forbidden_visual_treatments; the maps themselves are the transfer endpoints "
                "(handoff section 4)."
            ),
            "crop_cross_section": (
                "REMOVED in 3.0.0 and kept out. The generic crop-soil-water process "
                "cross-section stays in forbidden_visual_treatments; handoff section 5.2 item 4 "
                "replaces it with the coordinated state-and-irrigation region."
            ),
        },
        "string_classification": {},
        "caption_facts": {
            "_ownership": "caption/manuscript-owned",
            "_rule": (
                "Recorded here for caption drafting and audit only. The builder never promotes "
                "any of these strings into visible copy, and an assertion fails the build if one "
                "appears among strings classified title, direct_label, or annotation."
            ),
            "contract": (
                "paper/notes/fig01_production_handoff.md section 12, revised 2026-08-24, eight "
                "required items"
            ),
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
                "Inverse estimation uses PEST++ IES. This is relative reliability information, "
                "not calibrated uncertainty. The objective and the weight expression stay in the "
                "Methods."
            ),
            "item_4_outputs": (
                "The E1 example shows two kinds of daily evidence: the model's daily ET, aligned "
                "with the held-out flux record on a common date axis, and its simulated "
                "irrigation events, shown as magnitude-bearing applied-water stems that "
                "illustrate the model's daily output form. Separately, and not as an extension "
                "of that record, E3 aggregates daily gross applied water from E3 simulations to "
                "annual totals for comparison with meter records. The plotted E1 applied-water "
                "series is not an E3 meter pair."
            ),
            "item_4_note": (
                "Revised 2026-08-24. Architecture 3.0.0's caption conflated the two: it read the "
                "E1 example's applied water as the series aggregated for E3, which the "
                "artwork then drew as a linkage. Handoff section 12 item 4 now requires the "
                "distinction."
            ),
            "item_5_held_out": (
                "Flux ET evaluates E1 and E2; metered applied water evaluates E3. Flux ET and "
                "meter records were withheld from parameter estimation and from transfer-vector "
                "construction in every experiment. Where a model output and an observation are "
                "drawn together they are aligned for post-hoc comparison only; no arrow runs "
                "from a model output into an observation."
            ),
            "item_5_note": (
                "3.1.1 (review directive d1): the flux lane in panel (a) is now labelled simply "
                "'Flux ET' because the drawn trace is one US-Bi1 / E1 record. The E1-E2 "
                "flux-evaluation scope that the 'Flux ET · E1–E2' label used to assert is "
                "caption-owned and is carried by the first sentence of this item. The sentence "
                "about arrows was also corrected: since 3.1.0 there is no model-output-to-"
                "observation arrow to qualify, and since 3.1.1 the daily ET comparison carries "
                "no drawn glyph at all."
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
                "estimation with PEST++ IES; and daily meteorological forcing drives a "
                "mass-conserving balance that carries soil-water state between acquisitions. The "
                "example aligns daily SWIM-RS ET with held-out flux ET on a common date axis and "
                "shows simulated irrigation events as evidence of the model's daily output form. "
                "Separately, E3 aggregates daily gross applied water from E3 simulations to "
                "annual totals for comparison with meter records; the plotted E1 applied-water "
                "record is not an E3 meter pair. Flux and meter observations were withheld from "
                "parameter estimation and transferred-parameter construction. (b) The 60-site E1 "
                "CONUS cohort supplies separate irrigated and rainfed parameter sets. Both are "
                "applied without field-specific calibration across the 66-site, ten-country E2 "
                "experiment, whereas the irrigated set is applied to 50 metered fields in the "
                "San Luis Valley. E0 used the E1 cohort's flux observations after satellite "
                "calibration to select the vegetation formulation, so E1 flux evaluation is "
                "external to parameter estimation but not fully independent of model "
                "development. Map lines delineate study areas and do not necessarily depict "
                "accepted national boundaries."
            ),
            "working_caption_source": (
                "paper/notes/fig01_production_handoff.md section 12, revised 2026-08-24, "
                "transcribed verbatim"
            ),
            "working_caption_status": (
                "to be reconciled with the finished render (handoff section 12 and section 15 "
                "item 3)"
            ),
            "caption_owned_engine_label": (
                "PEST++ IES. Removed from the artwork by the 2026-08-24 revision (handoff "
                "section 5.3) and carried here."
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
                "Flux ET evaluates E1 and E2; metered applied water evaluates E3. Both were "
                "withheld from parameter estimation and from transfer-vector construction in "
                "every experiment. Where a model output and an observation are drawn together "
                "they are aligned for post-hoc comparison only; no arrow runs from a model "
                "output into an observation."
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
    _classify(
        arch["example_record"]["record_label"],
        arch["example_record"]["record_label_string_class"],
    )
    for row in arch["plotting_regions"]:
        _classify(row["heading"], row["heading_string_class"])
        _classify(row["axis_label"], row["axis_label_string_class"])
        for dl in row.get("direct_labels", []):
            _classify(dl["label"], dl["string_class"])
        for lane in row.get("lanes", []):
            _classify(lane["label"], lane["string_class"])
        agg = row.get("aggregation_mark")
        if agg:
            _classify(agg["label"], agg["string_class"])
    _classify(arch["date_axis"]["label"], arch["date_axis"]["string_class"])
    _classify(arch["inverse_cycle"]["title"], arch["inverse_cycle"]["title_string_class"])
    for stage in arch["inverse_cycle"]["stages"]:
        _classify(stage["label"], stage["string_class"])
    _classify(
        arch["inverse_cycle"]["exit"]["label"],
        arch["inverse_cycle"]["exit"]["string_class"],
    )
    _classify(arch["swe_constraint"]["label"], arch["swe_constraint"]["string_class"])
    _classify(arch["driver_routing"]["label"], arch["driver_routing"]["string_class"])
    for o in arch["outputs"]:
        _classify(o["label"], o["string_class"])
    _classify(arch["held_out"]["heading"], arch["held_out"]["heading_string_class"])
    for o in arch["held_out"]["observations"]:
        _classify(o["label"], o["string_class"])
    for node in arch["e3_evaluation_key"]["nodes"]:
        _classify(node["label"], node["string_class"])
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
    # No string retired by ANY superseded architecture may reappear in visible
    # copy.  The union is taken over every supersession block so that adding a
    # revision cannot silently drop an older block's retired strings.
    retired: set[str] = set()
    for block in arch.values():
        if isinstance(block, dict) and "removed_strings" in block:
            retired |= set(block["removed_strings"])
    if (
        not {
            "Flux ET · E1–E2",
            "Σ year",
            "annual sum",
            "PEST++ IES",
            "US-Bi1 · 2017",
            "Flux ET",
            "Held-Out Evaluation",
            "Inverse Estimation",
            "spread-weighted",
            "Daily State",
            "Daily Outputs",
            "Daily Gross Applied Water → Annual Total ⊢—⊣ Metered Water · E3",
        }
        <= retired
    ):
        raise BuildError(
            "fig01: the retired-string set lost an entry; every superseded architecture's "
            "removed_strings must stay under the revival guard"
        )
    revived = sorted(retired & set(cls))
    if revived:
        raise BuildError(f"fig01: superseded string(s) {revived} reached visible copy")

    # ---- section 11 (2026-08-25): visible record identification ----
    if cls.get(FIG01_RECORD_ID) not in {"title", "direct_label"}:
        raise BuildError(
            f"fig01: the record identification {FIG01_RECORD_ID!r} must be visible artwork copy; "
            "the reader may not be left to infer the site and year from the caption "
            "(handoff sections 5.1, 11 and 13)"
        )
    if not arch["example_record"]["site_id_is_visible"]:
        raise BuildError(
            "fig01: example_record.site_id_is_visible must be true for 3.1.0 and later"
        )
    if FIG01_EXAMPLE_SITE not in FIG01_RECORD_ID or FIG01_EXAMPLE_START[:4] not in FIG01_RECORD_ID:
        raise BuildError(
            f"fig01: the record identification {FIG01_RECORD_ID!r} does not name the frozen "
            f"site {FIG01_EXAMPLE_SITE!r} and year {FIG01_EXAMPLE_START[:4]!r}"
        )

    # ---- section 11 (2026-08-25): the identification is near-black, not gray ----
    # "US-Bi1 (2017) is absent from the visible record identification or is
    # rendered as muted gray secondary text" stops the build.
    rec = arch["example_record"]
    if rec["record_label_rendered_as_muted_gray"]:
        raise BuildError(
            f"fig01: the record identification {FIG01_RECORD_ID!r} is marked as muted gray "
            "secondary text. Handoff sections 5.1, 9 and 11 require near-black identification "
            "and forbid washing it into a gray subtitle."
        )
    if rec["record_label_color"] != FIG01_IDENTIFICATION_COLOR:
        raise BuildError(
            f"fig01: the record identification must be near-black {FIG01_IDENTIFICATION_COLOR!r}, "
            f"got {rec['record_label_color']!r} (handoff sections 5.1 and 9)"
        )
    if arch["typography"]["identification_color"] != FIG01_IDENTIFICATION_COLOR:
        raise BuildError(
            "fig01: typography.identification_color must be the near-black "
            f"{FIG01_IDENTIFICATION_COLOR!r} required by handoff section 9"
        )
    gray_scope = {s.lower() for s in arch["typography"]["muted_gray_scope"]}
    reader_facing = {
        "the record identification",
        "plotting-region names",
        "units",
        "map count lines",
        "inverse-stage labels",
    }
    if gray_scope & reader_facing:
        raise BuildError(
            "fig01: muted gray is reserved for axes, rules and quiet geographic context; "
            f"{sorted(gray_scope & reader_facing)} is reader-facing copy (handoff section 9)"
        )
    if "·" in FIG01_RECORD_ID:
        raise BuildError(
            "fig01: handoff section 5.1 retires the middle-dot record identification used in "
            f"revisions 1-3; {FIG01_RECORD_ID!r} still carries it"
        )

    # ---- section 11 (2026-08-25): five quantitative plotting regions ----
    regions = arch["plotting_regions"]
    if len(regions) > FIG01_MAX_PLOTTING_REGIONS:
        raise BuildError(
            f"fig01: panel (a) carries {len(regions)} full-height plotting regions; handoff "
            f"sections 4, 5.2 and 11 allow at most {FIG01_MAX_PLOTTING_REGIONS}. Root-zone "
            "depletion and irrigation share one region; daily ET and held-out flux ET share "
            "another."
        )
    if [r["order"] for r in regions] != list(range(1, len(regions) + 1)):
        raise BuildError(
            f"fig01: the plotting regions must be ordered 1..{len(regions)}, got "
            f"{[r['order'] for r in regions]}"
        )
    if {r["id"] for r in regions} != set(FIG01_DISPLAY_DOMAINS) - {
        "state_and_irrigation_secondary"
    }:
        raise BuildError(
            f"fig01: the frozen plotting-region ids {sorted(r['id'] for r in regions)} do not "
            "match the recorded display domains"
        )

    # ---- section 11 (2026-08-25): y-spine, two bounds, units, no clipping ----
    def _check_axis(region_id: str, axis: dict, key: str) -> None:
        if not axis["visible_spine"]:
            raise BuildError(
                f"fig01: plotting region {region_id!r} has no visible y-spine; every region must "
                "read as a quantitative small multiple (handoff sections 5.2, 9, 11 and 13)"
            )
        bounds = axis["labeled_bounds"]
        if len(bounds) < 2:
            raise BuildError(
                f"fig01: plotting region {region_id!r} labels {len(bounds)} bound value(s); at "
                "least two are required to define the displayed lower and upper bounds"
            )
        if not axis["units"]:
            raise BuildError(
                f"fig01: plotting region {region_id!r} declares no units; handoff section 5.2 "
                "requires units adjacent to the axis or the region label"
            )
        lo, hi = axis["display_domain"]
        if lo >= hi:
            raise BuildError(f"fig01: plotting region {region_id!r} has an empty display domain")
        dmin, dmax = _region_ranges[key]
        if dmin < lo or dmax > hi:
            raise BuildError(
                f"fig01: plotting region {region_id!r} clips its data: the frozen columns span "
                f"{dmin:.4g} to {dmax:.4g} but the recorded display limits are {lo} to {hi}. "
                "Handoff section 11 stops the build rather than let a plotted value be clipped."
            )

    for region in regions:
        _check_axis(region["id"], region["y_axis"], region["id"])
        if "secondary_y_axis" in region:
            _check_axis(
                region["id"],
                region["secondary_y_axis"],
                f"{region['id']}_secondary",
            )

    # ---- section 11 (2026-08-25): the merged state/irrigation region ----
    state = next(r for r in regions if r["id"] == "state_and_irrigation")
    if not state["merged"] or "secondary_y_axis" not in state:
        raise BuildError(
            "fig01: root-zone depletion and the irrigation stems share one coordinated region "
            "and each carries an explicit scale (handoff section 5.2 item 4)"
        )
    if state["irrigation_stems"]["normalized_to_state_scale"]:
        raise BuildError(
            "fig01: handoff section 5.2 item 4 prefers a compact secondary scale or a clearly "
            "separated internal sub-band to normalizing the irrigation stems"
        )

    # ---- section 11 (2026-08-25): one merged ET region, one y mapping ----
    etr = next(r for r in regions if r["id"] == "et_comparison")
    if not (etr["merged"] and etr["shared_axis"] and etr["shared_vertical_scale"]):
        raise BuildError(
            "fig01: daily SWIM-RS ET and held-out flux ET are plotted together on ONE axis with "
            "identical date support and vertical scale; they may not be separated into "
            "independent full-height lanes (handoff sections 5.2 item 5, 5.4 and 11)"
        )
    if {lane["id"] for lane in etr["lanes"]} != set(cdm_lanes := {"daily_et", "flux_et"}):
        raise BuildError(
            f"fig01: the merged ET region must carry exactly {sorted(cdm_lanes)}, got "
            f"{sorted(lane['id'] for lane in etr['lanes'])}"
        )

    # ---- section 11 (2026-08-24): one common date mapping for the ET traces ----
    cdm = arch["common_date_mapping"]
    if tuple(cdm["columns"]) != FIG01_COMMON_DATE_MAPPING_COLUMNS:
        raise BuildError(
            "fig01: the common date mapping must cover exactly the two compared ET traces "
            f"{list(FIG01_COMMON_DATE_MAPPING_COLUMNS)}, got {cdm['columns']}"
        )
    missing_cdm = [c for c in cdm["columns"] if c not in set(example.columns)]
    if missing_cdm:
        raise BuildError(
            f"fig01: common-date-mapping column(s) {missing_cdm} are absent from the frozen "
            "example record"
        )
    if cdm["axis"] != arch["date_axis"]["id"]:
        raise BuildError(
            f"fig01: the two ET traces must share the frozen date axis {arch['date_axis']['id']!r}"
        )
    if not (cdm["full_width"] and cdm["comparable_horizontal_extent"]):
        raise BuildError(
            "fig01: the SWIM-RS and flux ET traces are presented as a direct visual comparison, "
            "so both must be drawn full width at comparable horizontal extent (handoff sections "
            "5.4 and 11)"
        )
    if not (cdm["single_region"] and cdm["shared_vertical_scale"]):
        raise BuildError(
            "fig01: the two ET traces must share one plotting region and one vertical scale; "
            "different y mappings, or separation into independent full-height lanes, would "
            "misrepresent the comparison (handoff sections 5.2 item 5, 5.4 and 11)"
        )
    if cdm["region"] != "et_comparison":
        raise BuildError(
            f"fig01: the common date mapping must name the merged ET region, got {cdm['region']!r}"
        )
    if arch["held_out"]["narrow_right_hand_column"]:
        raise BuildError(
            "fig01: the held-out treatments must not be squeezed into one narrow right-hand "
            "column; the flux trace may not become a side strip (handoff section 5.4)"
        )
    out_lanes = {
        lane["id"]: lane for row in arch["plotting_regions"] for lane in row.get("lanes", [])
    }
    for lane_id in cdm["lanes"]:
        if lane_id not in out_lanes:
            raise BuildError(f"fig01: common-date-mapping lane {lane_id!r} is not a frozen lane")
    extents = {out_lanes[lane_id]["horizontal_extent"] for lane_id in cdm["lanes"]}
    if not all("full panel-(a) plotting width" in e for e in extents):
        raise BuildError(
            f"fig01: both compared ET lanes must span the full panel width, got {sorted(extents)}"
        )

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

    # ---- section 11 (2026-08-25): the inverse relationships form a closed cycle ----
    cyc = arch["inverse_cycle"]
    if [s["id"] for s in cyc["stages"]] != list(FIG01_CYCLE_STAGES):
        raise BuildError(
            f"fig01: the inverse cycle must run through {list(FIG01_CYCLE_STAGES)} in order, got "
            f"{[s['id'] for s in cyc['stages']]}"
        )
    for a, b in FIG01_CYCLE_EDGES:
        if (a, b) not in edges:
            raise BuildError(
                f"fig01: the inverse-estimation relationships do not form the directed internal "
                f"cycle 'Run Balance -> Compare -> Update Parameters -> Run Balance'; the edge "
                f"{a!r} -> {b!r} is missing (handoff sections 5.3, 10.3 and 11)"
            )
    # Walk the three edges and confirm they actually close on themselves rather
    # than merely being present.
    node = FIG01_CYCLE_STAGES[0]
    for _ in FIG01_CYCLE_STAGES:
        nxt = [d for s, d in FIG01_CYCLE_EDGES if s == node]
        if len(nxt) != 1:
            raise BuildError(f"fig01: cycle stage {node!r} does not have exactly one successor")
        node = nxt[0]
    if node != FIG01_CYCLE_STAGES[0]:
        raise BuildError(
            f"fig01: the inverse cycle does not close; walking from "
            f"{FIG01_CYCLE_STAGES[0]!r} ends at {node!r}"
        )
    if tuple(cyc["exit"]["edge"]) != FIG01_CYCLE_EXIT_EDGE or FIG01_CYCLE_EXIT_EDGE not in edges:
        raise BuildError(
            f"fig01: the {FIG01_CYCLE_EXIT_LABEL!r} exit "
            f"{FIG01_CYCLE_EXIT_EDGE[0]!r} -> {FIG01_CYCLE_EXIT_EDGE[1]!r} is absent; handoff "
            "section 5.3 requires a separate exit that leaves the update stage after convergence "
            "and feeds the displayed daily state and ET"
        )
    if cyc["exit"]["label"] != FIG01_CYCLE_EXIT_LABEL:
        raise BuildError(
            f"fig01: the cycle exit must be labelled {FIG01_CYCLE_EXIT_LABEL!r}, got "
            f"{cyc['exit']['label']!r}"
        )
    if arch["edge_labels"].get("update_parameters -> daily_balance") != FIG01_CYCLE_EXIT_LABEL:
        raise BuildError(
            f"fig01: the exit edge carries no {FIG01_CYCLE_EXIT_LABEL!r} label in edge_labels"
        )
    for c in cyc["constraint_inputs"]:
        if c["into"] != FIG01_CONSTRAINT_INTO:
            raise BuildError(
                f"fig01: the calibration constraint {c['from']!r} must enter "
                f"{FIG01_CONSTRAINT_INTO!r}, got {c['into']!r} (handoff section 5.3)"
            )
    for d in cyc["driver_inputs"]:
        if d["into"] != FIG01_DRIVER_INTO:
            raise BuildError(
                f"fig01: the driver {d['from']!r} must enter {FIG01_DRIVER_INTO!r}; NDVI and "
                "daily forcing drive the forward balance and are not calibration targets "
                "(handoff section 5.3)"
            )
    if cyc["boxed"] or cyc["title"] is not None:
        raise BuildError(
            "fig01: the inverse cycle stays compact and unboxed and carries no title; the three "
            "stage labels name it (handoff sections 5.3, 9 and 13)"
        )
    # Handoff section 10.3: held-out flux and meter observations are forbidden
    # as cycle inputs.  The generic firewall above covers directed edges; this
    # states the cycle-specific requirement explicitly so a negative test can
    # reach it.
    for obs in sorted(EVALUATION_NODES):
        for stage in FIG01_CYCLE_STAGES:
            if (obs, stage) in edges:
                raise BuildError(
                    f"fig01: held-out observation {obs!r} has an edge into the calibration cycle "
                    f"stage {stage!r} (handoff sections 5.3, 10.3 and 11)"
                )
            if [obs, stage] not in arch["forbidden_edges"]:
                raise BuildError(
                    f"fig01: {obs!r} -> {stage!r} must be listed as a forbidden edge so the "
                    "held-out firewall is asserted, not merely absent"
                )

    # ---- section 11 (2026-08-24): comparisons are ties, never arrows ----
    comparisons = [(c["a"], c["b"]) for c in arch["comparisons"]]
    for c in arch["comparisons"]:
        if c["directed"]:
            raise BuildError(
                f"fig01: comparison {c['a']!r}<->{c['b']!r} is marked directed; comparison with a "
                "held-out observation must be a neutral tie, never an arrow"
            )
        if (c["a"], c["b"]) in edges or (c["b"], c["a"]) in edges:
            raise BuildError(
                f"fig01: {c['a']!r}<->{c['b']!r} is both a comparison tie and a directed edge"
            )

    # ---- section 11 (2026-08-24): the E1 example never reaches the E3 meters ----
    # Reachability over the union of directed edges and undirected comparison
    # ties, so an indirect route through any intermediate node is caught too.
    adj: dict[str, set[str]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
    for a, b in comparisons:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    for start in FIG01_E1_APPLIED_WATER_NODES:
        seen_nodes = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in adj.get(node, ()):
                if nxt in seen_nodes:
                    continue
                if nxt == FIG01_E3_METER_NODE:
                    raise BuildError(
                        f"fig01: the E1 example's applied-water node {start!r} reaches the E3 "
                        f"meter node {FIG01_E3_METER_NODE!r} through {node!r}. E3 evaluates "
                        "applied water simulated for E3 fields, not this record; no drawn "
                        "relationship may connect them (handoff sections 5.2, 5.4 and 11)."
                    )
                seen_nodes.add(nxt)
                stack.append(nxt)
    aw = next(o for o in arch["outputs"] if o["id"] == "applied_water")
    if aw["compared_with"] is not None or aw["aggregation"] is not None:
        raise BuildError(
            "fig01: the E1 example's applied-water series carries no drawn comparison and no "
            "drawn aggregation"
        )
    for row in arch["plotting_regions"]:
        if row.get("aggregation_mark") is not None:
            raise BuildError(
                f"fig01: plotting region {row['id']!r} carries an aggregation mark; annual "
                "aggregation belongs only to the separate E3 evaluation key (handoff section 9)"
            )
        if row["id"] == "state_and_irrigation" and row.get("irrigation_events_repeated_here"):
            raise BuildError(
                "fig01: irrigation events are encoded once, as the applied-water stems; a second "
                "event lane must not be restored (handoff section 5.2)"
            )

    # ---- section 11 (2026-08-24): no data-like meter glyph without observations ----
    key = arch["e3_evaluation_key"]
    if key["frozen_observation_source"] is None:
        if key["data_like_glyph"]:
            raise BuildError(
                "fig01: no E3 observations are frozen in the Figure 1 package, so the E3 "
                "evaluation key must stay typographic and schematic; a data-like meter glyph "
                "would present a symbol as evidence (handoff sections 5.4 and 11)"
            )
        if key["treatment_class"] != "typographic_schematic":
            raise BuildError(
                "fig01: absent frozen E3 observations the evaluation key's treatment_class must "
                f"be 'typographic_schematic', got {key['treatment_class']!r}"
            )
        for node in key["nodes"]:
            if node["is_actual_data"]:
                raise BuildError(
                    f"fig01: E3 key node {node['id']!r} claims to be actual data, but no E3 "
                    "observations are frozen in this package"
                )
    else:
        # Actual E3 marks are permitted only with their own frozen values AND a
        # separately documented selection record (handoff sections 5.4 and 11).
        for field in ("frozen_observation_source", "frozen_observation_selection_record"):
            name = key.get(field)
            if not name or not (OUT / str(name)).exists():
                raise BuildError(
                    "fig01: actual E3 meter marks require both frozen observation values and a "
                    f"separately documented selection record; {field} is {name!r}"
                )
    meters_node = next(o for o in arch["held_out"]["observations"] if o["id"] == "meters")
    if meters_node["data_like_glyph"] or meters_node["is_actual_data"]:
        raise BuildError(
            "fig01: the 'Metered Water · E3' term must not be drawn as data while no E3 "
            "observations are frozen"
        )
    if meters_node["column"] is not None:
        raise BuildError("fig01: the E3 meter term has no source column and must not claim one")

    # ---- section 11 (2026-08-24): the E3 path branches at the irrigated token ----
    branch = arch["transfer_branch"]
    if branch["at"] != FIG01_TRANSFER_BRANCH_AT:
        raise BuildError(
            f"fig01: the transfer branch must sit at {FIG01_TRANSFER_BRANCH_AT!r}, got "
            f"{branch['at']!r}; a branch drawn at the E2 map reads as E2 -> E3"
        )
    if not branch["visible"]:
        raise BuildError(
            "fig01: the irrigated branch junction must be visible so both outgoing paths are "
            "seen to originate at the token (handoff sections 6.1, 6.3 and 13)"
        )
    if set(branch["outgoing"]) != set(FIG01_TRANSFER_DESTINATIONS):
        raise BuildError(
            f"fig01: the irrigated branch must serve {sorted(FIG01_TRANSFER_DESTINATIONS)}, got "
            f"{sorted(branch['outgoing'])}"
        )
    for dest in branch["outgoing"]:
        if (branch["at"], dest) not in edges:
            raise BuildError(
                f"fig01: the branch at {branch['at']!r} declares an outgoing path to {dest!r} "
                "that the frozen edge list does not carry"
            )
    if not any("E2" in r for r in branch["forbidden_routes"]):
        raise BuildError(
            "fig01: the branch must explicitly forbid the E3 route that runs along or beneath "
            "the E2 frame"
        )

    # ---- section 11 (2026-08-24): E0 placement, SWE inline, engine label ----
    if arch["development_tag"]["placement"] != "adjacent_to_e1_heading_or_parameter_relay_origin":
        raise BuildError(
            "fig01: the E0 tag belongs beside the E1 heading or the parameter-relay origin, not "
            "below the map as a detached footnote (handoff sections 6.4 and 7)"
        )
    if arch["swe_constraint"]["drawn_as_separate_node"] or arch["swe_constraint"]["drawn_as_edge"]:
        raise BuildError(
            "fig01: SWE is an inline label on the constraint route into 'Compare', not a "
            "separate chip with its own arrow (handoff section 5.3)"
        )
    if arch["swe_constraint"]["enters_stage"] != FIG01_CONSTRAINT_INTO:
        raise BuildError(
            f"fig01: the '+ SWE' constraint enters {FIG01_CONSTRAINT_INTO!r}, got "
            f"{arch['swe_constraint']['enters_stage']!r} (handoff section 5.3)"
        )
    if arch["inverse_cycle"]["engine_label_ownership"] != "caption":
        raise BuildError(
            "fig01: 'PEST++ IES' moves to the caption by default; restoring it to the artwork "
            "requires a final-size proof showing it adds specificity without enlarging the "
            "inverse element (handoff section 5.3)"
        )
    if arch["inverse_cycle"]["spread_label_condition"]["string"] in cls:
        raise BuildError(
            "fig01: 'spread-weighted' is caption-owned; handoff section 5.3 permits its return "
            "on the incoming target route only if a final-size proof shows it materially "
            "clarifies the member-spread encoding"
        )
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
            "Target for the revised Gate A proof per fig01_production_handoff.md section 4 "
            "(rewritten 2026-08-20, unchanged 2026-08-24): 190 mm x 120 mm with 3 mm outer "
            "margins (usable 184 x 114 mm), two labelled horizontal panels separated by a 3-4 mm "
            "gutter, no overall title inside the artwork. The rejected 145 mm composition and "
            "the superseded 110 mm map-plus-framework composition must not be restored. "
            "six_figure_plan.md section 3.1 was reconciled to 120 mm on 2026-08-20. The proof "
            "rendered against architecture 3.0.0 (fig01_evidence_190.png) is review provenance "
            "and must not be revised into the next proof."
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
        "context_datasets": {
            "_contract": (
                "handoff sections 6.4, 6.6 and 6.7 (revised 2026-08-24) direct the next Gate A "
                "proof to TEST faint CONUS state boundaries and a subtle, privacy-safe E3 basin "
                "context. Both sources below are archived in-repo, public domain, and hashed. "
                "Retention is a proof-level decision, so both layers are published as optional "
                "context and neither is required by the composition."
            ),
            "conus_states": {
                "layer": "conus_states_context",
                "name": "Natural Earth 50m admin 1 states provinces lakes",
                "version": _ne_states_version(),
                "path": str(NE_STATES),
                "sha256": sha256(NE_STATES),
                "archive_zip": str(NE_STATES.with_suffix(".zip")),
                "archive_zip_sha256": sha256(NE_STATES.with_suffix(".zip")),
                "url": (
                    "https://naciscdn.org/naturalearth/50m/cultural/"
                    "ne_50m_admin_1_states_provinces_lakes.zip"
                ),
                "license": "public domain (Natural Earth)",
                "archived": "2026-08-24",
                "selection": (
                    "iso_a2 == 'US' excluding AK and HI, clipped to the same "
                    "-125..-66.5 E, 24..49.5 N box conus_context uses; 49 features "
                    "(48 contiguous states plus the District of Columbia)"
                ),
                "note": (
                    "the same Natural Earth release (5.1.1) as the admin-0 source already used "
                    "for world_context, so the two boundary hierarchies are consistent"
                ),
            },
            "e3_basin": {
                "layer": "slv_basin_context",
                "name": "USGS Watershed Boundary Dataset, WBDHU8",
                "path": str(SLV_BASIN),
                "sha256": sha256(SLV_BASIN),
                "provenance_record": str(SLV_BASIN_SOURCE),
                "provenance_record_sha256": sha256(SLV_BASIN_SOURCE),
                "source_publication_date": "2025-01-07",
                "url": (
                    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/"
                    "GDB/WBD_National_GDB.zip"
                ),
                "license": (
                    "public domain (U.S. Geological Survey; 17 U.S.C. 105 - works of the U.S. "
                    "Government are not subject to copyright)"
                ),
                "archived": "2026-08-24",
                "selection": (
                    "the five HUC8 units of the Rio Grande headwaters accounting unit "
                    "(HUC6 130100): 13010001 Rio Grande Headwaters, 13010002 "
                    "Alamosa-Trinchera, 13010003 San Luis, 13010004 Saguache, 13010005 Conejos"
                ),
                "privacy_review": (
                    "HUC8 units span 1,987-6,576 km2. The builder asserts that no published unit "
                    "is smaller than 1000 km2 and that every generalized E3 display point falls "
                    "inside the layer, so the context cannot localize a field more precisely "
                    "than the approved 1 km EPSG:5070 centroid snap already does. No field "
                    "polygon, parcel, acreage, owner, or source-agency attribute is carried."
                ),
            },
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
            "conus_states_context": list(states_ctx.columns),
            "world_context": list(world_ctx.columns),
            "slv_context": list(slv_ctx.columns),
            "slv_basin_context": list(basin_ctx.columns),
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
            "conus_states_context_rows": int(len(states_ctx)),
            "slv_basin_context_rows": int(len(basin_ctx)),
            "record_identification_visible": arch["example_record"]["record_label"],
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
            "reader_facing_strings_architecture_3_1_0": {
                "PEST++ IES": "moved to the caption; the inverse node carries no engine subtitle",
                "SWE": "+ SWE (an inline label on the ETf-target relay, not a separate chip)",
                "annual sum": "Σ year",
                "Gross Applied Water": (
                    "Irrigation (the E1 example lane); 'Daily Gross Applied Water' is now "
                    "reserved for the separate E3 evaluation key"
                ),
                "irrigation": (
                    "removed; the duplicate triangle event lane in the daily-state row is gone "
                    "and the magnitude-bearing applied-water stems are the only irrigation "
                    "encoding"
                ),
                "2017": "US-Bi1 · 2017 (required visible record identification)",
            },
            "reader_facing_strings_architecture_3_1_0_note": (
                "Strings added by the 2026-08-24 revision, with no 3.0.0 predecessor: "
                "'US-Bi1 · 2017', '+ SWE', 'daily drivers', 'Irrigation', 'Daily Gross Applied "
                "Water', 'Σ year' and 'Daily Gross Applied Water → Σ year ⇄ Metered Water · E3'. "
                "Two of them were retired again by 3.1.1; see the next map."
            ),
            "reader_facing_strings_architecture_3_1_1": {
                "Flux ET · E1–E2": (
                    "Flux ET; the drawn trace is one US-Bi1 / E1 record and the E1-E2 "
                    "flux-evaluation scope is caption-owned"
                ),
                "Σ year": (
                    "Annual Total; the E3 key reads 'Daily Gross Applied Water → Annual Total "
                    "⊢—⊣ Metered Water · E3'"
                ),
            },
            "reader_facing_strings_architecture_3_1_1_note": (
                "The 2026-08-25 user review of r2 changed two strings and one string class. No "
                "string was added without a predecessor. 'Held-Out Evaluation' was unchanged as "
                "copy but moved from title to direct_label; 3.2.0 then retired it outright."
            ),
            "reader_facing_strings_architecture_3_2_0": {
                "US-Bi1 · 2017": (
                    "US-Bi1 (2017); near-black identification, no middle dot, no muted-gray "
                    "subtitle (handoff section 5.1)"
                ),
                "Flux ET": (
                    "Flux ET (Held Out); the held-out qualification travels with the trace in "
                    "the merged ET region (sections 5.2 item 5 and 5.4)"
                ),
                "Held-Out Evaluation": (
                    "removed; section 5.4 asks for the direct label 'rather than a separate "
                    "Held-Out Evaluation heading or rule competing for space'"
                ),
                "Inverse Estimation": (
                    "removed; the closed cycle is named by 'Run Balance', 'Compare' and 'Update "
                    "Parameters' (sections 4, 5.3, 9 and 13)"
                ),
                "spread-weighted": (
                    "removed; caption-owned by section 5.3, with a conditional return on the "
                    "incoming target route recorded in the architecture"
                ),
                "Daily State": "State + Irrigation (the merged region, section 4 wireframe)",
                "Daily Outputs": "Daily ET (the merged ET region, section 4 wireframe)",
                "Daily Gross Applied Water → Annual Total ⊢—⊣ Metered Water · E3": (
                    "Daily Gross Applied Water → Annual Total — Metered Water · E3; sections 4 "
                    "and 5.4 write the relation with a plain em dash instead of the improvised "
                    "'⊢—⊣' glyph"
                ),
            },
            "reader_facing_strings_architecture_3_2_0_note": (
                "Strings added by the 2026-08-25 handoff rewrite with no 3.1.1 predecessor: "
                "'Run Balance', 'Compare', 'Update Parameters' and 'Conditioned Parameters', the "
                "four labels of the closed inverse cycle. 'Daily ET' is not new -- it was "
                "already the model-trace direct label and now doubles as the merged region's "
                "facet name."
            ),
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
        "data_contract": (
            "paper/notes/fig01_production_handoff.md section 10, rewritten 2026-08-25. The "
            "panel (a) redesign is recorded in fig01_architecture.json under "
            "panel_a_redesign_2026_08_25; the earlier r2 user-review directives remain under "
            "review_directives_2026_08_25"
        ),
        "revision_level": (
            "Level 2 contract change (handoff sections 15.1 and 16). The frozen example, "
            "geography and evidence values are unchanged; the architecture, contract-bearing "
            "metadata and manifest entries are the expected rewrites."
        ),
        "review": {
            "handoff_rewrite_2026_08_25": {
                "date": "2026-08-25",
                "subject": "revision 3 of the proof",
                "status_line": (
                    "Revision 3 reviewed; panel (a) redesign and a new Gate A proof are required"
                ),
                "outcome": (
                    "panel (a) is redesigned into five quantitative plotting regions with a "
                    "closed inverse cycle; panel (b) is accepted as built in revision 3 "
                    "(handoff section 6.1). Architecture 3.2.0 freezes the new copy, grouping, "
                    "y-axis requirements and cycle relationships."
                ),
                "frozen_contract_changes": [
                    "five plotting regions replace the seven-lane structure",
                    "root-zone depletion and the irrigation stems merge into one coordinated "
                    "region with an explicit scale for each; the stems are not normalized",
                    "daily SWIM-RS ET and held-out flux ET merge onto one axis with identical "
                    "date support and vertical scale",
                    "inverse estimation becomes the closed cycle 'Run Balance -> Compare -> "
                    "Update Parameters -> Run Balance' with ETf and '+ SWE' entering Compare and "
                    "a 'Conditioned Parameters' exit from the update stage",
                    "'US-Bi1 · 2017' -> 'US-Bi1 (2017)' in near-black; 'Flux ET' -> 'Flux ET "
                    "(Held Out)'; 'Held-Out Evaluation', 'Inverse Estimation' and "
                    "'spread-weighted' retired",
                    "every region carries a visible y-spine, two labelled bounds, units, and "
                    "recorded display limits that clip no plotted value",
                    "near-black #202124 for all reader-facing identification; muted gray "
                    "confined to axes, rules and quiet geographic context",
                    "conus_states_context accepted and drawn; slv_basin_context archived but not "
                    "drawn (section 6.7)",
                ],
                "open": (
                    "a faint privacy-safe E3 hillshade remains optional and does not block "
                    "Gate B; it needs an archived, licensed, hashed DEM derivation first"
                ),
            },
            "user_review_2026_08_25": {
                "date": "2026-08-25",
                "subject": "the r2 proof",
                "outcome": (
                    "strong improvement; one revision before Gate B. Architecture 3.1.1 freezes "
                    "the subset of the directives that changes reader-facing copy, string class, "
                    "comparison treatment, or frozen geometry guidance."
                ),
                "frozen_contract_changes": [
                    "'Flux ET · E1–E2' -> 'Flux ET' (the drawn trace is one US-Bi1 / E1 record; "
                    "the E1-E2 flux-evaluation scope is caption-owned)",
                    "'Σ year' -> 'Annual Total' (a subscripted sigma would put 'year' below the "
                    "7.5 pt floor); the sigma exemption is removed from the caption guard",
                    "'Held-Out Evaluation' reclassified title -> direct_label, drawn as a "
                    "compact label on the held-out rule",
                    "the daily ET / flux ET comparison is shared alignment only, with no drawn "
                    "bracket or tie glyph",
                    "the irrigated triangle sits at the fork with its label to the left, and the "
                    "E3 leg leaves as a distinct diagonal or shallow curve",
                    "'+ SWE' sits on the incoming constraint route, not beside the inverse title",
                ],
                "open": (
                    "a faint privacy-safe E3 hillshade is user-optional and not required for "
                    "Gate B; it needs an archived, licensed DEM derivation first"
                ),
                "partly_superseded_by": (
                    "the 2026-08-25 handoff rewrite. Architecture 3.2.0 keeps the 'Annual Total' "
                    "operation, the shared-alignment ET comparison, the branch topology and the "
                    "'+ SWE' route, but replaces 'Flux ET' with 'Flux ET (Held Out)' and retires "
                    "'Held-Out Evaluation' entirely rather than reclassifying it."
                ),
            },
            "scientific_review": {
                "date": "2026-08-25",
                "outcome": (
                    "data contract revised for architecture 3.1.0, refined by 3.1.1, regrouped "
                    "without value change by 3.2.0"
                ),
                "detail": (
                    "All 2026-08-20 assertions are retained: E0 typed as model_development; "
                    "configured 60/66/50, the 13-site E1/E2 overlap, MB_Pch presence, and the "
                    "frozen source classes (E1 39 irrigated / 21 rainfed, E2 13 irrigated / 53 "
                    "rainfed); spread-weighted primaries and per-experiment ETf target "
                    "composition; the US-Bi1 example frozen from the audited fig03 and fig04 "
                    "artifacts with member marks reconciled to the frozen member count and "
                    "target mean; both transfer paths originating at E1 with no E2-to-E3 edge "
                    "and no evaluation-to-fitting edge. Added 2026-08-24: the false E1-to-E3 "
                    "linkage is removed and refused by a reachability check over edges and "
                    "comparison ties; comparison with a held-out observation must be a neutral "
                    "tie rather than a directed edge; the two ET traces must share one date "
                    "mapping at full width; the E3 evaluation key must stay typographic while no "
                    "E3 observations are frozen; the E3 transfer path must branch visibly at the "
                    "irrigated parameter token; and 'US-Bi1 · 2017' must appear as visible copy."
                ),
                "revised_2026_08_25": (
                    "Architecture 3.2.0 changes how these assertions are GROUPED and LABELLED, "
                    "not what they assert. The scientific content is unchanged: the same frozen "
                    "columns, the same 15 calibration captures, the same member count and target "
                    "mean, the same source classes, the same transfer topology, and the same "
                    "held-out firewall -- now extended so that no held-out observation may enter "
                    "any of the three inverse cycle stages. Two label changes carry scientific "
                    "weight: the drawn flux trace is qualified in place as 'Flux ET (Held Out)', "
                    "and 'spread-weighted' moves from artwork to caption, where the weighting "
                    "scheme can be stated precisely instead of compressed into two words."
                ),
                "supersedes": "2026-08-20 review of architecture 3.0.0",
            },
            "privacy_review": {
                "date": "2026-08-25",
                "outcome": (
                    "approved; the basin context layer is archived but no longer drawn, and the "
                    "optional hillshade stays unbuilt"
                ),
                "detail": (
                    "E3 display remains 1 km-snapped centroids; no exact polygons, no "
                    "source-agency identifiers, public source_key is a salted truncated hash. "
                    "The example record adds no restricted content: it is one public AmeriFlux "
                    "site. New for 3.1.0: slv_basin_context publishes five HUC8 watershed units "
                    "of 1,987-6,576 km2 from the public-domain USGS WBD. The builder refuses any "
                    "unit smaller than 1000 km2 and requires every generalized display point to "
                    "fall inside the layer, so the context adds orientation without adding "
                    "location precision. conus_states_context carries only state names and "
                    "postal codes. Open item carried forward: the E3 audit-key salt is published "
                    "in this public metadata file; the mechanism is left as-is."
                ),
                "revised_2026_08_25": (
                    "Handoff section 6.7 retains the CONUS state boundaries as drawn context, "
                    "keeps HUC8 watershed units out of the artwork, and leaves a faint "
                    "privacy-safe hillshade optional. slv_basin_context therefore stays in "
                    "fig01_scope.gpkg as archived, hashed provenance -- removing it would break "
                    "the Level 2 value-equality requirement -- but e3_map.optional_context_layer "
                    "is set to null so no proof draws it. Nothing is published that was not "
                    "already published under 3.1.0, and one layer moves from candidate artwork to "
                    "provenance only, which is a strict reduction in what a reader can see."
                ),
            },
            "visual_review": {
                "date": "2026-08-25",
                "outcome": (
                    "architecture 3.0.0 and its proof reviewed and superseded; the r2 proof "
                    "reviewed as a strong improvement and superseded by 3.1.1; revision 3 "
                    "reviewed and panel (a) sent back for redesign, panel (b) accepted as built; "
                    "3.2.0 frozen, no proof yet exists for it"
                ),
                "detail": (
                    "The evidence-first thesis survived review; the first proof "
                    "(fig01_evidence_190.png) did not. It compressed the actual flux record into "
                    "a side strip beside the full-width model trace, paired that actual trace "
                    "with a symbolic three-circle meter glyph inside one apparently equivalent "
                    "region, drew an 'annual sum' bracket across the E1 example's seasonal "
                    "applied-water record that asserted an E1-to-E3 data linkage, repeated the "
                    "irrigation events in a second triangle lane, boxed the inverse node and hung "
                    "a 'PEST++ IES' subtitle on it, ran a faint driver lane that read as an axis "
                    "spine, routed the E3 transfer path along the bottom of the E2 frame, and "
                    "left the record unidentified. Architecture 3.1.0 corrects each of those, "
                    "and the r2 proof rendered against it was accepted as a strong improvement. "
                    "The 2026-08-25 user review of r2 returned four must-address items -- the "
                    "'Flux ET · E1–E2' mislabel, the improvised 'Σ year' operator, 'Held-Out "
                    "Evaluation' as a left-column heading opening a large empty region, and a "
                    "right-margin comparison bracket that read as an empty box at thumbnail "
                    "scale -- plus refinements to the E3 branch geometry and the '+ SWE' "
                    "placement. Architecture 3.1.1 freezes those. Revision 3, rendered against "
                    "3.1.1, was reviewed on 2026-08-25: panel (b) was ACCEPTED as built, and "
                    "panel (a) was sent back for redesign. The r3 panel (a) still read as seven "
                    "thin sparklines without y-axes or units, so no reader could recover a "
                    "magnitude; the state and irrigation lanes were separated so event-to-state "
                    "response was invisible; the two ET traces sat in different lanes rather than "
                    "on one shared axis; the inverse element read as two disconnected L-shaped "
                    "legs under a heading instead of a closed iteration; and the record "
                    "identification was set as a muted gray middle-dot subtitle that read as "
                    "decoration. Architecture 3.2.0 freezes the five-region regrouping, the "
                    "y-axis contract, the closed cycle with its Conditioned Parameters exit, and "
                    "the near-black identification policy. A fresh Gate A revision must be "
                    "written to a NEW proof directory; no earlier proof image may be revised "
                    "into it."
                ),
                "gate_a_status": "not yet produced against architecture 3.2.0",
                "gate_b_status": "not started",
                "panel_b_status": (
                    "accepted as built in revision 3 (handoff section 6.1); subsequent panel (b) "
                    "changes are render-only unless a path source or destination changes"
                ),
                "context_layer_decision": {
                    "conus_states_context": "accepted, drawn as quiet muted-gray context",
                    "slv_basin_context": "archived and hashed, not drawn (handoff section 6.7)",
                    "e3_hillshade": (
                        "optional, not built; needs an archived, licensed, hashed DEM derivation "
                        "first and does not block Gate B"
                    ),
                },
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
            "conus_states_context": len(states_ctx),
            "world_context": len(world_ctx),
            "slv_context": len(slv_ctx),
            "slv_basin_context": len(basin_ctx),
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
            "conus_states_context is the archived Natural Earth 50m admin-1 layer, US units "
            "excluding Alaska and Hawaii, clipped to the same box; ACCEPTED 2026-08-25 (handoff "
            "section 6.7) and drawn as quiet muted-gray context in the E1 and E2 maps",
            "slv_context is a convex hull of the generalized E3 centroids buffered 0.15 degrees",
            "slv_basin_context is the archived five-unit HUC8 Rio Grande headwaters extract from "
            "the USGS WBD; REJECTED as artwork 2026-08-25 (handoff section 6.7 keeps HUC8 units "
            "out of the figure) and retained here as hashed provenance only, not drawn",
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
            "recorded provenance in fig01_example_selection.json. Re-audited 2026-08-24 against "
            "handoff sections 10.2 and 15.2: column values and the column set are unchanged, but "
            "irr_applied is now typed display-role e1_example_irrigation_stems_only -- it is E1 "
            "example evidence, never an E3 modeled series, and may not be aggregated in the "
            "artwork or connected to the separate 'Metered Water · E3' key."
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
            "metrics. The 2026-08-24 re-audit block records that no column value changed and "
            "adds per-column display roles and forbidden uses, so a downstream plotting script "
            "cannot silently promote the E1 example's irr_applied series into E3 evidence."
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
            "Architecture 3.2.0 (supersedes 3.1.1, which supersedes 3.1.0, 3.0.0 and 2.1.0): the "
            "two-panel evidence-first composition with panel (a) redesigned per the "
            "2026-08-25 handoff rewrite and panel (b) accepted as built in revision 3. Frozen "
            "reader-facing strings including the required near-black 'US-Bi1 (2017)' record "
            "identification, the FIVE panel (a) quantitative plotting regions with their source "
            "columns, per-region y-axis contract and recorded display domains -- root-zone "
            "depletion merged with the magnitude-bearing irrigation stems on two explicit "
            "scales, and daily ET merged with 'Flux ET (Held Out)' on one shared axis and "
            "vertical scale compared by shared alignment alone -- the closed directed inverse "
            "cycle 'Run Balance -> Compare -> Update Parameters -> Run Balance' with the ETf "
            "targets and '+ SWE' entering Compare, the NDVI and daily-forcing drivers entering "
            "Run Balance, and the labelled 'Conditioned Parameters' exit to the displayed daily "
            "trajectory; the separate typographic E3 evaluation key with its 'Annual Total' "
            "operation, the held-out boundary carried by the direct trace label rather than a "
            "heading, the three map nodes, the two class-specific parameter tokens with the "
            "visible irrigated branch junction, the allowed edges, the undirected comparisons, "
            "the forbidden-edge list extended so no held-out observation reaches any cycle "
            "stage, and a title/direct_label/annotation/proof_only classification of every "
            "visible string. "
            "Objective notation, retrieval members, weighting, the PEST++ IES engine name, "
            "realization and parameter counts, paired support, the ECOSTRESS sensitivity, the "
            "eight caption-contract items and the working caption live in a separate "
            "caption_facts block that the builder asserts never reaches visible copy."
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
            "Boundary dataset version and hash, the two archived optional context datasets with "
            "their URLs, hashes and public-domain licences, CRS provenance, legacy-to-current "
            "label map, E3 generalization method with its 2026-08-19 approval, cohort and "
            "irrigation-class assertions, the example-selection reference, font inventory, "
            "190 x 120 mm output dimensions, and the scientific / privacy / visual review record."
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


def _ne_states_version() -> str:
    p = NE_STATES.with_suffix(".VERSION.txt")
    if p.exists():
        return p.read_text().strip()
    return "unknown"


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

BUILDERS = {
    "fig01": lambda: build_fig01(),
    "fig02": lambda: build_fig02(),
    "fig03": lambda: build_fig03(),
    "fig04": lambda: build_fig04(),
    "fig05_e1": lambda: build_fig05_e1(),
    "fig05_e2": lambda: build_fig05_e2(),
    "fig06": lambda: build_fig06(),
    "fig06_bootstrap": lambda: build_fig06_bootstrap(),
    "obs_support": lambda: build_obs_support(),
}

# fig01 consumes the frozen fig03 example series (retained Figure 1 provenance,
# re-registered but never regenerated by build_fig03) and fig04 capture values,
# so a --all run must build it last.  Every other builder is independent.
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
        MANIFEST.partial_targets = targets
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
