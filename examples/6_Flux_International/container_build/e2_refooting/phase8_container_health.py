"""Phase 8: mandatory completeness validation of the corrected E2 container (Gate G8).

Checks, for every canonical site (plan §14 + CLAUDE.md "Container Validation"):
  * axes: daily time axis equals the config date range; ``geometry/uid`` equals the cohort
    shapefile (set and order); every time-series array is (n_days, n_fields);
  * ETf: >= 1 finite Landsat SSEBop and PT-JPL value per site, per-year capture counts;
  * NDVI: Landsat, Sentinel and fused series non-null with per-year/growing-season coverage;
  * meteorology: no all-NaN variable at any site (every array under ``meteorology/``);
  * no inherited calibration state (``calibration/`` group, resolved-state run, simulate events);
  * target identity: the container SSEBop array reconstructs exactly from the frozen grass-basis
    CSVs after the ingestor's own bounds (min_etf 0.05, max 2.0);
  * regression: PT-JPL, NDVI, meteorology and properties equal the baseline container for the
    common sites (only the SSEBop member and its derived dynamics may differ);
  * classifier audit: annual irrigation / groundwater-subsidy classes baseline -> corrected.

Outputs (QA root): container_health.json, etf_capture_counts.csv, ndvi_coverage.csv,
met_completeness.csv, irrigation_classifier_transition.csv. Exit 1 when any hard check fails.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re

import fiona
import numpy as np
import pandas as pd
import zarr

from swimrs.swim.config import ProjectConfig

HERE = os.path.dirname(os.path.abspath(__file__))
EX6 = os.path.dirname(os.path.dirname(HERE))
DATA = "/data/ssd1/swim/6_Flux_International/data"
QA_ROOT = os.path.join(DATA, "e2_etf_refooting")
CONFIG = os.path.join(EX6, "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr.toml")
BASELINE = os.path.join(DATA, "6_Flux_International_ls_ensemble_por_annual2yr.swim")
INGEST = {"min_etf": 0.05, "max_etf": 2.0}
ETF_PATHS = {
    "ssebop": "remote_sensing/etf/landsat/ssebop/no_mask",
    "ptjpl": "remote_sensing/etf/landsat/ptjpl/no_mask",
}
NDVI_PATHS = {
    "landsat": "remote_sensing/ndvi/landsat/no_mask",
    "sentinel": "remote_sensing/ndvi/sentinel/no_mask",
    "fused": "derived/merged_ndvi/no_mask",
}
REGRESSION_PATHS = [
    "remote_sensing/etf/landsat/ptjpl/no_mask",
    "remote_sensing/ndvi/landsat/no_mask",
    "remote_sensing/ndvi/sentinel/no_mask",
    "derived/merged_ndvi/no_mask",
    "properties/soils/awc",
    "properties/land_cover/glc10",
    "properties/land_cover/modis_lc",
]
GROWING_MONTHS = (4, 5, 6, 7, 8, 9)
GRASS_RE = re.compile(r"^ssebop_etf_grass_(?P<site>.+)_no_mask_(?P<year>\d{4})\.csv$")


# --------------------------------------------------------------------------- helpers
def shapefile_sids(path: str, id_col: str) -> list[str]:
    with fiona.open(path) as src:
        return [f["properties"][id_col] for f in src]


def open_root(path: str) -> zarr.Group:
    return zarr.open(path, mode="r")


def uids(root: zarr.Group) -> list[str]:
    return [str(u) for u in root["geometry/uid"][:]]


def dates(root: zarr.Group) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(root["time/daily"][:]))


def array_paths(group: zarr.Group, prefix: str = "") -> list[str]:
    out = []
    for key in group.keys():
        node = group[key]
        path = f"{prefix}{key}"
        if isinstance(node, zarr.Group):
            out.extend(array_paths(node, path + "/"))
        else:
            out.append(path)
    return out


def nan_equal(a: np.ndarray, b: np.ndarray, atol: float = 0.0) -> tuple[bool, float]:
    if a.shape != b.shape:
        return False, float("inf")
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        a = a.astype(float)
        b = b.astype(float)
        both = np.isfinite(a) & np.isfinite(b)
        same_mask = np.array_equal(np.isfinite(a), np.isfinite(b))
        diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        return same_mask and diff <= atol, diff
    return bool(np.array_equal(a, b)), 0.0


# --------------------------------------------------------------------------- checks
def check_axes(root: zarr.Group, cfg: ProjectConfig, sids: list[str]) -> dict:
    d = dates(root)
    expected = pd.date_range(cfg.start_dt, cfg.end_dt, freq="D")
    u = uids(root)
    ts_paths = [
        p
        for p in array_paths(root)
        if root[p].ndim == 2 and p.split("/")[0] in ("remote_sensing", "meteorology", "derived")
    ]
    bad_shape = [p for p in ts_paths if root[p].shape != (len(d), len(u))]
    return {
        "n_days": len(d),
        "n_fields": len(u),
        "time_axis_matches_config": bool(d.equals(expected)),
        "uid_set_matches_shapefile": set(u) == set(sids),
        "uid_order_matches_shapefile": u == sids,
        "missing_from_container": sorted(set(sids) - set(u)),
        "extra_in_container": sorted(set(u) - set(sids)),
        "n_timeseries_arrays": len(ts_paths),
        "arrays_with_wrong_shape": bad_shape,
        "source_shapefile_attr": root.attrs.get("source_shapefile"),
        "pass": bool(d.equals(expected)) and set(u) == set(sids) and not bad_shape,
    }


def check_no_calibration_state(root: zarr.Group) -> dict:
    ops = [e.get("operation") for e in root.attrs.get("provenance", {}).get("events", [])]
    runs = list(root["simulation/runs"].keys()) if "simulation/runs" in root else []
    problems = []
    if "calibration" in root:
        problems.append("calibration/ group present")
    if any(r.startswith("calibration") for r in runs):
        problems.append(f"simulation runs present: {runs}")
    if "simulate" in ops:
        problems.append("simulate events in provenance")
    return {
        "provenance_operations": ops,
        "simulation_runs": runs,
        "top_level_groups": sorted(root.keys()),
        "problems": problems,
        "pass": not problems,
    }


def etf_capture_counts(root: zarr.Group) -> tuple[pd.DataFrame, dict]:
    d = dates(root)
    u = uids(root)
    years = sorted(set(d.year))
    rows = []
    for model, path in ETF_PATHS.items():
        arr = np.asarray(root[path][:], float)
        for j, sid in enumerate(u):
            fin = np.isfinite(arr[:, j])
            per_year = {f"y{y}": int((fin & (d.year == y)).sum()) for y in years}
            rows.append(
                {
                    "site": sid,
                    "model": model,
                    "n_valid": int(fin.sum()),
                    "first": d[fin][0].strftime("%Y-%m-%d") if fin.any() else None,
                    "last": d[fin][-1].strftime("%Y-%m-%d") if fin.any() else None,
                    "min": float(np.nanmin(arr[:, j])) if fin.any() else None,
                    "max": float(np.nanmax(arr[:, j])) if fin.any() else None,
                    **per_year,
                }
            )
    table = pd.DataFrame(rows)
    zero = table[table["n_valid"] == 0]
    both = table.pivot(index="site", columns="model", values="n_valid")
    ss = np.isfinite(np.asarray(root[ETF_PATHS["ssebop"]][:], float))
    pj = np.isfinite(np.asarray(root[ETF_PATHS["ptjpl"]][:], float))
    summary = {
        "n_valid_by_model": table.groupby("model")["n_valid"].sum().to_dict(),
        "sites_with_zero_captures": zero[["site", "model"]].to_dict("records"),
        "min_site_captures_by_model": both.min().to_dict(),
        "paired_site_dates": int((ss & pj).sum()),
        "ptjpl_only_site_dates": int((pj & ~ss).sum()),
        "ssebop_only_site_dates": int((ss & ~pj).sum()),
        "pass": zero.empty,
    }
    return table, summary


def ndvi_coverage(root: zarr.Group) -> tuple[pd.DataFrame, dict]:
    d = dates(root)
    u = uids(root)
    years = sorted(set(d.year))
    rows = []
    for name, path in NDVI_PATHS.items():
        if path not in root:
            continue
        arr = np.asarray(root[path][:], float)
        for j, sid in enumerate(u):
            fin = np.isfinite(arr[:, j])
            per_year = np.array([(fin & (d.year == y)).sum() for y in years])
            gs = fin & np.isin(d.month, GROWING_MONTHS)
            gs_cells = {(y, m) for y, m in zip(d.year[gs], d.month[gs], strict=True)}
            rows.append(
                {
                    "site": sid,
                    "instrument": name,
                    "n_valid": int(fin.sum()),
                    "first": d[fin][0].strftime("%Y-%m-%d") if fin.any() else None,
                    "last": d[fin][-1].strftime("%Y-%m-%d") if fin.any() else None,
                    "years_with_zero_obs": int((per_year == 0).sum()),
                    "min_obs_per_year": int(per_year.min()),
                    "growing_season_month_coverage": len(gs_cells)
                    / (len(years) * len(GROWING_MONTHS)),
                }
            )
    table = pd.DataFrame(rows)
    fused = table[table["instrument"] == "fused"]
    problems = fused[(fused["n_valid"] == 0) | (fused["years_with_zero_obs"] > 0)]
    summary = {
        "fused_min_site_valid": int(fused["n_valid"].min()),
        "fused_min_growing_season_month_coverage": float(
            fused["growing_season_month_coverage"].min()
        ),
        "fused_sites_with_a_zero_year": problems["site"].tolist(),
        "landsat_sites_zero": table[(table["instrument"] == "landsat") & (table["n_valid"] == 0)][
            "site"
        ].tolist(),
        "sentinel_sites_zero": table[(table["instrument"] == "sentinel") & (table["n_valid"] == 0)][
            "site"
        ].tolist(),
        "pass": problems.empty,
    }
    return table, summary


def met_completeness(root: zarr.Group) -> tuple[pd.DataFrame, dict]:
    u = uids(root)
    rows = []
    paths = [p for p in array_paths(root["meteorology"], "meteorology/")]
    for path in paths:
        arr = np.asarray(root[path][:], float)
        for j, sid in enumerate(u):
            fin = np.isfinite(arr[:, j])
            rows.append(
                {
                    "site": sid,
                    "variable": path,
                    "n_valid": int(fin.sum()),
                    "n_nan": int((~fin).sum()),
                    "all_nan": bool(not fin.any()),
                    "min": float(np.nanmin(arr[:, j])) if fin.any() else None,
                    "max": float(np.nanmax(arr[:, j])) if fin.any() else None,
                }
            )
    table = pd.DataFrame(rows)
    bad = table[table["all_nan"]]
    summary = {
        "variables": paths,
        "all_nan_site_variables": bad[["site", "variable"]].to_dict("records"),
        "site_variables_with_any_nan": int((table["n_nan"] > 0).sum()),
        "max_nan_fraction": float((table["n_nan"] / (table["n_nan"] + table["n_valid"])).max()),
        "pass": bad.empty,
    }
    return table, summary


def expected_ssebop_from_grass(root: zarr.Group, grass_dir: str) -> np.ndarray:
    """Replay the ingestor on the frozen grass CSVs: date-keyed columns, bounds, float32."""
    d = dates(root)
    u = uids(root)
    pos = {sid: j for j, sid in enumerate(u)}
    tpos = {ts: i for i, ts in enumerate(d)}
    out = np.full((len(d), len(u)), np.nan, dtype=np.float64)
    for path in sorted(glob.glob(os.path.join(grass_dir, "ssebop_etf_grass_*_no_mask_*.csv"))):
        m = GRASS_RE.match(os.path.basename(path))
        if not m or m.group("site") not in pos:
            continue
        wide = pd.read_csv(path)
        if len(wide) != 1:
            raise ValueError(f"{path}: expected one row")
        j = pos[m.group("site")]
        for col in wide.columns[1:]:
            ts = pd.Timestamp(col.rsplit("_", 1)[1])
            if ts not in tpos:
                continue
            val = float(wide.iloc[0][col])
            if not (INGEST["min_etf"] <= val <= INGEST["max_etf"]):
                continue
            if np.isfinite(out[tpos[ts], j]):
                raise ValueError(f"{m.group('site')} {ts.date()}: two grass values for one date")
            out[tpos[ts], j] = val
    return out.astype(np.float32)


def check_target_identity(root: zarr.Group, grass_dir: str) -> dict:
    expected = expected_ssebop_from_grass(root, grass_dir)
    got = np.asarray(root[ETF_PATHS["ssebop"]][:], np.float32)
    same, diff = nan_equal(expected, got)
    return {
        "grass_dir": grass_dir,
        "expected_valid": int(np.isfinite(expected).sum()),
        "container_valid": int(np.isfinite(got).sum()),
        "valid_mask_identical": bool(np.array_equal(np.isfinite(expected), np.isfinite(got))),
        "max_abs_diff": diff,
        "pass": same,
    }


def check_regression(root: zarr.Group, base: zarr.Group) -> dict:
    u = uids(root)
    ub = uids(base)
    common = [s for s in u if s in set(ub)]
    ji = [u.index(s) for s in common]
    jb = [ub.index(s) for s in common]
    same_dates = dates(root).equals(dates(base))
    out = {"n_common_sites": len(common), "same_time_axis": same_dates, "arrays": {}}
    for path in REGRESSION_PATHS:
        if path not in root or path not in base:
            out["arrays"][path] = {"present_in_both": False}
            continue
        a = np.asarray(root[path][:])
        b = np.asarray(base[path][:])
        a = a[:, ji] if a.ndim == 2 else a[ji]
        b = b[:, jb] if b.ndim == 2 else b[jb]
        same, diff = nan_equal(a, b)
        out["arrays"][path] = {"present_in_both": True, "identical": same, "max_abs_diff": diff}
        if path == "properties/soils/awc" and not same:
            # The baseline container stored HWSD AWC as delivered (mm/m); the container
            # convention is m/m (HANDOFF_HWSD_AWC_UNITS_RECAL 2026-09-21). The refreshed
            # container must reproduce the baseline exactly after the x1000 conversion.
            same_units, diff_units = nan_equal(a.astype(float) * 1000.0, b.astype(float), atol=1e-3)
            out["arrays"][path].update(
                {
                    "identical_after_mm_per_m_conversion": same_units,
                    "max_abs_diff_after_conversion": diff_units,
                    "awc_units_stored": root["properties/soils"].attrs.get("awc_units_stored"),
                    "awc_units_source": root["properties/soils"].attrs.get("awc_units_source"),
                }
            )
            if same_units and root["properties/soils"].attrs.get("awc_units_stored") == "m/m":
                out["arrays"][path]["identical"] = True
                out["arrays"][path]["note"] = REGRESSION_AWC_NOTE
    for path in array_paths(root["meteorology"], "meteorology/"):
        if path in base:
            a = np.asarray(root[path][:])[:, ji]
            b = np.asarray(base[path][:])[:, jb]
            same, diff = nan_equal(a, b)
            out["arrays"][path] = {"present_in_both": True, "identical": same, "max_abs_diff": diff}
    ssebop_same, ssebop_diff = nan_equal(
        np.asarray(root[ETF_PATHS["ssebop"]][:])[:, ji],
        np.asarray(base[ETF_PATHS["ssebop"]][:])[:, jb],
    )
    out["ssebop_identical_to_baseline"] = ssebop_same  # expected False: the corrected member
    out["ssebop_max_abs_diff_vs_baseline"] = ssebop_diff
    out["explained_differences"] = REGRESSION_EXPLAINED
    out["pass"] = same_dates and all(
        v.get("identical", False)
        for p, v in out["arrays"].items()
        if v["present_in_both"] and p not in REGRESSION_EXPLAINED
    )
    return out


# Arrays allowed to differ from the baseline, each with the tracked cause. Any difference in an
# array not listed here fails the regression check; the Sentinel entry must additionally be
# reproduced by ``replay_sentinel_ndvi`` (both rules) before the check passes.
REGRESSION_AWC_NOTE = (
    "baseline stores HWSD AWC in mm/m as delivered; the refreshed container stores m/m "
    "(awc_units='mm/m' declared at ingest, 2026-09-21 HWSD AWC units recal); accepted only when "
    "new*1000 reproduces the baseline within 1e-3 mm/m"
)
REGRESSION_EXPLAINED = {
    "remote_sensing/ndvi/sentinel/no_mask": (
        "ingestor same-date Sentinel tile collapse changed max -> mean in c16263c (2026-08-13), "
        "after the baseline build (2026-06-12); verified by replaying the source directory "
        "with both rules"
    ),
    "derived/merged_ndvi/no_mask": "fused from the Sentinel series above (Landsat NDVI identical)",
}


def _parse_sentinel_csv_max(csv_file, known: set[str]) -> list[pd.Series]:
    """The pre-c16263c parse of one Sentinel CSV: same-date tiles collapsed by ``max``."""
    df = pd.read_csv(csv_file)
    if "sid" not in df.columns:
        first = df.columns[0]
        if first not in known:
            return []
        df.columns = ["sid"] + list(df.columns[1:])
        df["sid"] = df["sid"].astype(object)
        df.iloc[0, 0] = first
    cols = [c for c in df.columns if c != "sid" and len(c[:8]) == 8 and c[:8].isdigit()]
    out = []
    for _, row in df.iterrows():
        fid = str(row["sid"])
        if fid not in known:
            continue
        s = pd.Series(
            row[cols].values, index=pd.to_datetime([c[:8] for c in cols]), name=fid
        ).astype("float64")
        s = s.sort_index()
        if s.index.duplicated().any():
            s = s.groupby(s.index).max()
        out.append(s)
    return out


def replay_sentinel_ndvi(
    source_dir: str, d: pd.DatetimeIndex, u: list[str], rule: str, min_ndvi: float = 0.05
) -> np.ndarray:
    """Re-run the Sentinel NDVI ingest from the CSV directory in the ingestor's own file order.

    ``rule="mean"`` uses the live ``_parse_single_csv``; ``rule="max"`` uses the pre-c16263c
    collapse. Multiple CSVs per field merge by ``combine_first`` in glob order, then the
    ingestor's ``min_ndvi`` and consecutive-day filters apply, as in ``Ingestor.ndvi``.
    """
    from collections import defaultdict
    from pathlib import Path

    from swimrs.container.components.ingestor import Ingestor, _parse_single_csv

    known = set(u)
    per_field = defaultdict(list)
    for f in list(Path(source_dir).glob("*.csv")):
        series = (
            _parse_single_csv(f, "sid", "sentinel", known, None)
            if rule == "mean"
            else _parse_sentinel_csv_max(f, known)
        )
        for s in series:
            per_field[s.name].append(s)
    combined = []
    for fid, sl in per_field.items():
        c = sl[0]
        for s in sl[1:]:
            c = c.combine_first(s)
        c.name = fid
        combined.append(c)
    df = pd.concat(combined, axis=1).sort_index()
    df = Ingestor._apply_ndvi_filters(None, df, min_ndvi, True)
    return df.reindex(index=d, columns=u).to_numpy(dtype=np.float32)


def explain_sentinel_difference(root: zarr.Group, base: zarr.Group, source_dir: str) -> dict:
    """Attribute the Sentinel NDVI baseline difference to the collapse-rule change, exactly."""
    d = dates(root)
    u = uids(root)
    ub = uids(base)
    jb = [ub.index(s) for s in u]
    new = np.asarray(root[NDVI_PATHS["sentinel"]][:], np.float32)
    old = np.asarray(base[NDVI_PATHS["sentinel"]][:], np.float32)[:, jb]
    mean_ok, mean_diff = nan_equal(replay_sentinel_ndvi(source_dir, d, u, "mean"), new)
    max_ok, max_diff = nan_equal(replay_sentinel_ndvi(source_dir, d, u, "max"), old)
    both = np.isfinite(new) & np.isfinite(old)
    families = {
        "no_mask": len(glob.glob(os.path.join(source_dir, "ndvi_*_no_mask_*.csv")))
        - len(glob.glob(os.path.join(source_dir, "ndvi_*_sentinel_no_mask_*.csv"))),
        "sentinel_no_mask": len(
            glob.glob(os.path.join(source_dir, "ndvi_*_sentinel_no_mask_*.csv"))
        ),
    }
    return {
        "source_dir": source_dir,
        "csv_families": families,
        "cells_differing": int((both & (new != old)).sum()),
        "cells_valid_new_only": int((np.isfinite(new) & ~np.isfinite(old)).sum()),
        "cells_valid_baseline_only": int((~np.isfinite(new) & np.isfinite(old)).sum()),
        "mean_replay_reproduces_new": mean_ok,
        "mean_replay_max_abs_diff": mean_diff,
        "max_replay_reproduces_baseline": max_ok,
        "max_replay_max_abs_diff": max_diff,
        "pass": mean_ok and max_ok,
    }


def classifier_transition(root: zarr.Group, base: zarr.Group) -> tuple[pd.DataFrame, dict]:
    def parse(group: zarr.Group, path: str) -> dict[str, dict]:
        out = {}
        for sid, raw in zip(uids(group), group[path][:], strict=True):
            raw = raw.item() if hasattr(raw, "item") else raw
            out[sid] = json.loads(raw) if raw else {}
        return out

    irr_new, irr_old = (
        parse(root, "derived/dynamics/irr_data"),
        parse(base, "derived/dynamics/irr_data"),
    )
    gw_new, gw_old = (
        parse(root, "derived/dynamics/gwsub_data"),
        parse(base, "derived/dynamics/gwsub_data"),
    )
    rows = []
    fallow = {}
    for sid in uids(root):
        if sid not in irr_old:
            continue
        # the classifier also stores a per-site ``fallow_years`` list beside the year records
        fallow[sid] = {
            "baseline": sorted(irr_old[sid].get("fallow_years", [])),
            "corrected": sorted(irr_new[sid].get("fallow_years", [])),
        }
        years = sorted(k for k in set(irr_new[sid]) | set(irr_old[sid]) if str(k).isdigit())
        for y in years:
            n, o = irr_new[sid].get(y, {}), irr_old[sid].get(y, {})
            gn, go = gw_new.get(sid, {}).get(y, {}), gw_old.get(sid, {}).get(y, {})
            rows.append(
                {
                    "site": sid,
                    "year": int(y),
                    "fallow_baseline": int(y) in fallow[sid]["baseline"],
                    "fallow_corrected": int(y) in fallow[sid]["corrected"],
                    "irrigated_baseline": o.get("irrigated"),
                    "irrigated_corrected": n.get("irrigated"),
                    "f_irr_baseline": o.get("f_irr"),
                    "f_irr_corrected": n.get("f_irr"),
                    "n_irr_doys_baseline": len(o.get("irr_doys", [])),
                    "n_irr_doys_corrected": len(n.get("irr_doys", [])),
                    "subsidized_baseline": go.get("subsidized"),
                    "subsidized_corrected": gn.get("subsidized"),
                    "gw_ratio_baseline": go.get("ratio"),
                    "gw_ratio_corrected": gn.get("ratio"),
                }
            )
    t = pd.DataFrame(rows)
    t["irr_transition"] = (
        t["irrigated_baseline"].astype("Int64").astype(str)
        + "->"
        + t["irrigated_corrected"].astype("Int64").astype(str)
    )
    ever = t.groupby("site")[["irrigated_baseline", "irrigated_corrected"]].max()
    changed_sites = ever[ever["irrigated_baseline"] != ever["irrigated_corrected"]]
    summary = {
        "n_site_years": int(len(t)),
        "irrigated_site_years_baseline": int(t["irrigated_baseline"].fillna(0).sum()),
        "irrigated_site_years_corrected": int(t["irrigated_corrected"].fillna(0).sum()),
        "irr_transition_counts": t["irr_transition"].value_counts().to_dict(),
        "ever_irrigated_sites_baseline": int(ever["irrigated_baseline"].sum()),
        "ever_irrigated_sites_corrected": int(ever["irrigated_corrected"].sum()),
        "sites_changing_ever_irrigated": changed_sites.reset_index().to_dict("records"),
        "sites_with_any_year_change": sorted(
            t.loc[t["irrigated_baseline"] != t["irrigated_corrected"], "site"].unique()
        ),
        "subsidized_site_years_baseline": int(t["subsidized_baseline"].fillna(0).sum()),
        "subsidized_site_years_corrected": int(t["subsidized_corrected"].fillna(0).sum()),
        "fallow_site_years_baseline": int(t["fallow_baseline"].sum()),
        "fallow_site_years_corrected": int(t["fallow_corrected"].sum()),
        "sites_with_fallow_change": sorted(
            t.loc[t["fallow_baseline"] != t["fallow_corrected"], "site"].unique()
        ),
    }
    return t, summary


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--container", default=None, help="default: [paths] container of --config")
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--grass-dir", default=None, help="default: [paths.etf_sources] ssebop")
    ap.add_argument("--out-dir", default=QA_ROOT)
    args = ap.parse_args(argv)

    cfg = ProjectConfig()
    cfg.read_config(args.config)
    container = args.container or cfg.container_path
    grass_dir = args.grass_dir or cfg.etf_source_dirs["ssebop"]
    sids = shapefile_sids(cfg.fields_shapefile, cfg.feature_id_col)
    root = open_root(container)
    base = open_root(args.baseline)

    axes = check_axes(root, cfg, sids)
    calib = check_no_calibration_state(root)
    etf_table, etf = etf_capture_counts(root)
    ndvi_table, ndvi = ndvi_coverage(root)
    met_table, met = met_completeness(root)
    target = check_target_identity(root, grass_dir)
    regression = check_regression(root, base)
    sentinel_dir = os.path.join(cfg.sentinel_dir, "extracts", "ndvi", "no_mask")
    sentinel = explain_sentinel_difference(root, base, sentinel_dir)
    trans_table, transition = classifier_transition(root, base)

    checks = {
        "axes": axes,
        "no_calibration_state": calib,
        "etf_captures": etf,
        "ndvi": ndvi,
        "meteorology": met,
        "target_identity": target,
        "regression_vs_baseline": regression,
        "sentinel_difference_attribution": sentinel,
    }
    report = {
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": args.config,
        "container": container,
        "baseline": args.baseline,
        "shapefile": cfg.fields_shapefile,
        "container_attrs": {
            k: v for k, v in root.attrs.asdict().items() if k not in ("provenance",)
        },
        "ingest_rules": INGEST,
        "checks": checks,
        "classifier_transition": transition,
        "pass": all(c["pass"] for c in checks.values()),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    etf_table.to_csv(os.path.join(args.out_dir, "etf_capture_counts.csv"), index=False)
    ndvi_table.to_csv(os.path.join(args.out_dir, "ndvi_coverage.csv"), index=False)
    met_table.to_csv(os.path.join(args.out_dir, "met_completeness.csv"), index=False)
    trans_table.to_csv(
        os.path.join(args.out_dir, "irrigation_classifier_transition.csv"), index=False
    )
    with open(os.path.join(args.out_dir, "container_health.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
