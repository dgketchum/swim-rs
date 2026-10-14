"""Two-arm irrigation discrimination test for Example 10.

Runs the Example 10 fields forward under two prescribed irrigation regimes with
the frozen Run 23 irrigation-stratified class vectors (never fit to these
fields) and asks which arm the thermal ETf tracks, field-year by field-year.

    Arm I  every year irrigated (NDVI-derived windows), irrigated class vector
    Arm R  no year irrigated (scheduler off),           rainfed class vector

plus the two off-diagonal cells (Arm I container with the rainfed vector, Arm R
container with the irrigated vector) so the I-vs-R separation can be attributed
to the scheduler or to the vectors. No calibration, no IrrMapper input, no
Earth Engine except the optional ``irrmapper`` comparator pull.

Scoring is at the ETf level on Landsat capture dates: the arm's simulated ETf
(eta / ETo) against the four-member ensemble mean, with the across-member
spread (+0.1 floor, as the E1 calibration weights it) as the noise scale. The
daily ET comparison against the interpolated ensemble ETf x corrected ETo is
kept as a secondary metric.

Design: notes/two_arm_test_plan.md.

    uv run python examples/10_OR_Irrigation_Test/two_arm_test.py build-arms [--run e10cal] [--overwrite]
    uv run python examples/10_OR_Irrigation_Test/two_arm_test.py run [--run e10cal] [--cells I,R,...]
    uv run python examples/10_OR_Irrigation_Test/two_arm_test.py score [--run e10cal]
    uv run python examples/10_OR_Irrigation_Test/two_arm_test.py report [--run e10cal]
    uv run python examples/10_OR_Irrigation_Test/two_arm_test.py irrmapper      # Earth Engine; comparator only
    uv run python examples/10_OR_Irrigation_Test/two_arm_test.py all [--run e10cal] [--overwrite]
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from ex10_paths import HERE, LST_MODELS, load_config

from swimrs.container import open_container
from swimrs.process.input import build_swim_input
from swimrs.process.loop_fast import run_daily_loop_fast

REPO = HERE.parents[1]
VECTORS_PATH = REPO / "paper" / "data" / "final" / "e2_run23_transfer_vectors_by_irrigation.json"

# cell -> (arm container, class vector)
CELLS = {
    "I": ("I", "irrigated"),
    "R": ("R", "rainfed"),
    "I_rainfedvec": ("I", "rainfed"),
    "R_irrvec": ("R", "irrigated"),
}
PRIMARY = ("I", "R")
WINDOWS = {"jun_sep": (6, 9), "apr_oct": (4, 10), "full": (1, 12)}
PRIMARY_WINDOW = "jun_sep"
SPREAD_FLOOR = 0.1  # etf_weighting_spread_floor default in PestBuilder
BETWEEN_GAP = 0.1  # |bias| both arms must exceed for the between-arms flag
ETF_FIRST_YEAR = 1999  # first LST member year in this container
IRRMAPPER_YEARS = (1995, 2025)

# Classifier window-detection defaults (compute.dynamics signature).
WINDOW_KW = dict(lookback=10, ndvi_threshold=0.3, min_pos_days=10, ndvi_min_start=0.25)

ARM_DOC = {
    "I": "every year irrigated=1, f_irr=1.0, irr_doys from the classifier NDVI window "
    "detector (nearest-year backfill where none found); fallow_years=[]",
    "R": "every year irrigated=0, f_irr=0.0, irr_doys=[]; fallow_years=all years",
}


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def run_container(cfg, run):
    return os.path.join(cfg.data_dir, f"{cfg.project_name}_{run}.swim")


def arm_container(cfg, run, arm):
    return os.path.join(cfg.data_dir, f"{cfg.project_name}_{run}_arm{arm}.swim")


def results_dir(cfg, out=None):
    return out or os.path.join(cfg.project_ws, "results", "two_arm")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return None


def load_vectors():
    with open(VECTORS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# build-arms
# ---------------------------------------------------------------------------
def build_arms(cfg, run, overwrite=False, out=None):
    src = run_container(cfg, run)
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    c = open_container(src, mode="r")
    fids = list(c.field_uids)
    dates = pd.DatetimeIndex(np.array(c._root["time/daily"]))
    years = sorted(set(dates.year.tolist()))
    ndvi = c.query.dataframe("derived/merged_ndvi/no_mask", fields=fids)
    calc = c.compute

    audit = []
    arm_records = {"I": [], "R": []}
    for fid in fids:
        s = ndvi[fid]
        site_I, needs_backfill = {}, []
        for yr in years:
            ext = calc._get_extended_year_ndvi_fast(s, yr, years)
            doys = calc._detect_irrigation_windows(
                ext,
                WINDOW_KW["lookback"],
                WINDOW_KW["ndvi_threshold"],
                WINDOW_KW["min_pos_days"],
                yr,
                ndvi_min_start=WINDOW_KW["ndvi_min_start"],
            )
            if not doys:
                needs_backfill.append(yr)
            site_I[str(yr)] = {"irr_doys": doys, "irrigated": 1, "f_irr": 1.0}
        candidates = [int(y) for y, v in site_I.items() if v["irr_doys"]]
        for yr in needs_backfill:
            if candidates:
                nearest = min(candidates, key=lambda x: abs(x - yr))
                site_I[str(yr)]["irr_doys"] = list(site_I[str(nearest)]["irr_doys"])
        site_I["fallow_years"] = []
        arm_records["I"].append(json.dumps(site_I))

        site_R = {str(yr): {"irr_doys": [], "irrigated": 0, "f_irr": 0.0} for yr in years}
        site_R["fallow_years"] = list(years)
        arm_records["R"].append(json.dumps(site_R))

        n_days = [len(site_I[str(y)]["irr_doys"]) for y in years]
        audit.append(
            {
                "field": fid,
                "window_days_min": int(min(n_days)),
                "window_days_median": float(np.median(n_days)),
                "window_days_max": int(max(n_days)),
                "years_backfilled": len(needs_backfill),
                "backfilled": ",".join(str(y) for y in needs_backfill),
            }
        )
    c.close()

    for arm in ("I", "R"):
        dest = arm_container(cfg, run, arm)
        if os.path.exists(dest):
            if not overwrite:
                raise FileExistsError(f"{dest} exists; pass --overwrite")
            shutil.rmtree(dest)
            lock = dest + ".lock"
            if os.path.exists(lock):
                os.remove(lock)
        shutil.copytree(src, dest)
        ca = open_container(dest, mode="r+")
        if "calibration" in ca._root:
            raise RuntimeError(f"{src} carries a calibration group; refusing to seed arms from it")
        arr = ca._root["derived/dynamics/irr_data"]
        arr[:] = np.array(arm_records[arm], dtype=arr.dtype)
        ca.close()
        print(f"Arm {arm}: {dest}\n    {ARM_DOC[arm]}")

    audit_df = pd.DataFrame(audit)
    out = results_dir(cfg, out)
    os.makedirs(out, exist_ok=True)
    audit_df.to_csv(os.path.join(out, "arm_I_window_audit.csv"), index=False)
    print("\nArm I NDVI irrigation windows (days per year, 1995-2025):")
    print(audit_df.to_string(index=False))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def forward_run(cfg, container_path, vector, fids=None):
    """Forward run with one fixed parameter vector on every field; {fid: DataFrame}."""
    c = open_container(container_path, mode="r")
    fids = fids or list(c.field_uids)
    params_by_fid = {fid: dict(vector) for fid in fids}
    tmp_h5 = tempfile.NamedTemporaryFile(suffix=".h5", delete=False).name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(params_by_fid, tmp)
        params_json = tmp.name
    try:
        swim_input = build_swim_input(
            c,
            output_h5=tmp_h5,
            calibrated_params_path=params_json,
            start_date=cfg.start_dt,
            end_date=cfg.end_dt,
            refet_type=cfg.refet_type or "eto",
            etf_model=cfg.etf_target_model,
            met_source=cfg.met_source,
            fields=fids,
            empirical_kc_max=True,
            mask_mode=cfg.mask_mode,
            max_irr_rate=cfg.max_irr_rate or 100.0,
        )
        output, _ = run_daily_loop_fast(swim_input)
        dates = pd.date_range(swim_input.start_date, periods=swim_input.n_days, freq="D")
        eto = swim_input.get_time_series("eto")
        prcp = swim_input.get_time_series("prcp")
        results = {}
        for i, fid in enumerate(swim_input.fids):
            results[fid] = pd.DataFrame(
                {
                    "et_act": output.eta[:, i],
                    "etf_model": output.etf[:, i],
                    "etref": eto[:, i],
                    "prcp": prcp[:, i],
                    "kcb": output.kcb[:, i],
                    "ke": output.ke[:, i],
                    "ks": output.ks[:, i],
                    "kr": output.kr[:, i],
                    "irr_sim": output.irr_sim[:, i],
                    "depl_root": output.depl_root[:, i],
                    "swe": output.swe[:, i],
                    "runoff": output.runoff[:, i],
                    "dperc": output.dperc[:, i],
                },
                index=dates,
            )
        swim_input.close()
    finally:
        for p in (tmp_h5, params_json):
            if os.path.exists(p):
                os.remove(p)
        c.close()
    return results


def run_cells(cfg, run, cells, out=None):
    vectors = load_vectors()
    out = results_dir(cfg, out)
    meta = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "run_container": run_container(cfg, run),
        "arm_containers": {a: arm_container(cfg, run, a) for a in ("I", "R")},
        "arm_definitions": ARM_DOC,
        "vectors_path": str(VECTORS_PATH),
        "vectors_sha256": _sha256(VECTORS_PATH),
        "vectors": vectors,
        "cells": {k: {"arm": v[0], "vector": v[1]} for k, v in cells.items()},
        "forward_run": {
            "start": str(cfg.start_dt.date()),
            "end": str(cfg.end_dt.date()),
            "refet_type": cfg.refet_type,
            "etf_model": cfg.etf_target_model,
            "met_source": cfg.met_source,
            "mask_mode": cfg.mask_mode,
            "empirical_kc_max": True,
            "max_irr_rate": cfg.max_irr_rate or 100.0,
            "note": "eto consumed is meteorology/gridmet/eto_corr when present (build_swim_input)",
        },
        "window_detection": WINDOW_KW,
        "scoring": {
            "windows": WINDOWS,
            "primary_window": PRIMARY_WINDOW,
            "spread_floor": SPREAD_FLOOR,
            "etf_first_year": ETF_FIRST_YEAR,
            "members": list(LST_MODELS),
        },
    }
    for cell, (arm, vec) in cells.items():
        path = arm_container(cfg, run, arm)
        print(f"\n=== cell {cell}: arm {arm} container, {vec} vector ===")
        res = forward_run(cfg, path, vectors[vec])
        cdir = os.path.join(out, "arms", cell)
        os.makedirs(cdir, exist_ok=True)
        for fid, df in res.items():
            df.to_csv(os.path.join(cdir, f"{fid}.csv"), float_format="%.5f")
        irr = {fid: float(df["irr_sim"].sum()) for fid, df in res.items()}
        et = {fid: float(df["et_act"].mean() * 365.25) for fid, df in res.items()}
        print(
            f"  {len(res)} fields; mean annual ET {np.mean(list(et.values())):.0f} mm; "
            f"mean total irr_sim {np.mean(list(irr.values())):.0f} mm over the run"
        )
    with open(os.path.join(out, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nrun_metadata.json -> {out}")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def _load_arm(out, cell, fid):
    return pd.read_csv(os.path.join(out, "arms", cell, f"{fid}.csv"), index_col=0, parse_dates=True)


def _member_frames(c, fids):
    members = {
        m: c.query.dataframe(f"remote_sensing/etf/landsat/{m}/no_mask", fields=fids)
        for m in LST_MODELS
    }
    ens = c.query.dataframe("remote_sensing/etf/landsat/ensemble/no_mask", fields=fids)
    stack = np.stack([members[m].values for m in LST_MODELS])  # (m, t, f)
    n_members = np.isfinite(stack).sum(axis=0)
    std = pd.DataFrame(np.nanstd(stack, axis=0), index=ens.index, columns=ens.columns)
    std = std.where(n_members >= 2)  # std undefined with one member
    n_members = pd.DataFrame(n_members, index=ens.index, columns=ens.columns)
    return members, ens, std, n_members


def _window_mask(idx, window):
    lo, hi = WINDOWS[window]
    return (idx.month >= lo) & (idx.month <= hi)


def score(cfg, run, out=None):
    out = results_dir(cfg, out)
    with open(os.path.join(out, "run_metadata.json")) as f:
        meta = json.load(f)
    cells = list(meta["cells"])
    c = open_container(run_container(cfg, run), mode="r")
    fids = list(c.field_uids)
    members, ens, std, n_members = _member_frames(c, fids)
    eto_corr = c.query.dataframe("meteorology/gridmet/eto_corr", fields=fids)
    prcp = c.query.dataframe("meteorology/gridmet/prcp", fields=fids)
    c.close()

    rows, daily_rows, annual_rows = [], [], []
    for fid in fids:
        arms = {cell: _load_arm(out, cell, fid) for cell in cells}
        obs_ens = ens[fid]
        sigma = std[fid].fillna(0.0) + SPREAD_FLOOR
        cap = obs_ens.notna() & (obs_ens.index.year >= ETF_FIRST_YEAR)
        # daily ET from the interpolated ensemble ETf (secondary metric)
        etf_daily = obs_ens.interpolate(limit_area="inside")
        et_daily = etf_daily * eto_corr[fid]
        for yr in sorted(set(obs_ens.index.year)):
            if yr < ETF_FIRST_YEAR:
                continue
            yr_mask = obs_ens.index.year == yr
            for window in WINDOWS:
                wmask = yr_mask & _window_mask(obs_ens.index, window)
                dates = obs_ens.index[wmask & cap.values]
                if len(dates) == 0:
                    continue
                for cell in cells:
                    sim = arms[cell]["etf_model"].reindex(dates)
                    for target in ("ensemble", *LST_MODELS):
                        obs = (
                            obs_ens.reindex(dates)
                            if target == "ensemble"
                            else members[target][fid].reindex(dates)
                        )
                        ok = obs.notna() & sim.notna()
                        if ok.sum() == 0:
                            continue
                        r = (sim[ok] - obs[ok]).values
                        rows.append(
                            {
                                "field": fid,
                                "year": yr,
                                "window": window,
                                "cell": cell,
                                "target": target,
                                "n": int(ok.sum()),
                                "rmse": float(np.sqrt(np.mean(r**2))),
                                "bias": float(np.mean(r)),
                                "wsse": float(np.sum((r / sigma.reindex(dates)[ok].values) ** 2))
                                if target == "ensemble"
                                else np.nan,
                                "obs_mean": float(obs[ok].mean()),
                                "sim_mean": float(sim[ok].mean()),
                            }
                        )
                # secondary: daily ET against interpolated ensemble x eto_corr
                dmask = wmask & et_daily.notna().values
                if dmask.sum() > 0:
                    for cell in cells:
                        sim_et = arms[cell]["et_act"].values[dmask]
                        obs_et = et_daily.values[dmask]
                        r = sim_et - obs_et
                        daily_rows.append(
                            {
                                "field": fid,
                                "year": yr,
                                "window": window,
                                "cell": cell,
                                "n_days": int(dmask.sum()),
                                "rmse_mm": float(np.sqrt(np.mean(r**2))),
                                "bias_mm": float(np.mean(r)),
                                "obs_et_mm": float(obs_et.sum()),
                                "sim_et_mm": float(sim_et.sum()),
                            }
                        )
            # annual water balance and applied water
            p = float(prcp[fid].values[yr_mask].sum())
            et_rs = float(np.nansum(et_daily.values[yr_mask]))
            row = {
                "field": fid,
                "year": yr,
                "prcp_mm": p,
                "et_rs_mm": et_rs,
                "et_over_p": et_rs / p if p > 0 else np.nan,
                "n_captures": int((cap.values & yr_mask).sum()),
            }
            for cell in cells:
                a = arms[cell]
                row[f"irr_sim_mm_{cell}"] = float(a["irr_sim"].values[yr_mask].sum())
                row[f"et_mm_{cell}"] = float(a["et_act"].values[yr_mask].sum())
            annual_rows.append(row)

    pd.DataFrame(rows).to_csv(os.path.join(out, "capture_scores.csv"), index=False)
    pd.DataFrame(daily_rows).to_csv(os.path.join(out, "daily_scores.csv"), index=False)
    pd.DataFrame(annual_rows).to_csv(os.path.join(out, "annual_balance.csv"), index=False)
    print(f"capture_scores.csv ({len(rows)} rows), daily_scores.csv, annual_balance.csv -> {out}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _attributes(cfg, run):
    import geopandas as gpd

    gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")
    keep = [
        c for c in ("OPENET_ID", "Acres", "OWRD", "ITYPE", "srctype", "IRR_EFF") if c in gdf.columns
    ]
    attrs = gdf[keep].rename(columns={"OPENET_ID": "field"}).set_index("field")
    c = open_container(run_container(cfg, run), mode="r")
    fids = list(c.field_uids)
    lc = {
        k: np.array(c._root[f"properties/land_cover/{k}"])
        for k in ("cdl_cultivated", "glc10", "modis_lc")
    }
    irr = [json.loads(s) for s in np.array(c._root["derived/dynamics/irr_data"])]
    c.close()
    lc_df = pd.DataFrame(lc, index=fids)
    attrs = attrs.join(lc_df, how="right")
    clf = {}
    for fid, d in zip(fids, irr):
        for y, v in d.items():
            if y != "fallow_years" and isinstance(v, dict):
                clf[(fid, int(y))] = float(v.get("f_irr", 0.0))
    clf = pd.Series(clf, name="classifier_f_irr")
    clf.index.names = ["field", "year"]
    return attrs, clf


def _irrmapper(out):
    path = os.path.join(out, "irrmapper", "or_irr_test_irrmapper.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    idcol = "OPENET_ID" if "OPENET_ID" in df.columns else df.columns[0]
    long = df.melt(id_vars=[idcol], value_vars=[c for c in df.columns if c.startswith("irr_")])
    long["year"] = long["variable"].str.replace("irr_", "").astype(int)
    s = long.set_index([idcol, "year"])["value"].rename("irrmapper_frac")
    s.index.names = ["field", "year"]
    return s


def report(cfg, run, out=None):
    out = results_dir(cfg, out)
    cs = pd.read_csv(os.path.join(out, "capture_scores.csv"))
    ds = pd.read_csv(os.path.join(out, "daily_scores.csv"))
    ab = pd.read_csv(os.path.join(out, "annual_balance.csv")).set_index(["field", "year"])
    attrs, clf = _attributes(cfg, run)
    irrmapper = _irrmapper(out)

    def pivot(df, window, target, col):
        sub = df[(df.window == window) & (df.target == target)]
        return sub.pivot_table(index=["field", "year"], columns="cell", values=col)

    ens = pivot(cs, PRIMARY_WINDOW, "ensemble", "rmse")
    wsse = pivot(cs, PRIMARY_WINDOW, "ensemble", "wsse")
    n = pivot(cs, PRIMARY_WINDOW, "ensemble", "n")["I"]
    calls = pd.DataFrame(index=ens.index)
    calls["n_captures"] = n
    calls["rmse_I"] = ens["I"]
    calls["rmse_R"] = ens["R"]
    calls["delta_rmse"] = ens["R"] - ens["I"]  # >0: irrigated arm fits better
    calls["llr"] = (wsse["R"] - wsse["I"]) / 2.0
    bias = pivot(cs, PRIMARY_WINDOW, "ensemble", "bias")
    calls["bias_I"] = bias["I"]
    calls["bias_R"] = bias["R"]
    # off-diagonal cells: scheduler effect at fixed vector, vector effect at fixed scheduler
    if "I_rainfedvec" in ens.columns and "R_irrvec" in ens.columns:
        calls["delta_scheduler_rainfedvec"] = ens["R"] - ens["I_rainfedvec"]
        calls["delta_scheduler_irrvec"] = ens["R_irrvec"] - ens["I"]
        calls["delta_vector_armI"] = ens["I_rainfedvec"] - ens["I"]
        calls["delta_vector_armR"] = ens["R"] - ens["R_irrvec"]
    # member agreement: how many of the four members favor arm I
    agree = []
    for m in LST_MODELS:
        pm = pivot(cs, PRIMARY_WINDOW, m, "rmse")
        if {"I", "R"} <= set(pm.columns):
            agree.append(((pm["R"] - pm["I"]) > 0).reindex(calls.index).astype(float))
    calls["members_favoring_I"] = sum(agree) if agree else np.nan
    # sensitivities
    for window in ("apr_oct", "full"):
        pw = pivot(cs, window, "ensemble", "rmse")
        calls[f"delta_rmse_{window}"] = (pw["R"] - pw["I"]).reindex(calls.index)
    dd = ds[ds.window == PRIMARY_WINDOW].pivot_table(
        index=["field", "year"], columns="cell", values="rmse_mm"
    )
    calls["delta_daily_et_rmse_mm"] = (dd["R"] - dd["I"]).reindex(calls.index)
    calls = calls.join(
        ab[["prcp_mm", "et_rs_mm", "et_over_p", "irr_sim_mm_I", "et_mm_I", "et_mm_R"]]
    )
    calls = calls.join(clf)
    if irrmapper is not None:
        calls = calls.join(irrmapper)
    calls = calls.reset_index().merge(
        attrs.reset_index().rename(columns={"index": "field"}), on="field", how="left"
    )
    calls["control"] = calls["field"].str.startswith("ORx_99") | (calls["cdl_cultivated"] == 0)

    # provisional margin: 95th percentile of |delta| over control field-years
    ctrl = calls[calls.control & calls.n_captures.ge(3)]
    margin = float(np.nanpercentile(ctrl["delta_rmse"].abs(), 95)) if len(ctrl) else np.nan
    ctrl_median = float(ctrl["delta_rmse"].median()) if len(ctrl) else np.nan

    def call(row):
        if row.n_captures < 3 or np.isnan(row.delta_rmse):
            return "insufficient"
        if row.delta_rmse > margin:
            return "irrigated"
        if row.delta_rmse < -margin:
            return "rainfed"
        return "indeterminate"

    calls["margin"] = margin
    calls["call"] = calls.apply(call, axis=1)
    # neither arm fits: observed Jun-Sep ETf sits between the arms by more than the spread
    # floor on both sides (Arm I too wet, Arm R too dry). A rainfed/irrigated call carrying
    # this flag only says which arm misses by less, not that the arm fits.
    calls["rmse_best"] = calls[["rmse_I", "rmse_R"]].min(axis=1)
    calls["between_arms"] = (calls.bias_I > BETWEEN_GAP) & (calls.bias_R < -BETWEEN_GAP)
    calls.loc[calls.call == "insufficient", "between_arms"] = False
    calls = calls.sort_values(["field", "year"])
    calls.to_csv(os.path.join(out, "field_year_calls.csv"), index=False, float_format="%.4f")

    # field summary
    g = calls.groupby("field")
    summary = pd.DataFrame(
        {
            "control": g["control"].first(),
            "cdl_cultivated": g["cdl_cultivated"].first(),
            "ITYPE": g["ITYPE"].first(),
            "srctype": g["srctype"].first(),
            "IRR_EFF": g["IRR_EFF"].first(),
            "Acres": g["Acres"].first(),
            "years_scored": g["call"].apply(lambda s: int((s != "insufficient").sum())),
            "years_irrigated_call": g["call"].apply(lambda s: int((s == "irrigated").sum())),
            "years_rainfed_call": g["call"].apply(lambda s: int((s == "rainfed").sum())),
            "years_indeterminate": g["call"].apply(lambda s: int((s == "indeterminate").sum())),
            "years_between_arms": g["between_arms"].sum().astype(int),
            "median_rmse_best": g["rmse_best"].median(),
            "median_delta_rmse": g["delta_rmse"].median(),
            "median_llr": g["llr"].median(),
            "median_members_favoring_I": g["members_favoring_I"].median(),
            "median_et_over_p": g["et_over_p"].median(),
            "median_irr_sim_mm_I": g["irr_sim_mm_I"].median(),
            "classifier_years_irr": g["classifier_f_irr"].apply(lambda s: int((s > 0).sum())),
            "median_bias_R_ctrl_read": g["bias_R"].median(),
        }
    )
    if irrmapper is not None:
        summary["irrmapper_years_irr"] = g["irrmapper_frac"].apply(lambda s: int((s > 0.5).sum()))
        summary["irrmapper_mean_frac"] = g["irrmapper_frac"].mean()
    summary = summary.sort_values(["control", "median_delta_rmse"], ascending=[True, False])
    summary.to_csv(os.path.join(out, "field_summary.csv"), float_format="%.4f")

    null_rows = {
        "control_field_years": int(len(ctrl)),
        "control_delta_median": ctrl_median,
        "control_delta_p05": float(np.nanpercentile(ctrl["delta_rmse"], 5))
        if len(ctrl)
        else np.nan,
        "control_delta_p95": float(np.nanpercentile(ctrl["delta_rmse"], 95))
        if len(ctrl)
        else np.nan,
        "control_abs_delta_p95_margin": margin,
        "control_bias_R_median": float(ctrl["bias_R"].median()) if len(ctrl) else np.nan,
        "control_bias_I_median": float(ctrl["bias_I"].median()) if len(ctrl) else np.nan,
        "fallow_year_delta_median": float(
            calls[(~calls.control) & (calls.classifier_f_irr == 0) & calls.n_captures.ge(3)][
                "delta_rmse"
            ].median()
        ),
    }
    with open(os.path.join(out, "null_statistics.json"), "w") as f:
        json.dump(null_rows, f, indent=2)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\n=== Null statistics (controls = ORx_99* + CDL-uncultivated) ===")
    print(json.dumps(null_rows, indent=2))
    print("\n=== Field summary (sorted: controls last, by median delta) ===")
    print(summary.to_string())
    print(
        "\ndelta_rmse = RMSE_R - RMSE_I on Jun-Sep capture dates vs ensemble ETf; "
        "> margin -> irrigated, < -margin -> rainfed. See field_year_calls.csv."
    )
    _figures(cfg, run, out, calls)


def _figures(cfg, run, out, calls):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    c = open_container(run_container(cfg, run), mode="r")
    fids = list(c.field_uids)
    members, ens, _, _ = _member_frames(c, fids)
    c.close()
    years = list(range(ETF_FIRST_YEAR, int(ens.index.year.max()) + 1))
    ncol = 4
    nrow = int(np.ceil(len(years) / ncol))
    for fid in fids:
        armI = _load_arm(out, "I", fid)
        armR = _load_arm(out, "R", fid)
        fc = calls[calls.field == fid].set_index("year")
        fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.4 * nrow), sharey=True)
        for ax, yr in zip(axes.ravel(), years):
            t0, t1 = pd.Timestamp(yr, 3, 1), pd.Timestamp(yr, 11, 1)
            sl = slice(t0, t1)
            ax.plot(
                armI.loc[sl].index,
                armI.loc[sl, "etf_model"],
                color="tab:blue",
                lw=1,
                label="Arm I (irrigated)",
            )
            ax.plot(
                armR.loc[sl].index,
                armR.loc[sl, "etf_model"],
                color="tab:orange",
                lw=1,
                label="Arm R (rainfed)",
            )
            for m in LST_MODELS:
                s = members[m][fid].loc[sl].dropna()
                ax.plot(s.index, s.values, ".", color="0.6", ms=3)
            e = ens[fid].loc[sl].dropna()
            ax.plot(e.index, e.values, "o", color="k", ms=3.5, label="ensemble ETf")
            ax.axvspan(pd.Timestamp(yr, 6, 1), pd.Timestamp(yr, 9, 30), color="0.92", zorder=0)
            irr = armI.loc[sl, "irr_sim"]
            for d in irr.index[irr > 0]:
                ax.axvline(d, color="tab:blue", alpha=0.15, lw=0.6)
            txt = f"{yr}"
            if yr in fc.index:
                r = fc.loc[yr]
                txt += f"  d={r.delta_rmse:+.2f} {r.call}"
                if r.between_arms:
                    txt += "*"
                txt += f"  irr={r.irr_sim_mm_I:.0f}mm"
            ax.set_title(txt, fontsize=8, loc="left")
            ax.set_ylim(0, 1.4)
            ax.tick_params(labelsize=6)
        for ax in axes.ravel()[len(years) :]:
            ax.axis("off")
        axes.ravel()[0].legend(fontsize=6, loc="upper left")
        a = calls[calls.field == fid].iloc[0]
        fig.suptitle(
            f"{fid}  ITYPE={a.ITYPE} srctype={a.srctype} cdl_cult={a.cdl_cultivated} "
            f"control={bool(a.control)}  median ET/P={fc.et_over_p.median():.2f}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{fid}.png"), dpi=110)
        plt.close(fig)
    print(f"figures -> {fig_dir}")


# ---------------------------------------------------------------------------
# irrmapper comparator (Earth Engine)
# ---------------------------------------------------------------------------
def irrmapper(cfg, out=None):
    """IrrMapper annual irrigated fraction per field; comparator only, never an input."""
    import ee

    from swimrs.data_extraction.ee.ee_props import get_irrigation

    ee.Initialize(project="ee-dgketchum")
    out = os.path.join(results_dir(cfg, out), "irrmapper")
    get_irrigation(
        cfg.fields_shapefile,
        f"{cfg.project_name}_irrmapper",
        selector=cfg.feature_id_col,
        dest="local",
        out_dir=out,
        start_year=IRRMAPPER_YEARS[0],
        end_year=IRRMAPPER_YEARS[1],
        lanid=False,
    )


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("step", choices=["build-arms", "run", "score", "report", "irrmapper", "all"])
    p.add_argument("--run", default="e10cal", help="run container tag")
    p.add_argument("--cells", default=",".join(CELLS), help="comma list of cells to run")
    p.add_argument("--out", default=None, help="results dir (default {project_ws}/results/two_arm)")
    p.add_argument("--overwrite", action="store_true", help="replace existing arm containers")
    args = p.parse_args()
    cfg = load_config()
    cells = {k: CELLS[k] for k in args.cells.split(",") if k}

    if args.step in ("build-arms", "all"):
        build_arms(cfg, args.run, overwrite=args.overwrite, out=args.out)
    if args.step in ("run", "all"):
        run_cells(cfg, args.run, cells, out=args.out)
    if args.step in ("score", "all"):
        score(cfg, args.run, out=args.out)
    if args.step in ("report", "all"):
        report(cfg, args.run, out=args.out)
    if args.step == "irrmapper":
        irrmapper(cfg, out=args.out)


if __name__ == "__main__":
    main()
