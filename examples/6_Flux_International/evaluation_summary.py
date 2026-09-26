"""E2 evaluation summary and RUN_POLICY Category 6 archive for the GrassBasis
run, with the Gate G11 integrity checks.

Consumes what ``evaluate.py`` (daily, ``--monthly``, ``--etf``), ``derived_metrics.py``,
``pooled_metrics.py`` and ``transfer_ex5_params.py`` already wrote to the GrassBasis results
directory, re-derives every headline number from the per-site ``{fid}.csv`` series plus the
QAQC flux archive, and writes ``results/<run>/archive/6_evaluation/``:

* ``daily_paired_metrics.csv`` / ``monthly_paired_metrics.csv`` — the evaluator's rows plus KGE
  components (r, alpha, beta), MAE, and the reproduction delta against the evaluator
* ``overpass_split_metrics.csv`` — SWIM and the interpolated benchmark on retrieval days (a
  finite native member ETf exists on that date) versus between-retrieval days, identical flux
  pairing, both classes >= 10 paired days
* ``group_summary_metrics.csv`` (+ ``lulc_summary_metrics.csv``) — site medians by region,
  country, land cover, irrigation class, with n
* ``daily_exclusions.csv`` / ``monthly_exclusions.csv`` — every cohort site not scored, with
  the gate that removed it; counts reconcile to 66
* ``classifier_transition_vs_metrics.csv`` — the five classifier-transition sites with their
  irrigated-year counts and metric changes
* ``internal_baseline_comparison.csv`` — GrassBasis vs the frozen baseline (iteration-4
  posterior, as evaluated) and vs a read-only iteration-3 forward run of the baseline, scored on
  the intersection mask so the comparison is like-for-like (internal only, plan §17)
* ``transfer_refresh_summary.csv`` — the refreshed class-vector transfer next to the frozen
  2026-08 transfer
* ``site_daily_timeseries/{fid}.csv`` — RUN_POLICY columns from one forward run with the merged
  posterior (asserted identical to the evaluator's ``et_act``)
* ``evaluation_metadata.json`` and ``g11_gate_checks.json``

Read-only on every container and on the baseline results. No calibration, no Earth Engine.

    uv run python examples/6_Flux_International/evaluation_summary.py
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ex6_paths  # noqa: E402

REPO = ex6_paths.REPO
DEFAULT_CONFIG = ex6_paths.CANONICAL_CONFIG
DEFAULT_BASELINE_CONFIG = ex6_paths.BASELINE_CONFIG
DEFAULT_RUN_NAME = ex6_paths.CANONICAL_RUN
BASELINE_RUN_NAME = ex6_paths.BASELINE_RUN
# under the baseline run dir: the frozen classifier-recal evaluation the summary compares against
BASELINE_FROZEN_SUBDIR = Path("archive_recal20260702_classifier") / "6_evaluation"
TRANSFER_NEW_NAME = ex6_paths.TRANSFER_RUN  # under {project_ws}/results
# Historical: the Run 22 transfer scored on the superseded ETr-basis (baseline) footing.
TRANSFER_OLD_NAME = "e2_run22_transfer_by_irrigation_to_e3"
TRANSITION_NAME = "irrigation_classifier_transition.csv"  # under the QA root

MIN_DAILY = 10
MIN_DAILY_FOR_MONTHLY = 30
MIN_DAYS_PER_MONTH = 20
MIN_MONTHS = 6
METRICS = ["r2", "r", "alpha", "beta", "kge", "rmse", "mae", "bias"]
METRIC_DEFINITIONS = {
    "n": "paired observations (days or months) in the declared mask",
    "r2": "coefficient of determination, sklearn r2_score = 1 - SSE/SST (Nash-Sutcliffe form); NOT squared Pearson r",
    "r": "Pearson correlation",
    "alpha": "KGE variability ratio sd(model)/sd(obs)",
    "beta": "KGE bias ratio mean(model)/mean(obs)",
    "kge": "Kling-Gupta efficiency 2009: 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)",
    "rmse": "root mean square error (mm/d daily, mm/month monthly)",
    "mae": "mean absolute error",
    "bias": "mean(model - obs)",
}
DAILY_MASK = (
    "site passes VALIDATION_POLICY minimum (>=90 finite flux days and >=3 months with >=20 finite "
    "days); day counted when flux, SWIM ET and the interpolated Landsat-ensemble benchmark ET are "
    "all finite; >=10 such days; SWIM and benchmark scored on the identical day set"
)
MONTHLY_MASK = (
    "same site gate; >=30 overlapping days; monthly sums over flux-finite days only with >=20 such "
    "days per month; benchmark month NaN unless finite on every flux-finite day; month counted "
    "when flux, SWIM and benchmark sums are all finite; >=6 such months"
)


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def full_metrics(obs: np.ndarray, mod: np.ndarray, min_n: int = MIN_DAILY) -> dict:
    """calc_metrics-compatible metrics plus KGE components and MAE on finite pairs."""
    obs = np.asarray(obs, dtype=float)
    mod = np.asarray(mod, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(mod)
    o, m = obs[mask], mod[mask]
    out = {"n": int(len(o))}
    if len(o) < min_n:
        out.update({k: np.nan for k in METRICS})
        return out
    r = float(np.corrcoef(o, m)[0, 1])
    sst = float(np.sum((o - o.mean()) ** 2))
    sse = float(np.sum((m - o) ** 2))
    so, mo = float(np.std(o)), float(np.mean(o))
    alpha = float(np.std(m) / so) if so > 0 else np.nan
    beta = float(np.mean(m) / mo) if mo != 0 else np.nan
    out.update(
        {
            "r2": 1.0 - sse / sst if sst > 0 else np.nan,
            "r": r,
            "alpha": alpha,
            "beta": beta,
            "kge": 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)),
            "rmse": float(np.sqrt(np.mean((m - o) ** 2))),
            "mae": float(np.mean(np.abs(m - o))),
            "bias": float(np.mean(m - o)),
        }
    )
    return out


def daily_pairing(flux: pd.Series, swim: pd.Series, rs: pd.Series):
    """(dates, obs, swim, rs) on the declared daily mask, or (reason, None...) when excluded."""
    common = swim.index.intersection(flux.index)
    if len(common) < MIN_DAILY:
        return f"overlap_lt_{MIN_DAILY}_days", None, None, None, None
    obs = flux.loc[common].to_numpy(float)
    sv = swim.loc[common].to_numpy(float)
    rv = rs.reindex(common).to_numpy(float)
    mask = np.isfinite(obs) & np.isfinite(sv) & np.isfinite(rv)
    if int(mask.sum()) < MIN_DAILY:
        return f"paired_days_lt_{MIN_DAILY}", None, None, None, None
    return None, common[mask], obs[mask], sv[mask], rv[mask]


def monthly_pairing(flux: pd.Series, swim: pd.Series, rs: pd.Series):
    """(months, obs, swim, rs) on the declared monthly mask, or (reason, None...)."""
    from swimrs.calibrate.flux_utils import paired_monthly_sums

    common = swim.index.intersection(flux.index)
    if len(common) < MIN_DAILY_FOR_MONTHLY:
        return f"daily_overlap_lt_{MIN_DAILY_FOR_MONTHLY}", None, None, None, None
    s_m, f_m, r_m = paired_monthly_sums(
        swim.loc[common], flux.loc[common], rs.reindex(common), month_min_days=MIN_DAYS_PER_MONTH
    )
    idx = f_m.index
    r_on = r_m.reindex(idx)
    mask = f_m.notna() & s_m.reindex(idx).notna() & r_on.notna()
    months = idx[mask]
    if len(months) < MIN_MONTHS:
        return f"paired_months_lt_{MIN_MONTHS}", None, None, None, None
    return (
        None,
        months,
        f_m.loc[months].to_numpy(float),
        s_m.reindex(months).to_numpy(float),
        r_on.loc[months].to_numpy(float),
    )


def retrieval_mask(dates: pd.DatetimeIndex, member_frames: list[pd.Series]) -> np.ndarray:
    """True where at least one native member ETf is finite on that date (a retrieval day)."""
    hit = np.zeros(len(dates), dtype=bool)
    for s in member_frames:
        hit |= np.isfinite(s.reindex(dates).to_numpy(float))
    return hit


def overpass_split_row(fid, dates, obs, sv, rv, is_retrieval) -> dict | None:
    """Per-site retrieval vs between-retrieval metrics on the paired daily mask."""
    n_ret, n_btw = int(is_retrieval.sum()), int((~is_retrieval).sum())
    if n_ret < MIN_DAILY or n_btw < MIN_DAILY:
        return None
    row = {"fid": fid, "n_paired": int(len(dates)), "n_retrieval": n_ret, "n_between": n_btw}
    for cls, sel in (("retrieval", is_retrieval), ("between", ~is_retrieval)):
        for model, vals in (("swim", sv), ("rs", rv)):
            m = full_metrics(obs[sel], vals[sel])
            for k in METRICS:
                row[f"{k}_{model}_{cls}"] = m[k]
    for k in METRICS:
        row[f"{k}_swim_minus_rs_retrieval"] = row[f"{k}_swim_retrieval"] - row[f"{k}_rs_retrieval"]
        row[f"{k}_swim_minus_rs_between"] = row[f"{k}_swim_between"] - row[f"{k}_rs_between"]
        row[f"{k}_support_interaction"] = (
            row[f"{k}_swim_minus_rs_between"] - row[f"{k}_swim_minus_rs_retrieval"]
        )
    return row


def reconcile_counts(n_cohort: int, n_scored: int, exclusions: pd.DataFrame) -> dict:
    """Cohort = scored + excluded, with no site in both lists."""
    n_excl = int(len(exclusions))
    return {
        "n_cohort": int(n_cohort),
        "n_scored": int(n_scored),
        "n_excluded": n_excl,
        "reconciles": bool(n_cohort == n_scored + n_excl),
    }


def group_medians(metrics: pd.DataFrame, groups: pd.DataFrame, basis: str) -> pd.DataFrame:
    """Median (and n) of each metric for SWIM and benchmark within each grouping."""
    rows = []
    cols = [c for c in metrics.columns if c.rsplit("_", 1)[-1] in ("swim", "rs")]
    for kind in groups.columns:
        for label, sub in groups.groupby(kind):
            idx = metrics.index.intersection(sub.index)
            if len(idx) == 0:
                continue
            row = {
                "basis": basis,
                "group_kind": kind,
                "group": str(label),
                "n_sites": int(len(idx)),
            }
            for c in cols:
                row[f"{c}_median"] = float(metrics.loc[idx, c].median())
            rows.append(row)
    return pd.DataFrame(rows)


def _repro_delta(evaluator_row, row: dict) -> float:
    """Largest |evaluator - recomputed| over the shared metrics; NaN on both sides counts as 0,
    NaN on one side only as inf."""
    worst = 0.0
    for k in ("r2", "r", "rmse", "bias", "kge"):
        for s in ("swim", "rs"):
            a, b = float(evaluator_row[f"{k}_{s}"]), float(row[f"{k}_{s}"])
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) or np.isnan(b):
                return float("inf")
            worst = max(worst, abs(a - b))
    return worst


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# model runs (read-only containers)
# ---------------------------------------------------------------------------


def forward_run_full(cfg, container, fids, params_by_fid):
    """One forward run returning the RUN_POLICY time-series fields per site."""
    from swimrs.process.input import build_swim_input
    from swimrs.process.loop_fast import run_daily_loop_fast

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        temp_h5 = tmp.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(params_by_fid, tmp)
        params_json = tmp.name
    try:
        si = build_swim_input(
            container,
            output_h5=temp_h5,
            calibrated_params_path=params_json,
            start_date=cfg.start_dt,
            end_date=cfg.end_dt,
            refet_type=getattr(cfg, "refet_type", "eto") or "eto",
            etf_model=getattr(cfg, "etf_target_model", "ptjpl"),
            met_source=getattr(cfg, "met_source", "era5"),
            fields=fids,
            empirical_kc_max=True,
            mask_mode=getattr(cfg, "mask_mode", "none"),
        )
        out, _ = run_daily_loop_fast(si)
        dates = pd.date_range(si.start_date, periods=si.n_days, freq="D")
        eto = si.get_time_series("eto")
        prcp = si.get_time_series("prcp")
        res = {}
        for i, fid in enumerate(si.fids):
            res[fid] = pd.DataFrame(
                {
                    "swim_ET": out.eta[:, i],
                    "precip": prcp[:, i],
                    "eto": eto[:, i],
                    "ndvi_kcb": out.kcb[:, i],
                    "ks": out.ks[:, i],
                    "rz_depletion": out.depl_root[:, i],
                    "irr_applied": out.irr_sim[:, i],
                    "swim_etf": out.etf[:, i],
                    "swe": out.swe[:, i],
                },
                index=dates,
            )
        si.close()
    finally:
        for p in (temp_h5, params_json):
            if os.path.exists(p):
                os.remove(p)
    return res


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    p.add_argument("--results-root", default=None, help="default {project_ws}/results")
    p.add_argument("--baseline-config", default=str(DEFAULT_BASELINE_CONFIG))
    p.add_argument("--baseline-results", default=None, help="default <results-root>/<baseline run>")
    p.add_argument(
        "--baseline-frozen-eval",
        default=None,
        help=f"default <baseline-results>/{BASELINE_FROZEN_SUBDIR}",
    )
    p.add_argument("--baseline-pestrun", default=None, help="default the baseline pest_run_dir")
    p.add_argument("--baseline-iteration", type=int, default=3)
    p.add_argument(
        "--transfer-new", default=None, help=f"default <results-root>/{TRANSFER_NEW_NAME}"
    )
    p.add_argument(
        "--transfer-old", default=None, help=f"default <results-root>/{TRANSFER_OLD_NAME}"
    )
    p.add_argument("--transition-csv", default=None, help=f"default <qa-root>/{TRANSITION_NAME}")
    p.add_argument("--skip-baseline-forward", action="store_true")
    args = p.parse_args()

    import evaluate as ev
    import geopandas as gpd
    import zarr
    from archive_postcalibration import (
        merge_par_csvs,
        posterior_medians,
        read_irrigation_class,
    )

    from swimrs.calibrate.flux_utils import passes_site_minimum
    from swimrs.container import SwimContainer

    conf_path = Path(args.config)
    cfg = ev._load_config(conf_path)
    results_root = Path(args.results_root) if args.results_root else ex6_paths.results_root(cfg)
    if args.baseline_results is None:
        args.baseline_results = str(results_root / BASELINE_RUN_NAME)
    if args.baseline_frozen_eval is None:
        args.baseline_frozen_eval = str(Path(args.baseline_results) / BASELINE_FROZEN_SUBDIR)
    if args.baseline_pestrun is None:
        args.baseline_pestrun = ex6_paths.load_config(Path(args.baseline_config)).pest_run_dir
    if args.transfer_new is None:
        args.transfer_new = str(results_root / TRANSFER_NEW_NAME)
    if args.transfer_old is None:
        args.transfer_old = str(results_root / TRANSFER_OLD_NAME)
    if args.transition_csv is None:
        args.transition_csv = str(ex6_paths.qa_root(cfg) / TRANSITION_NAME)
    results = results_root / args.run_name
    archive = results / "archive"
    cat6 = archive / "6_evaluation"
    ts_dir = cat6 / "site_daily_timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = archive / "4_pest_outputs" / "merged" / "merged_posterior.csv"
    problems: list[str] = []
    checks: dict = {}

    container = SwimContainer.open(cfg.container_path, mode="r")
    root = zarr.open_group(cfg.container_path, mode="r")
    gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")
    id_col = cfg.feature_id_col if cfg.feature_id_col in gdf.columns else "sid"
    cohort = [str(s) for s in gdf[id_col]]
    fids = [f for f in cohort if f in set(container.field_uids)]
    gdf = gdf.drop_duplicates(id_col).set_index(gdf[id_col].astype(str))
    flux_sources = ev.load_flux_sources(cfg.fields_shapefile, cfg.feature_id_col)

    groups = pd.DataFrame(index=pd.Index(fids, name="site_id"))
    groups["region"] = ["CONUS" if f.startswith("US-") else "ex-CONUS" for f in fids]
    groups["country"] = [
        gdf.loc[f, "country"] if isinstance(gdf.loc[f, "country"], str) else f.split("-")[0]
        for f in fids
    ]
    groups["lulc"] = [f"glc10_{int(gdf.loc[f, 'glc10_lulc'])}" for f in fids]
    groups["irrigation_class"] = read_irrigation_class(root, fids)
    groups.to_csv(cat6 / "site_groups.csv")

    # ---- load evaluator outputs and per-site series ----
    ev_daily = pd.read_csv(results / "evaluation_metrics.csv", index_col=0)
    ev_monthly = pd.read_csv(results / "evaluation_monthly_metrics.csv", index_col=0)
    ev_excl = pd.read_csv(results / "evaluation_sites_excluded.csv")
    site_series = {}
    for fid in fids:
        path = results / f"{fid}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = df.index.normalize()
            site_series[fid] = df
    flux_by_fid = {fid: ev.load_flux_et(fid, flux_sources.get(fid)) for fid in fids}

    members = list(getattr(cfg, "etf_ensemble_members", None) or [])
    instrument = getattr(cfg, "etf_target_instrument", "landsat")
    member_series = {
        fid: [
            s
            for s in (
                ev._query_etf_series(container, f"remote_sensing/etf/{instrument}/{m}/no_mask", fid)
                for m in members
            )
            if s is not None
        ]
        for fid in fids
    }

    # ---- daily / monthly reproduction with components; exclusions ----
    daily_rows, monthly_rows, daily_excl, monthly_excl, split_rows = [], [], [], [], []
    daily_masks = {}
    for fid in fids:
        flux = flux_by_fid[fid]
        if flux.empty:
            daily_excl.append({"site": fid, "reason": "no_flux_data"})
            monthly_excl.append({"site": fid, "reason": "no_flux_data"})
            continue
        if not passes_site_minimum(flux):
            daily_excl.append({"site": fid, "reason": "below_site_minimum_90d_3mo"})
            monthly_excl.append({"site": fid, "reason": "below_site_minimum_90d_3mo"})
            continue
        df = site_series.get(fid)
        if df is None or "et_rs" not in df or not df["et_rs"].notna().any():
            daily_excl.append({"site": fid, "reason": "no_rs_benchmark"})
            monthly_excl.append({"site": fid, "reason": "no_rs_benchmark"})
            continue
        swim, rs = df["et_act"], df["et_rs"]

        reason, dates, obs, sv, rv = daily_pairing(flux, swim, rs)
        if reason:
            daily_excl.append({"site": fid, "reason": reason})
        else:
            daily_masks[fid] = dates
            row = {"fid": fid, "n": int(len(dates))}
            row["flux_network"] = (
                os.path.basename(os.path.dirname(flux.attrs.get("flux_file", ""))) or None
            )
            row["flux_et_col"] = flux.attrs.get("et_col")
            for model, vals in (("swim", sv), ("rs", rv)):
                m = full_metrics(obs, vals)
                for k in METRICS:
                    row[f"{k}_{model}"] = m[k]
            if fid in ev_daily.index:
                e = ev_daily.loc[fid]
                row["repro_max_abs_delta"] = _repro_delta(e, row)
                row["repro_n_delta"] = int(e["n"]) - row["n"]
            else:
                row["repro_max_abs_delta"] = np.nan
                row["repro_n_delta"] = np.nan
                problems.append(f"{fid}: scored here but absent from evaluation_metrics.csv")
            daily_rows.append(row)
            is_ret = retrieval_mask(dates, member_series[fid])
            srow = overpass_split_row(fid, dates, obs, sv, rv, is_ret)
            if srow is not None:
                split_rows.append(srow)

        reason, months, obs_m, sv_m, rv_m = monthly_pairing(flux, swim, rs)
        if reason:
            monthly_excl.append({"site": fid, "reason": reason})
        else:
            row = {"fid": fid, "n": int(len(months))}
            # the evaluator keeps a site at >= 6 paired months but its calc_metrics returns NaN
            # below 10 observations; reproduce that rule so the deltas are like-for-like
            for model, vals in (("swim", sv_m), ("rs", rv_m)):
                m = full_metrics(obs_m, vals, min_n=MIN_DAILY)
                for k in METRICS:
                    row[f"{k}_{model}"] = m[k]
            if fid in ev_monthly.index:
                e = ev_monthly.loc[fid]
                row["repro_max_abs_delta"] = _repro_delta(e, row)
                row["repro_n_delta"] = int(e["n"]) - row["n"]
            else:
                row["repro_max_abs_delta"] = np.nan
                row["repro_n_delta"] = np.nan
                problems.append(f"{fid}: monthly scored here but absent from evaluator output")
            monthly_rows.append(row)

    daily = pd.DataFrame(daily_rows).set_index("fid")
    monthly = pd.DataFrame(monthly_rows).set_index("fid")
    daily_excl_df = pd.DataFrame(daily_excl, columns=["site", "reason"])
    monthly_excl_df = pd.DataFrame(monthly_excl, columns=["site", "reason"])
    daily.to_csv(cat6 / "daily_paired_metrics.csv")
    monthly.to_csv(cat6 / "monthly_paired_metrics.csv")
    daily_excl_df.to_csv(cat6 / "daily_exclusions.csv", index=False)
    monthly_excl_df.to_csv(cat6 / "monthly_exclusions.csv", index=False)
    pd.DataFrame(split_rows).set_index("fid").to_csv(cat6 / "overpass_split_metrics.csv")

    # reproduction checks against the evaluator
    if set(daily.index) != set(ev_daily.index):
        problems.append(
            f"daily site set differs from evaluator: {sorted(set(daily.index) ^ set(ev_daily.index))}"
        )
    if set(monthly.index) != set(ev_monthly.index):
        problems.append(
            f"monthly site set differs from evaluator: {sorted(set(monthly.index) ^ set(ev_monthly.index))}"
        )
    # The benchmark series is float32 in the container (ETf arrays); the evaluator scored it in
    # memory and wrote it through CSV, so recomputed metrics differ by float32 round-trip noise
    # (~1e-7 daily, ~1e-6 on monthly sums). A mask or definition mismatch shows as >= 1e-3.
    tol = 1e-5
    if daily["repro_max_abs_delta"].max() > tol or (daily["repro_n_delta"] != 0).any():
        problems.append("daily metrics do not reproduce evaluation_metrics.csv from site files")
    if monthly["repro_max_abs_delta"].max() > tol or (monthly["repro_n_delta"] != 0).any():
        problems.append("monthly metrics do not reproduce evaluation_monthly_metrics.csv")
    checks["reproduction"] = {
        "daily_max_abs_delta": float(daily["repro_max_abs_delta"].max()),
        "monthly_max_abs_delta": float(monthly["repro_max_abs_delta"].max()),
        "tolerance": tol,
    }
    checks["daily_counts"] = reconcile_counts(len(fids), len(daily), daily_excl_df)
    checks["monthly_counts"] = reconcile_counts(len(fids), len(monthly), monthly_excl_df)
    for k in ("daily_counts", "monthly_counts"):
        if not checks[k]["reconciles"]:
            problems.append(f"{k} do not reconcile: {checks[k]}")
    ev_excl_sites = set(ev_excl["site"]) if len(ev_excl) else set()
    gate_excl = set(
        daily_excl_df.loc[
            daily_excl_df["reason"].isin(["no_flux_data", "below_site_minimum_90d_3mo"]), "site"
        ]
    )
    if ev_excl_sites != gate_excl:
        problems.append(
            f"evaluator exclusion file {sorted(ev_excl_sites)} != gate exclusions {sorted(gate_excl)}"
        )

    # ---- headline aggregates ----
    def _agg(frame, basis):
        rows = []
        for model in ("swim", "rs"):
            row = {"basis": basis, "model": model, "n_sites": int(len(frame))}
            for k in METRICS:
                row[f"{k}_median"] = float(frame[f"{k}_{model}"].median())
                row[f"{k}_mean"] = float(frame[f"{k}_{model}"].mean())
            rows.append(row)
        # paired site-level differences (SWIM minus benchmark)
        row = {"basis": basis, "model": "swim_minus_rs_paired", "n_sites": int(len(frame))}
        for k in METRICS:
            d = frame[f"{k}_swim"] - frame[f"{k}_rs"]
            row[f"{k}_median"] = float(d.median())
            row[f"{k}_mean"] = float(d.mean())
        rows.append(row)
        return rows

    headline = pd.DataFrame(_agg(daily, "daily") + _agg(monthly, "monthly"))
    headline.to_csv(cat6 / "headline_aggregates.csv", index=False)

    grp = pd.concat(
        [group_medians(daily, groups, "daily"), group_medians(monthly, groups, "monthly")]
    )
    grp.to_csv(cat6 / "group_summary_metrics.csv", index=False)
    grp.loc[grp["group_kind"] == "lulc"].to_csv(cat6 / "lulc_summary_metrics.csv", index=False)

    split = pd.DataFrame(split_rows).set_index("fid") if split_rows else pd.DataFrame()
    if len(split):
        srows = []
        for k in METRICS:
            srows.append(
                {
                    "metric": k,
                    "n_sites": int(len(split)),
                    "swim_retrieval_median": float(split[f"{k}_swim_retrieval"].median()),
                    "rs_retrieval_median": float(split[f"{k}_rs_retrieval"].median()),
                    "swim_between_median": float(split[f"{k}_swim_between"].median()),
                    "rs_between_median": float(split[f"{k}_rs_between"].median()),
                    "paired_swim_minus_rs_retrieval_median": float(
                        split[f"{k}_swim_minus_rs_retrieval"].median()
                    ),
                    "paired_swim_minus_rs_between_median": float(
                        split[f"{k}_swim_minus_rs_between"].median()
                    ),
                    "support_interaction_median": float(split[f"{k}_support_interaction"].median()),
                }
            )
        pd.DataFrame(srows).to_csv(cat6 / "overpass_split_summary.csv", index=False)

    # ---- internal baseline comparison (iteration-4 as evaluated; iteration-3 forward run) ----
    base_res = Path(args.baseline_results)
    frozen = Path(args.baseline_frozen_eval)
    base_daily_frozen = pd.read_csv(frozen / "evaluation_metrics.csv", index_col=0)
    base_monthly_frozen = pd.read_csv(frozen / "evaluation_monthly_metrics.csv", index_col=0)
    live_daily = pd.read_csv(base_res / "evaluation_metrics.csv", index_col=0)
    frozen_matches_live = set(live_daily.index) == set(base_daily_frozen.index) and np.allclose(
        live_daily.loc[base_daily_frozen.index, ["kge_swim", "r2_swim"]].to_numpy(float),
        base_daily_frozen[["kge_swim", "r2_swim"]].to_numpy(float),
    )
    checks["baseline_frozen_equals_live_root"] = bool(frozen_matches_live)

    base_series = {}
    for fid in fids:
        path = base_res / f"{fid}.csv"
        if path.exists():
            b = pd.read_csv(path, index_col=0, parse_dates=True)
            b.index = b.index.normalize()
            base_series[fid] = b

    base_it3 = {}
    if not args.skip_baseline_forward:
        bcfg = ev._load_config(Path(args.baseline_config))
        bcontainer = SwimContainer.open(bcfg.container_path, mode="r")
        try:
            paths = sorted(
                Path(args.baseline_pestrun).glob(
                    f"pest_archive/batch_*/{bcfg.project_name}.{args.baseline_iteration}.par.csv"
                )
            )
            # the baseline problem covers its 75-site container; decode against every uid
            bmed = posterior_medians(merge_par_csvs(paths), list(bcontainer.field_uids))
            bfids = [f for f in fids if f in bmed.index]
            bparams = {f: {k: float(v) for k, v in bmed.loc[f].items()} for f in bfids}
            uncovered = sorted(set(fids) - set(bfids))
            if uncovered:
                problems.append(f"baseline posterior does not cover GrassBasis sites {uncovered}")
            print(
                f"Baseline iteration-{args.baseline_iteration} forward run ({len(bparams)} sites)…"
            )
            base_it3 = ev.run_calibrated_model(bcfg, bcontainer, bfids, bparams)
        finally:
            bcontainer.close()

    comp_rows = []
    for fid in daily.index:
        flux = flux_by_fid[fid]
        gb = site_series[fid]
        b = base_series.get(fid)
        if b is None or "et_rs" not in b:
            continue
        common = gb.index.intersection(b.index).intersection(flux.index)
        obs = flux.loc[common].to_numpy(float)
        cols = {
            "grassbasis_swim": gb["et_act"].reindex(common).to_numpy(float),
            "grassbasis_rs": gb["et_rs"].reindex(common).to_numpy(float),
            "baseline_it4_swim": b["et_act"].reindex(common).to_numpy(float),
            "baseline_rs": b["et_rs"].reindex(common).to_numpy(float),
        }
        if fid in base_it3:
            cols["baseline_it3_swim"] = base_it3[fid]["et_act"].reindex(common).to_numpy(float)
        mask = np.isfinite(obs)
        for v in cols.values():
            mask &= np.isfinite(v)
        if int(mask.sum()) < MIN_DAILY:
            continue
        row = {
            "fid": fid,
            "n_common": int(mask.sum()),
            "n_grassbasis_mask": int(daily.loc[fid, "n"]),
        }
        for name, v in cols.items():
            m = full_metrics(obs[mask], v[mask])
            for k in METRICS:
                row[f"{k}_{name}"] = m[k]
        # monthly on the intersection: same rule as monthly_pairing but with every series
        f_d = flux.loc[common[mask]]
        mon_ok = True
        mon = {}
        from swimrs.calibrate.flux_utils import paired_monthly_sums

        for name, v in cols.items():
            s = pd.Series(v[mask], index=common[mask])
            s_m, f_m, _ = paired_monthly_sums(s, f_d, None, month_min_days=MIN_DAYS_PER_MONTH)
            mon[name] = s_m
            mon["_flux"] = f_m
        idx = mon["_flux"].index
        mmask = mon["_flux"].notna()
        for name in cols:
            mmask &= mon[name].reindex(idx).notna()
        if int(mmask.sum()) < MIN_MONTHS:
            mon_ok = False
        row["n_common_months"] = int(mmask.sum())
        if mon_ok:
            o_m = mon["_flux"].loc[mmask].to_numpy(float)
            for name in cols:
                m = full_metrics(o_m, mon[name].reindex(idx).loc[mmask].to_numpy(float), MIN_MONTHS)
                for k in METRICS:
                    row[f"{k}_{name}_monthly"] = m[k]
        comp_rows.append(row)
    comp = pd.DataFrame(comp_rows).set_index("fid")
    comp.to_csv(cat6 / "internal_baseline_comparison.csv")
    comp_summary = []
    for basis, suffix in (("daily", ""), ("monthly", "_monthly")):
        for name in [
            "grassbasis_swim",
            "baseline_it3_swim",
            "baseline_it4_swim",
            "grassbasis_rs",
            "baseline_rs",
        ]:
            col = f"kge_{name}{suffix}"
            if col not in comp:
                continue
            row = {"basis": basis, "series": name, "n_sites": int(comp[col].notna().sum())}
            for k in METRICS:
                row[f"{k}_median"] = float(comp[f"{k}_{name}{suffix}"].median())
            comp_summary.append(row)
        for a, bname in (
            ("grassbasis_swim", "baseline_it3_swim"),
            ("grassbasis_swim", "baseline_it4_swim"),
            ("grassbasis_rs", "baseline_rs"),
        ):
            if f"kge_{bname}{suffix}" not in comp:
                continue
            row = {
                "basis": basis,
                "series": f"{a}_minus_{bname}_paired",
                "n_sites": int(comp[f"kge_{bname}{suffix}"].notna().sum()),
            }
            for k in METRICS:
                d = comp[f"{k}_{a}{suffix}"] - comp[f"{k}_{bname}{suffix}"]
                row[f"{k}_median"] = float(d.median())
            comp_summary.append(row)
    pd.DataFrame(comp_summary).to_csv(
        cat6 / "internal_baseline_comparison_summary.csv", index=False
    )

    # ---- classifier transitions vs metric change ----
    tr = pd.read_csv(args.transition_csv)
    site_tr = tr.groupby("site").agg(
        irr_years_baseline=("irrigated_baseline", "sum"),
        irr_years_corrected=("irrigated_corrected", "sum"),
        n_years=("year", "count"),
    )
    site_tr["transition"] = np.where(
        site_tr["irr_years_baseline"] != site_tr["irr_years_corrected"], "changed", "unchanged"
    )
    for k in ("kge", "r2", "bias", "rmse"):
        if f"{k}_baseline_it4_swim" in comp:
            site_tr[f"{k}_grassbasis_minus_baseline_it4"] = (
                comp[f"{k}_grassbasis_swim"] - comp[f"{k}_baseline_it4_swim"]
            )
        if f"{k}_baseline_it3_swim" in comp:
            site_tr[f"{k}_grassbasis_minus_baseline_it3"] = (
                comp[f"{k}_grassbasis_swim"] - comp[f"{k}_baseline_it3_swim"]
            )
    site_tr.to_csv(cat6 / "classifier_transition_vs_metrics.csv")
    tr_summary = site_tr.groupby("transition").agg(
        n_sites=("n_years", "count"),
        **{
            c: (c, "median")
            for c in site_tr.columns
            if c.startswith(("kge_", "r2_", "bias_", "rmse_"))
        },
    )
    tr_summary.to_csv(cat6 / "classifier_transition_summary.csv")

    # ---- transfer refresh ----
    t_new, t_old = Path(args.transfer_new), Path(args.transfer_old)
    tframes = []
    for label, d in (("grassbasis_refresh", t_new), ("frozen_2026-08_e3", t_old)):
        f = d / "transfer_comparison_summary.csv"
        if f.exists():
            t = pd.read_csv(f)
            t.insert(0, "application", label)
            tframes.append(t)
        else:
            problems.append(f"transfer summary missing: {f}")
    if tframes:
        pd.concat(tframes).to_csv(cat6 / "transfer_refresh_summary.csv", index=False)
    t_meta = {}
    if (t_new / "run_metadata.json").exists():
        t_meta = json.loads((t_new / "run_metadata.json").read_text())

    # ---- Cat 6 site daily time series from one forward run with the merged posterior ----
    params_by_fid = ev.parse_pest_params(merged_csv, fids)
    print(f"GrassBasis forward run for time series ({len(params_by_fid)} sites)…")
    full = forward_run_full(cfg, container, fids, params_by_fid)
    weights = pd.concat(
        [
            pd.read_csv(p, usecols=["fid", "date", "weight_final"])
            for p in sorted((archive / "4_pest_outputs").glob("batch_*/weight_audit.csv"))
        ]
    )
    weights["date"] = pd.to_datetime(weights["date"])
    weights = weights.set_index(["fid", "date"])["weight_final"]
    max_ts_delta = 0.0
    for fid, df in full.items():
        ser = site_series.get(fid)
        if ser is not None:
            d = float(
                np.nanmax(
                    np.abs(df["swim_ET"].to_numpy() - ser["et_act"].reindex(df.index).to_numpy())
                )
            )
            max_ts_delta = max(max_ts_delta, d)
        out = pd.DataFrame(index=df.index)
        out.index.name = "date"
        out["flux_ET"] = (
            flux_by_fid[fid].reindex(df.index) if not flux_by_fid[fid].empty else np.nan
        )
        out["swim_ET"] = df["swim_ET"]
        out["benchmark_ET"] = (
            ser["et_rs"].reindex(df.index) if ser is not None and "et_rs" in ser else np.nan
        )
        for c in ("precip", "eto", "ndvi_kcb", "ks", "rz_depletion", "irr_applied"):
            out[c] = df[c]
        obs_etf = (
            pd.concat(member_series[fid], axis=1).mean(axis=1, skipna=True)
            if member_series[fid]
            else pd.Series(dtype=float)
        )
        out["observed_etf"] = obs_etf.reindex(df.index)
        out["is_overpass"] = out["observed_etf"].notna()
        out["sensor"] = np.where(out["is_overpass"], instrument, None)
        w = weights.loc[fid] if fid in weights.index.get_level_values(0) else pd.Series(dtype=float)
        out["obs_weight"] = w.reindex(df.index).to_numpy() if len(w) else np.nan
        out["swim_etf"] = df["swim_etf"]
        out["swe"] = df["swe"]
        out.to_csv(ts_dir / f"{fid}.csv", float_format="%.6g")
    checks["timeseries_vs_evaluator_max_abs_delta"] = max_ts_delta
    if max_ts_delta > 1e-6:
        problems.append(
            f"forward-run swim_ET differs from evaluator et_act by up to {max_ts_delta}"
        )

    # ---- copy the evaluator / derived / pooled products into Cat 6 ----
    for name in [
        "evaluation_metrics.csv",
        "evaluation_monthly_metrics.csv",
        "evaluation_etf_metrics.csv",
        "evaluation_summary.csv",
        "evaluation_sites_excluded.csv",
        "pooled_metrics_daily.csv",
        "pooled_metrics_monthly.csv",
        "derived/derived_per_model_benchmarks.csv",
        "derived/derived_conus_decomp_persite.csv",
        "derived/derived_conus_decomp_summary.csv",
        "derived/derived_murphy_decomp.csv",
        "derived/derived_uncal_baseline_persite.csv",
    ]:
        src = results / name
        name = Path(name).name
        if src.exists():
            (cat6 / name).write_bytes(src.read_bytes())
        else:
            problems.append(f"evaluator product missing: {src}")

    # ---- metadata + gate checks ----
    checks["identical_masks"] = {
        "swim_vs_benchmark": "same paired mask by construction (daily_pairing/monthly_pairing)",
        "internal_baseline_comparison": "intersection mask of flux, both SWIM series and both benchmarks",
        "transfer": "transfer_ex5_params.py scores every configuration on the ex5/e3 common support with the same flux-driven months",
    }
    checks["problems"] = problems
    checks["pass"] = not problems
    (cat6 / "g11_gate_checks.json").write_text(json.dumps(checks, indent=2, default=str))

    meta = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "run_name": args.run_name,
        "git_sha": _git_sha(),
        "config": str(conf_path),
        "config_sha256": sha256_file(conf_path),
        "container": str(cfg.container_path),
        "parameter_source": {
            "path": str(merged_csv),
            "sha256": sha256_file(merged_csv),
            "posterior_iteration": 3,
            "statistic": "median over realizations (base excluded)",
        },
        "flux_archive": str(cfg.flux_dir),
        "flux_sources": "per-site (network, et_col) from the cohort shapefile",
        "cohort_shapefile": str(cfg.fields_shapefile),
        "period": [str(cfg.start_dt.date()), str(cfg.end_dt.date())],
        "benchmark": (
            "Landsat SSEBop(grass basis) + PT-JPL member-mean ETf on retrieval dates, linearly "
            "interpolated to daily, x daily ERA5-Land ETo"
        ),
        "masks": {"daily": DAILY_MASK, "monthly": MONTHLY_MASK},
        "monthly_min_paired_months": MIN_MONTHS,
        "metric_definitions": METRIC_DEFINITIONS,
        "counts": {
            "cohort": len(fids),
            "daily_scored": int(len(daily)),
            "monthly_scored": int(len(monthly)),
            "overpass_split_scored": int(len(split)),
            "daily_exclusions": daily_excl_df["reason"].value_counts().to_dict(),
            "monthly_exclusions": monthly_excl_df["reason"].value_counts().to_dict(),
        },
        "baseline_reference": {
            "frozen_evaluation": str(frozen),
            "posterior_iteration_as_evaluated": 4,
            "iteration3_forward_run": not args.skip_baseline_forward,
            "note": "internal comparison only (plan §17); not a manuscript experiment",
        },
        "transfer_refresh": {
            "out_dir": str(t_new),
            "class_counts": (t_meta.get("stratified_transfer") or {}).get("class_counts"),
        },
        "gate_checks": str(cat6 / "g11_gate_checks.json"),
    }
    (cat6 / "evaluation_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    container.close()

    pd.set_option("display.width", 200)
    print("\nHEADLINE (site medians; n = sites)")
    print(
        headline[
            [
                "basis",
                "model",
                "n_sites",
                "r2_median",
                "kge_median",
                "r_median",
                "alpha_median",
                "beta_median",
                "rmse_median",
                "bias_median",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("\nINTERNAL BASELINE COMPARISON (intersection masks)")
    print(
        pd.DataFrame(comp_summary)[
            ["basis", "series", "n_sites", "r2_median", "kge_median", "bias_median"]
        ]
        .round(3)
        .to_string(index=False)
    )
    if len(split):
        print("\nRETRIEVAL vs BETWEEN (site medians)")
        print(pd.read_csv(cat6 / "overpass_split_summary.csv").round(3).to_string(index=False))
    print("\nEXCLUSIONS daily:", daily_excl_df["reason"].value_counts().to_dict())
    print("EXCLUSIONS monthly:", monthly_excl_df["reason"].value_counts().to_dict())
    print(f"\nCat 6 -> {cat6}")
    if problems:
        print("G11 CHECK FAILURES:")
        for q in problems:
            print("  -", q)
        sys.exit(1)
    print("G11 integrity checks: PASS")


if __name__ == "__main__":
    main()
