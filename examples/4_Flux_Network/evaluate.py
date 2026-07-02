"""Evaluate calibrated SWIM against flux tower ET and SSEBop NHM.

Runs the calibrated model in forecast mode and compares SWIM ET against
energy-balance-corrected flux tower ET (ET_corr) alongside interpolated
SSEBop NHM ET (ETf × ETo).

Usage:
    python evaluate.py [--par-csv PATH] [--sites SITE1,SITE2]
    python evaluate.py --etf  # compare SWIM ETf vs SSEBop NHM ETf at capture dates
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

from swimrs.calibrate.flux_utils import (
    paired_monthly_sums,
    passes_site_minimum,
    write_excluded_sites,
)
from swimrs.container import SwimContainer
from swimrs.process.input import build_swim_input
from swimrs.process.loop_fast import run_daily_loop_fast
from swimrs.swim.config import ProjectConfig

# Canonical exclusion list — sites with known data quality issues that should
# not appear in any comparative evaluation. Keep this list general so new
# exclusions can be added without ad hoc filters elsewhere.
EXCLUDED_SITES = {"MB_Pch"}


def apply_exclusions(fids):
    """Filter site list through the canonical exclusion policy."""
    before = len(fids)
    fids = [f for f in fids if f not in EXCLUDED_SITES]
    if before != len(fids):
        dropped = before - len(fids)
        print(
            f"Exclusion policy: dropped {dropped} site(s) {EXCLUDED_SITES & set(fids) or EXCLUDED_SITES}"
        )
    return fids


def load_config():
    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "4_Flux_Network.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd1/swim"):
        cfg.read_config(str(conf), calibrate=True)
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent), calibrate=True)
    return cfg


def parse_pest_params(par_csv, fids):
    """Parse PEST++ .par.csv into {fid: {param: value}} using median realization."""
    df = pd.read_csv(par_csv, index_col=0)

    numeric_rows = df.loc[df.index != "base"]
    row = numeric_rows.median()

    params_by_fid = {}
    for col in df.columns:
        parts = col.split("_ptype:")[0]
        parts = parts.replace("pname:p_", "")
        parts = parts.rsplit("_:0", 1)[0]

        matched_fid = None
        for fid in fids:
            if parts.lower().endswith(f"_{fid.lower()}"):
                matched_fid = fid
                param_name = parts[: -(len(fid) + 1)]
                break

        if matched_fid:
            if matched_fid not in params_by_fid:
                params_by_fid[matched_fid] = {}
            params_by_fid[matched_fid][param_name] = float(row[col])

    return params_by_fid


def run_calibrated_model(cfg, container, fids, calibrated_params):
    """Run model with calibrated parameters. Returns {fid: DataFrame}."""
    refet_type = (getattr(cfg, "refet_type", "eto") or "eto").lower()

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        temp_h5 = tmp.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(calibrated_params, tmp)
        params_json = tmp.name

    try:
        swim_input = build_swim_input(
            container,
            output_h5=temp_h5,
            calibrated_params_path=params_json,
            start_date=cfg.start_dt,
            end_date=cfg.end_dt,
            refet_type=refet_type,
            fields=fids,
            empirical_kc_max=True,
            mask_mode=getattr(cfg, "mask_mode", "irrigation"),
        )

        output, _ = run_daily_loop_fast(swim_input)
        dates = pd.date_range(swim_input.start_date, periods=swim_input.n_days, freq="D")
        etref = swim_input.get_time_series(refet_type)

        results = {}
        for i, fid in enumerate(swim_input.fids):
            results[fid] = pd.DataFrame(
                {
                    "et_act": output.eta[:, i],
                    "etf_model": output.etf[:, i],
                    "etref": etref[:, i],
                    "swe": output.swe[:, i],
                },
                index=dates,
            )

        swim_input.close()
    finally:
        for p in [temp_h5, params_json]:
            if os.path.exists(p):
                os.remove(p)

    return results


def load_flux_et(fid, flux_dir):
    """Load energy-balance-corrected ET from flux tower data."""
    path = os.path.join(flux_dir, f"{fid}_daily_data.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    if "ET_corr" in df.columns:
        return df["ET_corr"]
    return pd.Series(dtype=float)


def load_ssebop_etf(container, fid):
    """Load SSEBop NHM ETf from the no_mask path (full footprint)."""
    etf_path = "remote_sensing/etf/landsat/ssebop/no_mask"
    try:
        etf_df = container.query.dataframe(etf_path, fields=[fid])
        if fid in etf_df.columns:
            return etf_df[fid]
    except Exception:
        pass
    return None


def calc_metrics(obs, mod):
    """Calculate R2, Pearson r, RMSE, bias between obs and mod arrays."""
    mask = np.isfinite(obs) & np.isfinite(mod)
    obs, mod = obs[mask], mod[mask]
    if len(obs) < 10:
        return {"n": len(obs), "r2": np.nan, "r": np.nan, "rmse": np.nan, "bias": np.nan}
    r, _ = stats.pearsonr(obs, mod)
    r2 = r2_score(obs, mod)
    rmse = np.sqrt(mean_squared_error(obs, mod))
    bias = float((mod - obs).mean())
    return {"n": len(obs), "r2": r2, "r": r, "rmse": rmse, "bias": bias}


def evaluate(cfg, container, par_csv, fids, flux_dir):
    """Run calibrated model and evaluate against flux tower ET and SSEBop NHM.

    Both SWIM and SSEBop are scored on the exact same set of days per site
    (paired evaluation). Days where either model or flux is NaN are excluded
    from both scores.

    Returns DataFrame with per-field metrics for SWIM and SSEBop NHM.
    """
    fids = apply_exclusions(fids)
    print(f"Evaluating {len(fids)} fields from {par_csv}")

    calibrated_params = parse_pest_params(par_csv, fids)
    missing = [f for f in fids if f not in calibrated_params]
    if missing:
        print(f"WARNING: No calibrated params for: {missing}")

    print("Running calibrated model...")
    model_results = run_calibrated_model(cfg, container, fids, calibrated_params)

    rows = []
    excluded = []
    for fid in fids:
        flux_et = load_flux_et(fid, flux_dir)
        if flux_et.empty:
            print(f"  {fid}: no flux data, skipping")
            excluded.append({"site": fid, "reason": "no_flux_data"})
            continue
        if not passes_site_minimum(flux_et):
            print(f"  {fid}: below VALIDATION_POLICY site minimum (90 valid days / 3 months)")
            excluded.append({"site": fid, "reason": "below_site_minimum_90d_3mo"})
            continue

        model_df = model_results[fid]
        swim_et = model_df["et_act"]
        etref = model_df["etref"]

        # Common dates between model and flux
        common = swim_et.index.intersection(flux_et.index)
        if len(common) < 10:
            print(f"  {fid}: only {len(common)} overlapping days, skipping")
            continue

        obs = flux_et.loc[common].values
        swim_vals = swim_et.loc[common].values

        # SSEBop NHM ET (interpolated ETf × ETo)
        etf_series = load_ssebop_etf(container, fid)
        if etf_series is not None:
            etf_interp = etf_series.interpolate(method="linear")
            ssebop_et = etf_interp * etref
            ssebop_vals = ssebop_et.reindex(common).values
        else:
            ssebop_vals = np.full(len(common), np.nan)

        # Paired mask: all three must be finite on the same day
        paired_mask = np.isfinite(obs) & np.isfinite(swim_vals) & np.isfinite(ssebop_vals)
        n_paired = int(paired_mask.sum())

        row = {"fid": fid, "n": n_paired}

        if n_paired >= 10:
            m = calc_metrics(obs[paired_mask], swim_vals[paired_mask])
            for k in ["r2", "r", "rmse", "bias"]:
                row[f"{k}_swim"] = m[k]

            m = calc_metrics(obs[paired_mask], ssebop_vals[paired_mask])
            for k in ["r2", "r", "rmse", "bias"]:
                row[f"{k}_ssebop"] = m[k]
        else:
            for k in ["r2", "r", "rmse", "bias"]:
                row[f"{k}_swim"] = np.nan
                row[f"{k}_ssebop"] = np.nan

        rows.append(row)

        r2s = row.get("r2_swim", np.nan)
        r2b = row.get("r2_ssebop", np.nan)
        print(f"  {fid}: n_paired={n_paired:>5d}  R2_swim={r2s:.3f}  R2_ssebop={r2b:.3f}")

    write_excluded_sites(excluded, os.path.join(cfg.project_ws, "results"))

    if not rows:
        print("No fields with sufficient data for evaluation.")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(rows).set_index("fid")

    # Aggregate summary (only sites with finite paired metrics)
    has_both = metrics_df["r2_swim"].notna() & metrics_df["r2_ssebop"].notna()
    common_df = metrics_df.loc[has_both]

    print("\n" + "=" * 80)
    print(f"PAIRED AGGREGATE ({len(common_df)} fields, both models on identical days)")
    print("=" * 80)
    header = f"{'model':<12}"
    for stat in ["r2", "r", "rmse", "bias"]:
        header += f"  {stat + '_mean':>10}  {stat + '_med':>10}"
    print(header)
    print("-" * len(header))

    for model_name in ["swim", "ssebop"]:
        line = f"{model_name:<12}"
        for stat in ["r2", "r", "rmse", "bias"]:
            col = f"{stat}_{model_name}"
            if col in common_df.columns:
                vals = common_df[col].dropna()
                line += f"  {vals.mean():>10.3f}  {vals.median():>10.3f}"
            else:
                line += f"  {'n/a':>10}  {'n/a':>10}"
        print(line)

    return metrics_df


def evaluate_monthly(cfg, container, par_csv, fids, flux_dir):
    """Monthly aggregation of ET evaluation with strictly paired months.

    Intersects daily indices first, restricts every series to flux-valid days,
    then aggregates to monthly sums so all sides integrate the identical day
    set. Only months with at least 20 valid daily flux observations are kept.
    Both SWIM and SSEBop are scored on the exact same set of months per site.
    """
    fids = apply_exclusions(fids)
    print(f"Monthly evaluation: {len(fids)} fields from {par_csv}")

    calibrated_params = parse_pest_params(par_csv, fids)
    missing = [f for f in fids if f not in calibrated_params]
    if missing:
        print(f"WARNING: No calibrated params for: {missing}")

    print("Running calibrated model...")
    model_results = run_calibrated_model(cfg, container, fids, calibrated_params)

    rows = []
    excluded = []
    for fid in fids:
        flux_et = load_flux_et(fid, flux_dir)
        if flux_et.empty:
            excluded.append({"site": fid, "reason": "no_flux_data"})
            continue
        if not passes_site_minimum(flux_et):
            print(f"  {fid}: below VALIDATION_POLICY site minimum (90 valid days / 3 months)")
            excluded.append({"site": fid, "reason": "below_site_minimum_90d_3mo"})
            continue

        model_df = model_results[fid]
        swim_et = model_df["et_act"]
        etref = model_df["etref"]

        # Intersect daily indices first, then aggregate to monthly
        daily_common = swim_et.index.intersection(flux_et.index)
        if len(daily_common) < 30:
            continue

        swim_daily = swim_et.loc[daily_common]
        flux_daily = flux_et.loc[daily_common]

        # SSEBop NHM daily ET (interpolated ETf × ETo) on the same daily index
        etf_series = load_ssebop_etf(container, fid)
        if etf_series is not None:
            etf_interp = etf_series.reindex(daily_common).interpolate(method="linear")
            ssebop_daily = etf_interp * etref.reindex(daily_common)
        else:
            ssebop_daily = None

        # Monthly totals over flux-valid days only, so every series integrates
        # the identical day set; SSEBop months not finite on every valid day
        # become NaN instead of partial or fabricated-zero sums
        swim_monthly, flux_monthly, ssebop_monthly = paired_monthly_sums(
            swim_daily, flux_daily, ssebop_daily
        )
        if ssebop_monthly is None:
            ssebop_monthly = pd.Series(np.nan, index=swim_monthly.index)

        # Strictly paired months: flux, swim, and ssebop all finite
        all_idx = flux_monthly.index
        ssebop_on_idx = ssebop_monthly.reindex(all_idx)
        paired_mask = (
            flux_monthly.notna() & swim_monthly.reindex(all_idx).notna() & ssebop_on_idx.notna()
        )
        paired_months = all_idx[paired_mask]
        n_paired = len(paired_months)

        if n_paired < 6:
            continue

        obs = flux_monthly.loc[paired_months].values
        row = {"fid": fid, "n_months": n_paired}

        m = calc_metrics(obs, swim_monthly.reindex(paired_months).values)
        for k in ["r2", "r", "rmse", "bias"]:
            row[f"{k}_swim"] = m[k]

        m = calc_metrics(obs, ssebop_on_idx.loc[paired_months].values)
        for k in ["r2", "r", "rmse", "bias"]:
            row[f"{k}_ssebop"] = m[k]

        rows.append(row)
        print(
            f"  {fid}: n_months_paired={n_paired:>4d}  "
            f"R2_swim={row['r2_swim']:.3f}  R2_ssebop={row['r2_ssebop']:.3f}  "
            f"RMSE_swim={row['rmse_swim']:.2f}  RMSE_ssebop={row['rmse_ssebop']:.2f}"
        )

    write_excluded_sites(excluded, os.path.join(cfg.project_ws, "results"))

    if not rows:
        print("No fields with sufficient monthly data.")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(rows).set_index("fid")

    print("\n" + "=" * 80)
    print(f"PAIRED MONTHLY AGGREGATE ({len(metrics_df)} fields, identical months)")
    print("=" * 80)
    header = f"{'model':<12}"
    for stat in ["r2", "r", "rmse", "bias"]:
        header += f"  {stat + '_mean':>10}  {stat + '_med':>10}"
    print(header)
    print("-" * len(header))

    for model_name in ["swim", "ssebop"]:
        line = f"{model_name:<12}"
        for stat in ["r2", "r", "rmse", "bias"]:
            col = f"{stat}_{model_name}"
            if col in metrics_df.columns:
                vals = metrics_df[col].dropna()
                line += f"  {vals.mean():>10.3f}  {vals.median():>10.3f}"
            else:
                line += f"  {'n/a':>10}  {'n/a':>10}"
        print(line)

    return metrics_df


def evaluate_etf(cfg, container, par_csv, fids):
    """Compare SWIM ETf against SSEBop NHM ETf at Landsat capture dates.

    Isolates model skill from ETo conversion issues by comparing ETf directly.

    Returns DataFrame with per-field ETf metrics.
    """
    fids = apply_exclusions(fids)
    print(f"ETf evaluation: {len(fids)} fields from {par_csv}")

    calibrated_params = parse_pest_params(par_csv, fids)
    model_results = run_calibrated_model(cfg, container, fids, calibrated_params)

    rows = []
    for fid in fids:
        if fid not in model_results:
            continue
        swim_etf = model_results[fid]["etf_model"]

        etf_series = load_ssebop_etf(container, fid)
        if etf_series is None:
            continue

        obs_etf = etf_series.dropna()
        obs_etf = obs_etf[obs_etf > 0]
        if len(obs_etf) < 10:
            continue

        common = swim_etf.index.intersection(obs_etf.index)
        if len(common) < 10:
            continue

        s = swim_etf.loc[common].values
        o = obs_etf.loc[common].values
        valid = np.isfinite(s) & np.isfinite(o)
        s, o = s[valid], o[valid]
        if len(s) < 10:
            continue

        m = calc_metrics(o, s)
        rows.append({"fid": fid, **m})

    if not rows:
        print("No fields with sufficient ETf data.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("fid")

    print("\n" + "=" * 70)
    print("ETf: SWIM vs SSEBop NHM (at Landsat capture dates)")
    print("=" * 70)
    print(
        f"  Fields: {len(df)}  "
        f"R2_mean={df['r2'].mean():.3f}  R2_med={df['r2'].median():.3f}  "
        f"RMSE_mean={df['rmse'].mean():.3f}  bias_mean={df['bias'].mean():.3f}"
    )

    # Worst / best fields
    ranked = df.sort_values("r2")
    print("\nWorst 10 fields:")
    for fid, row in ranked.head(10).iterrows():
        print(f"  {fid:<20} R2={row['r2']:.3f}  RMSE={row['rmse']:.3f}")
    print("\nBest 10 fields:")
    for fid, row in ranked.tail(10).iterrows():
        print(f"  {fid:<20} R2={row['r2']:.3f}  RMSE={row['rmse']:.3f}")

    return df


def find_par_csv(results_dir, project_name):
    """Find the latest .par.csv in results directory."""
    for i in range(10, -1, -1):
        candidate = os.path.join(results_dir, f"{project_name}.{i}.par.csv")
        if os.path.exists(candidate):
            return candidate
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate calibrated SWIM against flux tower ET and SSEBop NHM"
    )
    parser.add_argument(
        "--par-csv", type=str, default=None, help="Override automatic par.csv discovery"
    )
    parser.add_argument(
        "--sites", type=str, default=None, help="Comma-separated site IDs (default: all)"
    )
    parser.add_argument(
        "--etf",
        action="store_true",
        help="Compare SWIM ETf vs SSEBop NHM ETf at capture dates (instead of ET vs flux)",
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Evaluate at monthly time step (sum ET, mean ETf)",
    )
    parser.add_argument(
        "--container",
        type=str,
        default=None,
        help="Override container path (default: derived from config)",
    )
    args = parser.parse_args()

    cfg = load_config()
    flux_dir = cfg.flux_dir
    results_dir = os.path.join(cfg.project_ws, "results")

    if args.par_csv:
        par_csv = args.par_csv
    else:
        par_csv = find_par_csv(results_dir, cfg.project_name)
    if par_csv is None:
        raise FileNotFoundError(f"No .par.csv found in {results_dir}")
    print(f"Using parameters: {par_csv}")

    if args.container:
        container_path = args.container
    else:
        container_path = os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")
    container = SwimContainer.open(container_path, mode="r")

    if args.sites:
        fids = [s.strip() for s in args.sites.split(",")]
    else:
        fids = container.field_uids
    fids = apply_exclusions(fids)

    try:
        if args.monthly:
            metrics = evaluate_monthly(cfg, container, par_csv, fids, flux_dir)
            out_csv = os.path.join(results_dir, "evaluation_monthly_metrics.csv")
        elif args.etf:
            metrics = evaluate_etf(cfg, container, par_csv, fids)
            out_csv = os.path.join(results_dir, "evaluation_etf_metrics.csv")
        else:
            metrics = evaluate(cfg, container, par_csv, fids, flux_dir)
            out_csv = os.path.join(results_dir, "evaluation_metrics.csv")
        os.makedirs(results_dir, exist_ok=True)
        metrics.to_csv(out_csv)
        print(f"\nMetrics saved to {out_csv}")
    finally:
        container.close()
