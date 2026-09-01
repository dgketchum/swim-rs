"""Replicate Volk et al. (2024) evaluation methodology with SWIM parameters.

Applies the exact statistical methods from the Nature Water paper:
  - Pooled Pearson r² (single regression across all site-days/months)
  - Daily evaluation restricted to Landsat overpass days only
  - sqrt(n)-weighted per-station MBE, MAE, RMSE
  - Linear regression slope via scipy.stats.linregress

Usage:
    python volk_replication.py
    python volk_replication.py --par-csv /path/to/par.csv --container /path/to/container.swim
"""

import argparse
import os

import numpy as np
import pandas as pd
from evaluate import (
    OPEN_SOURCE_MODELS,
    OPENET_SOURCE_DIRNAME,
    apply_exclusions,
    assert_may_source,
    load_config,
    load_flux_et,
    load_openet_etf_nomask,
    load_volk_monthly_et,
    load_volk_openet_et,
    parse_pest_params,
    resolve_flux_dir,
    run_calibrated_model,
)
from scipy import stats

from swimrs.calibrate.benchmark import reconstruct_daily_benchmark
from swimrs.container import SwimContainer

PAR_CSV_DEFAULT = "/data/ssd1/swim/5_Flux_Ensemble/results/run21/5_Flux_Ensemble.3.par.csv"
CONTAINER_DEFAULT = "/data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble_run21.swim"

MIN_DAILY_OBS = 6
MIN_MONTHLY_OBS = 3


def diy_daily_et(container, fid, etref):
    """DIY daily ET per model via the shared ETf-first reconstruction helper.

    ETf is interpolated under the Volk ±32-day temporal-support rule
    (openet-core semantics) and multiplied by the model's ETo. Unbounded
    ``ETf.interpolate()`` bridges gaps of any length and is invalid — it must
    never be reintroduced here.
    """
    etf_by_model = load_openet_etf_nomask(container, fid)
    et_daily = {}
    for mn, etf_series in etf_by_model.items():
        recon = reconstruct_daily_benchmark(
            capture_series=etf_series,
            capture_space="etf",
            eto=etref,
            eto_name="model_etref",
            label=f"{fid}:{mn}",
        )
        et_daily[mn] = recon.daily_et
    return et_daily


def sqrt_n_weighted_mean(values, counts):
    """Compute sqrt(n)-weighted mean across stations."""
    w = np.sqrt(counts)
    return float(np.sum(values * w) / np.sum(w))


def pooled_stats(all_obs, all_mod):
    """Pooled Pearson r², slope, and intercept from concatenated arrays."""
    mask = np.isfinite(all_obs) & np.isfinite(all_mod)
    obs, mod = all_obs[mask], all_mod[mask]
    if len(obs) < 10:
        return {"r2": np.nan, "slope": np.nan}
    r, _ = stats.pearsonr(obs, mod)
    slope, _, _, _, _ = stats.linregress(obs, mod)
    return {"r2": r**2, "slope": slope}


def pooled_alldays(model_results, fids, flux_dir, container=None):
    """Pooled daily evaluation on ALL days (not just overpass days).

    Evaluates SWIM on all flux-overlap days.  When *container* is provided,
    also evaluates each DIY OpenET model (interpolated ETf × ETo) and their
    ensemble on the same per-model paired day sets.

    Uses the same pooled Pearson r² and sqrt(n)-weighted aggregation as the
    Volk overpass-day and monthly metrics.
    """
    fids = apply_exclusions(fids)

    models_to_eval = OPEN_SOURCE_MODELS + ["ensemble"] if container else []
    all_models = ["swim"] + models_to_eval

    station_stats = {m: [] for m in all_models}
    pooled_obs = {m: [] for m in all_models}
    pooled_mod = {m: [] for m in all_models}

    for fid in fids:
        flux_et = load_flux_et(fid, flux_dir)
        if flux_et.empty:
            continue

        model_df = model_results.get(fid)
        if model_df is None:
            continue
        swim_et = model_df["et_act"]
        etref = model_df["etref"]

        # SWIM: all days with valid flux + SWIM
        common = flux_et.index.intersection(swim_et.index)
        obs = flux_et.loc[common].values
        mod = swim_et.loc[common].values
        mask = np.isfinite(obs) & np.isfinite(mod)
        obs, mod = obs[mask], mod[mask]

        if len(obs) < MIN_DAILY_OBS:
            continue

        mbe = float(np.mean(mod - obs))
        mae = float(np.mean(np.abs(mod - obs)))
        rmse = float(np.sqrt(np.mean((mod - obs) ** 2)))
        station_stats["swim"].append(
            {"fid": fid, "n": len(obs), "mbe": mbe, "mae": mae, "rmse": rmse}
        )
        pooled_obs["swim"].append(obs)
        pooled_mod["swim"].append(mod)

        # DIY OpenET models: ETf-first reconstruction (Volk ±32-day rule)
        if not container:
            continue

        et_daily_by_model = diy_daily_et(container, fid, etref)

        # Computed ensemble from available models
        if et_daily_by_model:
            model_arrays = []
            for mn in OPEN_SOURCE_MODELS:
                if mn in et_daily_by_model:
                    arr = et_daily_by_model[mn].reindex(common[mask]).values
                    model_arrays.append(arr)
            if model_arrays:
                stack = np.column_stack(model_arrays)
                et_daily_by_model["ensemble"] = pd.Series(
                    np.nanmean(stack, axis=1), index=common[mask]
                )

        for model_name in models_to_eval:
            if model_name not in et_daily_by_model:
                continue
            model_vals = et_daily_by_model[model_name].reindex(common[mask]).values
            pair_mask = np.isfinite(model_vals)
            o, m = obs[pair_mask], model_vals[pair_mask]
            if len(o) < MIN_DAILY_OBS:
                continue
            mbe = float(np.mean(m - o))
            mae = float(np.mean(np.abs(m - o)))
            rmse = float(np.sqrt(np.mean((m - o) ** 2)))
            station_stats[model_name].append(
                {"fid": fid, "n": len(o), "mbe": mbe, "mae": mae, "rmse": rmse}
            )
            pooled_obs[model_name].append(o)
            pooled_mod[model_name].append(m)

    # Aggregate and print
    print(f"\n{'=' * 90}")
    print("DAILY — ALL DAYS (pooled, sqrt(n)-weighted, DIY ETf×ETo)")
    print(f"{'=' * 90}")
    header = (
        f"{'Model':<12} {'N sta':>5} {'N days':>7} {'r2':>6} {'Slope':>6}"
        f" {'MBE':>10} {'MAE':>10} {'RMSE':>10}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for model_name in all_models:
        ss = station_stats[model_name]
        if not ss:
            continue
        n_sta = len(ss)
        counts = np.array([s["n"] for s in ss])
        n_days = int(counts.sum())

        all_obs_arr = np.concatenate(pooled_obs[model_name])
        all_mod_arr = np.concatenate(pooled_mod[model_name])
        ps = pooled_stats(all_obs_arr, all_mod_arr)

        mean_obs = float(np.mean(all_obs_arr[np.isfinite(all_obs_arr)]))
        mbe_w = sqrt_n_weighted_mean(np.array([s["mbe"] for s in ss]), counts)
        mae_w = sqrt_n_weighted_mean(np.array([s["mae"] for s in ss]), counts)
        rmse_w = sqrt_n_weighted_mean(np.array([s["rmse"] for s in ss]), counts)

        mbe_pct = 100 * mbe_w / mean_obs if mean_obs else np.nan
        mae_pct = 100 * mae_w / mean_obs if mean_obs else np.nan
        rmse_pct = 100 * rmse_w / mean_obs if mean_obs else np.nan

        label = (
            "SWIM"
            if model_name == "swim"
            else model_name.upper()
            if model_name == "ensemble"
            else model_name
        )
        print(
            f"{label:<12} {n_sta:>5} {n_days:>7} {ps['r2']:>6.3f} {ps['slope']:>6.3f}"
            f" {mbe_w:>+7.3f} ({mbe_pct:>+5.1f}%)"
            f" {mae_w:>7.3f} ({mae_pct:>5.1f}%)"
            f" {rmse_w:>7.3f} ({rmse_pct:>5.1f}%)"
        )

        rows.append(
            {
                "model": model_name,
                "n_stations": n_sta,
                "n_days": n_days,
                "r2_pooled": ps["r2"],
                "slope": ps["slope"],
                "mbe_weighted": mbe_w,
                "mae_weighted": mae_w,
                "rmse_weighted": rmse_w,
                "mbe_pct": mbe_pct,
                "mae_pct": mae_pct,
                "rmse_pct": rmse_pct,
            }
        )

    swim_obs = pooled_obs.get("swim", [])
    if swim_obs:
        mean_obs = float(np.mean(np.concatenate(swim_obs)))
        print(f"\nMean observed flux ET: {mean_obs:.2f} mm/day")

    return pd.DataFrame(rows)


def pooled_monthly_diy(model_results, fids, flux_dir, container):
    """Pooled monthly evaluation using DIY ETf×ETo for OpenET models.

    Sums daily ET to monthly totals (requiring ≥20 valid flux days per month).
    SWIM and each DIY OpenET model are paired on months where both SWIM and
    that model have data alongside flux.  Uses pooled Pearson r² and
    sqrt(n)-weighted aggregation.
    """
    fids = apply_exclusions(fids)
    min_days_per_month = 20

    models_to_eval = OPEN_SOURCE_MODELS + ["ensemble"]
    all_models = ["swim"] + models_to_eval

    station_stats = {m: [] for m in all_models}
    pooled_obs = {m: [] for m in all_models}
    pooled_mod = {m: [] for m in all_models}

    for fid in fids:
        flux_et = load_flux_et(fid, flux_dir)
        if flux_et.empty:
            continue

        model_df = model_results.get(fid)
        if model_df is None:
            continue
        swim_et = model_df["et_act"]
        etref = model_df["etref"]

        # Common daily dates for flux + SWIM
        common = flux_et.index.intersection(swim_et.index)
        flux_daily = flux_et.loc[common]
        swim_daily = swim_et.loc[common]

        # Monthly sums with ≥20-day filter
        flux_monthly = flux_daily.resample("MS").agg(["sum", "count"])
        valid_months = flux_monthly[flux_monthly["count"] >= min_days_per_month].index
        flux_monthly_et = flux_monthly.loc[valid_months, "sum"]

        swim_monthly = swim_daily.resample("MS").agg(["sum", "count"])
        swim_monthly = swim_monthly[swim_monthly["count"] >= min_days_per_month]
        swim_monthly_et = swim_monthly["sum"]

        # SWIM paired months
        swim_common = flux_monthly_et.index.intersection(swim_monthly_et.index)
        obs = flux_monthly_et.loc[swim_common].values
        mod = swim_monthly_et.loc[swim_common].values
        mask = np.isfinite(obs) & np.isfinite(mod)
        obs, mod = obs[mask], mod[mask]
        swim_common = swim_common[mask]

        if len(obs) < MIN_MONTHLY_OBS:
            continue

        mbe = float(np.mean(mod - obs))
        mae = float(np.mean(np.abs(mod - obs)))
        rmse = float(np.sqrt(np.mean((mod - obs) ** 2)))
        station_stats["swim"].append(
            {"fid": fid, "n": len(obs), "mbe": mbe, "mae": mae, "rmse": rmse}
        )
        pooled_obs["swim"].append(obs)
        pooled_mod["swim"].append(mod)

        # DIY OpenET models: ETf-first reconstruction (Volk ±32-day rule)
        et_daily_by_model = diy_daily_et(container, fid, etref)

        # Computed ensemble
        if et_daily_by_model:
            model_arrays = []
            for mn in OPEN_SOURCE_MODELS:
                if mn in et_daily_by_model:
                    arr = et_daily_by_model[mn].reindex(common).values
                    model_arrays.append(arr)
            if model_arrays:
                stack = np.column_stack(model_arrays)
                et_daily_by_model["ensemble"] = pd.Series(np.nanmean(stack, axis=1), index=common)

        for model_name in models_to_eval:
            if model_name not in et_daily_by_model:
                continue
            model_daily = et_daily_by_model[model_name].reindex(common)
            model_monthly = model_daily.resample("MS").agg(["sum", "count"])
            model_monthly = model_monthly[model_monthly["count"] >= min_days_per_month]
            model_monthly_et = model_monthly["sum"]

            paired = flux_monthly_et.index.intersection(swim_monthly_et.index).intersection(
                model_monthly_et.index
            )
            if len(paired) < MIN_MONTHLY_OBS:
                continue

            o = flux_monthly_et.loc[paired].values
            m = model_monthly_et.loc[paired].values
            pm = np.isfinite(o) & np.isfinite(m)
            o, m = o[pm], m[pm]
            if len(o) < MIN_MONTHLY_OBS:
                continue

            mbe = float(np.mean(m - o))
            mae = float(np.mean(np.abs(m - o)))
            rmse = float(np.sqrt(np.mean((m - o) ** 2)))
            station_stats[model_name].append(
                {"fid": fid, "n": len(o), "mbe": mbe, "mae": mae, "rmse": rmse}
            )
            pooled_obs[model_name].append(o)
            pooled_mod[model_name].append(m)

    # Aggregate and print
    print(f"\n{'=' * 100}")
    print("MONTHLY — ALL MONTHS (pooled, sqrt(n)-weighted, DIY ETf×ETo)")
    print(f"{'=' * 100}")
    header = (
        f"{'Model':<12} {'N sta':>5} {'N mo':>6} {'r2':>6} {'Slope':>6}"
        f" {'MBE':>12} {'MAE':>12} {'RMSE':>12}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for model_name in all_models:
        ss = station_stats[model_name]
        if not ss:
            continue
        n_sta = len(ss)
        counts = np.array([s["n"] for s in ss])
        n_months = int(counts.sum())

        all_obs_arr = np.concatenate(pooled_obs[model_name])
        all_mod_arr = np.concatenate(pooled_mod[model_name])
        ps = pooled_stats(all_obs_arr, all_mod_arr)

        mean_obs = float(np.mean(all_obs_arr[np.isfinite(all_obs_arr)]))
        mbe_w = sqrt_n_weighted_mean(np.array([s["mbe"] for s in ss]), counts)
        mae_w = sqrt_n_weighted_mean(np.array([s["mae"] for s in ss]), counts)
        rmse_w = sqrt_n_weighted_mean(np.array([s["rmse"] for s in ss]), counts)

        mbe_pct = 100 * mbe_w / mean_obs if mean_obs else np.nan
        mae_pct = 100 * mae_w / mean_obs if mean_obs else np.nan
        rmse_pct = 100 * rmse_w / mean_obs if mean_obs else np.nan

        label = (
            "SWIM"
            if model_name == "swim"
            else model_name.upper()
            if model_name == "ensemble"
            else model_name
        )
        print(
            f"{label:<12} {n_sta:>5} {n_months:>6} {ps['r2']:>6.3f} {ps['slope']:>6.3f}"
            f" {mbe_w:>+8.2f} ({mbe_pct:>+5.1f}%)"
            f" {mae_w:>8.2f} ({mae_pct:>5.1f}%)"
            f" {rmse_w:>8.2f} ({rmse_pct:>5.1f}%)"
        )

        rows.append(
            {
                "model": model_name,
                "n_stations": n_sta,
                "n_months": n_months,
                "r2_pooled": ps["r2"],
                "slope": ps["slope"],
                "mbe_weighted": mbe_w,
                "mae_weighted": mae_w,
                "rmse_weighted": rmse_w,
                "mbe_pct": mbe_pct,
                "mae_pct": mae_pct,
                "rmse_pct": rmse_pct,
            }
        )

    swim_obs = pooled_obs.get("swim", [])
    if swim_obs:
        mean_obs = float(np.mean(np.concatenate(swim_obs)))
        print(f"\nMean observed flux ET: {mean_obs:.2f} mm/month")

    return pd.DataFrame(rows)


def volk_daily(cfg, container, par_csv, fids, flux_dir):
    """Daily evaluation on overpass days only, using Volk methodology."""
    fids = apply_exclusions(fids)
    openet_daily_dir = assert_may_source(
        os.path.join(cfg.data_dir, OPENET_SOURCE_DIRNAME, "daily_data")
    )

    calibrated_params = parse_pest_params(par_csv, fids)
    print("Running calibrated model...")
    model_results = run_calibrated_model(cfg, container, fids, calibrated_params)

    models_to_eval = OPEN_SOURCE_MODELS + ["ensemble"]
    # Per-station accumulators
    station_stats = {m: [] for m in ["swim"] + models_to_eval}
    # Pooled accumulators
    pooled_obs = {m: [] for m in ["swim"] + models_to_eval}
    pooled_mod = {m: [] for m in ["swim"] + models_to_eval}

    for fid in fids:
        flux_et = load_flux_et(fid, flux_dir)
        if flux_et.empty:
            continue

        model_df = model_results.get(fid)
        if model_df is None:
            continue
        swim_et = model_df["et_act"]

        et_sparse = load_volk_openet_et(fid, openet_daily_dir)
        if not et_sparse:
            continue

        # Overpass dates = dates present in the sparse Volk CSV
        ens_sparse = et_sparse.get("ensemble")
        if ens_sparse is None:
            continue
        overpass_dates = ens_sparse.dropna().index

        # Common overpass dates across flux, SWIM, and ensemble
        common = overpass_dates.intersection(flux_et.index).intersection(swim_et.index)
        obs = flux_et.loc[common].values
        swim_vals = swim_et.loc[common].values
        mask = np.isfinite(obs) & np.isfinite(swim_vals)
        common = common[mask]
        obs = obs[mask]
        swim_vals = swim_vals[mask]

        if len(obs) < MIN_DAILY_OBS:
            continue

        # SWIM station stats
        mbe = float(np.mean(swim_vals - obs))
        mae = float(np.mean(np.abs(swim_vals - obs)))
        rmse = float(np.sqrt(np.mean((swim_vals - obs) ** 2)))
        station_stats["swim"].append(
            {"fid": fid, "n": len(obs), "mbe": mbe, "mae": mae, "rmse": rmse}
        )
        pooled_obs["swim"].append(obs)
        pooled_mod["swim"].append(swim_vals)

        # Per OpenET model station stats
        for model_name in models_to_eval:
            model_series = et_sparse.get(model_name)
            if model_series is None:
                continue
            model_vals = model_series.reindex(common).values
            pair_mask = np.isfinite(model_vals)
            o, m = obs[pair_mask], model_vals[pair_mask]
            if len(o) < MIN_DAILY_OBS:
                continue
            mbe = float(np.mean(m - o))
            mae = float(np.mean(np.abs(m - o)))
            rmse = float(np.sqrt(np.mean((m - o) ** 2)))
            station_stats[model_name].append(
                {"fid": fid, "n": len(o), "mbe": mbe, "mae": mae, "rmse": rmse}
            )
            pooled_obs[model_name].append(o)
            pooled_mod[model_name].append(m)

    # Aggregate
    print(f"\n{'=' * 90}")
    print("DAILY — OVERPASS DAYS ONLY (Volk methodology)")
    print(f"{'=' * 90}")
    header = f"{'Model':<12} {'N sta':>5} {'N days':>7} {'r2':>6} {'Slope':>6} {'MBE':>10} {'MAE':>10} {'RMSE':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for model_name in ["swim"] + models_to_eval:
        ss = station_stats[model_name]
        if not ss:
            continue
        n_sta = len(ss)
        counts = np.array([s["n"] for s in ss])
        n_days = int(counts.sum())

        all_obs = np.concatenate(pooled_obs[model_name])
        all_mod = np.concatenate(pooled_mod[model_name])
        ps = pooled_stats(all_obs, all_mod)

        mean_obs = float(np.mean(all_obs[np.isfinite(all_obs)]))
        mbe_w = sqrt_n_weighted_mean(np.array([s["mbe"] for s in ss]), counts)
        mae_w = sqrt_n_weighted_mean(np.array([s["mae"] for s in ss]), counts)
        rmse_w = sqrt_n_weighted_mean(np.array([s["rmse"] for s in ss]), counts)

        mbe_pct = 100 * mbe_w / mean_obs if mean_obs else np.nan
        mae_pct = 100 * mae_w / mean_obs if mean_obs else np.nan
        rmse_pct = 100 * rmse_w / mean_obs if mean_obs else np.nan

        label = (
            "SWIM"
            if model_name == "swim"
            else model_name.upper()
            if model_name == "ensemble"
            else model_name
        )
        print(
            f"{label:<12} {n_sta:>5} {n_days:>7} {ps['r2']:>6.3f} {ps['slope']:>6.3f}"
            f" {mbe_w:>+7.3f} ({mbe_pct:>+5.1f}%)"
            f" {mae_w:>7.3f} ({mae_pct:>5.1f}%)"
            f" {rmse_w:>7.3f} ({rmse_pct:>5.1f}%)"
        )

        rows.append(
            {
                "model": model_name,
                "n_stations": n_sta,
                "n_days": n_days,
                "r2_pooled": ps["r2"],
                "slope": ps["slope"],
                "mbe_weighted": mbe_w,
                "mae_weighted": mae_w,
                "rmse_weighted": rmse_w,
                "mbe_pct": mbe_pct,
                "mae_pct": mae_pct,
                "rmse_pct": rmse_pct,
            }
        )

    swim_obs = pooled_obs.get("swim", [])
    if swim_obs:
        mean_obs = float(np.mean(np.concatenate(swim_obs)))
        print(f"\nMean observed flux ET: {mean_obs:.2f} mm/day")

    return pd.DataFrame(rows), model_results


def volk_monthly(cfg, container, par_csv, fids, flux_dir):
    """Monthly evaluation using Volk methodology."""
    fids = apply_exclusions(fids)
    monthly_dir = assert_may_source(
        os.path.join(cfg.data_dir, OPENET_SOURCE_DIRNAME, "monthly_data")
    )

    calibrated_params = parse_pest_params(par_csv, fids)
    print("\nRunning calibrated model...")
    model_results = run_calibrated_model(cfg, container, fids, calibrated_params)

    models_to_eval = OPEN_SOURCE_MODELS + ["ensemble"]
    station_stats = {m: [] for m in ["swim"] + models_to_eval}
    pooled_obs = {m: [] for m in ["swim"] + models_to_eval}
    pooled_mod = {m: [] for m in ["swim"] + models_to_eval}

    for fid in fids:
        flux_et = load_flux_et(fid, flux_dir)
        if flux_et.empty:
            continue

        model_df = model_results.get(fid)
        if model_df is None:
            continue

        volk_monthly_et = load_volk_monthly_et(fid, monthly_dir)
        ens_monthly = volk_monthly_et.get("ensemble")
        if ens_monthly is None or ens_monthly.dropna().empty:
            continue

        # Monthly SWIM ET
        swim_daily = model_df["et_act"]
        swim_monthly = swim_daily.resample("MS").agg(["sum", "count"])
        swim_monthly = swim_monthly[swim_monthly["count"] >= 20]
        swim_monthly_et = swim_monthly["sum"]

        # Monthly flux ET
        flux_monthly = flux_et.resample("MS").agg(["sum", "count"])
        flux_monthly = flux_monthly[flux_monthly["count"] >= 20]
        flux_monthly_et = flux_monthly["sum"]

        # Paired months: all of Volk ensemble, SWIM, and flux must be valid
        ens_valid = ens_monthly.dropna().index
        common_months = ens_valid.intersection(swim_monthly_et.index).intersection(
            flux_monthly_et.index
        )

        obs = flux_monthly_et.loc[common_months].values
        swim_vals = swim_monthly_et.loc[common_months].values
        pair_mask = np.isfinite(obs) & np.isfinite(swim_vals)
        common_months = common_months[pair_mask]
        obs = obs[pair_mask]
        swim_vals = swim_vals[pair_mask]

        if len(obs) < MIN_MONTHLY_OBS:
            continue

        mbe = float(np.mean(swim_vals - obs))
        mae = float(np.mean(np.abs(swim_vals - obs)))
        rmse = float(np.sqrt(np.mean((swim_vals - obs) ** 2)))
        station_stats["swim"].append(
            {"fid": fid, "n": len(obs), "mbe": mbe, "mae": mae, "rmse": rmse}
        )
        pooled_obs["swim"].append(obs)
        pooled_mod["swim"].append(swim_vals)

        # Per OpenET model
        for model_name in models_to_eval:
            model_series = volk_monthly_et.get(model_name)
            if model_series is None:
                continue
            model_vals = model_series.reindex(common_months).values
            pm = np.isfinite(model_vals)
            o, m = obs[pm], model_vals[pm]
            if len(o) < MIN_MONTHLY_OBS:
                continue
            mbe = float(np.mean(m - o))
            mae = float(np.mean(np.abs(m - o)))
            rmse = float(np.sqrt(np.mean((m - o) ** 2)))
            station_stats[model_name].append(
                {"fid": fid, "n": len(o), "mbe": mbe, "mae": mae, "rmse": rmse}
            )
            pooled_obs[model_name].append(o)
            pooled_mod[model_name].append(m)

    # Aggregate
    print(f"\n{'=' * 100}")
    print("MONTHLY (Volk methodology)")
    print(f"{'=' * 100}")
    header = f"{'Model':<12} {'N sta':>5} {'N mo':>6} {'r2':>6} {'Slope':>6} {'MBE':>12} {'MAE':>12} {'RMSE':>12}"
    print(header)
    print("-" * len(header))

    rows = []
    for model_name in ["swim"] + models_to_eval:
        ss = station_stats[model_name]
        if not ss:
            continue
        n_sta = len(ss)
        counts = np.array([s["n"] for s in ss])
        n_months = int(counts.sum())

        all_obs = np.concatenate(pooled_obs[model_name])
        all_mod = np.concatenate(pooled_mod[model_name])
        ps = pooled_stats(all_obs, all_mod)

        mean_obs = float(np.mean(all_obs[np.isfinite(all_obs)]))
        mbe_w = sqrt_n_weighted_mean(np.array([s["mbe"] for s in ss]), counts)
        mae_w = sqrt_n_weighted_mean(np.array([s["mae"] for s in ss]), counts)
        rmse_w = sqrt_n_weighted_mean(np.array([s["rmse"] for s in ss]), counts)

        mbe_pct = 100 * mbe_w / mean_obs if mean_obs else np.nan
        mae_pct = 100 * mae_w / mean_obs if mean_obs else np.nan
        rmse_pct = 100 * rmse_w / mean_obs if mean_obs else np.nan

        label = (
            "SWIM"
            if model_name == "swim"
            else model_name.upper()
            if model_name == "ensemble"
            else model_name
        )
        print(
            f"{label:<12} {n_sta:>5} {n_months:>6} {ps['r2']:>6.3f} {ps['slope']:>6.3f}"
            f" {mbe_w:>+8.2f} ({mbe_pct:>+5.1f}%)"
            f" {mae_w:>8.2f} ({mae_pct:>5.1f}%)"
            f" {rmse_w:>8.2f} ({rmse_pct:>5.1f}%)"
        )

        rows.append(
            {
                "model": model_name,
                "n_stations": n_sta,
                "n_months": n_months,
                "r2_pooled": ps["r2"],
                "slope": ps["slope"],
                "mbe_weighted": mbe_w,
                "mae_weighted": mae_w,
                "rmse_weighted": rmse_w,
                "mbe_pct": mbe_pct,
                "mae_pct": mae_pct,
                "rmse_pct": rmse_pct,
            }
        )

    swim_obs = pooled_obs.get("swim", [])
    if swim_obs:
        mean_obs = float(np.mean(np.concatenate(swim_obs)))
        print(f"\nMean observed flux ET: {mean_obs:.2f} mm/month")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--par-csv", type=str, default=PAR_CSV_DEFAULT)
    parser.add_argument("--container", type=str, default=CONTAINER_DEFAULT)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    flux_dir = resolve_flux_dir(cfg)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.par_csv)

    container = SwimContainer.open(args.container, mode="r")
    fids = container.field_uids

    print(f"Parameters: {args.par_csv}")
    print(f"Container:  {args.container}")
    print(f"Flux dir:   {flux_dir}")
    print(f"Output dir: {output_dir}")

    try:
        daily_df, model_results = volk_daily(cfg, container, args.par_csv, fids, flux_dir)
        daily_out = os.path.join(output_dir, "volk_replication_daily.csv")
        daily_df.to_csv(daily_out, index=False)
        print(f"Saved: {daily_out}")

        alldays_df = pooled_alldays(model_results, fids, flux_dir, container=container)
        alldays_out = os.path.join(output_dir, "volk_replication_alldays.csv")
        alldays_df.to_csv(alldays_out, index=False)
        print(f"Saved: {alldays_out}")

        monthly_df = volk_monthly(cfg, container, args.par_csv, fids, flux_dir)
        monthly_out = os.path.join(output_dir, "volk_replication_monthly.csv")
        monthly_df.to_csv(monthly_out, index=False)
        print(f"Saved: {monthly_out}")

        monthly_diy_df = pooled_monthly_diy(model_results, fids, flux_dir, container)
        monthly_diy_out = os.path.join(output_dir, "volk_replication_monthly_diy.csv")
        monthly_diy_df.to_csv(monthly_diy_out, index=False)
        print(f"Saved: {monthly_diy_out}")
    finally:
        container.close()


if __name__ == "__main__":
    main()
