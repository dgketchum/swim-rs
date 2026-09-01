"""Evaluate calibrated SWIM against flux tower ET and OpenET models.

Runs the calibrated model in forecast mode and compares SWIM ET against
energy-balance-corrected flux tower ET (ET_corr) alongside 4 open-source
OpenET models (geeSEBAL, PT-JPL, SSEBop, SIMS) plus their ensemble mean.

OpenET model ET is computed directly from per-model ETf stored in the
container (ETf × ETo), not from external CSV files.

Usage:
    python evaluate.py [--par-csv PATH] [--sites SITE1,SITE2]
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

from swimrs.calibrate.benchmark import (
    BenchmarkConstructionError,
    reconstruct_daily_benchmark,
)
from swimrs.calibrate.flux_utils import (
    full_month_paired_sums,
    passes_site_minimum,
    write_excluded_sites,
)
from swimrs.container import SwimContainer
from swimrs.process.input import build_swim_input
from swimrs.process.loop_fast import run_daily_loop_fast
from swimrs.swim.config import ProjectConfig

OPEN_SOURCE_MODELS = ["geesebal", "ptjpl", "ssebop", "sims", "eemetric", "disalexi"]

EXCLUDED_SITES = {"MB_Pch"}


def apply_exclusions(fids):
    """Filter site list through the canonical exclusion policy."""
    before = len(fids)
    fids = [f for f in fids if f not in EXCLUDED_SITES]
    if before != len(fids):
        dropped = before - len(fids)
        print(f"Exclusion policy: dropped {dropped} site(s) {EXCLUDED_SITES}")
    return fids


VOLK_COLUMN_MAP = {
    "GEESEBAL_3x3": "geesebal",
    "PTJPL_3x3": "ptjpl",
    "SSEBOP_3x3": "ssebop",
    "SIMS_3x3": "sims",
    "EEMETRIC_3x3": "eemetric",
    "DISALEXI_3x3": "disalexi",
}

# May v2.1 is the only sanctioned OpenET capture source; the January
# "openet_flux" directory is rejected on source-version grounds.
OPENET_SOURCE_DIRNAME = "openet_flux_2pt1"
JANUARY_SOURCE_DIRNAME = "openet_flux"

_OPENET_ETO_CACHE = {}


def assert_may_source(path):
    """Hard-fail if a resolved OpenET data path is the January capture set."""
    parts = os.path.normpath(path).split(os.sep)
    if JANUARY_SOURCE_DIRNAME in parts:
        raise BenchmarkConstructionError(
            f"January OpenET source rejected: {path} — use {OPENET_SOURCE_DIRNAME}"
        )
    if OPENET_SOURCE_DIRNAME not in parts:
        raise BenchmarkConstructionError(f"OpenET data path is not the May v2.1 source: {path}")
    return path


def load_openet_eto(data_dir=None):
    """Load the extracted OpenET bias-corrected gridMET ETo (dates x sites).

    This is the sole ETo basis for benchmark reconstruction (identical to the
    run container's meteorology/gridmet/eto_corr; the archived
    site_daily_timeseries `eto` column is raw gridMET and must not be used).
    """
    candidates = []
    if data_dir:
        candidates.append(os.path.join(data_dir, "openet_refet", "openet_eto.csv"))
    candidates.append(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "openet_refet", "openet_eto.csv"
        )
    )
    for path in candidates:
        if os.path.exists(path):
            if path not in _OPENET_ETO_CACHE:
                wide = pd.read_csv(path, index_col="site_id")
                wide.columns = pd.to_datetime(wide.columns, format="%Y%m%d")
                _OPENET_ETO_CACHE[path] = (wide.T.sort_index(), path)
            return _OPENET_ETO_CACHE[path]
    raise FileNotFoundError(f"openet_eto.csv not found in: {candidates}")


def load_config(config_path=None):
    project_dir = Path(__file__).resolve().parent
    conf = Path(config_path) if config_path else project_dir / "5_Flux_Ensemble.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
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
            refet_type=getattr(cfg, "refet_type", "eto") or "eto",
            fields=fids,
            empirical_kc_max=True,
            mask_mode=getattr(cfg, "mask_mode", "irrigation"),
            transpiration_cover_scaling=getattr(cfg, "transpiration_cover_scaling", True),
            transpiration_cover_mode=getattr(cfg, "transpiration_cover_mode", None),
            cover_linear_ndvi_bare=getattr(cfg, "cover_linear_ndvi_bare", None),
            cover_linear_ndvi_full=getattr(cfg, "cover_linear_ndvi_full", None),
            kcb_ndvi_mode=getattr(cfg, "kcb_ndvi_mode", None),
            stress_depletion_fraction=getattr(cfg, "stress_depletion_fraction", None),
        )

        output, _ = run_daily_loop_fast(swim_input)
        dates = pd.date_range(swim_input.start_date, periods=swim_input.n_days, freq="D")
        etr = swim_input.get_time_series("etr")

        results = {}
        for i, fid in enumerate(swim_input.fids):
            results[fid] = pd.DataFrame(
                {
                    "et_act": output.eta[:, i],
                    "etf_model": output.etf[:, i],
                    "etref": etr[:, i],
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


def resolve_flux_dir(cfg):
    """Resolve the flux directory from config, with a shipped-data fallback."""
    if cfg.flux_dir and os.path.isdir(cfg.flux_dir):
        return cfg.flux_dir
    fallback = os.path.join(cfg.data_dir, "flux")
    if os.path.isdir(fallback):
        return fallback
    return cfg.flux_dir or os.path.join(cfg.data_dir, "daily_flux_files")


def load_openet_etf_nomask(container, fid):
    """Load per-model ETf from the container using no_mask (unmasked) data.

    Returns {model_name: pd.Series} of ETf values without irrigation masking.
    """
    etf_by_model = {}
    for model in OPEN_SOURCE_MODELS:
        etf_path = f"remote_sensing/etf/landsat/{model}/no_mask"
        try:
            etf_df = container.query.dataframe(etf_path, fields=[fid])
        except KeyError:
            continue
        if fid in etf_df.columns:
            series = etf_df[fid]
            if series.notna().any():
                etf_by_model[model] = series
    return etf_by_model


def load_volk_openet_et(fid, openet_daily_dir):
    """Load per-model daily ET from Volk OpenET 3x3 extractions.

    These CSVs contain actual ET (mm/day), not ETf fractions.
    Returns {model_name: pd.Series} of sparse ET on Landsat dates.
    """
    path = os.path.join(openet_daily_dir, f"{fid}.csv")
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path, index_col="DATE", parse_dates=True)

    et_by_model = {}
    for raw_col, model_name in VOLK_COLUMN_MAP.items():
        if raw_col in df.columns:
            et_by_model[model_name] = df[raw_col].astype(float)

    if "ensemble_mean_3x3" in df.columns:
        et_by_model["ensemble"] = df["ensemble_mean_3x3"].astype(float)

    return et_by_model


def calc_metrics(obs, mod):
    """Calculate R2, Pearson r, RMSE, bias, KGE between obs and mod arrays."""
    mask = np.isfinite(obs) & np.isfinite(mod)
    obs, mod = obs[mask], mod[mask]
    if len(obs) < 10:
        return {
            "n": len(obs),
            "r2": np.nan,
            "r": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "kge": np.nan,
        }
    r, _ = stats.pearsonr(obs, mod)
    r2 = r2_score(obs, mod)
    rmse = np.sqrt(mean_squared_error(obs, mod))
    bias = float((mod - obs).mean())
    alpha = np.std(mod) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(mod) / np.mean(obs) if np.mean(obs) > 0 else np.nan
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return {"n": len(obs), "r2": r2, "r": r, "rmse": rmse, "bias": bias, "kge": kge}


def _get_diy_openet(container, fid, irr_data, etref):
    """DIY: reconstruct daily ET from our sparse container ETf, ETf-first.

    Uses no_mask ETf exclusively — no fallback to masked ETf. Reconstruction
    goes through the shared helper (Volk ±32-day temporal-support rule, no
    unbounded tail padding).

    Returns {model_name: pd.Series} of daily ET (interpolated ETf × ETo).
    """
    etf_by_model = load_openet_etf_nomask(container, fid)
    et_daily = {}
    for model_name, etf_series in etf_by_model.items():
        recon = reconstruct_daily_benchmark(
            capture_series=etf_series,
            capture_space="etf",
            eto=etref,
            eto_name="model_etref",
            label=f"{fid}:{model_name}",
        )
        et_daily[model_name] = recon.daily_et
    return et_daily


def _get_volk_openet(fid, openet_daily_dir):
    """Volk: load pre-computed daily ET from OpenET 3x3 CSVs.

    Returns {model_name: pd.Series} of sparse ET (already mm/day on Landsat dates).
    """
    return load_volk_openet_et(fid, openet_daily_dir)


def evaluate(cfg, container, par_csv, fids, flux_dir, openet_source="diy"):
    """Run calibrated model and evaluate against flux tower ET and OpenET models.

    Parameters
    ----------
    openet_source : str
        'diy' — use our own ETf extracts from the container (ETf × ETo).
        'volk' — use Volk OpenET 3x3 daily ET extractions (already mm/day).

    Returns DataFrame with per-field metrics for SWIM and each OpenET model.
    """
    fids = apply_exclusions(fids)
    print(f"Evaluating {len(fids)} fields from {par_csv} (openet_source={openet_source})")

    # Load irrigation data from container
    irr_data = {}
    try:
        props = container.query.properties()
        for fid in fids:
            if fid in props and "irr" in props[fid]:
                irr_data[fid] = props[fid]["irr"]
    except Exception:
        pass

    if openet_source == "volk":
        openet_daily_dir = assert_may_source(
            os.path.join(cfg.data_dir, OPENET_SOURCE_DIRNAME, "daily_data")
        )
        openet_eto, openet_eto_path = load_openet_eto(cfg.data_dir)

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

        # Load all OpenET model daily ET on common dates
        if openet_source == "diy":
            et_daily_by_model = _get_diy_openet(container, fid, irr_data, etref)
        else:
            # ETf-first reconstruction on the common OpenET bias-corrected
            # gridMET ETo basis: ETf_i = ET_i/ETo_i at captures, linear-in-time
            # ETf interpolation under the Volk ±32-day rule, × daily ETo.
            # Direct interpolation of sparse ET is invalid (it smooths the
            # daily demand signal) and must never be reintroduced here.
            if fid not in openet_eto.columns:
                raise BenchmarkConstructionError(f"{fid}: no extracted OpenET ETo")
            site_eto = openet_eto[fid].astype("float64")
            et_sparse_by_model = _get_volk_openet(fid, openet_daily_dir)
            et_daily_by_model = {}
            for mn, s in et_sparse_by_model.items():
                if not s.notna().any():
                    et_daily_by_model[mn] = s
                    continue
                recon = reconstruct_daily_benchmark(
                    capture_series=s,
                    capture_space="et",
                    eto=site_eto,
                    eto_name=openet_eto_path,
                    label=f"{fid}:{mn}",
                )
                et_daily_by_model[mn] = recon.daily_et

        # Ensemble ET on common dates
        ens_vals = np.full(len(common), np.nan)
        if "ensemble" in et_daily_by_model:
            ens_vals = et_daily_by_model["ensemble"].reindex(common).values
        else:
            ensemble_source = getattr(cfg, "ensemble_source", "computed")
            if ensemble_source == "openet":
                ens_path = "remote_sensing/etf/landsat/ensemble/no_mask"
                try:
                    ens_df = container.query.dataframe(ens_path, fields=[fid])
                    if fid in ens_df.columns and ens_df[fid].notna().any():
                        recon = reconstruct_daily_benchmark(
                            capture_series=ens_df[fid],
                            capture_space="etf",
                            eto=etref,
                            eto_name="model_etref",
                            label=f"{fid}:ensemble",
                        )
                        ens_vals = recon.daily_et.reindex(common).values
                except KeyError:
                    pass
            else:
                # Computed ensemble: nanmean across available DIY models
                model_arrays = []
                for mn in OPEN_SOURCE_MODELS:
                    if mn in et_daily_by_model:
                        arr = et_daily_by_model[mn].reindex(common).values
                        model_arrays.append(arr)
                if model_arrays:
                    stack = np.column_stack(model_arrays)
                    ens_vals = np.nanmean(stack, axis=1)

        # Paired mask: flux, swim, and ensemble all finite on the same day
        paired_mask = np.isfinite(obs) & np.isfinite(swim_vals) & np.isfinite(ens_vals)
        n_paired = int(paired_mask.sum())

        row = {"fid": fid, "n": n_paired}

        if n_paired >= 10:
            m = calc_metrics(obs[paired_mask], swim_vals[paired_mask])
            for k in ["r2", "r", "rmse", "bias", "kge"]:
                row[f"{k}_swim"] = m[k]

            m = calc_metrics(obs[paired_mask], ens_vals[paired_mask])
            for k in ["r2", "r", "rmse", "bias", "kge"]:
                row[f"{k}_ensemble"] = m[k]
        else:
            for k in ["r2", "r", "rmse", "bias", "kge"]:
                row[f"{k}_swim"] = np.nan
                row[f"{k}_ensemble"] = np.nan

        # Per-model OpenET metrics: each model paired with flux+swim+model on same days.
        # SWIM is re-scored per model so swim-vs-ptjpl uses the same days as ptjpl-vs-flux.
        for model_name in OPEN_SOURCE_MODELS:
            if model_name not in et_daily_by_model:
                for k in ["r2", "r", "rmse", "bias", "kge"]:
                    row[f"{k}_{model_name}"] = np.nan
                    row[f"r2_swim_vs_{model_name}"] = np.nan
                continue

            model_vals = et_daily_by_model[model_name].reindex(common).values
            model_paired = np.isfinite(obs) & np.isfinite(swim_vals) & np.isfinite(model_vals)
            if model_paired.sum() >= 10:
                m = calc_metrics(obs[model_paired], model_vals[model_paired])
                row[f"r2_swim_vs_{model_name}"] = r2_score(
                    obs[model_paired], swim_vals[model_paired]
                )
            else:
                m = {"r2": np.nan, "r": np.nan, "rmse": np.nan, "bias": np.nan, "kge": np.nan}
                row[f"r2_swim_vs_{model_name}"] = np.nan

            for k in ["r2", "r", "rmse", "bias", "kge"]:
                row[f"{k}_{model_name}"] = m[k]

        rows.append(row)

        r2s = row.get("r2_swim", np.nan)
        r2e = row.get("r2_ensemble", np.nan)
        print(f"  {fid}: n_paired={n_paired:>5d}  R2_swim={r2s:.3f}  R2_ens={r2e:.3f}")

    write_excluded_sites(excluded, os.path.join(cfg.project_ws, "results"))

    if not rows:
        print("No fields with sufficient data for evaluation.")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(rows).set_index("fid")

    # Aggregate summary — only sites where both SWIM and ensemble have metrics
    all_models = ["swim"] + OPEN_SOURCE_MODELS + ["ensemble"]
    has_both = metrics_df["r2_swim"].notna() & metrics_df["r2_ensemble"].notna()
    common_df = metrics_df.loc[has_both]

    print("\n" + "=" * 80)
    print(f"PAIRED AGGREGATE ({len(common_df)} fields, SWIM+ensemble on identical days)")
    print("=" * 80)
    header = f"{'model':<12}"
    for stat in ["r2", "r", "rmse", "bias", "kge"]:
        header += f"  {stat + '_mean':>10}  {stat + '_med':>10}"
    print(header)
    print("-" * len(header))

    for model_name in all_models:
        line = f"{model_name:<12}"
        for stat in ["r2", "r", "rmse", "bias", "kge"]:
            col = f"{stat}_{model_name}"
            if col in common_df.columns:
                vals = common_df[col].dropna()
                line += f"  {vals.mean():>10.3f}  {vals.median():>10.3f}"
            else:
                line += f"  {'n/a':>10}  {'n/a':>10}"
        print(line)

    return metrics_df


def evaluate_etf(cfg, container, par_csv, fids):
    """Compare SWIM ETf against OpenET ETf from the container at capture dates.

    Runs the calibrated model and compares predicted ETf directly against
    per-model ETf observations stored in the container (at Landsat overpass
    dates only).  This isolates model skill from ETo conversion issues.

    Returns DataFrame with per-field, per-model ETf metrics.
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

        nomask_etf = load_openet_etf_nomask(container, fid)

        for model in OPEN_SOURCE_MODELS:
            if model not in nomask_etf:
                continue
            combined = nomask_etf[model]

            obs_etf = combined.dropna()
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
            rows.append({"fid": fid, "model": model, **m})

    if not rows:
        print("No fields with sufficient ETf data.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Per-field summary (median across models)
    by_fid = df.groupby("fid").agg(
        n=("n", "sum"),
        r2_median=("r2", "median"),
        rmse_median=("rmse", "median"),
        bias_median=("bias", "median"),
    )

    # Per-model summary
    print("\n" + "=" * 70)
    print("ETf: SWIM vs OpenET (at Landsat capture dates)")
    print("=" * 70)
    header = f"{'model':<12}  {'combos':>6}  {'r2_mean':>8}  {'r2_med':>8}  {'rmse_mean':>10}  {'bias_mean':>10}"
    print(header)
    print("-" * len(header))
    for model in OPEN_SOURCE_MODELS:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        print(
            f"{model:<12}  {len(sub):>6}  {sub['r2'].mean():>8.3f}  "
            f"{sub['r2'].median():>8.3f}  {sub['rmse'].mean():>10.3f}  "
            f"{sub['bias'].mean():>10.3f}"
        )
    print(
        f"{'ALL':<12}  {len(df):>6}  {df['r2'].mean():>8.3f}  "
        f"{df['r2'].median():>8.3f}  {df['rmse'].mean():>10.3f}  "
        f"{df['bias'].mean():>10.3f}"
    )

    # Worst / best fields
    ranked = by_fid.sort_values("r2_median")
    print("\nWorst 10 fields (median R2 across models):")
    for fid, row in ranked.head(10).iterrows():
        print(f"  {fid:<20} R2={row['r2_median']:.3f}  RMSE={row['rmse_median']:.3f}")
    print("\nBest 10 fields:")
    for fid, row in ranked.tail(10).iterrows():
        print(f"  {fid:<20} R2={row['r2_median']:.3f}  RMSE={row['rmse_median']:.3f}")

    # Set fid+model as index so saved CSVs are self-identifying
    df = df.set_index(["fid", "model"])
    return df


def load_volk_monthly_et(fid, monthly_dir):
    """Load Volk OpenET monthly ET totals (mm/month).

    Returns {model_name: pd.Series} indexed by month start date.
    """
    path = os.path.join(monthly_dir, f"{fid}.csv")
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path, index_col="DATE", parse_dates=True)

    et_by_model = {}
    for raw_col, model_name in VOLK_COLUMN_MAP.items():
        if raw_col in df.columns:
            et_by_model[model_name] = df[raw_col].astype(float)

    if "ensemble_mean_3x3" in df.columns:
        et_by_model["ensemble"] = df["ensemble_mean_3x3"].astype(float)

    return et_by_model


def evaluate_monthly(cfg, container, par_csv, fids, flux_dir):
    """Monthly ET comparison with strictly paired months.

    The Volk references are full calendar-month totals, so SWIM is summed over
    full months too, and only months with >= 28 valid daily flux observations
    are kept so the flux total misses at most a few days. SWIM and each OpenET
    model are scored on the exact same months per site. The ensemble defines
    the paired month index — all models share it.
    """
    fids = apply_exclusions(fids)
    monthly_dir = assert_may_source(
        os.path.join(cfg.data_dir, OPENET_SOURCE_DIRNAME, "monthly_data")
    )
    print(f"Monthly evaluation: {len(fids)} fields from {par_csv}")

    calibrated_params = parse_pest_params(par_csv, fids)
    print("Running calibrated model...")
    model_results = run_calibrated_model(cfg, container, fids, calibrated_params)

    all_models = OPEN_SOURCE_MODELS + ["ensemble"]
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

        # Intersect daily indices first, then aggregate to monthly
        daily_common = swim_et.index.intersection(flux_et.index)
        if len(daily_common) < 30:
            print(f"  {fid}: only {len(daily_common)} daily overlap, skipping")
            continue

        flux_daily = flux_et.loc[daily_common]

        # Full-calendar-month totals gated on nearly-complete flux months,
        # matching the full-month Volk reference totals
        swim_monthly, flux_monthly = full_month_paired_sums(swim_et, flux_daily)

        # Load Volk monthly ensemble to define the paired month index
        volk_monthly = load_volk_monthly_et(fid, monthly_dir)
        ens_monthly = volk_monthly.get("ensemble")

        if ens_monthly is not None:
            # Paired months: flux, swim, AND ensemble all finite
            all_idx = flux_monthly.index
            ens_on_idx = ens_monthly.reindex(all_idx)
            paired_mask = (
                flux_monthly.notna() & swim_monthly.reindex(all_idx).notna() & ens_on_idx.notna()
            )
            paired_months = all_idx[paired_mask]
        else:
            # No ensemble data — use flux ∩ swim months
            paired_months = swim_monthly.index.intersection(flux_monthly.index)

        n_paired = len(paired_months)
        if n_paired < 6:
            print(f"  {fid}: only {n_paired} paired months, skipping")
            continue

        obs = flux_monthly.loc[paired_months].values

        row = {"fid": fid, "n": n_paired}
        m = calc_metrics(obs, swim_monthly.reindex(paired_months).values)
        for k in ["r2", "r", "rmse", "bias", "kge"]:
            row[f"{k}_swim"] = m[k]

        # Score each model — per-model paired months (flux + swim + model all finite)
        swim_on_paired = swim_monthly.reindex(paired_months).values
        for model_name in all_models:
            if model_name not in volk_monthly:
                for k in ["r2", "r", "rmse", "bias", "kge"]:
                    row[f"{k}_{model_name}"] = np.nan
                continue

            model_vals = volk_monthly[model_name].reindex(paired_months).values
            model_valid = np.isfinite(model_vals) & np.isfinite(obs) & np.isfinite(swim_on_paired)
            if model_valid.sum() >= 6:
                m = calc_metrics(obs[model_valid], model_vals[model_valid])
            else:
                m = {"r2": np.nan, "r": np.nan, "rmse": np.nan, "bias": np.nan, "kge": np.nan}

            for k in ["r2", "r", "rmse", "bias", "kge"]:
                row[f"{k}_{model_name}"] = m[k]

        rows.append(row)
        r2s = row.get("r2_swim", np.nan)
        r2e = row.get("r2_ensemble", np.nan)
        print(f"  {fid}: n_paired={n_paired:>3d} mo  R2_swim={r2s:.3f}  R2_ens={r2e:.3f}")

    write_excluded_sites(excluded, os.path.join(cfg.project_ws, "results"))

    if not rows:
        print("No fields with sufficient data.")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(rows).set_index("fid")

    # Aggregate
    has_both = metrics_df["r2_swim"].notna() & metrics_df["r2_ensemble"].notna()
    common_df = metrics_df.loc[has_both]

    print("\n" + "=" * 80)
    print(f"PAIRED MONTHLY AGGREGATE ({len(common_df)} fields, identical months)")
    print("=" * 80)
    header = f"{'model':<12}"
    for stat in ["r2", "r", "rmse", "bias", "kge"]:
        header += f"  {stat + '_mean':>10}  {stat + '_med':>10}"
    print(header)
    print("-" * len(header))

    for model_name in ["swim"] + all_models:
        r2_col = f"r2_{model_name}"
        if r2_col not in common_df.columns:
            continue
        vals = common_df[r2_col].dropna()
        if vals.empty:
            continue
        line = f"{model_name:<12}"
        for stat in ["r2", "r", "rmse", "bias", "kge"]:
            col = f"{stat}_{model_name}"
            s = common_df[col].dropna()
            line += f"  {s.mean():>10.3f}  {s.median():>10.3f}"
        print(line)

    return metrics_df


def find_par_csv(results_dir, project_name):
    """Find the latest .par.csv in results directory."""
    for i in range(10, -1, -1):
        candidate = os.path.join(results_dir, f"{project_name}.{i}.par.csv")
        if os.path.exists(candidate):
            return candidate
    return None


def find_reference_par_csv(results_dir, project_name):
    """Resolve the canonical Example 5 parameter file when none is provided."""
    candidate_dirs = []

    run21_dir = os.path.join(results_dir, "run21")
    if os.path.isdir(run21_dir):
        candidate_dirs.append(run21_dir)

    candidate_dirs.append(results_dir)

    if os.path.isdir(results_dir):
        for name in sorted(os.listdir(results_dir)):
            path = os.path.join(results_dir, name)
            if os.path.isdir(path) and path not in candidate_dirs:
                candidate_dirs.append(path)

    for directory in candidate_dirs:
        par_csv = find_par_csv(directory, project_name)
        if par_csv is not None:
            return par_csv

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate calibrated SWIM against flux tower ET and OpenET models"
    )
    parser.add_argument(
        "--par-csv", type=str, default=None, help="Override automatic par.csv discovery"
    )
    parser.add_argument(
        "--sites", type=str, default=None, help="Comma-separated site IDs (default: all)"
    )
    parser.add_argument(
        "--openet-source",
        type=str,
        choices=["diy", "volk"],
        default="volk",
        help="'diy' = our ETf extracts × ETo, 'volk' = OpenET 3x3 daily ET CSVs",
    )
    parser.add_argument(
        "--etf",
        action="store_true",
        help="Compare SWIM ETf vs OpenET ETf at capture dates (instead of ET vs flux)",
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Monthly ET totals: SWIM vs flux vs Volk 3x3 monthly CSVs",
    )
    parser.add_argument(
        "--container",
        type=str,
        default=None,
        help="Override container path (default: derived from config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override config TOML (default: 5_Flux_Ensemble.toml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    flux_dir = resolve_flux_dir(cfg)
    results_dir = os.path.join(cfg.project_ws, "results")

    if args.par_csv:
        par_csv = args.par_csv
    else:
        par_csv = find_reference_par_csv(results_dir, cfg.project_name)
    if par_csv is None:
        raise FileNotFoundError(f"No .par.csv found in {results_dir}")
    print(f"Using parameters: {par_csv}")

    if args.container:
        container_path = args.container
    else:
        container_path = os.path.join(cfg.data_dir, f"{cfg.project_name}_run21.swim")
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
            metrics = evaluate(cfg, container, par_csv, fids, flux_dir, args.openet_source)
            out_csv = os.path.join(results_dir, "evaluation_metrics.csv")
        os.makedirs(results_dir, exist_ok=True)
        metrics.to_csv(out_csv)
        print(f"\nMetrics saved to {out_csv}")
    finally:
        container.close()
