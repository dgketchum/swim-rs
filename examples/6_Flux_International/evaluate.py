"""Evaluate calibrated SWIM against flux tower ET for international sites.

Runs the calibrated model and compares SWIM ET against energy-balance-corrected
flux tower ET (ET_corr) from the multi-network QAQC archive.

Key differences from CONUS examples:
    - ERA5-Land meteorology (not GridMET)
    - HWSD soils (not SSURGO)
    - No irrigation masking (mask_mode="none")
    - No OpenET reference models (international sites lack coverage)
    - Multi-network flux data search (ameriflux, fluxnet, icos, ozflux)

Usage:
    python evaluate.py [--config PATH] [--par-csv PATH] [--sites SITE1,SITE2,...] [--monthly]
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

QAQC_ROOT = "/nas/climate/flux_stations/qaqc"
QAQC_NETWORKS = ["ameriflux", "fluxnet", "icos", "ozflux"]


def _load_config(conf_path: Path | None = None):
    project_dir = Path(__file__).resolve().parent
    conf = conf_path if conf_path is not None else project_dir / "6_Flux_International.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd1/swim"):
        cfg.read_config(str(conf), calibrate=True)
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent), calibrate=True)
    return cfg


def _results_dir(cfg: ProjectConfig, conf_path: Path | None = None) -> str:
    base = os.path.join(cfg.project_ws, "results")
    if conf_path is None:
        return base
    return os.path.join(base, conf_path.stem)


def _group_results_dir(cfg: ProjectConfig, conf_path: Path | None = None) -> str:
    return os.path.join(_results_dir(cfg, conf_path), "group")


def _default_container_path(cfg: ProjectConfig) -> str:
    return getattr(cfg, "container_path", None) or os.path.join(
        cfg.data_dir, f"{cfg.project_name}.swim"
    )


def _query_etf_series(container: SwimContainer, path: str, fid: str) -> pd.Series | None:
    try:
        obs_df = container.query.dataframe(path, fields=[fid])
    except KeyError:
        return None
    except Exception:
        return None

    if fid not in obs_df.columns:
        return None

    series = obs_df[fid]
    if not series.notna().any():
        return None

    return series


def _load_target_etf_series(
    container: SwimContainer, cfg: ProjectConfig, fid: str
) -> pd.Series | None:
    mask = "no_mask" if getattr(cfg, "mask_mode", "none") == "none" else "irr"
    instrument = getattr(cfg, "etf_target_instrument", "landsat")
    etf_model = getattr(cfg, "etf_target_model", "ptjpl")

    if etf_model == "ensemble":
        direct_path = f"remote_sensing/etf/{instrument}/ensemble/{mask}"
        direct_series = _query_etf_series(container, direct_path, fid)
        if direct_series is not None:
            return direct_series

        member_names = list(getattr(cfg, "etf_ensemble_members", None) or [])
        member_series = []
        for model_name in member_names:
            path = f"remote_sensing/etf/{instrument}/{model_name}/{mask}"
            series = _query_etf_series(container, path, fid)
            if series is not None:
                member_series.append(series.rename(model_name))

        if member_series:
            return pd.concat(member_series, axis=1).mean(axis=1, skipna=True)

        return None

    path = f"remote_sensing/etf/{instrument}/{etf_model}/{mask}"
    return _query_etf_series(container, path, fid)


def _build_rs_eta_series(
    container: SwimContainer, cfg: ProjectConfig, fid: str, etref: pd.Series
) -> pd.Series | None:
    """Build daily RS ETa from native ETf, linear interpolation, and ETo."""
    etf_series = _load_target_etf_series(container, cfg, fid)
    if etf_series is None:
        return None

    daily_etf = etf_series.reindex(etref.index).interpolate(method="linear")
    rs_eta = daily_etf * etref.reindex(daily_etf.index)
    if not rs_eta.notna().any():
        return None

    return rs_eta


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


def _load_container_calibrated_params(
    container: SwimContainer, fids: list[str]
) -> dict[str, dict[str, float]]:
    """Load ingested calibration parameters from the container for selected fields."""
    from swimrs.container.components.ingestor import CALIBRATION_PARAMS

    root = container._root
    if "calibration/parameters" not in root:
        return {}

    missing_groups = [
        param for param in CALIBRATION_PARAMS if f"calibration/parameters/{param}" not in root
    ]
    if missing_groups:
        raise ValueError(
            "Container calibration is incomplete; missing parameter groups: "
            + ", ".join(sorted(missing_groups))
        )

    uid_to_idx = {uid: i for i, uid in enumerate(container.field_uids)}
    calibrated_mask = None
    if "calibration/metadata/calibrated" in root:
        calibrated_mask = np.asarray(root["calibration/metadata/calibrated"][:]).astype(bool)

    param_arrays = {
        param: np.asarray(root[f"calibration/parameters/{param}"][:], dtype=float)
        for param in CALIBRATION_PARAMS
    }

    params_by_fid = {}
    for fid in fids:
        idx = uid_to_idx.get(fid)
        if idx is None:
            continue
        if calibrated_mask is not None and (
            idx >= len(calibrated_mask) or not calibrated_mask[idx]
        ):
            continue

        field_params = {}
        missing_value = False
        for param, arr in param_arrays.items():
            if idx >= len(arr) or not np.isfinite(arr[idx]):
                missing_value = True
                break
            field_params[param] = float(arr[idx])

        if not missing_value:
            params_by_fid[fid] = field_params

    return params_by_fid


def _resolve_calibrated_params(
    container: SwimContainer, fids: list[str], par_csv: str | None = None
) -> tuple[dict[str, dict[str, float]], str]:
    """Resolve calibrated parameters from par.csv first, then container state."""
    if par_csv is not None:
        return parse_pest_params(par_csv, fids), str(par_csv)

    params_by_fid = _load_container_calibrated_params(container, fids)
    if not params_by_fid:
        raise FileNotFoundError(
            "No .par.csv found and container has no complete ingested calibration parameters "
            "for the requested sites."
        )

    return params_by_fid, f"container calibration ({container.path})"


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
            etf_model=getattr(cfg, "etf_target_model", "ptjpl"),
            met_source=getattr(cfg, "met_source", "era5"),
            fields=fids,
            empirical_kc_max=True,
            mask_mode=getattr(cfg, "mask_mode", "none"),
        )

        output, _ = run_daily_loop_fast(swim_input)
        dates = pd.date_range(swim_input.start_date, periods=swim_input.n_days, freq="D")
        eto = swim_input.get_time_series("eto")

        results = {}
        for i, fid in enumerate(swim_input.fids):
            results[fid] = pd.DataFrame(
                {
                    "et_act": output.eta[:, i],
                    "etf_model": output.etf[:, i],
                    "etref": eto[:, i],
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


def find_flux_file(site_id, network=None):
    """Find QAQC flux data file, preferring the declared network.

    Without a declared network, fall back to the fixed multi-network search
    order (first existing file wins).
    """
    fname = f"{site_id}_daily_data.csv"
    networks = [network] if network else QAQC_NETWORKS
    for net in networks:
        path = os.path.join(QAQC_ROOT, net, fname)
        if os.path.exists(path):
            return path
    return None


def load_flux_et(fid, source=None):
    """Load flux tower ET, honoring a declared (network, et_col) source.

    ``source`` is the (network, et_col) pair the cohort shapefile records for
    the site (see ``load_flux_sources``). Without it, the first file found in
    the network search order is used with the default column priority — which
    can silently pick a file with no ET column at all (e.g. the US-Ne sites'
    soil-moisture-only ameriflux files shadowing their fluxnet ET records).
    """
    network = et_col = None
    if source is not None:
        network, et_col = source
    path = find_flux_file(fid, network)
    if path is None:
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    # honor the shapefile-declared column first, then fall back
    for col in ([et_col] if et_col else []) + ["ET_corr", "ET", "LE_corr"]:
        if col in df.columns:
            series = df[col] if col != "LE_corr" else df[col] * 86400 / 2.45e6
            series.index = series.index.normalize()
            series.attrs["flux_file"] = path
            series.attrs["et_col"] = col
            return series
    return pd.Series(dtype=float)


def load_flux_sources(shapefile, id_col="sid"):
    """Per-site (network, et_col) declared in the cohort shapefile.

    Returns ``{site_id: (network, et_col)}``; empty when the shapefile lacks a
    ``network`` column, so callers fall back to the network search order.
    """
    import geopandas as gpd

    gdf = gpd.read_file(shapefile, engine="fiona")
    if id_col not in gdf.columns:
        id_col = "sid"
    if "network" not in gdf.columns:
        return {}
    sources = {}
    for _, r in gdf.iterrows():
        network = r["network"] if isinstance(r["network"], str) else None
        et_col = r["et_col"] if "et_col" in gdf.columns and isinstance(r["et_col"], str) else None
        sources[str(r[id_col])] = (network, et_col)
    return sources


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


def find_par_csv(results_dir, project_name):
    """Find the latest .par.csv in results directory."""
    for i in range(10, -1, -1):
        candidate = os.path.join(results_dir, f"{project_name}.{i}.par.csv")
        if os.path.exists(candidate):
            return candidate
    return None


def evaluate(cfg, container, par_csv, fids, results_dir=None, flux_sources=None):
    """Run calibrated model and evaluate against flux and RS-derived daily ETa.

    The RS benchmark is the native target ETf, linearly interpolated to daily,
    then multiplied by daily ETo. SWIM and RS ETa are both scored against flux
    on identical paired days. ``flux_sources`` maps site id to the shapefile-
    declared (network, et_col) truth source (see ``load_flux_sources``).

    Returns DataFrame with per-site paired metrics.
    """
    flux_sources = flux_sources or {}
    calibrated_params, param_source = _resolve_calibrated_params(container, fids, par_csv)
    print(f"Evaluating {len(fids)} sites from {param_source}")

    missing = [f for f in fids if f not in calibrated_params]
    if missing:
        print(f"WARNING: No calibrated params for: {missing} — skipping these sites")
        fids = [f for f in fids if f in calibrated_params]
    if not fids:
        print("No sites with calibrated parameters.")
        return pd.DataFrame()

    print("Running calibrated model...")
    model_results = run_calibrated_model(cfg, container, fids, calibrated_params)
    rs_eta_by_fid = {
        fid: _build_rs_eta_series(container, cfg, fid, df["etref"])
        for fid, df in model_results.items()
    }

    # Write per-site CSVs for publication_figures.py
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        for fid, df in model_results.items():
            out_df = df.copy()
            rs_eta = rs_eta_by_fid.get(fid)
            if rs_eta is not None:
                out_df["et_rs"] = rs_eta.reindex(out_df.index)
            out_df.to_csv(os.path.join(results_dir, f"{fid}.csv"))

    rows = []
    excluded = []
    for fid in fids:
        flux_et = load_flux_et(fid, flux_sources.get(fid))
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
        rs_eta = rs_eta_by_fid.get(fid)
        if rs_eta is None:
            print(f"  {fid}: no RS ETa benchmark, skipping")
            continue

        common = swim_et.index.intersection(flux_et.index)
        if len(common) < 10:
            print(f"  {fid}: only {len(common)} overlapping days, skipping")
            continue

        obs = flux_et.loc[common].values
        swim_vals = swim_et.loc[common].values
        rs_vals = rs_eta.reindex(common).values
        paired_mask = np.isfinite(obs) & np.isfinite(swim_vals) & np.isfinite(rs_vals)
        n_paired = int(paired_mask.sum())

        if n_paired < 10:
            print(f"  {fid}: only {n_paired} paired days with RS ETa benchmark, skipping")
            continue

        m_swim = calc_metrics(obs[paired_mask], swim_vals[paired_mask])
        m_rs = calc_metrics(obs[paired_mask], rs_vals[paired_mask])

        # record the truth source actually used (RUN_POLICY category-6 audit)
        flux_file = flux_et.attrs.get("flux_file", "")
        row = {
            "fid": fid,
            "n": n_paired,
            "flux_network": os.path.basename(os.path.dirname(flux_file)) if flux_file else None,
            "flux_et_col": flux_et.attrs.get("et_col"),
        }
        for k in ["r2", "r", "rmse", "bias", "kge"]:
            row[k] = m_swim[k]
            row[f"{k}_swim"] = m_swim[k]
            row[f"{k}_rs"] = m_rs[k]
        rows.append(row)

        print(
            f"  {fid}: n_paired={n_paired:>5d}  R2_swim={m_swim['r2']:.3f}  "
            f"R2_rs={m_rs['r2']:.3f}  KGE_swim={m_swim['kge']:.3f}  "
            f"KGE_rs={m_rs['kge']:.3f}"
        )

    if results_dir is not None:
        write_excluded_sites(excluded, results_dir)

    if not rows:
        print("No sites with sufficient data for evaluation.")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(rows).set_index("fid")
    has_both = metrics_df["r2_swim"].notna() & metrics_df["r2_rs"].notna()
    common_df = metrics_df.loc[has_both]

    print(f"\n{'=' * 70}")
    print(f"PAIRED DAILY AGGREGATE ({len(common_df)} sites)")
    print("=" * 70)
    header = f"{'model':<8}"
    for stat in ["r2", "r", "rmse", "bias", "kge"]:
        header += f"  {stat + '_mean':>10}  {stat + '_med':>10}"
    print(header)
    print("-" * len(header))
    for model_name in ["swim", "rs"]:
        line = f"{model_name:<8}"
        for stat in ["r2", "r", "rmse", "bias", "kge"]:
            col = f"{stat}_{model_name}"
            vals = common_df[col].dropna()
            line += f"  {vals.mean():>10.3f}  {vals.median():>10.3f}"
        print(line)

    # Worst / best
    ranked = common_df.sort_values("kge_swim")
    print("\nWorst 10 (by SWIM KGE):")
    for fid, row in ranked.head(10).iterrows():
        print(
            f"  {fid:<12} KGE_swim={row['kge_swim']:.3f}  KGE_rs={row['kge_rs']:.3f}  "
            f"R2_swim={row['r2_swim']:.3f}  R2_rs={row['r2_rs']:.3f}"
        )
    print("\nBest 10 (by SWIM KGE):")
    for fid, row in ranked.tail(10).iterrows():
        print(
            f"  {fid:<12} KGE_swim={row['kge_swim']:.3f}  KGE_rs={row['kge_rs']:.3f}  "
            f"R2_swim={row['r2_swim']:.3f}  R2_rs={row['r2_rs']:.3f}"
        )

    # Write evaluation_summary.csv with 'site' column for publication_figures.py
    if results_dir is not None:
        summary = metrics_df.reset_index().rename(columns={"fid": "site"})
        summary.to_csv(os.path.join(results_dir, "evaluation_summary.csv"), index=False)

    return metrics_df


def evaluate_etf(cfg, container, par_csv, fids, results_dir=None):
    """Compare SWIM ETf against the config-selected target ETf at overpass dates.

    Isolates ETf model skill from ETo bias: compares the calibrated Kcb curve
    directly against the container ETf observations defined by the active config
    (including ensemble or merged targets) on capture dates only. Works for all
    sites without flux tower data.
    """
    calibrated_params, param_source = _resolve_calibrated_params(container, fids, par_csv)
    print(f"ETf evaluation: {len(fids)} sites from {param_source}")

    missing = [f for f in fids if f not in calibrated_params]
    if missing:
        print(f"WARNING: No calibrated params for: {missing} — skipping")
        fids = [f for f in fids if f in calibrated_params]
    if not fids:
        return pd.DataFrame()

    print("Running calibrated model...")
    model_results = run_calibrated_model(cfg, container, fids, calibrated_params)

    rows = []
    for fid in fids:
        if fid not in model_results:
            continue

        swim_etf = model_results[fid]["etf_model"]

        obs_etf = _load_target_etf_series(container, cfg, fid)
        if obs_etf is None:
            print(f"  {fid}: no ETf in container for current config")
            continue

        obs_etf = obs_etf.dropna()
        common = swim_etf.index.intersection(obs_etf.index)
        if len(common) < 10:
            print(f"  {fid}: only {len(common)} overpass dates, skipping")
            continue

        o = obs_etf.loc[common].values
        s = swim_etf.loc[common].values
        m = calc_metrics(o, s)

        row = {
            "fid": fid,
            "n_overpass": len(common),
            "obs_etf_mean": float(np.nanmean(o)),
            "swim_etf_mean": float(np.nanmean(s)),
            **m,
        }
        rows.append(row)

        print(
            f"  {fid}: n={len(common):>4d}  R2={m['r2']:.3f}  r={m['r']:.3f}  "
            f"RMSE={m['rmse']:.3f}  bias={m['bias']:+.3f}  "
            f"obs={row['obs_etf_mean']:.3f}  swim={row['swim_etf_mean']:.3f}"
        )

    if not rows:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(rows).set_index("fid")

    valid = metrics_df["r2"].notna()
    vdf = metrics_df.loc[valid]

    print(f"\n{'=' * 70}")
    print(f"ETf AT OVERPASS AGGREGATE ({len(vdf)} sites)")
    print("=" * 70)
    for stat in ["r2", "r", "rmse", "bias", "kge"]:
        vals = vdf[stat].dropna()
        print(f"  {stat:>6s}:  mean={vals.mean():.3f}  median={vals.median():.3f}")
    print(f"  obs_etf_mean:  {vdf['obs_etf_mean'].mean():.3f}")
    print(f"  swim_etf_mean: {vdf['swim_etf_mean'].mean():.3f}")

    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(results_dir, "evaluation_etf_metrics.csv"))

    return metrics_df


def evaluate_monthly(cfg, container, par_csv, fids, results_dir=None, flux_sources=None):
    """Monthly ET totals: SWIM and RS-derived ETa vs flux tower.

    The RS benchmark is built from native target ETf, linearly interpolated to
    daily, then multiplied by ETo. All series are restricted to flux-valid
    days before monthly aggregation so sums integrate the identical day set.
    Both SWIM and RS ETa are scored on identical paired months.

    Returns DataFrame with per-site paired monthly metrics.
    """
    flux_sources = flux_sources or {}
    calibrated_params, param_source = _resolve_calibrated_params(container, fids, par_csv)
    print(f"Monthly evaluation: {len(fids)} sites from {param_source}")

    missing = [f for f in fids if f not in calibrated_params]
    if missing:
        print(f"WARNING: No calibrated params for: {missing} — skipping these sites")
        fids = [f for f in fids if f in calibrated_params]
    if not fids:
        print("No sites with calibrated parameters.")
        return pd.DataFrame()

    print("Running calibrated model...")
    model_results = run_calibrated_model(cfg, container, fids, calibrated_params)
    rs_eta_by_fid = {
        fid: _build_rs_eta_series(container, cfg, fid, df["etref"])
        for fid, df in model_results.items()
    }

    rows = []
    excluded = []
    for fid in fids:
        flux_et = load_flux_et(fid, flux_sources.get(fid))
        if flux_et.empty:
            excluded.append({"site": fid, "reason": "no_flux_data"})
            continue
        if not passes_site_minimum(flux_et):
            print(f"  {fid}: below VALIDATION_POLICY site minimum (90 valid days / 3 months)")
            excluded.append({"site": fid, "reason": "below_site_minimum_90d_3mo"})
            continue

        model_df = model_results[fid]
        swim_et = model_df["et_act"]
        rs_eta = rs_eta_by_fid.get(fid)
        if rs_eta is None:
            print(f"  {fid}: no RS ETa benchmark, skipping")
            continue

        daily_common = swim_et.index.intersection(flux_et.index)
        if len(daily_common) < 30:
            print(f"  {fid}: only {len(daily_common)} daily overlap, skipping")
            continue

        swim_daily = swim_et.loc[daily_common]
        flux_daily = flux_et.loc[daily_common]
        rs_daily = rs_eta.reindex(daily_common)

        # Monthly totals over flux-valid days only, so every series integrates
        # the identical day set; RS months not finite on every valid day
        # become NaN instead of partial sums
        swim_monthly, flux_monthly, rs_monthly = paired_monthly_sums(
            swim_daily, flux_daily, rs_daily
        )

        all_idx = flux_monthly.index
        rs_on_idx = rs_monthly.reindex(all_idx)
        paired_mask = (
            flux_monthly.notna() & swim_monthly.reindex(all_idx).notna() & rs_on_idx.notna()
        )
        paired_months = all_idx[paired_mask]
        n_paired = len(paired_months)
        if n_paired < 6:
            print(f"  {fid}: only {n_paired} paired months, skipping")
            continue

        obs = flux_monthly.loc[paired_months].values
        m_swim = calc_metrics(obs, swim_monthly.reindex(paired_months).values)
        m_rs = calc_metrics(obs, rs_on_idx.loc[paired_months].values)

        row = {"fid": fid, "n": n_paired}
        for k in ["r2", "r", "rmse", "bias", "kge"]:
            row[k] = m_swim[k]
            row[f"{k}_swim"] = m_swim[k]
            row[f"{k}_rs"] = m_rs[k]
        rows.append(row)

        print(
            f"  {fid}: n_paired={n_paired:>3d} mo  R2_swim={m_swim['r2']:.3f}  "
            f"R2_rs={m_rs['r2']:.3f}  KGE_swim={m_swim['kge']:.3f}  KGE_rs={m_rs['kge']:.3f}"
        )

    if results_dir is not None:
        write_excluded_sites(excluded, results_dir)

    if not rows:
        print("No sites with sufficient data.")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(rows).set_index("fid")
    has_both = metrics_df["r2_swim"].notna() & metrics_df["r2_rs"].notna()
    common_df = metrics_df.loc[has_both]

    print(f"\n{'=' * 70}")
    print(f"PAIRED MONTHLY AGGREGATE ({len(common_df)} sites)")
    print("=" * 70)
    header = f"{'model':<8}"
    for stat in ["r2", "r", "rmse", "bias", "kge"]:
        header += f"  {stat + '_mean':>10}  {stat + '_med':>10}"
    print(header)
    print("-" * len(header))
    for model_name in ["swim", "rs"]:
        line = f"{model_name:<8}"
        for stat in ["r2", "r", "rmse", "bias", "kge"]:
            col = f"{stat}_{model_name}"
            vals = common_df[col].dropna()
            line += f"  {vals.mean():>10.3f}  {vals.median():>10.3f}"
        print(line)

    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate calibrated SWIM against flux tower ET (international)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML config (default: 6_Flux_International.toml)",
    )
    parser.add_argument(
        "--par-csv",
        type=str,
        default=None,
        help="Override automatic par.csv discovery; otherwise fall back to ingested container calibration",
    )
    parser.add_argument(
        "--sites", type=str, default=None, help="Comma-separated site IDs (default: all)"
    )
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Monthly ET totals instead of daily",
    )
    parser.add_argument(
        "--etf",
        action="store_true",
        help="Compare SWIM ETf against the config-selected target ETf at overpass dates",
    )
    parser.add_argument(
        "--container",
        type=str,
        default=None,
        help="Override container path",
    )
    parser.add_argument(
        "--all-container-fields",
        action="store_true",
        help="Evaluate every field in the container instead of the configured "
        "shapefile cohort (default is the publication cohort shapefile)",
    )
    args = parser.parse_args()

    conf_path = Path(args.config) if args.config else None
    cfg = _load_config(conf_path)
    results_dir = _results_dir(cfg, conf_path)

    if args.par_csv:
        if not os.path.exists(args.par_csv):
            raise FileNotFoundError(f"Provided --par-csv does not exist: {args.par_csv}")
        par_csv = args.par_csv
    else:
        # Search in master dir first, then results/group
        master_dir = os.path.join(cfg.pest_run_dir, "master")
        par_csv = find_par_csv(master_dir, cfg.project_name)
        if par_csv is None:
            group_dir = _group_results_dir(cfg, conf_path)
            par_csv = find_par_csv(group_dir, cfg.project_name)
    if par_csv is None:
        print("No .par.csv found; will use ingested container calibration if available.")
    else:
        print(f"Using parameters: {par_csv}")

    if args.container:
        container_path = args.container
    else:
        container_path = _default_container_path(cfg)
    container = SwimContainer.open(container_path, mode="r")

    # Per-site (network, et_col) truth sources declared in the cohort
    # shapefile; used for all site lists so a --sites request also honors them
    flux_sources = {}
    if os.path.exists(cfg.fields_shapefile):
        flux_sources = load_flux_sources(cfg.fields_shapefile, cfg.feature_id_col)
    else:
        print(
            f"WARNING: cohort shapefile not found ({cfg.fields_shapefile}); "
            "flux files resolve by network search order"
        )

    if args.sites:
        fids = [s.strip() for s in args.sites.split(",")]
    elif args.all_container_fields:
        fids = container.field_uids
    else:
        # Default cohort = the configured fields shapefile (the publication
        # cohort), not every site in the container. The container may retain
        # sites excluded from the study cohort (e.g. natural-landcover sites
        # dropped from the 71-site cohort); restrict evaluation to the
        # shapefile and intersect with what the container actually holds.
        import geopandas as gpd

        gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")
        id_col = cfg.feature_id_col if cfg.feature_id_col in gdf.columns else "sid"
        cohort = [str(s) for s in gdf[id_col].tolist()]
        container_fids = set(container.field_uids)
        fids = [f for f in cohort if f in container_fids]
        missing = [f for f in cohort if f not in container_fids]
        if missing:
            print(f"WARNING: {len(missing)} cohort site(s) absent from container: {missing}")
        print(f"Cohort: {len(fids)} sites from {os.path.basename(cfg.fields_shapefile)}")

    try:
        if args.etf:
            metrics = evaluate_etf(cfg, container, par_csv, fids, results_dir=results_dir)
            out_csv = os.path.join(results_dir, "evaluation_etf_metrics.csv")
        elif args.monthly:
            metrics = evaluate_monthly(
                cfg, container, par_csv, fids, results_dir=results_dir, flux_sources=flux_sources
            )
            out_csv = os.path.join(results_dir, "evaluation_monthly_metrics.csv")
        else:
            metrics = evaluate(
                cfg, container, par_csv, fids, results_dir=results_dir, flux_sources=flux_sources
            )
            out_csv = os.path.join(results_dir, "evaluation_metrics.csv")
        os.makedirs(results_dir, exist_ok=True)
        metrics.to_csv(out_csv)
        print(f"\nMetrics saved to {out_csv}")
    finally:
        container.close()
