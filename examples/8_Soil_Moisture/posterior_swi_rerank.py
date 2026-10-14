"""ET-constrained posterior reranking with a fixed SMAP root-zone SWI target.

This is the Stage-0 test in ``notes/wp_c7_rootzone_swi_pareto.md``. It asks whether
the existing e8c1 posterior already contains a soil-moisture solution that preserves
the calibrated evapotranspiration result. It is forward-only: no inverse run and no
parameter changes are made.

Selection data:
  * ET: OpenET ensemble captures and final PEST measurement phi.
  * Soil moisture: SMAP L3 Enhanced transformed with the irregular recursive SWI
    filter at a fixed, literature-selected 20 d characteristic time.
  * Training: 2015-04-01 through 2020-12-31.
  * Satellite holdout: 2021-01-01 through 2025-12-31.

SCAN is read only after the single ET-eligible candidate has been selected. It never
sets the SWI time scale, site membership, parameter ranking, or ET gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from swimrs.calibrate.ssm_obs import recursive_soil_water_index
from swimrs.container import SwimContainer
from swimrs.process.input import build_swim_input
from swimrs.process.loop_fast import run_daily_loop_fast
from swimrs.process.state import FieldProperties

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/data/ssd1/swim/8_Soil_Moisture")
DEFAULT_CONTAINER = PROJECT_ROOT / "data" / "8_Soil_Moisture_e8c1.swim"
DEFAULT_ARCHIVE = PROJECT_ROOT / "results" / "e8c1" / "archive" / "4_pest_outputs"
DEFAULT_PAR = DEFAULT_ARCHIVE / "8_Soil_Moisture.3.par.csv"
DEFAULT_PHI = DEFAULT_ARCHIVE / "8_Soil_Moisture.phi.actual.csv"
DEFAULT_SMAP = HERE / "notes" / "paper" / "data" / "smap_l3_scan.parquet"
DEFAULT_OUT = PROJECT_ROOT / "results" / "e8c1" / "swi_rerank"
BASELINE = "posterior_median"
GROW_MONTHS = (4, 5, 6, 7, 8, 9, 10)

PARAMETER_ATTRS = {
    "ndvi_k": "ndvi_k",
    "ndvi_0": "ndvi_0",
    "swe_alpha": "swe_alpha",
    "swe_beta": "swe_beta",
    "kr_alpha": "kr_damp",
    "ks_alpha": "ks_damp",
}
PROPERTY_ATTRS = {
    "aw": "awc",
    "mad": "mad",
    "refill_frac": "refill_frac",
    "min_irr_days": "min_irr_days",
    "irr_depth": "irr_depth",
    "rew": "rew",
    "tew": "tew",
}


def _load_evaluate():
    spec = importlib.util.spec_from_file_location("ex8_swi_evaluate", HERE / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_row_params(row: pd.Series, fids: list[str]) -> dict[str, dict[str, float]]:
    """Parse one PEST par-CSV row into absolute values by field."""
    lower_fids = sorted(((fid.lower(), fid) for fid in fids), key=lambda item: -len(item[0]))
    result: dict[str, dict[str, float]] = {}
    for column, value in row.items():
        stem = column.split("_ptype:", 1)[0].replace("pname:p_", "").rsplit("_:0", 1)[0]
        stem_lower = stem.lower()
        for fid_lower, fid in lower_fids:
            suffix = f"_{fid_lower}"
            if stem_lower.endswith(suffix):
                result.setdefault(fid, {})[stem[: -len(suffix)]] = float(value)
                break
    return result


def copy_field_properties(base: FieldProperties) -> FieldProperties:
    """Deep-copy the array-valued field properties used by the fast loop."""
    return FieldProperties(
        n_fields=base.n_fields,
        fids=base.fids.copy(),
        awc=base.awc.copy(),
        ksat=base.ksat.copy(),
        rew=base.rew.copy(),
        tew=base.tew.copy(),
        cn2=base.cn2.copy(),
        zr_max=base.zr_max.copy(),
        zr_min=base.zr_min.copy(),
        mad=base.mad.copy(),
        irr_status=base.irr_status.copy(),
        perennial=base.perennial.copy(),
        gw_status=base.gw_status.copy(),
        ke_max=base.ke_max.copy(),
        kc_max=base.kc_max.copy(),
        f_sub=base.f_sub.copy(),
        refill_frac=base.refill_frac.copy(),
        min_irr_days=base.min_irr_days.copy(),
        irr_depth=base.irr_depth.copy(),
    )


def apply_parameter_values(base_parameters, base_properties, values_by_fid, fids):
    """Return independent parameter/property objects with absolute PEST values."""
    parameters = base_parameters.copy()
    properties = copy_field_properties(base_properties)
    known = set(PARAMETER_ATTRS) | set(PROPERTY_ATTRS)
    for field_idx, fid in enumerate(fids):
        for name, value in values_by_fid.get(fid, {}).items():
            if name in PARAMETER_ATTRS:
                getattr(parameters, PARAMETER_ATTRS[name])[field_idx] = value
            elif name in PROPERTY_ATTRS:
                getattr(properties, PROPERTY_ATTRS[name])[field_idx] = value
            else:
                raise ValueError(f"Unsupported posterior parameter {name!r} for {fid}")
    parsed = {name for field_values in values_by_fid.values() for name in field_values}
    missing = parsed - known
    if missing:
        raise ValueError(f"Unsupported posterior parameters: {sorted(missing)}")
    return parameters, properties


def _safe_median(frame: pd.DataFrame, column: str, mask=None) -> float:
    values = frame.loc[mask, column] if mask is not None else frame[column]
    values = pd.to_numeric(values, errors="coerce")
    return float(values.median()) if values.notna().any() else np.nan


def summarize_realization(site_metrics: pd.DataFrame) -> dict[str, float | str | int]:
    """Reduce one realization's per-site metrics without blending cohorts."""
    labels = site_metrics["realization"].astype(str).unique()
    if len(labels) != 1:
        raise ValueError("site_metrics must contain exactly one realization")
    active = site_metrics["scheduler_active"].astype(bool)
    return {
        "realization": labels[0],
        "n_sites": int(len(site_metrics)),
        "n_scheduler_active": int(active.sum()),
        "swi_train_active_anom_r": _safe_median(site_metrics, "swi_train_anom_r", active),
        "swi_train_active_rmse": _safe_median(site_metrics, "swi_train_anom_rmse", active),
        "swi_train_active_std_ratio": _safe_median(site_metrics, "swi_train_std_ratio", active),
        "swi_holdout_active_anom_r": _safe_median(site_metrics, "swi_holdout_anom_r", active),
        "swi_holdout_active_rmse": _safe_median(site_metrics, "swi_holdout_anom_rmse", active),
        "swi_holdout_active_std_ratio": _safe_median(site_metrics, "swi_holdout_std_ratio", active),
        "et_active_mae": _safe_median(site_metrics, "et_mae", active),
        "et_active_abs_bias": _safe_median(site_metrics, "et_abs_bias", active),
        "et_all_mae": _safe_median(site_metrics, "et_mae"),
    }


def apply_et_gates(
    summaries: pd.DataFrame,
    site_metrics: pd.DataFrame,
    phi_reference: float,
    phi_factor: float = 1.02,
    median_mae_tolerance: float = 0.003,
    site_mae_tolerance: float = 0.02,
) -> pd.DataFrame:
    """Attach the predeclared ET-preservation gates to every realization."""
    out = summaries.copy()
    out["realization"] = out["realization"].astype(str)
    baseline_rows = out[out["realization"] == BASELINE]
    if len(baseline_rows) != 1:
        raise ValueError(f"Expected exactly one {BASELINE!r} summary")
    baseline = baseline_rows.iloc[0]
    base_frame = site_metrics[site_metrics["realization"].astype(str) == BASELINE]
    base_frame = base_frame[base_frame["et_n"] >= 30]
    base_sites = base_frame.set_index("site_id")["et_mae"].dropna()

    max_regression = {}
    for realization, group in site_metrics.groupby(site_metrics["realization"].astype(str)):
        group = group[group["et_n"] >= 30]
        candidate = group.set_index("site_id")["et_mae"].dropna()
        common = candidate.index.intersection(base_sites.index)
        max_regression[str(realization)] = (
            float((candidate.loc[common] - base_sites.loc[common]).max())
            if len(common) == len(base_sites)
            else np.inf
        )
    out["max_site_et_mae_regression"] = out["realization"].map(max_regression)
    out["phi_ok"] = out["phi"].le(phi_reference * phi_factor)
    out["median_et_mae_ok"] = out["et_active_mae"].le(
        float(baseline["et_active_mae"]) + median_mae_tolerance
    )
    out["median_et_bias_ok"] = out["et_active_abs_bias"].le(
        float(baseline["et_active_abs_bias"]) + 1e-12
    )
    out["site_et_mae_ok"] = out["max_site_et_mae_regression"].le(site_mae_tolerance)
    out["et_eligible"] = out[
        ["phi_ok", "median_et_mae_ok", "median_et_bias_ok", "site_et_mae_ok"]
    ].all(axis=1)
    out.loc[out["realization"] == BASELINE, "et_eligible"] = True
    return out


def mark_pareto_front(summaries: pd.DataFrame) -> pd.DataFrame:
    """Mark non-dominated eligible candidates (maximize SWI r, minimize ET MAE)."""
    out = summaries.copy()
    out["pareto_front"] = False
    eligible = out[
        out["et_eligible"] & out["swi_train_active_anom_r"].notna() & out["et_active_mae"].notna()
    ]
    for idx, row in eligible.iterrows():
        dominated = (
            (eligible["swi_train_active_anom_r"] >= row["swi_train_active_anom_r"])
            & (eligible["et_active_mae"] <= row["et_active_mae"])
            & (
                (eligible["swi_train_active_anom_r"] > row["swi_train_active_anom_r"])
                | (eligible["et_active_mae"] < row["et_active_mae"])
            )
        ).any()
        out.loc[idx, "pareto_front"] = not dominated
    return out


def select_training_candidate(summaries: pd.DataFrame) -> pd.Series:
    """Select by training SWI only after ET gates; holdout columns are untouched."""
    candidates = summaries[
        (summaries["realization"].astype(str) != BASELINE)
        & summaries["et_eligible"].astype(bool)
        & summaries["swi_train_active_anom_r"].notna()
    ].copy()
    if candidates.empty:
        raise RuntimeError("No posterior realization passed the ET-preservation gates")
    candidates["train_std_error"] = (candidates["swi_train_active_std_ratio"] - 1.0).abs()
    return candidates.sort_values(
        ["swi_train_active_anom_r", "swi_train_active_rmse", "train_std_error", "realization"],
        ascending=[False, True, True, True],
    ).iloc[0]


def _anomaly_rmse(ev, observed: pd.Series, modeled: pd.Series) -> float:
    pair = pd.concat([observed.rename("obs"), modeled.rename("model")], axis=1, sort=False).dropna()
    if len(pair) < 3:
        return np.nan
    obs_anom = ev._deseasonalize(pair["obs"])
    model_anom = ev._deseasonalize(pair["model"])
    return float(np.sqrt(np.mean((model_anom - obs_anom) ** 2)))


def _interval_metrics(ev, observed, modeled, start, end):
    observed = observed.loc[start:end]
    modeled = modeled.loc[start:end]
    observed = observed[observed.index.month.isin(GROW_MONTHS)]
    modeled = modeled[modeled.index.month.isin(GROW_MONTHS)]
    # The reranker needs contemporaneous anomaly skill and amplitude. The more
    # expensive lag diagnostic is reserved for the final SCAN evaluation.
    metrics = ev.series_pair_metrics(observed, modeled, max_lag_days=0)
    return {
        "n": metrics["n"],
        "anom_r": metrics["anom_r"],
        "anom_rmse": _anomaly_rmse(ev, observed, modeled),
        "std_ratio": metrics["std_ratio"],
        "best_model_shift_days": metrics["best_model_shift_days"],
    }


def _et_metrics(observed: pd.Series, modeled: pd.Series):
    pair = pd.concat([observed.rename("obs"), modeled.rename("model")], axis=1, sort=False).dropna()
    pair = pair[pair.index.month.isin(GROW_MONTHS)]
    if len(pair) < 3:
        return {"n": len(pair), "mae": np.nan, "bias": np.nan, "abs_bias": np.nan}
    residual = pair["model"] - pair["obs"]
    bias = float(residual.mean())
    return {
        "n": len(pair),
        "mae": float(residual.abs().mean()),
        "bias": bias,
        "abs_bias": abs(bias),
    }


def score_output(
    ev,
    realization,
    output,
    properties,
    dates,
    fids,
    scheduler_active,
    swi_by_fid,
    et_by_fid,
    train_start,
    train_end,
    holdout_start,
    holdout_end,
):
    """Return one row per site for one forward output."""
    rows = []
    for idx, fid in enumerate(fids):
        theta = pd.Series(
            ev.theta_available(
                properties.awc[idx],
                output.zr[:, idx],
                output.depl_root[:, idx],
                output.daw3[:, idx],
                properties.zr_max[idx],
            ),
            index=dates,
        )
        train = _interval_metrics(ev, swi_by_fid[fid], theta, train_start, train_end)
        holdout = _interval_metrics(ev, swi_by_fid[fid], theta, holdout_start, holdout_end)
        et = _et_metrics(et_by_fid[fid], pd.Series(output.etf[:, idx], index=dates))
        rows.append(
            {
                "realization": str(realization),
                "site_id": fid,
                "scheduler_active": bool(scheduler_active[idx]),
                **{f"swi_train_{key}": value for key, value in train.items()},
                **{f"swi_holdout_{key}": value for key, value in holdout.items()},
                **{f"et_{key}": value for key, value in et.items()},
            }
        )
    return pd.DataFrame(rows)


def _read_phi(path: Path, realizations: list[str]):
    table = pd.read_csv(path)
    row = table.loc[table["iteration"] == table["iteration"].max()].iloc[0]
    by_realization = {r: float(row[r]) for r in realizations if r in row.index}
    return by_realization, float(np.mean(list(by_realization.values())))


def _read_remote_targets(container_path: Path, smap_path: Path, fids, swi_time_days):
    root = zarr.open_group(str(container_path), mode="r")
    container_dates = pd.DatetimeIndex(root["time/daily"][:])
    container_fids = [str(fid) for fid in root["geometry/uid"][:]]
    positions = {fid: idx for idx, fid in enumerate(container_fids)}
    et_array = root["remote_sensing/etf/landsat/ensemble/no_mask"]
    et_by_fid = {
        fid: pd.Series(et_array[:, positions[fid]], index=container_dates, dtype=float)
        for fid in fids
    }

    smap = pd.read_parquet(smap_path)
    smap["date"] = pd.to_datetime(smap["date"])
    swi_by_fid = {}
    for fid in fids:
        frame = smap[smap["site_id"].astype(str) == fid].sort_values("date")
        series = frame.drop_duplicates("date").set_index("date")["smap_l3_sm"].astype(float)
        values = recursive_soil_water_index(
            series.to_numpy(), series.index.to_numpy(), characteristic_time_days=swi_time_days
        )
        swi_by_fid[fid] = pd.Series(values, index=series.index)
    return swi_by_fid, et_by_fid


def _atomic_csv(frame: pd.DataFrame, path: Path):
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _bootstrap_median_ci(values, seed=42, n_boot=10_000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    return np.quantile(np.median(samples, axis=1), [0.025, 0.975]).tolist()


def _score_scan(ev, labels_outputs, properties_by_label, dates, fids, scheduler_active):
    sites = pd.read_csv(ev.SITES_CSV)
    theta_paths = dict(zip(sites["site_id"].astype(str), sites["theta_csv"], strict=True))
    rows = []
    for label, output in labels_outputs.items():
        properties = properties_by_label[label]
        for idx, fid in enumerate(fids):
            try:
                observed = ev._read_obs(theta_paths[fid])
            except FileNotFoundError:
                continue
            rootzone = ev.depth_weighted_rootzone(
                observed, max_depth_cm=float(properties.zr_max[idx] * 100.0)
            )
            modeled = pd.Series(
                ev.theta_available(
                    properties.awc[idx],
                    output.zr[:, idx],
                    output.depl_root[:, idx],
                    output.daw3[:, idx],
                    properties.zr_max[idx],
                ),
                index=dates,
            )
            rootzone = rootzone[rootzone.index.month.isin(GROW_MONTHS)]
            modeled = modeled[modeled.index.month.isin(GROW_MONTHS)]
            metrics = ev.series_pair_metrics(rootzone, modeled, max_lag_days=14)
            rows.append(
                {
                    "realization": str(label),
                    "site_id": fid,
                    "scheduler_active": bool(scheduler_active[idx]),
                    "model_depth_cm": float(properties.zr_max[idx] * 100.0),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _scan_decision(scan: pd.DataFrame, selected_label: str):
    baseline = scan[scan["realization"] == BASELINE].set_index("site_id")
    selected = scan[scan["realization"] == selected_label].set_index("site_id")
    common = baseline.index.intersection(selected.index)
    delta = (
        selected.loc[common, ["pearson", "anom_r", "std_ratio"]]
        - baseline.loc[common, ["pearson", "anom_r", "std_ratio"]]
    )
    active = selected.loc[common, "scheduler_active"].astype(bool)
    selected_active = selected.loc[common][active]
    inactive_delta = delta.loc[~active]
    pearson_delta = delta.loc[active, "pearson"]
    stable_active = int((pearson_delta >= -0.02).sum())
    inactive_pearson = float(inactive_delta["pearson"].median())
    inactive_anom = float(inactive_delta["anom_r"].median())
    medians = selected_active[["pearson", "anom_r", "std_ratio"]].median()
    gates = {
        "active_pearson_at_least_0.50": bool(medians["pearson"] >= 0.50),
        "active_anom_r_at_least_0.40": bool(medians["anom_r"] >= 0.40),
        "active_std_ratio_at_least_0.60": bool(medians["std_ratio"] >= 0.60),
        "at_least_9_active_sites_stable": bool(stable_active >= 9),
        "inactive_pearson_within_0.02": bool(inactive_pearson >= -0.02),
        "inactive_anom_r_within_0.02": bool(inactive_anom >= -0.02),
    }
    return {
        "selected_active_medians": medians.to_dict(),
        "active_sites_improved_or_within_0.02": stable_active,
        "inactive_median_deltas": {"pearson": inactive_pearson, "anom_r": inactive_anom},
        "active_paired_delta_median_ci95": {
            column: _bootstrap_median_ci(delta.loc[active, column])
            for column in ["pearson", "anom_r", "std_ratio"]
        },
        "sites_of_interest": {
            fid: selected.loc[fid, ["pearson", "anom_r", "std_ratio"]].to_dict()
            if fid in selected.index
            else None
            for fid in ["Manderfield", "Circleville", "Vallecitos"]
        },
        "gates": gates,
        "all_scan_gates_pass": all(gates.values()),
    }


def _selection_report(summaries, selected, complete, train_gate=0.08, holdout_gate=0.05):
    baseline = summaries[summaries["realization"] == BASELINE].iloc[0]
    train_delta = float(selected["swi_train_active_anom_r"] - baseline["swi_train_active_anom_r"])
    holdout_delta = float(
        selected["swi_holdout_active_anom_r"] - baseline["swi_holdout_active_anom_r"]
    )
    amplitude_train = float(selected["swi_train_active_std_ratio"])
    amplitude_holdout = float(selected["swi_holdout_active_std_ratio"])
    gates = {
        "training_anom_r_improvement_at_least_0.08": train_delta >= train_gate,
        "holdout_anom_r_improvement_at_least_0.05": holdout_delta >= holdout_gate,
        "training_std_ratio_in_0.8_to_1.2": 0.8 <= amplitude_train <= 1.2,
        "holdout_std_ratio_in_0.8_to_1.2": 0.8 <= amplitude_holdout <= 1.2,
        "et_preservation": bool(selected["et_eligible"]),
    }
    return {
        "selection_complete": complete,
        "selected_realization": str(selected["realization"]),
        "selection_rule": "maximum training SWI20 anomaly correlation among ET-eligible members",
        "baseline": baseline.replace({np.nan: None}).to_dict(),
        "selected": selected.replace({np.nan: None}).to_dict(),
        "training_anom_r_delta": train_delta,
        "holdout_anom_r_delta": holdout_delta,
        "satellite_gates": gates,
        "all_satellite_and_et_gates_pass": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", type=Path, default=DEFAULT_CONTAINER)
    parser.add_argument("--par-csv", type=Path, default=DEFAULT_PAR)
    parser.add_argument("--phi-csv", type=Path, default=DEFAULT_PHI)
    parser.add_argument("--smap-parquet", type=Path, default=DEFAULT_SMAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--swi-time-days", type=float, default=20.0)
    parser.add_argument("--train-start", default="2015-04-01")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--holdout-start", default="2021-01-01")
    parser.add_argument("--holdout-end", default="2025-12-31")
    parser.add_argument("--phi-factor", type=float, default=1.02)
    parser.add_argument("--median-et-mae-tolerance", type=float, default=0.003)
    parser.add_argument("--site-et-mae-tolerance", type=float, default=0.02)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--max-realizations", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-scan", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    site_csv = args.out_dir / "posterior_site_metrics.csv"
    summary_csv = args.out_dir / "posterior_summary.csv"
    selection_json = args.out_dir / "selection.json"

    ev = _load_evaluate()
    cfg = ev._load_config()
    sites = pd.read_csv(ev.SITES_CSV)
    fids = sites["site_id"].astype(str).tolist()
    par = pd.read_csv(args.par_csv, index_col=0)
    par.index = par.index.astype(str)
    realizations = sorted((idx for idx in par.index if idx.isdigit()), key=int)
    phi_by_realization, phi_reference = _read_phi(args.phi_csv, realizations)
    realizations = [
        realization for realization in realizations if realization in phi_by_realization
    ]
    requested = realizations[: args.max_realizations] if args.max_realizations else realizations
    complete = len(requested) == len(realizations)

    site_frames = []
    if args.resume and site_csv.exists():
        previous = pd.read_csv(site_csv, dtype={"realization": str, "site_id": str})
        site_frames.append(previous)
    elif site_csv.exists() or summary_csv.exists():
        raise FileExistsError("Output exists; pass --resume or choose a new --out-dir")
    prior = pd.concat(site_frames, ignore_index=True) if site_frames else pd.DataFrame()
    completed = set()
    if not prior.empty:
        counts = prior.groupby("realization")["site_id"].nunique()
        completed = set(counts[counts == len(fids)].index.astype(str))

    swi_by_fid, et_by_fid = _read_remote_targets(
        args.container, args.smap_parquet, fids, args.swi_time_days
    )
    container = SwimContainer.open(str(args.container), mode="r")
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp:
        input_h5 = temp.name
    si = None
    try:
        si = build_swim_input(
            container,
            output_h5=input_h5,
            start_date=cfg.start_dt,
            end_date=cfg.end_dt,
            refet_type=getattr(cfg, "refet_type", "eto") or "eto",
            fields=fids,
            empirical_kc_max=True,
            mask_mode=getattr(cfg, "mask_mode", "none"),
            transpiration_cover_scaling=getattr(cfg, "transpiration_cover_scaling", True),
            use_container_calibration=False,
        )
        dates = pd.date_range(si.start_date, periods=si.n_days, freq="D")
        scheduler_active = np.asarray(si.get_irr_flag(), dtype=bool).any(axis=0)
        base_parameters = si.parameters.copy()
        base_properties = copy_field_properties(si.properties)

        work = [(BASELINE, par.loc[realizations].median())]
        work.extend((realization, par.loc[realization]) for realization in requested)
        new_frames = []
        for run_number, (label, row) in enumerate(work, start=1):
            if str(label) in completed:
                print(f"[{run_number:>3}/{len(work)}] {label}: checkpoint exists, skipping")
                continue
            values = parse_row_params(row, fids)
            parameters, properties = apply_parameter_values(
                base_parameters, base_properties, values, fids
            )
            output, _ = run_daily_loop_fast(si, parameters=parameters, properties=properties)
            frame = score_output(
                ev,
                label,
                output,
                properties,
                dates,
                fids,
                scheduler_active,
                swi_by_fid,
                et_by_fid,
                args.train_start,
                args.train_end,
                args.holdout_start,
                args.holdout_end,
            )
            new_frames.append(frame)
            train_r = summarize_realization(frame)["swi_train_active_anom_r"]
            print(f"[{run_number:>3}/{len(work)}] {label}: active training SWI20 r={train_r:.3f}")
            if len(new_frames) % args.checkpoint_every == 0:
                checkpoint = pd.concat(site_frames + new_frames, ignore_index=True)
                _atomic_csv(checkpoint, site_csv)
        if new_frames:
            site_frames.extend(new_frames)
            _atomic_csv(pd.concat(site_frames, ignore_index=True), site_csv)

        metrics = pd.read_csv(site_csv, dtype={"realization": str, "site_id": str})
        keep = {BASELINE, *requested}
        metrics = metrics[metrics["realization"].isin(keep)].copy()
        summaries = pd.DataFrame(
            [summarize_realization(group) for _, group in metrics.groupby("realization")]
        )
        summaries["phi"] = summaries["realization"].map(phi_by_realization)
        summaries.loc[summaries["realization"] == BASELINE, "phi"] = phi_reference
        summaries = apply_et_gates(
            summaries,
            metrics,
            phi_reference,
            phi_factor=args.phi_factor,
            median_mae_tolerance=args.median_et_mae_tolerance,
            site_mae_tolerance=args.site_et_mae_tolerance,
        )
        summaries = mark_pareto_front(summaries)
        selected = select_training_candidate(summaries)
        report = _selection_report(summaries, selected, complete)
        _atomic_csv(summaries.sort_values("realization"), summary_csv)

        selected_label = str(selected["realization"])
        selected_values = parse_row_params(par.loc[selected_label], fids)
        with open(args.out_dir / "selected_params.json", "w") as stream:
            json.dump(selected_values, stream, indent=2, sort_keys=True)

        if args.score_scan and complete and report["all_satellite_and_et_gates_pass"]:
            labels_outputs = {}
            properties_by_label = {}
            for label, row in [
                (BASELINE, par.loc[realizations].median()),
                (selected_label, par.loc[selected_label]),
            ]:
                values = parse_row_params(row, fids)
                parameters, properties = apply_parameter_values(
                    base_parameters, base_properties, values, fids
                )
                output, _ = run_daily_loop_fast(si, parameters=parameters, properties=properties)
                labels_outputs[label] = output
                properties_by_label[label] = properties
            scan = _score_scan(
                ev, labels_outputs, properties_by_label, dates, fids, scheduler_active
            )
            _atomic_csv(scan, args.out_dir / "selected_scan_metrics.csv")
            report["scan"] = _scan_decision(scan, selected_label)
        elif args.score_scan:
            report["scan_skipped_reason"] = (
                "SCAN remains sealed until a complete rerank passes the predeclared "
                "satellite holdout and ET gates"
            )

        with open(selection_json, "w") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
        print(f"\nselected realization: {selected_label}")
        print(f"training SWI20 delta: {report['training_anom_r_delta']:+.3f}")
        print(f"holdout SWI20 delta: {report['holdout_anom_r_delta']:+.3f}")
        print(f"satellite + ET gates pass: {report['all_satellite_and_et_gates_pass']}")
        if "scan" in report:
            print(f"independent SCAN gates pass: {report['scan']['all_scan_gates_pass']}")
        print(f"wrote {selection_json}")
    finally:
        if si is not None:
            si.close()
        container.close()
        if os.path.exists(input_h5):
            os.remove(input_h5)


if __name__ == "__main__":
    main()
