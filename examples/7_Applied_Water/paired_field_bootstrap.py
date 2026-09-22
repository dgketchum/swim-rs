"""Paired field-clustered bootstrap for the Example 7 parameter-path comparison.

The locally calibrated and transferred simulations are evaluated on identical
``(site_id, year)`` meter records. Bootstrap replicates resample whole fields,
stratified by basin, so all years from a selected field enter both parameter
paths together. The reported contrast is always transfer minus local calibration.

This script is evaluation-only. It reads frozen ``per_field_year.csv`` outputs;
it does not run SWIM-RS, calibration, or data extraction.

    uv run python examples/7_Applied_Water/paired_field_bootstrap.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ex7_paths  # noqa: E402

MM_PER_FT = 304.8
# The pooled Run 22 vector (all 110 fields), not the paper's irrigated-class arm.
TRANSFER_LABEL = "transfer_run22"
OUT_LABEL = "local_vs_transfer_run22"

KEY_COLUMNS = ["site_id", "year"]
REQUIRED_COLUMNS = {
    "site_id",
    "year",
    "metered_depth_mm",
    "metered_volume_af",
    "acres",
    "sim_applied_mm",
    "basin",
}
TRUTH_COLUMNS = [
    "metered_depth_mm",
    "metered_volume_af",
    "acres",
    "basin",
    "method",
    "source",
    "crop",
]

METRIC_META = {
    "record_total_volume_r": ("unitless", "higher"),
    "record_total_volume_nse": ("unitless", "higher"),
    "record_total_volume_slope": ("unitless", "descriptive"),
    "record_total_volume_abs_slope_error": ("unitless", "lower"),
    "record_total_volume_bias_pct": ("percentage_points", "descriptive_signed"),
    "record_total_volume_abs_bias_pct": ("percentage_points", "lower"),
    "between_field_mean_depth_r": ("unitless", "higher"),
    "pooled_field_year_depth_r": ("unitless", "higher"),
    "pooled_field_year_depth_rmse_mm": ("mm_per_year", "lower"),
    "pooled_field_year_depth_bias_pct": ("percentage_points", "descriptive_signed"),
    "pooled_field_year_depth_abs_bias_pct": ("percentage_points", "lower"),
    "within_field_anomaly_r": ("unitless", "higher"),
    "within_field_anomaly_slope": ("unitless", "descriptive"),
    "within_field_anomaly_abs_slope_error": ("unitless", "lower"),
    "median_field_bias_pct": ("percentage_points", "descriptive_signed"),
    "median_abs_field_bias_pct": ("percentage_points", "lower"),
    "fraction_fields_within_20pct": ("fraction", "higher"),
    "fraction_fields_within_30pct": ("fraction", "higher"),
    "median_field_temporal_r": ("unitless", "higher"),
    "fraction_fields_temporal_r_gt_0_5": ("fraction", "higher"),
}

POINT_RECONCILIATION = {
    "record_total_volume_r": ("per_field_volume_total", "r", 3),
    "record_total_volume_nse": ("per_field_volume_total", "nse_1to1", 3),
    "record_total_volume_slope": ("per_field_volume_total", "slope", 3),
    "record_total_volume_bias_pct": ("per_field_volume_total", "bias_pct", 1),
    "between_field_mean_depth_r": ("per_field_mean_depth", "r", 3),
    "pooled_field_year_depth_r": ("pooled_field_year_depth", "r", 3),
    "pooled_field_year_depth_rmse_mm": ("pooled_field_year_depth", "rmse", 1),
    "pooled_field_year_depth_bias_pct": ("pooled_field_year_depth", "bias_pct", 1),
    "within_field_anomaly_r": ("within_field_anomaly_depth", "r", 3),
    "within_field_anomaly_slope": ("within_field_anomaly_depth", "slope", 3),
    "median_field_bias_pct": ("per_field_bias_pct", "median", 1),
    "fraction_fields_within_20pct": ("per_field_bias_pct", "within_20pct", 3),
    "fraction_fields_within_30pct": ("per_field_bias_pct", "within_30pct", 3),
    "median_field_temporal_r": ("per_field_temporal_r", "median", 3),
    "fraction_fields_temporal_r_gt_0_5": ("per_field_temporal_r", "frac_gt_0.5", 3),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_values(left: pd.Series, right: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        return bool(
            np.allclose(
                left.to_numpy(float),
                right.to_numpy(float),
                rtol=0.0,
                atol=1e-10,
                equal_nan=True,
            )
        )
    return left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str))


def load_paired_inputs(local_path: Path, transfer_path: Path) -> pd.DataFrame:
    """Load and strictly reconcile the two frozen field-year outputs."""
    local = pd.read_csv(local_path)
    transfer = pd.read_csv(transfer_path)

    for label, frame in (("local", local), ("transfer", transfer)):
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{label} input is missing required columns: {sorted(missing)}")
        if frame.duplicated(KEY_COLUMNS).any():
            dup = frame.loc[frame.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head()
            raise ValueError(f"{label} input has duplicate field-year keys:\n{dup}")

    local = local.sort_values(KEY_COLUMNS).reset_index(drop=True)
    transfer = transfer.sort_values(KEY_COLUMNS).reset_index(drop=True)
    if not local[KEY_COLUMNS].equals(transfer[KEY_COLUMNS]):
        local_keys = set(map(tuple, local[KEY_COLUMNS].itertuples(index=False, name=None)))
        transfer_keys = set(map(tuple, transfer[KEY_COLUMNS].itertuples(index=False, name=None)))
        missing = sorted(local_keys - transfer_keys)[:5]
        extra = sorted(transfer_keys - local_keys)[:5]
        raise ValueError(
            "Local and transfer field-year keys differ; "
            f"missing_from_transfer={missing}, extra_in_transfer={extra}"
        )

    shared_truth = [c for c in TRUTH_COLUMNS if c in local.columns and c in transfer.columns]
    for column in shared_truth:
        if not _same_values(local[column], transfer[column]):
            raise ValueError(
                f"Local and transfer inputs disagree in truth/metadata column {column!r}"
            )

    paired = local[KEY_COLUMNS + shared_truth].copy()
    paired["local_sim_applied_mm"] = local["sim_applied_mm"].to_numpy(float)
    paired["transfer_sim_applied_mm"] = transfer["sim_applied_mm"].to_numpy(float)
    if "sim_et_mm" in local.columns and "sim_et_mm" in transfer.columns:
        paired["local_sim_et_mm"] = local["sim_et_mm"].to_numpy(float)
        paired["transfer_sim_et_mm"] = transfer["sim_et_mm"].to_numpy(float)

    paired = paired.loc[paired["metered_depth_mm"] > 0].copy()
    if paired.empty:
        raise ValueError("No positive metered-depth records remain after reconciliation")
    comparison_values = paired[
        ["metered_depth_mm", "local_sim_applied_mm", "transfer_sim_applied_mm"]
    ].to_numpy(float)
    if not np.isfinite(comparison_values).all():
        raise ValueError("Paired comparison contains non-finite observed or simulated depths")
    basin_counts = paired.groupby("site_id")["basin"].nunique()
    if (basin_counts != 1).any():
        bad = basin_counts[basin_counts != 1].index.tolist()[:5]
        raise ValueError(f"Fields assigned to multiple basins: {bad}")
    return paired.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _correlation(obs: np.ndarray, sim: np.ndarray) -> float:
    if len(obs) < 3 or np.std(obs) == 0 or np.std(sim) == 0:
        return np.nan
    return float(np.corrcoef(obs, sim)[0, 1])


def _fit(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    keep = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[keep]
    sim = sim[keep]
    if len(obs) < 3:
        return {"r": np.nan, "nse": np.nan, "slope": np.nan, "bias_pct": np.nan, "rmse": np.nan}

    ss_total = float(np.sum((obs - obs.mean()) ** 2))
    ss_error = float(np.sum((obs - sim) ** 2))
    slope = (
        float(np.sum((obs - obs.mean()) * (sim - sim.mean())) / ss_total)
        if ss_total > 0
        else np.nan
    )
    bias_pct = (
        100.0 * float(sim.mean() - obs.mean()) / float(obs.mean()) if obs.mean() != 0 else np.nan
    )

    return {
        "r": _correlation(obs, sim),
        "nse": 1.0 - ss_error / ss_total if ss_total > 0 else np.nan,
        "slope": slope,
        "bias_pct": bias_pct,
        "rmse": float(np.sqrt(np.mean((obs - sim) ** 2))),
    }


def _finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def _finite_fraction_above(values: np.ndarray, threshold: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values > threshold)) if len(values) else np.nan


def prepare_fields(paired: pd.DataFrame, min_years: int = 4) -> dict:
    """Build field-level sufficient statistics and retained annual arrays."""
    rows = []
    records = []
    for site_id, sub in paired.groupby("site_id", sort=True):
        sub = sub.sort_values("year")
        obs = sub["metered_depth_mm"].to_numpy(float)
        local = sub["local_sim_applied_mm"].to_numpy(float)
        transfer = sub["transfer_sim_applied_mm"].to_numpy(float)
        acres = sub["acres"].to_numpy(float)

        obs_anomaly = obs - obs.mean()
        local_anomaly = local - local.mean()
        transfer_anomaly = transfer - transfer.mean()
        temporal_eligible = (
            len(obs) >= min_years and np.std(obs) > 0 and np.std(local) > 0 and np.std(transfer) > 0
        )
        local_temporal_r = _correlation(obs, local) if temporal_eligible else np.nan
        transfer_temporal_r = _correlation(obs, transfer) if temporal_eligible else np.nan

        obs_mean = float(obs.mean())
        local_mean = float(local.mean())
        transfer_mean = float(transfer.mean())
        rows.append(
            {
                "site_id": str(site_id),
                "basin": str(sub["basin"].iloc[0]),
                "n_years": int(len(sub)),
                "metered_mean_mm": obs_mean,
                "local_mean_mm": local_mean,
                "transfer_mean_mm": transfer_mean,
                "local_field_bias_pct": 100.0 * (local_mean - obs_mean) / obs_mean,
                "transfer_field_bias_pct": 100.0 * (transfer_mean - obs_mean) / obs_mean,
                "metered_volume_total_af": float(sub["metered_volume_af"].sum()),
                "local_volume_total_af": float(np.sum(local / MM_PER_FT * acres)),
                "transfer_volume_total_af": float(np.sum(transfer / MM_PER_FT * acres)),
                "local_temporal_r": local_temporal_r,
                "transfer_temporal_r": transfer_temporal_r,
            }
        )
        records.append(
            {
                "obs": obs,
                "local": local,
                "transfer": transfer,
                "obs_anomaly": obs_anomaly,
                "local_anomaly": local_anomaly,
                "transfer_anomaly": transfer_anomaly,
            }
        )

    fields = pd.DataFrame(rows).sort_values(["basin", "site_id"]).reset_index(drop=True)
    record_by_site = {
        str(site): record for site, record in zip(sorted(paired.site_id.unique()), records)
    }
    ordered_records = [record_by_site[site] for site in fields["site_id"]]
    return {"fields": fields, "records": ordered_records}


def _comparison(prepared: dict, indices: np.ndarray) -> dict[str, tuple[float, float]]:
    fields = prepared["fields"].iloc[indices]
    records = prepared["records"]

    obs = np.concatenate([records[i]["obs"] for i in indices])
    local = np.concatenate([records[i]["local"] for i in indices])
    transfer = np.concatenate([records[i]["transfer"] for i in indices])
    obs_anomaly = np.concatenate([records[i]["obs_anomaly"] for i in indices])
    local_anomaly = np.concatenate([records[i]["local_anomaly"] for i in indices])
    transfer_anomaly = np.concatenate([records[i]["transfer_anomaly"] for i in indices])

    record_local = _fit(fields["metered_volume_total_af"], fields["local_volume_total_af"])
    record_transfer = _fit(fields["metered_volume_total_af"], fields["transfer_volume_total_af"])
    between_local = _fit(fields["metered_mean_mm"], fields["local_mean_mm"])
    between_transfer = _fit(fields["metered_mean_mm"], fields["transfer_mean_mm"])
    pooled_local = _fit(obs, local)
    pooled_transfer = _fit(obs, transfer)
    anomaly_local = _fit(obs_anomaly, local_anomaly)
    anomaly_transfer = _fit(obs_anomaly, transfer_anomaly)

    local_bias = fields["local_field_bias_pct"].to_numpy(float)
    transfer_bias = fields["transfer_field_bias_pct"].to_numpy(float)
    local_temporal = fields["local_temporal_r"].to_numpy(float)
    transfer_temporal = fields["transfer_temporal_r"].to_numpy(float)

    return {
        "record_total_volume_r": (record_local["r"], record_transfer["r"]),
        "record_total_volume_nse": (record_local["nse"], record_transfer["nse"]),
        "record_total_volume_slope": (record_local["slope"], record_transfer["slope"]),
        "record_total_volume_abs_slope_error": (
            abs(record_local["slope"] - 1.0),
            abs(record_transfer["slope"] - 1.0),
        ),
        "record_total_volume_bias_pct": (
            record_local["bias_pct"],
            record_transfer["bias_pct"],
        ),
        "record_total_volume_abs_bias_pct": (
            abs(record_local["bias_pct"]),
            abs(record_transfer["bias_pct"]),
        ),
        "between_field_mean_depth_r": (between_local["r"], between_transfer["r"]),
        "pooled_field_year_depth_r": (pooled_local["r"], pooled_transfer["r"]),
        "pooled_field_year_depth_rmse_mm": (pooled_local["rmse"], pooled_transfer["rmse"]),
        "pooled_field_year_depth_bias_pct": (
            pooled_local["bias_pct"],
            pooled_transfer["bias_pct"],
        ),
        "pooled_field_year_depth_abs_bias_pct": (
            abs(pooled_local["bias_pct"]),
            abs(pooled_transfer["bias_pct"]),
        ),
        "within_field_anomaly_r": (anomaly_local["r"], anomaly_transfer["r"]),
        "within_field_anomaly_slope": (anomaly_local["slope"], anomaly_transfer["slope"]),
        "within_field_anomaly_abs_slope_error": (
            abs(anomaly_local["slope"] - 1.0),
            abs(anomaly_transfer["slope"] - 1.0),
        ),
        "median_field_bias_pct": (float(np.median(local_bias)), float(np.median(transfer_bias))),
        "median_abs_field_bias_pct": (
            float(np.median(np.abs(local_bias))),
            float(np.median(np.abs(transfer_bias))),
        ),
        "fraction_fields_within_20pct": (
            float(np.mean(np.abs(local_bias) <= 20.0)),
            float(np.mean(np.abs(transfer_bias) <= 20.0)),
        ),
        "fraction_fields_within_30pct": (
            float(np.mean(np.abs(local_bias) <= 30.0)),
            float(np.mean(np.abs(transfer_bias) <= 30.0)),
        ),
        "median_field_temporal_r": (
            _finite_median(local_temporal),
            _finite_median(transfer_temporal),
        ),
        "fraction_fields_temporal_r_gt_0_5": (
            _finite_fraction_above(local_temporal, 0.5),
            _finite_fraction_above(transfer_temporal, 0.5),
        ),
    }


def _scope_strata(fields: pd.DataFrame, scope: str) -> list[np.ndarray]:
    if scope == "all":
        return [
            np.flatnonzero(fields["basin"].to_numpy() == basin)
            for basin in sorted(fields["basin"].unique())
        ]
    return [np.flatnonzero(fields["basin"].to_numpy() == scope)]


def _sample_field_indices(strata: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Sample each field stratum at its observed size, with replacement."""
    return np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in strata])


def bootstrap_comparison(
    prepared: dict,
    bootstrap_reps: int = 10_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact interval summary and field-bootstrap delta replicates."""
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be positive")

    fields = prepared["fields"]
    scopes = ["all", *sorted(fields["basin"].unique())]
    summary_rows = []
    replicate_rows = []

    for scope_index, scope in enumerate(scopes):
        strata = _scope_strata(fields, scope)
        if any(len(idx) < 3 for idx in strata):
            raise ValueError(f"Scope {scope!r} has a stratum with fewer than three fields")
        point_indices = np.concatenate(strata)
        point = _comparison(prepared, point_indices)
        rng = np.random.default_rng(np.random.SeedSequence([seed, scope_index]))

        distributions = {metric: np.empty(bootstrap_reps, dtype=float) for metric in METRIC_META}
        for replicate in range(bootstrap_reps):
            sampled = _sample_field_indices(strata, rng)
            values = _comparison(prepared, sampled)
            row = {"scope": scope, "replicate": replicate}
            for metric, (local_value, transfer_value) in values.items():
                delta = transfer_value - local_value
                distributions[metric][replicate] = delta
                row[metric] = delta
            replicate_rows.append(row)

        n_fields = int(len(point_indices))
        n_field_years = int(fields.iloc[point_indices]["n_years"].sum())
        for metric, (local_value, transfer_value) in point.items():
            finite = distributions[metric][np.isfinite(distributions[metric])]
            if not len(finite):
                lower = upper = np.nan
            else:
                lower, upper = np.quantile(finite, [0.025, 0.975])
            unit, favorable = METRIC_META[metric]
            summary_rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "unit": unit,
                    "favorable_direction": favorable,
                    "n_fields": n_fields,
                    "n_field_years": n_field_years,
                    "local_value": local_value,
                    "transfer_value": transfer_value,
                    "delta_transfer_minus_local": transfer_value - local_value,
                    "bootstrap_seed": seed,
                    "bootstrap_reps": bootstrap_reps,
                    "n_finite_replicates": int(len(finite)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "interval_excludes_zero": bool(lower > 0 or upper < 0)
                    if np.isfinite(lower) and np.isfinite(upper)
                    else False,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(replicate_rows)


def _load_control_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _validate_control_summaries(local: dict | None, transfer: dict | None) -> None:
    if local is None and transfer is None:
        return
    if local is None or transfer is None:
        raise ValueError("Negative-control summary exists for only one parameter path")
    for key in ("n_control_fields", "n_control_field_years"):
        if local.get(key) != transfer.get(key):
            raise ValueError(f"Negative-control cohorts disagree for {key}")
    for label, summary in (("local", local), ("transfer", transfer)):
        if summary.get("sim_applied_mm_max") != 0.0:
            raise ValueError(f"{label} negative controls contain nonzero simulated irrigation")
        if summary.get("frac_years_gt_10mm") != 0.0:
            raise ValueError(f"{label} negative controls contain years above 10 mm")


def _reconcile_point_estimates(
    summary: pd.DataFrame,
    stats_path: Path,
    value_column: str,
) -> dict | None:
    if not stats_path.exists():
        return None
    stats = json.loads(stats_path.read_text())
    point = summary.loc[summary["scope"] == "all"].set_index("metric")
    checked = {}
    for metric, (group, key, decimals) in POINT_RECONCILIATION.items():
        calculated = float(point.loc[metric, value_column])
        archived = float(stats[group][key])
        rounded = round(calculated, decimals)
        if not np.isclose(rounded, archived, rtol=0.0, atol=10 ** (-(decimals + 2))):
            raise ValueError(
                f"Point-estimate reconciliation failed for {metric}: "
                f"calculated={calculated}, rounded={rounded}, archived={archived}"
            )
        checked[metric] = {
            "calculated": calculated,
            "archived": archived,
            "rounding_decimals": decimals,
        }
    return {
        "status": "passed",
        "stats_path": str(stats_path.resolve()),
        "stats_sha256": _sha256(stats_path),
        "n_checks": len(checked),
        "checks": checked,
    }


def run_analysis(
    local_path: Path,
    transfer_path: Path,
    out_dir: Path,
    bootstrap_reps: int = 10_000,
    seed: int = 42,
    min_years: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the reconciled analysis, write auditable outputs, and return them."""
    local_path = Path(local_path)
    transfer_path = Path(transfer_path)
    out_dir = Path(out_dir)
    paired = load_paired_inputs(local_path, transfer_path)
    prepared = prepare_fields(paired, min_years=min_years)
    summary, replicates = bootstrap_comparison(
        prepared,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )

    local_controls = _load_control_summary(local_path.parent / "negative_controls.json")
    transfer_controls = _load_control_summary(transfer_path.parent / "negative_controls.json")
    _validate_control_summaries(local_controls, transfer_controls)
    local_reconciliation = _reconcile_point_estimates(
        summary,
        local_path.parent / "field_accuracy_stats.json",
        "local_value",
    )
    transfer_reconciliation = _reconcile_point_estimates(
        summary,
        transfer_path.parent / "field_accuracy_stats.json",
        "transfer_value",
    )
    metadata = {
        "analysis": "paired basin-stratified field-clustered bootstrap",
        "delta_definition": "transfer minus local calibration",
        "bootstrap_interval": "2.5th and 97.5th percentiles",
        "bootstrap_seed": seed,
        "bootstrap_reps": bootstrap_reps,
        "minimum_years_for_temporal_r": min_years,
        "local_input": str(local_path.resolve()),
        "local_input_sha256": _sha256(local_path),
        "transfer_input": str(transfer_path.resolve()),
        "transfer_input_sha256": _sha256(transfer_path),
        "n_fields": int(prepared["fields"].shape[0]),
        "n_field_years": int(paired.shape[0]),
        "basin_field_counts": {
            str(k): int(v)
            for k, v in prepared["fields"]["basin"].value_counts().sort_index().items()
        },
        "key_columns": KEY_COLUMNS,
        "keys_identical": True,
        "sampling_unit": "field",
        "sampling_strata": "basin",
        "local_negative_controls": local_controls,
        "transfer_negative_controls": transfer_controls,
        "local_point_estimate_reconciliation": local_reconciliation,
        "transfer_point_estimate_reconciliation": transfer_reconciliation,
        "metric_metadata": {
            metric: {"unit": unit, "favorable_direction": favorable}
            for metric, (unit, favorable) in METRIC_META.items()
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out_dir / "paired_field_years.csv", index=False)
    prepared["fields"].to_csv(out_dir / "paired_field_metrics.csv", index=False)
    summary.to_csv(
        out_dir / "paired_field_bootstrap_summary.csv", index=False, float_format="%.12g"
    )
    replicates.to_csv(
        out_dir / "paired_field_bootstrap_replicates.csv", index=False, float_format="%.12g"
    )
    (out_dir / "paired_field_bootstrap_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return summary, replicates, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="project TOML (default: this example's)")
    parser.add_argument("--local", type=Path, default=None)
    parser.add_argument("--transfer", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-years", type=int, default=4)
    args = parser.parse_args()

    local = (
        args.local or ex7_paths.eval_dir(ex7_paths.LOCAL_LABEL, args.config) / "per_field_year.csv"
    )
    transfer = (
        args.transfer or ex7_paths.eval_dir(TRANSFER_LABEL, args.config) / "per_field_year.csv"
    )
    out = args.out or ex7_paths.eval_dir(OUT_LABEL, args.config)

    summary, _, metadata = run_analysis(
        local_path=local,
        transfer_path=transfer,
        out_dir=out,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
        min_years=args.min_years,
    )
    print(
        f"paired fields={metadata['n_fields']} field-years={metadata['n_field_years']} "
        f"reps={metadata['bootstrap_reps']}"
    )
    print(summary.loc[summary["scope"] == "all"].to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
