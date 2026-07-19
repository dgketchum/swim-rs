"""Run 22 overpass/non-overpass ET decomposition (evaluation-only).

Decomposes the canonical Run 22 daily flux evaluation into direct-benchmark
overpass days (dates with a finite raw Volk v2.1 ensemble_mean_3x3 value
before interpolation) and non-overpass days (paired dates whose OpenET value
exists only through linear interpolation). Consumes frozen Run 22 archive
CSVs and the raw Volk daily extractions only — it does not open the
container, rerun SWIM-RS, or touch any calibration artifact.

The archived site timeseries carry an ``is_overpass`` column derived from
the PEST calibration target (non-null observed_etf). That is a
calibration-capture flag, not a benchmark-retrieval flag, and it is
preserved in the audit output as ``is_calibration_capture`` counts. The
overpass/non-overpass split here is defined exclusively by the raw Volk
benchmark series.

Usage:
    uv run python overpass_decomposition.py \
        --run-dir /data/ssd1/swim/5_Flux_Ensemble/results/run22 \
        --openet-daily-dir /data/ssd1/swim/5_Flux_Ensemble/data/openet_flux/daily_data \
        --output-dir /data/ssd1/swim/5_Flux_Ensemble/results/run22/overpass_decomposition
"""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

MIN_PAIRED = 10
SUBSETS = ["all_days", "overpass", "non_overpass"]
METRIC_KEYS = ["nse", "kge", "r", "rmse", "mbe"]
DELTA_METRICS = ["nse", "kge", "rmse", "abs_mbe"]
IDENTITY_TOL = 1e-12
INTERPOLATION_RULE = (
    "raw Volk ensemble_mean_3x3 reindexed to a daily calendar spanning its first "
    "through last finite value, then pandas linear interpolation; no extrapolation "
    "outside that range (matches evaluate.py openet_source='volk')"
)


def calc_metrics(obs, mod, min_n=MIN_PAIRED):
    """NSE, KGE (Gupta 2009), Pearson r, RMSE, MBE — same math as evaluate.py.

    evaluate.py names NSE 'r2' (sklearn r2_score = 1 - SSE/SST); publication
    outputs here use 'nse'. MBE = mean(mod - obs), evaluate.py's 'bias'.
    """
    mask = np.isfinite(obs) & np.isfinite(mod)
    obs, mod = obs[mask], mod[mask]
    if len(obs) < min_n:
        return {"n": len(obs), **{k: np.nan for k in METRIC_KEYS}}
    r, _ = stats.pearsonr(obs, mod)
    nse = r2_score(obs, mod)
    rmse = np.sqrt(mean_squared_error(obs, mod))
    mbe = float((mod - obs).mean())
    alpha = np.std(mod) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(mod) / np.mean(obs) if np.mean(obs) > 0 else np.nan
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return {"n": len(obs), "nse": nse, "kge": kge, "r": r, "rmse": rmse, "mbe": mbe}


def reconstruct_benchmark(raw_ensemble):
    """Return (daily interpolated benchmark, direct benchmark dates).

    direct_dates are the finite raw values BEFORE interpolation — these define
    is_benchmark_overpass. The daily series spans first→last finite raw date
    and is linearly interpolated with no extrapolation.
    """
    s = pd.to_numeric(raw_ensemble, errors="coerce")
    if s.index.duplicated().any():
        raise ValueError("duplicate dates in raw benchmark series")
    finite = s[np.isfinite(s.values)]
    if finite.empty:
        raise ValueError("no finite raw benchmark values")
    direct_dates = finite.index
    daily_idx = pd.date_range(direct_dates.min(), direct_dates.max(), freq="D")
    daily = s.reindex(daily_idx).interpolate(method="linear")
    return daily, direct_dates


def classify_paired(frozen, bench_daily, direct_dates):
    """Pair flux/SWIM/benchmark by date and flag benchmark overpasses.

    Returns a DataFrame (index=date) with flux, swim, openet columns on dates
    where all three are finite, plus is_benchmark_overpass.
    """
    bench = bench_daily.reindex(frozen.index)
    df = pd.DataFrame(
        {
            "flux": frozen["flux_ET"].astype(float),
            "swim": frozen["swim_ET"].astype(float),
            "openet": bench.astype(float),
        },
        index=frozen.index,
    )
    paired = df[np.isfinite(df.values).all(axis=1)].copy()
    paired["is_benchmark_overpass"] = paired.index.isin(direct_dates)
    return paired


def subset_frames(paired):
    return {
        "all_days": paired,
        "overpass": paired[paired["is_benchmark_overpass"]],
        "non_overpass": paired[~paired["is_benchmark_overpass"]],
    }


def check_date_semantics(paired, direct_dates, fid):
    """Gate B/C invariants for one site; raises on violation."""
    over = paired[paired["is_benchmark_overpass"]]
    non = paired[~paired["is_benchmark_overpass"]]
    if not over.index.isin(direct_dates).all():
        raise AssertionError(f"{fid}: overpass date without finite raw benchmark value")
    if non.index.isin(direct_dates).any():
        raise AssertionError(f"{fid}: non_overpass date has a raw benchmark value")
    if not np.isfinite(non["openet"].values).all():
        raise AssertionError(f"{fid}: non_overpass date lacks finite interpolated value")
    lo, hi = direct_dates.min(), direct_dates.max()
    if (paired.index < lo).any() or (paired.index > hi).any():
        raise AssertionError(f"{fid}: paired date outside raw benchmark support")
    if len(over) + len(non) != len(paired):
        raise AssertionError(f"{fid}: overpass + non_overpass != all_days")


def bootstrap_median_ci(deltas, reps, seed):
    """95% site-bootstrap CI of the median paired delta. Resamples sites."""
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    medians = np.median(deltas[idx], axis=1)
    return (
        float(np.median(deltas)),
        float(np.percentile(medians, 2.5)),
        float(np.percentile(medians, 97.5)),
    )


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(repo_dir):
    def run(args):
        return subprocess.run(
            args, cwd=repo_dir, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "sha": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty_paths": len(run(["git", "status", "--porcelain"]).splitlines()),
    }


def _iqr(values):
    v = pd.Series(values).dropna()
    return float(v.quantile(0.25)), float(v.quantile(0.75))


def build_summary_row(subset, cohort, rows):
    """Cohort summary for one subset from eligible per-site metric rows."""
    days = rows["n_paired"].values
    q25, q75 = _iqr(days)
    out = {
        "subset": subset,
        "cohort": cohort,
        "n_sites": len(rows),
        "total_paired_site_days": int(days.sum()),
        "median_paired_days_per_site": float(np.median(days)),
        "iqr25_paired_days": q25,
        "iqr75_paired_days": q75,
    }
    for model in ["swim", "openet"]:
        for k in METRIC_KEYS:
            col = f"{k}_{model}"
            vals = rows[col]
            lo, hi = _iqr(vals)
            out[f"median_{col}"] = float(vals.median())
            out[f"iqr25_{col}"] = lo
            out[f"iqr75_{col}"] = hi
    return out


def gate_a_identity(metrics_df, canonical_csv, kge_reference_csv):
    """Gate A: reconstructed all_days must equal the archived Run 22 evaluation."""
    canon = pd.read_csv(canonical_csv, index_col="fid")
    kge_ref = pd.read_csv(kge_reference_csv, index_col="fid")
    all_days = metrics_df[metrics_df["subset"] == "all_days"].set_index("fid")

    if sorted(all_days.index) != sorted(canon.index):
        raise AssertionError("Gate A: site sets differ from canonical daily metrics")

    checks = [
        ("n_paired", canon["n"], 0.5),
        ("nse_swim", canon["r2_swim"], IDENTITY_TOL),
        ("r_swim", canon["r_swim"], IDENTITY_TOL),
        ("rmse_swim", canon["rmse_swim"], IDENTITY_TOL),
        ("mbe_swim", canon["bias_swim"], IDENTITY_TOL),
        ("nse_openet", canon["r2_ensemble"], IDENTITY_TOL),
        ("r_openet", canon["r_ensemble"], IDENTITY_TOL),
        ("rmse_openet", canon["rmse_ensemble"], IDENTITY_TOL),
        ("mbe_openet", canon["bias_ensemble"], IDENTITY_TOL),
        ("kge_swim", kge_ref["kge_swim"], IDENTITY_TOL),
        ("kge_openet", kge_ref["kge_ensemble"], IDENTITY_TOL),
    ]
    max_diffs = {}
    for col, ref, tol in checks:
        diff = (all_days[col] - ref.reindex(all_days.index)).abs().max()
        max_diffs[col] = float(diff)
        if not diff <= tol:
            raise AssertionError(f"Gate A: {col} max |diff| {diff:.3e} exceeds {tol:.0e}")
    return max_diffs


def main():
    parser = argparse.ArgumentParser(
        description="Run 22 overpass/non-overpass decomposition from frozen artifacts"
    )
    parser.add_argument("--run-dir", required=True, help="Run 22 results directory")
    parser.add_argument(
        "--openet-daily-dir", required=True, help="Raw Volk daily_data CSV directory"
    )
    parser.add_argument("--output-dir", required=True, help="Working output directory")
    parser.add_argument(
        "--kge-reference-csv",
        default=None,
        help="KGE-emitting exact Run 22 reproduction (default: sibling "
        "ablation_run22_e1_spread/evaluation_metrics.csv)",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    eval_dir = run_dir / "archive" / "6_evaluation"
    canonical_csv = eval_dir / "daily_paired_metrics.csv"
    timeseries_dir = eval_dir / "site_daily_timeseries"
    eval_metadata = eval_dir / "evaluation_metadata.json"
    kge_reference_csv = Path(
        args.kge_reference_csv
        or run_dir.parent / "ablation_run22_e1_spread" / "evaluation_metrics.csv"
    )
    openet_dir = Path(args.openet_daily_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fids = sorted(pd.read_csv(canonical_csv)["fid"])
    print(f"Canonical Run 22 daily cohort: {len(fids)} sites")

    metric_rows, audit_rows, input_manifest = [], [], {}
    for fid in fids:
        frozen_path = timeseries_dir / f"{fid}.csv"
        volk_path = openet_dir / f"{fid}.csv"
        frozen = pd.read_csv(frozen_path, index_col="date", parse_dates=True)
        volk = pd.read_csv(volk_path, index_col="DATE", parse_dates=True)
        input_manifest[f"frozen/{fid}.csv"] = sha256_file(frozen_path)
        input_manifest[f"volk/{fid}.csv"] = sha256_file(volk_path)

        bench_daily, direct_dates = reconstruct_benchmark(volk["ensemble_mean_3x3"])
        paired = classify_paired(frozen, bench_daily, direct_dates)
        check_date_semantics(paired, direct_dates, fid)

        calib_dates = frozen.index[frozen["is_overpass"].astype(bool)]
        n_inter = len(calib_dates.intersection(direct_dates))
        subs = subset_frames(paired)
        audit_rows.append(
            {
                "fid": fid,
                "n_calibration_captures": len(calib_dates),
                "n_benchmark_captures": len(direct_dates),
                "n_intersection": n_inter,
                "n_calibration_only": len(calib_dates) - n_inter,
                "n_benchmark_only": len(direct_dates) - n_inter,
                "n_paired_all_days": len(subs["all_days"]),
                "n_paired_overpass": len(subs["overpass"]),
                "n_paired_non_overpass": len(subs["non_overpass"]),
            }
        )

        for subset, sdf in subs.items():
            obs = sdf["flux"].values
            row = {
                "fid": fid,
                "subset": subset,
                "n_paired": len(sdf),
                "first_date": sdf.index.min().date().isoformat() if len(sdf) else "",
                "last_date": sdf.index.max().date().isoformat() if len(sdf) else "",
                "eligible": len(sdf) >= MIN_PAIRED,
                "exclusion_reason": ""
                if len(sdf) >= MIN_PAIRED
                else f"n_paired={len(sdf)} < {MIN_PAIRED}",
            }
            for model in ["swim", "openet"]:
                m = calc_metrics(obs, sdf[model].values)
                for k in METRIC_KEYS:
                    row[f"{k}_{model}"] = m[k]
            metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    audit_df = pd.DataFrame(audit_rows)

    # Gate C: per-site union identity (also enforced per-site in check_date_semantics)
    bad = audit_df[
        audit_df["n_paired_overpass"] + audit_df["n_paired_non_overpass"]
        != audit_df["n_paired_all_days"]
    ]
    if not bad.empty:
        raise AssertionError(f"Gate C: union violation at {list(bad['fid'])}")

    print("Running Gate A (45-site all-days identity vs canonical Run 22)...")
    gate_a_diffs = gate_a_identity(metrics_df, canonical_csv, kge_reference_csv)
    print(
        "Gate A passed: max per-column |diff| "
        f"{max(gate_a_diffs.values()):.2e} (tolerance {IDENTITY_TOL:.0e})"
    )

    # Cohorts
    eligible = metrics_df[metrics_df["eligible"]]
    eligible_by_subset = {s: set(eligible.loc[eligible["subset"] == s, "fid"]) for s in SUBSETS}
    common_split = sorted(eligible_by_subset["overpass"] & eligible_by_subset["non_overpass"])
    print(f"Common-split cohort: {len(common_split)} sites")

    summary_rows = []
    for subset in SUBSETS:
        sub_rows = eligible[eligible["subset"] == subset]
        summary_rows.append(build_summary_row(subset, "subset_eligible", sub_rows))
        summary_rows.append(
            build_summary_row(subset, "common_split", sub_rows[sub_rows["fid"].isin(common_split)])
        )
    summary_df = pd.DataFrame(summary_rows)

    # Non-overpass fraction: site-median and pooled, per cohort, from audit counts
    fractions = {}
    for cohort, cohort_fids in [
        ("subset_eligible", sorted(eligible_by_subset["non_overpass"])),
        ("common_split", common_split),
    ]:
        a = audit_df[audit_df["fid"].isin(cohort_fids)]
        site_frac = a["n_paired_non_overpass"] / a["n_paired_all_days"]
        fractions[cohort] = {
            "site_median_non_overpass_fraction": float(site_frac.median()),
            "pooled_non_overpass_fraction": float(
                a["n_paired_non_overpass"].sum() / a["n_paired_all_days"].sum()
            ),
        }
    for cohort, vals in fractions.items():
        mask = summary_df["cohort"] == cohort
        for k, v in vals.items():
            summary_df.loc[mask, k] = v

    # Paired site-level deltas with site-bootstrap CIs
    delta_rows = []
    wide = metrics_df.set_index(["fid", "subset"])
    for cohort, cohort_fids in [
        ("common_split", common_split),
        ("subset_eligible", None),
    ]:
        for subset in SUBSETS:
            sub = eligible[eligible["subset"] == subset]
            if cohort_fids is not None:
                sub = sub[sub["fid"].isin(cohort_fids)]
            deltas = {
                "nse": sub["nse_swim"] - sub["nse_openet"],
                "kge": sub["kge_swim"] - sub["kge_openet"],
                "rmse": sub["rmse_swim"] - sub["rmse_openet"],
                "abs_mbe": sub["mbe_swim"].abs() - sub["mbe_openet"].abs(),
            }
            for metric in DELTA_METRICS:
                med, lo, hi = bootstrap_median_ci(
                    deltas[metric].values, args.bootstrap_reps, args.seed
                )
                delta_rows.append(
                    {
                        "metric": metric,
                        "subset": subset,
                        "cohort": cohort,
                        "n_sites": len(sub),
                        "median_delta_swim_minus_openet": med,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "seed": args.seed,
                        "n_resamples": args.bootstrap_reps,
                    }
                )
    deltas_df = pd.DataFrame(delta_rows)
    _ = wide  # metrics indexed by (fid, subset); retained for interactive inspection

    metadata = {
        "run_name": "run22_overpass_decomposition",
        "analysis_date": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(str(Path(__file__).resolve().parents[2])),
        "inputs": {
            "canonical_daily_metrics": {
                "path": str(canonical_csv),
                "sha256": sha256_file(canonical_csv),
            },
            "kge_reference": {
                "path": str(kge_reference_csv),
                "sha256": sha256_file(kge_reference_csv),
            },
            "evaluation_metadata": {
                "path": str(eval_metadata),
                "sha256": sha256_file(eval_metadata),
            },
            "site_daily_timeseries_dir": str(timeseries_dir),
            "openet_daily_dir": str(openet_dir),
            "per_site_manifest_sha256": input_manifest,
        },
        "method": {
            "benchmark_overpass_definition": "finite raw Volk ensemble_mean_3x3 "
            "before interpolation",
            "interpolation_rule": INTERPOLATION_RULE,
            "min_paired_dates": MIN_PAIRED,
            "nse": "1 - SSE/SST (sklearn r2_score); labeled NSE, never r2",
            "kge": "Gupta et al. 2009: 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2), "
            "alpha = std ratio (ddof=0), beta = mean ratio",
            "mbe": "mean(model - flux), mm/day",
            "bootstrap": "site resampling with replacement; per-(subset,cohort) "
            "index matrix drawn once from default_rng(seed) and shared across metrics",
            "seed": args.seed,
            "bootstrap_reps": args.bootstrap_reps,
        },
        "gates": {
            "gate_a_identity_max_abs_diff_by_column": gate_a_diffs,
            "gate_a_tolerance": IDENTITY_TOL,
            "gate_b_date_semantics": "enforced per site (check_date_semantics)",
            "gate_c_union_and_pairing": "enforced per site and via audit counts",
            "common_split_cohort": common_split,
            "n_common_split_sites": len(common_split),
        },
        "cohort_fractions": fractions,
    }

    metrics_out = metrics_df[
        ["fid", "subset", "n_paired", "first_date", "last_date", "eligible", "exclusion_reason"]
        + [f"{k}_{m}" for m in ["swim", "openet"] for k in METRIC_KEYS]
    ]
    metrics_out.to_csv(out_dir / "overpass_split_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "overpass_split_summary.csv", index=False)
    deltas_df.to_csv(out_dir / "overpass_split_paired_deltas.csv", index=False)
    audit_df.to_csv(out_dir / "overpass_date_audit.csv", index=False)
    with open(out_dir / "overpass_split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nOutputs written to {out_dir}")
    print("\nCommon-split summary (medians):")
    show = summary_df[summary_df["cohort"] == "common_split"]
    cols = [
        "subset",
        "n_sites",
        "total_paired_site_days",
        "median_paired_days_per_site",
        "median_nse_swim",
        "median_nse_openet",
        "median_kge_swim",
        "median_kge_openet",
        "median_rmse_swim",
        "median_rmse_openet",
        "median_mbe_swim",
        "median_mbe_openet",
    ]
    print(show[cols].to_string(index=False))
    print("\nPaired deltas (common_split):")
    print(deltas_df[deltas_df["cohort"] == "common_split"].to_string(index=False))


if __name__ == "__main__":
    main()
