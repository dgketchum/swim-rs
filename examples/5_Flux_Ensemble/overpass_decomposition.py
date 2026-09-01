"""Run 22 overpass/non-overpass ET decomposition (evaluation-only).

Decomposes the canonical Run 22 daily flux evaluation into direct-benchmark
overpass days (dates with a finite raw May v2.1 ensemble_mean_3x3 capture
before interpolation) and non-overpass days (paired dates whose OpenET value
exists only through the ETf-first temporal reconstruction). Consumes frozen
Run 22 archive CSVs, the raw May v2.1 daily extractions, and the pinned
OpenET bias-corrected gridMET ETo — it does not open the container, rerun
SWIM-RS, or touch any calibration artifact.

The daily benchmark is reconstructed ETf-first with the shared helper
(``swimrs.calibrate.benchmark.reconstruct_daily_benchmark``): capture-date ET
is divided by the common ETo, ETf is interpolated under the Volk et al.
(2024) ±32-day temporal-support rule (openet-core semantics: linear when
both anchors are inside the window, one-sided flat fill otherwise, NaN when
neither), then multiplied by the same daily ETo. The archived
``site_daily_timeseries`` ``eto`` column is raw gridMET (ancillary only) and
is FORBIDDEN here; the sole benchmark ETo basis is
``data/openet_refet/openet_eto.csv``.

The archived site timeseries carry an ``is_overpass`` column derived from
the PEST calibration target (non-null observed_etf). That is a
calibration-capture flag, not a benchmark-retrieval flag, and it is
preserved in the audit output as ``is_calibration_capture`` counts. The
overpass/non-overpass split here is defined exclusively by the raw May v2.1
benchmark series.

The scoring cohort is supplied by the May v2.1 evidence rebuild
(``rebuild_e1_benchmark_evidence.py``) via ``--cohort-csv``; the
January-based archived ``daily_paired_metrics.csv`` is a superseded source
and is no longer read.

Usage:
    uv run python overpass_decomposition.py \
        --run-dir /data/ssd1/swim/5_Flux_Ensemble/results/run22 \
        --openet-daily-dir /data/ssd1/swim/5_Flux_Ensemble/data/openet_flux_2pt1/daily_data \
        --cohort-csv <rebuild output>/e2_primary_daily_site_metrics.csv \
        --output-dir /data/ssd1/swim/5_Flux_Ensemble/results/run22/archive/6_evaluation/overpass_split_etf_first_2pt1 \
        --support-contrast
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

from swimrs.calibrate.benchmark import (
    VOLK_WINDOW_DAYS,
    assert_inside_support,
    reconstruct_daily_benchmark,
)

MIN_PAIRED = 10
SUBSETS = ["all_days", "overpass", "non_overpass"]
METRIC_KEYS = ["nse", "kge", "r", "rmse", "mbe"]
DELTA_METRICS = ["nse", "kge", "rmse", "abs_mbe"]
IDENTITY_TOL = 1e-12
ETO_SOURCE = "openet_refet/openet_eto.csv"
INTERPOLATION_RULE = (
    "ETf-first: capture ETf = raw May v2.1 ensemble_mean_3x3 / OpenET bias-corrected "
    "gridMET ETo on capture dates; ETf interpolated in time under the Volk et al. "
    f"(2024) ±{VOLK_WINDOW_DAYS}-day temporal-support rule with openet-core semantics "
    "(linear when both anchors are within the window, one-sided flat fill otherwise "
    f"— including ≤{VOLK_WINDOW_DAYS}-day extension beyond the first/last capture — "
    "NaN when neither); then multiplied by the same daily ETo "
    "(swimrs.calibrate.benchmark.reconstruct_daily_benchmark; matches evaluate.py "
    "openet_source='volk'). Direct interpolation of sparse ET is invalid and must "
    "never be reintroduced here."
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


def assert_may_source(path):
    """Reject the superseded January capture set; require the May v2.1 dirs."""
    parts = Path(path).resolve().parts
    if "openet_flux" in parts or "openet_flux_2pt1" not in parts:
        raise ValueError(
            f"OpenET daily source must come from openet_flux_2pt1 (May v2.1); "
            f"got {path} — the January openet_flux/ set is superseded"
        )
    return Path(path)


def load_openet_eto(path):
    """Load the pinned OpenET bias-corrected gridMET ETo (dates × sites).

    This is the ONLY valid ETo basis for benchmark reconstruction. The
    archived site_daily_timeseries ``eto`` column is raw gridMET (ancillary
    only) and is forbidden here.
    """
    wide = pd.read_csv(path, index_col="site_id")
    wide.columns = pd.to_datetime(wide.columns, format="%Y%m%d")
    return wide.T.sort_index()


def reconstruct_benchmark(raw_ensemble, eto, label=""):
    """Return (daily benchmark ET, direct benchmark dates, reconstruction).

    Delegates to the shared ETf-first helper: capture-date ET is divided by
    the common ETo, ETf is interpolated under the Volk ±32-day window
    (openet-core semantics), and the result is multiplied by the same daily
    ETo. direct_dates are the finite raw captures BEFORE interpolation —
    these define is_benchmark_overpass. A capture date without ETo coverage
    (e.g. pre-1999) is a hard failure in the helper, never a silent fill.
    """
    s = pd.to_numeric(raw_ensemble, errors="coerce")
    recon = reconstruct_daily_benchmark(
        capture_series=s,
        capture_space="et",
        eto=eto,
        eto_name=ETO_SOURCE,
        label=label,
    )
    return recon.daily_et, recon.capture_dates, recon


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


def check_date_semantics(paired, recon, fid):
    """Gate B/C invariants for one site; raises on violation."""
    direct_dates = recon.capture_dates
    over = paired[paired["is_benchmark_overpass"]]
    non = paired[~paired["is_benchmark_overpass"]]
    if not over.index.isin(direct_dates).all():
        raise AssertionError(f"{fid}: overpass date without finite raw benchmark value")
    if non.index.isin(direct_dates).any():
        raise AssertionError(f"{fid}: non_overpass date has a raw benchmark value")
    if not np.isfinite(non["openet"].values).all():
        raise AssertionError(f"{fid}: non_overpass date lacks finite interpolated value")
    assert_inside_support(paired.index, recon, label=fid)
    support = recon.support_class.reindex(paired.index)
    if support.isna().any() or (support == "unsupported").any():
        raise AssertionError(f"{fid}: paired date outside benchmark temporal support")
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


def gate_a_identity(metrics_df, cohort_csv):
    """Gate A: reconstructed all_days must equal the May v2.1 rebuild metrics.

    Replaces the former identity check against the January-based archived
    ``daily_paired_metrics.csv`` (superseded source); the replacement is
    recorded in the output metadata. The reference is the rebuild's
    ``e2_primary_daily_site_metrics.csv`` (evaluate.py schema, KGE included).
    """
    ref = pd.read_csv(cohort_csv, index_col="fid")
    all_days = metrics_df[metrics_df["subset"] == "all_days"].set_index("fid")

    if sorted(all_days.index) != sorted(ref.index):
        raise AssertionError("Gate A: site sets differ from the rebuild daily metrics")

    checks = [
        ("n_paired", ref["n"], 0.5),
        ("nse_swim", ref["r2_swim"], IDENTITY_TOL),
        ("r_swim", ref["r_swim"], IDENTITY_TOL),
        ("rmse_swim", ref["rmse_swim"], IDENTITY_TOL),
        ("mbe_swim", ref["bias_swim"], IDENTITY_TOL),
        ("kge_swim", ref["kge_swim"], IDENTITY_TOL),
        ("nse_openet", ref["r2_ensemble"], IDENTITY_TOL),
        ("r_openet", ref["r_ensemble"], IDENTITY_TOL),
        ("rmse_openet", ref["rmse_ensemble"], IDENTITY_TOL),
        ("mbe_openet", ref["bias_ensemble"], IDENTITY_TOL),
        ("kge_openet", ref["kge_ensemble"], IDENTITY_TOL),
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
        "--openet-daily-dir",
        required=True,
        help="Raw May v2.1 daily_data CSV directory (openet_flux_2pt1)",
    )
    parser.add_argument(
        "--cohort-csv",
        required=True,
        help="Rebuilt e2_primary_daily_site_metrics.csv from "
        "rebuild_e1_benchmark_evidence.py — defines the cohort and the Gate A "
        "identity reference",
    )
    parser.add_argument(
        "--openet-eto-csv",
        default=str(Path(__file__).resolve().parent / "data" / ETO_SOURCE),
        help="Pinned OpenET bias-corrected gridMET ETo (the sole benchmark ETo basis)",
    )
    parser.add_argument("--output-dir", required=True, help="Working output directory")
    parser.add_argument(
        "--support-contrast",
        action="store_true",
        help="Also emit the §4.3 support contrast (all five metrics, "
        "non_overpass − overpass per model) with bootstrap CIs",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    eval_dir = run_dir / "archive" / "6_evaluation"
    timeseries_dir = eval_dir / "site_daily_timeseries"
    eval_metadata = eval_dir / "evaluation_metadata.json"
    cohort_csv = Path(args.cohort_csv)
    openet_dir = assert_may_source(args.openet_daily_dir)
    openet_eto_csv = Path(args.openet_eto_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    openet_eto = load_openet_eto(openet_eto_csv)
    fids = sorted(pd.read_csv(cohort_csv)["fid"])
    print(f"Rebuild daily cohort: {len(fids)} sites")

    metric_rows, audit_rows, input_manifest = [], [], {}
    for fid in fids:
        frozen_path = timeseries_dir / f"{fid}.csv"
        volk_path = openet_dir / f"{fid}.csv"
        frozen = pd.read_csv(frozen_path, index_col="date", parse_dates=True)
        volk = pd.read_csv(volk_path, index_col="DATE", parse_dates=True)
        input_manifest[f"frozen/{fid}.csv"] = sha256_file(frozen_path)
        input_manifest[f"volk/{fid}.csv"] = sha256_file(volk_path)

        if fid not in openet_eto.columns:
            raise AssertionError(f"{fid}: absent from {openet_eto_csv} — do not fill")
        bench_daily, direct_dates, recon = reconstruct_benchmark(
            volk["ensemble_mean_3x3"], openet_eto[fid], label=f"{fid}:ensemble_mean"
        )
        paired = classify_paired(frozen, bench_daily, direct_dates)
        check_date_semantics(paired, recon, fid)

        calib_dates = frozen.index[frozen["is_overpass"].astype(bool)]
        n_inter = len(calib_dates.intersection(direct_dates))
        subs = subset_frames(paired)
        support = recon.support_class.reindex(paired.index)
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
                "n_paired_interpolated": int((support == "interpolated").sum()),
                "n_paired_flat_fill": int((support == "flat_fill").sum()),
                "support_start": recon.support_start.date().isoformat(),
                "support_end": recon.support_end.date().isoformat(),
                "identity_max_abs_err": recon.identity_max_abs_err,
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

    print(f"Running Gate A ({len(fids)}-site all-days identity vs the May rebuild)...")
    gate_a_diffs = gate_a_identity(metrics_df, cohort_csv)
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

    # §4.3 support contrast: per-model metric change from overpass to
    # non_overpass support (all five metrics), common_split cohort only —
    # both subsets eligible by construction there
    contrast_df = contrast_persite = None
    if args.support_contrast:
        by = metrics_df.set_index(["fid", "subset"])
        persite_rows, contrast_rows = [], []
        for fid in common_split:
            for model in ["swim", "openet"]:
                for k in METRIC_KEYS:
                    o = by.loc[(fid, "overpass"), f"{k}_{model}"]
                    nv = by.loc[(fid, "non_overpass"), f"{k}_{model}"]
                    persite_rows.append(
                        {
                            "fid": fid,
                            "model": model,
                            "metric": k,
                            "overpass": o,
                            "non_overpass": nv,
                            "delta_non_minus_overpass": nv - o,
                        }
                    )
        contrast_persite = pd.DataFrame(persite_rows)
        for model in ["swim", "openet"]:
            for k in METRIC_KEYS:
                sub = contrast_persite[
                    (contrast_persite["model"] == model) & (contrast_persite["metric"] == k)
                ]
                med, lo, hi = bootstrap_median_ci(
                    sub["delta_non_minus_overpass"].values, args.bootstrap_reps, args.seed
                )
                contrast_rows.append(
                    {
                        "model": model,
                        "metric": k,
                        "cohort": "common_split",
                        "n_sites": len(sub),
                        "median_delta_non_minus_overpass": med,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "seed": args.seed,
                        "n_resamples": args.bootstrap_reps,
                    }
                )
        contrast_df = pd.DataFrame(contrast_rows)

    metadata = {
        "run_name": "run22_overpass_decomposition",
        "analysis_date": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(str(Path(__file__).resolve().parents[2])),
        "inputs": {
            "cohort_and_gate_a_reference": {
                "path": str(cohort_csv),
                "sha256": sha256_file(cohort_csv),
                "note": "rebuilt May v2.1 e2_primary_daily_site_metrics.csv; replaces "
                "the superseded January-based daily_paired_metrics.csv (and the "
                "separate KGE reference) as both cohort source and Gate A target",
            },
            "openet_eto": {
                "path": str(openet_eto_csv),
                "sha256": sha256_file(openet_eto_csv),
                "note": "sole benchmark ETo basis (OpenET bias-corrected gridMET, "
                "EE asset projects/openet/assets/reference_et/conus/gridmet/daily/v1); "
                "the archived site_daily_timeseries eto column is raw gridMET and is "
                "forbidden for reconstruction",
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
            "benchmark_overpass_definition": "finite raw May v2.1 ensemble_mean_3x3 "
            "before interpolation",
            "benchmark_construction": "OpenET-method temporal benchmark using a common "
            "OpenET bias-corrected gridMET ETo basis; site-series ETf reconstruction "
            "using the Volk et al. temporal-support rule (not a native-product "
            "reproduction)",
            "interpolation_rule": INTERPOLATION_RULE,
            "eto_source": ETO_SOURCE,
            "epoch": "captures and scored dates are confined to the extracted-ETo "
            "support (>= 1999-01-01) structurally: pairing intersects the "
            "reconstruction with the ETo calendar, and a capture without ETo "
            "coverage is a hard failure — never a raw-gridMET backfill",
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
            "gate_a_reference": "rebuilt May v2.1 e2_primary_daily_site_metrics.csv — "
            "consciously replaces the former identity check against the January-based "
            "archived daily_paired_metrics.csv + ablation KGE reference (superseded "
            "sources)",
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
    if contrast_df is not None:
        contrast_df.to_csv(out_dir / "e2_temporal_support_contrast.csv", index=False)
        contrast_persite.to_csv(out_dir / "e2_temporal_support_contrast_persite.csv", index=False)
        metadata["support_contrast"] = {
            "definition": "per-site metric change non_overpass - overpass for each "
            "model, all five metrics, common_split cohort; site-bootstrap 95% CI",
            "seed": args.seed,
            "bootstrap_reps": args.bootstrap_reps,
        }
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
    if contrast_df is not None:
        print("\nSupport contrast (common_split, non_overpass - overpass):")
        print(contrast_df.to_string(index=False))


if __name__ == "__main__":
    main()
