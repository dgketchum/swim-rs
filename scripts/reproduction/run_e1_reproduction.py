"""Example 5 (E1) reproduction, step by step, for walking in the debugger.

This is the E1 block of "3. Recompute the evidence" in examples/README.md
written as direct Python calls instead of command lines. Each step below
imports the Example 5 module and calls its functions the way the script's
``main`` does: the config, the open container, the site list, the posterior,
and every intermediate DataFrame are locals you can inspect, and stepping
into ``ev.evaluate_benchmark_daily`` leads through ``_collect_daily`` to
``run_calibrated_model``, the forward model run.

Run or debug it as-is: the data root and scratch directory are set in
``ROOT`` and ``SCRATCH`` below, and ``STEPS`` selects which steps run. The
repo-local package (swim_repo_local.tar.gz) must be unpacked into the clone.

Two steps call a module's ``main`` directly rather than re-stating its body:
``within_e1_transfer.main`` (its six forward runs and scoring are one long
inline function; step into it) and ``conditioned_ensemble_uncertainty.main``
(a function of keyword arguments). The runner writes the CSVs the README
compares; the provenance sidecars that record command-line arguments are
written only for the evaluator bundles, which the overpass decomposition
validates before it runs.

Comparisons are byte-level, like the README's ``cmp``. A rebuild on other
hardware may differ in the last digit of a few statistics; a cohort, sign, or
file-set difference is a failure.
"""

import filecmp
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EX5 = REPO / "examples" / "5_Flux_Ensemble"
CFG = EX5 / "5_Flux_Ensemble.toml"

# Edit these three to walk the script from the debugger with no configuration.
ROOT = Path.home() / "data" / "swim"  # the data root the swim_data_* packages were unpacked into
SCRATCH = Path.home() / "data" / "swim_scratch"  # outside the clone and the data root
STEPS = []  # e.g. ["daily", "cmp"]; empty means every step in README order

REPS = 10000  # bootstrap replicates (README: --bootstrap-reps 10000)
SEED = 42  # bootstrap seed (README: --bootstrap-seed 42)

os.environ["SWIM_ROOT"] = str(ROOT)  # the TOML root relocation every Example 5 module reads
sys.path.insert(0, str(EX5))  # the modules import each other as top-level names

import conditioned_ensemble_uncertainty as ceu  # noqa: E402
import e1_period_split as ps  # noqa: E402
import evaluate as ev  # noqa: E402
import ex5_paths  # noqa: E402
import overpass_decomposition as od  # noqa: E402
import pandas as pd  # noqa: E402
import run_weighting_ablation as rwa  # noqa: E402
import spread_error as se  # noqa: E402
import within_e1_transfer as wt  # noqa: E402

from swimrs.container import SwimContainer  # noqa: E402

DATA = ROOT / "5_Flux_Ensemble" / "data"
RUN23 = ROOT / "5_Flux_Ensemble" / "results" / "run23"
PAR = RUN23 / "5_Flux_Ensemble.3.par.csv"
CONTAINER = DATA / "5_Flux_Ensemble_run23.swim"
ABLATION_CONTAINER = DATA / "5_Flux_Ensemble_run23ablation.swim"
S = SCRATCH / "e1"

DIFFERED = []  # (scratch file, shipped file) pairs that are not byte-identical


def compare(scratch_dir, shipped_dir, files):
    for f in files:
        a, b = Path(scratch_dir) / f, Path(shipped_dir) / f
        if a.exists() and b.exists() and filecmp.cmp(a, b, shallow=False):
            print(f"  identical: {f}")
        else:
            print(f"  {'differs' if a.exists() and b.exists() else 'missing'}: {a} vs {b}")
            DIFFERED.append((a, b))


# --- 1. daily benchmark: one forward run, grouped estimands -----------------
# README: evaluate.py --openet-source volk --output-dir $S/daily --bootstrap-reps 10000 --bootstrap-seed 42 --quiet-sites
def step_daily():
    out = S / "daily"
    out.mkdir(parents=True, exist_ok=True)
    cfg = ev.load_config(CFG)
    flux_dir = ev.resolve_flux_dir(cfg)
    container = SwimContainer.open(str(CONTAINER), mode="r")
    try:
        fids = ev.apply_exclusions(container.field_uids)
        bundle = ev.evaluate_benchmark_daily(
            cfg,
            container,
            str(PAR),
            fids,
            flux_dir,
            openet_source="volk",
            bootstrap_reps=REPS,
            bootstrap_seed=SEED,
            quiet_sites=True,
            results_dir=str(out),
        )
    finally:
        container.close()
    ev.print_grouped_summary(bundle, "daily")
    written = ev.write_grouped_outputs(
        bundle, str(out), "daily", openet_source="volk", cli_args=_evaluate_cli_args(out)
    )
    bundle.site_metrics.to_csv(out / "evaluation_metrics.csv")
    print(f"Grouped estimates: {written['metrics']}")
    return bundle


# --- 2. monthly benchmark (Volk et al. 2024 monthly protocol) ---------------
# README: evaluate.py --monthly --output-dir $S/monthly --bootstrap-reps 10000 --bootstrap-seed 42 --quiet-sites
def step_monthly():
    out = S / "monthly"
    out.mkdir(parents=True, exist_ok=True)
    cfg = ev.load_config(CFG)
    flux_dir = ev.resolve_flux_dir(cfg)
    container = SwimContainer.open(str(CONTAINER), mode="r")
    try:
        fids = ev.apply_exclusions(container.field_uids)
        bundle = ev.evaluate_benchmark_monthly(
            cfg,
            container,
            str(PAR),
            fids,
            flux_dir,
            bootstrap_reps=REPS,
            bootstrap_seed=SEED,
            quiet_sites=True,
            results_dir=str(out),
        )
    finally:
        container.close()
    ev.print_grouped_summary(bundle, "monthly")
    written = ev.write_grouped_outputs(
        bundle,
        str(out),
        "monthly",
        openet_source="volk",
        cli_args=_evaluate_cli_args(out, monthly=True),
    )
    bundle.site_metrics.to_csv(out / "evaluation_monthly_metrics.csv")
    print(f"Grouped estimates: {written['metrics']}")
    return bundle


def _evaluate_cli_args(out, monthly=False):
    """What evaluate.py records in its provenance sidecar for the README command."""
    return {
        "par_csv": str(PAR),
        "sites": None,
        "openet_source": "volk",
        "etf": False,
        "monthly": monthly,
        "container": str(CONTAINER),
        "config": str(CFG),
        "output_dir": str(out),
        "bootstrap_reps": REPS,
        "bootstrap_seed": SEED,
        "site_effect_summary": False,
        "quiet_sites": True,
    }


# --- 3. compare the evaluator CSVs with the shipped Run 23 archive -----------
def step_cmp():
    shipped = RUN23 / "archive" / "6_evaluation"
    compare(
        S / "daily",
        shipped,
        [
            "evaluation_grouped_daily_metrics.csv",
            "evaluation_grouped_daily_contrasts.csv",
            "evaluation_paired_daily_records.csv",
        ],
    )
    compare(
        S / "monthly",
        shipped,
        ["evaluation_grouped_monthly_metrics.csv", "evaluation_grouped_monthly_contrasts.csv"],
    )


# --- 4. retrieval-support (overpass) decomposition of the daily record -------
# README: overpass_decomposition.py --evaluator-output-dir $S/daily --output-dir $S/temporal --bootstrap-reps 10000 --seed 42
def step_temporal():
    out = S / "temporal"
    out.mkdir(parents=True, exist_ok=True)
    frame, parent_meta, gate_report = od.validate_parent_bundle(str(S / "daily"))
    print(
        f"Parent identity gate passed: {gate_report['record_counts']['n_sites']} sites, "
        f"{gate_report['record_counts']['n_rows']:,} rows"
    )
    decomp = od.temporal_decomposition(frame, reps=REPS, seed=SEED)
    excluded = sorted(set(frame["fid"].unique()) - set(decomp.common_cohort))
    print(
        f"Common temporal cohort: {len(decomp.common_cohort)} sites (excluded: {excluded or 'none'})"
    )
    decomp.grouped_metrics.to_csv(out / od.OUT_METRICS, index=False)
    decomp.grouped_contrasts.to_csv(out / od.OUT_CONTRASTS, index=False)
    decomp.interactions.to_csv(out / od.OUT_INTERACTIONS, index=False)
    decomp.site_eligibility.to_csv(out / od.OUT_ELIGIBILITY, index=False)
    cols = ["aggregation", "metric", "estimate", "ci95_low", "ci95_high", "unit"]
    print(decomp.interactions[cols].to_string(index=False))
    compare(
        out,
        RUN23 / "temporal_decomposition",
        [od.OUT_METRICS, od.OUT_CONTRASTS, od.OUT_INTERACTIONS],
    )
    return decomp


# --- 5. spread-error: ensemble spread vs ETf error at the captures -----------
# README: spread_error.py --config $CFG --out-dir $S/spread_error
def step_spread():
    out = S / "spread_error"
    out.mkdir(parents=True, exist_ok=True)
    cfg = ev.load_config(CFG)
    flux_dir = ev.resolve_flux_dir(cfg)
    obs_meta_path = RUN23 / "archive" / "3_problem_definition" / "observation_metadata.csv"

    etf_obs = se.load_observation_metadata(str(obs_meta_path))
    etf_obs = etf_obs[etf_obs["site"].isin(ev.apply_exclusions(list(etf_obs["site"].unique())))]
    container = SwimContainer.open(str(CONTAINER), mode="r")
    try:
        sv = se.spot_verify(etf_obs, container, n=20)  # archive target/spread vs container ETf
        sv["detail"].to_csv(out / "spot_verify.csv", index=False)
        obs_df, excl, ref_path, kept_sites = se.build_observation_table(
            etf_obs, container, flux_dir
        )
    finally:
        container.close()
    print(f"Paired observations: {len(obs_df)} across {obs_df['site'].nunique()} sites")
    obs_df.to_csv(out / "spread_error_observations.csv", index=False)

    pooled = se.pooled_correlations(obs_df)
    qt = se.quintile_table(obs_df)
    qt.to_csv(out / "spread_error_quintiles.csv", index=False)
    rmse_ratio = qt["RMSE"].iloc[-1] / qt["RMSE"].iloc[0]
    monotonic = bool((qt["RMSE"].diff().dropna() > 0).all())
    ps_df, ps_sum = se.per_site_robustness(obs_df)
    ps_df.to_csv(out / "spread_error_persite.csv", index=False)
    wq = se.weight_quintile_contrast(obs_df)

    summary = {
        "n_observations": len(obs_df),
        "n_sites": int(obs_df["site"].nunique()),
        "reference_eto_path": ref_path,
        "min_eto_screen": se.MIN_ETO,
        "spot_verify_max_abs_diff_std": sv["max_abs_diff_std"],
        "spot_verify_max_abs_diff_mean": sv["max_abs_diff_mean"],
        "exclusion_no_flux_file": len(excl["no_flux_file"]),
        "exclusion_below_site_minimum": len(excl["below_site_minimum"]),
        "exclusion_obs_flux_nan": excl["obs_flux_nan"],
        "exclusion_obs_ref_nan": excl["obs_ref_nan"],
        "exclusion_obs_ref_below_min": excl["obs_ref_below_min"],
        "pooled_etf_pearson_r": pooled["etf"]["pearson_r"],
        "pooled_etf_spearman_rho": pooled["etf"]["spearman_rho"],
        "pooled_et_pearson_r": pooled["et"]["pearson_r"],
        "pooled_et_spearman_rho": pooled["et"]["spearman_rho"],
        "quintile_rmse_ratio_hi_lo": rmse_ratio,
        "quintile_rmse_monotonic": monotonic,
        "persite_n_sites": ps_sum["n_sites"],
        "persite_median_spearman": ps_sum["median_spearman"],
        "persite_frac_positive": ps_sum["frac_positive"],
        "weight_bottom_quintile_mean_abs_err": wq["bottom_mean_abs_err"],
        "weight_top_quintile_mean_abs_err": wq["top_mean_abs_err"],
        "weight_bottom_quintile_median_abs_err": wq["bottom_median_abs_err"],
        "weight_top_quintile_median_abs_err": wq["top_median_abs_err"],
    }
    pd.Series(summary).to_csv(out / "spread_error_summary.csv", header=False)
    compare(
        out,
        RUN23 / "spread_error",
        ["spread_error_summary.csv", "spread_error_quintiles.csv", "spread_error_persite.csv"],
    )
    return obs_df


# --- 6. conditioned-ensemble spread (gates A-E, then the bootstrap) ----------
# README: conditioned_ensemble_uncertainty.py   (rewrites $RUN23/conditioned_ensemble_uncertainty/ in place)
def step_conditioned():
    ceu.main(out_dir=None, n_boot=ceu.N_BOOT, run=ex5_paths.CANONICAL_RUN)


# --- 7. weighting-ablation summary from the two shipped calibration arms -----
# README: run_weighting_ablation.py --tag run23 --container $DATA/5_Flux_Ensemble_run23ablation.swim --summary-only
def step_ablation():
    cfg = rwa._load_config()
    results_root = Path(cfg.project_ws) / "results"
    e1_dir = results_root / "ablation_run23_e1_spread"
    e2_dir = results_root / "ablation_run23_e2_fixed_sd"
    summary_dir = results_root / "ablation_run23_summary"
    print(f"Summary-only: regenerating derived CSVs -> {summary_dir}")
    rwa.regenerate_summaries(str(e1_dir), str(e2_dir), str(summary_dir))


# --- 8. within-E1 held-out transfer: six forward runs, scored on flux --------
# README: within_e1_transfer.py --out $S/within_e1_transfer
def step_transfer():
    out = S / "within_e1_transfer"
    saved = sys.argv
    sys.argv = ["within_e1_transfer.py", "--out", str(out)]  # main() parses its own arguments
    try:
        wt.main()  # step in: per_site_median_table -> run_fixed_params x5 -> run_default_params
    finally:
        sys.argv = saved
    compare(
        out,
        RUN23 / "within_e1_transfer",
        [
            "summary_metrics.csv",
            "region_breakdown.csv",
            "win_rates.csv",
            "persite_daily.csv",
            "persite_monthly.csv",
        ],
    )


# --- 9. period-split gate on the frozen daily record (no --write) ------------
# README: e1_period_split.py
def step_split():
    metrics, contrasts, cohort, gate, dropped = ps.build(REPS, SEED)
    print(gate)
    print("dropped site-periods:", dropped)
    ps.report(metrics, contrasts)
    return metrics, contrasts


ALL_STEPS = {
    "daily": step_daily,
    "monthly": step_monthly,
    "cmp": step_cmp,
    "temporal": step_temporal,
    "spread": step_spread,
    "conditioned": step_conditioned,
    "ablation": step_ablation,
    "transfer": step_transfer,
    "split": step_split,
}


def main(steps):
    required = [
        CFG,
        PAR,
        CONTAINER,
        ABLATION_CONTAINER,
        EX5 / "data" / "openet_refet" / "openet_eto.csv",
    ]
    for p in required:
        if not p.exists():
            sys.exit(f"missing input: {p}")
    S.mkdir(parents=True, exist_ok=True)
    for name in steps:
        print(f"\n==== {name}")
        ALL_STEPS[name]()
    print("\n==== summary")
    if DIFFERED:
        print(
            f"{len(DIFFERED)} compared file(s) differ (diff each pair to see whether it is round-off):"
        )
        for a, b in DIFFERED:
            print(f"  {a}\n    vs {b}")
    else:
        print("every compared file is byte-identical to the shipped copy")
    return not DIFFERED


if __name__ == "__main__":
    chosen = sys.argv[1:] or STEPS or list(ALL_STEPS)
    unknown = [s for s in chosen if s not in ALL_STEPS]
    if unknown:
        sys.exit(f"unknown step(s) {unknown}; choose from: {' '.join(ALL_STEPS)}")
    sys.exit(0 if main(chosen) else 1)
