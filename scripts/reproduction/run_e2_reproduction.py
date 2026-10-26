"""Example 6 (E0 and E2) reproduction, step by step, for walking in the debugger.

This is the "E0 and E2 (Example 6)" block of "3. Recompute the evidence" in
examples/README.md written as Python calls instead of command lines. The
short scripts (``evaluate``, ``pooled_metrics``, ``derived_metrics``) are
re-stated here the way their ``__main__`` does it, so the config, the open
container, the cohort, and each DataFrame are locals you can inspect, and
stepping into ``ev.evaluate`` leads to ``run_calibrated_model``, the forward
model run. The longer scripts keep their whole body inside ``main`` and parse
their own arguments, so ``call_main`` sets ``sys.argv`` and calls ``main``;
set a breakpoint on the first line of that ``main`` (or step into the call).

Run or debug it as-is: the data root and scratch directory are set in
``ROOT`` and ``SCRATCH`` below, and ``STEPS`` selects which steps run. The
repo-local and reference-final packages must be unpacked into the clone
(``paper/data/final/`` supplies the transfer vector).

Most of this block writes into the shipped run directory, not scratch (the
README says so); the comparison for those is step 4, the frozen-package check
that ``promote_final`` and the test suite do. The two scratch steps (transfer,
product parity) are compared byte-for-byte with the shipped copies, like the
README's ``cmp``. A rebuild on other hardware may differ in the last digit of
a few statistics; a cohort, sign, or file-set difference is a failure.
"""

import filecmp
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EX6 = REPO / "examples" / "6_Flux_International"
RUN = "6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr"
CFG = EX6 / f"{RUN}.toml"
CFG_SIG = EX6 / f"{RUN}_fao56_sig.toml"  # E0 arm: unscaled sigmoid
CFG_LIN = EX6 / f"{RUN}_fao56.toml"  # E0 arm: unscaled linear

# Edit these three to walk the script from the debugger with no configuration.
ROOT = Path.home() / "data" / "swim"  # the data root the swim_data_* packages were unpacked into
SCRATCH = Path.home() / "data" / "swim_scratch"  # outside the clone and the data root
STEPS = []  # e.g. ["daily", "pooled"]; empty means every step in README order

os.environ["SWIM_ROOT"] = str(ROOT)  # the TOML root relocation every Example 6 module reads
sys.path.insert(0, str(EX6))  # the modules import each other as top-level names

import derived_metrics as dm  # noqa: E402
import evaluate as ev  # noqa: E402
import ex6_paths  # noqa: E402
import geopandas as gpd  # noqa: E402
import pooled_metrics as pm  # noqa: E402

from swimrs.container import SwimContainer  # noqa: E402


def _load(relpath):
    """Import a script that lives in a subdirectory of Example 6 (no package)."""
    path = EX6 / relpath
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RESULTS = ROOT / "6_Flux_International" / "results"
RUN_DIR = RESULTS / RUN
ARCHIVE = RUN_DIR / "archive"
CAT6 = ARCHIVE / "6_evaluation"
POST = ARCHIVE / "4_pest_outputs" / "merged" / "merged_posterior.csv"
POST_SIG = (
    RESULTS / f"{RUN}_fao56_sig" / "archive" / "4_pest_outputs" / "merged" / "merged_posterior.csv"
)
POST_LIN = (
    RESULTS / f"{RUN}_fao56" / "archive" / "4_pest_outputs" / "merged" / "merged_posterior.csv"
)
DATA = ROOT / "6_Flux_International" / "data"
CONTAINER = DATA / "6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim"
VECTOR = REPO / "paper" / "data" / "final" / ex6_paths.TRANSFER_VECTOR_JSON
MAPPING = RUN_DIR / "transfer_refresh" / "e3_irrigation_stratified_param_mapping.json"
S = SCRATCH / "e2"

DIFFERED = []  # (scratch file, shipped file) pairs that are not byte-identical


def compare(scratch_dir, shipped_dir, files):
    for f in files:
        a, b = Path(scratch_dir) / f, Path(shipped_dir) / f
        if a.exists() and b.exists() and filecmp.cmp(a, b, shallow=False):
            print(f"  identical: {f}")
        else:
            print(f"  {'differs' if a.exists() and b.exists() else 'missing'}: {a} vs {b}")
            DIFFERED.append((a, b))


def call_main(module, argv, name=None):
    """Run ``module.main()`` as the README command line would; returns its exit status."""
    saved = sys.argv
    sys.argv = [name or f"{module.__name__}.py", *map(str, argv)]
    try:
        rc = module.main()  # step in: the whole script body is this one function
    except SystemExit as e:
        rc = e.code
    finally:
        sys.argv = saved
    return rc or 0


# --- 1-3. daily, monthly, and ETf evaluation of the published posterior ----
# README: evaluate.py --config $CFG --par-csv $POST [--monthly | --etf]   (writes into $RESULTS/$RUN/)
def _open_cohort():
    """What evaluate.py's __main__ sets up before choosing a mode."""
    cfg = ev._load_config(CFG)
    results_dir = ev._results_dir(cfg, CFG)  # == RUN_DIR: the TOML stem is the run name
    container = SwimContainer.open(str(CONTAINER), mode="r")
    flux_sources = ev.load_flux_sources(cfg.fields_shapefile, cfg.feature_id_col)
    gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")  # the publication cohort
    id_col = cfg.feature_id_col if cfg.feature_id_col in gdf.columns else "sid"
    cohort = [str(s) for s in gdf[id_col].tolist()]
    container_fids = set(container.field_uids)
    fids = [f for f in cohort if f in container_fids]
    missing = [f for f in cohort if f not in container_fids]
    if missing:
        print(f"WARNING: {len(missing)} cohort site(s) absent from container: {missing}")
    print(f"Cohort: {len(fids)} sites from {os.path.basename(cfg.fields_shapefile)}")
    return cfg, container, fids, flux_sources, results_dir


def step_daily():
    cfg, container, fids, flux_sources, results_dir = _open_cohort()
    try:
        metrics = ev.evaluate(
            cfg, container, str(POST), fids, results_dir=results_dir, flux_sources=flux_sources
        )
    finally:
        container.close()
    metrics.to_csv(os.path.join(results_dir, "evaluation_metrics.csv"))
    return metrics


def step_monthly():
    cfg, container, fids, flux_sources, results_dir = _open_cohort()
    try:
        metrics = ev.evaluate_monthly(
            cfg, container, str(POST), fids, results_dir=results_dir, flux_sources=flux_sources
        )
    finally:
        container.close()
    metrics.to_csv(os.path.join(results_dir, "evaluation_monthly_metrics.csv"))
    return metrics


def step_etf():
    cfg, container, fids, _, results_dir = _open_cohort()
    try:
        metrics = ev.evaluate_etf(cfg, container, str(POST), fids, results_dir=results_dir)
    finally:
        container.close()
    metrics.to_csv(os.path.join(results_dir, "evaluation_etf_metrics.csv"))
    return metrics


# --- 4. pooled (Volk-style) metrics from the per-site CSVs step 1 wrote -----
# README: pooled_metrics.py --results-dir $RESULTS/$RUN
def step_pooled():
    cfg = ev._load_config(CFG)
    results_dir = str(RUN_DIR)
    fids = sorted(
        f[:-4]
        for f in os.listdir(results_dir)
        if f.endswith(".csv") and not f.startswith("evaluation")
    )
    print(f"Sites with per-site CSVs: {len(fids)}")
    flux_sources = pm.load_flux_sources(cfg.fields_shapefile)
    d_station, d_obs, d_mod = pm._accumulate(
        results_dir, fids, monthly=False, flux_sources=flux_sources
    )
    daily_df = pm._report("DAILY", "mm/day", d_station, d_obs, d_mod)
    daily_df.to_csv(os.path.join(results_dir, "pooled_metrics_daily.csv"), index=False)
    m_station, m_obs, m_mod = pm._accumulate(
        results_dir, fids, monthly=True, flux_sources=flux_sources
    )
    monthly_df = pm._report("MONTHLY", "mm/month", m_station, m_obs, m_mod)
    monthly_df.to_csv(os.path.join(results_dir, "pooled_metrics_monthly.csv"), index=False)
    return daily_df, monthly_df


# --- 5. derived metrics: per-member benchmarks, CONUS split, Murphy, uncalibrated baseline
# README: derived_metrics.py --config $CFG --uncalibrated --out $RESULTS/$RUN/derived
def step_derived():
    out = RUN_DIR / "derived"
    out.mkdir(parents=True, exist_ok=True)
    cfg = ev._load_config(CFG)
    results_dir = ev._results_dir(cfg, CFG)
    container = SwimContainer.open(str(CONTAINER), mode="r")
    try:
        fids = dm._cohort_fids(cfg, container)
        specs = dm._rs_model_specs(container, cfg)
        flux_sources = ev.load_flux_sources(cfg.fields_shapefile, cfg.feature_id_col)
        per_site = dm.collect(cfg, container, results_dir, fids, specs, flux_sources)
        print(f"Paired sites (>=10 common days, with flux): {len(per_site)}")
        bench = dm.per_model_benchmarks(per_site, specs)
        decomp_sites, decomp = dm.conus_decomp(per_site)
        mur = dm.murphy(per_site, specs)
        best, worst, allm = dm.top_bottom(results_dir)
        print("Running uncalibrated (default-parameter) forward baseline...")
        uncal_results = dm.run_uncalibrated_model(cfg, container, fids)
        uncal_df, uncal_sum = dm.uncal_baseline(per_site, uncal_results)
    finally:
        container.close()
    print(bench.to_string(index=False))
    print(decomp.to_string(index=False))
    print(uncal_sum.to_string())
    bench.to_csv(out / "derived_per_model_benchmarks.csv", index=False)
    decomp.to_csv(out / "derived_conus_decomp_summary.csv", index=False)
    decomp_sites.to_csv(out / "derived_conus_decomp_persite.csv", index=False)
    mur.to_csv(out / "derived_murphy_decomp.csv", index=False)
    uncal_df.to_csv(out / "derived_uncal_baseline_persite.csv", index=False)
    return per_site


# --- 6. irrigation-stratified E1 -> E2 parameter mapping ---------------------
# README: transfer/build_e2_irrigation_mapping.py --container $CONTAINER --out-dir $RESULTS/$RUN/transfer_refresh --allow-unexpected
def step_mapping():
    bim = _load("transfer/build_e2_irrigation_mapping.py")
    call_main(
        bim,
        ["--container", CONTAINER, "--out-dir", RUN_DIR / "transfer_refresh", "--allow-unexpected"],
    )


# --- 7. E1 transfer scored on the E2 cohort (three forward runs) -------------
# README: transfer_ex5_params.py --config $CFG --params paper/data/final/e2_run23_transfer_vector.json
#         --params-by-site $MAPPING --container $CONTAINER --e2-results-dir $RESULTS/$RUN --out $S/transfer
def step_transfer():
    import transfer_ex5_params as tp

    out = S / "transfer"
    call_main(
        tp,
        [
            "--config", CFG,
            "--params", VECTOR,
            "--params-by-site", MAPPING,
            "--container", CONTAINER,
            "--e2-results-dir", RUN_DIR,
            "--out", out,
        ],
    )  # fmt: skip
    compare(
        out,
        RESULTS / ex6_paths.TRANSFER_RUN,
        [
            "transfer_comparison_summary.csv",
            "transfer_comparison_persite.csv",
            "transfer_winrates.csv",
            "evaluation_metrics.csv",
            "evaluation_monthly_metrics.csv",
            "evaluation_metrics_strat.csv",
            "evaluation_monthly_metrics_strat.csv",
            "pooled_metrics_daily.csv",
            "pooled_metrics_monthly.csv",
        ],
    )


# --- 8. Cat 6 evaluation summary: paired metrics, groups, overpass split, G11 gates
# README: evaluation_summary.py --config $CFG --run-name $RUN   (writes $ARCHIVE/6_evaluation/)
def step_summary():
    import evaluation_summary as es

    rc = call_main(es, ["--config", CFG, "--run-name", RUN])
    if rc:
        raise SystemExit(f"evaluation_summary G11 checks failed (exit {rc})")


# --- 9. closure-pool summaries (the paper's primary E2 tier) -----------------
# README: closure_pool_summary.py --config $CFG --run-name $RUN
def step_closure():
    import closure_pool_summary as cp

    rc = call_main(cp, ["--config", CFG, "--run-name", RUN])
    if rc:
        raise SystemExit(f"closure_pool_summary failed (exit {rc})")


# --- 10. monthly 28-day-minimum sensitivity on the closure pool ---------------
# README: awc_recal/monthly_28day_sensitivity.py --run-name $RUN
def step_month28():
    m28 = _load("awc_recal/monthly_28day_sensitivity.py")
    call_main(m28, ["--run-name", RUN])


# --- 11. E0: pooled two-arm gates on the 37 disjoint sites and the 47-site pool
# README: six pooled_arm_compare.py invocations; GATE PASS expected for the two
#         grassbasis pairs, GATE FAIL (exit 1) expected for fao56_sig vs fao56
E0_SETS = {"disjoint37": "disjoint_sites.txt", "pool47": "closure_pool_sites.txt"}
E0_PAIRS = [  # (a_name, a_config, a_par, b_name, b_config, b_par, expected gate)
    ("grassbasis", CFG, POST, "fao56_sig", CFG_SIG, POST_SIG, True),
    ("grassbasis", CFG, POST, "fao56", CFG_LIN, POST_LIN, True),
    ("fao56_sig", CFG_SIG, POST_SIG, "fao56", CFG_LIN, POST_LIN, False),
]


def step_e0():
    import pooled_arm_compare as pac

    for tag, sites_file in E0_SETS.items():
        for a, a_cfg, a_par, b, b_cfg, b_par, expect_pass in E0_PAIRS:
            out = RESULTS / "e0_disjoint" / f"{a}_vs_{b}" / tag
            print(f"\n-- {a} vs {b} on {tag}")
            rc = call_main(
                pac,
                [
                    "--a-name", a, "--a-config", a_cfg, "--a-par", a_par,
                    "--b-name", b, "--b-config", b_cfg, "--b-par", b_par,
                    "--sites-file", EX6 / "e0_disjoint" / sites_file,
                    "--out-dir", out,
                ],
            )  # fmt: skip
            passed = rc == 0
            if passed != expect_pass:
                DIFFERED.append(
                    (out / "pooled_gate.json", f"expected GATE {'PASS' if expect_pass else 'FAIL'}")
                )
                print(f"  UNEXPECTED: gate {'passed' if passed else 'failed'}")


# --- 12. E1/E2 product parity on the shared CONUS sites ----------------------
# README: product_parity/e1_e2_product_parity.py --out-dir $S/product_parity
def step_parity():
    pp = _load("product_parity/e1_e2_product_parity.py")
    out = S / "product_parity"
    pp.main(["--out-dir", str(out)])  # main(argv) parses the list it is given
    compare(
        out,
        CAT6 / "e1_e2_product_parity",
        [
            "site_metrics.csv",
            "paired_delta_summary.csv",
            "pooled_summary.csv",
            "support_error_contrast.csv",
            "decomposition_pooled.csv",
            "decomposition_within_site_summary.csv",
            "capture_support_ledger_totals.csv",
            "site_windows.csv",
        ],
    )


# --- 13. frozen-package check: every promoted file must match the results root
# README: promote_final.py --config $CFG   (check mode: match / match~; never --write)
def step_promote():
    import promote_final as pf

    rc = call_main(pf, ["--config", CFG])
    if rc:
        DIFFERED.append((RESULTS, "paper/data/final (promote_final check)"))


ALL_STEPS = {
    "daily": step_daily,
    "monthly": step_monthly,
    "etf": step_etf,
    "pooled": step_pooled,
    "derived": step_derived,
    "mapping": step_mapping,
    "transfer": step_transfer,
    "summary": step_summary,
    "closure": step_closure,
    "month28": step_month28,
    "e0": step_e0,
    "parity": step_parity,
    "promote": step_promote,
}


def main(steps):
    required = [CFG, CFG_SIG, CFG_LIN, POST, POST_SIG, POST_LIN, CONTAINER, VECTOR]
    for p in required:
        if not p.exists():
            sys.exit(f"missing input: {p}")
    S.mkdir(parents=True, exist_ok=True)
    for name in steps:
        print(f"\n==== {name}")
        ALL_STEPS[name]()
    print("\n==== summary")
    if DIFFERED:
        print(f"{len(DIFFERED)} check(s) differ (diff each pair to see whether it is round-off):")
        for a, b in DIFFERED:
            print(f"  {a}\n    vs {b}")
    else:
        print("every compared file is byte-identical to the shipped copy; every gate as expected")
    return not DIFFERED


if __name__ == "__main__":
    chosen = sys.argv[1:] or STEPS or list(ALL_STEPS)
    unknown = [s for s in chosen if s not in ALL_STEPS]
    if unknown:
        sys.exit(f"unknown step(s) {unknown}; choose from: {' '.join(ALL_STEPS)}")
    sys.exit(0 if main(chosen) else 1)
