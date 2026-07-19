"""Run the Study 2 weighting ablation: E1 (spread) vs E2 (fixed_sd).

Launches both calibration runs sequentially, then evaluates each against
flux tower ET using the canonical Ex5 evaluation, and produces paired
comparison outputs.

Usage:
    python run_weighting_ablation.py
    python run_weighting_ablation.py --dry-run --debug-fields US-Bi1,US-Ne1,US-ARM
"""

import argparse
import json
import os
import time
from pathlib import Path

EXPERIMENTS = {
    "e1_spread": {
        "etf_weighting_mode": "spread",
        "etf_weighting_fixed_sd": 0.33,
        "etf_weighting_spread_floor": 0.1,
        "etf_weighting_min_members": 2,
    },
    "e2_fixed_sd": {
        "etf_weighting_mode": "fixed_sd",
        "etf_weighting_fixed_sd": 0.33,
        "etf_weighting_spread_floor": 0.1,
        "etf_weighting_min_members": 2,
    },
}


def _load_config():
    from swimrs.swim.config import ProjectConfig

    project_dir = Path(__file__).resolve().parent
    conf = project_dir / "5_Flux_Ensemble.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf), calibrate=True)
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent), calibrate=True)
    return cfg


def run_calibration(
    cfg, experiment_id, experiment_spec, results_dir, debug_fields=None, container_path=None
):
    from calibrate import run_pest_sequence

    cfg.etf_weighting_mode = experiment_spec["etf_weighting_mode"]
    cfg.etf_weighting_fixed_sd = experiment_spec["etf_weighting_fixed_sd"]
    cfg.etf_weighting_spread_floor = experiment_spec["etf_weighting_spread_floor"]
    cfg.etf_weighting_min_members = experiment_spec["etf_weighting_min_members"]

    print(f"\n{'=' * 80}")
    print(f"CALIBRATION: {experiment_id} (weighting_mode={cfg.etf_weighting_mode})")
    print(f"Container: {container_path}")
    print(f"Results: {results_dir}")
    print(f"{'=' * 80}\n")

    t0 = time.time()
    run_pest_sequence(
        cfg,
        results_dir,
        pdc_remove=False,
        debug_fields=debug_fields,
        container_path=container_path,
    )
    elapsed = time.time() - t0

    runtime = {
        "experiment_id": experiment_id,
        "weighting_mode": cfg.etf_weighting_mode,
        "container_path": container_path,
        "fixed_sd": cfg.etf_weighting_fixed_sd,
        "spread_floor": cfg.etf_weighting_spread_floor,
        "min_members": cfg.etf_weighting_min_members,
        "realizations": cfg.realizations,
        "workers": cfg.workers,
        "wall_seconds": round(elapsed, 1),
        "wall_minutes": round(elapsed / 60, 1),
    }
    with open(os.path.join(results_dir, "runtime.json"), "w") as f:
        json.dump(runtime, f, indent=2)

    # Save config snapshot
    spec_path = os.path.join(results_dir, "experiment_spec.json")
    with open(spec_path, "w") as f:
        json.dump(experiment_spec, f, indent=2)

    return elapsed


def _find_par_csv(results_dir, project):
    """Find the highest-iteration par.csv in a results directory."""
    for i in range(10, -1, -1):
        candidate = os.path.join(results_dir, f"{project}.{i}.par.csv")
        if os.path.exists(candidate):
            return candidate
    return None


def run_evaluation(
    cfg, experiment_id, results_dir, container_path, mode="daily", debug_fields=None
):
    """Run canonical evaluation for one experiment on the Run 22 footing.

    mode: 'daily' (volk source), 'monthly', or 'etf'.
    The container is the same Run 22-footed container used for calibration, and
    the flux reference is resolved from the TOML (Volk v2.1 daily_flux_files_2pt1)
    via resolve_flux_dir — NOT the legacy daily_flux_files directory.
    When debug_fields is set, limits evaluation to that subset only.
    """
    from evaluate import evaluate, evaluate_etf, evaluate_monthly, resolve_flux_dir

    from swimrs.container import SwimContainer

    project = cfg.project_name
    par_csv = _find_par_csv(results_dir, project)
    if par_csv is None:
        print(f"  WARNING: no par.csv found in {results_dir}, skipping evaluation")
        return

    container = SwimContainer.open(container_path, mode="r")
    flux_dir = resolve_flux_dir(cfg)

    if debug_fields is not None:
        fids = [f for f in debug_fields if f in container.field_uids]
    else:
        fids = container.field_uids

    try:
        if mode == "monthly":
            metrics = evaluate_monthly(cfg, container, par_csv, fids, flux_dir)
            out_csv = os.path.join(results_dir, "evaluation_monthly_metrics.csv")
        elif mode == "etf":
            metrics = evaluate_etf(cfg, container, par_csv, fids)
            out_csv = os.path.join(results_dir, "evaluation_etf_metrics.csv")
        else:
            metrics = evaluate(cfg, container, par_csv, fids, flux_dir, openet_source="volk")
            out_csv = os.path.join(results_dir, "evaluation_metrics.csv")
        metrics.to_csv(out_csv)
        print(f"  {experiment_id} {mode} metrics -> {out_csv}")
    finally:
        container.close()


def _build_paired_deltas(e1_path, e2_path, summary_dir, label):
    """Build paired delta CSV for one timescale/evaluation mode."""
    import pandas as pd

    if not os.path.exists(e1_path) or not os.path.exists(e2_path):
        print(f"  Skipping {label} comparison (missing files)")
        return None, None

    # ETf CSVs have a MultiIndex (fid, model); daily/monthly have fid only.
    with open(e1_path) as f:
        header = f.readline().strip().split(",")
    if "model" in header:
        idx_cols = [header.index("fid"), header.index("model")]
    else:
        idx_cols = 0

    e1 = pd.read_csv(e1_path, index_col=idx_cols)
    e2 = pd.read_csv(e2_path, index_col=idx_cols)
    common = e1.index.intersection(e2.index)
    e1, e2 = e1.loc[common], e2.loc[common]

    paired = pd.DataFrame(index=common)
    for metric in [
        "r2_swim",
        "rmse_swim",
        "bias_swim",
        "kge_swim",
        "r2",
        "rmse",
        "bias",
        "kge",
    ]:
        e1_col = metric if metric in e1.columns else None
        e2_col = metric if metric in e2.columns else None
        if e1_col and e2_col:
            paired[f"e1_{metric}"] = e1[metric]
            paired[f"e2_{metric}"] = e2[metric]
            paired[f"delta_{metric}"] = e1[metric] - e2[metric]

    r2_col = "r2_swim" if "r2_swim" in e1.columns else ("r2" if "r2" in e1.columns else None)
    if r2_col:
        paired["e1_wins_r2"] = e1[r2_col] > e2[r2_col]
        if "n" in e1.columns:
            paired["n_paired"] = e1["n"]

    out_path = os.path.join(summary_dir, f"paired_site_deltas_{label}.csv")
    paired.to_csv(out_path)
    print(f"  Paired deltas ({label}): {out_path}")
    return e1, e2


def _build_weight_summary(exp_dir, summary_dir, exp_id):
    """Build per-site weight summary from weight audit CSV."""
    import pandas as pd

    audit_path = os.path.join(exp_dir, "etf_weight_audit.csv")
    if not os.path.exists(audit_path):
        return
    audit = pd.read_csv(audit_path)
    if audit.empty:
        return

    grouped = audit.groupby("fid")
    rows = []
    wcol = "weight_final" if "weight_final" in audit.columns else "weight"
    total_weight_all = audit.loc[audit[wcol] > 0, wcol].sum()
    for fid, grp in grouped:
        nonzero = grp[grp[wcol] > 0]
        rows.append(
            {
                "fid": fid,
                "n_captures": len(grp),
                "n_eligible": int(grp["eligible"].sum()),
                "n_nonzero_weight": len(nonzero),
                "total_weight": nonzero[wcol].sum(),
                "weight_share": nonzero[wcol].sum() / total_weight_all
                if total_weight_all > 0
                else 0,
                "mean_weight": nonzero[wcol].mean() if len(nonzero) > 0 else 0,
                "max_weight": nonzero[wcol].max() if len(nonzero) > 0 else 0,
                "mean_member_std": grp["member_std"].mean(),
            }
        )

    df = pd.DataFrame(rows).set_index("fid")
    out_path = os.path.join(summary_dir, f"etf_weight_summary_by_site_{exp_id}.csv")
    df.to_csv(out_path)
    print(f"  Weight summary ({exp_id}): {out_path}")


PHI_REQUIRED_COLUMNS = ["iteration", "total_runs", "mean", "standard_deviation"]


def _read_phi_iterations(phi_path):
    """Read phi.meas.csv as one row per IES iteration, sorted by iteration.

    Actual pestpp-ies schema: ``iteration,total_runs,mean,standard_deviation,
    min,max,<realization columns...>`` — ROWS are iterations (0..noptmax), not
    realizations. The per-iteration ensemble-mean phi is the ``mean`` column.
    Raises on missing required columns, duplicate iterations, or a sequence
    that is not contiguous from 0, rather than silently mislabeling
    realization columns as iterations.
    """
    import pandas as pd

    phi = pd.read_csv(phi_path)
    missing = [c for c in PHI_REQUIRED_COLUMNS if c not in phi.columns]
    if missing:
        raise ValueError(f"{phi_path}: missing required phi columns {missing}")
    phi["iteration"] = phi["iteration"].astype(int)
    if phi["iteration"].duplicated().any():
        raise ValueError(f"{phi_path}: duplicate phi iterations {sorted(phi['iteration'])}")
    phi = phi.sort_values("iteration").reset_index(drop=True)
    expected = list(range(len(phi)))
    if list(phi["iteration"]) != expected:
        raise ValueError(
            f"{phi_path}: phi iterations not contiguous from 0: {list(phi['iteration'])}"
        )
    return phi


def _build_phi_summary(exp_dirs, summary_dir):
    """Parse phi.meas.csv from both runs and write phi_summary.csv.

    Phi is weight-scale-confounded across weighting schemes and is retained
    here as a descriptive convergence diagnostic only — it is NOT a
    model-skill claim and must not be compared across arms as accuracy.
    """
    import pandas as pd

    rows = []
    for exp_id, exp_dir in exp_dirs.items():
        phi_path = os.path.join(exp_dir, "5_Flux_Ensemble.phi.meas.csv")
        rt_path = os.path.join(exp_dir, "runtime.json")

        if not os.path.exists(phi_path):
            continue

        phi = _read_phi_iterations(phi_path)
        row = {"experiment_id": exp_id}
        for it, val in zip(phi["iteration"], phi["mean"]):
            row[f"phi_iter_{it}"] = round(float(val), 1)
        row["phi_initial"] = round(float(phi["mean"].iloc[0]), 1)
        row["phi_final"] = round(float(phi["mean"].iloc[-1]), 1)
        row["phi_reduction_pct"] = (
            round(100 * (1 - phi["mean"].iloc[-1] / phi["mean"].iloc[0]), 3)
            if phi["mean"].iloc[0] > 0
            else 0
        )
        row["n_phi_records"] = len(phi)
        row["nopt_iterations"] = int(phi["iteration"].iloc[-1])

        if os.path.exists(rt_path):
            with open(rt_path) as f:
                rt = json.load(f)
            row["wall_minutes"] = rt.get("wall_minutes", None)

        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        out_path = os.path.join(summary_dir, "phi_summary.csv")
        df.to_csv(out_path, index=False)
        print(f"  Phi summary: {out_path}")


def _build_ablation_summary(exp_dirs, summary_dir):
    """Build single ablation_summary.csv with one row per experiment.

    Reports the paper's standard metric set (NSE [=r2 column], KGE, RMSE, bias)
    as per-site cohort medians. The signed bias medians here are the
    descriptive cohort-median MBE; the accuracy-oriented bias contrast (paired
    absolute MBE) lives in paired_delta_summary.csv. Phi columns are excluded
    (weight-scale-confounded across arms; see phi_summary.csv).
    """
    import pandas as pd

    rows = []
    for exp_id, exp_dir in exp_dirs.items():
        row = {"experiment_id": exp_id}

        # Runtime
        rt_path = os.path.join(exp_dir, "runtime.json")
        if os.path.exists(rt_path):
            with open(rt_path) as f:
                rt = json.load(f)
            row.update(
                {
                    k: rt[k]
                    for k in ["weighting_mode", "realizations", "workers", "wall_minutes"]
                    if k in rt
                }
            )

        # Daily metrics (r2 column IS the paper's NSE)
        daily_path = os.path.join(exp_dir, "evaluation_metrics.csv")
        if os.path.exists(daily_path):
            d = pd.read_csv(daily_path, index_col=0)
            valid = d["r2_swim"].dropna()
            row["daily_n_sites"] = len(valid)
            row["daily_r2_median"] = round(valid.median(), 3)
            row["daily_rmse_median"] = round(d["rmse_swim"].dropna().median(), 3)
            row["daily_bias_median"] = round(d["bias_swim"].dropna().median(), 3)
            if "kge_swim" in d.columns:
                row["daily_kge_median"] = round(d["kge_swim"].dropna().median(), 3)

        # Monthly metrics
        monthly_path = os.path.join(exp_dir, "evaluation_monthly_metrics.csv")
        if os.path.exists(monthly_path):
            m = pd.read_csv(monthly_path, index_col=0)
            valid = m["r2_swim"].dropna()
            row["monthly_n_sites"] = len(valid)
            row["monthly_r2_median"] = round(valid.median(), 3)
            row["monthly_rmse_median"] = round(m["rmse_swim"].dropna().median(), 2)
            row["monthly_bias_median"] = round(m["bias_swim"].dropna().median(), 2)
            if "kge_swim" in m.columns:
                row["monthly_kge_median"] = round(m["kge_swim"].dropna().median(), 3)

        # Phi deliberately excluded here: weight-scale-confounded across arms.
        # Corrected per-iteration values live in the explicitly diagnostic
        # phi_summary.csv (_build_phi_summary).

        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        out_path = os.path.join(summary_dir, "ablation_summary.csv")
        df.to_csv(out_path, index=False)
        print(f"  Ablation summary: {out_path}")


DELTA_SPECS = [
    # (metric label, source column, delta definition, favorable direction, use_abs)
    ("nse", "r2_swim", "nse_spread - nse_fixed_sd", "positive", False),
    ("kge", "kge_swim", "kge_spread - kge_fixed_sd", "positive", False),
    ("rmse", "rmse_swim", "rmse_spread - rmse_fixed_sd", "negative", False),
    ("abs_mbe", "bias_swim", "abs(mbe_spread) - abs(mbe_fixed_sd)", "negative", True),
]


def _bootstrap_median_ci(deltas, reps, rng):
    """95% site-bootstrap CI of the median paired delta (resamples sites)."""
    import numpy as np

    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    idx = rng.integers(0, n, size=(reps, n))
    medians = np.median(deltas[idx], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def _build_paired_delta_summary(exp_dirs, summary_dir, seed=42, reps=10000):
    """Write paired_delta_summary.csv: paired spread-minus-fixed effects + CIs.

    The primary ablation evidence: per-site paired deltas (spread − fixed_sd)
    with deterministic 95% site-bootstrap intervals. Finite values are
    filtered per metric; sites (not days) are resampled; the bias contrast is
    accuracy-oriented absolute MBE, |MBE_spread| − |MBE_fixed| (signed
    cohort-median MBE stays in ablation_summary.csv as a descriptive
    quantity). No win-rate fields.

    One rng is seeded per scale and drawn sequentially across the metrics in
    DELTA_SPECS order — that order is load-bearing for reproducibility of the
    archived intervals; do not reorder DELTA_SPECS.
    """
    import numpy as np
    import pandas as pd

    rows = []
    for scale, fname in [
        ("daily", "evaluation_metrics.csv"),
        ("monthly", "evaluation_monthly_metrics.csv"),
    ]:
        e1_path = os.path.join(exp_dirs["e1_spread"], fname)
        e2_path = os.path.join(exp_dirs["e2_fixed_sd"], fname)
        if not (os.path.exists(e1_path) and os.path.exists(e2_path)):
            print(f"  Skipping paired delta summary ({scale}) — missing evaluation files")
            continue
        e1 = pd.read_csv(e1_path, index_col=0)
        e2 = pd.read_csv(e2_path, index_col=0)
        common = e1.index.intersection(e2.index)
        e1, e2 = e1.loc[common], e2.loc[common]

        rng = np.random.default_rng(seed)
        for metric, col, definition, favorable, use_abs in DELTA_SPECS:
            a = e1[col].abs() if use_abs else e1[col]
            b = e2[col].abs() if use_abs else e2[col]
            finite = np.isfinite(a.values) & np.isfinite(b.values)
            deltas = (a[finite] - b[finite]).values
            lo, hi = _bootstrap_median_ci(deltas, reps, rng)
            rows.append(
                {
                    "scale": scale,
                    "metric": metric,
                    "n_sites": int(finite.sum()),
                    "delta_definition": definition,
                    "favorable_direction": favorable,
                    "median_delta": float(np.median(deltas)),
                    "mean_delta": float(np.mean(deltas)),
                    "bootstrap_seed": seed,
                    "bootstrap_reps": reps,
                    "ci_lower": lo,
                    "ci_upper": hi,
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        out_path = os.path.join(summary_dir, "paired_delta_summary.csv")
        df.to_csv(out_path, index=False)
        print(f"  Paired delta summary: {out_path}")
        return df
    return None


def summarize_ablation(e1_dir, e2_dir, summary_dir):
    """Produce all paired comparison and diagnostic artifacts for E1 vs E2."""
    os.makedirs(summary_dir, exist_ok=True)

    # Paired deltas: daily, monthly, ETf
    for label, suffix in [
        ("daily", "_metrics.csv"),
        ("monthly", "_monthly_metrics.csv"),
        ("etf", "_etf_metrics.csv"),
    ]:
        _build_paired_deltas(
            os.path.join(e1_dir, f"evaluation{suffix}"),
            os.path.join(e2_dir, f"evaluation{suffix}"),
            summary_dir,
            label,
        )

    # Per-site weight summaries
    _build_weight_summary(e1_dir, summary_dir, "e1_spread")
    _build_weight_summary(e2_dir, summary_dir, "e2_fixed_sd")

    exp_dirs = {"e1_spread": e1_dir, "e2_fixed_sd": e2_dir}

    # Phi convergence summary (descriptive diagnostic only)
    _build_phi_summary(exp_dirs, summary_dir)

    # One-row-per-experiment ablation summary
    _build_ablation_summary(exp_dirs, summary_dir)

    # Paired uncertainty table (primary ablation evidence)
    _build_paired_delta_summary(exp_dirs, summary_dir)


def regenerate_summaries(e1_dir, e2_dir, summary_dir):
    """Regenerate only the derived summary CSVs from existing arm outputs.

    Rebuilds phi_summary.csv, ablation_summary.csv, and
    paired_delta_summary.csv. Does NOT touch calibration or evaluation
    outputs, the raw paired_site_deltas_*.csv files, or the weight summaries.
    """
    os.makedirs(summary_dir, exist_ok=True)
    exp_dirs = {"e1_spread": e1_dir, "e2_fixed_sd": e2_dir}
    _build_phi_summary(exp_dirs, summary_dir)
    _build_ablation_summary(exp_dirs, summary_dir)
    _build_paired_delta_summary(exp_dirs, summary_dir)


def main():
    parser = argparse.ArgumentParser(description="Run Study 2 weighting ablation (E1 vs E2)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use reduced realizations (20) for quick validation",
    )
    parser.add_argument(
        "--debug-fields",
        type=str,
        default=None,
        help="Comma-separated site IDs for debug subset",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration, only run evaluation and summary",
    )
    parser.add_argument(
        "--only",
        choices=["e1", "e2"],
        default=None,
        help="Run only one experiment (e1=spread, e2=fixed_sd)",
    )
    parser.add_argument(
        "--tag",
        default="run22",
        help="Results-dir tag: outputs land in ablation_{tag}_{exp}/ and "
        "ablation_{tag}_summary/ (default 'run22'). The untagged April dirs "
        "(ablation_e1_spread etc.) are the stale reference and must not be touched.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Regenerate only phi_summary.csv, ablation_summary.csv, and "
        "paired_delta_summary.csv from existing arm outputs. No calibration, "
        "no evaluation; raw paired_site_deltas_*.csv and weight summaries "
        "are left untouched.",
    )
    parser.add_argument(
        "--container",
        default=None,
        help="Container path for BOTH calibration and evaluation. REQUIRED for "
        "calibration/evaluation runs (not for --summary-only). For Run 22 "
        "footing pass the run22-seeded ablation container (calibration group "
        "already absent), e.g. {data}/5_Flux_Ensemble_run22ablation.swim. "
        "There is deliberately no default: the tag defaults to 'run22', and "
        "silently falling back to the stale base {data}/{project}.swim would "
        "overwrite Run-22-labeled results on the wrong footing.",
    )
    args = parser.parse_args()

    if not args.summary_only and args.container is None:
        parser.error(
            "--container is required for calibration/evaluation runs. Pass the "
            "container matching the tag (Run 22 footing: the run22-seeded "
            "ablation container, e.g. {data}/5_Flux_Ensemble_run22ablation.swim)."
        )

    cfg = _load_config()
    results_root = os.path.join(cfg.project_ws, "results")

    container_path = args.container

    tag = args.tag
    exp_dirs = {
        exp_id: os.path.join(results_root, f"ablation_{tag}_{exp_id}") for exp_id in EXPERIMENTS
    }
    summary_dir = os.path.join(results_root, f"ablation_{tag}_summary")

    debug_fields = None
    if args.debug_fields:
        debug_fields = [s.strip() for s in args.debug_fields.split(",")]

    if args.dry_run:
        cfg.realizations = 20
        cfg.workers = min(10, cfg.workers)
        print("DRY RUN: realizations=20, reduced workers")

    if args.summary_only:
        print(f"Summary-only: regenerating derived CSVs -> {summary_dir}")
        regenerate_summaries(exp_dirs["e1_spread"], exp_dirs["e2_fixed_sd"], summary_dir)
        return

    print(f"Container: {container_path}")
    print(f"Tag: {tag}  (summary -> {summary_dir})")

    # Determine which experiments to run
    run_ids = list(EXPERIMENTS.keys())
    if args.only == "e1":
        run_ids = ["e1_spread"]
    elif args.only == "e2":
        run_ids = ["e2_fixed_sd"]

    # Phase 1: Calibration
    if not args.skip_calibration:
        for exp_id in run_ids:
            run_calibration(
                cfg,
                exp_id,
                EXPERIMENTS[exp_id],
                exp_dirs[exp_id],
                debug_fields,
                container_path=container_path,
            )

    # Phase 2: Evaluation (daily + monthly + ETf)
    for exp_id in run_ids:
        exp_dir = exp_dirs[exp_id]
        if not os.path.exists(exp_dir):
            print(f"  Skipping evaluation for {exp_id} (no results dir)")
            continue
        print(f"\nEvaluating {exp_id}...")
        run_evaluation(
            cfg, exp_id, exp_dir, container_path, mode="daily", debug_fields=debug_fields
        )
        run_evaluation(
            cfg, exp_id, exp_dir, container_path, mode="monthly", debug_fields=debug_fields
        )
        run_evaluation(cfg, exp_id, exp_dir, container_path, mode="etf", debug_fields=debug_fields)

    # Phase 3: Summary
    print("\nSummarizing ablation...")
    summarize_ablation(exp_dirs["e1_spread"], exp_dirs["e2_fixed_sd"], summary_dir)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
