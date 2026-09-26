"""Freeze the irrigation-class split of the E1 OpenET benchmark (Table S11).

The frozen E1 package reports SWIM-RS and OpenET skill against the CONUS flux
towers on the whole cohort. This script partitions the same frozen records by
the E2 irrigation class (``irrigated`` / ``rainfed``, the class each site
carries in ``paper/data/final/e2_irrigation_stratified_fold_mad_domain.csv``)
and freezes the class-conditional KGE, RMSE and MBE for both series, with
whole-site bootstrap intervals on the SWIM-minus-OpenET contrast, at three
scales:

  daily      the frozen paired daily record (45 sites), pooled and
             station-weighted;
  monthly    the Volk-protocol monthly record replicated from the run archive
             with ``promote_e1_monthly.monthly_records_from_archive`` (pooled on
             every paired site, station-weighted on sites with at least three
             paired months);
  temporal   the frozen 43-site common temporal cohort split into retrieval and
             between-retrieval days, with the support interaction
             (between minus retrieval of SWIM minus OpenET) per class.

Nothing is computed that the frozen package does not already imply: before
anything is written the class partitions are reunited and must reproduce every
frozen grouped estimate (daily, monthly, temporal) to REPLICATION_TOL with the
frozen cohorts. Outputs land in ``e1_openet_benchmark/class_split/`` with a
MANIFEST.json (input hashes, git sha, gates, file hashes) and a copy of this
script; the parent MANIFEST.json gains an ``addenda.class_split`` pointer and a
``promotion_history`` entry. No existing frozen file is modified.

Usage:
    uv run python examples/5_Flux_Ensemble/e1_class_split.py            # gates + report
    uv run python examples/5_Flux_Ensemble/e1_class_split.py --write    # freeze
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from swimrs.evaluation.benchmark import (
    AGG_POOLED,
    AGG_WEIGHTED,
    POOLED_METRICS,
    TEMPORAL_ALL_DAYS,
    TEMPORAL_CLASS_BETWEEN,
    TEMPORAL_CLASS_RETRIEVAL,
    WEIGHTED_METRICS,
    _bootstrap_multiplicities,
    bootstrap_grouped_from_counts,
    grouped_point_estimates,
    paired_records_from_frame,
    read_paired_record_frame,
    temporal_class_records,
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import promote_e1_monthly as pm  # noqa: E402

REPO = HERE.parent.parent
FINAL = REPO / "paper" / "data" / "final"
PACKAGE = FINAL / "e1_openet_benchmark"
OUT_DIRNAME = "class_split"
CLASS_CSV = FINAL / "e2_irrigation_stratified_fold_mad_domain.csv"
CLASSES = ("irrigated", "rainfed")
CONTRAST = "swim_minus_openet"
INTERACTION = "between_retrieval_minus_retrieval_of_swim_minus_openet"
REPLICATION_TOL = 1e-6
SCHEMA = "e1_openet_benchmark_class_split/v1"
UNITS = {"daily": "mm d-1", "monthly": "mm month-1", "temporal": "mm d-1"}
FAVORABLE = {
    "kge": "positive",
    "rmse": "negative",
    "mbe": "directional_only",
    "r": "positive",
    "r2": "positive",
    "slope0": "directional_only",
}
ALL_DAYS = "all_days"


class ClassSplitError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# inputs


def load_class_map(path=CLASS_CSV):
    df = pd.read_csv(path)[["fid", "irr_class"]].drop_duplicates()
    dup = df[df.duplicated("fid", keep=False)]
    if not dup.empty:
        raise ClassSplitError(f"sites with more than one class: {sorted(dup.fid.unique())}")
    bad = set(df.irr_class) - set(CLASSES)
    if bad:
        raise ClassSplitError(f"unknown irrigation classes {sorted(bad)}")
    return df.set_index("fid")["irr_class"]


def require_classed(fids, classes, context):
    missing = sorted(f for f in fids if f not in classes.index)
    if missing:
        raise ClassSplitError(f"{context}: sites without an irrigation class: {missing}")


def split_records(records, classes):
    """{class: tuple of records} in the records' order; every record classed."""
    require_classed([r.fid for r in records], classes, "split")
    return {c: tuple(r for r in records if classes[r.fid] == c) for c in CLASSES}


# ---------------------------------------------------------------------------
# estimation


def _ci(arr):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


def estimate_class(records, aggregation, reps, seed, context):
    """Point estimates and bootstrap replicates for one class cohort and aggregation."""
    if not records:
        raise ClassSplitError(f"{context}: empty class cohort")
    est = grouped_point_estimates(records, aggregations=(aggregation,))
    _, counts = _bootstrap_multiplicities(len(records), reps, seed)
    boot = bootstrap_grouped_from_counts(
        records, counts, context=context, aggregations=(aggregation,)
    )
    return est, boot, counts


def metric_rows(scale, temporal_class, irr_class, aggregation, records, est, boot, reps, seed):
    n_sites = len(records)
    n_pairs = int(sum(r.n for r in records))
    metrics = POOLED_METRICS if aggregation == AGG_POOLED else WEIGHTED_METRICS
    rows, contrasts = [], []
    for metric in metrics:
        unit = UNITS[scale] if metric in ("rmse", "mbe") else "dimensionless"
        for model in ("swim", "openet_ensemble"):
            key = (aggregation, model, metric)
            lo, hi = _ci(boot[key])
            rows.append(
                {
                    "scale": scale,
                    "temporal_class": temporal_class,
                    "irr_class": irr_class,
                    "aggregation": aggregation,
                    "model": model,
                    "metric": metric,
                    "estimate": float(est[key]),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "unit": unit,
                    "n_sites": n_sites,
                    "n_pairs": n_pairs,
                    "bootstrap_unit": "site",
                    "bootstrap_reps": reps,
                    "bootstrap_seed": seed,
                }
            )
        key = (aggregation, CONTRAST, metric)
        lo, hi = _ci(boot[key])
        contrasts.append(
            {
                "scale": scale,
                "temporal_class": temporal_class,
                "irr_class": irr_class,
                "aggregation": aggregation,
                "metric": metric,
                "contrast": CONTRAST,
                "estimate": float(est[(aggregation, "swim", metric)])
                - float(est[(aggregation, "openet_ensemble", metric)]),
                "ci95_low": lo,
                "ci95_high": hi,
                "unit": unit,
                "n_sites": n_sites,
                "n_pairs": n_pairs,
                "bootstrap_unit": "site",
                "bootstrap_reps": reps,
                "bootstrap_seed": seed,
                "favorable_direction": FAVORABLE[metric],
            }
        )
    return rows, contrasts


def gate_against_frozen(
    records, frozen_csv, key_cols, filt, aggregations, context, tol=REPLICATION_TOL
):
    """Reunited class records must reproduce every frozen grouped estimate."""
    frozen = pd.read_csv(frozen_csv)
    for col, val in filt.items():
        frozen = frozen[frozen[col] == val]
    got = frozen.set_index(key_cols)["estimate"]
    est = grouped_point_estimates(records, aggregations=aggregations)
    worst = 0.0
    for (agg, model, metric), value in est.items():
        if (agg, model, metric) not in got.index:
            raise ClassSplitError(f"{context}: frozen table lacks {(agg, model, metric)}")
        worst = max(worst, abs(float(got.loc[(agg, model, metric)]) - value))
    if worst > tol:
        raise ClassSplitError(f"{context}: max abs difference {worst:.3e} exceeds {tol}")
    return f"PASS: {len(est)} estimates reproduced, max abs diff {worst:.3e} at tolerance {tol}"


# ---------------------------------------------------------------------------
# scales


def daily_scale(frame, classes, reps, seed):
    records = paired_records_from_frame(frame)
    parts = split_records(records, classes)
    gate = gate_against_frozen(
        records,
        PACKAGE / "daily" / "evaluation_grouped_daily_metrics.csv",
        ["aggregation", "model", "metric"],
        {"scale": "daily"},
        (AGG_POOLED, AGG_WEIGHTED),
        "daily",
    )
    rows, contrasts, cohort = [], [], []
    for irr_class, recs in parts.items():
        for agg in (AGG_POOLED, AGG_WEIGHTED):
            est, boot, _ = estimate_class(recs, agg, reps, seed, f"daily {irr_class} {agg}")
            r, c = metric_rows("daily", ALL_DAYS, irr_class, agg, recs, est, boot, reps, seed)
            rows += r
            contrasts += c
        cohort += [{"fid": r.fid, "irr_class": irr_class, "n_daily": r.n} for r in recs]
    return rows, contrasts, pd.DataFrame(cohort), gate


def monthly_scale(classes, reps, seed):
    meta = json.load(open(PACKAGE / "monthly" / "evaluation_grouped_monthly_metadata.json"))
    run_meta_paths = meta["paths"]
    ts_dir = (
        Path(run_meta_paths["par_csv"]).parent
        / "archive"
        / "6_evaluation"
        / "site_daily_timeseries"
    )
    excluded = set(meta.get("static_exclusions", []))
    records = pm.monthly_records_from_archive(
        ts_dir,
        run_meta_paths["flux_dir"],
        run_meta_paths["openet_monthly_dir"],
        static_exclusions=excluded,
    )
    weighted = tuple(r for r in records if r.n >= pm.WEIGHTED_MIN_MONTHS)
    for agg, recs in ((AGG_POOLED, records), (AGG_WEIGHTED, weighted)):
        mine = tuple(sorted((r.fid, r.n) for r in recs))
        if mine != pm.cohort_from_meta(meta, agg):
            raise ClassSplitError(
                f"monthly {agg}: replicated cohort differs from the frozen cohort"
            )
    frozen_csv = PACKAGE / "monthly" / "evaluation_grouped_monthly_metrics.csv"
    keys = ["aggregation", "model", "metric"]
    gate_p = gate_against_frozen(records, frozen_csv, keys, {}, (AGG_POOLED,), "monthly pooled")
    gate_w = gate_against_frozen(
        weighted, frozen_csv, keys, {}, (AGG_WEIGHTED,), "monthly weighted"
    )
    rows, contrasts, cohort = [], [], []
    parts_p = split_records(records, classes)
    parts_w = split_records(weighted, classes)
    for irr_class in CLASSES:
        for agg, recs in ((AGG_POOLED, parts_p[irr_class]), (AGG_WEIGHTED, parts_w[irr_class])):
            est, boot, _ = estimate_class(recs, agg, reps, seed, f"monthly {irr_class} {agg}")
            r, c = metric_rows("monthly", ALL_DAYS, irr_class, agg, recs, est, boot, reps, seed)
            rows += r
            contrasts += c
        weighted_fids = {r.fid for r in parts_w[irr_class]}
        cohort += [
            {
                "fid": r.fid,
                "irr_class": irr_class,
                "n_monthly": r.n,
                "in_monthly_weighted": r.fid in weighted_fids,
            }
            for r in parts_p[irr_class]
        ]
    gates = {
        "monthly_cohorts": "PASS: pooled and station-weighted cohorts equal the frozen cohorts"
    }
    gates["monthly_pooled_identity"] = gate_p
    gates["monthly_weighted_identity"] = gate_w
    return rows, contrasts, pd.DataFrame(cohort), gates, str(ts_dir)


def temporal_scale(frame, classes, reps, seed):
    elig = pd.read_csv(PACKAGE / "temporal" / "evaluation_temporal_site_eligibility.csv")
    common = tuple(sorted(elig.loc[elig["in_common_cohort"].astype(bool), "fid"]))
    require_classed(common, classes, "temporal")
    frozen_csv = PACKAGE / "temporal" / "evaluation_temporal_grouped_metrics.csv"
    keys = ["aggregation", "model", "metric"]
    parts_all = temporal_class_records(frame, common)
    gates = {}
    for part_name, recs in parts_all.items():
        gates[f"temporal_{part_name}_identity"] = gate_against_frozen(
            recs,
            frozen_csv,
            keys,
            {"temporal_class": "all_days_common" if part_name == TEMPORAL_ALL_DAYS else part_name},
            (AGG_POOLED, AGG_WEIGHTED),
            f"temporal {part_name}",
        )
    rows, contrasts, interactions, cohort = [], [], [], []
    for irr_class in CLASSES:
        fids = tuple(f for f in common if classes[f] == irr_class)
        parts = temporal_class_records(frame, fids)
        _, counts = _bootstrap_multiplicities(len(fids), reps, seed)
        boots = {}
        for part_name, recs in parts.items():
            label = "all_days_common" if part_name == TEMPORAL_ALL_DAYS else part_name
            for agg in (AGG_POOLED, AGG_WEIGHTED):
                est = grouped_point_estimates(recs, aggregations=(agg,))
                boot = bootstrap_grouped_from_counts(
                    recs, counts, context=f"temporal {irr_class} {label} {agg}", aggregations=(agg,)
                )
                boots[(part_name, agg)] = (est, boot)
                r, c = metric_rows("temporal", label, irr_class, agg, recs, est, boot, reps, seed)
                rows += r
                contrasts += c
        n_ret = int(sum(r.n for r in parts[TEMPORAL_CLASS_RETRIEVAL]))
        n_btw = int(sum(r.n for r in parts[TEMPORAL_CLASS_BETWEEN]))
        for agg in (AGG_POOLED, AGG_WEIGHTED):
            est_r, boot_r = boots[(TEMPORAL_CLASS_RETRIEVAL, agg)]
            est_b, boot_b = boots[(TEMPORAL_CLASS_BETWEEN, agg)]
            metrics = POOLED_METRICS if agg == AGG_POOLED else WEIGHTED_METRICS
            for metric in metrics:
                point = (est_b[(agg, "swim", metric)] - est_b[(agg, "openet_ensemble", metric)]) - (
                    est_r[(agg, "swim", metric)] - est_r[(agg, "openet_ensemble", metric)]
                )
                arr = boot_b[(agg, CONTRAST, metric)] - boot_r[(agg, CONTRAST, metric)]
                lo, hi = _ci(arr)
                interactions.append(
                    {
                        "irr_class": irr_class,
                        "aggregation": agg,
                        "metric": metric,
                        "interaction": INTERACTION,
                        "estimate": float(point),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "unit": UNITS["temporal"] if metric in ("rmse", "mbe") else "dimensionless",
                        "n_sites": len(fids),
                        "n_pairs_retrieval": n_ret,
                        "n_pairs_between_retrieval": n_btw,
                        "bootstrap_unit": "site",
                        "bootstrap_reps": reps,
                        "bootstrap_seed": seed,
                    }
                )
        cohort += [
            {"fid": r.fid, "irr_class": irr_class, "n_retrieval": r.n, "n_between_retrieval": b.n}
            for r, b in zip(
                parts[TEMPORAL_CLASS_RETRIEVAL], parts[TEMPORAL_CLASS_BETWEEN], strict=True
            )
        ]
    return rows, contrasts, interactions, pd.DataFrame(cohort), gates


# ---------------------------------------------------------------------------
# freeze


def _git(args):
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def build(reps, seed):
    classes = load_class_map()
    frame = read_paired_record_frame(PACKAGE / "daily" / "evaluation_paired_daily_records.csv")
    d_rows, d_con, d_cohort, d_gate = daily_scale(frame, classes, reps, seed)
    m_rows, m_con, m_cohort, m_gates, ts_dir = monthly_scale(classes, reps, seed)
    t_rows, t_con, t_int, t_cohort, t_gates = temporal_scale(frame, classes, reps, seed)
    metrics = pd.DataFrame(d_rows + m_rows + t_rows)
    contrasts = pd.DataFrame(d_con + m_con + t_con)
    interactions = pd.DataFrame(t_int)
    cohort = (
        d_cohort.merge(m_cohort, on=["fid", "irr_class"], how="outer")
        .merge(t_cohort, on=["fid", "irr_class"], how="outer")
        .sort_values(["irr_class", "fid"])
        .reset_index(drop=True)
    )
    cohort["in_temporal_common"] = cohort["n_retrieval"].notna()
    gates = {"daily_identity": d_gate, **m_gates, **t_gates}
    class_counts = {
        "daily": {c: int((d_cohort.irr_class == c).sum()) for c in CLASSES},
        "monthly_pooled": {c: int((m_cohort.irr_class == c).sum()) for c in CLASSES},
        "monthly_weighted": {
            c: int(((m_cohort.irr_class == c) & m_cohort.in_monthly_weighted).sum())
            for c in CLASSES
        },
        "temporal_common": {c: int((t_cohort.irr_class == c).sum()) for c in CLASSES},
    }
    inputs = {
        str(CLASS_CSV.relative_to(REPO)): pm.sha256_file(CLASS_CSV),
        "paper/data/final/e1_openet_benchmark/daily/evaluation_paired_daily_records.csv": pm.sha256_file(
            PACKAGE / "daily" / "evaluation_paired_daily_records.csv"
        ),
        "paper/data/final/e1_openet_benchmark/monthly/evaluation_grouped_monthly_metadata.json": pm.sha256_file(
            PACKAGE / "monthly" / "evaluation_grouped_monthly_metadata.json"
        ),
        "paper/data/final/e1_openet_benchmark/temporal/evaluation_temporal_site_eligibility.csv": pm.sha256_file(
            PACKAGE / "temporal" / "evaluation_temporal_site_eligibility.csv"
        ),
        "paper/data/final/e1_openet_benchmark/MANIFEST.json": pm.sha256_file(
            PACKAGE / "MANIFEST.json"
        ),
    }
    return {
        "metrics": metrics,
        "contrasts": contrasts,
        "interactions": interactions,
        "cohort": cohort,
        "gates": gates,
        "class_counts": class_counts,
        "inputs": inputs,
        "monthly_archive_ts_dir": ts_dir,
    }


FILES = {
    "metrics": "e1_class_split_metrics.csv",
    "contrasts": "e1_class_split_contrasts.csv",
    "interactions": "e1_class_split_interactions.csv",
    "cohort": "e1_class_split_cohorts.csv",
}


def write(result, reps, seed, out_dir):
    out_dir.mkdir(parents=True, exist_ok=False)
    for key, name in FILES.items():
        result[key].to_csv(out_dir / name, index=False)
    script_copy = out_dir / Path(__file__).name
    shutil.copy2(__file__, script_copy)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    head = _git(["rev-parse", "HEAD"]).strip()
    dirty = [
        ln[3:] for ln in _git(["status", "--porcelain"]).splitlines() if not ln.startswith("??")
    ]
    parent = json.load(open(PACKAGE / "MANIFEST.json"))
    files = {name: pm.sha256_file(out_dir / name) for name in [*FILES.values(), script_copy.name]}
    manifest = {
        "schema_version": SCHEMA,
        "status": "frozen_for_supplement_table_s11",
        "promoted_at": now,
        "promoted_at_git_sha": head,
        "worktree_modified_tracked_paths": len(dirty),
        "parent_package": {
            "schema_version": parent["schema_version"],
            "internal_archive_id": parent["internal_archive_id"],
            "promoted_at": parent["promoted_at"],
            "promoted_at_git_sha": parent["promoted_at_git_sha"],
        },
        "scope": (
            "Irrigation-class split (E2 fold classes) of the frozen E1 daily record, the "
            "Volk-protocol monthly record replicated from the run archive, and the common "
            "temporal cohort; pooled and sqrt(n) station-weighted KGE/RMSE/MBE for SWIM-RS and "
            "the OpenET ensemble, SWIM-minus-OpenET contrasts with whole-site bootstrap "
            "intervals, and the per-class retrieval/between-retrieval support interaction"
        ),
        "class_source": str(CLASS_CSV.relative_to(REPO)),
        "monthly_archive_series": result["monthly_archive_ts_dir"],
        "class_counts": result["class_counts"],
        "bootstrap": {
            "reps": reps,
            "seed": seed,
            "unit": "site",
            "interval": "95% percentile",
            "note": "one multiplicity matrix per class cohort; the temporal partitions of a class share its draws",
        },
        "gates": result["gates"],
        "inputs_sha256": result["inputs"],
        "files": files,
        "consumers": [
            "paper/text/supp.md Table S11 (class-split bias)",
            "paper/text/main.md abstract, Results 3.2 class-split sentence, Results 3.3 temporal sentence, Discussion 4.3",
        ],
    }
    (out_dir / "MANIFEST.json").write_bytes(pm._json_bytes(manifest))
    parent.setdefault("addenda", {})["class_split"] = {
        "dir": f"{OUT_DIRNAME}/",
        "manifest": f"{OUT_DIRNAME}/MANIFEST.json",
        "manifest_sha256": pm.sha256_file(out_dir / "MANIFEST.json"),
        "files_sha256": {f"{OUT_DIRNAME}/{k}": v for k, v in files.items()},
        "note": "derived from the frozen daily/monthly/temporal records; no frozen file changed",
    }
    parent["promotion_history"].append(
        {
            "promoted_at": now,
            "git_sha": head,
            "scope": "addendum",
            "change": "irrigation-class split of the frozen records added under class_split/ (Table S11)",
            "producer": "examples/5_Flux_Ensemble/e1_class_split.py",
        }
    )
    (PACKAGE / "MANIFEST.json").write_bytes(pm._json_bytes(parent))
    return manifest


def report(result):
    m = result["metrics"]
    show = m[m.metric.isin(["kge", "rmse", "mbe"])].pivot_table(
        index=["scale", "temporal_class", "aggregation", "irr_class", "n_sites", "n_pairs"],
        columns=["model", "metric"],
        values="estimate",
    )
    pd.set_option("display.width", 220)
    print(show.round(3).to_string())
    print()
    c = result["contrasts"][result["contrasts"].metric.isin(["kge", "rmse", "mbe"])]
    print(
        c[
            [
                "scale",
                "temporal_class",
                "aggregation",
                "irr_class",
                "metric",
                "estimate",
                "ci95_low",
                "ci95_high",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print()
    print(
        result["interactions"][
            ["irr_class", "aggregation", "metric", "estimate", "ci95_low", "ci95_high"]
        ]
        .round(3)
        .to_string(index=False)
    )
    print()
    for k, v in result["gates"].items():
        print(f"{k}: {v}")
    print(json.dumps(result["class_counts"]))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--write", action="store_true", help="freeze under e1_openet_benchmark/class_split/"
    )
    args = ap.parse_args(argv)
    out_dir = PACKAGE / OUT_DIRNAME
    if args.write and out_dir.exists():
        raise ClassSplitError(f"{out_dir} exists; move it aside (never overwrite a frozen output)")
    result = build(args.reps, args.seed)
    report(result)
    if args.write:
        manifest = write(result, args.reps, args.seed, out_dir)
        print(
            f"\nfrozen {len(manifest['files'])} files under {out_dir}; parent MANIFEST.json updated"
        )
    else:
        print("\ncheck mode: nothing written (pass --write to freeze)")


if __name__ == "__main__":
    main()
