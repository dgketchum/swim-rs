"""Freeze the E1 daily record split at the 2016 start of the calibration targets.

The frozen E1 package scores SWIM-RS and the reconstructed OpenET benchmark on
the whole paired daily record. The calibration targets begin in 2016, so the
paired days before that date are a temporal holdout. This script partitions the
frozen paired daily record into ``pre_2016`` (date < 2016-01-01) and
``from_2016`` and freezes pooled and station-weighted KGE, RMSE and MBE for both
series in each period, with whole-site bootstrap intervals on the SWIM-minus-
OpenET contrast.

A site enters a period only when it keeps at least MIN_OBS_FOR_METRICS paired
days there (the frozen per-site floor); dropped site-periods are listed in the
cohort file. Before anything is written the unsplit record must reproduce every
frozen daily grouped estimate to REPLICATION_TOL. Outputs land in
``e1_openet_benchmark/period_split/`` with a MANIFEST.json and a copy of this
script; the parent MANIFEST.json gains an ``addenda.period_split`` pointer. No
existing frozen file is modified.

Usage:
    uv run python examples/5_Flux_Ensemble/e1_period_split.py            # gates + report
    uv run python examples/5_Flux_Ensemble/e1_period_split.py --write    # freeze
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from swimrs.evaluation.benchmark import (
    AGG_POOLED,
    AGG_WEIGHTED,
    MIN_OBS_FOR_METRICS,
    POOLED_METRICS,
    WEIGHTED_METRICS,
    _bootstrap_multiplicities,
    bootstrap_grouped_from_counts,
    grouped_point_estimates,
    paired_records_from_frame,
    read_paired_record_frame,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FINAL = REPO / "paper" / "data" / "final"
PACKAGE = FINAL / "e1_openet_benchmark"
OUT_DIRNAME = "period_split"
SPLIT_DATE = pd.Timestamp("2016-01-01")
PERIODS = ("pre_2016", "from_2016")
CONTRAST = "swim_minus_openet"
REPLICATION_TOL = 1e-6
SCHEMA = "e1_openet_benchmark_period_split/v1"
FAVORABLE = {
    "kge": "positive",
    "rmse": "negative",
    "mbe": "directional_only",
    "r": "positive",
    "r2": "positive",
    "slope0": "directional_only",
}


class PeriodSplitError(RuntimeError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(args):
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def _ci(arr):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


def gate_against_frozen(records, tol=REPLICATION_TOL):
    frozen = pd.read_csv(PACKAGE / "daily" / "evaluation_grouped_daily_metrics.csv")
    frozen = frozen[frozen["scale"] == "daily"]
    got = frozen.set_index(["aggregation", "model", "metric"])["estimate"]
    est = grouped_point_estimates(records, aggregations=(AGG_POOLED, AGG_WEIGHTED))
    worst = 0.0
    for key, value in est.items():
        if key not in got.index:
            raise PeriodSplitError(f"frozen table lacks {key}")
        worst = max(worst, abs(float(got.loc[key]) - value))
    if worst > tol:
        raise PeriodSplitError(f"max abs difference {worst:.3e} exceeds {tol}")
    return f"PASS: {len(est)} estimates reproduced, max abs diff {worst:.3e} at tolerance {tol}"


def period_frames(frame):
    pre = frame[frame["date"] < SPLIT_DATE]
    post = frame[frame["date"] >= SPLIT_DATE]
    return {"pre_2016": pre, "from_2016": post}


def period_records(sub):
    """Records for one period; sites under the per-site floor are dropped and reported."""
    counts = sub.groupby("fid").size()
    keep = sorted(counts.index[counts >= MIN_OBS_FOR_METRICS])
    dropped = {f: int(n) for f, n in counts.items() if n < MIN_OBS_FOR_METRICS}
    records = paired_records_from_frame(sub[sub["fid"].isin(keep)])
    return records, dropped


def metric_rows(period, aggregation, records, est, boot, reps, seed):
    n_sites = len(records)
    n_pairs = int(sum(r.n for r in records))
    metrics = POOLED_METRICS if aggregation == AGG_POOLED else WEIGHTED_METRICS
    rows, contrasts = [], []
    for metric in metrics:
        unit = "mm d-1" if metric in ("rmse", "mbe") else "dimensionless"
        for model in ("swim", "openet_ensemble"):
            key = (aggregation, model, metric)
            lo, hi = _ci(boot[key])
            rows.append(
                {
                    "scale": "daily",
                    "period": period,
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
                "scale": "daily",
                "period": period,
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


def build(reps, seed):
    frame = read_paired_record_frame(PACKAGE / "daily" / "evaluation_paired_daily_records.csv")
    frame = frame.assign(date=pd.to_datetime(frame["date"]))
    gate = gate_against_frozen(paired_records_from_frame(frame))
    rows, contrasts, cohort, dropped_all = [], [], [], {}
    for period, sub in period_frames(frame).items():
        records, dropped = period_records(sub)
        dropped_all[period] = dropped
        for agg in (AGG_POOLED, AGG_WEIGHTED):
            est = grouped_point_estimates(records, aggregations=(agg,))
            _, counts = _bootstrap_multiplicities(len(records), reps, seed)
            boot = bootstrap_grouped_from_counts(
                records, counts, context=f"{period} {agg}", aggregations=(agg,)
            )
            r, c = metric_rows(period, agg, records, est, boot, reps, seed)
            rows += r
            contrasts += c
        cohort += [
            {"fid": r.fid, "period": period, "n_days": r.n, "retained": True} for r in records
        ]
        cohort += [
            {"fid": f, "period": period, "n_days": n, "retained": False} for f, n in dropped.items()
        ]
    cohort = pd.DataFrame(cohort).sort_values(["period", "fid"]).reset_index(drop=True)
    return pd.DataFrame(rows), pd.DataFrame(contrasts), cohort, gate, dropped_all


def report(metrics, contrasts):
    view = metrics[metrics["metric"].isin(("kge", "rmse", "mbe"))]
    print(
        view.pivot_table(
            index=["period", "aggregation", "model"], columns="metric", values="estimate"
        ).round(3)
    )
    print(
        contrasts[contrasts["metric"].isin(("kge", "rmse", "mbe"))][
            ["period", "aggregation", "metric", "estimate", "ci95_low", "ci95_high", "n_sites"]
        ].round(3)
    )


def write(metrics, contrasts, cohort, gate, dropped, reps, seed):
    out = PACKAGE / OUT_DIRNAME
    out.mkdir(exist_ok=True)
    files = {
        "e1_period_split_metrics.csv": metrics,
        "e1_period_split_contrasts.csv": contrasts,
        "e1_period_split_cohort.csv": cohort,
    }
    for name, df in files.items():
        df.to_csv(out / name, index=False)
    script_copy = out / Path(__file__).name
    shutil.copy2(__file__, script_copy)
    inputs = {
        "paper/data/final/e1_openet_benchmark/daily/evaluation_paired_daily_records.csv": _sha256(
            PACKAGE / "daily" / "evaluation_paired_daily_records.csv"
        ),
        "paper/data/final/e1_openet_benchmark/daily/evaluation_grouped_daily_metrics.csv": _sha256(
            PACKAGE / "daily" / "evaluation_grouped_daily_metrics.csv"
        ),
        "paper/data/final/e1_openet_benchmark/MANIFEST.json": _sha256(PACKAGE / "MANIFEST.json"),
    }
    parent = json.load(open(PACKAGE / "MANIFEST.json"))
    manifest = {
        "schema_version": SCHEMA,
        "status": "frozen_for_results_reporting",
        "promoted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "promoted_at_git_sha": _git(["rev-parse", "HEAD"]).strip(),
        "worktree_modified_tracked_paths": len(
            [ln for ln in _git(["status", "--short"]).splitlines() if not ln.startswith("??")]
        ),
        "parent_package": {
            "schema_version": parent["schema_version"],
            "internal_archive_id": parent.get("internal_archive_id"),
            "promoted_at": parent["promoted_at"],
            "promoted_at_git_sha": parent["promoted_at_git_sha"],
        },
        "scope": (
            "Temporal holdout split of the frozen E1 daily record at the 2016-01-01 start of the "
            "calibration targets: pooled and sqrt(n) station-weighted KGE/RMSE/MBE for SWIM-RS and "
            "the reconstructed OpenET ensemble in each period, with whole-site bootstrap intervals "
            "on the SWIM-minus-OpenET contrast"
        ),
        "split_rule": "pre_2016: date < 2016-01-01; from_2016: date >= 2016-01-01",
        "site_floor": (
            f"a site enters a period only with at least {MIN_OBS_FOR_METRICS} paired days in it "
            "(the frozen per-site floor); dropped site-periods are listed in the cohort file"
        ),
        "dropped_site_periods": dropped,
        "period_counts": {
            p: {
                "n_sites": int(cohort[(cohort.period == p) & cohort.retained].shape[0]),
                "n_pairs": int(cohort[(cohort.period == p) & cohort.retained].n_days.sum()),
            }
            for p in PERIODS
        },
        "bootstrap": {
            "reps": reps,
            "seed": seed,
            "unit": "site",
            "interval": "95% percentile",
            "note": "one multiplicity matrix per period cohort; pooled and weighted share draws",
        },
        "gates": {"daily_identity": gate},
        "inputs_sha256": inputs,
        "files": {name: _sha256(out / name) for name in [*files, script_copy.name]},
        "consumers": [
            "paper/text/main.md Results 3.2 pre-2016 sentence; Discussion 4.2 temporal-holdout sentence; abstract, 4.1, Conclusions, highlight (time as a transfer axis)"
        ],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    parent.setdefault("addenda", {})[OUT_DIRNAME] = {
        "dir": f"{OUT_DIRNAME}/",
        "manifest": f"{OUT_DIRNAME}/MANIFEST.json",
        "manifest_sha256": _sha256(out / "MANIFEST.json"),
        "files_sha256": {f"{OUT_DIRNAME}/{k}": v for k, v in manifest["files"].items()},
        "note": "derived from the frozen daily record; no frozen file changed",
    }
    (PACKAGE / "MANIFEST.json").write_text(json.dumps(parent, indent=2) + "\n")
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--write", action="store_true")
    p.add_argument("--reps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    metrics, contrasts, cohort, gate, dropped = build(a.reps, a.seed)
    print(gate)
    print("dropped site-periods:", dropped)
    report(metrics, contrasts)
    if a.write:
        write(metrics, contrasts, cohort, gate, dropped, a.reps, a.seed)


if __name__ == "__main__":
    main()
