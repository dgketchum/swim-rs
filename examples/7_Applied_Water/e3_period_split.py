"""Freeze the E3 applied-water record split at the 2016 start of the calibration targets.

The frozen Fig. 6 field-year record (``paper/data/final/figures/fig06_field_years.csv``)
carries the 404 scored SLV field-years for both parameter paths, with within-field
anomalies taken about each field's full-record mean. The calibration targets begin
in 2016, so the field-years before that date are a temporal holdout. This script
splits the record into ``pre_2016`` (year < 2016) and ``from_2016`` and freezes,
for each treatment and period, the pooled annual-depth r, RMSE and percent bias
and the within-field anomaly r and OLS slope, with the anomalies left on the
full-record basis (a split of the Table 6 statistics, not a re-centering). The
pre-minus-post difference of each statistic gets a whole-field bootstrap
interval.

Gate: the unsplit record must reproduce the Table 6 point values frozen in
``fig06_bootstrap_effects.csv`` for both treatments. Outputs land in
``paper/data/final/e3_applied_water_period_split/`` with a MANIFEST.json and a
copy of this script. No existing frozen file is modified.

Usage:
    uv run python examples/7_Applied_Water/e3_period_split.py            # gate + report
    uv run python examples/7_Applied_Water/e3_period_split.py --write    # freeze
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FINAL = REPO / "paper" / "data" / "final"
FIELD_YEARS = FINAL / "figures" / "fig06_field_years.csv"
EFFECTS = FINAL / "figures" / "fig06_bootstrap_effects.csv"
FIG_MANIFEST = FINAL / "figures" / "fig_manifest.json"
OUT = FINAL / "e3_applied_water_period_split"
SCHEMA = "e3_applied_water_period_split/v1"
SPLIT_YEAR = 2016
PERIODS = ("pre_2016", "from_2016")
TREATMENTS = ("local_calibration", "e1_irrigated_transfer")
REPLICATION_TOL = 1e-9
STATS = (
    "pooled_annual_depth_r",
    "pooled_rmse_mm",
    "pooled_bias_pct",
    "within_field_anomaly_r",
    "within_field_anomaly_slope",
)
GATE_MAP = {
    "pooled_annual_depth_r": "pooled_annual_depth_r",
    "pooled_rmse_mm": "pooled_rmse_mm",
    "within_field_anomaly_r": "within_field_anomaly_r",
    "within_field_anomaly_slope": "within_field_anomaly_slope",
}
VALUE_COL = {
    "local_calibration": "value_local_calibration",
    "e1_irrigated_transfer": "value_e1_irrigated_transfer",
}


class PeriodSplitError(RuntimeError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(args):
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


COLS = ("metered_depth_mm", "sim_applied_mm", "anomaly_metered_mm", "anomaly_sim_mm")


def stats(g):
    return stats_arrays(*(g[c].to_numpy(float) for c in COLS))


def stats_arrays(met, sim, am, asim):
    slope = np.polyfit(am, asim, 1)[0]
    return {
        "pooled_annual_depth_r": float(np.corrcoef(met, sim)[0, 1]),
        "pooled_rmse_mm": float(np.sqrt(np.mean((sim - met) ** 2))),
        "pooled_bias_pct": float((sim.sum() - met.sum()) / met.sum() * 100.0),
        "within_field_anomaly_r": float(np.corrcoef(am, asim)[0, 1]),
        "within_field_anomaly_slope": float(slope),
    }


def load():
    d = pd.read_csv(FIELD_YEARS)
    if set(d.treatment.unique()) != set(TREATMENTS):
        raise PeriodSplitError(f"unexpected treatments {sorted(d.treatment.unique())}")
    if (d.basin != "SLV").any():
        raise PeriodSplitError("non-SLV rows in the frozen field-year record")
    d["period"] = np.where(d.year < SPLIT_YEAR, "pre_2016", "from_2016")
    return d


def gate(d):
    frozen = pd.read_csv(EFFECTS).set_index("statistic")
    worst = 0.0
    n = 0
    for t in TREATMENTS:
        s = stats(d[d.treatment == t])
        for mine, theirs in GATE_MAP.items():
            worst = max(worst, abs(s[mine] - float(frozen.loc[theirs, VALUE_COL[t]])))
            n += 1
        worst = max(
            worst,
            abs(abs(s["pooled_bias_pct"]) - float(frozen.loc["pooled_abs_bias_pct", VALUE_COL[t]])),
        )
        n += 1
    if worst > REPLICATION_TOL:
        raise PeriodSplitError(f"Table 6 values not reproduced: max abs diff {worst:.3e}")
    return f"PASS: {n} Table 6 point values reproduced, max abs diff {worst:.3e} at tolerance {REPLICATION_TOL}"


def build(reps, seed):
    d = load()
    g = gate(d)
    rows, diffs = [], []
    for t in TREATMENTS:
        sub = d[d.treatment == t].reset_index(drop=True)
        fields = np.array(sorted(sub.site_id.unique()))
        rows_by_field = [np.flatnonzero(sub.site_id.to_numpy() == f) for f in fields]
        arrays = [sub[c].to_numpy(float) for c in COLS]
        is_pre = (sub.period == "pre_2016").to_numpy()
        point = {}
        for p in PERIODS:
            gp = sub[sub.period == p]
            point[p] = stats(gp)
            rows.append(
                {
                    "treatment": t,
                    "period": p,
                    "n_field_years": len(gp),
                    "n_fields": gp.site_id.nunique(),
                    "mean_metered_mm": float(gp.metered_depth_mm.mean()),
                    **point[p],
                }
            )
        rng = np.random.default_rng(seed)
        boot = {s: np.empty(reps) for s in STATS}
        for i in range(reps):
            pick = rng.integers(0, len(fields), len(fields))
            idx = np.concatenate([rows_by_field[j] for j in pick])
            pre = idx[is_pre[idx]]
            post = idx[~is_pre[idx]]
            spre = stats_arrays(*(a[pre] for a in arrays))
            spost = stats_arrays(*(a[post] for a in arrays))
            for s in STATS:
                boot[s][i] = spre[s] - spost[s]
        for s in STATS:
            lo, hi = np.percentile(boot[s], [2.5, 97.5])
            diffs.append(
                {
                    "treatment": t,
                    "statistic": s,
                    "contrast": "pre_2016_minus_from_2016",
                    "estimate": point["pre_2016"][s] - point["from_2016"][s],
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "n_fields": len(fields),
                    "bootstrap_unit": "field (all years retained per sampled field)",
                    "bootstrap_reps": reps,
                    "bootstrap_seed": seed,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(diffs), g


def write(metrics, diffs, g, reps, seed):
    OUT.mkdir(exist_ok=True)
    files = {"e3_period_split_metrics.csv": metrics, "e3_period_split_contrasts.csv": diffs}
    for name, df in files.items():
        df.to_csv(OUT / name, index=False)
    script_copy = OUT / Path(__file__).name
    shutil.copy2(__file__, script_copy)
    fig_manifest = json.load(open(FIG_MANIFEST))
    manifest = {
        "schema_version": SCHEMA,
        "status": "frozen_for_results_reporting",
        "promoted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "promoted_at_git_sha": _git(["rev-parse", "HEAD"]).strip(),
        "worktree_modified_tracked_paths": len(
            [ln for ln in _git(["status", "--short"]).splitlines() if not ln.startswith("??")]
        ),
        "parent_package": {
            "package": fig_manifest.get("package"),
            "frozen_utc": fig_manifest.get("frozen_utc"),
            "repo_git_sha": fig_manifest.get("repo_git_sha"),
            "field_years_sha256_in_fig_manifest": fig_manifest["output_files"][
                "fig06_field_years.csv"
            ]["sha256"],
        },
        "scope": (
            "Temporal holdout split of the 404 scored E3 field-years (50 SLV fields, both parameter "
            "paths) at the 2016 start of the calibration targets: pooled annual-depth r, RMSE and "
            "percent bias, within-field anomaly r and OLS slope per period, and whole-field "
            "bootstrap intervals on the pre-minus-post difference"
        ),
        "split_rule": "pre_2016: year < 2016; from_2016: year >= 2016",
        "anomaly_basis": (
            "within-field anomalies are the frozen full-record anomalies (each field's 2011-2021 "
            "record mean removed), split by period without re-centering; a split of the Table 6 "
            "statistics, not a per-period re-estimation"
        ),
        "period_counts": {
            p: {
                "n_field_years": int(metrics[(metrics.period == p)].n_field_years.iloc[0]),
                "n_fields": int(metrics[(metrics.period == p)].n_fields.iloc[0]),
            }
            for p in PERIODS
        },
        "bootstrap": {
            "reps": reps,
            "seed": seed,
            "unit": "field (all years retained per sampled field)",
            "interval": "95% percentile",
            "note": "one RNG per treatment; pre and post statistics computed on the same field draw",
        },
        "gates": {"table6_identity": g},
        "inputs_sha256": {
            str(p.relative_to(REPO)): _sha256(p) for p in (FIELD_YEARS, EFFECTS, FIG_MANIFEST)
        },
        "files": {name: _sha256(OUT / name) for name in [*files, script_copy.name]},
        "consumers": [
            "paper/text/main.md Results 3.4 period-split sentence; Conclusions Landsat-era sentence"
        ],
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--write", action="store_true")
    p.add_argument("--reps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    metrics, diffs, g = build(a.reps, a.seed)
    print(g)
    print(metrics.round(3).to_string(index=False))
    print(
        diffs[["treatment", "statistic", "estimate", "ci95_low", "ci95_high"]]
        .round(3)
        .to_string(index=False)
    )
    if a.write:
        write(metrics, diffs, g, a.reps, a.seed)


if __name__ == "__main__":
    main()
