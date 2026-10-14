"""Freeze the E2 paired transfer-minus-calibration effects on the 47-site closure pool.

The frozen E2 transfer table reports arm medians only. This script derives,
from the frozen per-site comparison, the median within-site difference between
the irrigation-stratified transfer and calibration at each site (and, for
reference, between each SWIM-RS arm and the Landsat ensemble), with whole-site
bootstrap intervals, on the full 47-site pool and on four partitions:

  overlap   the pool sites that belong to the 60-site E1 source pool from
            which the transferred class vectors were built (Methods 2.6);
  unseen    the pool sites absent from parameter-set construction;
  irrigated / rainfed   the E2 parameter class (closure_pool_sites.csv).

Gates: the 47-site arm medians must reproduce ``transfer_refresh_summary.csv``
and the calibrated-minus-ensemble paired median must reproduce
``paired_delta_bootstrap.csv`` (closure_corrected, daily). Outputs land in
``e2_closure_pool/transfer_paired_effects/`` with a MANIFEST.json and a copy of
this script; the parent MANIFEST.json gains an ``addenda`` pointer. No existing
frozen file is modified.

Usage:
    uv run python examples/6_Flux_International/e2_transfer_paired_effects.py           # gates + report
    uv run python examples/6_Flux_International/e2_transfer_paired_effects.py --write   # freeze
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
PACKAGE = FINAL / "e2_closure_pool"
OUT_DIRNAME = "transfer_paired_effects"
PERSITE = PACKAGE / "transfer" / "transfer_comparison_persite.csv"
POOL = PACKAGE / "closure_pool" / "closure_pool_sites.csv"
REFRESH = PACKAGE / "closure_pool" / "transfer_refresh_summary.csv"
PAIRED = PACKAGE / "closure_pool" / "paired_delta_bootstrap.csv"
E1_SOURCE = FINAL / "e2_run23_transfer_vectors_by_irrigation_metadata.json"
SCHEMA = "e2_closure_pool_transfer_paired_effects/v1"
REPLICATION_TOL = 1e-9
ARMS = {
    "e3_uncal": "Uncalibrated parameters",
    "ex5_transfer_strat": "Transferred by irrigation class",
    "e3_cal": "Calibrated at each site",
    "ls_ensemble": "Landsat ensemble",
}
REFRESH_LABELS = {
    "e3_uncal": "E3 uncalibrated/default",
    "ex5_transfer_strat": "Ex5 stratified transfer",
    "e3_cal": "E3 calibrated",
    "ls_ensemble": "LS ensemble",
}
CONTRASTS = (
    ("ex5_transfer_strat", "e3_cal"),
    ("e3_cal", "ls_ensemble"),
    ("ex5_transfer_strat", "ls_ensemble"),
)
METRICS = ("kge", "rmse", "bias", "abs_bias")
FAVORABLE = {
    "kge": "positive",
    "rmse": "negative",
    "bias": "directional_only",
    "abs_bias": "negative",
}


class PairedEffectsError(RuntimeError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(args):
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout


def load():
    pool = pd.read_csv(POOL, index_col=0)
    persite = pd.read_csv(PERSITE, index_col=0)
    missing = sorted(set(pool.index) - set(persite.index))
    if missing:
        raise PairedEffectsError(f"pool sites missing from the per-site table: {missing}")
    persite = persite.loc[pool.index].copy()
    for arm in ARMS:
        persite[f"{arm}_abs_bias"] = persite[f"{arm}_bias"].abs()
    meta = json.load(open(E1_SOURCE))
    source = set()
    for cls in meta["classes"].values():
        source.update(cls["fids"])
    if len(source) != meta["n_source_sites"]:
        raise PairedEffectsError(
            f"E1 source pool has {len(source)} fids, metadata says {meta['n_source_sites']}"
        )
    persite["irr_class"] = pool["irrigation_class"]
    persite["in_e1_source_pool"] = [f in source for f in persite.index]
    return persite, len(source)


def partitions(persite):
    return {
        "all": persite,
        "unseen": persite[~persite.in_e1_source_pool],
        "overlap": persite[persite.in_e1_source_pool],
        "irrigated": persite[persite.irr_class == "irrigated"],
        "rainfed": persite[persite.irr_class == "rainfed"],
    }


def gates(persite):
    out = {}
    refresh = pd.read_csv(REFRESH)
    refresh = refresh[refresh.basis == "daily"].set_index("config")
    worst = 0.0
    for arm, label in REFRESH_LABELS.items():
        for m, col in (("kge", "kge_med"), ("rmse", "rmse_med"), ("bias", "bias_med")):
            worst = max(worst, abs(persite[f"{arm}_{m}"].median() - refresh.loc[label, col]))
    if worst > REPLICATION_TOL:
        raise PairedEffectsError(f"arm medians differ from transfer_refresh_summary by {worst:.3e}")
    out["arm_medians_identity"] = (
        f"PASS: 12 daily arm medians reproduced, max abs diff {worst:.3e} at tolerance {REPLICATION_TOL}"
    )
    paired = pd.read_csv(PAIRED)
    paired = paired[(paired.basis == "daily") & (paired.tier == "closure_corrected")].set_index(
        "metric"
    )
    worst = 0.0
    for m in ("kge", "rmse", "bias"):
        d = (persite[f"e3_cal_{m}"] - persite[f"ls_ensemble_{m}"]).median()
        worst = max(worst, abs(d - paired.loc[m, "median"]))
    if worst > REPLICATION_TOL:
        raise PairedEffectsError(f"calibrated-minus-ensemble medians differ by {worst:.3e}")
    out["paired_cal_minus_ensemble_identity"] = (
        f"PASS: 3 paired medians reproduced, max abs diff {worst:.3e} at tolerance {REPLICATION_TOL}"
    )
    return out


def bootstrap_median(d, rng, reps):
    d = np.asarray(d, float)
    n = len(d)
    medians = np.median(d[rng.integers(0, n, size=(reps, n))], axis=1)
    return (
        float(np.median(d)),
        float(np.percentile(medians, 2.5)),
        float(np.percentile(medians, 97.5)),
    )


def build(reps, seed):
    persite, n_source = load()
    gate = gates(persite)
    med_rows, eff_rows = [], []
    for part, sub in partitions(persite).items():
        for arm, label in ARMS.items():
            med_rows.append(
                {
                    "partition": part,
                    "n_sites": len(sub),
                    "arm": arm,
                    "parameter_source": label,
                    **{m: float(sub[f"{arm}_{m}"].median()) for m in METRICS},
                }
            )
        rng = np.random.default_rng(seed)  # one RNG per partition block
        for a, b in CONTRASTS:
            for m in METRICS:
                d = (sub[f"{a}_{m}"] - sub[f"{b}_{m}"]).to_numpy(float)
                med, lo, hi = bootstrap_median(d, rng, reps)
                eff_rows.append(
                    {
                        "partition": part,
                        "n_sites": len(sub),
                        "contrast": f"{a}_minus_{b}",
                        "metric": m,
                        "median": med,
                        "ci_low": lo,
                        "ci_high": hi,
                        "frac_sites_first_higher": float(np.mean(d > 0)),
                        "unit": "mm d-1" if m != "kge" else "dimensionless",
                        "favorable_direction": FAVORABLE[m],
                        "bootstrap_unit": "site",
                        "bootstrap_reps": reps,
                        "bootstrap_seed": seed,
                    }
                )
    sites = persite[["irr_class", "in_e1_source_pool"]].copy()
    for arm in ARMS:
        for m in ("kge", "rmse", "bias"):
            sites[f"{arm}_{m}"] = persite[f"{arm}_{m}"]
    sites.index.name = "fid"
    return pd.DataFrame(med_rows), pd.DataFrame(eff_rows), sites.reset_index(), gate, n_source


def report(medians, effects):
    print(
        medians[medians.arm != "e3_uncal"]
        .pivot_table(index=["partition", "arm"], values="kge")
        .round(3)
    )
    print(
        effects[effects.contrast == "ex5_transfer_strat_minus_e3_cal"][
            ["partition", "n_sites", "metric", "median", "ci_low", "ci_high"]
        ].round(3)
    )


def write(medians, effects, sites, gate, n_source, reps, seed):
    out = PACKAGE / OUT_DIRNAME
    out.mkdir(exist_ok=True)
    files = {
        "e2_transfer_partition_medians.csv": medians,
        "e2_transfer_paired_effects.csv": effects,
        "e2_transfer_partition_sites.csv": sites,
    }
    for name, df in files.items():
        df.to_csv(out / name, index=False)
    script_copy = out / Path(__file__).name
    shutil.copy2(__file__, script_copy)
    inputs = {
        str(p.relative_to(REPO)): _sha256(p)
        for p in (PERSITE, POOL, REFRESH, PAIRED, E1_SOURCE, PACKAGE / "MANIFEST.json")
    }
    parent = json.load(open(PACKAGE / "MANIFEST.json"))
    counts = {p: int(n) for p, n in medians.groupby("partition")["n_sites"].first().items()}
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
            "promoted_at": parent["promoted_at"],
            "promoted_at_git_sha": parent["promoted_at_git_sha"],
        },
        "scope": (
            "Median within-site daily differences between the irrigation-stratified transfer, "
            "calibration at each site, and the Landsat ensemble on the 47-site closure pool and "
            "on its E1-overlap / unseen and irrigated / rainfed partitions, with whole-site "
            "bootstrap intervals; arm medians per partition"
        ),
        "overlap_rule": (
            f"overlap = pool site whose fid is among the {n_source} E1 source sites "
            "(irrigated + rainfed classes) in e2_run23_transfer_vectors_by_irrigation_metadata.json; "
            "unseen = the rest (Methods 2.6: 10 overlap, 37 unseen)"
        ),
        "partition_counts": counts,
        "bootstrap": {
            "reps": reps,
            "seed": seed,
            "unit": "site",
            "interval": "95% percentile",
            "note": "one RNG per partition block, contrasts and metrics drawn in file order",
        },
        "gates": gate,
        "inputs_sha256": inputs,
        "files": {name: _sha256(out / name) for name in [*files, script_copy.name]},
        "consumers": [
            "paper/text/main.md Results 3.3 paired transfer sentence and unseen/overlap sentence; Discussion 4.2"
        ],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    parent.setdefault("addenda", {})[OUT_DIRNAME] = {
        "dir": f"{OUT_DIRNAME}/",
        "manifest": f"{OUT_DIRNAME}/MANIFEST.json",
        "manifest_sha256": _sha256(out / "MANIFEST.json"),
        "files_sha256": {f"{OUT_DIRNAME}/{k}": v for k, v in manifest["files"].items()},
        "note": "derived from the frozen transfer per-site table; no frozen file changed",
    }
    (PACKAGE / "MANIFEST.json").write_text(json.dumps(parent, indent=2) + "\n")
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--write", action="store_true")
    p.add_argument("--reps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260908)
    a = p.parse_args()
    medians, effects, sites, gate, n_source = build(a.reps, a.seed)
    for k, v in gate.items():
        print(k, v)
    report(medians, effects)
    if a.write:
        write(medians, effects, sites, gate, n_source, a.reps, a.seed)


if __name__ == "__main__":
    main()
