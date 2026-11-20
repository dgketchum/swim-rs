"""Freeze the ET-level split of E1 retrieval-date bias (main text, Fig. 3 paragraph).

The frozen E1 temporal decomposition reports SWIM-RS and OpenET mean bias on
retrieval dates for the 40-site common temporal cohort. This script partitions
those same retrieval-date records by daily ET level and freezes the mean bias
error (MBE, modeled minus observed) of both series in each level, pooled and
sqrt(n) station-weighted, with whole-site bootstrap intervals on two contrasts:
the signed MBE difference (SWIM minus OpenET) and the difference in absolute
station-weighted MBE (|SWIM| minus |OpenET|, negative favors SWIM-RS).

Two stratifiers are used:

  flux_et   tower ET, bins [0, 2), [2, 4), [4, 6), [6, inf) mm d-1. Binning on
            the observation pushes any imperfect model toward positive bias in
            the low bins and negative bias in the high bins; the contrast
            compares both series on the same days, so it is not removed by it.
  eto       OpenET bias-corrected GridMET ETo (the reference ET the benchmark
            itself uses), bins [0, 3), [3, 5), [5, 7), [7, inf) mm d-1. It is
            independent of the tower and serves as a check on the flux_et split.

Each stratum is reported for all cohort sites and for the irrigated and rainfed
classes of ``class_split/e1_class_split_cohorts.csv``. A site enters a binned
station-weighted estimate with at least MIN_DAYS_PER_BIN days in that bin.

Before anything is written, the unbinned retrieval-date MBE must reproduce the
frozen temporal (all sites) and class_split (irrigated, rainfed) estimates to
REPLICATION_TOL, and the ETo file must match the hash recorded in the frozen
daily metadata. Outputs land in ``e1_openet_benchmark/et_level_split/`` with a
MANIFEST.json and a copy of this script; the parent MANIFEST.json gains an
``addenda.et_level_split`` pointer and a ``promotion_history`` entry. No
existing frozen file is otherwise modified.

Usage:
    uv run python examples/5_Flux_Ensemble/e1_et_level_split.py            # gates + report
    uv run python examples/5_Flux_Ensemble/e1_et_level_split.py --write    # freeze
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "paper/data/final/e1_openet_benchmark"
RECORDS = PKG / "daily/evaluation_paired_daily_records.csv"
DAILY_META = PKG / "daily/evaluation_grouped_daily_metadata.json"
COHORTS = PKG / "class_split/e1_class_split_cohorts.csv"
CLASS_METRICS = PKG / "class_split/e1_class_split_metrics.csv"
TEMPORAL_METRICS = PKG / "temporal/evaluation_temporal_grouped_metrics.csv"
ETO = REPO / "examples/5_Flux_Ensemble/data/openet_refet/openet_eto.csv"
OUT = PKG / "et_level_split"

BINS = {
    "flux_et": [0.0, 2.0, 4.0, 6.0, np.inf],
    "eto": [0.0, 3.0, 5.0, 7.0, np.inf],
}
MIN_DAYS_UNBINNED = 10
MIN_DAYS_PER_BIN = 3
BOOT_REPS = 10000
BOOT_SEED = 42
REPLICATION_TOL = 1e-6
N_SITES = 40
N_RETRIEVAL = 4541
SAFE_FLOAT = "%.17g"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records():
    r = pd.read_csv(RECORDS, parse_dates=["date"])
    c = pd.read_csv(COHORTS)
    r = r.merge(c[["fid", "irr_class", "in_temporal_common"]], on="fid", how="left")
    assert r["irr_class"].notna().all(), "site missing from class cohorts"
    r = r[(r["temporal_class"] == "retrieval") & r["in_temporal_common"].astype(bool)].copy()
    eto = pd.read_csv(ETO).set_index("site_id").stack().rename("eto").reset_index()
    eto.columns = ["fid", "date", "eto"]
    eto["date"] = pd.to_datetime(eto["date"], format="%Y%m%d")
    r = r.merge(eto, on=["fid", "date"], how="left")
    assert r["eto"].notna().all(), "ETo missing on a retrieval date"
    r["err_swim"] = r["swim_et_mm_d"] - r["flux_et_mm_d"]
    r["err_openet"] = r["openet_et_mm_d"] - r["flux_et_mm_d"]
    return r


def site_table(g, sites, min_days):
    s = g.groupby("fid").agg(
        n=("err_swim", "size"), swim=("err_swim", "mean"), openet=("err_openet", "mean")
    )
    s = s[s["n"] >= min_days].reindex(sites)
    return s


def weighted(s, mult):
    """sqrt(n) station-weighted means; mult is (reps, sites) multiplicity or (sites,)."""
    ok = s["n"].notna().to_numpy()
    w = np.where(ok, np.sqrt(s["n"].fillna(0).to_numpy()), 0.0)
    q_s = s["swim"].fillna(0).to_numpy()
    q_o = s["openet"].fillna(0).to_numpy()
    wm = mult * w
    den = wm.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (wm * q_s).sum(axis=-1) / den, (wm * q_o).sum(axis=-1) / den


def gates(r):
    out = {}
    meta = json.load(open(DAILY_META))
    want = meta["input_hashes"]["openet_eto_csv"]
    want = want.get("sha256", want) if isinstance(want, dict) else want
    got = sha256(ETO)
    assert got == want, f"ETo hash {got} != frozen {want}"
    out["eto_hash"] = "PASS: ETo csv matches the frozen daily metadata"
    assert r["fid"].nunique() == N_SITES and len(r) == N_RETRIEVAL, (r["fid"].nunique(), len(r))
    out["cohort"] = f"PASS: {N_SITES} sites, {N_RETRIEVAL} retrieval days"

    t = pd.read_csv(TEMPORAL_METRICS)
    t = t[(t["temporal_class"] == "retrieval") & (t["metric"] == "mbe")]
    cm = pd.read_csv(CLASS_METRICS)
    cm = cm[
        (cm["scale"] == "temporal")
        & (cm["temporal_class"] == "retrieval")
        & (cm["metric"] == "mbe")
    ]
    diffs = []
    for cls in ("all", "irrigated", "rainfed"):
        g = r if cls == "all" else r[r["irr_class"] == cls]
        sites = sorted(g["fid"].unique())
        s = site_table(g, sites, MIN_DAYS_UNBINNED)
        sw_s, sw_o = weighted(s, np.ones(len(sites)))
        mine = {
            ("pooled_observations", "swim"): g["err_swim"].mean(),
            ("pooled_observations", "openet_ensemble"): g["err_openet"].mean(),
            ("sqrt_n_weighted_site_metric", "swim"): sw_s,
            ("sqrt_n_weighted_site_metric", "openet_ensemble"): sw_o,
        }
        ref = t if cls == "all" else cm[cm["irr_class"] == cls]
        for (agg, model), v in mine.items():
            row = ref[(ref["aggregation"] == agg) & (ref["model"] == model)]
            assert len(row) == 1, (cls, agg, model)
            diffs.append(abs(float(row["estimate"].iloc[0]) - float(v)))
    mx = max(diffs)
    assert mx < REPLICATION_TOL, f"unbinned MBE not reproduced, max abs diff {mx:.3e}"
    out["unbinned_identity"] = (
        f"PASS: {len(diffs)} frozen retrieval-date MBE estimates reproduced, "
        f"max abs diff {mx:.3e} at tolerance {REPLICATION_TOL:g}"
    )
    return out


def split(r):
    metrics, contrasts = [], []
    rng = np.random.default_rng(BOOT_SEED)
    for cls in ("all", "irrigated", "rainfed"):
        g0 = r if cls == "all" else r[r["irr_class"] == cls]
        sites = sorted(g0["fid"].unique())
        k = len(sites)
        mult = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(BOOT_REPS)])
        for strat, edges in BINS.items():
            col = "flux_et_mm_d" if strat == "flux_et" else "eto"
            for lo, hi in zip(edges[:-1], edges[1:]):
                g = g0[(g0[col] >= lo) & (g0[col] < hi)]
                s = site_table(g, sites, MIN_DAYS_PER_BIN)
                n_sites = int(s["n"].notna().sum())
                n_days = int(s["n"].sum())
                sw_s, sw_o = weighted(s, np.ones(k))
                bs_s, bs_o = weighted(s, mult)
                keep = np.isfinite(bs_s)
                base = dict(
                    stratifier=strat,
                    bin_low=lo,
                    bin_high=hi,
                    irr_class=cls,
                    unit="mm d-1",
                    n_sites=n_sites,
                    n_days=n_days,
                    n_days_pooled=len(g),
                )
                for model, pooled, swv, bsv in (
                    ("swim", g["err_swim"].mean(), sw_s, bs_s),
                    ("openet_ensemble", g["err_openet"].mean(), sw_o, bs_o),
                ):
                    metrics.append(
                        {
                            **base,
                            "model": model,
                            "aggregation": "pooled_observations",
                            "mbe": pooled,
                        }
                    )
                    lo95, hi95 = np.percentile(bsv[keep], [2.5, 97.5])
                    metrics.append(
                        {
                            **base,
                            "model": model,
                            "aggregation": "sqrt_n_weighted_site_metric",
                            "mbe": swv,
                            "ci95_low": lo95,
                            "ci95_high": hi95,
                        }
                    )
                for name, est, draws in (
                    ("mbe_swim_minus_openet", sw_s - sw_o, bs_s - bs_o),
                    (
                        "abs_mbe_swim_minus_abs_mbe_openet",
                        abs(sw_s) - abs(sw_o),
                        np.abs(bs_s) - np.abs(bs_o),
                    ),
                ):
                    lo95, hi95 = np.percentile(draws[keep], [2.5, 97.5])
                    contrasts.append(
                        {
                            **base,
                            "contrast": name,
                            "aggregation": "sqrt_n_weighted_site_metric",
                            "estimate": est,
                            "ci95_low": lo95,
                            "ci95_high": hi95,
                            "valid_reps": int(keep.sum()),
                        }
                    )
    return pd.DataFrame(metrics), pd.DataFrame(contrasts)


def git_state():
    sha = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    mod = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return sha, len(mod)


def report(m, c):
    sw = m[m["aggregation"] == "sqrt_n_weighted_site_metric"].pivot_table(
        index=["irr_class", "stratifier", "bin_low", "bin_high", "n_sites", "n_days"],
        columns="model",
        values="mbe",
    )
    a = c[c["contrast"] == "abs_mbe_swim_minus_abs_mbe_openet"].set_index(
        ["irr_class", "stratifier", "bin_low", "bin_high", "n_sites", "n_days"]
    )[["estimate", "ci95_low", "ci95_high"]]
    print(sw.join(a).round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="freeze outputs under et_level_split/")
    args = ap.parse_args()

    r = load_records()
    g = gates(r)
    for k, v in g.items():
        print(f"{k}: {v}")
    m, c = split(r)
    report(m, c)
    if not args.write:
        return

    if OUT.exists():
        sys.exit(f"exists: {OUT}")
    OUT.mkdir(parents=True)
    files = {
        "e1_et_level_split_metrics.csv": m,
        "e1_et_level_split_contrasts.csv": c,
    }
    for name, df in files.items():
        df.to_csv(OUT / name, index=False, float_format=SAFE_FLOAT)
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)

    sha, n_mod = git_state()
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    parent_path = PKG / "MANIFEST.json"
    inputs = [RECORDS, DAILY_META, COHORTS, CLASS_METRICS, TEMPORAL_METRICS, ETO, parent_path]
    out_files = list(files) + [Path(__file__).name]
    manifest = {
        "schema_version": "e1_openet_benchmark_et_level_split/v1",
        "status": "frozen_for_main_text_fig3_paragraph",
        "promoted_at": now,
        "promoted_at_git_sha": sha,
        "worktree_modified_tracked_paths": n_mod,
        "scope": (
            "Retrieval-date MBE of SWIM-RS and the OpenET ensemble on the 40-site common temporal cohort, "
            "split by tower ET level and by ETo level, for all sites and by irrigation class; pooled and "
            "sqrt(n) station-weighted, with whole-site bootstrap intervals on the SWIM-minus-OpenET MBE "
            "difference and on the absolute-MBE difference"
        ),
        "bins_mm_d": {k: [float(x) for x in v] for k, v in BINS.items()},
        "min_days_per_site_per_bin": MIN_DAYS_PER_BIN,
        "caveat": (
            "flux_et bins condition on the observation, which pushes any imperfect series toward positive "
            "bias at low ET and negative bias at high ET; eto bins are independent of the tower"
        ),
        "bootstrap": {
            "reps": BOOT_REPS,
            "seed": BOOT_SEED,
            "unit": "site",
            "interval": "95% percentile",
            "note": "one multiplicity matrix per class cohort, shared by all bins of that cohort",
        },
        "gates": g,
        "inputs_sha256": {str(p.relative_to(REPO)): sha256(p) for p in inputs},
        "files": {f: sha256(OUT / f) for f in out_files},
    }
    with open(OUT / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    parent = json.load(open(parent_path))
    parent.setdefault("addenda", {})["et_level_split"] = {
        "dir": "et_level_split/",
        "manifest": "et_level_split/MANIFEST.json",
        "manifest_sha256": sha256(OUT / "MANIFEST.json"),
        "files_sha256": {f"et_level_split/{f}": sha256(OUT / f) for f in out_files},
        "note": "derived from the frozen daily/temporal/class_split records; no frozen file changed",
    }
    parent.setdefault("promotion_history", []).append(
        {
            "promoted_at": now,
            "git_sha": sha,
            "scope": "addendum",
            "change": "ET-level split of retrieval-date bias added under et_level_split/ (main text, Fig. 3 paragraph)",
            "producer": "examples/5_Flux_Ensemble/e1_et_level_split.py",
        }
    )
    with open(parent_path, "w") as f:
        json.dump(parent, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
