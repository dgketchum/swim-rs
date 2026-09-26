"""Diff two PEST++ problem definitions (.pst) observation by observation.

Two uses in the Run 23 workflow:

* replay gate: a problem definition rebuilt at HEAD against the Run 22
  container must equal the archived Run 22 definition (builder drift);
* weight audit: the Run 23 definition against Run 22 shows how the target
  values and weights moved under the corrected ETf members.

Observations are aligned by name; observations present on one side only are
counted. Parameter data and ``pestpp_options`` are compared as well.

Usage:
    uv run python compare_problem_definition.py --reference <run22.pst> \\
        --candidate <run23.pst> --out-dir <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyemu


def _pct(x, q):
    return float(np.percentile(x, q)) if x.size else float("nan")


def compare(reference, candidate):
    ra = pyemu.Pst(str(reference))
    rb = pyemu.Pst(str(candidate))
    oa = ra.observation_data.set_index("obsnme")
    ob = rb.observation_data.set_index("obsnme")
    common = oa.index.intersection(ob.index)
    only_a = oa.index.difference(ob.index)
    only_b = ob.index.difference(oa.index)

    obs = pd.DataFrame(
        {
            "obgnme": oa.loc[common, "obgnme"],
            "obsval_ref": oa.loc[common, "obsval"].astype(float),
            "obsval_cand": ob.loc[common, "obsval"].astype(float),
            "weight_ref": oa.loc[common, "weight"].astype(float),
            "weight_cand": ob.loc[common, "weight"].astype(float),
        }
    )
    obs["obsval_diff"] = obs["obsval_cand"] - obs["obsval_ref"]
    obs["weight_rel"] = np.where(
        obs["weight_ref"] > 0, obs["weight_cand"] / obs["weight_ref"] - 1.0, np.nan
    )
    nz = obs[(obs["weight_ref"] > 0) | (obs["weight_cand"] > 0)]
    rel = nz["weight_rel"].dropna().to_numpy()
    summary = {
        "n_obs_ref": int(len(oa)),
        "n_obs_cand": int(len(ob)),
        "n_common": int(len(common)),
        "n_only_ref": int(len(only_a)),
        "n_only_cand": int(len(only_b)),
        "n_weighted_common": int(len(nz)),
        "obsval_identical": bool(np.array_equal(obs["obsval_ref"], obs["obsval_cand"])),
        "weight_identical": bool(np.array_equal(obs["weight_ref"], obs["weight_cand"])),
        "obsval_max_abs_diff": float(obs["obsval_diff"].abs().max()) if len(obs) else 0.0,
        "obsval_mean_diff_weighted": float(nz["obsval_diff"].mean()) if len(nz) else 0.0,
        "weight_mean_ref": float(nz["weight_ref"].mean()) if len(nz) else 0.0,
        "weight_mean_cand": float(nz["weight_cand"].mean()) if len(nz) else 0.0,
        "weight_rel_p05": _pct(rel, 5),
        "weight_rel_p50": _pct(rel, 50),
        "weight_rel_p95": _pct(rel, 95),
        "weight_frac_abs_gt_10pct": float((np.abs(rel) > 0.10).mean()) if rel.size else 0.0,
        "weight_frac_abs_gt_25pct": float((np.abs(rel) > 0.25).mean()) if rel.size else 0.0,
        "weight_zeroed": int(((obs["weight_ref"] > 0) & (obs["weight_cand"] == 0)).sum()),
        "weight_activated": int(((obs["weight_ref"] == 0) & (obs["weight_cand"] > 0)).sum()),
    }
    by_group = (
        nz.groupby("obgnme")
        .agg(
            n=("weight_ref", "size"),
            obsval_mean_diff=("obsval_diff", "mean"),
            weight_mean_ref=("weight_ref", "mean"),
            weight_mean_cand=("weight_cand", "mean"),
        )
        .reset_index()
    )

    pa = ra.parameter_data.set_index("parnme")
    pb = rb.parameter_data.set_index("parnme")
    pcols = ["parval1", "parlbnd", "parubnd", "pargp", "partrans"]
    par_same = pa.index.equals(pb.index) and all(
        pa[c].astype(str).equals(pb[c].astype(str)) for c in pcols
    )
    summary["parameter_data_identical"] = bool(par_same)
    summary["n_par_ref"] = int(len(pa))
    summary["n_par_cand"] = int(len(pb))
    opt_a, opt_b = dict(ra.pestpp_options), dict(rb.pestpp_options)
    summary["pestpp_options_identical"] = opt_a == opt_b
    summary["pestpp_options_diff"] = {
        k: [opt_a.get(k), opt_b.get(k)]
        for k in set(opt_a) | set(opt_b)
        if opt_a.get(k) != opt_b.get(k)
    }
    summary["control_data"] = {
        "noptmax": [int(ra.control_data.noptmax), int(rb.control_data.noptmax)],
    }
    return summary, obs, by_group


def main():
    p = argparse.ArgumentParser(description="Diff two PEST++ problem definitions")
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, obs, by_group = compare(args.reference, args.candidate)
    obs.to_csv(out / "observation_diff.csv")
    by_group.to_csv(out / "observation_diff_by_group.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    identical = (
        summary["obsval_identical"]
        and summary["weight_identical"]
        and summary["n_only_ref"] == 0
        and summary["n_only_cand"] == 0
        and summary["parameter_data_identical"]
        and summary["pestpp_options_identical"]
    )
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"PROBLEM_DEFINITION_IDENTICAL: {'yes' if identical else 'no'}")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
