"""Verify a chain rehearsal reproduced the production calibration.

PEST++-IES uses a fixed random seed (358183147, the default, never overridden
in these runs), so a rehearsal that changes only the job layout should
reproduce production parameters to within floating-point noise. Anything larger
is a behavioural change in the chain rewrite, not sampling.

Checks, in order of what they would catch:
  1. field coverage   — same fields present, same number calibrated
  2. partitioning     — same field-to-batch assignment (batches are joint
                        inverse problems, so a different split is a different
                        experiment and makes 3 meaningless)
  3. parameter values — per-parameter relative difference against tolerance
"""

import argparse

import numpy as np
import pandas as pd
from nwi_container_params import dump_container

TOL = 1e-6  # relative; anything above this is not floating-point noise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, help="CSV from nwi_container_params.py")
    ap.add_argument("--containers", nargs="+", required=True, help="rehearsal containers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()

    base = pd.read_csv(args.baseline, dtype={"fid": str})
    reh = pd.concat([dump_container(c) for c in args.containers], ignore_index=True)
    reh["fid"] = reh["fid"].astype(str)

    merged = base.merge(
        reh, on=["label", "fid"], suffixes=("_prod", "_reh"), how="outer", indicator=True
    )
    ok = True

    print("=== 1. field coverage ===")
    only = merged["_merge"].value_counts()
    print(
        f"  both {int(only.get('both', 0))}  production-only {int(only.get('left_only', 0))}  rehearsal-only {int(only.get('right_only', 0))}"
    )
    if int(only.get("left_only", 0)) or int(only.get("right_only", 0)):
        print("  FAIL: field sets differ")
        ok = False
    for side in ("prod", "reh"):
        col = f"meta_calibrated_{side}"
        if col in merged:
            print(f"  calibrated ({side}): {int(merged[col].fillna(0).sum())}")
    if "meta_calibrated_prod" in merged and "meta_calibrated_reh" in merged:
        mism = int(
            (
                merged["meta_calibrated_prod"].fillna(-1)
                != merged["meta_calibrated_reh"].fillna(-1)
            ).sum()
        )
        print(f"  calibrated-flag mismatches: {mism}")
        if mism:
            ok = False

    print("\n=== 2. partitioning ===")
    if "meta_batch_id_prod" in merged and "meta_batch_id_reh" in merged:
        diff = merged["meta_batch_id_prod"].fillna(-1) != merged["meta_batch_id_reh"].fillna(-1)
        print(f"  fields assigned to a different batch: {int(diff.sum())} / {len(merged)}")
        if int(diff.sum()):
            print("  NOTE: partitioning differs, so parameter differences below are expected")
            print(
                merged.loc[diff, ["label", "fid", "meta_batch_id_prod", "meta_batch_id_reh"]]
                .head(10)
                .to_string(index=False)
            )
            ok = False
    else:
        print("  batch_id absent from one side; skipped")

    print("\n=== 3. parameter values ===")
    params = sorted(
        {c[:-5] for c in merged.columns if c.endswith("_prod") and not c.startswith("meta_")}
    )
    rows = []
    for p in params:
        a = pd.to_numeric(merged[f"{p}_prod"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(merged[f"{p}_reh"], errors="coerce").to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        denom = np.where(np.abs(a[both]) > 0, np.abs(a[both]), 1.0)
        rel = np.abs(b[both] - a[both]) / denom
        rows.append(
            {
                "param": p,
                "n": int(both.sum()),
                "max_rel": float(rel.max()) if rel.size else np.nan,
                "median_rel": float(np.median(rel)) if rel.size else np.nan,
                "n_over_tol": int((rel > args.tol).sum()),
            }
        )
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False, float_format=lambda v: f"{v:.3e}"))
    if int(rep["n_over_tol"].sum()):
        print(f"\n  {int(rep['n_over_tol'].sum())} value(s) exceed rel tol {args.tol:g}")
        ok = False

    merged.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(
        "\nVERDICT:",
        "PASS — rehearsal reproduced production" if ok else "DIFFERENCES FOUND — see above",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
