"""Dump a container's calibration group to a tidy CSV, one row per field.

Used to snapshot a production answer before a rehearsal overwrites it, and to
diff the rehearsal's result against that snapshot. Reads only — never opens a
container for write, so it is safe to run against a live production store.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


def dump_container(path):
    path = Path(path)
    g = zarr.open_group(str(path), mode="r")
    # uid comes back as a numpy StringDType, which will not cast straight to
    # a fixed-width str dtype; go through a Python list.
    uid = np.array([str(v) for v in g["geometry"]["uid"][:]], dtype=object)
    cal = g["calibration"]

    cols = {"label": path.stem, "fid": uid}
    meta = cal["metadata"]
    for key in meta:
        cols[f"meta_{key}"] = np.asarray(meta[key][:])
    for key in cal["parameters"]:
        cols[key] = np.asarray(cal["parameters"][key][:])
    unc = cal.get("uncertainty", None)
    if unc is not None:
        for key in unc:
            cols[f"sd_{key}"] = np.asarray(unc[key][:])

    n = len(uid)
    frame = pd.DataFrame({k: v for k, v in cols.items() if np.ndim(v) == 0 or len(v) == n})
    frame.attrs["n_batches_completed"] = cal.attrs.get("n_batches_completed")
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--containers", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = [dump_container(c) for c in args.containers]
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False)

    n_cal = int(out["meta_calibrated"].sum()) if "meta_calibrated" in out else -1
    print(
        f"wrote {args.out}: {len(out)} fields, {n_cal} calibrated, {out['label'].nunique()} containers"
    )
    for label, grp in out.groupby("label"):
        done = int(grp["meta_calibrated"].sum()) if "meta_calibrated" in grp else -1
        print(f"  {label}: {len(grp)} fields, {done} calibrated")


if __name__ == "__main__":
    main()
