"""Gate G1 (HWSD AWC units recal): the refreshed GrassBasis container vs its superseded copy.

Asserts, on the refreshed container:

* ``properties/soils/awc`` is stored in m/m: every value finite, in (0, 1], 66/66 non-null,
  and equal to the superseded container's mm/m value / 1000 (same HWSD CSV, declared units);
* the ``properties/soils`` attrs record ``awc_units_source = "mm/m"`` and
  ``awc_units_stored = "m/m"``;
* no calibration state (``calibration/`` group, calibration simulation runs, simulate events);
* every array outside ``properties/soils`` is content-identical to the superseded copy —
  in particular ``derived/dynamics/irr_data`` (soils do not feed the classifier; assert, do not
  assume), ``derived/dynamics/{gwsub_data,ke_max,kc_max}``, the merged NDVI, ETf, meteorology
  and snow arrays;
* the phase8 classifier-transition CSV written for the refreshed container is identical to the
  canonical copy (``--transition-new`` vs ``--transition-canon``).

Exit 1 on any failure; the report is written as JSON to ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "container_build" / "e2_refooting"))
from phase8_container_health import check_no_calibration_state  # noqa: E402

E2 = Path("/data/ssd1/swim/6_Flux_International")
DEFAULT_NEW = E2 / "data" / "6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim"
DEFAULT_OLD = (
    E2
    / "results"
    / "superseded_awc320_20260921"
    / "containers"
    / "6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim"
)
DEFAULT_TRANSITION_CANON = E2 / "data" / "e2_etf_refooting" / "irrigation_classifier_transition.csv"
DEFAULT_TRANSITION_NEW = (
    E2 / "data" / "awc_recal" / "qa_canon" / "irrigation_classifier_transition.csv"
)
DEFAULT_OUT = E2 / "data" / "awc_recal" / "gate_g1_container.json"


def array_paths(root: zarr.Group) -> dict[str, zarr.Array]:
    return {
        name: node for name, node in root.members(max_depth=None) if isinstance(node, zarr.Array)
    }


def content_hash(x: np.ndarray) -> str:
    if x.dtype.kind in ("U", "S", "O", "T"):
        data = "\x1f".join(str(v) for v in x.ravel().tolist()).encode()
    else:
        data = np.ascontiguousarray(x).tobytes()
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", default=str(DEFAULT_NEW))
    ap.add_argument("--old", default=str(DEFAULT_OLD))
    ap.add_argument("--transition-new", default=str(DEFAULT_TRANSITION_NEW))
    ap.add_argument("--transition-canon", default=str(DEFAULT_TRANSITION_CANON))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--allow-calibration-state",
        action="store_true",
        help="Do not fail on calibration state (use only for a pre-clean inspection)",
    )
    args = ap.parse_args()

    new = zarr.open_group(args.new, mode="r")
    old = zarr.open_group(args.old, mode="r")
    problems: list[str] = []
    report: dict = {"new": args.new, "old": args.old}

    # --- AWC units ---
    awc_new = np.asarray(new["properties/soils/awc"][:], float)
    awc_old = np.asarray(old["properties/soils/awc"][:], float)
    n = int(awc_new.size)
    nn = int(np.isfinite(awc_new).sum())
    report["awc"] = {
        "n": n,
        "non_null": nn,
        "min": float(np.nanmin(awc_new)),
        "max": float(np.nanmax(awc_new)),
        "old_min_mm_per_m": float(np.nanmin(awc_old)),
        "old_max_mm_per_m": float(np.nanmax(awc_old)),
        "n_below_100_mm_per_m": int(np.sum(awc_new * 1000.0 < 100.0)),
    }
    if nn != n or n != 66:
        problems.append(f"awc non-null {nn}/{n} (expected 66/66)")
    if not (np.all(awc_new > 0.0) and np.all(awc_new <= 1.0)):
        problems.append(f"awc outside (0, 1]: min {awc_new.min()} max {awc_new.max()}")
    if not np.allclose(awc_new * 1000.0, awc_old, rtol=0, atol=1e-3):
        problems.append(
            "awc_new * 1000 != awc_old (HWSD mm/m): refreshed values do not match source"
        )
    attrs = dict(new["properties/soils"].attrs)
    report["soils_attrs"] = attrs
    if attrs.get("awc_units_source") != "mm/m" or attrs.get("awc_units_stored") != "m/m":
        problems.append(f"properties/soils attrs do not record mm/m -> m/m: {attrs}")

    # --- calibration state ---
    state = check_no_calibration_state(new)
    report["no_calibration_state"] = state
    if not state["pass"] and not args.allow_calibration_state:
        problems.append(f"calibration state present: {state['problems']}")

    # --- every other array identical ---
    arrs_new = array_paths(new)
    arrs_old = array_paths(old)
    only_new = sorted(set(arrs_new) - set(arrs_old))
    only_old = sorted(k for k in set(arrs_old) - set(arrs_new) if not k.startswith("calibration/"))
    only_old = [k for k in only_old if not k.startswith("simulation/")]
    diffs = []
    checked = 0
    for name in sorted(set(arrs_new) & set(arrs_old)):
        if name == "properties/soils/awc":
            continue
        if name.startswith(("calibration/", "simulation/")):
            continue
        a = np.asarray(arrs_new[name][:])
        b = np.asarray(arrs_old[name][:])
        checked += 1
        if a.shape != b.shape or content_hash(a) != content_hash(b):
            diffs.append(name)
    report["arrays"] = {
        "checked_identical": checked,
        "differ": diffs,
        "only_in_new": only_new,
        "only_in_old_non_calibration": only_old,
    }
    if diffs:
        problems.append(f"arrays differ from superseded copy: {diffs}")
    if only_new or only_old:
        problems.append(f"array set differs: only_new={only_new} only_old={only_old}")
    for key in ("derived/dynamics/irr_data", "derived/dynamics/gwsub_data"):
        if key not in arrs_new:
            problems.append(f"missing {key}")

    # --- classifier transition CSV ---
    tn, tc = Path(args.transition_new), Path(args.transition_canon)
    if tn.exists() and tc.exists():
        dn = pd.read_csv(tn)
        dc = pd.read_csv(tc)
        same = dn.shape == dc.shape and list(dn.columns) == list(dc.columns) and dn.equals(dc)
        report["transition_csv_identical"] = bool(same)
        if not same:
            problems.append("irrigation_classifier_transition.csv differs from the canonical copy")
    else:
        problems.append(f"transition CSV missing: new={tn.exists()} canon={tc.exists()}")

    report["problems"] = problems
    report["pass"] = not problems
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print(f"G1 {'PASS' if not problems else 'FAIL'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
