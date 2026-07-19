"""Generate the RUN_POLICY Category 2 (input data audit) archive for E1.

E1 (Example 4) had no archiver of its own; this fills the Cat 2 gap flagged in
the julyphysics review. It documents the container's data completeness at
calibration time so missing-data failures can be diagnosed after the fact.

E1's single active calibration target is Landsat SSEBop NHM ETf at the full
footprint (``remote_sensing/etf/landsat/ssebop/no_mask``); ``is_active_target``
is therefore True for that one stream. Meteorology is GridMET.

Usage:
    uv run python examples/4_Flux_Network/archive_input_audit.py \
        --container /data/ssd1/swim/4_Flux_Network/data/4_Flux_Network_julyphysics.swim \
        --archive-root /data/ssd1/swim/4_Flux_Network/results/julyphysics/archive \
        [--eval-exclusions /data/ssd1/swim/4_Flux_Network/results/julyphysics/evaluation_sites_excluded.csv] \
        [--calib-excluded ...]

Corrected ETo (``eto_corr``) is gated alongside raw ``eto`` per the container-
validation policy (check both raw and corrected met paths). The canonical
julyphysics calibration passed no ``--exclude``, so ``--calib-excluded`` is
empty by default; MB_Pch is an EVALUATION-only exclusion (it was calibrated) and
is recorded in the evaluation ledger, not the calibration one.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

# E1 active calibration target: single Landsat SSEBop NHM stream, full footprint.
ETF_SENSOR = "landsat"
ETF_MODEL = "ssebop"
ETF_MASK = "no_mask"
# Meteorology variables that must not be all-NaN (RUN_POLICY minimum checks).
# eto_corr is gated too: the container-validation policy requires checking both
# the raw and corrected met paths, even though E1's forward model uses raw eto
# (refet_type = "eto").
MET_EVAL = ["eto", "eto_corr", "srad", "tmax", "tmin", "prcp"]
MET_REPORT = ["eto", "eto_corr", "srad", "tmax", "tmin", "prcp"]


def _strs(arr):
    return [s.decode() if isinstance(s, bytes) else str(s) for s in np.asarray(arr)]


def _nonnull_per_site(arr2d):
    """Count of non-null (time, site) values along the time axis, per site."""
    return np.sum(~np.isnan(np.asarray(arr2d)), axis=0)


def capture_input_audit(audit, container, eval_exclusions=None, calib_excluded=None):
    audit.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(container), mode="r")

    uid = _strs(root["geometry/uid"])
    lc_class = _strs(root["geometry/properties/lc_class"])
    state = _strs(root["geometry/properties/state"])
    days = np.asarray(root["time/daily"][:])
    n = len(uid)

    etf = root[f"remote_sensing/etf/{ETF_SENSOR}/{ETF_MODEL}/{ETF_MASK}"]
    etf_nn = _nonnull_per_site(etf)
    ndvi = np.asarray(root[f"derived/merged_ndvi/{ETF_MASK}"][:])
    ndvi_nn = _nonnull_per_site(ndvi)
    met = {v: _nonnull_per_site(root[f"meteorology/gridmet/{v}"]) for v in MET_REPORT}

    # etf_capture_counts.csv (per site, per sensor/model/mask, is_active_target)
    pd.DataFrame(
        [
            {
                "site": uid[i],
                "sensor": ETF_SENSOR,
                "model": ETF_MODEL,
                "mask_mode": ETF_MASK,
                "non_null": int(etf_nn[i]),
                "is_active_target": True,
            }
            for i in range(n)
        ]
    ).to_csv(audit / "etf_capture_counts.csv", index=False)

    # ndvi_coverage.csv (per site: non-null, first/last obs date, seasonal flag)
    nd_rows = []
    for i in range(n):
        obs = ~np.isnan(ndvi[:, i])
        dts = days[obs]
        months = sorted({int(str(np.datetime64(d, "M")).split("-")[1]) for d in dts})
        nd_rows.append(
            {
                "site": uid[i],
                "non_null": int(ndvi_nn[i]),
                "first_date": str(np.datetime64(dts.min(), "D")) if dts.size else "",
                "last_date": str(np.datetime64(dts.max(), "D")) if dts.size else "",
                "months_covered": len(months),
                "seasonal_ok": len(months) >= 6,
            }
        )
    pd.DataFrame(nd_rows).to_csv(audit / "ndvi_coverage.csv", index=False)

    # met_completeness.csv (per site non-null for the reported met vars)
    pd.DataFrame(
        [{"site": uid[i], **{v: int(met[v][i]) for v in MET_REPORT}} for i in range(n)]
    ).to_csv(audit / "met_completeness.csv", index=False)

    # per_field_audit.csv (compact one-row-per-field roll-up)
    pd.DataFrame(
        [
            {
                "site": uid[i],
                "state": state[i],
                "lc_class": lc_class[i],
                "etf_ssebop_nn": int(etf_nn[i]),
                "ndvi_nn": int(ndvi_nn[i]),
                "met_eto_nn": int(met["eto"][i]),
                "met_prcp_nn": int(met["prcp"][i]),
            }
            for i in range(n)
        ]
    ).to_csv(audit / "per_field_audit.csv", index=False)

    # calibration_sites_excluded.csv: sites actually withheld from the CALIBRATION
    # run. The canonical julyphysics run passed no --exclude, so this is empty
    # unless a site had zero usable ETf target or NDVI coverage. MB_Pch is an
    # EVALUATION-only exclusion (it was calibrated) and is recorded in the
    # evaluation ledger, not here.
    calib_excluded = list(calib_excluded or [])
    cal_rows = [{"site": s, "reason": "calibration_run_exclusion"} for s in calib_excluded]
    for i in range(n):
        if etf_nn[i] == 0:
            cal_rows.append({"site": uid[i], "reason": "no_active_target_etf"})
        elif ndvi_nn[i] == 0:
            cal_rows.append({"site": uid[i], "reason": "no_ndvi"})
    pd.DataFrame(cal_rows, columns=["site", "reason"]).to_csv(
        audit / "calibration_sites_excluded.csv", index=False
    )

    # evaluation ledgers: copy the evaluator's daily record and its monthly
    # sibling (the monthly path withholds more sites — <30 paired days and the
    # monthly metric floor) so both cohorts reconcile in the archive.
    if eval_exclusions and Path(eval_exclusions).exists():
        ep = Path(eval_exclusions)
        pd.read_csv(ep).to_csv(audit / "evaluation_sites_excluded.csv", index=False)
        monthly = ep.with_name("evaluation_sites_excluded_monthly.csv")
        if monthly.exists():
            pd.read_csv(monthly).to_csv(
                audit / "evaluation_sites_excluded_monthly.csv", index=False
            )

    # container_health.json: the gate and per-stream completeness summary
    etf_zero = int((etf_nn == 0).sum())
    ndvi_zero = int((ndvi_nn == 0).sum())
    met_bad = {v: int((met[v] == 0).sum()) for v in MET_EVAL}
    met_allnan_vars = int(sum(1 for v in MET_EVAL if met_bad[v] > 0))
    gate = "PASS" if (etf_zero == 0 and ndvi_zero == 0 and met_allnan_vars == 0) else "HALT"
    health = {
        "container": str(container),
        "n_fields": n,
        "n_days": int(len(days)),
        "active_target": f"{ETF_SENSOR}/{ETF_MODEL}/{ETF_MASK}",
        "etf_zero_sites": etf_zero,
        "etf_capture_median": int(np.median(etf_nn)),
        "ndvi_zero_sites": ndvi_zero,
        "ndvi_median": int(np.median(ndvi_nn)),
        "met_zero_sites_by_var": met_bad,
        "met_vars_with_allnan_site": met_allnan_vars,
        "gate": gate,
    }
    (audit / "container_health.json").write_text(json.dumps(health, indent=2))
    print(
        f"  Cat 2: gate={gate}  etf_zero={etf_zero}  ndvi_zero={ndvi_zero}  "
        f"met_allnan_vars={met_allnan_vars}  (etf_med={health['etf_capture_median']}, "
        f"ndvi_med={health['ndvi_median']})"
    )
    return gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--container", required=True)
    p.add_argument("--archive-root", required=True, help="run's archive/ directory")
    p.add_argument("--eval-exclusions", default=None)
    p.add_argument(
        "--calib-excluded",
        nargs="*",
        default=[],
        help="Sites actually withheld from the calibration run (calibrate.py "
        "--exclude). Default none — the canonical julyphysics run excluded nothing.",
    )
    args = p.parse_args()

    audit = Path(args.archive_root) / "2_input_audit"
    gate = capture_input_audit(audit, args.container, args.eval_exclusions, args.calib_excluded)
    print(f"Cat 2 input audit written -> {audit}  (gate={gate})")


if __name__ == "__main__":
    main()
