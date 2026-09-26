"""Read-only array-by-array diff of two Example 5 containers.

Used as the two Run 23 gates:

* null rebuild: a container rebuilt from base with the *original* extract
  tables must equal the reference run's container on every array (builder
  and calculator drift check);
* corrected rebuild: only the ET-denominated members, the ensemble and the
  ETf-dependent dynamics may differ, and no site may change irrigation class
  or groundwater-subsidy status.

Numeric arrays are compared with NaN-equal semantics. The dynamics JSON
arrays (``irr_data``, ``gwsub_data``) are compared per site and per year.

Usage:
    uv run python diff_run_containers.py --reference <run22.swim> \\
        --candidate <run23.swim> --out-dir <results/run23/container_diff>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

FSUB_THRESHOLD = 0.2  # process/loop_fast.py groundwater-subsidy gate


def _md_table(df):
    """Plain markdown table without the optional tabulate dependency."""
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else f"{v:.6g}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _arrays(root):
    out = {}

    def walk(g, prefix=""):
        for k, v in g.members():
            p = f"{prefix}/{k}" if prefix else k
            if isinstance(v, zarr.Group):
                walk(v, p)
            else:
                out[p] = v

    walk(root)
    return out


def _compare_numeric(a, b):
    a = np.asarray(a[:], dtype=np.float64)
    b = np.asarray(b[:], dtype=np.float64)
    if a.shape != b.shape:
        return {"status": "shape", "shape_a": a.shape, "shape_b": b.shape}
    fa, fb = np.isfinite(a), np.isfinite(b)
    same_nan = fa == fb
    both = fa & fb
    diff = np.zeros(a.shape, dtype=bool)
    diff[both] = a[both] != b[both]
    diff |= ~same_nan
    n_diff = int(diff.sum())
    rec = {
        "status": "identical" if n_diff == 0 else "changed",
        "n_diff": n_diff,
        "finite_a": int(fa.sum()),
        "finite_b": int(fb.sum()),
        "only_a": int((fa & ~fb).sum()),
        "only_b": int((fb & ~fa).sum()),
    }
    if n_diff:
        d = np.abs(a[both] - b[both])
        rec["max_abs_diff"] = float(d.max()) if d.size else 0.0
        nz = both & (a != 0)
        if nz.any():
            r = b[nz] / a[nz]
            r = r[r != 1.0]
            if r.size:
                rec["ratio_median"] = float(np.median(r))
                rec["ratio_mean"] = float(r.mean())
                rec["ratio_sd"] = float(r.std(ddof=1)) if r.size > 1 else 0.0
                rec["ratio_p01"] = float(np.percentile(r, 1))
                rec["ratio_p99"] = float(np.percentile(r, 99))
                rec["n_ratio"] = int(r.size)
    return rec


def _compare_generic(a, b):
    a = a[:]
    b = b[:]
    if a.shape != b.shape:
        return {"status": "shape", "shape_a": a.shape, "shape_b": b.shape}
    diff = np.array([x != y for x, y in zip(np.ravel(a), np.ravel(b))])
    n = int(diff.sum())
    return {"status": "identical" if n == 0 else "changed", "n_diff": n}


def _json_col(arr, uids):
    out = {}
    for uid, s in zip(uids, arr[:]):
        s = s.item() if hasattr(s, "item") else s
        out[uid] = json.loads(s) if s else {}
    return out


def _irr_years(d):
    """Years flagged irrigated (per-year dicts only; ``fallow_years`` is a list)."""
    return {yr for yr, rec in d.items() if isinstance(rec, dict) and rec.get("irrigated")}


def _site_gate(gw, irr_years):
    """Site gw_status and the set of years where subsidy is actually active."""
    yearly = {
        int(yr): float(rec.get("f_sub", 0.0))
        for yr, rec in gw.items()
        if isinstance(rec, dict) and yr.isdigit()
    }
    non_irr = [v for yr, v in yearly.items() if str(yr) not in irr_years]
    site = bool(non_irr) and float(np.mean(non_irr)) > FSUB_THRESHOLD
    active = (
        {yr for yr, v in yearly.items() if str(yr) not in irr_years and v > FSUB_THRESHOLD}
        if site
        else set()
    )
    return site, active


def diff_dynamics(ra, rb, uids):
    rows = []
    irr_a = _json_col(ra["derived/dynamics/irr_data"], uids)
    irr_b = _json_col(rb["derived/dynamics/irr_data"], uids)
    gw_a = _json_col(ra["derived/dynamics/gwsub_data"], uids)
    gw_b = _json_col(rb["derived/dynamics/gwsub_data"], uids)
    kc_a, kc_b = ra["derived/dynamics/kc_max"][:], rb["derived/dynamics/kc_max"][:]
    ke_a, ke_b = ra["derived/dynamics/ke_max"][:], rb["derived/dynamics/ke_max"][:]

    irr_flips, gw_flips, site_gate_flips, active_flips = [], [], [], []
    for i, uid in enumerate(uids):
        ya, yb = _irr_years(irr_a[uid]), _irr_years(irr_b[uid])
        irr_same_json = irr_a[uid] == irr_b[uid]
        if ya != yb:
            irr_flips.append((uid, sorted(ya ^ yb)))
        # Effective model gate (process/input.py): site gw_status is the
        # non-irrigated-year mean f_sub > 0.2; subsidy then applies only in
        # non-irrigated years whose own f_sub > 0.2.
        site_a, active_a = _site_gate(gw_a[uid], ya)
        site_b, active_b = _site_gate(gw_b[uid], yb)
        if site_a != site_b:
            site_gate_flips.append((uid, site_a, site_b))
        if active_a != active_b:
            active_flips.append((uid, sorted(active_a ^ active_b)))
        years = sorted(set(gw_a[uid]) | set(gw_b[uid]))
        gw_flip_years = []
        for yr in years:
            a = gw_a[uid].get(yr, {})
            b = gw_b[uid].get(yr, {})
            sa, sb = int(a.get("subsidized", 0)), int(b.get("subsidized", 0))
            fa, fb = float(a.get("f_sub", 0.0)), float(b.get("f_sub", 0.0))
            ga, gb = (sa == 1 and fa > FSUB_THRESHOLD), (sb == 1 and fb > FSUB_THRESHOLD)
            if sa != sb or ga != gb:
                gw_flip_years.append(yr)
            rows.append(
                {
                    "site": uid,
                    "year": yr,
                    "subsidized_ref": sa,
                    "subsidized_cand": sb,
                    "f_sub_ref": fa,
                    "f_sub_cand": fb,
                    "gate_ref": ga,
                    "gate_cand": gb,
                    "irr_year_ref": yr in ya,
                    "irr_year_cand": yr in yb,
                }
            )
        if gw_flip_years:
            gw_flips.append((uid, gw_flip_years))
        rows.append(
            {
                "site": uid,
                "year": "site",
                "kc_max_ref": float(kc_a[i]),
                "kc_max_cand": float(kc_b[i]),
                "ke_max_ref": float(ke_a[i]),
                "ke_max_cand": float(ke_b[i]),
                "irr_json_identical": irr_same_json,
                "n_irr_years_ref": len(ya),
                "n_irr_years_cand": len(yb),
                "gw_status_ref": site_a,
                "gw_status_cand": site_b,
                "gw_active_years_ref": len(active_a),
                "gw_active_years_cand": len(active_b),
            }
        )
    gates = {
        "irr_flips": irr_flips,
        "gw_year_flips": gw_flips,
        "gw_site_flips": site_gate_flips,
        "gw_active_flips": active_flips,
    }
    return pd.DataFrame(rows), gates


def main():
    p = argparse.ArgumentParser(description="Diff two Example 5 containers")
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ra = zarr.open(args.reference, mode="r")
    rb = zarr.open(args.candidate, mode="r")
    aa, ab = _arrays(ra), _arrays(rb)

    records = []
    for path in sorted(set(aa) | set(ab)):
        if path not in aa or path not in ab:
            records.append(
                {"array": path, "status": "only_reference" if path in aa else "only_candidate"}
            )
            continue
        a, b = aa[path], ab[path]
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            rec = _compare_numeric(a, b)
        else:
            rec = _compare_generic(a, b)
        records.append({"array": path, **rec})
    arrays_df = pd.DataFrame(records)
    arrays_df.to_csv(out / "arrays.csv", index=False)

    uids = [u.item() if hasattr(u, "item") else u for u in ra["geometry/uid"][:]]
    dyn_df, gates = diff_dynamics(ra, rb, uids)
    irr_flips = gates["irr_flips"]
    gw_flips = gates["gw_year_flips"]
    dyn_df.to_csv(out / "dynamics.csv", index=False)

    changed = arrays_df[arrays_df["status"] != "identical"]
    site = dyn_df[dyn_df["year"] == "site"]
    kc_moved = site[site["kc_max_ref"] != site["kc_max_cand"]]
    ke_moved = site[site["ke_max_ref"] != site["ke_max_cand"]]
    irr_json_diff = site[~site["irr_json_identical"].astype(bool)]

    lines = [
        "# Container diff",
        "",
        f"reference: `{args.reference}`",
        f"candidate: `{args.candidate}`",
        "",
        f"arrays compared: {len(arrays_df)}; changed: {len(changed)}",
        "",
        "## Changed arrays",
        "",
    ]
    if changed.empty:
        lines.append("none")
    else:
        lines.append(_md_table(changed))
    lines += [
        "",
        "## Dynamics",
        "",
        f"irr_data JSON differs at {len(irr_json_diff)} sites: {list(irr_json_diff['site'])}",
        f"irrigated-year set flips: {len(irr_flips)} sites: {irr_flips}",
        f"gwsub per-year subsidized/f_sub>0.2 flips: {len(gw_flips)} sites: {gw_flips}",
        f"site gw_status flips (non-irrigated-year mean f_sub > 0.2): "
        f"{len(gates['gw_site_flips'])}: {gates['gw_site_flips']}",
        f"active subsidy site-year set changes: {len(gates['gw_active_flips'])}: "
        f"{gates['gw_active_flips']}",
        f"kc_max moved at {len(kc_moved)} sites; ke_max moved at {len(ke_moved)} sites",
        "",
    ]
    if not ke_moved.empty:
        lines.append(
            _md_table(ke_moved[["site", "ke_max_ref", "ke_max_cand", "kc_max_ref", "kc_max_cand"]])
        )
    lines += [
        "",
        "## Gates",
        "",
        f"IRR_GATE: {'PASS' if not irr_flips and irr_json_diff.empty else 'FAIL'}",
        f"GWSUB_SITE_GATE: {'PASS' if not gates['gw_site_flips'] else 'FAIL'}",
        f"GWSUB_ACTIVE_YEARS_GATE: {'PASS' if not gates['gw_active_flips'] else 'FAIL'}",
        f"GWSUB_PER_YEAR_FLAGS: {'unchanged' if not gw_flips else 'changed'}",
        f"ALL_IDENTICAL: {'yes' if changed.empty else 'no'}",
    ]
    (out / "container_diff.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-6:]))
    print(f"changed arrays: {list(changed['array'])}")
    print(f"report: {out / 'container_diff.md'}")


if __name__ == "__main__":
    main()
