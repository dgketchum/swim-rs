"""Post-calibration review for one NWI unit: parameter gate + output gate.

Generalized from nwi/review_32009_output.py (the Esmeralda pilot review that
passed 2026-08-22/24) so any partition can be reviewed from its TOML.

Parameter gate (reads {pest_run_dir}/posterior/calibration.json, written by
nwi_posterior_export.py): coverage of calibrated fields and the rail /
high-uncertainty flag census.

Output gate (reads the persisted run in the container):
  1. capture-day ETf fit vs the >=3-member Landsat ensemble mean, with the
     mask resolved per field-year from derived/dynamics/irr_data to match
     mask_mode="irrigation"
  2. annual / growing-season water balance by irrigation status, incl. closure
  3. irrigated-year detail: applied depth, irrigation event structure
     (events/yr, depth, inter-event gap) against posterior mad
  4. E/T partition and Ks engagement

Transpiration: when the run carries the `evap` output (added after the pilot),
T = eta - evap is reported directly. The pilot's cover-scaling identity
T = min(fc_t*Ks*Kcb*ETo, eta) is always reported too, so numbers stay
comparable with the 32009 review.

Usage:
    uv run python scripts/nwi_review_output.py --config /path/to/32019a.toml
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from swimrs.container import SwimContainer
from swimrs.swim.config import ProjectConfig

MEMBERS = ("ssebop", "sims", "geesebal", "eemetric", "ptjpl", "disalexi")
KCB_BASE = 0.15
MIN_MEMBERS = 3  # capture day requires this many members
GS_MONTHS = (4, 10)


def dist(s, prec=1):
    """Distribution one-liner. Use prec=3 for dimensionless quantities
    (ETf bias/rmse/r, T/ET, mad) where 1 decimal hides the signal."""
    s = pd.Series(s).dropna()
    if s.empty:
        return "n=0"
    w = 7 if prec <= 1 else 6 + prec
    return (
        f"n={len(s)} mean={s.mean():{w}.{prec}f} p10={s.quantile(0.1):{w}.{prec}f} "
        f"med={s.median():{w}.{prec}f} p90={s.quantile(0.9):{w}.{prec}f}"
    )


def irrigation_events(irr_daily_mm, gap_days=2):
    """Cluster consecutive irrigation days into events.

    Returns (n_events, mean_depth_mm, median_gap_days). Days closer together
    than gap_days belong to the same event, so a multi-day application is one
    event rather than several.
    """
    idx = np.nonzero(irr_daily_mm > 0)[0]
    if idx.size == 0:
        return 0, np.nan, np.nan
    splits = np.nonzero(np.diff(idx) >= gap_days)[0] + 1
    groups = np.split(idx, splits)
    depths = [float(irr_daily_mm[g].sum()) for g in groups]
    starts = np.array([g[0] for g in groups])
    gaps = np.diff(starts) if starts.size > 1 else np.array([np.nan])
    return len(groups), float(np.mean(depths)), float(np.nanmedian(gaps))


def parameter_gate(pest_run_dir, lines):
    path = os.path.join(pest_run_dir, "posterior", "calibration.json")
    lines.append("== 0. parameter gate ==")
    if not os.path.exists(path):
        lines.append(f"  SKIP: no posterior export at {path}")
        return
    d = json.load(open(path))
    n, ncal = d.get("n_fields"), d.get("n_calibrated")
    lines.append(f"  fields={n} calibrated={ncal} batches={d.get('n_batches')}")
    lines.append(f"  flagged fields: {d.get('n_flagged_fields')} ({d.get('n_flags')} flags)")
    for k, v in sorted(d.get("flags_by_type", {}).items(), key=lambda kv: -kv[1]):
        pct = 100.0 * v / n if n else float("nan")
        lines.append(f"    {k:32s} {v:5d}  ({pct:5.1f}% of fields)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Unit TOML")
    ap.add_argument("--run-id", default="posterior_full")
    ap.add_argument("--out-dir", default=None, help="Default {project_workspace}/review")
    ap.add_argument("--spinup-years", type=int, default=2, help="Years skipped after start_date")
    ap.add_argument("--kc-max", type=float, default=1.35, help="Ceiling for the fc_t identity")
    args = ap.parse_args()

    cfg = ProjectConfig()
    cfg.read_config(args.config)
    label = (
        str(cfg.project_name)
        if hasattr(cfg, "project_name")
        else os.path.basename(args.config).split(".")[0]
    )
    workspace = (
        cfg.project_workspace if hasattr(cfg, "project_workspace") else os.path.dirname(args.config)
    )
    out_dir = args.out_dir or os.path.join(workspace, "review")
    os.makedirs(out_dir, exist_ok=True)

    c = SwimContainer.open(cfg.container_path, mode="r")
    root = c._root
    pf = root[f"simulation/runs/{args.run_id}"]
    uid = pf["fields/uid"][:]
    time = pd.DatetimeIndex(pf["time/daily"][:])
    years, months = time.year.values, time.month.values
    n_days, n_fields = len(time), len(uid)

    out = {k: pf["outputs"][k][:] for k in pf["outputs"].array_keys()}
    eto = root["meteorology/gridmet/eto_corr"][:]
    assert eto.shape == (n_days, n_fields), f"eto {eto.shape} != run {(n_days, n_fields)}"

    yr_start = int(years.min()) + args.spinup_years
    yr_end = int(years.max())
    yr_list = list(range(yr_start, yr_end + 1))

    # irrigation status per field-year, as the model resolved it
    irr_json = [json.loads(s) for s in root["derived/dynamics/irr_data"][:]]
    irrigated = np.zeros((len(yr_list), n_fields), dtype=bool)
    f_irr = np.full((len(yr_list), n_fields), np.nan)
    for j, d in enumerate(irr_json):
        if not isinstance(d, dict):
            continue
        for i, y in enumerate(yr_list):
            rec = d.get(str(y), {})
            if not isinstance(rec, dict):
                continue
            irrigated[i, j] = bool(rec.get("irrigated", 0))
            f_irr[i, j] = rec.get("f_irr", np.nan)

    yr_idx = np.searchsorted(yr_list, years)
    in_por = (years >= yr_start) & (years <= yr_end)
    irr_daily = np.zeros((n_days, n_fields), dtype=bool)
    irr_daily[in_por] = irrigated[yr_idx[in_por]]

    # --- 1. capture-day ETf fit -------------------------------------------
    obs_stack = np.full((len(MEMBERS), n_days, n_fields), np.nan, dtype=np.float32)
    for m, name in enumerate(MEMBERS):
        a_irr = root[f"remote_sensing/etf/landsat/{name}/irr"][:]
        a_inv = root[f"remote_sensing/etf/landsat/{name}/inv_irr"][:]
        obs_stack[m] = np.where(irr_daily, a_irr, a_inv)
    n_members = np.isfinite(obs_stack).sum(axis=0)
    with np.errstate(invalid="ignore"):
        obs_mean = np.nanmean(obs_stack, axis=0)
    obs_mean[n_members < MIN_MEMBERS] = np.nan

    sim_etf = out["etf"]
    rows = []
    for j in range(n_fields):
        mask = np.isfinite(obs_mean[:, j]) & in_por
        o, s = obs_mean[mask, j], sim_etf[mask, j]
        if len(o) < 10:
            rows.append((uid[j], len(o), np.nan, np.nan, np.nan))
            continue
        rows.append(
            (
                uid[j],
                len(o),
                float(np.mean(s - o)),
                float(np.sqrt(np.mean((s - o) ** 2))),
                float(np.corrcoef(o, s)[0, 1]),
            )
        )
    etf_fit = pd.DataFrame(rows, columns=["uid", "n_obs", "bias", "rmse", "r"])
    etf_fit["ever_irr"] = irrigated.any(axis=0)
    etf_fit.to_csv(f"{out_dir}/etf_capture_fit.csv", index=False)

    # --- 2/3/4. water balance, partition, events --------------------------
    gs = (months >= GS_MONTHS[0]) & (months <= GS_MONTHS[1])
    comp = ["eta", "irr_sim", "gw_sim", "rain", "melt", "dperc", "dperc_irr", "runoff", "et_irr"]
    has_evap = "evap" in out
    recs = []
    for i, y in enumerate(yr_list):
        ymask = years == y
        gmask = ymask & gs
        yi = np.where(ymask)[0]
        dS = -(out["depl_root"][yi[-1]] - out["depl_root"][yi[0]])
        sums = {k: out[k][ymask].sum(axis=0) for k in comp}
        gs_eta = out["eta"][gmask].sum(axis=0)
        kcb, ks = out["kcb"][gmask], out["ks"][gmask]
        fc_t = np.clip((kcb - KCB_BASE) / (args.kc_max - KCB_BASE), 0.0, 1.0)
        gs_t_id = np.minimum(fc_t * ks * kcb * eto[gmask], out["eta"][gmask]).sum(axis=0)
        gs_t_dir = (out["eta"][gmask] - out["evap"][gmask]).sum(axis=0) if has_evap else None
        ks_engaged = (ks < 0.9).mean(axis=0)
        for j in range(n_fields):
            rec = {
                "uid": uid[j],
                "year": y,
                "irrigated": irrigated[i, j],
                "f_irr": f_irr[i, j],
                **{k: float(sums[k][j]) for k in comp},
                "dS_root": float(dS[j]),
                "gs_eta": float(gs_eta[j]),
                "gs_t_identity": float(gs_t_id[j]),
                "gs_t_et_identity": float(gs_t_id[j] / gs_eta[j]) if gs_eta[j] > 0 else np.nan,
                "ks_lt_0p9_gs_frac": float(ks_engaged[j]),
                "irr_days": int((out["irr_sim"][ymask, j] > 0).sum()),
            }
            if has_evap:
                rec["gs_t_direct"] = float(gs_t_dir[j])
                rec["gs_t_et_direct"] = float(gs_t_dir[j] / gs_eta[j]) if gs_eta[j] > 0 else np.nan
            n_ev, depth, gap = irrigation_events(out["irr_sim"][ymask, j])
            rec.update(n_events=n_ev, event_depth_mm=depth, event_gap_days=gap)
            recs.append(rec)
    fy = pd.DataFrame(recs)
    fy["input"] = fy["rain"] + fy["melt"] + fy["irr_sim"] + fy["gw_sim"]
    fy["output"] = fy["eta"] + fy["dperc"] + fy["runoff"]
    fy["closure_resid"] = fy["input"] - fy["output"] - fy["dS_root"]
    fy.to_csv(f"{out_dir}/field_year_water_balance.csv", index=False)

    # per-field event structure vs posterior mad/aw
    ev = (
        fy[fy.irrigated]
        .groupby("uid")
        .agg(
            ev_per_yr=("n_events", "mean"),
            depth_med=("event_depth_mm", "median"),
            gap_med=("event_gap_days", "median"),
            irr_med=("irr_sim", "median"),
        )
    )
    if "calibration" in root:
        par = root["calibration/parameters"]
        cal_uid = root["fields/uid"][:] if "fields/uid" in root else uid
        for p in ("mad", "aw"):
            if p in par:
                ev[p] = pd.Series(np.asarray(par[p][:]), index=cal_uid).reindex(ev.index)
    ev.to_csv(f"{out_dir}/irrigation_events_by_field.csv")
    c.close()

    # --- summary -----------------------------------------------------------
    lines = [
        f"NWI {label} review — run {args.run_id}, years {yr_start}-{yr_end}",
        f"fields: {n_fields}; ever-irrigated: {int(irrigated.any(axis=0).sum())}; "
        f"irrigated field-years: {int(fy.irrigated.sum())} / {len(fy)}",
        "",
    ]
    parameter_gate(cfg.pest_run_dir if hasattr(cfg, "pest_run_dir") else "", lines)
    lines.append("")
    lines.append(f"== 1. capture-day ETf fit (sim vs >={MIN_MEMBERS}-member mean) ==")
    for lab, sub in [
        ("irrigated fields", etf_fit[etf_fit.ever_irr]),
        ("never-irrigated", etf_fit[~etf_fit.ever_irr]),
    ]:
        lines.append(f"  {lab}: n_fields={len(sub)}  n_obs med={sub.n_obs.median():.0f}")
        for k in ("bias", "rmse", "r"):
            lines.append(f"    {k:5s}: {dist(sub[k], prec=3)}")
    lines.append("")
    lines.append("== 2. annual water balance (mm/yr, field-year distributions) ==")
    for lab, sub in [
        ("IRRIGATED field-years", fy[fy.irrigated]),
        ("NON-IRRIGATED field-years", fy[~fy.irrigated]),
    ]:
        lines.append(f"  {lab}:")
        for k in [
            "eta",
            "gs_eta",
            "irr_sim",
            "gw_sim",
            "rain",
            "melt",
            "dperc",
            "runoff",
            "et_irr",
            "closure_resid",
        ]:
            lines.append(f"    {k:13s}: {dist(sub[k])}")
    lines.append("")
    lines.append("== 3. irrigated-year detail ==")
    sub = fy[fy.irrigated]
    lines.append(f"  irr days/yr    : {dist(sub.irr_days.astype(float))}")
    lines.append(f"  events/yr      : {dist(sub.n_events.astype(float))}")
    lines.append(f"  depth/event mm : {dist(sub.event_depth_mm)}")
    lines.append(f"  gap days       : {dist(sub.event_gap_days)}")
    if len(ev) and "mad" in ev.columns and ev["mad"].notna().any():
        good = ev.dropna(subset=["mad", "depth_med"])
        if len(good) > 2:
            lines.append(
                f"  corr(mad, depth/event) = {np.corrcoef(good['mad'], good['depth_med'])[0, 1]:.2f}"
            )
        lines.append(f"  posterior mad  : {dist(ev['mad'], prec=3)}")
    lines.append("")
    lines.append("== 4. E/T partition and Ks ==")
    for lab, sub in [("IRRIGATED", fy[fy.irrigated]), ("NON-IRRIGATED", fy[~fy.irrigated])]:
        lines.append(f"  {lab}:")
        lines.append(f"    GS T/ET (identity): {dist(sub.gs_t_et_identity, prec=3)}")
        if has_evap:
            lines.append(f"    GS T/ET (direct)  : {dist(sub.gs_t_et_direct, prec=3)}")
        lines.append(f"    GS frac days Ks<0.9: {dist(sub.ks_lt_0p9_gs_frac, prec=3)}")
    if not has_evap:
        lines.append("  NOTE: run predates the `evap` output; identity-based T only.")
    lines.append("")
    lines.append("== 5. sanity ==")
    nonirr = fy[~fy.irrigated]
    lines.append(
        f"  irr_sim>1mm in non-irrigated years: {int((nonirr.irr_sim > 1.0).sum())} field-years"
    )
    lines.append(
        f"  |closure| > 5 mm/yr: {int((fy.closure_resid.abs() > 5).sum())} / {len(fy)} field-years"
    )

    txt = "\n".join(lines)
    with open(f"{out_dir}/review_summary.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
