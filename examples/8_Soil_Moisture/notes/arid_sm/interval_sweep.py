"""Forward interval-sweep experiment at the arid, IrrMapper-confirmed irrigated SCAN
sites (Circleville, Morgan, Manderfield, UT).

Hypothesis (from diagnose_arid.py): the model decorrelates from in-situ theta at these
sites NOT because it lacks amplitude (std_ratio is healthy 0.49-0.65) and NOT because of
a constant lag (phase_gain ~0), but because the MAD scheduler with a small calibrated
`mad` crosses its trigger `raw = mad*taw` every few days and floods in frequent small
top-ups that pin theta near field capacity. Real flood-rotation irrigation is infrequent
and large: depletion accumulates for 1-3 weeks, then a flood refills to FC -> a sawtooth.

Lever: `min_irr_days` (WP-C1 return-interval gate) blocks re-triggering within N days of
the last event. Sweeping it over the FROZEN e8split posterior (no recalibration) tests
whether imposing a realistic return interval recovers the theta sawtooth. The Pareto cost
is measured against the RS ETf ensemble (the calibration target) on capture dates, plus
growing-season mean Ks and seasonal ET total: the cost appears only if the longer interval
drives depletion past the stress threshold (p_stress*taw, fixed 0.5) and Ks drops.

Forward-runs only. Reads validation theta strictly for scoring; ETf obs is the model's own
calibration target (RS, not validation). No EE, no calibration.
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
E8 = HERE.parent.parent
E5 = E8.parent / "5_Flux_Ensemble"
if str(E5) not in sys.path:
    sys.path.insert(0, str(E5))


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev8 = _load_module(str(E8 / "evaluate.py"), "ex8_evaluate")
_load_config = ev8._load_config
theta_available = ev8.theta_available

ARID = ["Circleville", "Morgan", "Manderfield"]
DEPTHS = ["soil_vwc_5", "soil_vwc_10", "soil_vwc_20", "soil_vwc_50", "soil_vwc_101"]
GROW = range(4, 11)
SWEEP = [None, 7, 10, 14, 21, 30]  # None = posterior baseline (keep calibrated min_irr_days)
CONFIG = str(E8 / "8_Soil_Moisture_e8split.toml")
CONTAINER = "/data/ssd1/swim/8_Soil_Moisture/data/8_Soil_Moisture_e8split.swim"
PAR = "/data/ssd1/swim/8_Soil_Moisture/results/e8split/8_Soil_Moisture.3.par.csv"
SITES_CSV = E8 / "data" / "scan_sites.csv"


def deseason(s):
    return s - s.groupby(s.index.dayofyear).transform("mean")


def load_obs_etf(container_path, fids):
    """Observed 6-member Landsat ETf ensemble (no_mask), capture dates only, per fid."""
    z = zarr.open_group(container_path, mode="r")
    uids = [str(u) for u in z["geometry/uid"][:]]
    days = pd.to_datetime(z["time/daily"][:])
    arr = z["remote_sensing/etf/landsat/ensemble/no_mask"][:]  # (n_days, n_fields)
    out = {}
    for fid in fids:
        j = uids.index(fid)
        s = pd.Series(arr[:, j], index=days).dropna()  # capture dates only
        out[fid] = s
    return out


def build(cfg, container_path, par, fids):
    from evaluate import parse_pest_params

    from swimrs.container import SwimContainer
    from swimrs.process.input import build_swim_input

    container = SwimContainer(container_path, mode="r")
    calibrated = parse_pest_params(par, fids)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as t:
        h5 = t.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as t:
        json.dump(calibrated, t)
        pj = t.name
    si = build_swim_input(
        container,
        output_h5=h5,
        calibrated_params_path=pj,
        start_date=cfg.start_dt,
        end_date=cfg.end_dt,
        refet_type=getattr(cfg, "refet_type", "eto") or "eto",
        fields=fids,
        empirical_kc_max=True,
        mask_mode=getattr(cfg, "mask_mode", "none"),
        transpiration_cover_scaling=getattr(cfg, "transpiration_cover_scaling", True),
        stress_depletion_fraction=getattr(cfg, "stress_depletion_fraction", None),
    )
    return si, (h5, pj)


def run_one(si, min_irr_override):
    """Run the frozen posterior, optionally overriding min_irr_days at ALL built (arid)
    fields. Returns {fid: DataFrame}. Baseline (override None) keeps the posterior array."""
    from swimrs.process.loop_fast import run_daily_loop_fast

    props = si.properties
    saved = None if props.min_irr_days is None else props.min_irr_days.copy()
    if min_irr_override is not None:
        props.min_irr_days = np.full(len(si.fids), float(min_irr_override))
    try:
        out, _ = run_daily_loop_fast(si)
    finally:
        props.min_irr_days = saved
    dates = pd.date_range(si.start_date, periods=si.n_days, freq="D")
    awc = np.asarray(props.awc, dtype=float)
    zr_max = np.asarray(props.zr_max, dtype=float)
    res = {}
    for i, fid in enumerate(si.fids):
        depl = out.depl_root[:, i]
        ta = theta_available(awc[i], out.zr[:, i], depl, out.daw3[:, i], zr_max[i])
        res[fid] = pd.DataFrame(
            {
                "theta_avail": ta,
                "irr_sim": out.irr_sim[:, i],
                "eta": out.eta[:, i],
                "etf": out.etf[:, i],
                "ks": out.ks[:, i],
                "depl_root": depl,
            },
            index=dates,
        )
    return res


def score(m, rz, obs_etf):
    """theta skill (vs rootzone) + ET cost (vs obs etf, Ks, seasonal ET)."""
    j = pd.concat([m["theta_avail"].rename("mod"), rz], axis=1).dropna()
    j = j[j.index.month.isin(GROW)]
    if len(j) < 60:
        return None
    r = pearsonr(j["mod"], j["rootzone_theta"])[0]
    sr = j["mod"].std() / j["rootzone_theta"].std()
    ao = deseason(j["rootzone_theta"]).dropna()
    am = deseason(j["mod"]).reindex(ao.index).dropna()
    ao = ao.reindex(am.index)
    anom_r = pearsonr(ao, am)[0] if len(am) > 60 else np.nan

    # ET cost: model etf vs obs etf ensemble on capture dates (growing season)
    me = m["etf"].reindex(obs_etf.index).dropna()
    oe = obs_etf.reindex(me.index)
    mask = np.array([d.month in GROW for d in me.index])
    me, oe = me[mask], oe[mask]
    etf_bias = float((me - oe).mean()) if len(me) else np.nan
    etf_rmse = float(np.sqrt(((me - oe) ** 2).mean())) if len(me) else np.nan
    etf_r = pearsonr(me, oe)[0] if len(me) > 10 else np.nan

    gm = m[m.index.month.isin(GROW)]
    mean_ks = float(gm["ks"].mean())
    model_irr_days = int((gm["irr_sim"] > 1.0).sum())
    # seasonal ET total per year (mm/yr), averaged
    et_yr = gm["eta"].groupby(gm.index.year).sum()
    et_seasonal = float(et_yr.mean())
    irr_yr = gm["irr_sim"].groupby(gm.index.year).sum()
    irr_seasonal = float(irr_yr.mean())
    return dict(
        pearson=round(r, 3),
        anom_r=round(anom_r, 3),
        std_ratio=round(sr, 3),
        mean_ks=round(mean_ks, 3),
        etf_bias=round(etf_bias, 4),
        etf_rmse=round(etf_rmse, 4),
        etf_r=round(etf_r, 3),
        et_mm_yr=round(et_seasonal, 1),
        irr_mm_yr=round(irr_seasonal, 1),
        model_irr_days=model_irr_days,
    )


def main():
    cfg = _load_config(CONFIG)
    sites = pd.read_csv(SITES_CSV)
    theta_by = dict(zip(sites.site_id.astype(str), sites.theta_csv))
    obs_etf = load_obs_etf(CONTAINER, ARID)

    si, tmp = build(cfg, CONTAINER, PAR, ARID)
    try:
        rootzone = {}
        for fid in ARID:
            o = pd.read_parquet(theta_by[fid])
            o["date"] = pd.to_datetime(o["date"])
            o = o.set_index("date")
            rootzone[fid] = o[DEPTHS].mean(axis=1).rename("rootzone_theta")

        rows = []
        for mid in SWEEP:
            res = run_one(si, mid)
            for fid in ARID:
                s = score(res[fid], rootzone[fid], obs_etf[fid])
                if s is None:
                    continue
                rows.append(dict(site=fid, min_irr_days=("post" if mid is None else mid), **s))
    finally:
        si.close()
        for p in tmp:
            if os.path.exists(p):
                os.remove(p)

    df = pd.DataFrame(rows)
    out = HERE / "interval_sweep.csv"
    df.to_csv(out, index=False)
    for fid in ARID:
        print(f"\n=== {fid} ===")
        print(df[df.site == fid].drop(columns="site").to_string(index=False))
    print(f"\nwrote {out}")
    print(
        "\n--- read: theta recovers if pearson/std_ratio RISE with interval; ET-safe if "
        "etf_bias/etf_r and mean_ks stay ~flat (no stress penalty). Sweet spot = max theta "
        "skill before etf_bias goes negative / mean_ks drops."
    )


if __name__ == "__main__":
    main()
