"""Volume-preserving SMAP-timing experiment at the arid irrigated SCAN sites.

The diagnosis established that the arid-irrigated theta decorrelation is a TIMING problem:
amplitude is tunable (interval sweep) and the active-root-zone representation is decent
(Morgan 5-20 cm r=0.52), but the model irrigates on the wrong DAYS because it infers
timing from ET demand while the farmer's schedule is exogenous. SMAP surface anomaly
carries that timing signal (anom_r 0.47-0.65, exceeding the model's own rootzone skill).

This experiment isolates TIMING from VOLUME. For each site-year:
  1. Run the frozen e8split posterior -> record that year's growing-season irrigation
     VOLUME (mm) and the GridMET rain series.
  2. Detect SMAP irrigation events: daily-interpolated SMAP surface increments above a
     threshold in rain-free windows (rain-driven wet-ups are already in the forcing).
  3. Redistribute the SAME posterior volume onto the SMAP event days (split by SMAP jump
     magnitude), zero on other growing-season days. Non-detected years fall back to the
     scheduler (NaN) so the crop is never starved.
  4. Prescribe via the run_daily_loop_fast `prescribed_irr` bypass and re-score.

Because volume is preserved per year, ET/ETf is held ~constant BY CONSTRUCTION -- the only
thing that changes is WHEN water is applied. If theta correlation rises, timing is the
recoverable lever and SMAP supplies it. If not, the timing is not extractable from SMAP.

SMAP is a remote-sensing product (permitted as model input); SCAN in-situ theta is read
strictly for scoring. No calibration, no EE.
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
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
theta_available = ev8.theta_available

ARID = ["Circleville", "Morgan", "Manderfield"]
DEPTHS_FULL = ["soil_vwc_5", "soil_vwc_10", "soil_vwc_20", "soil_vwc_50", "soil_vwc_101"]
DEPTHS_ACTIVE = ["soil_vwc_5", "soil_vwc_10", "soil_vwc_20"]  # agronomic root zone
GROW = range(4, 11)
INC_THRESH = 0.015  # SMAP volumetric daily increment flagging a wet-up (m3/m3)
RAIN_THRESH = 3.0  # mm in prior 3 d above which a wet-up is deemed rain-driven
CONFIG = str(E8 / "8_Soil_Moisture_e8split.toml")
CONTAINER = "/data/ssd1/swim/8_Soil_Moisture/data/8_Soil_Moisture_e8split.swim"
PAR = "/data/ssd1/swim/8_Soil_Moisture/results/e8split/8_Soil_Moisture.3.par.csv"
SMAP = E8 / "notes" / "paper" / "data" / "smap_l3_scan.parquet"
SITES_CSV = E8 / "data" / "scan_sites.csv"


def deseason(s):
    return s - s.groupby(s.index.dayofyear).transform("mean")


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


def run(si, prescribed=None):
    from swimrs.process.loop_fast import run_daily_loop_fast

    out, _ = run_daily_loop_fast(si, prescribed_irr=prescribed)
    dates = pd.date_range(si.start_date, periods=si.n_days, freq="D")
    props = si.properties
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
                "rain": out.rain[:, i],
                "eta": out.eta[:, i],
                "etf": out.etf[:, i],
                "ks": out.ks[:, i],
            },
            index=dates,
        )
    return res


def smap_daily(smap_df, fid, index):
    s = (
        smap_df[smap_df.site_id == fid][["date", "smap_l3_sm"]]
        .dropna()
        .set_index("date")["smap_l3_sm"]
    )
    s = s[~s.index.duplicated()]
    # daily reindex + short-gap interpolation (<=5 d), no extrapolation
    d = s.reindex(index).interpolate(limit=5, limit_area="inside")
    return d


def score_pair(mod, tgt):
    j = pd.concat([mod.rename("mod"), tgt.rename("tgt")], axis=1).dropna()
    j = j[j.index.month.isin(GROW)]
    if len(j) < 60:
        return None
    r = pearsonr(j["mod"], j["tgt"])[0]
    ao = deseason(j["tgt"]).dropna()
    am = deseason(j["mod"]).reindex(ao.index).dropna()
    ao = ao.reindex(am.index)
    anom_r = pearsonr(ao, am)[0] if len(am) > 60 else np.nan
    return dict(
        pearson=round(r, 3),
        anom_r=round(anom_r, 3),
        std_ratio=round(j["mod"].std() / j["tgt"].std(), 3),
        n=len(j),
    )


def build_prescription(base, smapd, index):
    """Volume-preserving SMAP-timed prescription for one field.
    Returns (pres_series over index, n_years_timed, total_events)."""
    rain = base["rain"].reindex(index).fillna(0.0)
    rain3 = rain.rolling(3, min_periods=1).sum()
    inc = smapd.diff()
    grow = pd.Series(index.month.isin(GROW), index=index)
    event = (inc > INC_THRESH) & (rain3 < RAIN_THRESH) & grow
    jump = inc.where(event, 0.0).clip(lower=0.0)

    pres = pd.Series(np.nan, index=index)
    n_years, n_events = 0, 0
    for yr, idx_yr in pd.Series(index, index=index).groupby(index.year):
        gmask = idx_yr.index.month.isin(GROW)
        gdays = idx_yr.index[gmask]
        if len(gdays) == 0:
            continue
        vol = float(base.loc[gdays, "irr_sim"].sum())  # posterior growing-season volume
        ev_days = jump.loc[gdays]
        ev_days = ev_days[ev_days > 0]
        if vol <= 0 or len(ev_days) == 0:
            continue  # leave NaN -> scheduler fallback for this year
        weights = ev_days / ev_days.sum()
        pres.loc[gdays] = 0.0  # suppress scheduler on all growing days this year
        pres.loc[ev_days.index] = (weights * vol).values
        n_years += 1
        n_events += len(ev_days)
    return pres, n_years, n_events


def main():
    cfg = ev8._load_config(CONFIG)
    sites = pd.read_csv(SITES_CSV)
    theta_by = dict(zip(sites.site_id.astype(str), sites.theta_csv))
    smap_df = pd.read_parquet(SMAP)
    smap_df["date"] = pd.to_datetime(smap_df["date"])

    si, tmp = build(cfg, CONTAINER, PAR, ARID)
    try:
        index = pd.date_range(si.start_date, periods=si.n_days, freq="D")
        base = run(si)  # posterior baseline

        # build volume-preserving SMAP-timed prescription (n_days, n_fields)
        pres_arr = np.full((si.n_days, len(si.fids)), np.nan)
        meta = {}
        for i, fid in enumerate(si.fids):
            smapd = smap_daily(smap_df, fid, index)
            pres, ny, ne = build_prescription(base[fid], smapd, index)
            pres_arr[:, i] = pres.values
            meta[fid] = (ny, ne)

        timed = run(si, prescribed=pres_arr)
    finally:
        si.close()
        for p in tmp:
            if os.path.exists(p):
                os.remove(p)

    rows = []
    for fid in ARID:
        obs = pd.read_parquet(theta_by[fid])
        obs["date"] = pd.to_datetime(obs["date"])
        obs = obs.set_index("date")
        rz_full = obs[DEPTHS_FULL].mean(axis=1)
        rz_act = obs[DEPTHS_ACTIVE].mean(axis=1)
        for tag, target in [("full_5-101", rz_full), ("active_5-20", rz_act)]:
            for run_tag, r in [("baseline", base[fid]), ("smap_timed", timed[fid])]:
                s = score_pair(r["theta_avail"], target)
                if s is None:
                    continue
                # ET preservation check
                gm = r[r.index.month.isin(GROW)]
                et = float(gm["eta"].groupby(gm.index.year).sum().mean())
                irr = float(gm["irr_sim"].groupby(gm.index.year).sum().mean())
                rows.append(
                    dict(
                        site=fid,
                        target=tag,
                        run=run_tag,
                        **s,
                        et_mm_yr=round(et, 1),
                        irr_mm_yr=round(irr, 1),
                        mean_ks=round(float(gm["ks"].mean()), 3),
                        yrs_timed=meta[fid][0],
                        events=meta[fid][1],
                    )
                )
    df = pd.DataFrame(rows)
    out = HERE / "smap_timing.csv"
    df.to_csv(out, index=False)
    for fid in ARID:
        print(f"\n=== {fid}  (SMAP-timed {meta[fid][0]} yrs, {meta[fid][1]} events) ===")
        print(
            df[df.site == fid].drop(columns=["site", "yrs_timed", "events"]).to_string(index=False)
        )
    print(f"\nwrote {out}")
    print(
        "\n--- read: TIMING is the lever iff pearson/anom_r RISE baseline->smap_timed while "
        "et_mm_yr/irr_mm_yr stay ~equal (volume preserved). Compare on the active_5-20 target "
        "(the depth the bucket represents). Flat/negative => SMAP timing not extractable here."
    )


if __name__ == "__main__":
    main()
