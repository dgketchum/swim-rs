"""Decompose the model-vs-observed soil-moisture mismatch at the arid, IrrMapper-
confirmed irrigated SCAN sites (Circleville, Morgan, Manderfield, all UT).

The question: WHY does simulated theta decorrelate from in-situ theta at genuinely
irrigated arid sites (as opposed to the humid eastern sites that may be mislabeled)?

Decomposition, per site, growing season (Apr-Oct):
  1. Per-depth Pearson  (model theta_avail / surface proxy vs each SCAN sensor depth)
     -> which depth, if any, does the single-bucket model represent?
  2. Amplitude ratio     sigma(model)/sigma(obs)  -> variance collapse vs healthy swing.
  3. Lagged cross-correlation (model vs obs, lags -30..+30 d) -> is the decorrelation a
     PHASE error (a lag recovers r) or structural (no lag helps)?
  4. Irrigation-event alignment: model irr_sim pulses vs observed theta wet-ups.
  5. Precip attribution: do observed wet-ups coincide with GridMET precip (rain-driven)
     or occur in rain-free windows (irrigation-driven, invisible to the model's forcing)?
  6. SMAP timing skill at the site (anomaly-r vs 5 cm) -> is an RS timing signal available?

Forward-runs only (e8split posterior), no calibration, no EE. Reads validation theta
strictly for scoring. Writes a per-site diagnostic table + a phase/amplitude summary.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
E8 = HERE.parent.parent
E5 = E8.parent / "5_Flux_Ensemble"
# Ex5 must be importable as bare `evaluate` (Ex8 evaluate imports parse_pest_params from it).
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
CONFIG = str(E8 / "8_Soil_Moisture_e8split.toml")
CONTAINER = "/data/ssd1/swim/8_Soil_Moisture/data/8_Soil_Moisture_e8split.swim"
PAR = "/data/ssd1/swim/8_Soil_Moisture/results/e8split/8_Soil_Moisture.3.par.csv"
SMAP = E8 / "notes" / "paper" / "data" / "smap_l3_scan.parquet"
SITES_CSV = E8 / "data" / "scan_sites.csv"


def deseason(s):
    return s - s.groupby(s.index.dayofyear).transform("mean")


def run_full(cfg, container, par, fids):
    """Forward-run and return {fid: DataFrame} with theta_avail + surface proxy +
    irr_sim + rain + eta + etf + ks, indexed by date. Mirrors evaluate.run_model but
    dumps the water-balance fluxes too."""
    import json
    import tempfile

    from evaluate import parse_pest_params

    from swimrs.container import SwimContainer
    from swimrs.process.input import build_swim_input
    from swimrs.process.loop_fast import run_daily_loop_fast

    if isinstance(container, str):
        container = SwimContainer(container, mode="r")
    calibrated = parse_pest_params(par, fids)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as t:
        h5 = t.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as t:
        json.dump(calibrated, t)
        pj = t.name
    try:
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
        out, _ = run_daily_loop_fast(si)
        dates = pd.date_range(si.start_date, periods=si.n_days, freq="D")
        awc = np.asarray(si.properties.awc, dtype=float)
        zr_max = np.asarray(si.properties.zr_max, dtype=float)
        res = {}
        for i, fid in enumerate(si.fids):
            depl = out.depl_root[:, i]
            ta = theta_available(awc[i], out.zr[:, i], depl, out.daw3[:, i], zr_max[i])
            res[fid] = pd.DataFrame(
                {
                    "theta_avail": ta,
                    "surface_proxy": -out.depl_ze[:, i],
                    "irr_sim": out.irr_sim[:, i],
                    "rain": out.rain[:, i],
                    "eta": out.eta[:, i],
                    "etf": out.etf[:, i],
                    "ks": out.ks[:, i],
                    "depl_root": depl,
                },
                index=dates,
            )
        si.close()
    finally:
        for p in (h5, pj):
            if os.path.exists(p):
                os.remove(p)
    return res


def lag_xcorr(obs, mod, max_lag=30):
    """Best Pearson over integer day lags in [-max_lag, +max_lag].
    Positive lag = model LEADS obs (model shifted later to match)."""
    best_r, best_lag = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        m = mod.shift(lag)
        d = pd.concat([obs, m], axis=1).dropna()
        if len(d) < 60:
            continue
        r = pearsonr(d.iloc[:, 0], d.iloc[:, 1])[0]
        if r > best_r:
            best_r, best_lag = r, lag
    return best_r, best_lag


def main():
    cfg = _load_config(CONFIG)
    sites = pd.read_csv(SITES_CSV)
    theta_by = dict(zip(sites.site_id.astype(str), sites.theta_csv))
    smap = pd.read_parquet(SMAP)
    smap["date"] = pd.to_datetime(smap["date"])

    model = run_full(cfg, CONTAINER, PAR, ARID)

    rows, depth_rows = [], []
    for fid in ARID:
        m = model[fid]
        obs = pd.read_parquet(theta_by[fid])
        obs["date"] = pd.to_datetime(obs["date"])
        obs = obs.set_index("date")
        rz = obs[DEPTHS].mean(axis=1).rename("rootzone_theta")  # simple profile mean

        # growing-season paired frame on theta_avail vs rootzone
        j = pd.concat([m["theta_avail"].rename("mod"), rz], axis=1).dropna()
        j = j[j.index.month.isin(GROW)]
        if len(j) < 60:
            continue
        r0 = pearsonr(j["mod"], j["rootzone_theta"])[0]
        sr = j["mod"].std() / j["rootzone_theta"].std()
        best_r, best_lag = lag_xcorr(j["rootzone_theta"], j["mod"])
        # anomaly-space (deseasonalized) correlation
        ao = deseason(j["rootzone_theta"]).dropna()
        am = deseason(j["mod"]).reindex(ao.index).dropna()
        ao = ao.reindex(am.index)
        anom_r = pearsonr(ao, am)[0] if len(am) > 60 else np.nan

        # per-depth correlations (which sensor does the bucket track best?)
        for dcol in DEPTHS:
            dd = pd.concat([m["theta_avail"].rename("mod"), obs[dcol]], axis=1).dropna()
            dd = dd[dd.index.month.isin(GROW)]
            if len(dd) < 60:
                continue
            depth_rows.append(
                dict(
                    site=fid,
                    depth=dcol,
                    r=round(pearsonr(dd["mod"], dd[dcol])[0], 3),
                    std_ratio=round(dd["mod"].std() / dd[dcol].std(), 3),
                    n=len(dd),
                )
            )

        # irrigation-event count: model pulses vs observed rain-free wet-ups
        gm = m[m.index.month.isin(GROW)]
        model_events = int((gm["irr_sim"] > 1.0).sum())  # days with real application
        # observed wet-ups: rootzone rises > 0.02 day-over-day
        rz_g = rz[rz.index.month.isin(GROW)]
        wetup = (rz_g.diff() > 0.02).sum()
        # precip attribution: fraction of observed wet-ups within 2 d of GridMET rain
        rain_g = gm["rain"].reindex(rz_g.index).fillna(0.0)
        rain_recent = rain_g.rolling(3, min_periods=1).sum()
        wet_days = rz_g.diff() > 0.02
        rain_driven = int(((wet_days) & (rain_recent > 2.0)).sum())
        n_wet = int(wet_days.sum())

        # SMAP surface timing skill (anom-r vs 5cm)
        sm = (
            smap[smap.site_id == fid][["date", "smap_l3_sm"]]
            .dropna()
            .set_index("date")["smap_l3_sm"]
        )
        s5 = obs["soil_vwc_5"].dropna()
        sj = pd.concat([sm.rename("smap"), s5.rename("s5")], axis=1).dropna()
        sj = sj[sj.index.month.isin(GROW)]
        if len(sj) > 60:
            aso = deseason(sj["s5"]).dropna()
            asm = deseason(sj["smap"]).reindex(aso.index).dropna()
            aso = aso.reindex(asm.index)
            smap_anom_r = pearsonr(aso, asm)[0]
            # does SMAP surface anomaly track the MODEL surface proxy? (can it re-phase?)
            msp = m["surface_proxy"].reindex(sj.index)
            amsp = deseason(msp).reindex(asm.index).dropna()
            asm2 = asm.reindex(amsp.index)
            smap_vs_model_surface = pearsonr(asm2, amsp)[0] if len(amsp) > 60 else np.nan
        else:
            smap_anom_r = smap_vs_model_surface = np.nan

        rows.append(
            dict(
                site=fid,
                n=len(j),
                pearson=round(r0, 3),
                anom_r=round(anom_r, 3),
                std_ratio=round(sr, 3),
                best_lag_r=round(best_r, 3),
                best_lag_days=best_lag,
                phase_gain=round(best_r - r0, 3),
                model_irr_days=model_events,
                obs_wetups=n_wet,
                obs_wetups_rain_driven=rain_driven,
                obs_wetups_irrig_driven=n_wet - rain_driven,
                smap_anom_r=round(smap_anom_r, 3),
                smap_vs_modelsurf=round(smap_vs_model_surface, 3),
            )
        )

    summ = pd.DataFrame(rows)
    depth = pd.DataFrame(depth_rows)
    print("=== ARID IRRIGATED SITES: phase/amplitude decomposition (e8split) ===")
    print(summ.to_string(index=False))
    print("\n=== per-depth model tracking (which sensor does the bucket follow?) ===")
    print(depth.to_string(index=False))
    out = HERE / "arid_diagnosis.csv"
    summ.to_csv(out, index=False)
    depth.to_csv(HERE / "arid_depth.csv", index=False)
    print(f"\nwrote {out}")

    # interpretation guide
    print("\n--- read ---")
    print(
        "phase_gain high (best_lag_r >> pearson) => decorrelation is TIMING; a lag recovers skill."
    )
    print(
        "std_ratio ~0.5-1.0 with pearson~0 => healthy amplitude, wrong phase (schedule ignorance)."
    )
    print(
        "obs_wetups_irrig_driven >> model_irr timing alignment => model misses the real schedule."
    )
    print("smap_anom_r high => an RS timing signal EXISTS to re-phase the scheduler.")


if __name__ == "__main__":
    main()
