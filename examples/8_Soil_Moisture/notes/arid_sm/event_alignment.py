"""Why did SMAP-timed redistribution HURT correlation? Direct alignment diagnostic:
do SMAP-detected rain-free wet-up days coincide with OBSERVED SCAN rootzone wet-ups?

For each arid site, growing season:
  - SMAP event days   = daily-interp SMAP increment > INC_THRESH in rain-free window.
  - SCAN wetup days    = active-zone (5-20 cm) rootzone rise > 0.01/day.
  - model irr days     = posterior irr_sim > 1 mm.
Report the hit rate of each candidate schedule against the SCAN wet-ups (fraction of SCAN
wet-ups with a same/±1 d event) and its precision (fraction of events landing on a SCAN
wet-up). If SMAP events have LOW precision/recall vs SCAN wet-ups, the discrete detector
is the failure -- SMAP surface timing does not map to SCAN rootzone wet-ups event-for-event
even though their smoothed anomalies correlate.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
E8 = HERE.parent.parent
E5 = E8.parent / "5_Flux_Ensemble"
if str(E5) not in sys.path:
    sys.path.insert(0, str(E5))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ev8 = _load(str(E8 / "evaluate.py"), "ex8_evaluate")
diag = _load(str(HERE / "diagnose_arid.py"), "arid_diag")
st = _load(str(HERE / "smap_timing.py"), "smap_timing")

ARID = ["Circleville", "Morgan", "Manderfield"]
ACTIVE = ["soil_vwc_5", "soil_vwc_10", "soil_vwc_20"]
GROW = range(4, 11)


def _daynums(idx):
    # normalize to integer day counts; robust to datetime64[us] vs [ns] unit mismatch
    return pd.DatetimeIndex(idx).normalize().values.astype("datetime64[D]").astype(np.int64)


def within(days_a, days_b, tol=1):
    """fraction of days_a that have a day_b within +-tol days."""
    a = _daynums(days_a)
    if len(a) == 0:
        return np.nan
    b = np.array(sorted(_daynums(days_b)))
    hit = 0
    for d in a:
        if b.size and np.min(np.abs(b - d)) <= tol:
            hit += 1
    return round(hit / len(a), 3)


def main():
    cfg = ev8._load_config(diag.CONFIG)
    sites = pd.read_csv(diag.SITES_CSV)
    theta_by = dict(zip(sites.site_id.astype(str), sites.theta_csv))
    smap_df = pd.read_parquet(diag.SMAP)
    smap_df["date"] = pd.to_datetime(smap_df["date"])

    si, tmp = st.build(cfg, diag.CONTAINER, diag.PAR, ARID)
    import os

    try:
        index = pd.date_range(si.start_date, periods=si.n_days, freq="D")
        base = st.run(si)
    finally:
        si.close()
        for p in tmp:
            if os.path.exists(p):
                os.remove(p)

    rows = []
    for fid in ARID:
        b = base[fid]
        obs = pd.read_parquet(theta_by[fid])
        obs["date"] = pd.to_datetime(obs["date"])
        obs = obs.set_index("date")
        rz = obs[ACTIVE].mean(axis=1)

        # SCAN active-zone wet-ups (growing season)
        rz_g = rz[rz.index.month.isin(GROW)]
        scan_wet = rz_g.index[rz_g.diff() > 0.01]

        # SMAP events (same detector as the timing experiment)
        smapd = st.smap_daily(smap_df, fid, index)
        rain = b["rain"].reindex(index).fillna(0.0)
        rain3 = rain.rolling(3, min_periods=1).sum()
        inc = smapd.diff()
        grow = pd.Series(index.month.isin(GROW), index=index)
        smap_ev = index[(inc > st.INC_THRESH) & (rain3 < st.RAIN_THRESH) & grow]

        # model posterior irr days
        bg = b[b.index.month.isin(GROW)]
        model_irr = bg.index[bg["irr_sim"] > 1.0]

        rows.append(
            dict(
                site=fid,
                scan_wetups=len(scan_wet),
                smap_events=len(smap_ev),
                model_irr_days=len(model_irr),
                # recall: SCAN wet-ups explained by each schedule
                recall_smap=within(scan_wet, smap_ev, 1),
                recall_model=within(scan_wet, model_irr, 1),
                # precision: events landing on a SCAN wet-up
                prec_smap=within(smap_ev, scan_wet, 1),
                prec_model=within(model_irr, scan_wet, 1),
                # do SMAP events even coincide with rain? (rain leakage check)
                smap_on_rain=within(smap_ev, index[rain3.values > st.RAIN_THRESH], 0),
            )
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(HERE / "event_alignment.csv", index=False)
    print(f"\nwrote {HERE / 'event_alignment.csv'}")
    print(
        "\n--- read: if recall_smap/prec_smap are LOW (~chance), SMAP discrete events do not "
        "map to SCAN rootzone wet-ups -> the surface-event detector cannot pin rootzone timing, "
        "explaining why volume-redistribution onto SMAP events hurt correlation."
    )


if __name__ == "__main__":
    main()
