"""Representation ceiling: score the frozen e8split posterior theta_avail against
cumulative SCAN depth windows at the arid irrigated sites.

The single surface-fed root-zone bucket cannot reproduce a deep sensor that does not
respond to surface irrigation (below the wetting front, or driven by deep drainage /
capillary rise / a water table). Scoring against the full 5-101 cm mean therefore
penalizes the model for a physics it does not claim. Widening the window from 5 cm
downward shows where the correlation ceiling collapses -> the agronomically-active depth
the model actually represents, vs the deep sensors that are irreducibly decorrelated.

Forward-run only, no calibration. Reads validation theta strictly for scoring.
"""

import importlib.util
import sys
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
diag = _load_module(str(HERE / "diagnose_arid.py"), "arid_diag")

ARID = ["Circleville", "Morgan", "Manderfield"]
WINDOWS = [
    ["soil_vwc_5"],
    ["soil_vwc_5", "soil_vwc_10"],
    ["soil_vwc_5", "soil_vwc_10", "soil_vwc_20"],
    ["soil_vwc_5", "soil_vwc_10", "soil_vwc_20", "soil_vwc_50"],
    ["soil_vwc_5", "soil_vwc_10", "soil_vwc_20", "soil_vwc_50", "soil_vwc_101"],
]
GROW = range(4, 11)


def deseason(s):
    return s - s.groupby(s.index.dayofyear).transform("mean")


def main():
    cfg = ev8._load_config(diag.CONFIG)
    sites = pd.read_csv(diag.SITES_CSV)
    theta_by = dict(zip(sites.site_id.astype(str), sites.theta_csv))
    model = diag.run_full(cfg, diag.CONTAINER, diag.PAR, ARID)

    rows = []
    for fid in ARID:
        m = model[fid]
        obs = pd.read_parquet(theta_by[fid])
        obs["date"] = pd.to_datetime(obs["date"])
        obs = obs.set_index("date")
        for w in WINDOWS:
            tgt = obs[w].mean(axis=1).rename("tgt")
            j = pd.concat([m["theta_avail"].rename("mod"), tgt], axis=1).dropna()
            j = j[j.index.month.isin(GROW)]
            if len(j) < 60:
                continue
            r = pearsonr(j["mod"], j["tgt"])[0]
            ao = deseason(j["tgt"]).dropna()
            am = deseason(j["mod"]).reindex(ao.index).dropna()
            ao = ao.reindex(am.index)
            anom_r = pearsonr(ao, am)[0] if len(am) > 60 else np.nan
            rows.append(
                dict(
                    site=fid,
                    window=f"5-{w[-1].split('_')[-1]}cm",
                    pearson=round(r, 3),
                    anom_r=round(anom_r, 3),
                    std_ratio=round(j["mod"].std() / j["tgt"].std(), 3),
                    n=len(j),
                )
            )
    df = pd.DataFrame(rows)
    out = HERE / "depth_window.csv"
    df.to_csv(out, index=False)
    for fid in ARID:
        print(f"\n=== {fid} ===")
        print(df[df.site == fid].drop(columns="site").to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
