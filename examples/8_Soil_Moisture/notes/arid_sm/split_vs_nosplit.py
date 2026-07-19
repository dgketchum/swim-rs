"""Matched split-vs-no-split at the 3 arid IrrMapper-domain irrigated SCAN sites.

Answers directly: does the WP-C7 mad-split help the GENUINE arid sites (as opposed to
the cohort median, which is buoyed by likely-mislabeled humid eastern sites)? Scores the
e8cal posterior (base config, stress_depletion_fraction=None -> legacy overloaded mad) and
the e8split posterior (split config, sdf=0.5 -> p_stress decoupled from mad_irr) with the
SAME scorer, at both the full 5-101 cm rootzone mean and the 5-20 cm agronomic root zone.

Forward runs only, no calibration, no EE. Validation theta read strictly for scoring.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
E8 = HERE.parent.parent
E5 = E8.parent / "5_Flux_Ensemble"
if str(E5) not in sys.path:
    sys.path.insert(0, str(E5))


def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


ev8 = _load(str(E8 / "evaluate.py"), "ex8_evaluate")
st = _load(str(HERE / "smap_timing.py"), "smap_timing")  # reuse build()/run()

ARID = ["Circleville", "Morgan", "Manderfield"]
FULL = ["soil_vwc_5", "soil_vwc_10", "soil_vwc_20", "soil_vwc_50", "soil_vwc_101"]
ACTIVE = ["soil_vwc_5", "soil_vwc_10", "soil_vwc_20"]
GROW = range(4, 11)

RUNS = {
    "nosplit": dict(
        config=str(E8 / "8_Soil_Moisture.toml"),
        container="/data/ssd1/swim/8_Soil_Moisture/data/8_Soil_Moisture_e8cal.swim",
        par="/data/ssd1/swim/8_Soil_Moisture/results/e8cal/archive/4_pest_outputs/8_Soil_Moisture.3.par.csv",
    ),
    "split": dict(
        config=str(E8 / "8_Soil_Moisture_e8split.toml"),
        container="/data/ssd1/swim/8_Soil_Moisture/data/8_Soil_Moisture_e8split.swim",
        par="/data/ssd1/swim/8_Soil_Moisture/results/e8split/8_Soil_Moisture.3.par.csv",
    ),
}


def score(mod, tgt):
    j = pd.concat([mod.rename("m"), tgt.rename("t")], axis=1).dropna()
    j = j[j.index.month.isin(GROW)]
    if len(j) < 60:
        return None
    return dict(
        pearson=round(pearsonr(j["m"], j["t"])[0], 3),
        std_ratio=round(j["m"].std() / j["t"].std(), 3),
        n=len(j),
    )


def main():
    sites = pd.read_csv(E8 / "data" / "scan_sites.csv")
    theta_by = dict(zip(sites.site_id.astype(str), sites.theta_csv))
    obs = {}
    for fid in ARID:
        o = pd.read_parquet(theta_by[fid])
        o["date"] = pd.to_datetime(o["date"])
        o = o.set_index("date")
        obs[fid] = dict(full=o[FULL].mean(axis=1), active=o[ACTIVE].mean(axis=1))

    theta = {}
    for tag, cfgd in RUNS.items():
        cfg = ev8._load_config(cfgd["config"])
        si, tmp = st.build(cfg, cfgd["container"], cfgd["par"], ARID)
        try:
            res = st.run(si)
        finally:
            si.close()
            for p in tmp:
                if os.path.exists(p):
                    os.remove(p)
        theta[tag] = {fid: res[fid]["theta_avail"] for fid in ARID}

    rows = []
    for fid in ARID:
        for depth, tgt in [("full_5-101", obs[fid]["full"]), ("active_5-20", obs[fid]["active"])]:
            c = score(theta["nosplit"][fid], tgt)
            s = score(theta["split"][fid], tgt)
            if c and s:
                rows.append(
                    dict(
                        site=fid,
                        target=depth,
                        pearson_nosplit=c["pearson"],
                        pearson_split=s["pearson"],
                        d_pearson=round(s["pearson"] - c["pearson"], 3),
                        sr_nosplit=c["std_ratio"],
                        sr_split=s["std_ratio"],
                        n=c["n"],
                    )
                )
    df = pd.DataFrame(rows)
    out = HERE / "split_vs_nosplit_arid.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
