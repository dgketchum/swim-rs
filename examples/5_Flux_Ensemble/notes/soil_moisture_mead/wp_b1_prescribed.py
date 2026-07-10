"""WP-B1 step 4: Mead prescribed-irrigation attribution (forward-run, READ-ONLY).

Discovery that reshapes the plan: the documented Ne1/Ne2 irrigation record is NOT
an external acquisition -- it lives in the AmeriFlux BADM (GRP_DM_WATER) we already
hold (Ne1 49 events, Ne2 37 events, 2014-2021, amounts in the free-text comments;
Ne3 has zero -- the rainfed control). Combined with BASE theta+ET (wp_b1_parse_base),
Mead is a full local theta+ET+irrigation triad in the OpenET era, so the decisive
prescribed-irrigation experiment can run entirely on data in hand.

This forces the model with the real irrigation via the WP-B0 hook
(build_swim_input(prescribed_irr_path=...)) and reads the plan's outcome matrix:
  theta good + ET holds -> scheduler is the deficiency (C1, = H1)
  theta good + ET collapses at low theta -> stress/rooting (C2, = H2)
  theta still wrong even with real irrigation -> vertical structure (C3)

Two configs, both forward runs (no recalibration):
  (i)  baseline   -- internal scheduler (the pinned behavior)
  (ii) prescribed -- observed daily irrigation injected; scheduler suppressed
       (0 mm) on all other days in the record window so the field receives ONLY
       the documented water.

Prescribed irrigation is management metadata, a diagnostic physics-bypass -- never a
production or calibration input. Nothing here writes to any container/result/pestrun.
"""

import glob
import json
import os
import re
import sys
import tempfile

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from evaluate import load_config, parse_pest_params  # noqa: E402
from wp_b1_baseline import BASE, LABEL, skill  # noqa: E402

from swimrs.container import SwimContainer  # noqa: E402
from swimrs.process.input import build_swim_input  # noqa: E402
from swimrs.process.loop_fast import run_daily_loop_fast  # noqa: E402

SITES = ["US-Ne1", "US-Ne2", "US-Ne3"]
CONTAINER = "/data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble_run22.swim"
PAR = "/data/ssd1/swim/5_Flux_Ensemble/results/run22/5_Flux_Ensemble.3.par.csv"
OUT = os.path.dirname(__file__)
PRESCRIBED_CSV = "/data/ssd1/swim/soil_moisture/mead_base/mead_prescribed_irr.csv"
REC_START, REC_END = "2014-01-01", "2021-12-31"  # documented-irrigation window
GROW = range(4, 11)


def build_prescribed_series():
    """BADM GRP_DM_WATER -> daily prescribed irrigation (mm/day) per field.

    In-window (2014-2021): 0.0 everywhere (suppress the scheduler), then each
    documented event's amount spread evenly over its DM_DATE_START..DM_DATE_END
    span. Out-of-window: left as NaN (scheduler runs; not analysed). Ne3 is all
    0.0 in-window -- the rainfed control receives no irrigation.
    """
    idx = pd.date_range(REC_START, REC_END, freq="D")
    out = pd.DataFrame(index=idx)
    out.index.name = "date"
    for site in SITES:
        s = site.split("-")[1]
        series = pd.Series(0.0, index=idx)  # in-window baseline: no scheduler water
        cands = glob.glob(
            f"/nas/climate/ameriflux/amf_new/AMF_US-{s}_BASE-BADM_*-5/AMF_US-{s}_BIF_*.xlsx"
        )
        if cands:
            x = pd.read_excel(sorted(cands)[-1], dtype=str)
            piv = x[x.VARIABLE_GROUP == "GRP_DM_WATER"].pivot_table(
                index="GROUP_ID", columns="VARIABLE", values="DATAVALUE", aggfunc="first"
            )
            if "DM_WATER" in piv.columns:
                piv = piv[piv["DM_WATER"] == "Irrigation"]
                for _, row in piv.iterrows():
                    m = re.search(r'(\d*\.?\d+)\s*(?:in|")', str(row.get("DM_COMMENT", "")))
                    if not m:
                        continue
                    inch = float(m.group(1))
                    if inch > 3.0:
                        continue
                    mm = inch * 25.4
                    d0 = pd.to_datetime(
                        str(row["DM_DATE_START"])[:8], format="%Y%m%d", errors="coerce"
                    )
                    de = row.get("DM_DATE_END")
                    d1 = (
                        pd.to_datetime(str(de)[:8], format="%Y%m%d", errors="coerce")
                        if pd.notna(de)
                        else d0
                    )
                    if pd.isna(d0):
                        continue
                    if pd.isna(d1) or d1 < d0:
                        d1 = d0
                    span = pd.date_range(d0.normalize(), d1.normalize(), freq="D")
                    span = span[(span >= idx[0]) & (span <= idx[-1])]
                    if len(span):
                        series.loc[span] += mm / len(span)
        out[site] = series
    out.to_csv(PRESCRIBED_CSV)
    return out


def run_model(cfg, container, fids, prescribed_path=None):
    calibrated = parse_pest_params(PAR, fids)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        temp_h5 = tmp.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(calibrated, tmp)
        params_json = tmp.name
    try:
        si = build_swim_input(
            container,
            output_h5=temp_h5,
            calibrated_params_path=params_json,
            start_date=cfg.start_dt,
            end_date=cfg.end_dt,
            refet_type=getattr(cfg, "refet_type", "eto") or "eto",
            fields=fids,
            empirical_kc_max=True,
            mask_mode=getattr(cfg, "mask_mode", "irrigation"),
            transpiration_cover_scaling=getattr(cfg, "transpiration_cover_scaling", True),
            prescribed_irr_path=prescribed_path,
        )
        output, _ = run_daily_loop_fast(si)
        dates = pd.date_range(si.start_date, periods=si.n_days, freq="D")
        awc = np.asarray(si.properties.awc, dtype=float)
        zr_max = np.asarray(si.properties.zr_max, dtype=float)
        results = {}
        for i, fid in enumerate(si.fids):
            sw = awc[i] * output.zr[:, i] - output.depl_root[:, i] + output.daw3[:, i]
            results[fid] = pd.DataFrame(
                {
                    "theta_avail": sw / (zr_max[i] * 1000.0),
                    "et_model": output.eta[:, i],
                    "irr_sim": output.irr_sim[:, i],
                    "depl_root": output.depl_root[:, i],
                },
                index=dates,
            )
        si.close()
    finally:
        for p in [temp_h5, params_json]:
            if os.path.exists(p):
                os.remove(p)
    return results


def main():
    cfg = load_config()
    build_prescribed_series()
    container = SwimContainer.open(CONTAINER, mode="r")
    base = run_model(cfg, container, SITES, prescribed_path=None)
    pres = run_model(cfg, container, SITES, prescribed_path=PRESCRIBED_CSV)

    rows = []
    for site in SITES:
        obs = pd.read_parquet(BASE.format(site))
        # analysis window: OpenET era within the documented-irrigation record
        win = lambda d: d[(d.index >= "2016-01-01") & (d.index <= REC_END)]  # noqa: E731
        for tag, m in [("baseline", base[site]), ("prescribed", pres[site])]:
            df = win(m.join(obs, how="inner"))
            gs = df[df.index.month.isin(GROW)]
            irr_yr = m["irr_sim"].resample("YE").sum()
            irr_mean = irr_yr[(irr_yr.index >= "2016") & (irr_yr.index <= REC_END)].mean()
            sk_t = skill(gs["theta_mean"], gs["theta_avail"])
            sk_e = skill(gs["et_le_corr"], gs["et_model"])
            rows.append(
                dict(
                    site=site,
                    regime=LABEL[site],
                    config=tag,
                    irr_mm_yr=round(float(irr_mean), 0),
                    theta_r=round(sk_t["pearson"], 3),
                    theta_anom_r=round(sk_t["anom_r"], 3),
                    theta_sigma_ratio=round(sk_t["sigma_ratio"], 3),
                    et_r=round(sk_e["pearson"], 3),
                    et_anom_r=round(sk_e["anom_r"], 3),
                    et_bias=round(sk_e["bias"], 3),
                    et_rmse=round(sk_e["rmse"], 3),
                )
            )

        # figure: theta and ET, baseline vs prescribed, over the record window
        obw = win(obs)
        fig, ax = plt.subplots(2, 1, figsize=(13, 8))
        ax[0].plot(win(obs).index, win(obs)["theta_mean"], lw=0.5, color="k", label="BASE theta")
        ax0b = ax[0].twinx()
        ax0b.plot(
            win(base[site]).index,
            win(base[site])["theta_avail"],
            lw=0.5,
            color="tab:blue",
            alpha=0.7,
            label="SWIM baseline",
        )
        ax0b.plot(
            win(pres[site]).index,
            win(pres[site])["theta_avail"],
            lw=0.5,
            color="tab:red",
            alpha=0.7,
            label="SWIM prescribed",
        )
        ax[0].set_title(f"{site} {LABEL[site]}: theta baseline vs prescribed irrigation")
        ax[0].set_ylabel("BASE theta (m3/m3)")
        ax0b.set_ylabel("SWIM theta_avail")
        ax[0].legend(loc="upper left", fontsize=7)
        ax0b.legend(loc="upper right", fontsize=7)
        eo = (
            win(obs)[["et_le_corr"]]
            .join(win(base[site])[["et_model"]].rename(columns={"et_model": "base"}))
            .join(win(pres[site])[["et_model"]].rename(columns={"et_model": "pres"}))
        )
        ax[1].plot(eo.index, eo["et_le_corr"], lw=0.4, color="k", alpha=0.6, label="EC ET corr")
        ax[1].plot(eo.index, eo["base"], lw=0.4, color="tab:blue", alpha=0.7, label="SWIM baseline")
        ax[1].plot(
            eo.index, eo["pres"], lw=0.4, color="tab:red", alpha=0.7, label="SWIM prescribed"
        )
        ax[1].set_ylabel("ET mm/day")
        ax[1].legend(loc="upper right", fontsize=7)
        ax[1].set_title("ET baseline vs prescribed vs observed")
        fig.tight_layout()
        fig.savefig(f"{OUT}/wp_b1_{site}_prescribed.png", dpi=110)
        plt.close(fig)

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/wp_b1_prescribed_outcome.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    print("=== PRESCRIBED-IRRIGATION OUTCOME (OpenET era within 2016-2021 record) ===")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
