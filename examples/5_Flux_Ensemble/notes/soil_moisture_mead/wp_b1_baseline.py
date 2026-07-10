"""WP-B1 step 3: baseline forward-run comparison at Mead (READ-ONLY).

Runs Run-22 (Ex5 container + params) forward at US-Ne1/Ne2/Ne3 and compares:
  * modeled theta_avail  vs  BASE profile theta (wp_b1_parse_base.py output)
  * modeled ET (output.eta)  vs  BASE eddy-covariance ET (raw + closure-corrected)
in the OpenET era (2016+), growing season (Apr-Oct), plus the whole record.

Also reports how much the model irrigates each field (annual mm) against the
BADM GRP_DM_WATER management record, to test the over-irrigation pathology that
the plan's H1/H2 question hinges on. Nothing here writes to any swim container,
result, or pestrun directory.

Feeds the DECISION GATE: pursue prescribed-irrigation (step 4) only if there is a
clear irrigated over-application pathology AND adequate theta+ET overlap.
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
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from evaluate import load_config, parse_pest_params  # noqa: E402

from swimrs.container import SwimContainer  # noqa: E402
from swimrs.process.input import build_swim_input  # noqa: E402
from swimrs.process.loop_fast import run_daily_loop_fast  # noqa: E402

SITES = ["US-Ne1", "US-Ne2", "US-Ne3"]
LABEL = {
    "US-Ne1": "Ne1 irrigated cont. maize",
    "US-Ne2": "Ne2 irrigated maize-soy",
    "US-Ne3": "Ne3 rainfed maize-soy",
}
CONTAINER = "/data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble_run22.swim"
PAR = "/data/ssd1/swim/5_Flux_Ensemble/results/run22/5_Flux_Ensemble.3.par.csv"
BASE = "/data/ssd1/swim/soil_moisture/mead_base/{}_daily.parquet"
OUT = os.path.dirname(__file__)
GROW = range(4, 11)  # Apr-Oct
OE_YEAR = 2016  # OpenET era


def run_model(cfg, container, fids):
    """Forward Run-22; return {fid: DataFrame(theta_avail, et_model, etf, irr_sim, ...)}."""
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
        )
        output, _ = run_daily_loop_fast(si)
        dates = pd.date_range(si.start_date, periods=si.n_days, freq="D")
        awc = np.asarray(si.properties.awc, dtype=float)
        zr_max = np.asarray(si.properties.zr_max, dtype=float)
        prcp = si.get_time_series("prcp")
        results = {}
        for i, fid in enumerate(si.fids):
            depl = output.depl_root[:, i]
            soil_water = awc[i] * output.zr[:, i] - depl + output.daw3[:, i]
            results[fid] = pd.DataFrame(
                {
                    "theta_avail": soil_water / (zr_max[i] * 1000.0),
                    "et_model": output.eta[:, i],
                    "etf_model": output.etf[:, i],
                    "irr_sim": output.irr_sim[:, i],
                    "depl_root": depl,
                    "precip": prcp[:, i],
                },
                index=dates,
            )
        si.close()
    finally:
        for p in [temp_h5, params_json]:
            if os.path.exists(p):
                os.remove(p)
    return results


def deseasonalize(s):
    return s - s.groupby(s.index.dayofyear).transform("mean")


def skill(obs, mod):
    """Pearson, Spearman, anomaly-r, sigma ratio, bias, RMSE on a paired frame."""
    d = pd.concat([obs.rename("o"), mod.rename("m")], axis=1).dropna()
    if len(d) < 30:
        return None
    an_o = deseasonalize(d["o"])
    an_m = deseasonalize(d["m"])
    a = pd.concat([an_o, an_m], axis=1).dropna()
    return dict(
        n=len(d),
        pearson=pearsonr(d["o"], d["m"])[0],
        spearman=spearmanr(d["o"], d["m"])[0],
        anom_r=pearsonr(a.iloc[:, 0], a.iloc[:, 1])[0] if len(a) > 30 else np.nan,
        sigma_obs=d["o"].std(),
        sigma_mod=d["m"].std(),
        sigma_ratio=d["m"].std() / d["o"].std() if d["o"].std() > 0 else np.nan,
        bias=(d["m"] - d["o"]).mean(),
        rmse=float(np.sqrt(((d["m"] - d["o"]) ** 2).mean())),
    )


def observed_irrigation_mm_by_year(site):
    """Annual irrigation total (mm) from the BADM GRP_DM_WATER record.

    Amount is embedded in the free-text DM_COMMENT as 'N.NN in' (inches).
    We take the first inch value per event and sum by DM_DATE_START year. This
    is a coarse read of the management record intended for the over-application
    gate, not the final prescribed-irrigation series (that is step 4).
    """
    s = site.split("-")[1]
    cands = glob.glob(
        f"/nas/climate/ameriflux/amf_new/AMF_US-{s}_BASE-BADM_*-5/AMF_US-{s}_BIF_*.xlsx"
    )
    if not cands:
        return pd.Series(dtype=float)
    x = pd.read_excel(sorted(cands)[-1], dtype=str)
    sub = x[x.VARIABLE_GROUP == "GRP_DM_WATER"]
    if sub.empty:
        return pd.Series(dtype=float)
    piv = sub.pivot_table(index="GROUP_ID", columns="VARIABLE", values="DATAVALUE", aggfunc="first")
    piv = piv[piv.get("DM_WATER") == "Irrigation"]
    if piv.empty:
        return pd.Series(dtype=float)
    yr = pd.to_datetime(piv["DM_DATE_START"].str[:8], format="%Y%m%d", errors="coerce").dt.year

    def first_inch(t):
        # amount is 'N.NN in' / '.NN in' / 'N.NN"'; leading-dot decimals need \d*\.?\d+.
        m = re.search(r'(\d*\.?\d+)\s*(?:in|")', str(t))
        if not m:
            return np.nan
        v = float(m.group(1))
        return v if v <= 3.0 else np.nan  # per-pass physical cap for a center pivot

    inch = piv["DM_COMMENT"].fillna("").apply(first_inch)
    return (inch * 25.4).groupby(yr).sum().dropna()


def main():
    cfg = load_config()
    container = SwimContainer.open(CONTAINER, mode="r")
    model = run_model(cfg, container, SITES)

    theta_rows, et_rows, irr_rows = [], [], []
    for site in SITES:
        m = model[site]
        obs = pd.read_parquet(BASE.format(site))
        df = m.join(obs, how="inner")

        # ---- irrigation: modeled vs observed annual totals ----
        mod_irr_yr = df["irr_sim"].resample("YE").sum()
        obs_irr_yr = observed_irrigation_mm_by_year(site)
        # OpenET-era means (where the observed record exists: 2016-2021)
        oe = df[df.index.year >= OE_YEAR]
        mod_irr_mean = mod_irr_yr[mod_irr_yr.index.year >= OE_YEAR].mean()
        obs_irr_mean = obs_irr_yr.mean() if len(obs_irr_yr) else np.nan
        irr_rows.append(
            dict(
                site=site,
                regime=LABEL[site],
                model_irr_mm_yr=round(float(mod_irr_mean), 0),
                obs_irr_mm_yr=round(float(obs_irr_mean), 0)
                if np.isfinite(obs_irr_mean)
                else np.nan,
                obs_irr_years=",".join(map(str, obs_irr_yr.index.astype(int)))
                if len(obs_irr_yr)
                else "",
                model_irr_days_per_yr=round(
                    float((oe["irr_sim"] > 0).groupby(oe.index.year).sum().mean()), 1
                ),
            )
        )

        # ---- skill: theta and ET, growing season, OpenET era & full record ----
        for era_tag, sub in [("2016+", oe), ("full", df)]:
            gs = sub[sub.index.month.isin(GROW)]
            for obs_col, mvar, kind in [
                ("theta_mean", "theta_avail", "theta_profmean"),
                ("theta_deep", "theta_avail", "theta_deep"),
            ]:
                sk = skill(gs[obs_col], gs[mvar])
                if sk:
                    theta_rows.append(dict(site=site, era=era_tag, target=kind, **sk))
            for obs_col, tag in [("et_le_raw", "et_raw"), ("et_le_corr", "et_corr")]:
                sk = skill(gs[obs_col], gs["et_model"])
                if sk:
                    et_rows.append(dict(site=site, era=era_tag, obs_et=tag, **sk))

        # ---- figures (OpenET-era growing seasons) ----
        oe_gs = oe[oe.index.month.isin(GROW)]
        fig, ax = plt.subplots(3, 1, figsize=(13, 10))
        # theta
        dd = oe[["theta_mean", "theta_avail"]].dropna()
        ax[0].plot(dd.index, dd["theta_mean"], lw=0.5, label="BASE profile theta (m3/m3)")
        ax0b = ax[0].twinx()
        ax0b.plot(dd.index, dd["theta_avail"], lw=0.5, color="tab:orange", label="SWIM theta_avail")
        sk = skill(oe_gs["theta_mean"], oe_gs["theta_avail"])
        ax[0].set_title(
            f"{site} {LABEL[site]}: theta  (grow r={sk['pearson']:.2f} anom={sk['anom_r']:.2f} "
            f"sigma_ratio={sk['sigma_ratio']:.2f})"
        )
        ax[0].set_ylabel("BASE theta")
        ax0b.set_ylabel("SWIM theta_avail")
        ax[0].legend(loc="upper left", fontsize=7)
        ax0b.legend(loc="upper right", fontsize=7)
        # ET
        ee = oe[["et_le_raw", "et_le_corr", "et_model"]].dropna(subset=["et_model"])
        ax[1].plot(ee.index, ee["et_le_raw"], lw=0.4, alpha=0.7, label="EC ET raw")
        ax[1].plot(ee.index, ee["et_le_corr"], lw=0.4, alpha=0.7, label="EC ET closure-corr")
        ax[1].plot(ee.index, ee["et_model"], lw=0.5, color="k", alpha=0.8, label="SWIM ET")
        ske = skill(oe_gs["et_le_corr"], oe_gs["et_model"])
        ax[1].set_title(
            f"ET  (grow vs corr: r={ske['pearson']:.2f} bias={ske['bias']:+.2f} "
            f"rmse={ske['rmse']:.2f} mm/d)"
        )
        ax[1].set_ylabel("ET mm/day")
        ax[1].legend(loc="upper right", fontsize=7)
        # irrigation
        ax[2].bar(mod_irr_yr.index.year, mod_irr_yr.values, width=0.4, label="SWIM irr_sim")
        if len(obs_irr_yr):
            ax[2].bar(
                obs_irr_yr.index + 0.4, obs_irr_yr.values, width=0.4, label="BADM observed irr"
            )
        ax[2].set_title("annual irrigation (mm)")
        ax[2].set_ylabel("mm/yr")
        ax[2].legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{OUT}/wp_b1_{site}_baseline.png", dpi=110)
        plt.close(fig)

    theta = pd.DataFrame(theta_rows)
    et = pd.DataFrame(et_rows)
    irr = pd.DataFrame(irr_rows)
    theta.to_csv(f"{OUT}/wp_b1_theta_skill.csv", index=False)
    et.to_csv(f"{OUT}/wp_b1_et_skill.csv", index=False)
    irr.to_csv(f"{OUT}/wp_b1_irrigation.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print("=== IRRIGATION: modeled vs observed (BADM) ===")
    print(irr.to_string(index=False))
    print("\n=== THETA skill (growing season) ===")
    print(theta.round(3).to_string(index=False))
    print("\n=== ET skill (growing season) ===")
    print(et.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
