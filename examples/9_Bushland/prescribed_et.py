"""WP-B2 -- Bushland prescribed-irrigation ET check (forward-run comparison).

The question: if we PRESCRIBE the real metered daily irrigation, does the model
reproduce the gold-standard weighing-lysimeter ET *without* the over-application
the internal scheduler produces? A "yes" corroborates the WP-A H1 verdict -- the
scheduler tops the root zone up unnecessarily; ET does not require it.

This is a FORWARD-run comparison (no PEST++). Two configurations are run at one
fixed, reasonable cropland parameter vector (the frozen Example 5 Run-22 cropland
median transfer vector, ``data/ex5_cropland_params.json``, applied uniformly to
all four fields -- documented, defensible, and not tuned to Bushland):

    (i)  scheduler  -- the internal irrigation scheduler decides irr_sim;
    (ii) prescribed -- the metered daily irrigation REPLACES irr_sim per field/day
         over the OpenET-era crop-years (WP-B0 override via prescribed_irr_path).

Both modeled ET series (``output.eta``) are compared to the measured lysimeter ET
and to the OpenET RS-ETa reference (ensemble ETf x ETo). We report, per crop-year:
does prescribed real-irrigation ET match lysimeter ET? how much LESS irrigation
does the real record apply than the scheduler (the over-application quantum)? and
does ET hold despite less water (the H1 signature)?

Metered irrigation is a diagnostic physics-bypass input; the lysimeter ET is the
validation target. Neither is ever a model parameter or a calibration input.

    uv run python examples/9_Bushland/prescribed_et.py \
        --container /data/ssd1/swim/9_Bushland/data/9_Bushland_e9for.swim
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

HERE = Path(__file__).resolve().parent
E5 = HERE.parent / "5_Flux_Ensemble"
if str(E5) not in sys.path:
    sys.path.insert(0, str(E5))

from evaluate import load_openet_etf_nomask  # noqa: E402  (Example 5 helper)

from swimrs.container import SwimContainer  # noqa: E402
from swimrs.process.input import build_swim_input  # noqa: E402
from swimrs.process.loop_fast import run_daily_loop_fast  # noqa: E402
from swimrs.swim.config import ProjectConfig  # noqa: E402

SITES_CSV = HERE / "data" / "bushland_sites.csv"
FIELD_YEARS_CSV = HERE / "data" / "bushland_field_years.csv"
PRESCRIBED_IRR = HERE / "data" / "bushland_prescribed_irr.parquet"
LYSIMETER_ET = HERE / "data" / "bushland_lysimeter_et.parquet"
TRANSFER_PARAMS = HERE / "data" / "ex5_cropland_params.json"

# Peak-season months for the daily-timing correlation (maize + soybean, TX).
PEAK_MONTHS = range(5, 10)  # May-Sep


def _load_config() -> ProjectConfig:
    conf = HERE / "9_Bushland.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(HERE.parent))
    return cfg


def run_forward(cfg, container, fids, params_by_fid, prescribed_irr_path=None):
    """Forward-run one configuration; return {fid: DataFrame(et_act, irr_sim, ...)}."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        temp_h5 = tmp.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(params_by_fid, tmp)
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
            mask_mode=getattr(cfg, "mask_mode", "none"),
            transpiration_cover_scaling=getattr(cfg, "transpiration_cover_scaling", True),
            prescribed_irr_path=prescribed_irr_path,
        )
        output, _ = run_daily_loop_fast(si)
        dates = pd.date_range(si.start_date, periods=si.n_days, freq="D")
        etr = si.get_time_series("etr")  # etref[:, i]
        results = {}
        for i, fid in enumerate(si.fids):
            results[fid] = pd.DataFrame(
                {
                    "et_act": output.eta[:, i],
                    "etf_model": output.etf[:, i],
                    "irr_sim": output.irr_sim[:, i],
                    "etref": etr[:, i],
                },
                index=dates,
            )
        si.close()
    finally:
        for p in (temp_h5, params_json):
            if os.path.exists(p):
                os.remove(p)
    return results


def openet_eta(container, fid, etref):
    """OpenET RS-ETa reference: mean over members of etf.interp(linear) x etref."""
    etf_by_model = load_openet_etf_nomask(container, fid)
    if not etf_by_model:
        return pd.Series(dtype=float)
    frames = []
    for _model, series in etf_by_model.items():
        et = series.interpolate(method="linear").reindex(etref.index) * etref
        frames.append(et)
    return pd.concat(frames, axis=1).mean(axis=1, skipna=True)


def _metrics(obs, mod):
    """n, obs/mod sum, ratio, bias, rmse, Pearson r over finite paired days."""
    m = np.isfinite(obs) & np.isfinite(mod)
    o, p = obs[m], mod[m]
    if len(o) < 3:
        return dict(n=int(len(o)), obs_mm=np.nan, mod_mm=np.nan, ratio=np.nan,
                    bias=np.nan, rmse=np.nan, r=np.nan)  # fmt: skip
    return dict(
        n=int(len(o)),
        obs_mm=float(o.sum()),
        mod_mm=float(p.sum()),
        ratio=float(p.sum() / o.sum()) if o.sum() else np.nan,
        bias=float((p - o).mean()),
        rmse=float(np.sqrt(((p - o) ** 2).mean())),
        r=float(pearsonr(o, p)[0]) if len(o) > 3 else np.nan,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--container", required=True, help="Forward-run container (.swim)")
    ap.add_argument("--out-dir", default=str(HERE / "results"), help="Output directory")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg = _load_config()
    sites = pd.read_csv(SITES_CSV)
    fids = sites["site_id"].astype(str).tolist()
    field_years = pd.read_csv(FIELD_YEARS_CSV)

    with open(TRANSFER_PARAMS) as f:
        vector = json.load(f)
    params_by_fid = {fid: dict(vector) for fid in fids}

    lys_et = pd.read_parquet(LYSIMETER_ET)
    lys_et.index = pd.to_datetime(lys_et.index)
    metered = pd.read_parquet(PRESCRIBED_IRR)
    metered.index = pd.to_datetime(metered.index)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "et_figures"
    if not args.no_figures:
        fig_dir.mkdir(exist_ok=True)

    container = SwimContainer.open(args.container, mode="r")
    try:
        print("Forward run (i): internal scheduler ...")
        sched = run_forward(cfg, container, fids, params_by_fid, prescribed_irr_path=None)
        print("Forward run (ii): prescribed metered irrigation ...")
        presc = run_forward(
            cfg, container, fids, params_by_fid, prescribed_irr_path=str(PRESCRIBED_IRR)
        )
        openet = {fid: openet_eta(container, fid, sched[fid]["etref"]) for fid in fids}
    finally:
        container.close()

    daily_rows = []
    summary_rows = []
    for _, fy in field_years.iterrows():
        fid, year, crop = str(fy["site_id"]), int(fy["year"]), fy["crop"]
        if fid not in sched:
            print(f"  {fid} {year}: not in container, skipping")
            continue
        yr = slice(f"{year}-01-01", f"{year}-12-31")
        d = pd.DataFrame(
            {
                "lys_et": lys_et[fid].reindex(sched[fid].loc[yr].index),
                "et_sched": sched[fid].loc[yr, "et_act"],
                "et_presc": presc[fid].loc[yr, "et_act"],
                "et_openet": openet[fid].reindex(sched[fid].loc[yr].index),
                "irr_sched": sched[fid].loc[yr, "irr_sim"],
                "irr_presc": presc[fid].loc[yr, "irr_sim"],
                "irr_metered": metered[fid].reindex(sched[fid].loc[yr].index),
                "eto": sched[fid].loc[yr, "etref"],
            }
        )
        d.insert(0, "crop", crop)
        d.insert(0, "year", year)
        d.insert(0, "site_id", fid)
        daily_rows.append(d.reset_index().rename(columns={"index": "date"}))

        # metrics over days with lysimeter ET present (annual) and peak season
        obs = d["lys_et"].to_numpy()
        peak = d.index.month.isin(PEAK_MONTHS)
        m_sched = _metrics(obs, d["et_sched"].to_numpy())
        m_presc = _metrics(obs, d["et_presc"].to_numpy())
        m_openet = _metrics(obs, d["et_openet"].to_numpy())
        m_sched_pk = _metrics(obs[peak], d["et_sched"].to_numpy()[peak])
        m_presc_pk = _metrics(obs[peak], d["et_presc"].to_numpy()[peak])

        irr_metered_mm = float(d["irr_metered"].sum())
        irr_sched_mm = float(d["irr_sched"].sum())
        summary_rows.append(
            {
                "site_id": fid,
                "year": year,
                "crop": crop,
                "n_obs_days": m_sched["n"],
                "lys_et_mm": round(m_sched["obs_mm"], 1),
                "et_sched_mm": round(m_sched["mod_mm"], 1),
                "et_presc_mm": round(m_presc["mod_mm"], 1),
                "et_openet_mm": round(m_openet["mod_mm"], 1),
                "irr_metered_mm": round(irr_metered_mm, 1),
                "irr_sched_mm": round(irr_sched_mm, 1),
                "overapp_mm": round(irr_sched_mm - irr_metered_mm, 1),
                "presc_ratio": round(m_presc["ratio"], 3),
                "sched_ratio": round(m_sched["ratio"], 3),
                "openet_ratio": round(m_openet["ratio"], 3),
                "presc_bias": round(m_presc["bias"], 3),
                "sched_bias": round(m_sched["bias"], 3),
                "presc_rmse": round(m_presc["rmse"], 3),
                "sched_rmse": round(m_sched["rmse"], 3),
                "presc_r_peak": round(m_presc_pk["r"], 3),
                "sched_r_peak": round(m_sched_pk["r"], 3),
            }
        )

        if not args.no_figures:
            _figure(fig_dir, fid, year, crop, d)

    daily = pd.concat(daily_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(["site_id", "year"])
    daily.to_csv(out_dir / "bushland_prescribed_daily.csv", index=False)
    summary.to_csv(out_dir / "bushland_prescribed_summary.csv", index=False)

    if not args.no_figures:
        _summary_figure(fig_dir, summary)
    _report(summary)
    print(f"\nwrote {out_dir / 'bushland_prescribed_daily.csv'}")
    print(f"wrote {out_dir / 'bushland_prescribed_summary.csv'}")
    if not args.no_figures:
        print(f"wrote per-field-year figures -> {fig_dir}")


def _figure(fig_dir, fid, year, crop, d):
    """Two-panel: daily ET (lysimeter vs scheduler vs prescribed vs OpenET) + irrigation."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True, height_ratios=[2, 1])
    ax1.plot(d.index, d["lys_et"], lw=1.4, color="k", label="lysimeter ET (measured)")
    ax1.plot(d.index, d["et_presc"], lw=1.0, color="tab:green", label="model ET (prescribed irr)")
    ax1.plot(
        d.index, d["et_sched"], lw=1.0, color="tab:red", alpha=0.8, label="model ET (scheduler)"
    )
    ax1.plot(d.index, d["et_openet"], lw=0.8, color="tab:blue", alpha=0.6, label="OpenET RS-ETa")
    ax1.set_ylabel("ET (mm/day)")
    ax1.set_title(f"Bushland {fid} {year} {crop} -- prescribed vs scheduler ET")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.bar(d.index, d["irr_metered"], width=2.0, color="tab:green", label="metered irrigation")
    ax2.bar(
        d.index,
        -d["irr_sched"],
        width=2.0,
        color="tab:red",
        alpha=0.7,
        label="scheduler irrigation",
    )
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("irrigation (mm/day)\nmetered up / scheduler down")
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{fid}_{year}_{crop}.png", dpi=110)
    plt.close(fig)


def _summary_figure(fig_dir, summary):
    """Two-panel cohort summary: crop-year ET totals and irrigation totals."""
    s = summary.reset_index(drop=True)
    labels = [f"{r.site_id}\n{r.year}\n{r.crop[:4]}" for r in s.itertuples()]
    x = np.arange(len(s))
    w = 0.2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax1.bar(x - 1.5 * w, s["lys_et_mm"], w, color="k", label="lysimeter ET (measured)")
    ax1.bar(x - 0.5 * w, s["et_openet_mm"], w, color="tab:blue", alpha=0.7, label="OpenET RS-ETa")
    ax1.bar(
        x + 0.5 * w, s["et_sched_mm"], w, color="tab:red", alpha=0.8, label="model ET (scheduler)"
    )
    ax1.bar(x + 1.5 * w, s["et_presc_mm"], w, color="tab:green", label="model ET (prescribed irr)")
    ax1.set_ylabel("season ET (mm)")
    ax1.set_title("Bushland WP-B2: does prescribing metered irrigation reproduce lysimeter ET?")
    ax1.legend(loc="lower right", fontsize=8, ncol=2)

    ax2.bar(
        x - 0.5 * w,
        s["irr_metered_mm"],
        2 * w,
        color="tab:green",
        label="metered irrigation (real)",
    )
    ax2.bar(
        x + 0.5 * w,
        s["irr_sched_mm"],
        2 * w,
        color="tab:red",
        alpha=0.8,
        label="scheduler irrigation",
    )
    ax2.set_ylabel("season irrigation (mm)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "_cohort_summary.png", dpi=120)
    plt.close(fig)


def _report(summary):
    pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 60)
    print("\n=== Per crop-year: prescribed vs scheduler ET vs lysimeter ET ===")
    cols = [
        "site_id", "year", "crop", "n_obs_days", "lys_et_mm", "et_presc_mm", "et_sched_mm",
        "et_openet_mm", "irr_metered_mm", "irr_sched_mm", "overapp_mm",
        "presc_ratio", "sched_ratio", "presc_bias", "sched_bias", "presc_r_peak", "sched_r_peak",
    ]  # fmt: skip
    print(summary[cols].to_string(index=False))

    print("\n=== Median across 12 field-years ===")
    med = summary[
        ["irr_metered_mm", "irr_sched_mm", "overapp_mm", "presc_ratio", "sched_ratio",
         "presc_bias", "sched_bias", "presc_rmse", "sched_rmse", "presc_r_peak", "sched_r_peak"]
    ].median()  # fmt: skip
    print(med.round(3).to_string())

    total_metered = summary["irr_metered_mm"].sum()
    total_sched = summary["irr_sched_mm"].sum()
    print("\n=== Over-application quantum (sum over 12 field-years) ===")
    print(f"  metered irrigation total : {total_metered:.0f} mm")
    print(f"  scheduler irrigation total: {total_sched:.0f} mm")
    print(f"  scheduler - metered       : {total_sched - total_metered:+.0f} mm "
          f"({100 * (total_sched - total_metered) / total_metered:+.0f}% vs metered)")  # fmt: skip


if __name__ == "__main__":
    main()
