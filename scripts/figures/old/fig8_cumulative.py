"""
Figure 8: Seasonal Cumulative ET and Bias Accumulation

(a) Cumulative ET over representative growing seasons at 3-4 sites,
    lines for SWIM, OpenET ensemble, and flux tower.
(b) Distribution of seasonal bias (mm/season) across the matched cohort
    for SWIM vs ensemble vs individual models.

Usage:
    python paper/figures/fig8_cumulative.py
"""

import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swimrs.container import SwimContainer
from swimrs.process.input import build_swim_input
from swimrs.process.loop_fast import run_daily_loop_fast
from swimrs.swim.config import ProjectConfig

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
    }
)

# Paths — Example 5
DATA_DIR = "/data/ssd1/swim/5_Flux_Ensemble/data"
CONTAINER_PATH = os.path.join(DATA_DIR, "5_Flux_Ensemble_corrected_mad.swim")
PAR_CSV = "/data/ssd1/swim/5_Flux_Ensemble/results/run13_mad_ensemble/5_Flux_Ensemble.3.par.csv"
FLUX_DIR = os.path.join(DATA_DIR, "daily_flux_files")
VOLK_DIR = os.path.join(DATA_DIR, "openet_flux", "daily_data")
EVAL_CSV = "/data/ssd1/swim/5_Flux_Ensemble/results/evaluation_metrics.csv"

OUT_DIR = Path(__file__).resolve().parent
OUT_PNG = OUT_DIR / "fig8_cumulative.png"
OUT_PDF = OUT_DIR / "fig8_cumulative.pdf"

# Representative sites and years for panel (a)
CUMULATIVE_SITES = [
    {"fid": "US-Ne1", "year": 2010, "label": "US-Ne1 — Irrigated Maize"},
    {"fid": "US-Tw3", "year": 2017, "label": "US-Tw3 — Rice Paddy"},
    {"fid": "RIP760", "year": 2018, "label": "RIP760 — Irrigated Cropland"},
]

# Models for panel (b)
MODELS = ["swim", "ensemble", "ptjpl", "sims", "ssebop", "geesebal"]
MODEL_LABELS = {
    "swim": "SWIM",
    "ensemble": "Ensemble",
    "ptjpl": "PT-JPL",
    "sims": "SIMS",
    "ssebop": "SSEBop",
    "geesebal": "geeSEBAL",
}
MODEL_COLORS = {
    "swim": "#4C72B0",
    "ensemble": "#222222",
    "ptjpl": "#DD8452",
    "sims": "#55A868",
    "ssebop": "#CCB974",
    "geesebal": "#64B5CD",
}

SWIM_COLOR = "#4C72B0"
ENS_COLOR = "#DD8452"
FLUX_COLOR = "black"

# Growing season: April 1 - October 31
GS_START_DOY = 91
GS_END_DOY = 304


def load_config():
    project_dir = Path(__file__).resolve().parents[2] / "examples" / "5_Flux_Ensemble"
    conf = project_dir / "5_Flux_Ensemble.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd1/swim"):
        cfg.read_config(str(conf), calibrate=True)
    else:
        cfg.read_config(str(conf), project_root_override=str(project_dir.parent), calibrate=True)
    return cfg


def parse_pest_params(par_csv, fids):
    df = pd.read_csv(par_csv, index_col=0)
    numeric_rows = df.loc[df.index != "base"]
    row = numeric_rows.median()
    params_by_fid = {}
    for col in df.columns:
        parts = col.split("_ptype:")[0].replace("pname:p_", "").rsplit("_:0", 1)[0]
        matched_fid = None
        for fid in fids:
            if parts.lower().endswith(f"_{fid.lower()}"):
                matched_fid = fid
                param_name = parts[: -(len(fid) + 1)]
                break
        if matched_fid:
            if matched_fid not in params_by_fid:
                params_by_fid[matched_fid] = {}
            params_by_fid[matched_fid][param_name] = float(row[col])
    return params_by_fid


def run_swim(cfg, container, fids, calibrated_params):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        temp_h5 = tmp.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(calibrated_params, tmp)
        params_json = tmp.name
    try:
        swim_input = build_swim_input(
            container,
            output_h5=temp_h5,
            calibrated_params_path=params_json,
            start_date=cfg.start_dt,
            end_date=cfg.end_dt,
            refet_type=getattr(cfg, "refet_type", "eto") or "eto",
            fields=fids,
            empirical_kc_max=True,
            mask_mode=getattr(cfg, "mask_mode", "irrigation"),
        )
        output, _ = run_daily_loop_fast(swim_input)
        dates = pd.date_range(swim_input.start_date, periods=swim_input.n_days, freq="D")
        results = {}
        for i, fid in enumerate(swim_input.fids):
            results[fid] = pd.Series(output.eta[:, i], index=dates)
        swim_input.close()
    finally:
        for p in [temp_h5, params_json]:
            if os.path.exists(p):
                os.remove(p)
    return results


def load_flux_et(fid):
    path = os.path.join(FLUX_DIR, f"{fid}_daily_data.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    if "ET_corr" in df.columns:
        return df["ET_corr"]
    return pd.Series(dtype=float)


def load_volk_ensemble_et(fid):
    path = os.path.join(VOLK_DIR, f"{fid}.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col="DATE", parse_dates=True)
    if "ensemble_mean_3x3" in df.columns:
        return df["ensemble_mean_3x3"].astype(float)
    return pd.Series(dtype=float)


def growing_season_slice(year):
    """Return date range for Apr 1 - Oct 31 of a given year."""
    return slice(f"{year}-04-01", f"{year}-10-31")


def annual_median_bias_all_sites(swim_out, fids):
    """Compute per-site annual median bias for SWIM and ensemble.

    For each site, compute full calendar year bias for each year, then take the
    median across years. Returns one row per site (not per site-year).
    """
    records = []
    for fid in fids:
        flux_et = load_flux_et(fid)
        swim_et = swim_out.get(fid)
        ens_et_sparse = load_volk_ensemble_et(fid)

        if swim_et is None or len(flux_et) == 0:
            continue

        # Interpolate ensemble to daily, but only within its actual date range
        if len(ens_et_sparse) > 0:
            ens_valid = ens_et_sparse.dropna()
            ens_min_year = ens_valid.index.min().year
            ens_max_year = ens_valid.index.max().year
            ens_et = ens_et_sparse.reindex(swim_et.index).interpolate(method="linear")
        else:
            ens_et = pd.Series(dtype=float)
            ens_min_year = ens_max_year = None

        swim_yearly_bias = []
        ens_yearly_bias = []

        common_years = sorted(set(flux_et.index.year) & set(swim_et.index.year))
        for yr in common_years:
            yr_slice = slice(f"{yr}-01-01", f"{yr}-12-31")
            f = flux_et.loc[yr_slice].dropna()
            s = swim_et.loc[yr_slice]

            if len(f) < 200:
                continue

            common_dates = f.index.intersection(s.index)
            fv = f.loc[common_dates].values
            sv = s.loc[common_dates].values
            valid = np.isfinite(fv) & np.isfinite(sv)

            if valid.sum() < 200:
                continue

            # Strictly paired: only include years where flux, swim, AND ensemble
            # are all finite on the same days. No unpaired fallback.
            if len(ens_et) > 0 and ens_min_year is not None and ens_min_year <= yr <= ens_max_year:
                paired = valid & np.isfinite(ens_et.reindex(common_dates).values)
                if paired.sum() > 200:
                    swim_yearly_bias.append(sv[paired].sum() - fv[paired].sum())
                    ens_yearly_bias.append(
                        ens_et.reindex(common_dates).values[paired].sum() - fv[paired].sum()
                    )

        if swim_yearly_bias:
            records.append(
                {
                    "fid": fid,
                    "swim_bias": float(np.median(swim_yearly_bias)),
                    "ens_bias": float(np.median(ens_yearly_bias)) if ens_yearly_bias else np.nan,
                }
            )

    return pd.DataFrame(records)


def plot_cumulative(axes, swim_out):
    """Panel (a): cumulative ET curves at representative sites."""
    for idx, site in enumerate(CUMULATIVE_SITES):
        ax = axes[idx]
        fid = site["fid"]
        yr = site["year"]
        gs = growing_season_slice(yr)

        swim_et = swim_out.get(fid)
        flux_et = load_flux_et(fid)
        ens_sparse = load_volk_ensemble_et(fid)

        if swim_et is None:
            ax.set_title(f"{site['label']} — no data")
            continue

        # Get growing season data
        s = swim_et.loc[gs].copy()
        f = flux_et.loc[gs].dropna() if len(flux_et) > 0 else pd.Series(dtype=float)

        if len(ens_sparse) > 0:
            e = ens_sparse.reindex(swim_et.index).interpolate(method="linear").loc[gs]
        else:
            e = pd.Series(dtype=float)

        # Compute cumulative on common valid dates
        if len(f) > 0:
            # Cumulate only on days where all plotted series are finite (no zero-fill)
            f_aligned = f.reindex(s.index)
            if len(e) > 0:
                e_aligned = e.reindex(s.index)
                paired = f_aligned.notna() & s.notna() & e_aligned.notna()
            else:
                e_aligned = pd.Series(dtype=float)
                paired = f_aligned.notna() & s.notna()

            s_cum = s.where(paired).fillna(0).cumsum()
            f_cum = f_aligned.where(paired).fillna(0).cumsum()

            ax.plot(s_cum.index, s_cum.values, color=SWIM_COLOR, lw=1.8, label="SWIM")
            ax.plot(
                f_cum.index, f_cum.values, color=FLUX_COLOR, lw=1.2, ls="--", label="Flux Tower"
            )

            has_ensemble = (
                len(e_aligned) > 0 and paired.any() and e_aligned.loc[paired].notna().any()
            )
            if has_ensemble:
                e_cum = e_aligned.where(paired).fillna(0).cumsum()
                ax.plot(e_cum.index, e_cum.values, color=ENS_COLOR, lw=1.5, label="Ensemble")

            # Annotate final totals
            swim_total = s_cum.iloc[-1]
            flux_total = f_cum.iloc[-1]
            ax.text(
                0.97,
                0.55,
                f"SWIM: {swim_total:.0f} mm",
                transform=ax.transAxes,
                fontsize=8,
                ha="right",
                color=SWIM_COLOR,
                fontweight="bold",
            )
            ax.text(
                0.97,
                0.45,
                f"Flux: {flux_total:.0f} mm",
                transform=ax.transAxes,
                fontsize=8,
                ha="right",
                color=FLUX_COLOR,
            )
            if has_ensemble:
                ens_total = e_cum.iloc[-1]
                ax.text(
                    0.97,
                    0.35,
                    f"Ens: {ens_total:.0f} mm",
                    transform=ax.transAxes,
                    fontsize=8,
                    ha="right",
                    color=ENS_COLOR,
                )

        ax.set_ylabel("Cumulative ET (mm)")
        ax.set_title(f"{site['label']} ({yr})", loc="left", fontweight="bold", fontsize=10)
        import matplotlib.dates as mdates

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", rotation=0)
        if idx == 0:
            ax.legend(loc="upper left", framealpha=0.9, fontsize=8)


def plot_seasonal_bias(ax, bias_df):
    """Panel (b): box plot of annual median bias per site for SWIM vs ensemble."""
    swim_bias = bias_df["swim_bias"].dropna()
    ens_bias = bias_df["ens_bias"].dropna()

    bp = ax.boxplot(
        [swim_bias, ens_bias],
        positions=[0, 1],
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
        medianprops=dict(color="black", lw=1.5),
        whiskerprops=dict(color="gray", lw=1),
        capprops=dict(color="gray", lw=1),
    )

    bp["boxes"][0].set_facecolor(SWIM_COLOR)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(ENS_COLOR)
    bp["boxes"][1].set_alpha(0.7)

    ax.axhline(0, color="black", ls=":", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SWIM", "Ensemble"])
    ax.set_ylabel("Annual Bias (mm/year)")
    ax.set_ylim(top=400)
    ax.set_title("Annual Bias\n(median across years)", loc="left", fontweight="bold", fontsize=10)

    swim_med = swim_bias.median()
    ens_med = ens_bias.median()
    n_sites = len(swim_bias)
    ax.text(
        0.05,
        0.95,
        f"n = {n_sites} sites\nSWIM median: {swim_med:+.0f} mm\nEns median: {ens_med:+.0f} mm",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def main():
    print("Loading config and container...")
    cfg = load_config()
    container = SwimContainer.open(CONTAINER_PATH, mode="r")

    # Use only the monthly matched cohort (same 33 sites as Fig 5)
    monthly_df = pd.read_csv(
        "/data/ssd1/swim/5_Flux_Ensemble/results/evaluation_monthly_metrics.csv"
    )
    matched_monthly = monthly_df.dropna(subset=["r2_ensemble"])
    fids = matched_monthly["fid"].tolist()

    # Only need to run model for cumulative sites + all sites for bias
    print(f"Parsing calibrated parameters for {len(fids)} sites...")
    cal_params = parse_pest_params(PAR_CSV, fids)

    print("Running SWIM forward model...")
    swim_out = run_swim(cfg, container, fids, cal_params)
    container.close()

    print("Computing annual median bias per site...")
    bias_df = annual_median_bias_all_sites(swim_out, fids)
    matched_bias = bias_df.dropna(subset=["ens_bias"])
    print(f"Sites with both SWIM and ensemble bias: {len(matched_bias)}")

    # Layout: 3 cumulative panels + 1 bias panel
    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.8], wspace=0.35)

    cum_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    bias_ax = fig.add_subplot(gs[0, 3])

    plot_cumulative(cum_axes, swim_out)
    plot_seasonal_bias(bias_ax, matched_bias)

    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
