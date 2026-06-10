"""
Figure 4: Daily ET Accuracy Summary

(a) SWIM ET vs flux tower ET — pooled scatter, 45-site matched cropland cohort
(b) OpenET ensemble ET vs flux tower ET — same sites
(c) Paired site-level R² delta histogram (SWIM minus ensemble)

Usage:
    python paper/figures/fig4_accuracy.py
"""

import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

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

OUT_DIR = Path(__file__).resolve().parent
OUT_PNG = OUT_DIR / "fig4_accuracy.png"
OUT_PDF = OUT_DIR / "fig4_accuracy.pdf"


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
            results[fid] = pd.DataFrame(
                {"et_act": output.eta[:, i]},
                index=dates,
            )
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
    """Load Volk OpenET ensemble daily ET, interpolated to daily."""
    path = os.path.join(VOLK_DIR, f"{fid}.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col="DATE", parse_dates=True)
    if "ensemble_mean_3x3" in df.columns:
        return df["ensemble_mean_3x3"].astype(float)
    return pd.Series(dtype=float)


def collect_daily_pairs(swim_out, fids):
    """Collect matched daily (flux, swim, ensemble) triplets across all sites."""
    records = []
    site_metrics = []

    for fid in fids:
        flux_et = load_flux_et(fid)
        swim_df = swim_out.get(fid)
        ens_et = load_volk_ensemble_et(fid)

        if swim_df is None or len(flux_et) == 0:
            continue

        swim_et = swim_df["et_act"]

        # Interpolate ensemble to daily for fair comparison
        if len(ens_et) > 0:
            ens_daily = ens_et.reindex(swim_et.index).interpolate(method="linear")
        else:
            ens_daily = pd.Series(dtype=float)

        # Match on common dates with valid flux data
        common = flux_et.index.intersection(swim_et.index)
        if len(common) < 10:
            continue

        f = flux_et.loc[common].values
        s = swim_et.loc[common].values
        valid = np.isfinite(f) & np.isfinite(s) & (f >= 0)

        for i, dt in enumerate(common):
            if not valid[i]:
                continue
            e_val = ens_daily.get(dt, np.nan) if len(ens_daily) > 0 else np.nan
            records.append({"fid": fid, "date": dt, "flux": f[i], "swim": s[i], "ensemble": e_val})

        # Per-site metrics — strictly paired: flux, swim, AND ensemble all finite
        if len(ens_daily) > 0:
            ev = ens_daily.reindex(common).values
            paired = valid & np.isfinite(ev)
            if paired.sum() >= 10:
                swim_r2 = r2_score(f[paired], s[paired])
                ens_r2 = r2_score(f[paired], ev[paired])
            else:
                swim_r2 = np.nan
                ens_r2 = np.nan
        else:
            swim_r2 = r2_score(f[valid], s[valid]) if valid.sum() >= 10 else np.nan
            ens_r2 = np.nan

        site_metrics.append({"fid": fid, "swim_r2": swim_r2, "ens_r2": ens_r2})

    return pd.DataFrame(records), pd.DataFrame(site_metrics)


def plot_scatter(ax, obs, mod, color, label, panel_label):
    """Density scatter with 1:1 line and annotated metrics."""
    mask = np.isfinite(obs) & np.isfinite(mod)
    x, y = obs[mask], mod[mask]

    cmap = "Blues" if "SWIM" in label else "Oranges"
    ax.hexbin(x, y, gridsize=60, cmap=cmap, mincnt=1, bins="log")
    lim = 10.0
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Flux Tower ET (mm/day)")
    ax.set_ylabel(f"{label} ET (mm/day)")

    r2 = r2_score(x, y)
    rmse = np.sqrt(mean_squared_error(x, y))
    bias = float((y - x).mean())
    r, _ = stats.pearsonr(x, y)

    textstr = f"n = {len(x):,}\nR² = {r2:.3f}\nRMSE = {rmse:.2f}\nBias = {bias:+.2f}"
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_title(f"{panel_label} {label}", loc="left", fontweight="bold")


def plot_delta_hist(ax, site_metrics):
    """Histogram of per-site R² deltas (SWIM minus ensemble)."""
    matched = site_metrics.dropna(subset=["swim_r2", "ens_r2"])
    delta = matched["swim_r2"] - matched["ens_r2"]

    ax.hist(delta, bins=20, color="#4C72B0", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.axvline(
        delta.median(), color="#DD8452", ls="-", lw=1.5, label=f"Median = {delta.median():+.3f}"
    )

    wins = (delta > 0).sum()
    total = len(delta)
    ax.set_xlabel("R² Delta (SWIM - Ensemble)")
    ax.set_ylabel("Number of Sites")
    ax.set_title("(c) Site-Level R² Advantage", loc="left", fontweight="bold")

    textstr = f"n = {total}\nSWIM wins: {wins}/{total} ({100 * wins / total:.0f}%)"
    ax.text(
        0.95,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend(loc="upper left", framealpha=0.9)


def main():
    print("Loading config and container...")
    cfg = load_config()
    container = SwimContainer.open(CONTAINER_PATH, mode="r")

    # Use only the 45-site matched cohort (sites with ensemble in evaluation metrics)
    eval_df = pd.read_csv("/data/ssd1/swim/5_Flux_Ensemble/results/evaluation_metrics.csv")
    matched_eval = eval_df.dropna(subset=["r2_ensemble"])
    fids = matched_eval["fid"].tolist()

    print(f"Parsing calibrated parameters for {len(fids)} sites...")
    cal_params = parse_pest_params(PAR_CSV, fids)

    print("Running SWIM forward model...")
    swim_out = run_swim(cfg, container, fids, cal_params)

    print("Collecting daily paired data...")
    pairs, site_metrics = collect_daily_pairs(swim_out, fids)
    container.close()

    # Filter to matched cohort (sites with ensemble data)
    matched_pairs = pairs.dropna(subset=["ensemble"])
    matched_sites = site_metrics.dropna(subset=["ens_r2"])
    print(f"Matched cohort: {len(matched_sites)} sites, {len(matched_pairs):,} daily pairs")

    # Plot
    fig, axes_arr = plt.subplots(1, 3, figsize=(14, 4.5))
    axes = fig.axes

    plot_scatter(
        axes[0],
        matched_pairs["flux"].values,
        matched_pairs["swim"].values,
        "#4C72B0",
        "SWIM",
        "(a)",
    )

    plot_scatter(
        axes[1],
        matched_pairs["flux"].values,
        matched_pairs["ensemble"].values,
        "#DD8452",
        "OpenET Ensemble",
        "(b)",
    )

    plot_delta_hist(axes[2], matched_sites)

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
