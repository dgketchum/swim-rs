"""
Figure 3: Daily ET Time Series at Representative Sites

Four panels showing SWIM ET, SSEBop ET, and flux tower ET at:
  (a) US-Ne1 — irrigated cropland
  (b) US-ARM — grassland
  (c) US-NC3 — evergreen forest (managed loblolly pine)
  (d) US-CMW — wetland/riparian (SWIM failure case)

Usage:
    python paper/figures/fig3_timeseries.py
"""

import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
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

# Paths
DATA_DIR = "/data/ssd1/swim/4_Flux_Network/data"
CONTAINER_PATH = os.path.join(DATA_DIR, "4_Flux_Network.swim")
PAR_CSV = "/data/ssd1/swim/4_Flux_Network/results/4_Flux_Network.3.par.csv"
FLUX_DIR = os.path.join(DATA_DIR, "daily_flux_files")

OUT_DIR = Path(__file__).resolve().parent
OUT_PNG = OUT_DIR / "fig3_timeseries.png"
OUT_PDF = OUT_DIR / "fig3_timeseries.pdf"

IRR_THRESHOLD = 0.3

SITES = [
    {"fid": "US-Ne1", "label": "US-Ne1 — Irrigated Cropland", "years": (2007, 2013)},
    {"fid": "US-ARM", "label": "US-ARM — Grassland", "years": (2007, 2013)},
    {"fid": "US-NC3", "label": "US-NC3 — Evergreen Forest", "years": (2014, 2019)},
    {"fid": "US-CMW", "label": "US-CMW — Wetland/Riparian", "years": (2007, 2013)},
]


def load_config():
    project_dir = Path(__file__).resolve().parents[2] / "examples" / "4_Flux_Network"
    conf = project_dir / "4_Flux_Network.toml"
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
        etr = swim_input.get_time_series("etr")

        results = {}
        for i, fid in enumerate(swim_input.fids):
            results[fid] = pd.DataFrame(
                {"et_act": output.eta[:, i], "etref": etr[:, i]},
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


def load_ssebop_et(container, fid, etref):
    """Load SSEBop ETf using year-by-year mask switching matching the Ex4 evaluator.

    Default to inv_irr; switch to irr for years where irrigation fraction >= 0.3.
    This replicates the logic in evaluate.py:load_ssebop_etf().
    """
    import json

    irr_threshold = 0.3

    # Determine irrigated years from container dynamics
    irr_years = set()
    try:
        gwsub_path = "derived/dynamics/irr_data"
        irr_arr = container._root[gwsub_path]
        uids = container._root["geometry/uid"][:]
        idx = list(uids).index(fid) if fid in uids else -1
        if idx >= 0:
            raw = irr_arr[idx]
            if hasattr(raw, "item"):
                raw = raw.item()
            if raw:
                irr_data = json.loads(raw)
                for k, v in irr_data.items():
                    if k == "fallow_years":
                        continue
                    if isinstance(v, dict) and v.get("f_irr", 0.0) >= irr_threshold:
                        irr_years.add(int(k))
    except Exception:
        pass

    # Load both masks
    etf_inv = etf_irr = None
    for mask in ["inv_irr", "irr"]:
        etf_path = f"remote_sensing/etf/landsat/ssebop/{mask}"
        try:
            etf_df = container.query.dataframe(etf_path, fields=[fid])
            if fid in etf_df.columns and etf_df[fid].notna().any():
                if mask == "inv_irr":
                    etf_inv = etf_df[fid]
                else:
                    etf_irr = etf_df[fid]
        except Exception:
            pass

    if etf_inv is None and etf_irr is None:
        return pd.Series(dtype=float)

    # Composite: inv_irr default, irr for irrigated years
    combined = (
        etf_inv.copy() if etf_inv is not None else pd.Series(float("nan"), index=etf_irr.index)
    )
    if etf_irr is not None and irr_years:
        irr_mask = combined.index.year.isin(irr_years)
        combined.loc[irr_mask] = etf_irr.loc[irr_mask]
    if etf_inv is None and etf_irr is not None:
        combined = etf_irr.copy()

    etf_daily = combined.reindex(etref.index).interpolate(method="linear")
    return (etf_daily * etref).clip(lower=0)


def main():
    fids = [s["fid"] for s in SITES]
    print("Loading config and container...")
    cfg = load_config()
    container = SwimContainer.open(CONTAINER_PATH, mode="r")

    print(f"Parsing calibrated parameters for {len(fids)} sites...")
    cal_params = parse_pest_params(PAR_CSV, fids)

    print("Running SWIM forward model...")
    swim_out = run_swim(cfg, container, fids, cal_params)

    fig, _ = plt.subplots(4, 1, figsize=(12, 12), sharex=False)
    axes = fig.axes
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for idx, (site, label) in enumerate(zip(SITES, panel_labels)):
        ax = axes[idx]
        fid = site["fid"]
        y0, y1 = site["years"]

        flux_et = load_flux_et(fid)
        swim_df = swim_out.get(fid)
        if swim_df is None:
            ax.set_title(f"{label} {site['label']} — NO SWIM DATA")
            continue

        swim_et = swim_df["et_act"]
        etref = swim_df["etref"]
        ssebop_et = load_ssebop_et(container, fid, etref)

        # Window to plotting years
        date_mask = slice(f"{y0}-01-01", f"{y1}-12-31")
        swim_w = swim_et.loc[date_mask]
        flux_w = flux_et.loc[date_mask] if len(flux_et) > 0 else pd.Series(dtype=float)
        ssebop_w = ssebop_et.loc[date_mask] if len(ssebop_et) > 0 else pd.Series(dtype=float)

        # 7-day rolling mean for smoother lines
        swim_smooth = swim_w.rolling(7, center=True, min_periods=3).mean()
        ssebop_smooth = ssebop_w.rolling(7, center=True, min_periods=3).mean()

        ax.plot(swim_smooth.index, swim_smooth.values, color="#4C72B0", lw=1.2, label="SWIM")
        ax.plot(ssebop_smooth.index, ssebop_smooth.values, color="#DD8452", lw=1.0, label="SSEBop")

        if len(flux_w) > 0:
            # Subsample flux points for readability
            flux_valid = flux_w.dropna()
            ax.scatter(
                flux_valid.index,
                flux_valid.values,
                s=3,
                alpha=0.4,
                color="black",
                label="Flux Tower",
                zorder=1,
            )

        ax.set_ylabel("ET (mm/day)")
        ax.set_ylim(bottom=-0.5)
        ax.set_title(f"{label} {site['label']}", loc="left", fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9, markerscale=3)

    axes[-1].set_xlabel("Date")
    fig.tight_layout()

    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")

    container.close()


if __name__ == "__main__":
    main()
