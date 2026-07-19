"""Soil-moisture evaluation for Example 8: modeled theta_avail vs in-situ SCAN theta.

Runs the calibrated SCAN model forward (loop exports daily daw3 + zr) and converts
the kernel's total plant-available water to a volumetric content redistributed over
the MAX rooting depth — the like-for-like quantity established in the Mead first
pass (examples/5_Flux_Ensemble/notes/soil_moisture_mead/):

    soil_water  = awc*zr - depl_root + daw3          # available water (mm) to zr_max
    theta_avail = soil_water / (zr_max * 1000)       # m3/m3, available-water content

theta_avail is above-wilting-point content; observed SCAN theta is total VWC, so the
two differ by a ~constant wilting-point offset that does not affect correlation.
Comparison is in scale-invariant space (Pearson, Spearman, anomaly-r after removing
the day-of-year climatology), which sidesteps sensor calibration offsets. In-situ
theta is used ONLY here at evaluation — never as a model input.

Observations are paired to the model layer they physically represent (see PAIRS):
the root-zone bucket (theta_avail, integrated over zr_max) is scored against a
DEPTH-WEIGHTED SCAN profile mean (not the unweighted mean, which over-weights the
shallow high-variance sensors a bulk average cannot track), and the surface evap
layer (surface_sm_proxy = -depl_ze) is scored against the 5 cm sensor — the SMAP
analog. The legacy unweighted-mean and single-deep-sensor pairings are kept for
reference so the correction is auditable.

Observed theta comes from the screened SCAN archive (validation-only):
    /data/ssd1/swim/soil_moisture/scan/SCAN_<station>.parquet
    schema: date, soil_vwc_{5,10,20,50,101}, profile_mean_theta
joined to model site_id via examples/8_Soil_Moisture/data/scan_sites.csv.

Validation: the rerun's depl_root is checked against the container's stored
rz_depletion where available (bit-for-bit sanity, mirroring run_theta_mead.py).

    uv run python examples/8_Soil_Moisture/evaluate.py \
        --container /data/ssd1/swim/8_Soil_Moisture/data/8_Soil_Moisture_e8cal.swim \
        --par-csv /data/ssd1/swim/8_Soil_Moisture/results/e8cal/8_Soil_Moisture.3.par.csv
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
E5 = HERE.parent / "5_Flux_Ensemble"
if str(E5) not in sys.path:
    sys.path.insert(0, str(E5))

from evaluate import parse_pest_params  # noqa: E402  (Example 5 helper)

from swimrs.container import SwimContainer  # noqa: E402
from swimrs.process.input import build_swim_input  # noqa: E402
from swimrs.process.loop_fast import run_daily_loop_fast  # noqa: E402
from swimrs.swim.config import ProjectConfig  # noqa: E402

SITES_CSV = HERE / "data" / "scan_sites.csv"
DIAG_CSV = HERE / "notes" / "paper" / "data" / "e8cal_dynamic_range_diag.csv"
GROW_MONTHS = range(4, 11)  # Apr-Oct

# Explicit like-for-like (model_var, obs_var, label) pairings. The root-zone bucket
# (theta_avail, a depth-integrated average over zr_max) is compared against a
# DEPTH-WEIGHTED SCAN profile mean, not the unweighted mean that over-weights the
# shallow high-variance sensors. The surface evap layer (surface_sm_proxy = -depl_ze)
# is compared against the shallowest sensor (5 cm) — the SMAP analog. The legacy
# unweighted-mean and single deep-sensor pairings are retained for reference.
PAIRS = [
    ("theta_avail", "rootzone_theta", "rootzone depth-wtd"),
    ("surface_sm_proxy", "soil_vwc_5", "surface 5cm (SMAP-analog)"),
    ("theta_avail", "profile_mean_theta", "legacy unwtd mean"),
    ("theta_avail", "soil_vwc_50", "deep sensor 50cm"),
]


def _load_config(config_path=None) -> ProjectConfig:
    conf = Path(config_path) if config_path else HERE / "8_Soil_Moisture.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(HERE.parent))
    return cfg


def theta_available(awc, zr, depl_root, daw3, zr_max):
    """Plant-available water content redistributed over the max rooting depth (m3/m3).

    awc [mm/m], zr / zr_max [m], depl_root / daw3 [mm]. Scalar or array. The
    available water in the current root zone (awc*zr - depl_root) plus the deep
    below-root store (daw3), spread over zr_max metres of soil -> m3/m3.
    """
    soil_water = awc * zr - depl_root + daw3  # mm available water to zr_max
    return soil_water / (zr_max * 1000.0)


def run_model(cfg, container, par_csv, fids):
    """Forward-run the calibrated model; return {fid: DataFrame(theta_avail, ...)}."""
    calibrated = parse_pest_params(par_csv, fids)
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
            mask_mode=getattr(cfg, "mask_mode", "none"),
            transpiration_cover_scaling=getattr(cfg, "transpiration_cover_scaling", True),
            stress_depletion_fraction=getattr(cfg, "stress_depletion_fraction", None),
        )
        output, _ = run_daily_loop_fast(si)
        dates = pd.date_range(si.start_date, periods=si.n_days, freq="D")
        awc = np.asarray(si.properties.awc, dtype=float)  # mm/m
        zr_max = np.asarray(si.properties.zr_max, dtype=float)  # m
        results = {}
        for i, fid in enumerate(si.fids):
            depl = output.depl_root[:, i]
            daw3 = output.daw3[:, i]
            zr = output.zr[:, i]
            ze = output.depl_ze[:, i]  # surface evap-layer depletion (mm)
            theta_avail = theta_available(awc[i], zr, depl, daw3, zr_max[i])  # m3/m3
            results[fid] = pd.DataFrame(
                {
                    "depl_root": depl,
                    "theta_avail": theta_avail,
                    "sm_proxy": -depl,  # raw first-pass proxy for reference
                    # surface-layer moisture proxy (drier = more depleted); the
                    # SMAP/shallow-sensor analog. Correlation is scale-invariant so
                    # -depl_ze suffices without converting to a VWC.
                    "surface_sm_proxy": -ze,
                },
                index=dates,
            )
        si.close()
    finally:
        for p in (temp_h5, params_json):
            if os.path.exists(p):
                os.remove(p)
    return results


def _read_obs(theta_csv):
    """Read a cleaned SCAN theta parquet/csv; return DataFrame indexed by date."""
    path = theta_csv
    if not os.path.exists(path):
        alt = path.replace(".parquet", ".csv")
        path = alt if os.path.exists(alt) else path
    obs = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    date_col = (
        "date" if "date" in obs.columns else ("datetime" if "datetime" in obs.columns else None)
    )
    if date_col is None:
        raise ValueError(f"No date column in {path}")
    obs[date_col] = pd.to_datetime(obs[date_col])
    return obs.set_index(date_col).sort_index()


def _deseasonalize(s):
    return s - s.groupby(s.index.dayofyear).transform("mean")


def depth_weighted_rootzone(obs, max_depth_cm=101.0):
    """Depth-weighted mean of the soil_vwc_* sensors (midpoint-layer weights).

    Each sensor represents the soil layer bounded by the midpoints to its
    neighbours (deepest layer capped at the deepest sensor — no extrapolation).
    Weighting by layer thickness down-weights the shallow high-variance sensors so
    the observed quantity matches theta_avail's bulk root-zone character, rather
    than the unweighted profile mean which gives a 5 cm probe the same weight as a
    50 cm probe. Per-row missing sensors are renormalized over whatever is present.
    """
    cols, depths = [], []
    for c in obs.columns:
        m = re.fullmatch(r"soil_vwc_(\d+)", c)
        if m:
            cols.append(c)
            depths.append(float(m.group(1)))
    if not cols:
        return pd.Series(np.nan, index=obs.index)
    order = np.argsort(depths)
    depths = np.asarray(depths)[order]
    cols = [cols[i] for i in order]
    bounds = [0.0]
    for j in range(len(depths) - 1):
        bounds.append(0.5 * (depths[j] + depths[j + 1]))
    bounds.append(min(max_depth_cm, depths[-1]))
    thick = np.array([max(0.0, bounds[j + 1] - bounds[j]) for j in range(len(depths))])
    sub = obs[cols].astype(float)
    present = sub.notna().values
    w = np.where(present, thick[None, :], 0.0)
    wsum = w.sum(axis=1)
    num = np.nansum(np.where(present, sub.values * thick[None, :], 0.0), axis=1)
    out = np.where(wsum > 0.0, num / np.where(wsum > 0.0, wsum, 1.0), np.nan)
    return pd.Series(out, index=obs.index)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--container", required=True, help="Calibrated SCAN container (.swim)")
    ap.add_argument("--par-csv", required=True, help="Calibrated par.csv (…3.par.csv)")
    ap.add_argument("--out-dir", default=str(HERE / "results"), help="Output directory")
    ap.add_argument("--no-figures", action="store_true", help="Skip per-site figures")
    ap.add_argument(
        "--config", default=None, help="Override config TOML (default: 8_Soil_Moisture.toml)"
    )
    args = ap.parse_args()

    cfg = _load_config(args.config)
    sites = pd.read_csv(SITES_CSV)
    fids = sites["site_id"].astype(str).tolist()
    theta_by_fid = dict(zip(sites["site_id"].astype(str), sites["theta_csv"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "theta_figures"
    if not args.no_figures:
        fig_dir.mkdir(exist_ok=True)

    container = SwimContainer.open(args.container, mode="r")
    model = run_model(cfg, container, args.par_csv, fids)

    rows = []
    for fid in fids:
        if fid not in model:
            print(f"  {fid}: no model output (not in container), skipping")
            continue
        try:
            obs = _read_obs(theta_by_fid[fid])
        except FileNotFoundError:
            print(f"  {fid}: observed theta missing, skipping")
            continue

        obs = obs.copy()
        obs["rootzone_theta"] = depth_weighted_rootzone(obs)
        mdf = model[fid]

        for mcol, ycol, _lbl in PAIRS:
            if mcol not in mdf.columns or ycol not in obs.columns:
                continue
            df = mdf[[mcol]].join(obs[[ycol]], how="inner")
            gs = df[df.index.month.isin(GROW_MONTHS)]
            d = gs[[ycol, mcol]].dropna()
            if len(d) < 30:
                continue
            an_o = _deseasonalize(d[ycol]).dropna()
            an_m = _deseasonalize(d[mcol]).reindex(an_o.index).dropna()
            an_o = an_o.reindex(an_m.index)
            rows.append(
                dict(
                    site_id=fid,
                    obs_var=ycol,
                    model_var=mcol,
                    n=len(d),
                    pearson=round(pearsonr(d[ycol], d[mcol])[0], 3),
                    spearman=round(spearmanr(d[ycol], d[mcol])[0], 3),
                    anom_r=round(pearsonr(an_o, an_m)[0], 3) if len(an_m) > 30 else np.nan,
                    std_ratio=round(d[mcol].std() / d[ycol].std(), 3)
                    if d[ycol].std() > 0
                    else np.nan,
                )
            )

        if not args.no_figures:
            fdf = mdf[["theta_avail"]].join(obs[["rootzone_theta"]], how="inner")
            dd = fdf[fdf.index.month.isin(GROW_MONTHS)][["rootzone_theta", "theta_avail"]].dropna()
            if len(dd) > 30:

                def _nrm(x):
                    return (x - x.min()) / (x.max() - x.min())

                fig, ax = plt.subplots(figsize=(13, 4))
                ax.plot(
                    dd.index,
                    _nrm(dd["rootzone_theta"]),
                    lw=0.6,
                    label="observed root-zone theta (depth-wtd, norm)",
                )
                ax.plot(
                    dd.index,
                    _nrm(dd["theta_avail"]),
                    lw=0.6,
                    alpha=0.8,
                    label="SWIM theta_avail (norm)",
                )
                r = pearsonr(dd["rootzone_theta"], dd["theta_avail"])[0]
                ax.set_title(f"{fid}: n={len(dd)}  Pearson r={r:.2f}")
                ax.legend(loc="upper right", fontsize=8)
                ax.set_ylabel("normalized [0,1]")
                fig.tight_layout()
                fig.savefig(fig_dir / f"{fid}_theta.png", dpi=110)
                plt.close(fig)

    res = pd.DataFrame(rows)
    out_csv = out_dir / "scan_theta_correlations.csv"
    res.to_csv(out_csv, index=False)
    pd.set_option("display.width", 180, "display.max_rows", 400)
    print(f"\n=== modeled vs observed SCAN theta ({len(res)} site-metric rows) ===")
    if not res.empty:
        print(res.to_string(index=False))
        print("\n--- median across sites, by pairing ---")
        for mcol, ycol, lbl in PAIRS:
            sub = res[(res.model_var == mcol) & (res.obs_var == ycol)]
            if sub.empty:
                continue
            med = sub[["pearson", "spearman", "anom_r", "std_ratio"]].median().round(3)
            print(
                f"  [{lbl:<26}] {mcol} vs {ycol}: "
                f"n_sites={len(sub)}  pearson={med.pearson}  "
                f"spearman={med.spearman}  anom_r={med.anom_r}  std_ratio={med.std_ratio}"
            )

        # Irrigation split (authoritative labels from the water-balance algorithm,
        # never an a-priori override). SCAN theta stays validation-only throughout.
        if DIAG_CSV.exists():
            irr = pd.read_csv(DIAG_CSV).set_index("site_id")["irrigated"]
            res2 = res.assign(irrigated=res.site_id.map(irr))
            print("\n--- median by pairing x irrigation (pearson / anom_r / std_ratio) ---")
            for mcol, ycol, lbl in PAIRS:
                sub = res2[(res2.model_var == mcol) & (res2.obs_var == ycol)]
                if sub.empty or sub.irrigated.isna().all():
                    continue
                g = sub.groupby("irrigated")[["pearson", "anom_r", "std_ratio"]].median().round(3)
                print(f"\n  [{lbl}] {mcol} vs {ycol}")
                print(g.to_string().replace("\n", "\n    "))
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
