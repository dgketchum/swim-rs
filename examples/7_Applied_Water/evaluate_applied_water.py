"""Score simulated irrigation against metered applied water (Example 7).

SWIM-RS simulates irrigation internally from the satellite record; the metered
volume is revealed only here, at scoring. For each field we sum simulated daily
``irr_sim`` (mm) per calendar year into a simulated annual applied depth and
compare it to the withheld metered depth in ``metered_truth.csv``.

Two parameter sources (see ``--params-json`` / ``--par-csv``):
  * calibrated  — per-field PEST++ IES posterior (E2 method)
  * transfer    — the fixed E2 cropland median vector, applied with no calibration

Outputs (under --out): per-field-year table, summary metrics, implied-efficiency
distribution, per-crop panel, negative-control check, and a sim-vs-metered scatter.

    uv run python examples/7_Applied_Water/evaluate_applied_water.py --par-csv <par.csv>
    uv run python examples/7_Applied_Water/evaluate_applied_water.py \
        --params-json examples/6_Flux_International/transfer/ex5_cropland_params.json \
        --label transfer
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
E6 = HERE.parent / "6_Flux_International"
if str(E6) not in sys.path:
    sys.path.insert(0, str(E6))

import evaluate as e6  # noqa: E402  (reuse parse_pest_params)

from swimrs.container import open_container  # noqa: E402
from swimrs.process.input import build_swim_input  # noqa: E402
from swimrs.process.loop_fast import run_daily_loop_fast  # noqa: E402
from swimrs.swim.config import ProjectConfig  # noqa: E402

TRUTH = HERE / "data" / "metered_truth.csv"


def _load_config() -> ProjectConfig:
    conf = HERE / "7_Applied_Water.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(HERE.parent))
    return cfg


def _kge(obs: np.ndarray, sim: np.ndarray) -> float:
    m = np.isfinite(obs) & np.isfinite(sim)
    if m.sum() < 3 or np.std(obs[m]) == 0:
        return np.nan
    r = np.corrcoef(obs[m], sim[m])[0, 1]
    alpha = np.std(sim[m]) / np.std(obs[m])
    beta = np.mean(sim[m]) / np.mean(obs[m]) if np.mean(obs[m]) != 0 else np.nan
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def _metrics(df: pd.DataFrame) -> dict:
    """df has columns metered_depth_mm (obs) and sim_applied_mm (sim)."""
    obs = df.metered_depth_mm.to_numpy(float)
    sim = df.sim_applied_mm.to_numpy(float)
    m = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[m], sim[m]
    if len(obs) < 3:
        return {"n": len(obs)}
    bias = sim.mean() - obs.mean()
    ss_res = float(np.sum((obs - sim) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    return {
        "n": int(len(obs)),
        "obs_mean_mm": round(float(obs.mean()), 1),
        "sim_mean_mm": round(float(sim.mean()), 1),
        "bias_mm": round(float(bias), 1),
        "bias_pct": round(float(100 * bias / obs.mean()), 1),
        "rmse_mm": round(float(np.sqrt(np.mean((obs - sim) ** 2))), 1),
        "r2": round(1 - ss_res / ss_tot if ss_tot > 0 else np.nan, 3),
        "kge": round(_kge(obs, sim), 3),
        "eff_median": round(float(np.median(sim / np.where(obs > 0, obs, np.nan))), 3),
    }


def run_applied(cfg, container, fids, params) -> dict:
    """Forward run; return {fid: DataFrame(index=date, [irr_sim, et_act, etref])}."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        temp_h5 = tmp.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(params, tmp)
        params_json = tmp.name
    try:
        swim_input = build_swim_input(
            container,
            output_h5=temp_h5,
            calibrated_params_path=params_json,
            start_date=cfg.start_dt,
            end_date=cfg.end_dt,
            refet_type=getattr(cfg, "refet_type", "eto") or "eto",
            etf_model=getattr(cfg, "etf_target_model", "ensemble"),
            met_source=getattr(cfg, "met_source", "gridmet"),
            fields=fids,
            empirical_kc_max=True,
            mask_mode=getattr(cfg, "mask_mode", "none"),
        )
        output, _ = run_daily_loop_fast(swim_input)
        dates = pd.date_range(swim_input.start_date, periods=swim_input.n_days, freq="D")
        eto = swim_input.get_time_series("eto")
        results = {}
        for i, fid in enumerate(swim_input.fids):
            results[fid] = pd.DataFrame(
                {
                    "irr_sim": output.irr_sim[:, i],
                    "et_act": output.eta[:, i],
                    "etref": eto[:, i],
                },
                index=dates,
            )
        swim_input.close()
    finally:
        for p in (temp_h5, params_json):
            if os.path.exists(p):
                os.remove(p)
    return results


def _resolve_params(container, fids, par_csv, params_json):
    if params_json:
        vec = json.load(open(params_json))
        # a flat {param: value} vector -> apply unchanged to every field
        if all(not isinstance(v, dict) for v in vec.values()):
            return {fid: dict(vec) for fid in fids}
        return {fid: vec[fid] for fid in fids if fid in vec}
    if par_csv:
        return e6.parse_pest_params(par_csv, fids)
    raise SystemExit("Provide --par-csv or --params-json.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--container", default=None)
    ap.add_argument("--par-csv", default=None)
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--label", default=None, help="calibrated|transfer (names outputs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = _load_config()
    label = args.label or ("transfer" if args.params_json else "calibrated")
    container_path = args.container or os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")
    out_dir = Path(args.out or (Path(cfg.project_ws) / "results" / f"applied_{label}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = pd.read_csv(TRUTH)
    # basin/crop live in the fields shapefile, not the truth table
    fields_gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")
    meta = fields_gdf.drop_duplicates("site_id").set_index("site_id")
    container = open_container(container_path, mode="r")
    fids = [f for f in container.field_uids if f in set(truth.site_id)]

    params = _resolve_params(container, fids, args.par_csv, args.params_json)
    series = run_applied(cfg, container, list(params), params)

    # annual simulated applied depth (+ ET) per field-year
    rows = []
    for fid, df in series.items():
        ann = df.groupby(df.index.year).agg(
            sim_applied_mm=("irr_sim", "sum"), sim_et_mm=("et_act", "sum")
        )
        for yr, r in ann.iterrows():
            rows.append(
                {
                    "site_id": fid,
                    "year": int(yr),
                    "sim_applied_mm": float(r.sim_applied_mm),
                    "sim_et_mm": float(r.sim_et_mm),
                }
            )
    sim = pd.DataFrame(rows)

    # ---- metered fields: join on (site_id, year) ----
    metered = truth[truth.metered_depth_mm > 0]
    paired = metered.merge(sim, on=["site_id", "year"], how="inner")
    paired["basin"] = paired.site_id.map(meta.basin) if "basin" in meta else ""
    paired["crop"] = paired.site_id.map(meta.crop) if "crop" in meta else ""
    paired["efficiency"] = paired.sim_applied_mm / paired.metered_depth_mm.replace(0, np.nan)
    paired.to_csv(out_dir / "per_field_year.csv", index=False)

    summary = {"all": _metrics(paired)}
    for b, sub in paired.groupby("basin"):
        summary[f"basin:{b}"] = _metrics(sub)
    for c, sub in paired.groupby("crop"):
        if len(sub) >= 5:
            summary[f"crop:{c}"] = _metrics(sub)
    # field-aggregated (mean over years per field)
    fa = (
        paired.groupby("site_id")
        .agg(
            metered_depth_mm=("metered_depth_mm", "mean"), sim_applied_mm=("sim_applied_mm", "mean")
        )
        .reset_index()
    )
    summary["field_aggregated"] = _metrics(fa)
    pd.DataFrame(summary).T.to_csv(out_dir / "summary_metrics.csv")

    # ---- negative controls: simulated irrigation must be ~0 ----
    controls = truth[truth.source == "ESPA_rainfed_control"].site_id.unique()
    csim = sim[sim.site_id.isin(controls)]
    ctl = {
        "n_control_fields": int(len(controls)),
        "n_control_field_years": int(len(csim)),
        "sim_applied_mm_mean": round(float(csim.sim_applied_mm.mean()) if len(csim) else 0.0, 2),
        "sim_applied_mm_max": round(float(csim.sim_applied_mm.max()) if len(csim) else 0.0, 2),
        "frac_years_gt_10mm": round(
            float((csim.sim_applied_mm > 10).mean()) if len(csim) else 0.0, 3
        ),
    }
    (out_dir / "negative_controls.json").write_text(json.dumps(ctl, indent=2))

    print(f"[{label}] paired field-years: {len(paired)}")
    print("all:", summary["all"])
    print("controls:", ctl)

    _scatter(paired, out_dir / "sim_vs_metered.png", label)
    print("wrote", out_dir)


def _scatter(paired: pd.DataFrame, path: Path, label: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    for b, sub in paired.groupby("basin"):
        ax.scatter(sub.metered_depth_mm, sub.sim_applied_mm, s=14, alpha=0.5, label=str(b))
    lim = [0, float(np.nanmax([paired.metered_depth_mm.max(), paired.sim_applied_mm.max()])) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Metered applied depth (mm/yr)")
    ax.set_ylabel("Simulated applied depth (mm/yr)")
    ax.set_title(f"SWIM-RS irrigation vs metered ({label})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
