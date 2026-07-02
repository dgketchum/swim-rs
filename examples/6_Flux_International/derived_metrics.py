"""Recompute the derived E3 (Example 6) validation analyses for the manuscript.

These are the sub-analyses that the headline ``evaluate.py`` run does not emit
directly but the results overview reports:

    1. Per-model RS benchmarks  - SWIM vs SSEBop / PT-JPL / Ensemble RS ETa,
       per-site, on the "all days" basis (RS ETf interpolated between captures
       times daily ETo), paired against flux on each model's availability.
    2. CONUS vs ex-CONUS decomposition - per-site KGE, bias ratio (beta),
       variability ratio (alpha), MAE, RMSE; medians within US- vs non-US.
    3. Murphy (1988) MSE decomposition - phase / bias / variance fractions and
       median Pearson r for SWIM, SSEBop-interp, PT-JPL-interp.
    4. Top / bottom sites by SWIM KGE.

All inputs are read from an already-evaluated results directory: the per-site
``{fid}.csv`` files (which carry SWIM ``et_act`` and ``etref``), the container
(for individual-model ETf series), and the QAQC flux archive. No model rerun is
performed, so the SWIM and ensemble-RS numbers reconcile exactly with the
``evaluation_metrics.csv`` produced by ``evaluate.py``.

Usage:
    python derived_metrics.py --config PATH [--results-dir PATH] [--out PATH]
"""

import argparse
import os
import tempfile
from pathlib import Path

import evaluate as ev
import numpy as np
import pandas as pd

import swimrs.process.input as swim_input_mod
from swimrs.container import SwimContainer
from swimrs.process.input import build_swim_input
from swimrs.process.loop_fast import run_daily_loop_fast

# Instrument sort priority for the per-model benchmark table.
_INSTRUMENT_ORDER = {"landsat": 0, "sentinel": 1, "ecostress": 2}


def _mask_key(cfg):
    return "no_mask" if getattr(cfg, "mask_mode", "none") == "none" else "irr"


def _rs_model_specs(container, cfg):
    """RS series to benchmark: every individual constituent ETf series present
    in the container (each under its native instrument) plus the config-selected
    calibration target.

    Individual models are labelled ``"{instrument}/{model}"`` so that, e.g.,
    Landsat PT-JPL and ECOSTRESS PT-JPL stay distinct (they are both ``ptjpl``).
    The ``merged`` instrument is skipped here — its synthetic series (e.g.
    ``triple``) is the calibration target, added as the final ``target`` spec and
    loaded via the same path evaluate.py uses, so the target row reconciles with
    evaluation_metrics.csv. The target label is ``etf_target_model`` (``ensemble``
    for Experiment A, ``triple`` for Experiment B).
    """
    mask = _mask_key(cfg)
    specs = []
    try:
        etf_grp = container._root["remote_sensing/etf"]
    except KeyError:
        etf_grp = None
    if etf_grp is not None:
        items = []
        for instrument in etf_grp:
            if instrument == "merged":
                continue
            inst_grp = etf_grp[instrument]
            for model in inst_grp:
                if mask in inst_grp[model]:
                    items.append((instrument, model))
        items.sort(key=lambda im: (_INSTRUMENT_ORDER.get(im[0], 99), im[1]))
        for instrument, model in items:
            specs.append(
                {
                    "label": f"{instrument}/{model}",
                    "kind": "individual",
                    "instrument": instrument,
                    "model": model,
                }
            )
    target_label = getattr(cfg, "etf_target_model", "ensemble") or "ensemble"
    specs.append({"label": target_label, "kind": "target", "instrument": None, "model": None})
    return specs


def _cohort_fids(cfg, container):
    """Resolve the publication cohort exactly as evaluate.py does."""
    import geopandas as gpd

    gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")
    id_col = cfg.feature_id_col if cfg.feature_id_col in gdf.columns else "sid"
    cohort = [str(s) for s in gdf[id_col].tolist()]
    container_fids = set(container.field_uids)
    return [f for f in cohort if f in container_fids]


def _model_rs_eta(container, cfg, spec, fid, etref):
    """Build daily RS ETa for one RS spec (an individual model or the target)."""
    if spec["kind"] == "target":
        etf_series = ev._load_target_etf_series(container, cfg, fid)
    else:
        path = f"remote_sensing/etf/{spec['instrument']}/{spec['model']}/{_mask_key(cfg)}"
        etf_series = ev._query_etf_series(container, path, fid)

    if etf_series is None:
        return None

    daily_etf = etf_series.reindex(etref.index).interpolate(method="linear")
    rs_eta = daily_etf * etref.reindex(daily_etf.index)
    if not rs_eta.notna().any():
        return None
    return rs_eta


def _murphy_decomp(obs, mod):
    """Murphy (1988) MSE decomposition into bias / variance / phase fractions.

    MSE = (mean_mod - mean_obs)^2            (bias, systematic offset)
        + (std_mod  - std_obs)^2             (variance / amplitude mismatch)
        + 2*std_mod*std_obs*(1 - r)          (phase / lack of correlation)

    Uses population std (ddof=0) so the three components sum exactly to the MSE.
    """
    obs = np.asarray(obs, dtype=float)
    mod = np.asarray(mod, dtype=float)
    mse = float(np.mean((mod - obs) ** 2))
    if mse <= 0:
        return None
    sm, so = float(np.std(mod)), float(np.std(obs))
    r = float(np.corrcoef(obs, mod)[0, 1])
    bias_c = (float(np.mean(mod)) - float(np.mean(obs))) ** 2
    var_c = (sm - so) ** 2
    phase_c = 2.0 * sm * so * (1.0 - r)
    return {
        "r": r,
        "mse": mse,
        "bias_frac": bias_c / mse,
        "var_frac": var_c / mse,
        "phase_frac": phase_c / mse,
    }


def run_uncalibrated_model(cfg, container, fids):
    """Forward run with default (uncalibrated) parameters.

    The container holds ingested calibration, so build_swim_input would normally
    load it even when ``calibrated_params_path`` is None. We temporarily force the
    "no calibration" branch so every site runs with the model's default/initial
    parameters (kc_min=0.15, ndvi_k=10, ndvi_0=0.55/0.15 by irrigation,
    swe_alpha=0.5, swe_beta=2.0, kr/ks_damp=0.2, with soil-derived aw). Returns
    {fid: Series of uncalibrated daily ET}.
    """
    orig = swim_input_mod._container_has_calibration
    swim_input_mod._container_has_calibration = lambda c: False
    temp_h5 = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            temp_h5 = tmp.name
        swim_input = build_swim_input(
            container,
            output_h5=temp_h5,
            calibrated_params_path=None,
            start_date=cfg.start_dt,
            end_date=cfg.end_dt,
            refet_type=getattr(cfg, "refet_type", "eto") or "eto",
            etf_model=getattr(cfg, "etf_target_model", "ptjpl"),
            met_source=getattr(cfg, "met_source", "era5"),
            fields=fids,
            empirical_kc_max=True,
            mask_mode=getattr(cfg, "mask_mode", "none"),
        )
        output, _ = run_daily_loop_fast(swim_input)
        dates = pd.date_range(swim_input.start_date, periods=swim_input.n_days, freq="D")
        results = {
            fid: pd.Series(output.eta[:, i], index=dates) for i, fid in enumerate(swim_input.fids)
        }
        swim_input.close()
    finally:
        swim_input_mod._container_has_calibration = orig
        if temp_h5 and os.path.exists(temp_h5):
            os.remove(temp_h5)
    return results


def uncal_baseline(per_site, uncal_results):
    """Compare calibrated vs uncalibrated SWIM against flux on common sites.

    Both runs share the same daily date index, so each site is paired against
    flux on the same days. Reports per-site calibrated and uncalibrated metrics
    and the medians over the common evaluable set.
    """
    rows = []
    for fid, d in per_site.items():
        uncal = uncal_results.get(fid)
        if uncal is None:
            continue
        obs = d["obs"]
        cal = d["swim"]
        unc = uncal.reindex(obs.index)
        mc = ev.calc_metrics(obs.values, cal.values)
        mu = ev.calc_metrics(obs.values, unc.values)
        if not (np.isfinite(mc["r2"]) and np.isfinite(mu["r2"])):
            continue
        rows.append(
            {
                "fid": fid,
                "n": mc["n"],
                "r2_cal": mc["r2"],
                "r2_uncal": mu["r2"],
                "rmse_cal": mc["rmse"],
                "rmse_uncal": mu["rmse"],
                "bias_cal": mc["bias"],
                "bias_uncal": mu["bias"],
                "kge_cal": mc["kge"],
                "kge_uncal": mu["kge"],
            }
        )
    df = pd.DataFrame(rows)
    summary = pd.Series(
        {
            "n_sites": len(df),
            "r2_uncal_med": df["r2_uncal"].median(),
            "r2_cal_med": df["r2_cal"].median(),
            "rmse_uncal_med": df["rmse_uncal"].median(),
            "rmse_cal_med": df["rmse_cal"].median(),
            "bias_uncal_med": df["bias_uncal"].median(),
            "bias_cal_med": df["bias_cal"].median(),
            "abs_bias_uncal_med": df["bias_uncal"].abs().median(),
            "abs_bias_cal_med": df["bias_cal"].abs().median(),
            "kge_uncal_med": df["kge_uncal"].median(),
            "kge_cal_med": df["kge_cal"].median(),
        }
    )
    return df, summary


def collect(cfg, container, results_dir, fids, specs, flux_sources=None):
    """Pair SWIM, per-model RS ETa, and flux for every cohort site (all days)."""
    from swimrs.calibrate.flux_utils import passes_site_minimum

    flux_sources = flux_sources or {}
    per_site = {}
    for fid in fids:
        site_csv = os.path.join(results_dir, f"{fid}.csv")
        if not os.path.exists(site_csv):
            continue
        sdf = pd.read_csv(site_csv, index_col=0, parse_dates=True)
        swim = sdf["et_act"]
        etref = sdf["etref"]

        flux = ev.load_flux_et(fid, flux_sources.get(fid))
        if flux.empty or not passes_site_minimum(flux):
            continue

        common = swim.index.intersection(flux.index)
        if len(common) < 10:
            continue

        obs = flux.loc[common]
        swim_c = swim.loc[common]

        rs_by_model = {}
        for spec in specs:
            rs = _model_rs_eta(container, cfg, spec, fid, etref)
            if rs is not None:
                rs_by_model[spec["label"]] = rs.reindex(common)

        per_site[fid] = {
            "obs": obs,
            "swim": swim_c,
            "rs": rs_by_model,
        }
    return per_site


def per_model_benchmarks(per_site, specs):
    """Per-site SWIM vs each RS model, paired on SWIM+RS+flux (all days)."""
    rows = []
    for model in [s["label"] for s in specs]:
        swim_r2, rs_r2, swim_kge, rs_kge = [], [], [], []
        r2_wins, kge_wins, n_sites = 0, 0, 0
        for fid, d in per_site.items():
            if model not in d["rs"]:
                continue
            obs = d["obs"].values
            swim = d["swim"].values
            rs = d["rs"][model].values
            m = np.isfinite(obs) & np.isfinite(swim) & np.isfinite(rs)
            if m.sum() < 10:
                continue
            ms = ev.calc_metrics(obs[m], swim[m])
            mr = ev.calc_metrics(obs[m], rs[m])
            if not (np.isfinite(ms["r2"]) and np.isfinite(mr["r2"])):
                continue
            n_sites += 1
            swim_r2.append(ms["r2"])
            rs_r2.append(mr["r2"])
            swim_kge.append(ms["kge"])
            rs_kge.append(mr["kge"])
            r2_wins += int(ms["r2"] > mr["r2"])
            kge_wins += int(ms["kge"] > mr["kge"])
        rows.append(
            {
                "rs_model": model,
                "n_sites": n_sites,
                "swim_r2_med": np.median(swim_r2) if swim_r2 else np.nan,
                "rs_r2_med": np.median(rs_r2) if rs_r2 else np.nan,
                "swim_kge_med": np.median(swim_kge) if swim_kge else np.nan,
                "rs_kge_med": np.median(rs_kge) if rs_kge else np.nan,
                "swim_r2_win": r2_wins / n_sites if n_sites else np.nan,
                "swim_kge_win": kge_wins / n_sites if n_sites else np.nan,
            }
        )
    return pd.DataFrame(rows)


def conus_decomp(per_site):
    """Per-site SWIM-vs-flux KGE/beta/alpha/MAE/RMSE, grouped US- vs non-US."""
    rows = []
    for fid, d in per_site.items():
        obs = d["obs"].values
        swim = d["swim"].values
        m = np.isfinite(obs) & np.isfinite(swim)
        if m.sum() < 10:
            continue
        o, s = obs[m], swim[m]
        so, sm = np.std(o), np.std(s)
        mo, mm = np.mean(o), np.mean(s)
        r = float(np.corrcoef(o, s)[0, 1])
        alpha = sm / so if so > 0 else np.nan
        beta = mm / mo if mo > 0 else np.nan
        kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
        rows.append(
            {
                "fid": fid,
                "subset": "CONUS" if fid.startswith("US-") else "Ex-CONUS",
                "kge": kge,
                "beta": beta,
                "alpha": alpha,
                "mae": float(np.mean(np.abs(s - o))),
                "rmse": float(np.sqrt(np.mean((s - o) ** 2))),
            }
        )
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("subset")
        .agg(
            sites=("fid", "count"),
            kge_med=("kge", "median"),
            beta_med=("beta", "median"),
            alpha_med=("alpha", "median"),
            mae_med=("mae", "median"),
            rmse_med=("rmse", "median"),
        )
        .reset_index()
    )
    return df, summary


def murphy(per_site, specs):
    """Per-site Murphy decomposition for SWIM and each interpolated RS model."""
    products = [("swim", None)] + [(s["label"], s["label"]) for s in specs]
    rows = []
    for label, model in products:
        rs_list, phase, bias, var = [], [], [], []
        for fid, d in per_site.items():
            obs = d["obs"].values
            if label == "swim":
                mod = d["swim"].values
            else:
                if model not in d["rs"]:
                    continue
                mod = d["rs"][model].values
            m = np.isfinite(obs) & np.isfinite(mod)
            if m.sum() < 10:
                continue
            dec = _murphy_decomp(obs[m], mod[m])
            if dec is None:
                continue
            rs_list.append(dec["r"])
            phase.append(dec["phase_frac"])
            bias.append(dec["bias_frac"])
            var.append(dec["var_frac"])
        rows.append(
            {
                "product": label,
                "n_sites": len(rs_list),
                "r_med": np.median(rs_list) if rs_list else np.nan,
                "phase_frac_med": np.median(phase) if phase else np.nan,
                "bias_frac_med": np.median(bias) if bias else np.nan,
                "var_frac_med": np.median(var) if var else np.nan,
            }
        )
    return pd.DataFrame(rows)


def top_bottom(results_dir, n=10):
    """Best/worst sites by SWIM KGE from evaluation_metrics.csv."""
    path = os.path.join(results_dir, "evaluation_metrics.csv")
    df = pd.read_csv(path)
    df = df.dropna(subset=["kge_swim"]).sort_values("kge_swim", ascending=False)
    best = df.head(n)[["fid", "kge_swim"]].copy()
    worst = df[df["kge_swim"] < 0][["fid", "kge_swim"]].copy()
    return best, worst, df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--container", default=None)
    parser.add_argument("--out", default=None, help="Optional dir for CSV outputs")
    parser.add_argument(
        "--uncalibrated",
        action="store_true",
        help="Also run an uncalibrated (default-parameter) forward baseline",
    )
    args = parser.parse_args()

    conf_path = Path(args.config)
    cfg = ev._load_config(conf_path)
    results_dir = args.results_dir or ev._results_dir(cfg, conf_path)
    container_path = args.container or ev._default_container_path(cfg)
    container = SwimContainer.open(container_path, mode="r")

    try:
        fids = _cohort_fids(cfg, container)
        specs = _rs_model_specs(container, cfg)
        flux_sources = {}
        if Path(cfg.fields_shapefile).exists():
            flux_sources = ev.load_flux_sources(cfg.fields_shapefile, cfg.feature_id_col)
        print(f"Cohort: {len(fids)} sites from {os.path.basename(cfg.fields_shapefile)}")
        print(f"RS models benchmarked: {', '.join(s['label'] for s in specs)}")
        print(f"Results dir: {results_dir}\n")

        per_site = collect(cfg, container, results_dir, fids, specs, flux_sources)
        print(f"Paired sites (>=10 common days, with flux): {len(per_site)}\n")

        bench = per_model_benchmarks(per_site, specs)
        decomp_sites, decomp = conus_decomp(per_site)
        mur = murphy(per_site, specs)
        best, worst, allm = top_bottom(results_dir)

        uncal_df, uncal_sum = None, None
        if args.uncalibrated:
            print("Running uncalibrated (default-parameter) forward baseline...")
            uncal_results = run_uncalibrated_model(cfg, container, fids)
            uncal_df, uncal_sum = uncal_baseline(per_site, uncal_results)
    finally:
        container.close()

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    print("=" * 72)
    print("1. PER-MODEL RS BENCHMARKS (all days; SWIM vs each RS model)")
    print("=" * 72)
    print(bench.to_string(index=False))

    print("\n" + "=" * 72)
    print("2. CONUS vs EX-CONUS DECOMPOSITION (SWIM vs flux, all days)")
    print("=" * 72)
    print(decomp.to_string(index=False))
    print(
        "\n   CONUS (US-) sites:",
        sorted(decomp_sites.loc[decomp_sites.subset == "CONUS", "fid"]),
    )
    print(
        "   Ex-CONUS sites:  ",
        sorted(decomp_sites.loc[decomp_sites.subset == "Ex-CONUS", "fid"]),
    )

    print("\n" + "=" * 72)
    print("3. MURPHY (1988) MSE DECOMPOSITION (fractions; all days)")
    print("=" * 72)
    print(mur.to_string(index=False))

    print("\n" + "=" * 72)
    print("4. TOP / BOTTOM SITES by SWIM KGE")
    print("=" * 72)
    print(f"Total scored sites: {len(allm)}")
    print("\nBest 10:")
    print(best.to_string(index=False))
    print(f"\nWorst (KGE < 0): {len(worst)} sites")
    print(worst.to_string(index=False))

    if uncal_sum is not None:
        print("\n" + "=" * 72)
        print("5. UNCALIBRATED BASELINE (default params; calibrated vs uncalibrated)")
        print("=" * 72)
        print(uncal_sum.to_string())
        n = int(uncal_sum["n_sites"])
        bu, bc = uncal_sum["bias_uncal_med"], uncal_sum["bias_cal_med"]
        ru, rc = uncal_sum["rmse_uncal_med"], uncal_sum["rmse_cal_med"]
        r2u, r2c = uncal_sum["r2_uncal_med"], uncal_sum["r2_cal_med"]
        print(
            f"\n   Over {n} common sites: median bias {bu:+.3f} -> {bc:+.3f} mm/d, "
            f"RMSE {ru:.3f} -> {rc:.3f} mm/d, R2 {r2u:.3f} -> {r2c:.3f}"
        )

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        bench.to_csv(os.path.join(args.out, "derived_per_model_benchmarks.csv"), index=False)
        decomp.to_csv(os.path.join(args.out, "derived_conus_decomp_summary.csv"), index=False)
        decomp_sites.to_csv(os.path.join(args.out, "derived_conus_decomp_persite.csv"), index=False)
        mur.to_csv(os.path.join(args.out, "derived_murphy_decomp.csv"), index=False)
        if uncal_df is not None:
            uncal_df.to_csv(
                os.path.join(args.out, "derived_uncal_baseline_persite.csv"), index=False
            )
        print(f"\nCSV outputs written to {args.out}")


if __name__ == "__main__":
    main()
