"""Spread-error analysis for Example 5 / Experiment E2 (Flux Ensemble).

Tests whether the OpenET six-member inter-model ensemble spread at each
calibration observation predicts the error of the ensemble-mean ETf target
against flux-tower truth. This backs the manuscript's spread-based observation
weighting ``weight = ETf_obs / (sigma_ensemble + 0.1)`` against reviewer
pushback.

Per observation i (site x Landsat overpass date with valid flux):
    spread_i = std across available members' ETf   (= archived ensemble_std)
    target_i = ensemble-mean ETf                   (= archived target_etf)
    truth_i  = flux ETf = flux ET_corr / reference ETo (grass, eto_corr)
    err_i    = target_i - truth_i

The per-observation spread and target are read from the RUN_POLICY Category-3
weight-decomposition table (``observation_metadata.csv``) in the Run 22 archive
and spot-verified against the container member ETf. The ensemble target is the
simple per-overpass member mean (nanmean) and ``ensemble_std`` is the sample
standard deviation (ddof=1) across available members -- the exact quantities in
the weight formula. Reference ETo matches the forward-model reference
(``meteorology/gridmet/eto_corr``, grass reference, refet_type=eto).

Read-only analysis over existing Run 22 data. No calibration, no Earth Engine.

Usage:
    uv run python spread_error.py --config 5_Flux_Ensemble.toml
"""

import argparse
import os

import numpy as np
import pandas as pd
from evaluate import apply_exclusions, load_config, load_flux_et, resolve_flux_dir
from scipy import stats

from swimrs.calibrate.flux_utils import passes_site_minimum
from swimrs.container import SwimContainer

MEMBERS = ["ssebop", "sims", "geesebal", "eemetric", "ptjpl", "disalexi"]

# Canonical Run 22 artifacts (VALIDATION_POLICY E2 section).
DEFAULT_RUN_DIR = "run22"
DEFAULT_CONTAINER = "/data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble_run22.swim"

WEIGHT_FLOOR = 0.1  # weight = target / (spread + WEIGHT_FLOOR)
MIN_ETO = 0.5  # reject overpass days with ETo < 0.5 mm/d (matches ETf extraction screen)
PERSITE_MIN_OBS = 20  # per-site robustness requires >= 20 paired observations


def load_observation_metadata(obs_meta_path):
    """Load the Cat-3 weight-decomposition table and keep ETf ensemble rows.

    Returns a DataFrame with columns: site, date, member_count, target (ETf
    ensemble mean = obsval), spread (ensemble_std). These are the exact
    per-observation quantities used in the calibration objective and weights.
    """
    obs = pd.read_csv(obs_meta_path)
    etf = obs[(obs["model"] == "ensemble") & obs["member_count"].notna()].copy()
    etf["date"] = pd.to_datetime(etf["date"])
    etf = etf.rename(columns={"target_etf": "target", "ensemble_std": "spread"})
    etf["member_count"] = etf["member_count"].astype(int)
    return etf[["site", "date", "member_count", "target", "spread"]].reset_index(drop=True)


def spot_verify(etf_obs, container, n=20, seed=0):
    """Recompute mean/std from container member ETf for n sampled rows.

    Confirms the archived target/spread reproduce from the raw member values
    (nanmean and sample std, ddof=1). Returns a diagnostics dict.
    """
    sample = etf_obs.sample(min(n, len(etf_obs)), random_state=seed)
    # Preload member frames once per site to limit container reads.
    member_cache = {}
    d_mean, d_std, d_cnt = [], [], []
    rows = []
    for _, r in sample.iterrows():
        fid, d = r["site"], r["date"]
        if fid not in member_cache:
            frames = {}
            for m in MEMBERS:
                mdf = container.query.dataframe(
                    f"remote_sensing/etf/landsat/{m}/no_mask", fields=[fid]
                )
                if fid in mdf.columns:
                    frames[m] = mdf[fid]
            member_cache[fid] = frames
        vals = []
        for m, s in member_cache[fid].items():
            if d in s.index and np.isfinite(s.loc[d]):
                vals.append(float(s.loc[d]))
        vals = np.array(vals)
        if len(vals) == 0:
            continue
        rmean = float(np.mean(vals))
        rstd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        d_mean.append(abs(rmean - r["target"]))
        d_std.append(abs(rstd - r["spread"]))
        d_cnt.append(abs(len(vals) - r["member_count"]))
        rows.append(
            {
                "site": fid,
                "date": d.date().isoformat(),
                "n_container": len(vals),
                "member_count_archive": int(r["member_count"]),
                "mean_container": rmean,
                "target_archive": r["target"],
                "std_container": rstd,
                "spread_archive": r["spread"],
            }
        )
    return {
        "n_checked": len(rows),
        "max_abs_diff_mean": float(np.nanmax(d_mean)) if d_mean else np.nan,
        "max_abs_diff_std": float(np.nanmax(d_std)) if d_std else np.nan,
        "max_abs_diff_count": float(np.nanmax(d_cnt)) if d_cnt else np.nan,
        "detail": pd.DataFrame(rows),
    }


def load_reference_eto(container, fids):
    """Load daily reference ETo (grass) matching the forward model.

    Forward model prefers ``{refet}_corr`` when present (input.py); refet_type
    is eto for E2, so this returns ``meteorology/gridmet/eto_corr`` (OpenET
    bias-corrected grass ETo), falling back to raw ``eto``.
    """
    for path in ["meteorology/gridmet/eto_corr", "meteorology/gridmet/eto"]:
        try:
            df = container.query.dataframe(path, fields=fids)
        except KeyError:
            continue
        if not df.empty:
            return df, path
    raise KeyError("No reference ETo array found in container")


def build_observation_table(etf_obs, container, flux_dir):
    """Pair each ETf ensemble observation with flux ETf truth.

    Returns (obs_df, exclusions) where obs_df has one row per pairwise-complete
    observation and exclusions records site- and observation-level drops.
    """
    fids = sorted(etf_obs["site"].unique())
    ref_df, ref_path = load_reference_eto(container, fids)

    exclusions = {
        "no_flux_file": [],
        "below_site_minimum": [],
        "obs_flux_nan": 0,
        "obs_ref_nan": 0,
        "obs_ref_below_min": 0,
    }

    records = []
    kept_sites = []
    for fid in fids:
        flux_et = load_flux_et(fid, flux_dir)
        if flux_et.empty:
            exclusions["no_flux_file"].append(fid)
            continue
        if not passes_site_minimum(flux_et):
            exclusions["below_site_minimum"].append(fid)
            continue

        ref = ref_df[fid] if fid in ref_df.columns else None
        site_obs = etf_obs[etf_obs["site"] == fid]
        n_site_records = 0
        for _, r in site_obs.iterrows():
            d = r["date"]
            fval = flux_et.loc[d] if d in flux_et.index else np.nan
            if not np.isfinite(fval):
                exclusions["obs_flux_nan"] += 1
                continue
            rval = ref.loc[d] if (ref is not None and d in ref.index) else np.nan
            if not np.isfinite(rval):
                exclusions["obs_ref_nan"] += 1
                continue
            if rval < MIN_ETO:
                exclusions["obs_ref_below_min"] += 1
                continue

            flux_etf = float(fval) / float(rval)
            err_etf = float(r["target"]) - flux_etf
            spread = float(r["spread"])
            weight = float(r["target"]) / (spread + WEIGHT_FLOOR)
            # ET space (mm/d)
            target_et = float(r["target"]) * float(rval)
            err_et = target_et - float(fval)
            records.append(
                {
                    "site": fid,
                    "date": d.date().isoformat(),
                    "member_count": int(r["member_count"]),
                    "spread": spread,
                    "target": float(r["target"]),
                    "flux_etf": flux_etf,
                    "err_etf": err_etf,
                    "abs_err_etf": abs(err_etf),
                    "ref_eto": float(rval),
                    "flux_et": float(fval),
                    "target_et": target_et,
                    "err_et": err_et,
                    "abs_err_et": abs(err_et),
                    "spread_et": spread * float(rval),
                    "weight": weight,
                }
            )
            n_site_records += 1
        if n_site_records > 0:
            kept_sites.append(fid)

    obs_df = pd.DataFrame(records)
    return obs_df, exclusions, ref_path, kept_sites


def pooled_correlations(obs_df):
    """Pearson and Spearman of spread vs |err| in ETf and ET space."""
    out = {}
    for space, scol, ecol in [
        ("etf", "spread", "abs_err_etf"),
        ("et", "spread_et", "abs_err_et"),
    ]:
        s = obs_df[scol].values
        e = obs_df[ecol].values
        mask = np.isfinite(s) & np.isfinite(e)
        s, e = s[mask], e[mask]
        pr, pp = stats.pearsonr(s, e)
        sr, sp = stats.spearmanr(s, e)
        out[space] = {
            "n": int(mask.sum()),
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_rho": float(sr),
            "spearman_p": float(sp),
        }
    return out


def quintile_table(obs_df, n_bins=5):
    """Binned spread-skill table: quintiles of ETf spread vs ensemble error."""
    df = obs_df.dropna(subset=["spread", "err_etf"]).copy()
    df["bin"] = pd.qcut(df["spread"], n_bins, labels=False, duplicates="drop")
    rows = []
    for b, g in df.groupby("bin"):
        err = g["err_etf"].values
        rows.append(
            {
                "quintile": int(b) + 1,
                "n": len(g),
                "mean_spread": float(g["spread"].mean()),
                "spread_lo": float(g["spread"].min()),
                "spread_hi": float(g["spread"].max()),
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err**2))),
            }
        )
    tab = pd.DataFrame(rows)
    tab["rmse_over_spread"] = tab["RMSE"] / tab["mean_spread"]
    return tab


def per_site_robustness(obs_df, min_obs=PERSITE_MIN_OBS):
    """Per-site Spearman(spread, |err|) for sites with >= min_obs paired obs."""
    rows = []
    for fid, g in obs_df.groupby("site"):
        g = g.dropna(subset=["spread", "abs_err_etf"])
        if len(g) < min_obs:
            continue
        rho, p = stats.spearmanr(g["spread"].values, g["abs_err_etf"].values)
        rows.append({"site": fid, "n": len(g), "spearman_rho": float(rho), "p": float(p)})
    df = pd.DataFrame(rows)
    summary = {
        "n_sites": len(df),
        "median_spearman": float(df["spearman_rho"].median()) if len(df) else np.nan,
        "frac_positive": float((df["spearman_rho"] > 0).mean()) if len(df) else np.nan,
    }
    return df, summary


def weight_quintile_contrast(obs_df, n_bins=5):
    """Realized |err| for top vs bottom weight quintile under w=target/(spread+floor)."""
    df = obs_df.dropna(subset=["weight", "abs_err_etf"]).copy()
    df["wbin"] = pd.qcut(df["weight"], n_bins, labels=False, duplicates="drop")
    top = int(df["wbin"].max())
    lo = df[df["wbin"] == 0]
    hi = df[df["wbin"] == top]
    return {
        "n_bottom": len(lo),
        "n_top": len(hi),
        "bottom_mean_abs_err": float(lo["abs_err_etf"].mean()),
        "bottom_median_abs_err": float(lo["abs_err_etf"].median()),
        "top_mean_abs_err": float(hi["abs_err_etf"].mean()),
        "top_median_abs_err": float(hi["abs_err_etf"].median()),
        "bottom_mean_weight": float(lo["weight"].mean()),
        "top_mean_weight": float(hi["weight"].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Canonical Run 22 TOML")
    parser.add_argument("--container", type=str, default=DEFAULT_CONTAINER)
    parser.add_argument(
        "--obs-metadata",
        type=str,
        default=None,
        help="Cat-3 observation_metadata.csv (default: Run 22 archive)",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    flux_dir = resolve_flux_dir(cfg)
    results_dir = os.path.join(cfg.project_ws, "results", DEFAULT_RUN_DIR)

    obs_meta_path = args.obs_metadata or os.path.join(
        results_dir, "archive", "3_problem_definition", "observation_metadata.csv"
    )
    out_dir = args.out_dir or os.path.join(results_dir, "spread_error")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Config:          {args.config}")
    print(f"Container:       {args.container}")
    print(f"Obs metadata:    {obs_meta_path}")
    print(f"Flux dir:        {flux_dir}")
    print(f"Output dir:      {out_dir}")

    etf_obs = load_observation_metadata(obs_meta_path)
    etf_obs = etf_obs[etf_obs["site"].isin(apply_exclusions(list(etf_obs["site"].unique())))]
    print(
        f"\nETf ensemble observations: {len(etf_obs)} rows, "
        f"{etf_obs['site'].nunique()} sites, "
        f"{etf_obs['date'].min().date()} -> {etf_obs['date'].max().date()}"
    )

    container = SwimContainer.open(args.container, mode="r")
    try:
        # 0. Spot-verify archive target/spread against container member ETf.
        sv = spot_verify(etf_obs, container, n=20)
        print(
            f"\nSpot-verify ({sv['n_checked']} rows): "
            f"max|mean diff|={sv['max_abs_diff_mean']:.2e}, "
            f"max|std diff|={sv['max_abs_diff_std']:.2e}, "
            f"max|count diff|={sv['max_abs_diff_count']:.0f}"
        )
        sv["detail"].to_csv(os.path.join(out_dir, "spot_verify.csv"), index=False)

        # 1. Pair with flux ETf truth.
        obs_df, excl, ref_path, kept_sites = build_observation_table(etf_obs, container, flux_dir)
    finally:
        container.close()

    print(f"\nReference ETo path: {ref_path}")
    print(
        f"Exclusions: no_flux_file={len(excl['no_flux_file'])} "
        f"below_site_minimum={len(excl['below_site_minimum'])} "
        f"obs_flux_nan={excl['obs_flux_nan']} obs_ref_nan={excl['obs_ref_nan']} "
        f"obs_ref_below_min={excl['obs_ref_below_min']}"
    )
    print(f"Paired observations: {len(obs_df)} across {obs_df['site'].nunique()} sites")

    obs_df.to_csv(os.path.join(out_dir, "spread_error_observations.csv"), index=False)

    # 2. Pooled association.
    pooled = pooled_correlations(obs_df)
    print("\n=== Pooled spread vs |err| ===")
    for space, d in pooled.items():
        print(
            f"  {space.upper():>3}: n={d['n']}  Pearson r={d['pearson_r']:.3f} "
            f"(p={d['pearson_p']:.1e})  Spearman rho={d['spearman_rho']:.3f} "
            f"(p={d['spearman_p']:.1e})"
        )

    # 3. Binned spread-skill quintile table.
    qt = quintile_table(obs_df)
    qt.to_csv(os.path.join(out_dir, "spread_error_quintiles.csv"), index=False)
    print("\n=== Spread quintile table (ETf space) ===")
    print(qt.to_string(index=False))
    rmse_ratio = qt["RMSE"].iloc[-1] / qt["RMSE"].iloc[0]
    monotonic = bool(np.all(np.diff(qt["RMSE"].values) > 0))
    print(f"  High/low quintile RMSE ratio: {rmse_ratio:.3f}  monotonic_increasing={monotonic}")

    # 4. Per-site robustness.
    ps_df, ps_sum = per_site_robustness(obs_df)
    ps_df.to_csv(os.path.join(out_dir, "spread_error_persite.csv"), index=False)
    print("\n=== Per-site robustness (>= 20 paired obs) ===")
    print(
        f"  n_sites={ps_sum['n_sites']}  median Spearman={ps_sum['median_spearman']:.3f}  "
        f"frac positive={ps_sum['frac_positive']:.3f}"
    )

    # 5. Weight-quintile contrast.
    wq = weight_quintile_contrast(obs_df)
    print("\n=== Weight-quintile |err| contrast (w = target/(spread+0.1)) ===")
    print(
        f"  bottom-quintile (n={wq['n_bottom']}, mean w={wq['bottom_mean_weight']:.2f}): "
        f"mean|err|={wq['bottom_mean_abs_err']:.3f} median|err|={wq['bottom_median_abs_err']:.3f}"
    )
    print(
        f"  top-quintile    (n={wq['n_top']}, mean w={wq['top_mean_weight']:.2f}): "
        f"mean|err|={wq['top_mean_abs_err']:.3f} median|err|={wq['top_median_abs_err']:.3f}"
    )

    # Summary artifact.
    summary = {
        "n_observations": len(obs_df),
        "n_sites": int(obs_df["site"].nunique()),
        "reference_eto_path": ref_path,
        "min_eto_screen": MIN_ETO,
        "spot_verify_max_abs_diff_std": sv["max_abs_diff_std"],
        "spot_verify_max_abs_diff_mean": sv["max_abs_diff_mean"],
        "exclusion_no_flux_file": len(excl["no_flux_file"]),
        "exclusion_below_site_minimum": len(excl["below_site_minimum"]),
        "exclusion_obs_flux_nan": excl["obs_flux_nan"],
        "exclusion_obs_ref_nan": excl["obs_ref_nan"],
        "exclusion_obs_ref_below_min": excl["obs_ref_below_min"],
        "pooled_etf_pearson_r": pooled["etf"]["pearson_r"],
        "pooled_etf_spearman_rho": pooled["etf"]["spearman_rho"],
        "pooled_et_pearson_r": pooled["et"]["pearson_r"],
        "pooled_et_spearman_rho": pooled["et"]["spearman_rho"],
        "quintile_rmse_ratio_hi_lo": rmse_ratio,
        "quintile_rmse_monotonic": monotonic,
        "persite_n_sites": ps_sum["n_sites"],
        "persite_median_spearman": ps_sum["median_spearman"],
        "persite_frac_positive": ps_sum["frac_positive"],
        "weight_bottom_quintile_mean_abs_err": wq["bottom_mean_abs_err"],
        "weight_top_quintile_mean_abs_err": wq["top_mean_abs_err"],
        "weight_bottom_quintile_median_abs_err": wq["bottom_median_abs_err"],
        "weight_top_quintile_median_abs_err": wq["top_median_abs_err"],
    }
    pd.Series(summary).to_csv(os.path.join(out_dir, "spread_error_summary.csv"), header=False)
    print(f"\nArtifacts written to {out_dir}")


if __name__ == "__main__":
    main()
