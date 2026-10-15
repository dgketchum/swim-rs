"""Figure 4: ensemble spread as reliability information."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths
from .manifest import MANIFEST, BuildError, require_columns, require_unique, sha256, write_table
from .paths import E1_CONTAINER, E1_RUN, FINAL, SCRIPT_VERSION, ex5_paths

# --------------------------------------------------------------------------
# Figure 4 -- ensemble spread as reliability information
# --------------------------------------------------------------------------

E1_MEMBERS = ["ssebop", "ptjpl", "sims", "geesebal", "eemetric", "disalexi"]


def _e1_spread_counts() -> dict:
    """Capture, site and association counts from the canonical run's spread_error summary."""
    p = E1_RUN / "spread_error" / "spread_error_summary.csv"
    if not p.exists():
        raise BuildError(f"fig04 source missing: {p}")
    summ = pd.read_csv(p, header=None, index_col=0).squeeze("columns")
    n_assoc = int(float(summ["persite_n_sites"]))
    return {
        "captures": int(float(summ["n_observations"])),
        "capture_sites": int(float(summ["n_sites"])),
        "association_sites": n_assoc,
        "association_positive": int(round(float(summ["persite_frac_positive"]) * n_assoc)),
    }


def build_fig04() -> None:
    exp = _e1_spread_counts()
    src_obs = E1_RUN / "spread_error" / "spread_error_observations.csv"
    src_q = FINAL / "e2_spread_error_quintiles.csv"
    src_ps = FINAL / "e2_spread_error_persite.csv"
    src_sum = FINAL / "e2_spread_error_summary.csv"
    src_unc_ps = E1_RUN / "conditioned_ensemble_uncertainty" / "uncertainty_persite.csv"
    src_unc_obs = E1_RUN / "conditioned_ensemble_uncertainty" / "uncertainty_observations.csv"
    src_unc_sum = E1_RUN / "conditioned_ensemble_uncertainty" / "uncertainty_summary.csv"
    src_wd = FINAL / "e2_weighting_ablation_paired_deltas.csv"
    src_wdd = FINAL / "e2_weighting_ablation_daily_site_deltas.csv"
    src_wdm = FINAL / "e2_weighting_ablation_monthly_site_deltas.csv"
    for p in (
        src_obs,
        src_q,
        src_ps,
        src_sum,
        src_unc_ps,
        src_unc_obs,
        src_unc_sum,
        src_wd,
        src_wdd,
        src_wdm,
    ):
        if not p.exists():
            raise BuildError(f"fig04 source missing: {p}")

    obs = pd.read_csv(src_obs, parse_dates=["date"])
    require_columns(
        obs,
        ["site", "date", "member_count", "spread", "target", "flux_etf", "weight"],
        "spread_error_observations",
    )
    require_unique(obs, ["site", "date"], "spread_error_observations")
    if len(obs) != exp["captures"]:
        raise BuildError(f"fig04: expected {exp['captures']} paired captures, got {len(obs)}")
    if obs["site"].nunique() != exp["capture_sites"]:
        raise BuildError(
            f"fig04: expected {exp['capture_sites']} sites, got {obs['site'].nunique()}"
        )

    members = _e1_member_etf(obs)
    cap = obs.merge(members, on=["site", "date"], how="left", validate="one_to_one")

    unc_obs = pd.read_csv(src_unc_obs, parse_dates=["date"])
    keep = [
        "site",
        "date",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "spread_conditioned_iqr",
        "spread_conditioned_std",
        "width90",
        "etf_effective",
        "abs_error_effective",
        "covered_90",
    ]
    cap = cap.merge(unc_obs[keep], on=["site", "date"], how="left", validate="one_to_one")

    recomputed = cap[[f"etf_{m}" for m in E1_MEMBERS]].notna().sum(axis=1)
    mismatch = int((recomputed != cap["member_count"]).sum())
    if mismatch:
        raise BuildError(
            f"fig04: member-count mismatch between container arrays and archived metadata at {mismatch} captures"
        )

    cap.insert(0, "legacy_prefix", "e2_")
    cap.insert(0, "experiment", "E1")
    cap = cap.rename(
        columns={
            "site": "site_id",
            "spread": "spread_retrieval",
            "target": "ensemble_mean_etf",
            "flux_etf": "flux_derived_etf",
            "err_etf": "error_etf",
            "abs_err_etf": "abs_error_etf",
            "etf_effective": "swim_etf_effective",
            "abs_error_effective": "swim_abs_error_effective",
        }
    )
    n_cap = write_table(cap, "fig04_spread_capture_values.csv")

    q = pd.read_csv(src_q)
    q.insert(0, "legacy_prefix", "e2_")
    q.insert(0, "experiment", "E1")
    n_q = write_table(q, "fig04_spread_quintiles.csv")

    ps = pd.read_csv(src_ps).rename(
        columns={"site": "site_id", "spearman_rho": "rho_retrieval_vs_abs_ensemble_error"}
    )
    unc_ps = pd.read_csv(src_unc_ps).rename(columns={"site": "site_id"})
    assoc = ps.merge(
        unc_ps[
            [
                "site_id",
                "n",
                "eligible",
                "rho_retrieval",
                "rho_conditioned_iqr",
                "rho_conditioned_std",
                "rho_conditioned_width90",
                "delta_rho",
                "coverage_90",
                "median_width90",
                "median_iqr",
                "median_spread_retrieval",
            ]
        ],
        on="site_id",
        how="outer",
        suffixes=("_spread_analysis", "_conditioned"),
    )
    assoc.insert(0, "legacy_prefix", "e2_")
    assoc.insert(0, "experiment", "E1")
    n_assoc = write_table(assoc, "fig04_site_associations.csv")

    elig = unc_ps[unc_ps["eligible"].astype(bool)]
    n_assoc = exp["association_sites"]
    if len(elig) != n_assoc:
        raise BuildError(f"fig04: expected {n_assoc} eligible sites, got {len(elig)}")
    if len(ps) != n_assoc:
        raise BuildError(f"fig04: expected {n_assoc} spread-analysis sites, got {len(ps)}")
    n_pos = int((ps["rho_retrieval_vs_abs_ensemble_error"] > 0).sum())
    if n_pos != exp["association_positive"]:
        raise BuildError(
            f"fig04: expected {exp['association_positive']} of {n_assoc} positive within-site "
            f"spread-error associations, got {n_pos}"
        )

    cond = unc_ps.copy()
    cond.insert(0, "legacy_prefix", "e2_")
    cond.insert(0, "experiment", "E1")
    summary = pd.read_csv(src_unc_sum)
    _sum = summary.set_index("metric")
    _drho = _sum.loc["median_delta_rho"]
    _cov90 = float(_sum.loc["pooled_coverage_90", "point"])
    n_cond = write_table(cond, "fig04_conditioned_spread.csv")

    wd = pd.read_csv(src_wd)
    require_columns(
        wd,
        ["scale", "metric", "n_sites", "median_delta", "ci_lower", "ci_upper"],
        "weighting_ablation_paired_deltas",
    )
    wdd = pd.read_csv(src_wdd)
    wdm = pd.read_csv(src_wdm)

    def _site_deltas(df, scale):
        require_columns(
            df,
            [
                "fid",
                "n_paired",
                "delta_r2_swim",
                "delta_kge_swim",
                "delta_rmse_swim",
                "e1_bias_swim",
                "e2_bias_swim",
                "delta_bias_swim",
            ],
            f"weighting_ablation_{scale}_site_deltas",
        )
        signed_mbe_delta = df["e1_bias_swim"] - df["e2_bias_swim"]
        if not np.allclose(
            signed_mbe_delta,
            df["delta_bias_swim"],
            equal_nan=True,
            rtol=0.0,
            atol=1e-12,
        ):
            raise BuildError(
                f"fig04 weighting: archived {scale} delta_bias_swim does not equal "
                "MBE_spread - MBE_fixed_sd"
            )
        return pd.DataFrame(
            {
                "experiment": "E1",
                "legacy_prefix": "e2_",
                "record_type": "site_delta",
                "scale": scale,
                "site_id": df["fid"],
                "metric": None,
                "n_paired": df["n_paired"],
                "d_nse": df["delta_r2_swim"],
                "d_kge": df["delta_kge_swim"],
                "d_rmse": df["delta_rmse_swim"],
                "d_mbe": signed_mbe_delta,
                "d_abs_mbe": df["e1_bias_swim"].abs() - df["e2_bias_swim"].abs(),
            }
        )

    site_rows = pd.concat(
        [_site_deltas(wdd, "daily"), _site_deltas(wdm, "monthly")], ignore_index=True
    )
    metric_rename = {
        "nse": "nse",
        "r2": "nse",
        "kge": "kge",
        "rmse": "rmse",
        "abs_bias": "abs_mbe",
        "abs_mbe": "abs_mbe",
    }
    eff_rows = wd.copy()
    eff_rows["metric_display"] = eff_rows["metric"].map(metric_rename)
    if eff_rows["metric_display"].isna().any():
        raise BuildError("fig04 weighting: unmapped metric label")
    eff_rows = pd.DataFrame(
        {
            "experiment": "E1",
            "legacy_prefix": "e2_",
            "record_type": "cohort_effect",
            "scale": eff_rows["scale"],
            "site_id": None,
            "metric": eff_rows["metric_display"],
            "n_paired": eff_rows["n_sites"],
            "median_delta": eff_rows["median_delta"],
            "ci95_lo": eff_rows["ci_lower"],
            "ci95_hi": eff_rows["ci_upper"],
            "delta_definition": eff_rows["delta_definition"],
            "favorable_direction": eff_rows["favorable_direction"],
            "bootstrap_seed": eff_rows["bootstrap_seed"],
            "bootstrap_reps": eff_rows["bootstrap_reps"],
        }
    )

    # Signed MBE is a principal reader-facing metric, but the archived ablation
    # summary intentionally contains the accuracy-oriented |MBE| contrast only.
    # Expose the already archived signed site deltas through the figure contract
    # with the same whole-site median-bootstrap design. This is a display-level
    # aggregation: it does not rerun evaluation, calibration, or the model.
    # Every site in the frozen ablation cohort (gated against the promoted E1
    # daily/monthly primaries by promote_e1_run.py) must carry a finite delta:
    # 45 daily / 30 monthly on Run 22, 45 / 42 on Run 23 (Volk month protocol).
    signed_mbe_rows = []
    for scale, expected_n in (("daily", len(wdd)), ("monthly", len(wdm))):
        values = site_rows.loc[site_rows["scale"] == scale, "d_mbe"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) != expected_n:
            raise BuildError(
                f"fig04 weighting: expected {expected_n} finite {scale} signed-MBE "
                f"effects, got {len(values)}"
            )
        rng = np.random.default_rng(42)
        indices = rng.integers(0, len(values), size=(10_000, len(values)))
        bootstrap_medians = np.median(values[indices], axis=1)
        signed_mbe_rows.append(
            {
                "experiment": "E1",
                "legacy_prefix": "e2_",
                "record_type": "cohort_effect",
                "scale": scale,
                "site_id": None,
                "metric": "mbe",
                "n_paired": len(values),
                "median_delta": float(np.median(values)),
                "ci95_lo": float(np.percentile(bootstrap_medians, 2.5)),
                "ci95_hi": float(np.percentile(bootstrap_medians, 97.5)),
                "delta_definition": "mbe_spread - mbe_fixed_sd",
                "favorable_direction": "directional",
                "bootstrap_seed": 42,
                "bootstrap_reps": 10_000,
            }
        )
    eff_rows = pd.concat(
        [eff_rows, pd.DataFrame(signed_mbe_rows)],
        ignore_index=True,
    )
    weighting = pd.concat([eff_rows, site_rows], ignore_index=True)
    n_w = write_table(weighting, "fig04_weighting_effects.csv")
    # archived cohort effects + the two signed-MBE effects + one row per site and
    # scale (85 on Run 22: 8 + 2 + 45 + 30; 97 on Run 23: 8 + 2 + 45 + 42)
    expected_w = len(wd) + 2 + len(wdd) + len(wdm)
    if n_w != expected_w:
        raise BuildError(f"fig04 weighting: expected {expected_w} display rows, got {n_w}")

    # panel (a) example selection
    eligible = ps.merge(unc_ps[["site_id", "eligible"]], on="site_id", how="left")
    eligible = eligible[eligible["n"] >= 20].copy()
    med = float(eligible["rho_retrieval_vs_abs_ensemble_error"].median())
    eligible["rank_value"] = (eligible["rho_retrieval_vs_abs_ensemble_error"] - med).abs()
    eligible = eligible.sort_values(["rank_value", "site_id"]).reset_index(drop=True)
    chosen_site = str(eligible.loc[0, "site_id"])
    caps = cap[cap["site_id"] == chosen_site].copy()
    payload = {
        "figure": "fig04",
        "panel": "a",
        "experiment": "E1",
        "legacy_prefix": "e2_",
        "selection_rule": {
            "step_1": "sites with at least 20 paired capture-level observations in the E1 spread-error analysis",
            "step_2": "rank by abs(within-site Spearman rho(retrieval spread, |ensemble-mean ETf error|) - median rho among eligible sites)",
            "step_3": "select the nearest site; ties broken by ascending site_id",
            "step_4": "freeze the eligible list, ranking values, chosen site, and the exact capture dates",
        },
        "n_eligible_sites": int(len(eligible)),
        "median_rho_among_eligible": med,
        "ranking": eligible[
            ["site_id", "n", "rho_retrieval_vs_abs_ensemble_error", "rank_value"]
        ].to_dict("records"),
        "selected_site": chosen_site,
        "selected_site_rho": float(eligible.loc[0, "rho_retrieval_vs_abs_ensemble_error"]),
        "selected_captures": [
            {
                "date": str(pd.Timestamp(r["date"]).date()),
                "member_count": int(r["member_count"]),
                "ensemble_mean_etf": float(r["ensemble_mean_etf"]),
                "spread_retrieval": float(r["spread_retrieval"]),
                "flux_derived_etf": float(r["flux_derived_etf"]),
                **{
                    f"etf_{m}": (None if pd.isna(r[f"etf_{m}"]) else float(r[f"etf_{m}"]))
                    for m in E1_MEMBERS
                },
            }
            for _, r in caps.sort_values("date").iterrows()
        ],
        "generator_script": "scripts/figures/build_figure_data.py",
        "generator_version": SCRIPT_VERSION,
    }
    (paths.OUT / "fig04_example_selection.json").write_text(json.dumps(payload, indent=2))

    srcs = {
        "spread_error_observations": src_obs,
        "e2_spread_error_quintiles": src_q,
        "e2_spread_error_persite": src_ps,
        "e2_spread_error_summary": src_sum,
        "uncertainty_persite": src_unc_ps,
        "uncertainty_observations": src_unc_obs,
        "uncertainty_summary": src_unc_sum,
        "e2_weighting_ablation_paired_deltas": src_wd,
        "e2_weighting_ablation_daily_site_deltas": src_wdd,
        "e2_weighting_ablation_monthly_site_deltas": src_wdm,
    }
    base = {
        "sources": {k: {"path": str(p), "sha256": sha256(p)} for k, p in srcs.items()},
        "container_source": {
            "path": str(E1_CONTAINER),
            "arrays": [f"remote_sensing/etf/landsat/{m}/no_mask" for m in E1_MEMBERS],
        },
        "experiment_mapping": {"E1": "legacy e2_*"},
        "cohort_key": "site_id (+ date for capture-level rows)",
        "inclusion_rule": (
            f"Capture-level: {exp['captures']:,} ETf ensemble calibration targets across "
            f"{exp['capture_sites']} sites that "
            "have a finite same-day flux ET_corr and a reference ETo >= 0.5 mm d-1; sites "
            "must pass the VALIDATION_POLICY minimum and MB_Pch is excluded. Site-level "
            f"associations: the {exp['association_sites']} sites with at least 20 paired "
            "captures."
        ),
        "temporal_support_rule": "Landsat acquisition dates retained as ETf calibration targets, 2016-2025, paired to same-day flux observations (Volk v2.1 record ends mid-2022).",
        "units": {
            "etf": "dimensionless ET fraction",
            "spread_retrieval": "dimensionless ETf standard deviation across valid members (ddof=1)",
            "weight": "dimensionless PEST observation weight, ETf_obs/(sigma+0.1)",
            "rmse_daily": "mm d-1",
            "rmse_monthly": "mm month-1",
            "mbe_daily": "mm d-1",
            "mbe_monthly": "mm month-1",
        },
        "deterministic_seed": 42,
        "configured_counts": {"E1": 60},
        "evaluated_counts": {
            "captures": exp["captures"],
            "capture_sites": exp["capture_sites"],
            "association_sites": exp["association_sites"],
            # Derived from the frozen site-delta files (45 / 30 Run 22; 45 / 42 Run 23
            # once the monthly ablation scored on the Volk 2024 month protocol).
            "weighting_daily_sites": int(len(wdd)),
            "weighting_monthly_sites": int(len(wdm)),
            "weighting_monthly_finite_sites": int(
                wdm[["delta_r2_swim", "delta_kge_swim", "delta_rmse_swim", "delta_bias_swim"]]
                .notna()
                .all(axis=1)
                .sum()
            ),
        },
    }
    MANIFEST.add(
        "fig04_spread_capture_values.csv",
        rows=n_cap,
        display_transformations=[
            f"six OpenET member ETf values joined from the {ex5_paths.CANONICAL_RUN} container and cross-checked against the archived member_count (exact match required)",
            "conditioned-ensemble quantiles joined from uncertainty_observations.csv on (site, date)",
            "legacy column 'spread' renamed spread_retrieval; 'target' renamed ensemble_mean_etf",
        ],
        **base,
    )
    MANIFEST.add(
        "fig04_spread_quintiles.csv",
        rows=n_q,
        display_transformations=["experiment label columns prepended; values unchanged"],
        note=f"Quintiles of retrieval spread over the {exp['captures']:,} pooled captures; MAE/RMSE in ETf units. Descriptive, not a calibrated uncertainty function.",
        **base,
    )
    MANIFEST.add(
        "fig04_site_associations.csv",
        rows=n_assoc,
        display_transformations=[
            "spread-analysis per-site Spearman rho joined to the conditioned-ensemble per-site rho on site_id",
            "legacy column spearman_rho renamed rho_retrieval_vs_abs_ensemble_error",
        ],
        **base,
    )
    MANIFEST.add(
        "fig04_conditioned_spread.csv",
        rows=n_cond,
        display_transformations=["experiment label columns prepended; values unchanged"],
        note=(
            "Conditioned-ensemble diagnostic. delta_rho = rho(retrieval spread, |err|) - "
            f"rho(conditioned IQR, |err|). Cohort estimate {_drho['point']:+.3f} "
            f"[{_drho['ci_lo']:+.3f}, {_drho['ci_hi']:+.3f}] over {int(_drho['n_sites'])} "
            f"eligible sites ({int(_drho['n_boot']):,} site-bootstrap, seed {int(_drho['seed'])}). "
            f"Pooled 90% conditioned envelope covers {100.0 * _cov90:.1f}% of flux-derived ETf; "
            "this is an empirical coverage diagnostic and must never be shown as a "
            "predictive interval."
        ),
        cohort_summary=summary.to_dict("records"),
        **base,
    )
    MANIFEST.add(
        "fig04_weighting_effects.csv",
        rows=n_w,
        display_transformations=[
            "legacy arm labels e1_*/e2_* mapped to spread-weighted / fixed-scale (0.33 ETf) arms",
            "legacy metric label r2 renamed nse after evaluator verification; abs_bias renamed abs_mbe",
            "signed site MBE effect exposed as MBE_spread - MBE_fixed_sd and verified against archived delta_bias_swim",
            "signed-MBE cohort effect summarized by 10,000 whole-site bootstrap resamples with seed 42",
            "delta_abs_mbe recomputed at site level as |MBE_spread| - |MBE_fixed|",
        ],
        note=(
            "record_type=cohort_effect rows carry 10,000-resample whole-site bootstrap "
            "95% intervals (seed 42); the signed-MBE rows are a targeted display-level "
            "aggregation of the archived site deltas, while the other rows are frozen "
            "upstream summaries. record_type=site_delta rows carry the raw paired site "
            "values. Arms differ only in the weight denominator (sigma_ensemble+0.1 versus "
            "fixed 0.33 ETf). Objective-function values are not comparable across arms."
        ),
        **base,
    )
    MANIFEST.add(
        "fig04_example_selection.json",
        rows=None,
        selected_site=chosen_site,
        n_selected_captures=len(caps),
        note=(
            "Retained audit record for the superseded Figure 4 capture example; no visual "
            "inspection entered the choice and the active main-figure design does not use it."
        ),
        cohort_key="site_id",
        inclusion_rule=base["inclusion_rule"],
        deterministic_seed=None,
    )
    print(
        f"  fig04: captures {n_cap}, quintiles {n_q}, associations {n_assoc}, "
        f"conditioned {n_cond}, weighting {n_w}; example site {chosen_site} ({len(caps)} captures)"
    )


def _e1_member_etf(obs: pd.DataFrame) -> pd.DataFrame:
    import zarr

    z = zarr.open(str(E1_CONTAINER), mode="r")
    uid = [str(x) for x in z["geometry/uid"][:]]
    t = pd.to_datetime(z["time/daily"][:])
    tpos = pd.Series(np.arange(len(t)), index=t)
    arrays = {
        m: np.asarray(z[f"remote_sensing/etf/landsat/{m}/no_mask"][:], dtype=float)
        for m in E1_MEMBERS
    }
    rows = []
    for site, grp in obs.groupby("site"):
        j = uid.index(site)
        idx = tpos.reindex(grp["date"]).to_numpy()
        if np.isnan(idx).any():
            raise BuildError(f"fig04: capture date outside container time axis at {site}")
        idx = idx.astype(int)
        rec = {"site": site, "date": grp["date"].values}
        for m in E1_MEMBERS:
            rec[f"etf_{m}"] = arrays[m][idx, j]
        rows.append(pd.DataFrame(rec))
    return pd.concat(rows, ignore_index=True)
