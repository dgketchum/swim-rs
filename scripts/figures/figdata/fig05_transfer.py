"""Figure 5: parameter transfer."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .cohorts import EXPECTED, FIG01_CLASS_COUNTS
from .manifest import (
    MANIFEST,
    BuildError,
    require_columns,
    require_count,
    require_unique,
    sha256,
    write_table,
)
from .paths import (
    E1_WITHIN_STRAT,
    E2_CONTAINER,
    E2_FROZEN_POOL,
    E2_FROZEN_TRANSFER,
    E2_MAPPING_META,
    E2_RESULTS,
    E2_TRANSFER,
    FINAL,
    ex5_paths,
)
from .sources import (
    E2_CONFIGS,
    check_pool_medians,
    e1_configured,
    e2_configured,
    gate_closure_pool,
    gate_pool_summary,
    restrict_to_pool,
)

# --------------------------------------------------------------------------
# Figure 5 -- parameter transfer
# --------------------------------------------------------------------------

E1_ARMS = {
    "default": ("generic_defaults", "uncalibrated generic prior parameter set"),
    "loro_strat": (
        "heldout_transfer_class_specific_loro",
        "irrigation-class-specific leave-region-out transfer (primary)",
    ),
    "loso_strat": (
        "heldout_transfer_class_specific_loso",
        "irrigation-class-specific leave-one-site-out transfer (fold-granularity sensitivity)",
    ),
    "loro": (
        "heldout_transfer_pooled_loro",
        "superseded pooled leave-region-out transfer (provenance only)",
    ),
    "loso": (
        "heldout_transfer_pooled_loso",
        "superseded pooled leave-one-site-out transfer (provenance only)",
    ),
    "local": ("local_calibration", f"{ex5_paths.CANONICAL_RUN} local site calibration"),
}


def build_fig05_e1() -> None:
    src_d = E1_WITHIN_STRAT / "persite_daily.csv"
    src_m = E1_WITHIN_STRAT / "persite_monthly.csv"
    src_sum = FINAL / "e2_irrigation_stratified_transfer_summary.csv"
    src_mad = FINAL / "e2_irrigation_stratified_fold_mad_domain.csv"
    src_fold = E1_WITHIN_STRAT / "class_fold_support.csv"
    for p in (src_d, src_m, src_sum, src_mad, src_fold):
        if not p.exists():
            raise BuildError(f"fig05 E1 source missing: {p}")

    d = pd.read_csv(src_d)
    m = pd.read_csv(src_m)
    require_unique(d, ["fid"], "E1 transfer persite_daily")
    require_unique(m, ["fid"], "E1 transfer persite_monthly")
    require_count(len(d), EXPECTED["E1_transfer_daily"], "fig05 E1 daily cohort")
    m = m[m["local_kge"].notna()].copy()
    require_count(len(m), EXPECTED["E1_transfer_monthly"], "fig05 E1 monthly cohort")

    mad = pd.read_csv(src_mad)
    fold = pd.read_csv(src_fold)

    rows = []
    for scale, df in (("daily", d), ("monthly", m)):
        for arm, (name, prov) in E1_ARMS.items():
            need = [f"{arm}_{k}" for k in ("n", "r2", "kge", "rmse", "bias")]
            require_columns(df, need, f"E1 transfer {scale} arm {arm}")
            block = pd.DataFrame(
                {
                    "experiment": "E1",
                    "legacy_prefix": "e2_",
                    "scale": scale,
                    "site_id": df["fid"],
                    "region": df["region"],
                    "irrigation_class": df["irr_class"],
                    "treatment": name,
                    "legacy_arm": arm,
                    "treatment_provenance": prov,
                    "n_paired": df[f"{arm}_n"],
                    "nse": df[f"{arm}_r2"],
                    "kge": df[f"{arm}_kge"],
                    "rmse": df[f"{arm}_rmse"],
                    "mbe": df[f"{arm}_bias"],
                    "r": df[f"{arm}_r"],
                }
            )
            for metric, col in (
                ("nse", "r2"),
                ("kge", "kge"),
                ("rmse", "rmse"),
            ):
                block[f"d_{metric}_vs_local"] = df[f"{arm}_{col}"] - df[f"local_{col}"]
                block[f"d_{metric}_vs_default"] = df[f"{arm}_{col}"] - df[f"default_{col}"]
            block["d_abs_mbe_vs_local"] = df[f"{arm}_bias"].abs() - df["local_bias"].abs()
            block["d_abs_mbe_vs_default"] = df[f"{arm}_bias"].abs() - df["default_bias"].abs()
            rows.append(block)
    site = pd.concat(rows, ignore_index=True)

    madm = mad.set_index(["arm", "fid"])
    site["fold_mad"] = [
        madm["mad"].get((a, f), np.nan) for a, f in zip(site["legacy_arm"], site["site_id"])
    ]
    site["fold_mad_in_class_prior"] = [
        madm["mad_in_class_prior"].get((a, f), np.nan)
        for a, f in zip(site["legacy_arm"], site["site_id"])
    ]
    foldm = fold.set_index("fid")
    site["n_train_loro"] = site["site_id"].map(foldm["n_train_loro"])
    site["n_train_loro_strat"] = site["site_id"].map(foldm["n_train_loro_strat"])

    summ = pd.read_csv(src_sum)
    summ = summ[summ["experiment"] == "E2_within_held_out"].copy()
    summ["treatment"] = summ["config"].map(lambda c: E1_ARMS.get(c, (c, ""))[0])
    summ_out = pd.DataFrame(
        {
            "experiment": "E1",
            "legacy_prefix": "e2_",
            "scale": summ["basis"],
            "site_id": None,
            "region": None,
            "irrigation_class": summ["stratum"],
            "treatment": summ["treatment"],
            "legacy_arm": summ["config"],
            "treatment_provenance": summ["label"],
            "record_type": "cohort_summary",
            "metric": summ["metric"].map(
                {"r2": "nse", "kge": "kge", "rmse": "rmse", "bias": "mbe", "abs_bias": "abs_mbe"}
            ),
            "median_common": summ["median_common"],
            "ci95_lo": summ["ci_lo"],
            "ci95_hi": summ["ci_hi"],
            "n_common": summ["n_common"],
        }
    )
    if summ_out["metric"].isna().any():
        raise BuildError("fig05 E1: unmapped summary metric label")
    site["record_type"] = "site_metric"
    out = pd.concat([site, summ_out], ignore_index=True)
    n = write_table(out, "fig05_e1_heldout_transfer.csv")

    MANIFEST.add(
        "fig05_e1_heldout_transfer.csv",
        rows=n,
        sources={
            k: {"path": str(p), "sha256": sha256(p)}
            for k, p in {
                "within_e1_stratified_persite_daily": src_d,
                "within_e1_stratified_persite_monthly": src_m,
                "e2_irrigation_stratified_transfer_summary": src_sum,
                "e2_irrigation_stratified_fold_mad_domain": src_mad,
                "class_fold_support": src_fold,
            }.items()
        },
        experiment_mapping={"E1": "legacy e2_*; legacy experiment label E2_within_held_out"},
        cohort_key="site_id",
        inclusion_rule=(
            # Cohort counts match the canonical run's within_e1_transfer output.
            f"Common paired cohort of the {ex5_paths.CANONICAL_RUN} 45-site daily and "
            "31-site monthly "
            "evaluations, evaluated identically under all six arms. 29 irrigated / 16 "
            "rainfed daily sites. Held-out folds exclude each evaluated site from the "
            "parameter set applied to it; the class-specific vector for each fold is the "
            "median of per-site posterior medians within the site's irrigation class."
        ),
        temporal_support_rule="Daily common-support days; monthly complete calendar months with a 10-month finite-metric floor.",
        units={"rmse_daily": "mm d-1", "rmse_monthly": "mm month-1", "mbe": "mm d-1 / mm month-1"},
        display_transformations=[
            "legacy arm labels mapped to reader-facing treatment names (see treatment/legacy_arm columns)",
            "legacy r2 -> nse; legacy bias -> mbe",
            "paired site deltas computed against the local-calibration and generic-default arms",
            "fold-level mad prior legality joined from the frozen fold_mad_domain table",
        ],
        deterministic_seed=1234,
        bootstrap="paired site bootstrap, 2,000 draws, seed 1234 (frozen upstream harness default)",
        configured_counts={"E1": 60},
        evaluated_counts={"daily_sites": 45, "monthly_sites": 31, "irrigated": 29, "rainfed": 16},
        note=(
            "Primary reader-facing treatment is heldout_transfer_class_specific_loro. The "
            "pooled LORO/LOSO arms are retained with treatment_provenance marking them "
            "superseded; they must not be plotted as coequal main treatments."
        ),
    )
    print(f"  fig05 E1: {n} rows (45 daily + 31 monthly sites x 6 arms + cohort summaries)")


def build_fig05_e2() -> None:
    """E2 cross-environment transfer package on the closure-corrected 47-site pool.

    Per-site metrics come from the by-irrigation transfer run (all 63 daily
    common-cohort sites, then restricted to the frozen closure pool); cohort
    summaries are the frozen 47-pool medians from the closure-pool
    summary.  The two must agree: the median of each restricted per-site
    column is asserted against the frozen summary row.
    """
    src_ps = E2_TRANSFER / "transfer_comparison_persite.csv"
    src_mstrat = E2_TRANSFER / "evaluation_monthly_metrics_strat.csv"
    src_mcal = E2_RESULTS / "evaluation_monthly_metrics.csv"
    src_frozen_ps = E2_FROZEN_TRANSFER / "transfer_comparison_persite.csv"
    src_pool = E2_FROZEN_POOL / "closure_pool_sites.csv"
    src_refresh = E2_FROZEN_POOL / "transfer_refresh_summary.csv"
    src_pool_meta = E2_FROZEN_POOL / "closure_pool_metadata.json"
    for p in (src_ps, src_mstrat, src_mcal, src_frozen_ps, src_pool, src_refresh, src_pool_meta):
        if not p.exists():
            raise BuildError(f"fig05 E2 source missing: {p}")
    if sha256(src_ps) != sha256(src_frozen_ps):
        raise BuildError(
            "fig05 E2: results-root transfer_comparison_persite.csv differs from the frozen copy "
            f"{src_frozen_ps}"
        )
    pool_meta = json.loads(src_pool_meta.read_text())
    if pool_meta.get("problems"):
        raise BuildError(
            f"fig05 E2: closure pool metadata reports problems {pool_meta['problems']}"
        )

    pool = pd.read_csv(src_pool)
    pool_ids = gate_closure_pool(pool, "fig05 E2")
    pool = pool.set_index("fid")

    ps_all = pd.read_csv(src_ps, index_col=0)
    ps_all.index = ps_all.index.astype(str)
    ps_all.index.name = "site_id"
    require_count(len(ps_all), EXPECTED["E2_daily"], "fig05 E2 daily common cohort (all tiers)")
    ps = restrict_to_pool(ps_all, pool_ids, EXPECTED["E2_pool_daily"], "fig05 E2 daily pool")

    summ = gate_pool_summary(pd.read_csv(src_refresh), "fig05 E2")

    rows = []
    for cfg, (name, prov) in E2_CONFIGS.items():
        need = [f"{cfg}_{k}" for k in ("kge", "r2", "rmse", "bias", "mae")]
        require_columns(ps.reset_index(), need, f"E2 persite config {cfg}")
        check_pool_medians(ps, summ, cfg, "fig05 E2")
        rows.append(
            pd.DataFrame(
                {
                    "experiment": "E2",
                    "legacy_prefix": "e3_",
                    "record_type": "site_metric",
                    "scale": "daily",
                    "site_id": ps.index,
                    "treatment": name,
                    "legacy_config": cfg,
                    "treatment_provenance": prov,
                    "nse": ps[f"{cfg}_r2"].values,
                    "kge": ps[f"{cfg}_kge"].values,
                    "rmse": ps[f"{cfg}_rmse"].values,
                    "mbe": ps[f"{cfg}_bias"].values,
                    "mae": ps[f"{cfg}_mae"].values,
                }
            )
        )
    daily = pd.concat(rows, ignore_index=True)

    # monthly: per-site available for three of four treatments
    mstrat_all = pd.read_csv(src_mstrat)
    mcal_all = pd.read_csv(src_mcal)
    require_count(len(mstrat_all), EXPECTED["E2_monthly_support"], "fig05 E2 monthly strat (all)")
    require_count(len(mcal_all), EXPECTED["E2_monthly_support"], "fig05 E2 monthly cal (all)")
    if set(mstrat_all["fid"]) != set(mcal_all["fid"]):
        raise BuildError("fig05 E2 monthly: site sets differ between transfer and canonical runs")
    mstrat = mstrat_all[mstrat_all["fid"].astype(str).isin(pool_ids)].copy()
    mcal = mcal_all[mcal_all["fid"].astype(str).isin(pool_ids)].copy()
    require_count(len(mstrat), EXPECTED["E2_pool_monthly_support"], "fig05 E2 monthly strat pool")
    require_count(len(mcal), EXPECTED["E2_pool_monthly_support"], "fig05 E2 monthly cal pool")

    def _m(df, suffix, name, prov):
        return pd.DataFrame(
            {
                "experiment": "E2",
                "legacy_prefix": "e3_",
                "record_type": "site_metric",
                "scale": "monthly",
                "site_id": df["fid"].astype(str),
                "treatment": name,
                "legacy_config": suffix,
                "treatment_provenance": prov,
                "nse": df[f"r2_{suffix}"],
                "kge": df[f"kge_{suffix}"],
                "rmse": df[f"rmse_{suffix}"],
                "mbe": df[f"bias_{suffix}"],
                "mae": np.nan,
                "n_paired": df["n"],
                "finite_metric": df[f"kge_{suffix}"].notna(),
            }
        )

    monthly = pd.concat(
        [
            _m(mstrat, "swim", *E2_CONFIGS["ex5_transfer_strat"]),
            _m(mcal, "swim", *E2_CONFIGS["e3_cal"]),
            _m(mcal, "rs", *E2_CONFIGS["ls_ensemble"]),
        ],
        ignore_index=True,
    )
    n_finite = int(monthly[monthly["treatment"] == "local_calibration"]["finite_metric"].sum())
    require_count(n_finite, EXPECTED["E2_pool_monthly_finite"], "fig05 E2 monthly finite-metric")
    fin_cal = mcal.loc[mcal["kge_swim"].notna(), "fid"].astype(str)
    if set(fin_cal) != set(pool.index[pool["monthly_metrics_finite"].astype(bool)].astype(str)):
        raise BuildError("fig05 E2: finite-metric monthly sites differ from the frozen pool flags")
    for cfg, col_df, sfx in (
        ("e3_cal", mcal, "swim"),
        ("ls_ensemble", mcal, "rs"),
        ("ex5_transfer_strat", mstrat, "swim"),
    ):
        frozen = summ[(summ["legacy_config"] == cfg) & (summ["basis"] == "monthly")]
        require_count(len(frozen), 1, f"fig05 E2 frozen monthly summary for {cfg}")
        got = float(col_df[f"kge_{sfx}"].median())
        want = float(frozen["kge_med"].iloc[0])
        if not np.isclose(got, want, rtol=0, atol=1e-6):
            raise BuildError(f"fig05 E2: pool monthly KGE median {cfg} {got!r} != frozen {want!r}")

    summ_out = pd.DataFrame(
        {
            "experiment": "E2",
            "legacy_prefix": "e3_",
            "record_type": "cohort_summary",
            "scale": summ["basis"],
            "site_id": None,
            "treatment": summ["legacy_config"].map(lambda c: E2_CONFIGS[c][0]),
            "legacy_config": summ["legacy_config"],
            "treatment_provenance": summ["config"],
            "nse": summ["r2_med"],
            "kge": summ["kge_med"],
            "rmse": summ["rmse_med"],
            "mbe": summ["bias_med"],
            "mae": summ["mae_med"],
            "n_paired": summ["n_sites"],
            "alpha": np.nan,
            "beta": np.nan,
        }
    )

    out = pd.concat([daily, monthly, summ_out], ignore_index=True)
    n = write_table(out, "fig05_e2_cross_environment_transfer.csv")

    # geography / class lookup, over the full configured 66
    cfg66 = e2_configured()
    region = pool["region"].reindex(cfg66["site_id"]).to_numpy()
    bbox = (cfg66["lon"].between(-125, -66)) & (cfg66["lat"].between(24, 50))
    cfg66["conus"] = np.where(pd.notna(region), region == "CONUS", bbox).astype(bool)
    cfg66["in_daily_cohort"] = cfg66["site_id"].isin(ps_all.index)
    cfg66["in_monthly_cohort"] = cfg66["site_id"].isin(set(mcal_all["fid"].astype(str)))
    cfg66["in_closure_pool"] = cfg66["site_id"].isin(pool_ids)
    cfg66["in_monthly_pool"] = cfg66["site_id"].isin(set(mcal["fid"].astype(str)))
    e1_sites = set(e1_configured()["site_id"])
    cfg66["also_in_e1"] = cfg66["site_id"].isin(e1_sites)
    require_count(int(cfg66["also_in_e1"].sum()), EXPECTED["E1_E2_overlap"], "E1/E2 overlap")
    require_count(
        int((cfg66["also_in_e1"] & cfg66["in_closure_pool"]).sum()),
        EXPECTED["E1_E2_overlap_pool"],
        "E1/E2 overlap within the closure pool",
    )
    require_count(int(cfg66["in_daily_cohort"].sum()), EXPECTED["E2_daily"], "E2 daily cohort flag")
    require_count(int(cfg66["in_closure_pool"].sum()), EXPECTED["E2_pool_daily"], "E2 pool flag")
    n_conus = int(cfg66.loc[cfg66["in_closure_pool"], "conus"].sum())
    n_ex = int((~cfg66.loc[cfg66["in_closure_pool"], "conus"]).sum())
    require_count(n_conus, EXPECTED["E2_pool_conus"], "E2 pool CONUS split")
    require_count(n_ex, EXPECTED["E2_pool_ex_conus"], "E2 pool ex-CONUS split")
    n_equipped = int(cfg66["equipped_for_irrigation"].sum())
    require_count(n_equipped, FIG01_CLASS_COUNTS["e2_irrigated"], "E2 equipped sites")
    n_equipped_pool = int(cfg66.loc[cfg66["in_closure_pool"], "equipped_for_irrigation"].sum())
    require_count(
        n_equipped_pool, pool_meta["irrigation_class_pool"]["irrigated"], "E2 pool equipped sites"
    )
    cfg66.insert(0, "legacy_prefix", "e3_")
    cfg66.insert(0, "experiment", "E2")
    n_lu = write_table(cfg66, "fig05_geography_lookup.csv")

    MANIFEST.add(
        "fig05_e2_cross_environment_transfer.csv",
        rows=n,
        sources={
            k: {"path": str(p), "sha256": sha256(p)}
            for k, p in {
                "transfer_comparison_persite": src_ps,
                "transfer_comparison_persite_frozen": src_frozen_ps,
                "evaluation_monthly_metrics_strat": src_mstrat,
                "evaluation_monthly_metrics_canonical": src_mcal,
                "closure_pool_sites": src_pool,
                "transfer_refresh_summary": src_refresh,
                "closure_pool_metadata": src_pool_meta,
            }.items()
        },
        experiment_mapping={"E2": "legacy e3_*"},
        cohort_key="site_id",
        inclusion_rule=(
            "66 configured E2 sites; 63 with >=10 paired daily observations form the all-tier "
            "daily common cohort; the paper's E2 pool is the 47 closure-corrected (EBR) towers "
            "among them (16 raw-ET towers excluded, decision 2026-09-08). Of the 47, 43 have "
            "paired monthly support and 39 meet the 10-month finite-metric criterion. All four "
            "treatments are scored on identical rs-gated common-support days."
        ),
        temporal_support_rule="Daily common-support days 2013-2025; monthly are sums of paired days (>=20 paired days per month, >=6 paired months).",
        units={"rmse_daily": "mm d-1", "rmse_monthly": "mm month-1", "mbe": "mm d-1 / mm month-1"},
        display_transformations=[
            "legacy config e3_uncal -> generic_defaults",
            "legacy config ex5_transfer_strat -> e1_irrigation_class_transfer",
            "legacy config e3_cal -> local_calibration",
            "legacy config ls_ensemble -> landsat_benchmark",
            "legacy r2 -> nse; legacy bias -> mbe",
            "superseded pooled transfer (ex5_transfer) deliberately excluded from the display package",
            "cohort_summary rows are the frozen 47-pool medians (transfer_refresh_summary.csv); "
            "alpha/beta medians are not frozen for the pool and are null",
        ],
        deterministic_seed=1234,
        configured_counts={"E2": 66},
        evaluated_counts={
            "daily_sites_all_tiers": EXPECTED["E2_daily"],
            "daily_sites": EXPECTED["E2_pool_daily"],
            "monthly_support_sites": EXPECTED["E2_pool_monthly_support"],
            "monthly_finite_metric_sites": EXPECTED["E2_pool_monthly_finite"],
            "conus_daily": n_conus,
            "ex_conus_daily": n_ex,
            "also_in_e1": EXPECTED["E1_E2_overlap_pool"],
        },
        known_gap=(
            "Per-site MONTHLY metrics for the generic-defaults treatment are not persisted "
            "by transfer_ex5_params.py (the uncalibrated arm is a fresh in-process forward "
            "run). Only the daily cohort median is available and is frozen as a record_type="
            "cohort_summary row. Site-level monthly distributions can therefore be drawn "
            "for three of four treatments."
        ),
    )
    MANIFEST.add(
        "fig05_geography_lookup.csv",
        rows=n_lu,
        sources={
            "e2_container_geometry": {"path": str(E2_CONTAINER), "arrays": ["geometry/*"]},
            "e3_irrigation_stratified_param_mapping_metadata": {
                "path": str(E2_MAPPING_META),
                "sha256": sha256(E2_MAPPING_META),
            },
            "closure_pool_sites": {"path": str(src_pool), "sha256": sha256(src_pool)},
        },
        experiment_mapping={"E2": "legacy e3_*"},
        cohort_key="site_id",
        inclusion_rule=(
            "All 66 configured E2 sites, flagged for all-tier daily/monthly cohort membership "
            "and for the closure-corrected evaluation pool."
        ),
        display_transformations=[
            "CONUS flag taken from the frozen closure-pool region where present; otherwise derived from the container lon/lat bounding box (-125..-66 E, 24..50 N)",
            "equipped_for_irrigation is stage 1 of the two-stage classifier, recovered per the frozen param-mapping metadata",
        ],
        units={"lat": "degrees north", "lon": "degrees east"},
        deterministic_seed=None,
        configured_counts={"E2": 66},
        evaluated_counts={
            "daily_all_tiers": EXPECTED["E2_daily"],
            "closure_pool": EXPECTED["E2_pool_daily"],
            "conus_pool": n_conus,
            "ex_conus_pool": n_ex,
            "equipped": n_equipped,
            "equipped_pool": n_equipped_pool,
            "also_in_e1": EXPECTED["E1_E2_overlap"],
            "also_in_e1_pool": EXPECTED["E1_E2_overlap_pool"],
        },
    )
    print(
        f"  fig05 E2: {n} rows ({EXPECTED['E2_pool_daily']} daily x 4 treatments + "
        f"{EXPECTED['E2_pool_monthly_support']} monthly x 3 + {len(summ)} cohort summaries), "
        f"lookup {n_lu} rows ({n_conus} CONUS / {n_ex} ex-CONUS pool; "
        f"{EXPECTED['E1_E2_overlap_pool']} also in E1)"
    )
