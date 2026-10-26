"""Cohort counts, the artifact-prefix-to-paper experiment map, and metric definitions."""

from __future__ import annotations

EXPERIMENT_MAP = {
    "E1": {
        "legacy_prefix": "e2_",
        "repository": "examples/5_Flux_Ensemble",
        "reader_facing_role": "CONUS cropland ET evaluation, reconstruction, ensemble reliability, held-out transfer",
        "configured_n": 60,
        "configured_unit": "cropland flux sites",
    },
    "E2": {
        "legacy_prefix": "e3_",
        "repository": "examples/6_Flux_International",
        "reader_facing_role": "Ten-country cropland evaluation and E1-derived parameter transfer under changed inputs",
        "configured_n": 66,
        "configured_unit": "cropland flux sites",
    },
    "E3": {
        "legacy_prefix": "e4_",
        "repository": "examples/7_Applied_Water",
        "reader_facing_role": "San Luis Valley metered applied-water evaluation",
        "configured_n": 50,
        "configured_unit": "metered fields",
    },
}

# Cohort counts asserted everywhere.  A mismatch stops the affected table.
EXPECTED = {
    "E0_configured": 37,
    "E0_container_fields": 66,
    "E0_pooled_sites": 37,
    "E0_pooled_daily": 49289,
    "E0_pooled_monthly": 1514,
    "E0_pooled_monthly_sites": 34,
    "E0_effect_daily_sites": 37,
    "E0_effect_monthly_sites": 33,
    "E0_iso_daily_wins": 30,
    "E0_iso_monthly_wins": 26,
    "E0_whole_daily_wins": 32,
    "E0_whole_monthly_wins": 28,
    "E1_configured": 60,
    # Evaluation-derived E1 counts, fixed by the benchmark record and cohorts
    # (spread-error counts are read per run).
    "E1_daily": 45,
    "E1_monthly_finite": 29,
    "E1_transfer_daily": 45,
    "E1_transfer_monthly": 31,
    "E1_split_common": 43,
    "E1_pool_retrieval": 4972,
    "E1_pool_between_retrieval": 54300,
    "E1_pool_temporal_common": 59272,
    "E2_configured": 66,
    "E2_daily": 63,
    "E2_monthly_support": 56,
    "E2_monthly_finite": 50,
    # Closure-corrected (EBR) evaluation pool: the paper's E2 numbers; the 16 raw-ET
    # towers are scored but do not appear in the paper.
    "E2_pool_daily": 47,
    "E2_pool_monthly_support": 43,
    "E2_pool_monthly_finite": 39,
    "E2_pool_conus": 29,
    "E2_pool_ex_conus": 18,
    "E2_pool_countries": 8,
    "E2_pool_continents": 3,
    "E3_fields": 50,
    # 408 metered SLV field-years minus the four mislinked 2019 rows in
    # ex7_paths.EXCLUDED_FIELD_YEARS.
    "E3_field_years": 404,
    "E1_E2_overlap": 13,
    "E1_E2_overlap_pool": 10,
}

METRIC_DEFS = {
    "nse": {
        "definition": "Nash-Sutcliffe efficiency, 1 - SSE/SST, computed as sklearn.metrics.r2_score(flux_obs, model) with the observation vector as y_true",
        "direction": "higher is better",
        "units": "dimensionless",
        "legacy_column": "r2_*",
        "verification": "examples/5_Flux_Ensemble/evaluate.py::calc_metrics L228 and examples/6_Flux_International/evaluate.py::calc_metrics L357 both call r2_score(obs, mod) with obs=flux truth; that is 1 - SS_res/SS_tot about the observed mean, i.e. NSE, not a squared Pearson correlation. Pearson r is emitted separately as 'r'. Verified 2026-08-19.",
    },
    "kge": {
        "definition": "Kling-Gupta efficiency, Gupta et al. (2009): 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2), alpha = sd(mod)/sd(obs), beta = mean(mod)/mean(obs)",
        "direction": "higher is better",
        "units": "dimensionless",
        "legacy_column": "kge_*",
        "verification": "examples/5_Flux_Ensemble/evaluate.py L231-233; identical form in examples/6_Flux_International/evaluate.py L360-362.",
    },
    "rmse": {
        "definition": "root mean square error of model against flux ET",
        "direction": "lower is better",
        "units": "mm d-1 daily; mm month-1 monthly",
        "legacy_column": "rmse_*",
        "verification": "sqrt(sklearn.metrics.mean_squared_error(obs, mod))",
    },
    "mbe": {
        "definition": "mean bias error, mean(model - flux)",
        "direction": "zero is best; sign retained",
        "units": "mm d-1 daily; mm month-1 monthly",
        "legacy_column": "bias_*",
        "verification": "float((mod - obs).mean())",
    },
    "r": {
        "definition": "Pearson correlation coefficient between model and flux ET",
        "direction": "higher is better",
        "units": "dimensionless",
        "legacy_column": "r_*",
        "verification": "scipy.stats.pearsonr",
    },
    "mae": {
        "definition": "mean absolute error of model against flux ET",
        "direction": "lower is better",
        "units": "mm d-1 daily; mm month-1 monthly",
        "legacy_column": "mae_*",
        "verification": "mean(|mod - obs|) in transfer_ex5_params.py",
    },
}


# Frozen E1/E2 source-class split used to construct the transfer parameter sets.
# These are the assignment used to build the vectors, not annual irrigation activation.
FIG01_CLASS_COUNTS = {
    "e1_irrigated": 39,
    "e1_rainfed": 21,
    # E2 classes come from the mapping built on the E2 calibration container.
    "e2_irrigated": 14,
    "e2_rainfed": 52,
}
