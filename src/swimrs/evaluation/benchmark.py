"""Pure SWIM-OpenET grouped-benchmark estimators and the paired-record contract.

Owns the estimator math and data contracts shared by the Example 5 evaluator
(``examples/5_Flux_Ensemble/evaluate.py``) and the temporal decomposition
(``examples/5_Flux_Ensemble/overpass_decomposition.py``):

- pooled KGE/RMSE/MBE/r/r^2/slope0 on concatenated exactly-paired cohorts;
- sqrt(n)-weighted site KGE/RMSE/MBE;
- whole-site paired bootstrap with shared draws for both models;
- the ``e1_openet_paired_daily/v1`` record-level daily artifact contract
  (flux/SWIM/OpenET on the exact three-way mask, with OpenET temporal-support
  classes and retrieval/between-retrieval temporal classes);
- the common temporal cohort and the cross-model support interaction
  ``(SWIM - OpenET between_retrieval) - (SWIM - OpenET retrieval)``.

This module is pure: numpy/pandas plus stdlib only. It must never import the
container, the model stack, project configuration, example scripts, or
plotting code.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --- Grouped-benchmark contract (VALIDATION_POLICY: SWIM-OpenET Benchmark
# Aggregation). One constant per threshold/label so gates cannot drift apart.
MIN_OBS_FOR_METRICS = 10
BOOTSTRAP_REPS_DEFAULT = 10_000
BOOTSTRAP_SEED_DEFAULT = 42
PRIMARY_METRICS = ("kge", "rmse", "mbe")
POOLED_METRICS = ("kge", "rmse", "mbe", "r", "r2", "slope0")
WEIGHTED_METRICS = ("kge", "rmse", "mbe")
GROUPED_MODEL_ORDER = ("swim", "openet_ensemble")
MODEL_DISPLAY = {"swim": "SWIM-RS", "openet_ensemble": "OpenET ensemble"}
AGG_POOLED = "pooled_observations"
AGG_WEIGHTED = "sqrt_n_weighted_site_metric"
MOMENT_IDENTITY_ATOL = 1e-12

# Reader-facing benchmark label and stable machine tokens (development
# chronology stays out of scientific display labels; tokens carry provenance).
BENCHMARK_LABELS = {
    "volk": "OpenET ensemble v2.1 3x3",
    "diy": "Container ETf reconstruction (diagnostic)",
}
BENCHMARK_SOURCE_MACHINE_TOKENS = {
    "volk": "openet_flux_2pt1",
    "diy": "container_etf_diy_diagnostic",
}
CONSTRUCTION_TOKENS = {
    "daily": "etf_first_volk_window",
    "monthly": "independent_full_month_openet_totals_v2pt1",
}
FAVORABLE_DIRECTION = {
    "kge": "positive",
    "r": "positive",
    "r2": "positive",
    "rmse": "negative",
    "mbe": "directional_only",
    "slope0": "directional_only",
}
ERROR_METRIC_UNITS = {"daily": "mm d-1", "monthly": "mm month-1"}

GROUPED_METRIC_COLUMNS = (
    "scale",
    "model",
    "metric",
    "estimate",
    "ci95_low",
    "ci95_high",
    "unit",
    "aggregation",
    "n_sites",
    "n_pairs",
    "weight_rule",
    "bootstrap_unit",
    "bootstrap_reps",
    "bootstrap_seed",
    "benchmark",
    "benchmark_source",
    "benchmark_construction",
    "kge_variant",
    "r2_definition",
    "slope_constraint",
)
GROUPED_CONTRAST_COLUMNS = (
    "scale",
    "aggregation",
    "metric",
    "contrast",
    "estimate",
    "ci95_low",
    "ci95_high",
    "unit",
    "n_sites",
    "n_pairs",
    "bootstrap_unit",
    "bootstrap_reps",
    "bootstrap_seed",
    "favorable_direction",
)

GROUPED_FORMULAS = {
    "kge": (
        "1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2); "
        "alpha = std(mod, ddof=0)/std(obs, ddof=0); beta = mean(mod)/mean(obs); "
        "KGE 2009 with population standard deviation"
    ),
    "rmse": "sqrt(mean((mod - obs)^2))",
    "mbe": "mean(mod - obs), signed, modeled minus observed",
    "r": "Pearson correlation(obs, mod)",
    "r2": "squared Pearson correlation (never NSE / sklearn r2_score)",
    "slope0": "sum(obs*mod)/sum(obs^2), least-squares slope forced through the origin",
    "weighted": "sum_i(sqrt(n_i)*Q_i)/sum_i(sqrt(n_i)) for Q in {kge, rmse, mbe}",
}
GROUPED_MASK_DEFINITION = (
    "per site: finite(flux) & finite(swim) & finite(openet_ensemble) on the "
    "exact same dates/months; both models scored on identical support"
)

# --- Paired daily record contract (e1_openet_paired_daily/v1) ---------------
PAIRED_RECORD_SCHEMA_VERSION = "e1_openet_paired_daily/v1"
PAIRED_RECORD_FILENAME = "evaluation_paired_daily_records.csv"
PAIRED_RECORD_COLUMNS = (
    "fid",
    "date",
    "flux_et_mm_d",
    "swim_et_mm_d",
    "openet_et_mm_d",
    "openet_support_class",
    "temporal_class",
)
PAIRED_RECORD_SORT_ORDER = "fid then date, ascending"
SUPPORT_CLASSES = ("capture", "interpolated", "flat_fill")
TEMPORAL_CLASS_RETRIEVAL = "retrieval"
TEMPORAL_CLASS_BETWEEN = "between_retrieval"
TEMPORAL_CLASSES = (TEMPORAL_CLASS_RETRIEVAL, TEMPORAL_CLASS_BETWEEN)
TEMPORAL_CLASS_DEFINITION = (
    "retrieval when openet_support_class == 'capture'; between_retrieval "
    "otherwise. Support classes derive from finite raw ensemble capture "
    "availability BEFORE temporal reconstruction; calibration-target capture "
    "flags never participate."
)
TEMPORAL_INTERACTION_FORMULA = (
    "(swim - openet_ensemble | between_retrieval) - (swim - openet_ensemble | retrieval)"
)
TEMPORAL_ALL_DAYS = "all_days_common"
TEMPORAL_CLASS_ORDER = (TEMPORAL_ALL_DAYS, TEMPORAL_CLASS_RETRIEVAL, TEMPORAL_CLASS_BETWEEN)
TEMPORAL_COHORT_TOKEN = "common_temporal"

TEMPORAL_METRIC_COLUMNS = (
    "temporal_class",
    "cohort",
    "model",
    "metric",
    "estimate",
    "ci95_low",
    "ci95_high",
    "unit",
    "aggregation",
    "n_sites",
    "n_pairs",
    "weight_rule",
    "bootstrap_unit",
    "bootstrap_reps",
    "bootstrap_seed",
    "benchmark_source",
)
TEMPORAL_CONTRAST_COLUMNS = (
    "temporal_class",
    "cohort",
    "aggregation",
    "metric",
    "contrast",
    "estimate",
    "ci95_low",
    "ci95_high",
    "unit",
    "n_sites",
    "n_pairs",
    "bootstrap_unit",
    "bootstrap_reps",
    "bootstrap_seed",
    "benchmark_source",
    "favorable_direction",
)
TEMPORAL_INTERACTION_COLUMNS = (
    "cohort",
    "aggregation",
    "metric",
    "interaction",
    "estimate",
    "ci95_low",
    "ci95_high",
    "unit",
    "n_sites",
    "n_pairs_retrieval",
    "n_pairs_between_retrieval",
    "bootstrap_unit",
    "bootstrap_reps",
    "bootstrap_seed",
    "benchmark_source",
    "favorable_direction",
)
TEMPORAL_ELIGIBILITY_COLUMNS = (
    "fid",
    "n_all_days",
    "n_retrieval",
    "n_interpolated",
    "n_flat_fill",
    "n_between_retrieval",
    "eligible_retrieval",
    "eligible_between_retrieval",
    "in_common_cohort",
    "exclusion_reason",
)


class GroupedEstimationError(ValueError):
    """A grouped estimand could not be computed; names the degenerate condition."""


def _validate_bootstrap_reps(reps):
    """Return an integer bootstrap count, rejecting ambiguous or negative values."""
    if isinstance(reps, bool | np.bool_) or not isinstance(reps, int | np.integer):
        raise GroupedEstimationError("bootstrap repetitions must be a non-negative integer")
    reps = int(reps)
    if reps < 0:
        raise GroupedEstimationError("bootstrap repetitions must be a non-negative integer")
    return reps


@dataclass(frozen=True)
class PairedSiteSeries:
    """Exactly paired flux/SWIM/OpenET-ensemble series for one retained site.

    All three arrays share the ensemble-headline common-support mask
    ``finite(flux) & finite(swim) & finite(openet_ensemble)``; values must be
    finite, one-dimensional, equal length, and at least MIN_OBS_FOR_METRICS.

    ``support_class`` is optional per-date OpenET temporal-support metadata
    (required for the canonical daily paired-record artifact): one of
    ``SUPPORT_CLASSES`` per retained date, aligned one-to-one with ``index``.
    ``'unsupported'`` is prohibited — an unsupported date can never survive
    the three-way mask.
    """

    fid: str
    index: pd.DatetimeIndex
    observed: np.ndarray
    swim: np.ndarray
    openet: np.ndarray
    support_class: tuple | None = None

    def __post_init__(self):
        for name in ("observed", "swim", "openet"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.ndim != 1:
                raise GroupedEstimationError(f"{self.fid}: {name} must be one-dimensional")
            object.__setattr__(self, name, arr)
        idx = self.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(idx)
            object.__setattr__(self, "index", idx)
        n = len(self.observed)
        if not (len(self.swim) == len(self.openet) == len(idx) == n):
            raise GroupedEstimationError(f"{self.fid}: unequal paired array lengths")
        if n < MIN_OBS_FOR_METRICS:
            raise GroupedEstimationError(
                f"{self.fid}: {n} paired observations < MIN_OBS_FOR_METRICS={MIN_OBS_FOR_METRICS}"
            )
        for name in ("observed", "swim", "openet"):
            if not np.isfinite(getattr(self, name)).all():
                raise GroupedEstimationError(
                    f"{self.fid}: non-finite values remain in {name} after masking"
                )
        if idx.has_duplicates:
            raise GroupedEstimationError(f"{self.fid}: duplicate timestamps in paired index")
        if not idx.is_monotonic_increasing:
            raise GroupedEstimationError(f"{self.fid}: paired index is not monotonic")
        if self.support_class is not None:
            support = tuple(str(s) for s in self.support_class)
            object.__setattr__(self, "support_class", support)
            if len(support) != n:
                raise GroupedEstimationError(
                    f"{self.fid}: support metadata length {len(support)} != paired length {n}"
                )
            bad = sorted(set(support) - set(SUPPORT_CLASSES))
            if bad:
                if "unsupported" in bad:
                    raise GroupedEstimationError(
                        f"{self.fid}: 'unsupported' support class is prohibited in the "
                        "paired record (an unsupported date cannot pass the three-way mask)"
                    )
                raise GroupedEstimationError(f"{self.fid}: unknown support classes {bad}")

    @property
    def n(self):
        return len(self.observed)

    def model_series(self, model):
        if model == "swim":
            return self.swim
        if model == "openet_ensemble":
            return self.openet
        raise GroupedEstimationError(f"unknown model token {model!r}")

    def temporal_class(self):
        """Per-date retrieval/between_retrieval labels from the support classes."""
        if self.support_class is None:
            raise GroupedEstimationError(f"{self.fid}: no support metadata on paired record")
        return tuple(
            TEMPORAL_CLASS_RETRIEVAL if s == "capture" else TEMPORAL_CLASS_BETWEEN
            for s in self.support_class
        )


def build_paired_site_series(fid, index, flux, swim, openet, support_class=None):
    """Apply the ensemble-headline three-way mask and build a PairedSiteSeries."""
    flux = np.asarray(flux, dtype=np.float64)
    swim = np.asarray(swim, dtype=np.float64)
    openet = np.asarray(openet, dtype=np.float64)
    mask = np.isfinite(flux) & np.isfinite(swim) & np.isfinite(openet)
    idx = pd.DatetimeIndex(index)[mask]
    support = None
    if support_class is not None:
        support_arr = np.asarray(support_class, dtype=object)
        if support_arr.shape[0] != mask.shape[0]:
            raise GroupedEstimationError(
                f"{fid}: support metadata length {support_arr.shape[0]} != "
                f"unmasked length {mask.shape[0]}"
            )
        support = tuple(support_arr[mask])
    return PairedSiteSeries(
        fid=fid,
        index=idx,
        observed=flux[mask],
        swim=swim[mask],
        openet=openet[mask],
        support_class=support,
    )


def _check_cohort(records):
    if len(records) == 0:
        raise GroupedEstimationError("empty cohort: no retained paired sites")
    fids = [r.fid for r in records]
    if len(set(fids)) != len(fids):
        dupes = sorted({f for f in fids if fids.count(f) > 1})
        raise GroupedEstimationError(f"duplicate site IDs in cohort: {dupes}")


def site_sufficient_stats(observed, modeled):
    """Per-site pooled-metric sufficient statistics.

    Returns ``[n, sum_o, sum_x, sum_o2, sum_x2, sum_ox]`` (float64), from
    which every pooled metric is derivable without re-concatenating arrays.
    """
    o = np.asarray(observed, dtype=np.float64)
    x = np.asarray(modeled, dtype=np.float64)
    return np.array(
        [o.size, o.sum(), x.sum(), (o * o).sum(), (x * x).sum(), (o * x).sum()],
        dtype=np.float64,
    )


def _moment_matrix(records, model):
    return np.vstack([site_sufficient_stats(r.observed, r.model_series(model)) for r in records])


def pooled_metrics_from_moments(moments, context=""):
    """All six pooled metrics from summed sufficient statistics.

    ``moments`` is a 6-vector or an ``(reps, 6)`` array of summed site
    statistics. Degenerate variance/mean conditions raise instead of being
    silently discarded. Returns scalars for a 6-vector input.
    """
    m = np.asarray(moments, dtype=np.float64)
    squeeze = m.ndim == 1
    if squeeze:
        m = m[None, :]
    n = m[:, 0]
    if np.any(n <= 0):
        raise GroupedEstimationError(f"{context}: empty pooled sample (n <= 0)")
    mean_o = m[:, 1] / n
    mean_x = m[:, 2] / n
    var_o = m[:, 3] / n - mean_o**2
    var_x = m[:, 4] / n - mean_x**2
    cov_ox = m[:, 5] / n - mean_o * mean_x
    if np.any(var_o <= 0):
        raise GroupedEstimationError(
            f"{context}: degenerate observed variance (constant observations)"
        )
    if np.any(var_x <= 0):
        raise GroupedEstimationError(f"{context}: degenerate modeled variance (constant model)")
    if np.any(mean_o == 0):
        raise GroupedEstimationError(f"{context}: zero observed mean (KGE beta undefined)")
    if np.any(m[:, 3] == 0):
        raise GroupedEstimationError(f"{context}: zero observed sum of squares (slope0 undefined)")
    r = cov_ox / np.sqrt(var_o * var_x)
    alpha = np.sqrt(var_x / var_o)
    beta = mean_x / mean_o
    out = {
        "kge": 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2),
        "rmse": np.sqrt((m[:, 4] - 2.0 * m[:, 5] + m[:, 3]) / n),
        "mbe": (m[:, 2] - m[:, 1]) / n,
        "r": r,
        "r2": r * r,
        "slope0": m[:, 5] / m[:, 3],
    }
    for k, v in out.items():
        if not np.isfinite(v).all():
            raise GroupedEstimationError(f"{context}: non-finite pooled {k}")
    if squeeze:
        return {k: float(v[0]) for k, v in out.items()}
    return out


def pooled_metrics_direct(observed, modeled):
    """All six pooled metrics by direct computation on concatenated arrays.

    Population standard deviation (ddof=0) inside KGE 2009; ``r2`` is squared
    Pearson correlation; the slope is least-squares with the intercept forced
    to zero.
    """
    o = np.asarray(observed, dtype=np.float64)
    x = np.asarray(modeled, dtype=np.float64)
    if o.size == 0 or o.size != x.size:
        raise GroupedEstimationError("empty or mismatched pooled arrays")
    mean_o, mean_x = o.mean(), x.mean()
    std_o, std_x = o.std(), x.std()
    if std_o <= 0:
        raise GroupedEstimationError("degenerate observed variance (constant observations)")
    if std_x <= 0:
        raise GroupedEstimationError("degenerate modeled variance (constant model)")
    if mean_o == 0:
        raise GroupedEstimationError("zero observed mean (KGE beta undefined)")
    r = float(((o - mean_o) * (x - mean_x)).mean() / (std_o * std_x))
    alpha = std_x / std_o
    beta = mean_x / mean_o
    return {
        "kge": float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)),
        "rmse": float(np.sqrt(np.mean((x - o) ** 2))),
        "mbe": float(np.mean(x - o)),
        "r": r,
        "r2": r * r,
        "slope0": float(np.sum(o * x) / np.sum(o * o)),
    }


def site_secondary_metrics(observed, modeled, min_n=MIN_OBS_FOR_METRICS):
    """Legacy-compatible per-site NSE/KGE/r/RMSE/MBE with unambiguous keys.

    - ``nse`` means 1 - SSE/SST (what evaluate.py's legacy per-site table
      labels ``r2``); squared Pearson is never emitted by this helper.
    - ``mbe`` is signed modeled minus observed (legacy label ``bias``).
    Below ``min_n`` finite pairs, every metric is NaN (secondary-product
    semantics; primary grouped estimators raise instead).
    """
    o = np.asarray(observed, dtype=np.float64)
    x = np.asarray(modeled, dtype=np.float64)
    mask = np.isfinite(o) & np.isfinite(x)
    o, x = o[mask], x[mask]
    keys = ("nse", "kge", "r", "rmse", "mbe")
    if len(o) < min_n:
        return {"n": len(o), **{k: np.nan for k in keys}}
    mean_o, mean_x = o.mean(), x.mean()
    std_o, std_x = o.std(), x.std()
    sst = np.sum((o - mean_o) ** 2)
    sse = np.sum((x - o) ** 2)
    nse = float(1.0 - sse / sst) if sst > 0 else np.nan
    r = (
        float(((o - mean_o) * (x - mean_x)).mean() / (std_o * std_x))
        if std_o > 0 and std_x > 0
        else np.nan
    )
    alpha = std_x / std_o if std_o > 0 else np.nan
    beta = mean_x / mean_o if mean_o > 0 else np.nan
    kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    return {
        "n": len(o),
        "nse": nse,
        "kge": kge,
        "r": r,
        "rmse": float(np.sqrt(np.mean((x - o) ** 2))),
        "mbe": float(np.mean(x - o)),
    }


def pooled_metrics(records, model):
    """Pooled metrics for one model, moment-based with a direct-identity gate.

    The moment-derived values must reproduce direct concatenation to
    MOMENT_IDENTITY_ATOL; a mismatch is a formula defect, not noise.
    """
    _check_cohort(records)
    moments = _moment_matrix(records, model).sum(axis=0)
    from_moments = pooled_metrics_from_moments(moments, context=f"pooled {model}")
    obs = np.concatenate([r.observed for r in records])
    mod = np.concatenate([r.model_series(model) for r in records])
    direct = pooled_metrics_direct(obs, mod)
    for k in POOLED_METRICS:
        if abs(from_moments[k] - direct[k]) > MOMENT_IDENTITY_ATOL:
            raise GroupedEstimationError(
                f"pooled {model} {k}: moment-based {from_moments[k]!r} vs direct "
                f"{direct[k]!r} disagree beyond {MOMENT_IDENTITY_ATOL}"
            )
    return direct


def site_metric_triads(records, model):
    """Per-site KGE/RMSE/MBE arrays (record order) for one model."""
    _check_cohort(records)
    out = {q: np.empty(len(records), dtype=np.float64) for q in WEIGHTED_METRICS}
    for i, rec in enumerate(records):
        m = pooled_metrics_direct(rec.observed, rec.model_series(model))
        for q in WEIGHTED_METRICS:
            out[q][i] = m[q]
    return out


def sqrt_n_weighted_metrics(site_triads, n_obs):
    """sqrt(n)-weighted site KGE/RMSE/MBE: sum(sqrt(n_i)*Q_i)/sum(sqrt(n_i))."""
    w = np.sqrt(np.asarray(n_obs, dtype=np.float64))
    return {q: float(np.sum(w * site_triads[q]) / np.sum(w)) for q in WEIGHTED_METRICS}


def grouped_point_estimates(records):
    """{(aggregation, model, metric): value} on the original cohort."""
    _check_cohort(records)
    n_obs = np.array([r.n for r in records], dtype=np.float64)
    est = {}
    for model in GROUPED_MODEL_ORDER:
        pm = pooled_metrics(records, model)
        for k in POOLED_METRICS:
            est[(AGG_POOLED, model, k)] = pm[k]
        wm = sqrt_n_weighted_metrics(site_metric_triads(records, model), n_obs)
        for k in WEIGHTED_METRICS:
            est[(AGG_WEIGHTED, model, k)] = wm[k]
    return est


def _bootstrap_multiplicities(n_sites, reps, seed):
    """Whole-site draws: index matrix (reps, n_sites) and multiplicity counts."""
    reps = _validate_bootstrap_reps(reps)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_sites, size=(reps, n_sites))
    counts = np.zeros((reps, n_sites), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(reps), n_sites), idx.ravel()), 1.0)
    return idx, counts


def _require_finite(arr, context):
    if not np.isfinite(arr).all():
        raise GroupedEstimationError(f"{context}: non-finite bootstrap replicate")


def bootstrap_grouped_from_counts(records, counts, context=""):
    """Grouped bootstrap replicates from an explicit site multiplicity matrix.

    ``counts`` has shape (reps, n_sites) in record order; passing one matrix
    to several record partitions of the SAME site cohort structurally
    guarantees the partitions share identical site draws. A duplicated site
    contributes its full observation block and its sqrt(n) weight with the
    same multiplicity, and SWIM/OpenET share the identical site draws. Returns
    ``{(aggregation, model_or_'swim_minus_openet', metric): (reps,) array}``.
    """
    _check_cohort(records)
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != len(records):
        raise GroupedEstimationError(
            f"{context}: multiplicity matrix shape {counts.shape} does not match "
            f"{len(records)} sites"
        )
    n_obs = np.array([r.n for r in records], dtype=np.float64)
    w = np.sqrt(n_obs)
    out = {}
    for model in GROUPED_MODEL_ORDER:
        rep_moments = counts @ _moment_matrix(records, model)
        pm = pooled_metrics_from_moments(
            rep_moments, context=f"{context} bootstrap pooled {model}".strip()
        )
        for k in POOLED_METRICS:
            arr = np.asarray(pm[k], dtype=np.float64)
            _require_finite(arr, f"{context} bootstrap pooled {model} {k}".strip())
            out[(AGG_POOLED, model, k)] = arr
        triads = site_metric_triads(records, model)
        den = counts @ w
        for k in WEIGHTED_METRICS:
            arr = (counts @ (w * triads[k])) / den
            _require_finite(arr, f"{context} bootstrap weighted {model} {k}".strip())
            out[(AGG_WEIGHTED, model, k)] = arr
    for agg, metrics in ((AGG_POOLED, POOLED_METRICS), (AGG_WEIGHTED, WEIGHTED_METRICS)):
        for k in metrics:
            out[(agg, "swim_minus_openet", k)] = (
                out[(agg, "swim", k)] - out[(agg, "openet_ensemble", k)]
            )
    return out


def bootstrap_grouped(records, reps, seed, context=""):
    """Whole-site bootstrap replicates for every grouped estimand and contrast.

    Sites are resampled with replacement (draws per replicate = number of
    retained sites); see ``bootstrap_grouped_from_counts`` for the replicate
    semantics.
    """
    _check_cohort(records)
    _, counts = _bootstrap_multiplicities(len(records), reps, seed)
    return bootstrap_grouped_from_counts(records, counts, context=context)


def site_effect_summary(records, reps, seed, scale):
    """Secondary diagnostic: median across sites of per-site SWIM - OpenET.

    Labeled ``median_paired_site_effect``; never the cohort headline. Uses the
    same whole-site paired bootstrap draws as the grouped estimates (same seed,
    same draw shape).
    """
    _check_cohort(records)
    reps = _validate_bootstrap_reps(reps)
    unit_error = ERROR_METRIC_UNITS[scale]
    deltas = {
        q: site_metric_triads(records, "swim")[q]
        - site_metric_triads(records, "openet_ensemble")[q]
        for q in WEIGHTED_METRICS
    }
    idx = None
    if reps > 0:
        idx, _ = _bootstrap_multiplicities(len(records), reps, seed)
    rows = []
    for q in WEIGHTED_METRICS:
        lo = hi = np.nan
        if idx is not None:
            rep_medians = np.median(deltas[q][idx], axis=1)
            _require_finite(rep_medians, f"{scale} site-effect {q}")
            lo, hi = np.percentile(rep_medians, [2.5, 97.5])
        rows.append(
            {
                "scale": scale,
                "aggregation": "median_paired_site_effect",
                "metric": q,
                "contrast": "swim_minus_openet",
                "estimate": float(np.median(deltas[q])),
                "ci95_low": lo,
                "ci95_high": hi,
                "unit": "dimensionless" if q == "kge" else unit_error,
                "n_sites": len(records),
                "bootstrap_unit": "site",
                "bootstrap_reps": reps,
                "bootstrap_seed": int(seed),
            }
        )
    return pd.DataFrame(rows)


def grouped_metric_tables(records, scale, reps, seed, openet_source="volk"):
    """Long-form grouped estimate and contrast tables (artifact contract).

    18 metric rows per scale (6 pooled x 2 models + 3 weighted x 2 models) and
    9 contrast rows, deterministically ordered by aggregation (pooled first),
    model (swim first), and declared metric order. CI fields are null when
    ``reps == 0`` (development runs only).
    """
    _check_cohort(records)
    reps = _validate_bootstrap_reps(reps)
    unit_error = ERROR_METRIC_UNITS[scale]
    est = grouped_point_estimates(records)
    boot = bootstrap_grouped(records, reps, seed, context=scale) if reps > 0 else None
    n_sites = len(records)
    n_pairs = int(sum(r.n for r in records))
    benchmark = BENCHMARK_LABELS[openet_source]
    source_token = BENCHMARK_SOURCE_MACHINE_TOKENS[openet_source]
    construction = CONSTRUCTION_TOKENS[scale]

    def _ci(key):
        if boot is None:
            return np.nan, np.nan
        lo, hi = np.percentile(boot[key], [2.5, 97.5])
        return float(lo), float(hi)

    def _unit(metric):
        return unit_error if metric in ("rmse", "mbe") else "dimensionless"

    metric_rows, contrast_rows = [], []
    for agg, metrics in ((AGG_POOLED, POOLED_METRICS), (AGG_WEIGHTED, WEIGHTED_METRICS)):
        for model in GROUPED_MODEL_ORDER:
            for k in metrics:
                lo, hi = _ci((agg, model, k))
                metric_rows.append(
                    {
                        "scale": scale,
                        "model": model,
                        "metric": k,
                        "estimate": est[(agg, model, k)],
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "unit": _unit(k),
                        "aggregation": agg,
                        "n_sites": n_sites,
                        "n_pairs": n_pairs,
                        "weight_rule": "none" if agg == AGG_POOLED else "sqrt(n_site)",
                        "bootstrap_unit": "site",
                        "bootstrap_reps": int(reps),
                        "bootstrap_seed": int(seed),
                        "benchmark": benchmark,
                        "benchmark_source": source_token,
                        "benchmark_construction": construction,
                        "kge_variant": "2009" if k == "kge" else "",
                        "r2_definition": "pearson_r_squared" if k == "r2" else "",
                        "slope_constraint": "intercept_forced_zero" if k == "slope0" else "",
                    }
                )
        for k in metrics:
            lo, hi = _ci((agg, "swim_minus_openet", k))
            contrast_rows.append(
                {
                    "scale": scale,
                    "aggregation": agg,
                    "metric": k,
                    "contrast": "swim_minus_openet",
                    "estimate": est[(agg, "swim", k)] - est[(agg, "openet_ensemble", k)],
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "unit": _unit(k),
                    "n_sites": n_sites,
                    "n_pairs": n_pairs,
                    "bootstrap_unit": "site",
                    "bootstrap_reps": int(reps),
                    "bootstrap_seed": int(seed),
                    "favorable_direction": FAVORABLE_DIRECTION[k],
                }
            )
    metrics_df = pd.DataFrame(metric_rows, columns=list(GROUPED_METRIC_COLUMNS))
    contrasts_df = pd.DataFrame(contrast_rows, columns=list(GROUPED_CONTRAST_COLUMNS))
    return metrics_df, contrasts_df


def grouped_metadata(records, scale, reps, seed, openet_source, collect_meta):
    """Provenance sidecar content for one grouped evaluation scale."""
    _check_cohort(records)
    reps = _validate_bootstrap_reps(reps)
    meta = {
        "scale": scale,
        "benchmark": BENCHMARK_LABELS[openet_source],
        "benchmark_source": BENCHMARK_SOURCE_MACHINE_TOKENS[openet_source],
        "benchmark_construction": CONSTRUCTION_TOKENS[scale],
        "kge_variant": "2009",
        "formulas": dict(GROUPED_FORMULAS),
        "weight_formula": GROUPED_FORMULAS["weighted"],
        "mask_definition": GROUPED_MASK_DEFINITION,
        "min_obs_for_metrics": MIN_OBS_FOR_METRICS,
        "site_minimum_gate": "90 valid flux days and 3 months with >= 20 valid days",
        "sites": [{"fid": r.fid, "n": int(r.n)} for r in records],
        "n_sites": len(records),
        "n_pairs": int(sum(r.n for r in records)),
        "bootstrap": {
            "unit": "site",
            "reps": int(reps),
            "seed": int(seed),
            "interval": "percentile_2.5_97.5",
        },
    }
    meta.update(collect_meta or {})
    return meta


# ---------------------------------------------------------------------------
# Paired-record serialization contract (e1_openet_paired_daily/v1)
# ---------------------------------------------------------------------------


def paired_records_to_frame(records):
    """Flatten annotated PairedSiteSeries into the v1 record frame.

    Every record must carry support metadata; the frame is deterministically
    sorted by fid then date with the exact contract column order.
    """
    _check_cohort(records)
    for rec in records:
        if rec.support_class is None:
            raise GroupedEstimationError(
                f"{rec.fid}: paired record has no support metadata; the canonical "
                "daily record artifact requires per-date OpenET support classes"
            )
    frames = []
    for rec in sorted(records, key=lambda r: r.fid):
        frames.append(
            pd.DataFrame(
                {
                    "fid": rec.fid,
                    "date": rec.index,
                    "flux_et_mm_d": rec.observed,
                    "swim_et_mm_d": rec.swim,
                    "openet_et_mm_d": rec.openet,
                    "openet_support_class": list(rec.support_class),
                    "temporal_class": list(rec.temporal_class()),
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)[list(PAIRED_RECORD_COLUMNS)]
    validate_paired_record_frame(frame)
    return frame


def validate_paired_record_frame(frame):
    """Enforce every v1 row- and cohort-level invariant; return count summary."""
    if list(frame.columns) != list(PAIRED_RECORD_COLUMNS):
        raise GroupedEstimationError(
            f"paired record schema violation: columns {list(frame.columns)} != "
            f"{list(PAIRED_RECORD_COLUMNS)} (unrecognized or missing columns are "
            f"not accepted under {PAIRED_RECORD_SCHEMA_VERSION})"
        )
    if len(frame) == 0:
        raise GroupedEstimationError("paired record frame is empty")
    if frame["fid"].isna().any():
        raise GroupedEstimationError("paired record has null fid values")
    dates = frame["date"]
    if not pd.api.types.is_datetime64_any_dtype(dates):
        raise GroupedEstimationError("paired record 'date' column is not datetime-typed")
    if (dates != dates.dt.normalize()).any():
        raise GroupedEstimationError("paired record dates carry time-of-day components")
    for col in ("flux_et_mm_d", "swim_et_mm_d", "openet_et_mm_d"):
        vals = frame[col].to_numpy()
        if not np.issubdtype(vals.dtype, np.floating):
            raise GroupedEstimationError(f"paired record column {col} is not float-typed")
        if not np.isfinite(vals).all():
            raise GroupedEstimationError(f"paired record column {col} has non-finite values")
    support = frame["openet_support_class"]
    if (support == "unsupported").any():
        raise GroupedEstimationError(
            "paired record contains prohibited 'unsupported' support class"
        )
    bad_support = sorted(set(support.unique()) - set(SUPPORT_CLASSES))
    if bad_support:
        raise GroupedEstimationError(f"paired record has unknown support classes {bad_support}")
    temporal = frame["temporal_class"]
    bad_temporal = sorted(set(temporal.unique()) - set(TEMPORAL_CLASSES))
    if bad_temporal:
        raise GroupedEstimationError(f"paired record has unknown temporal classes {bad_temporal}")
    expected_temporal = np.where(
        support.to_numpy() == "capture", TEMPORAL_CLASS_RETRIEVAL, TEMPORAL_CLASS_BETWEEN
    )
    if (temporal.to_numpy() != expected_temporal).any():
        raise GroupedEstimationError(
            "paired record temporal_class is inconsistent with openet_support_class "
            f"({TEMPORAL_CLASS_DEFINITION})"
        )
    if frame.duplicated(subset=["fid", "date"]).any():
        dupes = frame.loc[frame.duplicated(subset=["fid", "date"]), ["fid", "date"]]
        raise GroupedEstimationError(
            f"paired record has duplicate (fid, date) rows (first: "
            f"{dupes.iloc[0]['fid']} {dupes.iloc[0]['date'].date()})"
        )
    order = frame.sort_values(["fid", "date"], kind="mergesort").index
    if not (order.to_numpy() == np.arange(len(frame))).all():
        raise GroupedEstimationError(
            f"paired record rows are not sorted by {PAIRED_RECORD_SORT_ORDER}"
        )
    counts = frame.groupby("fid", sort=True).size()
    thin = counts[counts < MIN_OBS_FOR_METRICS]
    if not thin.empty:
        raise GroupedEstimationError(
            f"paired record sites below MIN_OBS_FOR_METRICS={MIN_OBS_FOR_METRICS}: "
            f"{sorted(thin.index)}"
        )
    support_counts = support.value_counts().to_dict()
    n_retrieval = int((temporal == TEMPORAL_CLASS_RETRIEVAL).sum())
    return {
        "n_sites": int(counts.size),
        "n_rows": int(len(frame)),
        "n_retrieval": n_retrieval,
        "n_between_retrieval": int(len(frame) - n_retrieval),
        "support_class_counts": {c: int(support_counts.get(c, 0)) for c in SUPPORT_CLASSES},
    }


def write_paired_record_frame(frame, path):
    """Serialize the validated record frame deterministically and atomically.

    Floats at round-trip precision (%.17g), dates as YYYY-MM-DD, fixed column
    order, rows already sorted by fid then date.
    """
    validate_paired_record_frame(frame)
    out = frame.copy()
    out["date"] = pd.DatetimeIndex(out["date"]).strftime("%Y-%m-%d")
    tmp = f"{path}.tmp"
    out.to_csv(tmp, index=False, float_format="%.17g")
    os.replace(tmp, path)
    return path


def read_paired_record_frame(path):
    """Load and fully validate a v1 paired-record CSV (round-trip floats)."""
    frame = pd.read_csv(
        path,
        dtype={
            "fid": str,
            "flux_et_mm_d": np.float64,
            "swim_et_mm_d": np.float64,
            "openet_et_mm_d": np.float64,
            "openet_support_class": str,
            "temporal_class": str,
        },
        float_precision="round_trip",
    )
    if list(frame.columns) != list(PAIRED_RECORD_COLUMNS):
        raise GroupedEstimationError(
            f"{path}: columns {list(frame.columns)} != {list(PAIRED_RECORD_COLUMNS)}"
        )
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")
    validate_paired_record_frame(frame)
    return frame


def paired_records_from_frame(frame, fids=None, temporal_class=None):
    """Rebuild PairedSiteSeries from a validated record frame.

    ``fids`` restricts to a site subset (all frame sites otherwise);
    ``temporal_class`` restricts to one temporal class. Records are returned
    in ascending fid order; a resulting site below MIN_OBS_FOR_METRICS raises
    (temporal cohorts must be gated with ``temporal_cohort_from_frame`` before
    subsetting).
    """
    sub = frame
    if temporal_class is not None:
        if temporal_class not in TEMPORAL_CLASSES:
            raise GroupedEstimationError(f"unknown temporal class {temporal_class!r}")
        sub = sub[sub["temporal_class"] == temporal_class]
    if fids is not None:
        want = sorted(fids)
        have = set(sub["fid"].unique())
        missing = [f for f in want if f not in have]
        if missing:
            raise GroupedEstimationError(
                f"sites missing from paired record"
                f"{' (' + temporal_class + ')' if temporal_class else ''}: {missing}"
            )
        sub = sub[sub["fid"].isin(want)]
    records = []
    for fid, grp in sub.groupby("fid", sort=True):
        records.append(
            PairedSiteSeries(
                fid=fid,
                index=pd.DatetimeIndex(grp["date"]),
                observed=grp["flux_et_mm_d"].to_numpy(),
                swim=grp["swim_et_mm_d"].to_numpy(),
                openet=grp["openet_et_mm_d"].to_numpy(),
                support_class=tuple(grp["openet_support_class"]),
            )
        )
    return tuple(records)


# ---------------------------------------------------------------------------
# Temporal cohort and cross-model support interaction
# ---------------------------------------------------------------------------


def temporal_cohort_from_frame(frame, min_obs=MIN_OBS_FOR_METRICS):
    """Per-site temporal eligibility plus the common temporal cohort.

    A site enters the common cohort only with at least ``min_obs`` paired
    dates in BOTH the retrieval and between_retrieval classes. Returns
    ``(eligibility_df, common_fids)`` with deterministic fid ordering.
    """
    validate_paired_record_frame(frame)
    rows = []
    common = []
    for fid, grp in frame.groupby("fid", sort=True):
        support = grp["openet_support_class"]
        n_all = len(grp)
        n_ret = int((support == "capture").sum())
        n_interp = int((support == "interpolated").sum())
        n_flat = int((support == "flat_fill").sum())
        n_btw = n_interp + n_flat
        ok_ret = n_ret >= min_obs
        ok_btw = n_btw >= min_obs
        reasons = []
        if not ok_ret:
            reasons.append(f"n_retrieval={n_ret} < {min_obs}")
        if not ok_btw:
            reasons.append(f"n_between_retrieval={n_btw} < {min_obs}")
        if ok_ret and ok_btw:
            common.append(fid)
        rows.append(
            {
                "fid": fid,
                "n_all_days": n_all,
                "n_retrieval": n_ret,
                "n_interpolated": n_interp,
                "n_flat_fill": n_flat,
                "n_between_retrieval": n_btw,
                "eligible_retrieval": ok_ret,
                "eligible_between_retrieval": ok_btw,
                "in_common_cohort": ok_ret and ok_btw,
                "exclusion_reason": "; ".join(reasons),
            }
        )
    eligibility = pd.DataFrame(rows, columns=list(TEMPORAL_ELIGIBILITY_COLUMNS))
    if not common:
        raise GroupedEstimationError(
            "no common temporal cohort: no site has "
            f">= {min_obs} paired dates in both temporal classes"
        )
    return eligibility, tuple(common)


def temporal_class_records(frame, fids):
    """Class-partitioned PairedSiteSeries for one site cohort, identical order.

    Returns ``{TEMPORAL_ALL_DAYS: records, 'retrieval': records,
    'between_retrieval': records}``; every partition holds the same fids in
    the same ascending order, each site's retrieval and between_retrieval
    subsets are disjoint, and their union is the site's all-days record.
    """
    fids = tuple(sorted(fids))
    parts = {
        TEMPORAL_ALL_DAYS: paired_records_from_frame(frame, fids=fids),
        TEMPORAL_CLASS_RETRIEVAL: paired_records_from_frame(
            frame, fids=fids, temporal_class=TEMPORAL_CLASS_RETRIEVAL
        ),
        TEMPORAL_CLASS_BETWEEN: paired_records_from_frame(
            frame, fids=fids, temporal_class=TEMPORAL_CLASS_BETWEEN
        ),
    }
    for name, records in parts.items():
        if tuple(r.fid for r in records) != fids:
            raise GroupedEstimationError(f"{name}: partition fid order differs from cohort order")
    for a, r, b in zip(
        parts[TEMPORAL_ALL_DAYS],
        parts[TEMPORAL_CLASS_RETRIEVAL],
        parts[TEMPORAL_CLASS_BETWEEN],
        strict=True,
    ):
        ret_idx = r.index
        btw_idx = b.index
        if len(ret_idx.intersection(btw_idx)) != 0:
            raise GroupedEstimationError(f"{a.fid}: retrieval and between_retrieval overlap")
        if not ret_idx.union(btw_idx).equals(a.index):
            raise GroupedEstimationError(
                f"{a.fid}: retrieval union between_retrieval != all paired days"
            )
    return parts


@dataclass(frozen=True)
class TemporalDecomposition:
    """Grouped temporal estimates, contrasts, interactions, and eligibility."""

    site_eligibility: pd.DataFrame
    grouped_metrics: pd.DataFrame
    grouped_contrasts: pd.DataFrame
    interactions: pd.DataFrame
    common_cohort: tuple
    class_counts: dict


def temporal_decomposition(frame, reps, seed, openet_source="volk", min_obs=MIN_OBS_FOR_METRICS):
    """Retrieval/between-retrieval grouped estimates and the support interaction.

    One whole-site bootstrap multiplicity matrix (from ``seed``) is shared
    across both models, every metric, and all three temporal partitions of the
    common cohort, so within-class contrasts and the cross-model interaction
    are paired at the site-draw level. CI fields are null only when ``reps``
    is explicitly zero.
    """
    reps = _validate_bootstrap_reps(reps)
    eligibility, common = temporal_cohort_from_frame(frame, min_obs=min_obs)
    parts = temporal_class_records(frame, common)
    unit_error = ERROR_METRIC_UNITS["daily"]
    source_token = BENCHMARK_SOURCE_MACHINE_TOKENS[openet_source]
    n_sites = len(common)

    counts = None
    if reps > 0:
        _, counts = _bootstrap_multiplicities(n_sites, reps, seed)

    est = {}
    boot = {}
    n_pairs = {}
    for name, records in parts.items():
        est[name] = grouped_point_estimates(records)
        n_pairs[name] = int(sum(r.n for r in records))
        if counts is not None:
            boot[name] = bootstrap_grouped_from_counts(records, counts, context=name)

    def _ci(name, key):
        if counts is None:
            return np.nan, np.nan
        lo, hi = np.percentile(boot[name][key], [2.5, 97.5])
        return float(lo), float(hi)

    def _unit(metric):
        return unit_error if metric in ("rmse", "mbe") else "dimensionless"

    metric_rows, contrast_rows = [], []
    for name in TEMPORAL_CLASS_ORDER:
        for agg, metrics in ((AGG_POOLED, POOLED_METRICS), (AGG_WEIGHTED, WEIGHTED_METRICS)):
            for model in GROUPED_MODEL_ORDER:
                for k in metrics:
                    lo, hi = _ci(name, (agg, model, k))
                    metric_rows.append(
                        {
                            "temporal_class": name,
                            "cohort": TEMPORAL_COHORT_TOKEN,
                            "model": model,
                            "metric": k,
                            "estimate": est[name][(agg, model, k)],
                            "ci95_low": lo,
                            "ci95_high": hi,
                            "unit": _unit(k),
                            "aggregation": agg,
                            "n_sites": n_sites,
                            "n_pairs": n_pairs[name],
                            "weight_rule": "none" if agg == AGG_POOLED else "sqrt(n_site)",
                            "bootstrap_unit": "site",
                            "bootstrap_reps": int(reps),
                            "bootstrap_seed": int(seed),
                            "benchmark_source": source_token,
                        }
                    )
            for k in metrics:
                lo, hi = _ci(name, (agg, "swim_minus_openet", k))
                contrast_rows.append(
                    {
                        "temporal_class": name,
                        "cohort": TEMPORAL_COHORT_TOKEN,
                        "aggregation": agg,
                        "metric": k,
                        "contrast": "swim_minus_openet",
                        "estimate": (
                            est[name][(agg, "swim", k)] - est[name][(agg, "openet_ensemble", k)]
                        ),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "unit": _unit(k),
                        "n_sites": n_sites,
                        "n_pairs": n_pairs[name],
                        "bootstrap_unit": "site",
                        "bootstrap_reps": int(reps),
                        "bootstrap_seed": int(seed),
                        "benchmark_source": source_token,
                        "favorable_direction": FAVORABLE_DIRECTION[k],
                    }
                )

    interaction_rows = []
    for agg in (AGG_POOLED, AGG_WEIGHTED):
        for k in PRIMARY_METRICS:
            point = (
                est[TEMPORAL_CLASS_BETWEEN][(agg, "swim", k)]
                - est[TEMPORAL_CLASS_BETWEEN][(agg, "openet_ensemble", k)]
            ) - (
                est[TEMPORAL_CLASS_RETRIEVAL][(agg, "swim", k)]
                - est[TEMPORAL_CLASS_RETRIEVAL][(agg, "openet_ensemble", k)]
            )
            lo = hi = np.nan
            if counts is not None:
                # same replicate arrays from the shared multiplicity matrix:
                # the interaction is paired at the site-draw level
                reps_arr = (
                    boot[TEMPORAL_CLASS_BETWEEN][(agg, "swim_minus_openet", k)]
                    - boot[TEMPORAL_CLASS_RETRIEVAL][(agg, "swim_minus_openet", k)]
                )
                _require_finite(reps_arr, f"interaction {agg} {k}")
                lo, hi = (float(v) for v in np.percentile(reps_arr, [2.5, 97.5]))
            interaction_rows.append(
                {
                    "cohort": TEMPORAL_COHORT_TOKEN,
                    "aggregation": agg,
                    "metric": k,
                    "interaction": "between_retrieval_minus_retrieval_of_swim_minus_openet",
                    "estimate": point,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "unit": _unit(k),
                    "n_sites": n_sites,
                    "n_pairs_retrieval": n_pairs[TEMPORAL_CLASS_RETRIEVAL],
                    "n_pairs_between_retrieval": n_pairs[TEMPORAL_CLASS_BETWEEN],
                    "bootstrap_unit": "site",
                    "bootstrap_reps": int(reps),
                    "bootstrap_seed": int(seed),
                    "benchmark_source": source_token,
                    "favorable_direction": FAVORABLE_DIRECTION[k],
                }
            )

    return TemporalDecomposition(
        site_eligibility=eligibility,
        grouped_metrics=pd.DataFrame(metric_rows, columns=list(TEMPORAL_METRIC_COLUMNS)),
        grouped_contrasts=pd.DataFrame(contrast_rows, columns=list(TEMPORAL_CONTRAST_COLUMNS)),
        interactions=pd.DataFrame(interaction_rows, columns=list(TEMPORAL_INTERACTION_COLUMNS)),
        common_cohort=tuple(common),
        class_counts={name: n_pairs[name] for name in TEMPORAL_CLASS_ORDER},
    )
