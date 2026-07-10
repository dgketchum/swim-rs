"""SMAP surface-soil-moisture (SSM) anomaly observation construction for WP-C4.

Pure, dependency-light helpers shared by three consumers so the anomaly is defined
*identically* on the observation and prediction sides:

  1. ``PestBuilder`` (build time) — turns a SMAP L3 daily series into the per-field
     observation anomaly baked into the .pst, and the capture-day index shipped to
     workers.
  2. the auto-generated ``custom_forward_run.py`` (worker, per realization) — turns
     the modeled surface-layer depletion into the prediction anomaly on the same
     capture days.
  3. the unit test.

The SSM calibration target is the SMAP L3 *satellite* product only. In-situ SCAN
theta is never read here — it stays validation-only (see examples/8_Soil_Moisture).

Anomaly definition (both sides): subtract the day-of-year climatology (mean over the
matched capture days), exactly as ``examples/8_Soil_Moisture/evaluate._deseasonalize``.
Amplitude is preserved (no divide by sigma) so the objective penalizes a model whose
surface-moisture variance has collapsed (the WP-A irrigated-site failure mode).
"""

from __future__ import annotations

import numpy as np

GROW_MONTHS = (4, 5, 6, 7, 8, 9, 10)  # Apr-Oct, matching the rest of Example 8


def recursive_soil_water_index(
    values: np.ndarray,
    times: np.ndarray,
    characteristic_time_days: float = 20.0,
) -> np.ndarray:
    """Convert intermittent surface moisture to a root-zone soil-water index.

    Implements the recursive exponential filter of Albergel et al. (2008),
    including the time-step-dependent gain used for irregular satellite
    captures. The fixed characteristic time is an externally chosen physical
    assumption; this helper contains no in-situ soil-moisture fitting.

    The returned array has the same shape as ``values``. It contains SWI only
    where ``values`` is finite and NaN elsewhere, which preserves the original
    satellite capture calendar.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times)
    if values.ndim != 1 or times.ndim != 1 or values.shape != times.shape:
        raise ValueError("values and times must be one-dimensional arrays of equal length")
    if not np.isfinite(characteristic_time_days) or characteristic_time_days <= 0:
        raise ValueError("characteristic_time_days must be finite and greater than zero")

    try:
        times_ns = times.astype("datetime64[ns]").astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("times must be convertible to datetime64") from exc
    if np.any(np.diff(times_ns) <= 0):
        raise ValueError("times must be strictly increasing")

    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size == 0:
        return out

    first = valid[0]
    out[first] = values[first]
    previous_swi = values[first]
    previous_gain = 1.0
    previous_time = times_ns[first]
    day_ns = 86_400_000_000_000.0

    for idx in valid[1:]:
        dt_days = (times_ns[idx] - previous_time) / day_ns
        decay = np.exp(-dt_days / characteristic_time_days)
        gain = previous_gain / (previous_gain + decay)
        previous_swi = previous_swi + gain * (values[idx] - previous_swi)
        out[idx] = previous_swi
        previous_gain = gain
        previous_time = times_ns[idx]
    return out


def deseasonalize_doy(values: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """Remove the day-of-year climatology: subtract, per DOY, the mean over samples.

    Matches ``evaluate._deseasonalize`` (pandas groupby(dayofyear) - transform mean)
    when applied to the same set of days. A DOY with a single sample yields 0 (that
    day equals its own mean), identical to the pandas path.
    """
    values = np.asarray(values, dtype=float)
    doy = np.asarray(doy, dtype=int)
    out = values.copy()
    for d in np.unique(doy):
        m = doy == d
        out[m] = values[m] - values[m].mean()
    return out


def surface_theta_from_depl_ze(depl_ze: np.ndarray, ze: float = 0.1) -> np.ndarray:
    """FAO-56 surface-layer volumetric water content proxy (m3/m3), offset-free.

    theta_surf = theta_fc - depl_ze / (1000*ze). The per-field theta_fc offset cancels
    under deseasonalization, so we return only the depletion term ``-depl_ze/(1000*ze)``
    (m3/m3): a drier surface (larger depletion) gives a more negative value. ``ze`` is
    the evaporation-layer depth in metres (FAO-56 default 0.10 m); 1000*ze converts the
    mm depletion to a volumetric fraction over that layer.
    """
    depl_ze = np.asarray(depl_ze, dtype=float)
    return -depl_ze / (1000.0 * float(ze))


def build_ssm_observation(
    smap_values: np.ndarray,
    months: np.ndarray,
    years: np.ndarray,
    doy: np.ndarray,
    grow_months: tuple = GROW_MONTHS,
    min_seasons: int = 2,
    gated: bool = False,
):
    """Build the per-field SMAP anomaly observation on the full daily model grid.

    Parameters
    ----------
    smap_values : (n_days,) SMAP L3 surface soil moisture aligned to the model date
        grid; NaN on days with no valid L3 retrieval (kept, not dropped).
    months, years, doy : (n_days,) calendar fields of the model date grid.
    grow_months : growing-season months to weight (Apr-Oct).
    min_seasons : require at least this many distinct growing-season years with a valid
        SMAP capture before the site is weighted (record-length safety gate).
    gated : if True (footprint-compromised pixel, e.g. Vallecitos), return no obs.

    Returns
    -------
    anom_full : (n_days,) anomaly on capture days, NaN elsewhere.
    capture_idx : int indices (into the grid) of the weighted capture days.
    used : bool, whether the site contributes any weighted SSM obs.
    """
    n = len(smap_values)
    anom_full = np.full(n, np.nan, dtype=float)
    if gated:
        return anom_full, np.array([], dtype=int), False

    smap_values = np.asarray(smap_values, dtype=float)
    months = np.asarray(months, dtype=int)
    years = np.asarray(years, dtype=int)
    doy = np.asarray(doy, dtype=int)

    grow = np.isin(months, np.asarray(grow_months, dtype=int))
    valid = np.isfinite(smap_values) & grow
    if valid.sum() < 30 or np.unique(years[valid]).size < int(min_seasons):
        return anom_full, np.array([], dtype=int), False

    idx = np.where(valid)[0]
    anom = deseasonalize_doy(smap_values[idx], doy[idx])
    anom_full[idx] = anom
    return anom_full, idx, True


def model_ssm_prediction(
    depl_ze: np.ndarray,
    doy: np.ndarray,
    capture_idx: np.ndarray,
    ze: float = 0.1,
) -> np.ndarray:
    """Modeled SSM prediction anomaly on the full daily grid (worker side).

    Deseasonalizes the surface-layer volumetric proxy on the *same* capture days as
    the observation, so obs and pred share one anomaly definition. Zero off capture
    days (weight is zero there anyway).
    """
    depl_ze = np.asarray(depl_ze, dtype=float)
    n = len(depl_ze)
    pred_full = np.zeros(n, dtype=float)
    capture_idx = np.asarray(capture_idx, dtype=int)
    if capture_idx.size == 0:
        return pred_full
    theta = surface_theta_from_depl_ze(depl_ze[capture_idx], ze=ze)
    pred_full[capture_idx] = deseasonalize_doy(theta, np.asarray(doy, dtype=int)[capture_idx])
    return pred_full
