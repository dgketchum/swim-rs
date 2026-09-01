"""Shared daily-benchmark reconstruction for sparse capture-date ET records.

Implements the binding ETf-first construction for OpenET-style daily
benchmarks::

    ETf_i = ET_i / ETo_i                       (at capture dates i)
    ET_t  = interp(ETf)_t * ETo_t              (at daily dates t)

per OpenET's documented temporal method ("The fraction of reference ET is
calculated for each overpass, linearly interpolated for the days in between,
and multiplied by daily reference ET values", https://etdata.org/methods/).

Temporal support follows the Volk et al. (2024, Nature Water,
https://www.nature.com/articles/s44221-023-00181-7) rule — "linearly
interpolating between the nearest unmasked (cloud free) pixels in time within
+/-32 days" — implemented with the exact operational semantics of
``openet.core.interpolate.daily`` (``interp_days=32``): for each day ``t`` the
previous anchor is the nearest capture in ``[t-32, t-1]`` and the next anchor
the nearest in ``[t, t+32]``; when both exist the value is linear in time
between them, when only one exists that value is held flat (including up to
32 days beyond the first/last capture), and when neither exists the day is
unsupported (NaN). Applied here to delivered, spatially aggregated site
series — a "site-series ETf reconstruction using the Volk et al.
temporal-support rule", not a native pixel-level reproduction.

Direct interpolation of sparse ET in time is NOT valid: interp(ETf*ETo) !=
interp(ETf)*ETo whenever ETo varies between captures.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

VOLK_WINDOW_DAYS = 32


class BenchmarkConstructionError(ValueError):
    """Raised when a daily benchmark cannot be constructed validly."""


def _as_day_ints(index):
    return index.values.astype("datetime64[D]").astype(np.int64)


def _window_anchors(day_ints, capture_ints, window_days):
    """Previous/next capture anchors for each day under the Volk window rule.

    Returns ``(prev_idx, next_idx, prev_ok, next_ok)`` where the previous
    anchor is the nearest capture in ``[t - window, t - 1]`` and the next
    anchor the nearest capture in ``[t, t + window]`` (a capture day is its
    own next anchor, so capture-date identity is structural).
    """
    next_idx = np.searchsorted(capture_ints, day_ints, side="left")
    prev_idx = next_idx - 1
    n = len(capture_ints)
    prev_ok = prev_idx >= 0
    prev_ok &= day_ints - capture_ints[np.clip(prev_idx, 0, n - 1)] <= window_days
    next_ok = next_idx < n
    next_ok &= capture_ints[np.clip(next_idx, 0, n - 1)] - day_ints <= window_days
    return prev_idx, next_idx, prev_ok, next_ok


def _validate_daily_index(index, label, name):
    if not isinstance(index, pd.DatetimeIndex):
        raise BenchmarkConstructionError(f"{label}: {name} index must be a DatetimeIndex")
    if index.has_duplicates:
        raise BenchmarkConstructionError(f"{label}: duplicate dates in {name}")
    if not index.is_monotonic_increasing:
        raise BenchmarkConstructionError(f"{label}: {name} index is not monotonic")


@dataclass(frozen=True)
class BenchmarkReconstruction:
    """Result of a daily ETf-first benchmark reconstruction."""

    daily_et: pd.Series
    daily_etf: pd.Series
    capture_dates: pd.DatetimeIndex
    capture_et: pd.Series
    capture_etf: pd.Series
    support_class: pd.Series
    support_start: pd.Timestamp
    support_end: pd.Timestamp
    n_captures: int
    n_interpolated_days: int
    n_flat_filled_days: int
    n_unsupported_days: int
    max_capture_gap_days: int
    identity_max_abs_err: float
    eto_name: str
    window_days: int
    label: str = field(default="")


def reconstruct_daily_benchmark(
    *,
    capture_series,
    capture_space,
    eto,
    eto_name,
    index=None,
    identity_tol=1e-10,
    window_days=VOLK_WINDOW_DAYS,
    label="",
):
    """Reconstruct a daily ET benchmark from sparse captures, ETf-first.

    Parameters
    ----------
    capture_series : pd.Series
        Sparse capture-date values (ET in mm/day, or ETf, per
        ``capture_space``). Non-finite entries are dropped; every finite
        capture anchors the interpolation regardless of any other data
        availability on that day.
    capture_space : {'et', 'etf'}
        Space of ``capture_series``. ``'et'`` divides by same-day ``eto``
        before interpolating; ``'etf'`` interpolates directly.
    eto : pd.Series
        Daily reference ET. The SAME series is used for the capture-date
        division and the daily multiplication (single-argument by design).
        Must cover every capture date with a finite, strictly positive value
        and every target day.
    eto_name : str
        Provenance name of the ETo series, recorded on the result.
    index : pd.DatetimeIndex, optional
        Target daily index (default: ``eto.index``).
    identity_tol : float
        Maximum allowed |reconstructed - supplied| on capture dates.
    window_days : int
        Volk temporal-support window (default 32).
    label : str
        Identifier used in error messages (e.g. "US-Ne1:ensemble_mean").
    """
    if capture_space not in ("et", "etf"):
        raise BenchmarkConstructionError(f"{label}: capture_space must be 'et' or 'etf'")
    if not isinstance(eto, pd.Series):
        raise BenchmarkConstructionError(f"{label}: eto must be a pd.Series")
    _validate_daily_index(eto.index, label, "eto")

    if index is None:
        index = eto.index
    _validate_daily_index(index, label, "target")
    missing_days = index.difference(eto.index)
    if len(missing_days):
        raise BenchmarkConstructionError(
            f"{label}: eto missing {len(missing_days)} target days "
            f"(first: {missing_days[0].date()})"
        )

    captures = pd.Series(capture_series, dtype="float64")
    captures = captures[np.isfinite(captures.values)]
    if captures.empty:
        raise BenchmarkConstructionError(f"{label}: no finite capture values")
    if captures.index.has_duplicates:
        dupes = captures.index[captures.index.duplicated()].unique()
        raise BenchmarkConstructionError(
            f"{label}: duplicate capture dates (first: {dupes[0].date()})"
        )
    captures = captures.sort_index()

    outside = captures.index.difference(eto.index)
    if len(outside):
        raise BenchmarkConstructionError(
            f"{label}: {len(outside)} capture dates have no eto "
            f"(first: {outside[0].date()}) — do not fill"
        )
    cap_eto = eto.loc[captures.index].astype("float64")
    bad_eto = cap_eto[~(np.isfinite(cap_eto.values) & (cap_eto.values > 0.0))]
    if len(bad_eto):
        raise BenchmarkConstructionError(
            f"{label}: non-finite or non-positive eto on {len(bad_eto)} capture "
            f"dates (first: {bad_eto.index[0].date()}) — do not fill"
        )

    if capture_space == "et":
        capture_et = captures
        capture_etf = captures / cap_eto
    else:
        capture_etf = captures
        capture_et = captures * cap_eto

    cap_ints = _as_day_ints(captures.index)
    day_ints = _as_day_ints(index)
    prev_idx, next_idx, prev_ok, next_ok = _window_anchors(day_ints, cap_ints, window_days)

    n = len(cap_ints)
    etf_vals = capture_etf.values
    prev_v = etf_vals[np.clip(prev_idx, 0, n - 1)]
    next_v = etf_vals[np.clip(next_idx, 0, n - 1)]
    prev_t = cap_ints[np.clip(prev_idx, 0, n - 1)]
    next_t = cap_ints[np.clip(next_idx, 0, n - 1)]

    daily_etf = np.full(len(index), np.nan)
    both = prev_ok & next_ok
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (day_ints - prev_t) / np.where(next_t > prev_t, next_t - prev_t, 1)
    daily_etf[both] = (prev_v + (next_v - prev_v) * frac)[both]
    daily_etf[prev_ok & ~next_ok] = prev_v[prev_ok & ~next_ok]
    daily_etf[~prev_ok & next_ok] = next_v[~prev_ok & next_ok]

    daily_etf = pd.Series(daily_etf, index=index, name="benchmark_etf")
    daily_et = (daily_etf * eto.loc[index].astype("float64")).rename("benchmark_et")

    is_capture = index.isin(captures.index)
    finite = np.isfinite(daily_etf.values)
    interpolated = finite & ~is_capture & both
    flat_filled = finite & ~is_capture & (prev_ok ^ next_ok)
    support_class = pd.Series("unsupported", index=index, name="support_class")
    support_class[is_capture] = "capture"
    support_class[interpolated] = "interpolated"
    support_class[flat_filled] = "flat_fill"

    cap_in_index = captures.index.intersection(index)
    if capture_space == "et":
        identity = (daily_et.loc[cap_in_index] - capture_et.loc[cap_in_index]).abs()
    else:
        identity = (daily_etf.loc[cap_in_index] - capture_etf.loc[cap_in_index]).abs()
    identity_max = float(identity.max()) if len(identity) else 0.0
    if identity_max > identity_tol:
        worst = identity.idxmax()
        raise BenchmarkConstructionError(
            f"{label}: capture-date identity violated "
            f"(max |err| {identity_max:.3e} on {worst.date()}, tol {identity_tol:.1e})"
        )

    supported = index[finite]
    gaps = np.diff(cap_ints)
    return BenchmarkReconstruction(
        daily_et=daily_et,
        daily_etf=daily_etf,
        capture_dates=captures.index,
        capture_et=capture_et,
        capture_etf=capture_etf,
        support_class=support_class,
        support_start=supported[0] if len(supported) else pd.NaT,
        support_end=supported[-1] if len(supported) else pd.NaT,
        n_captures=len(captures),
        n_interpolated_days=int(interpolated.sum()),
        n_flat_filled_days=int(flat_filled.sum()),
        n_unsupported_days=int((~finite).sum()),
        max_capture_gap_days=int(gaps.max()) if len(gaps) else 0,
        identity_max_abs_err=identity_max,
        eto_name=eto_name,
        window_days=window_days,
        label=label,
    )


def classify_temporal_support(index, capture_dates, window_days=VOLK_WINDOW_DAYS):
    """Classify each day of ``index`` under the Volk temporal-support rule.

    Returns a Series of {'capture', 'interpolated', 'flat_fill',
    'unsupported'} derived from raw capture availability only — never from
    pairing or flux availability.
    """
    caps = pd.DatetimeIndex(capture_dates).sort_values()
    if caps.has_duplicates:
        caps = caps.unique()
    day_ints = _as_day_ints(index)
    cap_ints = _as_day_ints(caps)
    _, _, prev_ok, next_ok = _window_anchors(day_ints, cap_ints, window_days)
    is_capture = index.isin(caps)
    out = pd.Series("unsupported", index=index, name="support_class")
    out[(prev_ok ^ next_ok) & ~is_capture] = "flat_fill"
    out[prev_ok & next_ok & ~is_capture] = "interpolated"
    out[is_capture] = "capture"
    return out


def assert_inside_support(scored_index, recon, label=""):
    """Raise unless every scored date has a finite reconstructed value."""
    values = recon.daily_et.reindex(scored_index)
    bad = values.index[~np.isfinite(values.values)]
    if len(bad):
        raise BenchmarkConstructionError(
            f"{label or recon.label}: {len(bad)} scored dates outside benchmark "
            f"support (first: {bad[0].date()})"
        )


def pair_on_common_dates(**series):
    """Intersect named daily series on dates where every series is finite.

    Pairing happens only AFTER reconstruction: pass the reconstructed
    benchmark, the model series, and the (closure-corrected) flux series and
    score all of them on the identical returned date set.
    """
    if not series:
        raise BenchmarkConstructionError("pair_on_common_dates: no series given")
    df = pd.concat(series, axis=1, join="inner")
    return df.dropna()
