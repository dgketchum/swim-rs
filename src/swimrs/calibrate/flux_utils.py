"""Utilities for flux tower site selection and ensemble parameter generation.

Relocated from swimrs.prep (deprecated) for use by examples 5/6 and viz modules.
Also hosts the shared flux-evaluation gates and monthly aggregation helpers
required by examples/VALIDATION_POLICY.md.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd


def passes_site_minimum(flux_daily, min_days=90, min_months=3, month_min_days=20):
    """VALIDATION_POLICY site-minimum gate for headline-table inclusion.

    A site qualifies when its daily flux record has at least ``min_days``
    valid (finite) observations and at least ``min_months`` months with
    ``month_min_days`` or more valid days.
    """
    valid = flux_daily.dropna()
    if len(valid) < min_days:
        return False
    qualifying = int((valid.resample("MS").count() >= month_min_days).sum())
    return qualifying >= min_months


def paired_monthly_sums(swim_daily, flux_daily, ref_daily=None, month_min_days=20):
    """Monthly ET totals summed over flux-valid days only.

    Restricts every series to days with a finite flux observation before
    resampling, so monthly sums integrate the identical day set on all sides,
    then keeps months with at least ``month_min_days`` valid flux days. A
    reference month is NaN unless the reference is finite on every valid day
    of that month — partial or empty reference months are not fabricated from
    whichever days happen to be present.

    Returns ``(swim_monthly, flux_monthly, ref_monthly)``; ``ref_monthly`` is
    None when ``ref_daily`` is None.
    """
    valid_days = flux_daily.dropna().index.intersection(swim_daily.index)
    swim_valid = swim_daily.loc[valid_days]
    flux_valid = flux_daily.loc[valid_days]

    flux_count = flux_valid.resample("MS").count()
    months = flux_count[flux_count >= month_min_days].index

    swim_monthly = swim_valid.resample("MS").sum().reindex(months)
    flux_monthly = flux_valid.resample("MS").sum().reindex(months)

    ref_monthly = None
    if ref_daily is not None:
        ref_valid = ref_daily.reindex(valid_days)
        ref_monthly = ref_valid.resample("MS").sum(min_count=1).reindex(months)
        ref_count = ref_valid.notna().resample("MS").sum().reindex(months).fillna(0)
        ref_monthly[ref_count < flux_count.reindex(months)] = np.nan

    return swim_monthly, flux_monthly, ref_monthly


def full_month_paired_sums(swim_daily, flux_daily, month_min_days=28):
    """Full-calendar-month totals gated on nearly-complete flux months.

    For comparison against references reported only as full-month totals
    (e.g. Volk OpenET monthly ET): SWIM is summed over the full calendar
    month, and months with fewer than ``month_min_days`` valid flux days are
    dropped so the flux total misses at most a few days.

    Returns ``(swim_monthly, flux_monthly)``.
    """
    flux_count = flux_daily.resample("MS").count()
    months = flux_count[flux_count >= month_min_days].index
    swim_monthly = swim_daily.resample("MS").sum().reindex(months)
    flux_monthly = flux_daily.resample("MS").sum().reindex(months)
    return swim_monthly, flux_monthly


# Volk et al. (2024) monthly protocol (flux-data-qaqc defaults): a month is a
# valid total when more than 80% of its days carry ET, and it enters the
# monthly comparison when at most five of those days were gap-filled.
VOLK_MONTH_COMPLETENESS = 0.8
VOLK_MAX_FILLED_DAYS = 5


def volk_gap_fill_et(flux_daily, eto_daily):
    """Gap-fill daily flux ET as flux-data-qaqc ``QaQc._ET_gap_fill`` (refET ETo).

    Reproduces the chain applied to the Volk et al. (2024) station records,
    over the full index of ``flux_daily`` (the station file's date span):
    EToF = ET / gridMET ETo on every day; EToF outliers beyond
    Q1 - 1.5 IQR or Q3 + 1.5 IQR (whole-record quartiles) set to NaN; 7-day
    centered rolling mean with at least two values; pandas linear
    interpolation (interior gaps and trailing days, never leading days);
    fill ET = ETo x smoothed EToF on days where ET is missing.

    ``eto_daily`` is the raw gridMET grass reference ET (mm d-1), reindexed
    to the flux index; a day without ETo cannot be filled.

    Returns ``(filled, gap)``: the filled daily ET and a boolean Series
    marking the filled days.
    """
    flux = flux_daily.astype(float)
    eto = eto_daily.reindex(flux.index).astype(float)
    etof = flux / eto
    q1, q3 = etof.quantile(0.25), etof.quantile(0.75)
    iqr = q3 - q1
    filtered = etof.mask((etof < q1 - 1.5 * iqr) | (etof > q3 + 1.5 * iqr))
    filtered = filtered.rolling(7, min_periods=2, center=True).mean()
    filtered = filtered.interpolate(method="linear")
    et_fill = eto * filtered
    gap = flux.isna() & et_fill.notna()
    filled = flux.copy()
    filled[gap] = et_fill[gap]
    return filled, gap


def volk_monthly_total(daily, thresh=VOLK_MONTH_COMPLETENESS):
    """Monthly total as flux-data-qaqc ``util.monthly_resample(agg='sum')``.

    The month's sum plus its remaining missing days at the month's daily
    mean; NaN when the count of valid days is at most ``thresh`` times the
    days in the month. Indexed by month start.
    """
    grouped = daily.astype(float).resample("MS")
    count, total, mean = grouped.count(), grouped.sum(), grouped.mean()
    days_in_month = pd.Series(count.index.days_in_month, index=count.index, dtype=float)
    out = total + (days_in_month - count) * mean
    out[count <= thresh * days_in_month] = np.nan
    return out


def volk_full_month_paired_sums(
    swim_daily,
    flux_daily,
    eto_daily,
    thresh=VOLK_MONTH_COMPLETENESS,
    max_filled_days=VOLK_MAX_FILLED_DAYS,
):
    """Full-calendar-month SWIM and flux totals under the Volk et al. (2024) rules.

    Flux ET is gap-filled with :func:`volk_gap_fill_et`, totaled with
    :func:`volk_monthly_total`, and a month is admitted when its total is
    finite and at most ``max_filled_days`` of its days were filled. SWIM is
    summed over the full calendar month and is NaN for any month the SWIM
    series does not cover completely.

    Returns ``(swim_monthly, flux_monthly, filled_days)`` on the admitted
    months; ``filled_days`` is the per-month count of gap-filled flux days.
    """
    filled, gap = volk_gap_fill_et(flux_daily, eto_daily)
    flux_monthly = volk_monthly_total(filled, thresh=thresh)
    filled_days = gap.resample("MS").sum().reindex(flux_monthly.index).fillna(0).astype(int)
    months = flux_monthly.index[flux_monthly.notna() & (filled_days <= max_filled_days)]

    swim_grouped = swim_daily.astype(float).resample("MS")
    swim_monthly = swim_grouped.sum()
    swim_count = swim_grouped.count()
    swim_monthly[swim_count < swim_count.index.days_in_month] = np.nan

    return (
        swim_monthly.reindex(months),
        flux_monthly.reindex(months),
        filled_days.reindex(months),
    )


def write_excluded_sites(excluded, results_dir, filename="evaluation_sites_excluded.csv"):
    """Write the per-site exclusion record required by RUN_POLICY Category 2.

    ``excluded`` is a list of ``{"site": ..., "reason": ...}`` dicts; an empty
    list still writes the (header-only) file so the artifact reflects the
    current run.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    pd.DataFrame(excluded, columns=["site", "reason"]).to_csv(path, index=False)
    return path


def get_flux_sites(
    sites, crop_only=False, return_df=False, western_only=False, index_col=None, header=None
):
    if sites.endswith(".shp"):
        sdf = gpd.read_file(sites, engine="fiona")
        sdf.index = sdf[index_col]

    else:
        sdf = pd.read_csv(sites, index_col=0, header=header)

    if crop_only:
        sdf = sdf[sdf["General classification"] == "Croplands"]

    if western_only:
        target_states = ["AZ", "CA", "CO", "ID", "MT", "NM", "NV", "OR", "UT", "WA", "WY"]
        state_idx = [i for i, r in sdf.iterrows() if r["State"] in target_states]
        sdf = sdf.loc[state_idx]

    sites_ = list(set(sdf.index.unique().to_list()))

    sites_.sort()
    if return_df:
        return sites_, sdf
    else:
        return sites_


def get_ensemble_parameters(skip=None, include=None, masks=("irr", "inv_irr")):
    ensemble_params = []

    for mask in masks:
        for model in ["openet", "eemetric", "geesebal", "ptjpl", "sims", "ssebop", "disalexi"]:
            if skip and model in skip:
                continue
            if include and model not in include:
                continue

            ensemble_params.append((f"{model}", "etf", f"{mask}"))

        ensemble_params.append(("none", "ndvi", f"{mask}"))

    return ensemble_params
