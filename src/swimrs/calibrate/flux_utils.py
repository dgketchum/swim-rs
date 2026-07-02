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
