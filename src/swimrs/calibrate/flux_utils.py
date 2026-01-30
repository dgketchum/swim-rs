"""Utilities for flux tower site selection and ensemble parameter generation.

Relocated from swimrs.prep (deprecated) for use by examples 5/6 and viz modules.
"""

import geopandas as gpd
import pandas as pd


def get_flux_sites(
    sites, crop_only=False, return_df=False, western_only=False, index_col=None, header=None
):
    if sites.endswith(".shp"):
        sdf = gpd.read_file(sites)
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
