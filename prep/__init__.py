import pandas as pd
import geopandas as gpd

# Global estimation of effective plant rooting depth: Implications for hydrological modeling
# https://doi.org/10.1002/2016WR019392
# mapped to Land Cover Type 1: Annual International Geosphere-Biosphere Programme (IGBP) classification
# from https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1#bands

MEAN_EFFECTIVE_ROOTING_DEPTH = {
    "1": {"rooting_depth": 0.43,
          "zr_multiplier": 5,
          "description": "Evergreen Needleleaf Forests: dominated by evergreen conifer trees (canopy >2m). Tree cover >60%."},
    "2": {"rooting_depth": 3.14,
          "zr_multiplier": 2,
          "description": "Evergreen Broadleaf Forests: dominated by evergreen broadleaf and palmate trees (canopy >2m). Tree cover >60%."},
    "3": {"rooting_depth": 0.38,
          "zr_multiplier": 5,
          "description": "Deciduous Needleleaf Forests: dominated by deciduous needleleaf (larch) trees (canopy >2m). Tree cover >60%."},
    "4": {"rooting_depth": 1.07,
          "zr_multiplier": 5,
          "description": "Deciduous Broadleaf Forests: dominated by deciduous broadleaf trees (canopy >2m). Tree cover >60%."},
    "5": {"rooting_depth": 0.54,
          "zr_multiplier": 5,
          "description": "Mixed Forests: dominated by neither deciduous nor evergreen (40-60% of each) tree type (canopy >2m). Tree cover >60%."},
    "6": {"rooting_depth": 0.37,
          "zr_multiplier": 3,
          "description": "Closed Shrublands: dominated by woody perennials (1-2m height) >60% cover."},
    "7": {"rooting_depth": 0.37,
          "zr_multiplier": 3,
          "description": "Open Shrublands: dominated by woody perennials (1-2m height) 10-60% cover."},
    "8": {"rooting_depth": 0.80, "zr_multiplier": 3, "description": "Woody Savannas: tree cover 30-60% (canopy >2m)."},

    "9": {"rooting_depth": 0.80, "zr_multiplier": 3, "description": "Savannas: tree cover 10-30% (canopy >2m)."},

    "10": {"rooting_depth": 0.51, "zr_multiplier": 3,
           "description": "Grasslands: dominated by herbaceous annuals (<2m)."},

    "11": {"rooting_depth": 0.37, "zr_multiplier": 3, "description": "Wetlands"},

    "12": {"rooting_depth": 0.55, "zr_multiplier": 3, "description": "Cropland, same depth as shrublands"},
    "13": {"rooting_depth": 0.55, "zr_multiplier": 3, "description": "Developed"},
    "14": {"rooting_depth": 0.55, "zr_multiplier": 1,
           "description": "Cropland/Natural Mosiac, same depth as shrublands"},
    "16": {"rooting_depth": 0.41, "zr_multiplier": 5, "description": "Desert vegetation."}
}

MAX_EFFECTIVE_ROOTING_DEPTH = {
    "1": {"rooting_depth": 1.34, "zr_multiplier": 5,
          "description": "Evergreen Needleleaf Forests: dominated by evergreen conifer trees (canopy >2m). Tree cover >60%."},
    "2": {"rooting_depth": 7.99, "zr_multiplier": 5,
          "description": "Evergreen Broadleaf Forests: dominated by evergreen broadleaf and palmate trees (canopy >2m). Tree cover >60%."},
    "3": {"rooting_depth": 0.84, "zr_multiplier": 5,
          "description": "Deciduous Needleleaf Forests: dominated by deciduous needleleaf (larch) trees (canopy >2m). Tree cover >60%."},
    "4": {"rooting_depth": 2.09, "zr_multiplier": 5,
          "description": "Deciduous Broadleaf Forests: dominated by deciduous broadleaf trees (canopy >2m). Tree cover >60%."},
    "5": {"rooting_depth": 1.94, "zr_multiplier": 5,
          "description": "Mixed Forests: dominated by neither deciduous nor evergreen (40-60% of each) tree type (canopy >2m). Tree cover >60%."},
    "6": {"rooting_depth": 1.12, "zr_multiplier": 3,
          "description": "Closed Shrublands: dominated by woody perennials (1-2m height) >60% cover."},
    "7": {"rooting_depth": 1.12, "zr_multiplier": 3,
          "description": "Open Shrublands: dominated by woody perennials (1-2m height) 10-60% cover."},
    "8": {"rooting_depth": 2.28, "zr_multiplier": 3, "description": "Woody Savannas: tree cover 30-60% (canopy >2m)."},
    "9": {"rooting_depth": 2.28, "zr_multiplier": 3, "description": "Savannas: tree cover 10-30% (canopy >2m)."},
    "10": {"rooting_depth": 1.18, "zr_multiplier": 3,
           "description": "Grasslands: dominated by herbaceous annuals (<2m)."},
    "11": {"rooting_depth": 1.12, "zr_multiplier": 3, "description": "Wetlands"},
    "12": {"rooting_depth": 1.12, "zr_multiplier": 2, "description": "Cropland, same depth as shrublands"},
    "13": {"rooting_depth": 1.12, "zr_multiplier": 1, "description": "Developed"},
    "14": {"rooting_depth": 1.12, "zr_multiplier": 1,
           "description": "Cropland/Natural Mosiac, same depth as shrublands"},
    "16": {"rooting_depth": 1.43, "zr_multiplier": 5, "description": "Desert vegetation."}
}


def get_flux_sites(sites, crop_only=False, return_df=False, western_only=False,
                   index_col=None, header=None):
    """"""

    if sites.endswith('.shp'):
        sdf = gpd.read_file(sites)
        sdf.index = sdf[index_col]

    else:
        sdf = pd.read_csv(sites, index_col=0, header=header)

    if crop_only:
        sdf = sdf[sdf['General classification'] == 'Croplands']

    if western_only:
        target_states = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
        state_idx = [i for i, r in sdf.iterrows() if r['State'] in target_states]
        sdf = sdf.loc[state_idx]

    sites_ = list(set(sdf.index.unique().to_list()))

    sites_.sort()
    if return_df:
        return sites_, sdf
    else:
        return sites_


def get_ensemble_parameters(skip=None, include=None):
    """"""
    ensemble_params = []

    for mask in ['irr', 'inv_irr']:

        for model in ['openet', 'eemetric', 'geesebal', 'ptjpl', 'sims', 'ssebop', 'disalexi']:

            if skip and model in skip:
                continue
            if include and model not in include:
                continue

            ensemble_params.append(f'{model}_etf_{mask}')
            ensemble_params.append(f'{model}_etf_{mask}_ct')

        ensemble_params.append(f'ndvi_{mask}')
        ensemble_params.append(f'ndvi_{mask}_ct')

    return ensemble_params


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
