# Global estimation of effective plant rooting depth: Implications for hydrological modeling
# https://doi.org/10.1002/2016WR019392
# mapped to Land Cover Type 1: Annual International Geosphere-Biosphere Programme (IGBP) classification
# from https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1#bands

MAX_EFFECTIVE_ROOTING_DEPTH = {
    "1": {"rooting_depth": 0.43,
          "description": "Evergreen Needleleaf Forests: dominated by evergreen conifer trees (canopy >2m). Tree cover >60%."},
    "2": {"rooting_depth": 3.14,
          "description": "Evergreen Broadleaf Forests: dominated by evergreen broadleaf and palmate trees (canopy >2m). Tree cover >60%."},
    "3": {"rooting_depth": 0.38,
          "description": "Deciduous Needleleaf Forests: dominated by deciduous needleleaf (larch) trees (canopy >2m). Tree cover >60%."},
    "4": {"rooting_depth": 1.07,
          "description": "Deciduous Broadleaf Forests: dominated by deciduous broadleaf trees (canopy >2m). Tree cover >60%."},
    "5": {"rooting_depth": 0.54,
          "description": "Mixed Forests: dominated by neither deciduous nor evergreen (40-60% of each) tree type (canopy >2m). Tree cover >60%."},
    "6": {"rooting_depth": 0.37,
          "description": "Closed Shrublands: dominated by woody perennials (1-2m height) >60% cover."},
    "7": {"rooting_depth": 0.37,
          "description": "Open Shrublands: dominated by woody perennials (1-2m height) 10-60% cover."},
    "8": {"rooting_depth": 0.80, "description": "Woody Savannas: tree cover 30-60% (canopy >2m)."},
    "9": {"rooting_depth": 0.80, "description": "Savannas: tree cover 10-30% (canopy >2m)."},
    "10": {"rooting_depth": 0.51, "description": "Grasslands: dominated by herbaceous annuals (<2m)."},
    "11": {"rooting_depth": 0.55, "description": "Cropland, same depth as shrublands"},
    "12": {"rooting_depth": 0.41, "description": "Desert vegetation."}
}

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
