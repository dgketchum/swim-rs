"""Landsat C2 L2 NDVI matching the EE implementation in ee_utils.py.

Chain (order matters, parity with ee_utils.landsat_c2_sr + harmonize):
scale SR -> mask QA_PIXEL bits 1-5 + QA_RADSAT>0 + SR fill (DN 0) ->
SBAF harmonize TM/ETM+ to OLI (Roy et al. 2016) -> NDVI.
"""

import numpy as np

from swimrs.data_extraction.mpc import stac
from swimrs.data_extraction.mpc.grid import NODATA, read_window

SR_SCALE = 0.0000275
SR_OFFSET = -0.2

# Roy et al. (2016) Table 2, OLS surface reflectance — same numbers as
# ee_utils.SBAF_COEFFICIENTS; L4/5 (TM) and L7 (ETM+) share coefficients,
# OLI (L8/9) is the identity reference.
SBAF_COEFFICIENTS = {
    "TM": {
        "red_slope": 0.9047,
        "red_intercept": 0.0061,
        "nir_slope": 0.8462,
        "nir_intercept": 0.0412,
    },
    "ETM": {
        "red_slope": 0.9047,
        "red_intercept": 0.0061,
        "nir_slope": 0.8462,
        "nir_intercept": 0.0412,
    },
    "OLI": {"red_slope": 1.0, "red_intercept": 0.0, "nir_slope": 1.0, "nir_intercept": 0.0},
}

PLATFORM_TO_SENSOR = {
    "landsat-4": "TM",
    "landsat-5": "TM",
    "landsat-7": "ETM",
    "landsat-8": "OLI",
    "landsat-9": "OLI",
}


def qa_clear_mask(qa_pixel, qa_radsat):
    """True where the pixel is usable, replicating ee_utils.landsat_c2_sr.

    QA_PIXEL bits 1 (dilated cloud), 2 (cirrus), 3 (cloud), 4 (cloud
    shadow), 5 (snow) must all be unset, and QA_RADSAT must be 0 (no
    saturated bands). Bit 0 (fill) is handled by the SR DN==0 check in
    scene_ndvi, mirroring EE's native fill masking.
    """
    qa = qa_pixel.astype(np.uint16)
    flagged = np.zeros(qa.shape, dtype=bool)
    for bit in (1, 2, 3, 4, 5):
        flagged |= ((qa >> bit) & 1).astype(bool)
    flagged |= qa_radsat.astype(np.uint16) > 0
    return ~flagged


def harmonize(red, nir, platform):
    """SBAF-adjust scaled reflectance to the OLI reference."""
    coef = SBAF_COEFFICIENTS[PLATFORM_TO_SENSOR[platform.lower()]]
    red_h = red * coef["red_slope"] + coef["red_intercept"]
    nir_h = nir * coef["nir_slope"] + coef["nir_intercept"]
    return red_h, nir_h


def scene_ndvi(item_dict, bounds):
    """Cloud-masked, harmonized NDVI for one scene, windowed to `bounds`.

    `bounds` is (xmin, ymin, xmax, ymax) in the scene's native UTM CRS.
    Returns (ndvi float32 with NODATA where masked, GridSpec).
    """
    red_dn, grid = read_window(stac.asset_href(item_dict, "red"), bounds, fill_value=0)
    nir_dn, _ = read_window(stac.asset_href(item_dict, "nir08"), bounds, fill_value=0)
    qa_pixel, _ = read_window(stac.asset_href(item_dict, "qa_pixel"), bounds, fill_value=1)
    qa_radsat, _ = read_window(stac.asset_href(item_dict, "qa_radsat"), bounds, fill_value=0)

    valid = qa_clear_mask(qa_pixel, qa_radsat)
    valid &= (red_dn != 0) & (nir_dn != 0)  # SR DN 0 = fill

    red = red_dn.astype(np.float64) * SR_SCALE + SR_OFFSET
    nir = nir_dn.astype(np.float64) * SR_SCALE + SR_OFFSET
    red, nir = harmonize(red, nir, item_dict["properties"]["platform"])

    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / denom
    valid &= denom != 0

    out = np.full(ndvi.shape, NODATA, dtype=np.float32)
    out[valid] = ndvi[valid]
    return out, grid
