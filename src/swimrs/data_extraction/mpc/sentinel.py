"""Sentinel-2 L2A NDVI matching the EE implementation in ee_utils.py.

Parity notes vs ee_utils.sentinel2_sr / harmonize_sentinel_to_oli:
- SCL classes 3 (cloud shadow), 8 (cloud medium), 9 (cloud high),
  10 (cirrus) are masked; everything else is kept.
- MSI -> OLI SBAF (HLS bandpass) on B04 (red) and B8A (narrow NIR).
- EE's S2_SR_HARMONIZED removes the +1000 DN offset that ESA added with
  processing baseline 04.00 (Jan 2022). MPC serves L2A as-distributed, so
  the offset must be subtracted here for baseline >= 04.00 items.
- Work happens on the 20 m B8A/SCL grid; 10 m B04 is block-averaged 2x2.
"""

import numpy as np

from swimrs.data_extraction.mpc import stac
from swimrs.data_extraction.mpc.grid import NODATA, block_reduce_mean, read_window

S2_SCALE = 0.0001
S2_DN_NODATA = 0
BASELINE_OFFSET_DN = 1000  # subtract for processing baseline >= 04.00
SCL_MASKED_CLASSES = (3, 8, 9, 10)

MSI_SBAF = {
    "red_slope": 0.9763,
    "red_intercept": 0.00095,
    "nir_slope": 0.99745,
    "nir_intercept": -0.00005,
}


def dn_offset(item_dict):
    """DN offset to subtract before scaling, from the processing baseline."""
    baseline = item_dict["properties"].get("s2:processing_baseline", "0.0")
    try:
        needs_offset = float(baseline) >= 4.0
    except ValueError:
        needs_offset = False
    return BASELINE_OFFSET_DN if needs_offset else 0


def scl_clear_mask(scl):
    """True where the SCL class is not cloud/shadow/cirrus."""
    return ~np.isin(scl, SCL_MASKED_CLASSES)


def scene_ndvi(item_dict, bounds):
    """Cloud-masked, harmonized NDVI for one S2 granule at 20 m.

    `bounds` is (xmin, ymin, xmax, ymax) in the granule's native UTM CRS;
    it is snapped outward to the 20 m grid so the 10 m red band block-
    averages cleanly. Returns (ndvi float32 with NODATA, GridSpec).
    """
    bounds = _snap_to_grid(bounds, 20.0)
    nir_dn, grid = read_window(stac.asset_href(item_dict, "B8A"), bounds, fill_value=0)
    scl, _ = read_window(stac.asset_href(item_dict, "SCL"), bounds, fill_value=0)
    red_dn10, _ = read_window(stac.asset_href(item_dict, "B04"), bounds, fill_value=0)
    red_dn = block_reduce_mean(red_dn10, 2, nodata=S2_DN_NODATA)

    offset = dn_offset(item_dict)
    valid = scl_clear_mask(scl)
    valid &= (nir_dn != S2_DN_NODATA) & np.isfinite(red_dn)

    red = (red_dn - offset) * S2_SCALE
    nir = (nir_dn.astype(np.float64) - offset) * S2_SCALE
    red = red * MSI_SBAF["red_slope"] + MSI_SBAF["red_intercept"]
    nir = nir * MSI_SBAF["nir_slope"] + MSI_SBAF["nir_intercept"]

    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / denom
    valid &= denom != 0

    out = np.full(ndvi.shape, NODATA, dtype=np.float32)
    out[valid] = ndvi[valid]
    return out, grid


def _snap_to_grid(bounds, res):
    xmin, ymin, xmax, ymax = bounds
    return (
        np.floor(xmin / res) * res,
        np.floor(ymin / res) * res,
        np.ceil(xmax / res) * res,
        np.ceil(ymax / res) * res,
    )
