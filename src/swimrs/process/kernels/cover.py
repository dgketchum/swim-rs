"""Vegetation cover fraction calculations.

Pure physics kernels for computing fractional vegetation cover and
exposed soil fraction from crop coefficients.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

__all__ = ["fractional_cover", "fractional_cover_from_ndvi", "exposed_soil_fraction"]


@njit(cache=True, fastmath=True, parallel=True)
def fractional_cover(
    kcb: NDArray[np.float64],
    kc_min: NDArray[np.float64],
    kc_max: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate fractional vegetation cover from basal crop coefficient.

    fc = (Kcb - Kc_min) / (Kc_max - Kc_min)

    Physical constraints:
        - 0 <= fc < 1 (capped at 0.99 for numerical stability in few calculation)
        - fc = 0 when Kcb = Kc_min (bare soil)
        - fc approaches 1 when Kcb approaches Kc_max (full cover)

    Parameters
    ----------
    kcb : (n_fields,)
        Basal crop coefficient, typically [0, Kc_max]
    kc_min : (n_fields,)
        Minimum crop coefficient (bare soil), typically ~0.15
    kc_max : (n_fields,)
        Maximum crop coefficient (full cover), typically [1.0, 1.3]

    Returns
    -------
    fc : (n_fields,)
        Fractional vegetation cover, bounded [0, 0.99]

    Notes
    -----
    The upper bound of 0.99 ensures that few (fraction of soil exposed to
    evaporation) remains positive, avoiding division by zero in subsequent
    evaporation calculations.

    References
    ----------
    FAO-56 Eq. 76: fc = ((Kcb - Kc_min) / (Kc_max - Kc_min))^(1+0.5h)
    Simplified here without the height correction term.
    """
    n = kcb.shape[0]
    fc = np.empty(n, dtype=np.float64)

    for i in prange(n):
        kc_range = kc_max[i] - kc_min[i]
        if kc_range < 1e-6:
            # Avoid division by zero
            fc[i] = 0.0
        else:
            # Ensure kcb is at least kc_min
            kcb_clipped = kcb[i]
            if kcb_clipped < kc_min[i]:
                kcb_clipped = kc_min[i]

            fc_raw = (kcb_clipped - kc_min[i]) / kc_range

            # Clip to [0, 0.99] range
            if fc_raw > 0.99:
                fc[i] = 0.99
            elif fc_raw < 0.0:
                fc[i] = 0.0
            else:
                fc[i] = fc_raw

    return fc


@njit(cache=True, fastmath=True, parallel=True)
def fractional_cover_from_ndvi(
    ndvi: NDArray[np.float64],
    ndvi_bare: NDArray[np.float64],
    ndvi_full: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate fractional vegetation cover directly from NDVI.

    fc = (NDVI - NDVIs) / (NDVIv - NDVIs)  # Carlson & Ripley (1997)

    This approach decouples canopy cover from transpiration demand (Kcb),
    allowing the evaporation shading effect to reflect actual vegetation
    density rather than the calibrated NDVI-Kcb relationship.

    Physical constraints:
        - 0 <= fc < 1 (capped at 0.99 for numerical stability)
        - fc = 0 when NDVI <= NDVIs (bare soil)
        - fc approaches 1 when NDVI >= NDVIv (full cover)

    Parameters
    ----------
    ndvi : (n_fields,)
        Normalized Difference Vegetation Index, typically [0, 1]
    ndvi_bare : (n_fields,)
        NDVI of bare/dormant conditions, typically 5th percentile of site record
    ndvi_full : (n_fields,)
        NDVI of full vegetation cover, typically 95th percentile of site record

    Returns
    -------
    fc : (n_fields,)
        Fractional vegetation cover, bounded [0, 0.99]

    References
    ----------
    Carlson, T.N. and Ripley, D.A. (1997). On the relation between NDVI,
    fractional vegetation cover, and leaf area index. Remote Sensing of
    Environment, 62(3), 241-252.
    """
    n = ndvi.shape[0]
    fc = np.empty(n, dtype=np.float64)

    for i in prange(n):
        ndvi_range = ndvi_full[i] - ndvi_bare[i]
        if ndvi_range < 1e-6:
            # Avoid division by zero
            fc[i] = 0.0
        else:
            fc_raw = (ndvi[i] - ndvi_bare[i]) / ndvi_range

            # Clip to [0, 0.99] range
            if fc_raw > 0.99:
                fc[i] = 0.99
            elif fc_raw < 0.0:
                fc[i] = 0.0
            else:
                fc[i] = fc_raw

    return fc


@njit(cache=True, fastmath=True, parallel=True)
def exposed_soil_fraction(
    fc: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate fraction of soil exposed to evaporation.

    few = 1 - fc

    Physical constraints:
        - 0 < few <= 1
        - few represents the fraction of ground not covered by vegetation
          that is available for soil evaporation

    Parameters
    ----------
    fc : (n_fields,)
        Fractional vegetation cover, bounded [0, 0.99]

    Returns
    -------
    few : (n_fields,)
        Fraction of soil exposed and wetted, bounded [0.01, 1.0]

    Notes
    -----
    The minimum value of 0.01 ensures numerical stability in evaporation
    calculations. In reality, even dense canopies have some exposed soil
    or mulch surface.

    References
    ----------
    FAO-56 Eq. 75: few = min(1-fc, fw) where fw is wetted fraction.
    Simplified here assuming fw = 1 (full wetting from precipitation/irrigation).
    """
    n = fc.shape[0]
    few = np.empty(n, dtype=np.float64)

    for i in prange(n):
        few_raw = 1.0 - fc[i]

        # Ensure minimum exposed fraction for numerical stability
        if few_raw < 0.01:
            few[i] = 0.01
        elif few_raw > 1.0:
            few[i] = 1.0
        else:
            few[i] = few_raw

    return few
