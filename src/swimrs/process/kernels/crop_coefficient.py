"""Crop coefficient calculations from vegetation indices.

Pure physics kernels for computing basal crop coefficient (Kcb) from
remotely-sensed vegetation indices like NDVI.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

__all__ = ["kcb_sigmoid", "kcb_linear", "kcb_affine"]


@njit(cache=True, fastmath=True, parallel=True)
def kcb_sigmoid(
    ndvi: NDArray[np.float64],
    kc_max: NDArray[np.float64],
    ndvi_k: NDArray[np.float64],
    ndvi_0: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Basal crop coefficient from NDVI using sigmoid function.

    Kcb = Kc_max / (1 + exp(-k * (NDVI - NDVI_0)))

    Physical constraints:
        - 0 <= Kcb <= Kc_max
        - Monotonically increasing with NDVI
        - At NDVI = NDVI_0: Kcb = Kc_max * 0.5

    Parameters
    ----------
    ndvi : (n_fields,)
        Normalized Difference Vegetation Index, typically [-0.2, 0.9]
    kc_max : (n_fields,)
        Maximum crop coefficient per field, typically [1.0, 1.3]
    ndvi_k : (n_fields,)
        Sigmoid steepness parameter, typically [4, 10]
        Higher values = sharper transition from bare to vegetated
    ndvi_0 : (n_fields,)
        Sigmoid inflection point, typically [0.1, 0.7]

    Returns
    -------
    kcb : (n_fields,)
        Basal crop coefficient, bounded [0, Kc_max]

    References
    ----------
    Adapted from FAO-56 dual crop coefficient approach with remote sensing.
    """
    n = ndvi.shape[0]
    kcb = np.empty(n, dtype=np.float64)

    for i in prange(n):
        exponent = -ndvi_k[i] * (ndvi[i] - ndvi_0[i])
        # Clip exponent to prevent overflow (exp(20) ~ 4.8e8, exp(-20) ~ 2e-9)
        if exponent > 20.0:
            exponent = 20.0
        elif exponent < -20.0:
            exponent = -20.0

        sigmoid = 1.0 / (1.0 + np.exp(exponent))
        kcb_raw = kc_max[i] * sigmoid

        # Ensure Kcb doesn't exceed Kc_max
        if kcb_raw > kc_max[i]:
            kcb[i] = kc_max[i]
        elif kcb_raw < 0.0:
            kcb[i] = 0.0
        else:
            kcb[i] = kcb_raw

    return kcb


@njit(cache=True, fastmath=True, parallel=True)
def kcb_linear(
    ndvi: NDArray[np.float64],
    kc_min: NDArray[np.float64],
    kc_max: NDArray[np.float64],
    ndvi_min: NDArray[np.float64],
    ndvi_max: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Basal crop coefficient from NDVI using linear relationship.

    Kcb = Kc_min + (Kc_max - Kc_min) * (NDVI - NDVI_min) / (NDVI_max - NDVI_min)

    Physical constraints:
        - Kc_min <= Kcb <= Kc_max
        - Linear interpolation between endpoints

    Parameters
    ----------
    ndvi : (n_fields,)
        Normalized Difference Vegetation Index
    kc_min : (n_fields,)
        Minimum crop coefficient (bare soil), typically ~0.15
    kc_max : (n_fields,)
        Maximum crop coefficient (full cover), typically [1.0, 1.3]
    ndvi_min : (n_fields,)
        NDVI corresponding to bare soil, typically ~0.1
    ndvi_max : (n_fields,)
        NDVI corresponding to full vegetation, typically ~0.8

    Returns
    -------
    kcb : (n_fields,)
        Basal crop coefficient, bounded [Kc_min, Kc_max]

    Notes
    -----
    Simpler than sigmoid but can extrapolate poorly at extreme NDVI values.
    Values are clipped to [Kc_min, Kc_max] range.
    """
    n = ndvi.shape[0]
    kcb = np.empty(n, dtype=np.float64)

    for i in prange(n):
        ndvi_range = ndvi_max[i] - ndvi_min[i]
        if ndvi_range < 1e-6:
            # Avoid division by zero
            kcb[i] = kc_min[i]
        else:
            frac = (ndvi[i] - ndvi_min[i]) / ndvi_range
            kcb_raw = kc_min[i] + (kc_max[i] - kc_min[i]) * frac

            # Clip to valid range
            if kcb_raw > kc_max[i]:
                kcb[i] = kc_max[i]
            elif kcb_raw < kc_min[i]:
                kcb[i] = kc_min[i]
            else:
                kcb[i] = kcb_raw

    return kcb


@njit(cache=True, fastmath=True, parallel=True)
def kcb_affine(
    ndvi: NDArray[np.float64],
    kc_max: NDArray[np.float64],
    ndvi_alpha: NDArray[np.float64],
    ndvi_beta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Basal crop coefficient from NDVI using the linear NDVI-Kcb relation.

    Kcb = ndvi_beta * NDVI + ndvi_alpha

    This is the standard remote-sensing crop-coefficient form and the relation
    SWIM's predecessor used before the sigmoid (``model/obs_kcb_daily.py``,
    removed in ``d77bc84``). Slope and intercept are both free, so the curve is
    two-parameter like the sigmoid.

    Parameters
    ----------
    ndvi : (n_fields,)
        Normalized Difference Vegetation Index
    kc_max : (n_fields,)
        Maximum crop coefficient per field, used only as the upper clip
    ndvi_alpha : (n_fields,)
        Intercept, historically ~0.2 with bounds [-0.7, 1.5]
    ndvi_beta : (n_fields,)
        Slope, historically ~1.25 with bounds [0.5, 1.7]

    Returns
    -------
    kcb : (n_fields,)
        Basal crop coefficient, bounded [0, Kc_max]

    Notes
    -----
    Unlike :func:`kcb_linear`, the endpoints are not NDVI values: the ramp is
    parameterized directly by slope and intercept, which is what makes it a
    drop-in two-parameter replacement for the logistic.

    The clip to ``[0, Kc_max]`` is a deliberate departure from the legacy
    implementation, which had none. Downstream ``fc`` and root-depth ratios
    assume ``Kcb <= Kc_max``; an unclipped ramp violates that at high NDVI.
    """
    n = ndvi.shape[0]
    kcb = np.empty(n, dtype=np.float64)

    for i in prange(n):
        kcb_raw = ndvi_beta[i] * ndvi[i] + ndvi_alpha[i]

        if kcb_raw > kc_max[i]:
            kcb[i] = kc_max[i]
        elif kcb_raw < 0.0:
            kcb[i] = 0.0
        else:
            kcb[i] = kcb_raw

    return kcb
