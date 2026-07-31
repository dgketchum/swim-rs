"""Transpiration cover-scaling formulations.

SWIM computes actual ET with a cover-scaled transpiration term,

    Kc_act = fc_t * Ks * Kcb + Ke

where ``fc_t`` is a canopy-cover weight on transpiration. This module defines
the alternative forms of ``fc_t`` used by the E1 cover-form experiment and
resolves the configured name to the integer code consumed inside the JIT loop.

Forms
-----
``none`` (0)
    ``fc_t = 1`` -- the FAO-56 canonical dual crop coefficient, ``Ks*Kcb + Ke``.
``kcb`` (1)
    ``fc_t = fc = (Kcb - Kc_min)/(Kc_max - Kc_min)``, clipped ``[0, 0.99]``:
    FAO-56 Eq. 76 without the height exponent, i.e. the SWIM default.
``sigmoid`` (2)
    ``fc_t = 1/(1 + exp(-k(NDVI - NDVI_0))) = Kcb/Kc_max`` -- the bare logistic
    in NDVI, untied from the Kc_min offset and rescaling of Eq. 76.
``linear`` (3)
    ``fc_t = (NDVI - NDVI_lo)/(NDVI_hi - NDVI_lo)`` -- the classic linear
    NDVI-to-cover ramp. With no explicit endpoints the ramp is placed by the
    same calibrated ``(ndvi_k, ndvi_0)`` as the other forms, spanning the
    logistic's 10%-90% range, so every arm carries identical free parameters
    and only the *shape* of the cover function differs.

In every case ``few = 1 - fc`` (the evaporation-exposed soil fraction) and the
root-depth ratio keep using the Kcb-derived ``fc``: the mode changes the
transpiration weight only.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = [
    "COVER_MODE_CODES",
    "COVER_MODE_NAMES",
    "COVER_MODE_KCB",
    "COVER_MODE_LINEAR",
    "COVER_MODE_NONE",
    "COVER_MODE_SIGMOID",
    "resolve_cover_mode",
    "transpiration_cover_factor",
]

COVER_MODE_NONE = 0
COVER_MODE_KCB = 1
COVER_MODE_SIGMOID = 2
COVER_MODE_LINEAR = 3

COVER_MODE_CODES = {
    "none": COVER_MODE_NONE,
    "kcb": COVER_MODE_KCB,
    "sigmoid": COVER_MODE_SIGMOID,
    "linear": COVER_MODE_LINEAR,
}
COVER_MODE_NAMES = {code: name for name, code in COVER_MODE_CODES.items()}

# ln(9): half-width of the logistic between its 10% and 90% points, used to
# place the linear ramp on the same (ndvi_k, ndvi_0) footing as the sigmoid.
_LOGIT_90 = 2.1972245773362196

# Ceiling shared by every cover-weighted form, inherited from the historical
# fc clip so the default mode reproduces prior runs bit-for-bit.
_FC_MAX = 0.99


def resolve_cover_mode(mode=None, cover_scaling=None) -> int:
    """Resolve a cover-mode name (and/or the legacy boolean) to an integer code.

    Parameters
    ----------
    mode : str | int | None
        Cover-mode name (``none``/``kcb``/``sigmoid``/``linear``) or its code.
        ``None`` falls back to *cover_scaling*.
    cover_scaling : bool | None
        Legacy ``transpiration_cover_scaling`` toggle. ``True`` (or ``None``,
        the historical default) maps to ``kcb``; ``False`` maps to ``none``.

    Raises
    ------
    ValueError
        If *mode* is unknown, or if an explicit *mode* contradicts an explicit
        *cover_scaling* (silently honoring one of two conflicting settings is
        how a whole ablation arm ends up running the wrong physics).
    """
    if mode is None:
        if cover_scaling is None:
            return COVER_MODE_KCB
        return COVER_MODE_KCB if bool(cover_scaling) else COVER_MODE_NONE

    if isinstance(mode, str):
        key = mode.strip().lower()
        if key not in COVER_MODE_CODES:
            raise ValueError(
                f"Unknown transpiration cover mode {mode!r}. "
                f"Expected one of {sorted(COVER_MODE_CODES)}."
            )
        code = COVER_MODE_CODES[key]
    elif isinstance(mode, int | np.integer) and not isinstance(mode, bool):
        code = int(mode)
        if code not in COVER_MODE_NAMES:
            raise ValueError(
                f"Unknown transpiration cover mode code {code!r}. "
                f"Expected one of {sorted(COVER_MODE_NAMES)}."
            )
    else:
        raise TypeError(f"transpiration cover mode must be a str or int, got {type(mode)!r}")

    if cover_scaling is not None:
        implied = COVER_MODE_KCB if bool(cover_scaling) else COVER_MODE_NONE
        conflict = (code == COVER_MODE_NONE) != (implied == COVER_MODE_NONE)
        if conflict:
            raise ValueError(
                f"transpiration_cover_mode={COVER_MODE_NAMES[code]!r} contradicts "
                f"transpiration_cover_scaling={bool(cover_scaling)!r}. Set one or the other."
            )
    return code


@njit(cache=True)
def transpiration_cover_factor(
    mode: int,
    ndvi: NDArray[np.float64],
    fc: NDArray[np.float64],
    sigmoid: NDArray[np.float64],
    ndvi_k: NDArray[np.float64],
    ndvi_0: NDArray[np.float64],
    lin_lo: float,
    lin_hi: float,
) -> NDArray[np.float64]:
    """Per-field transpiration cover weight ``fc_t`` for one day.

    Parameters
    ----------
    mode : int
        Cover-mode code (see module docstring).
    ndvi, fc, sigmoid : (n_fields,)
        Day's NDVI, the Kcb-derived fractional cover, and the raw NDVI logistic
        (``Kcb/Kc_max``). ``fc`` and ``sigmoid`` are passed in rather than
        recomputed so the caller's clipping is the single source of truth.
    ndvi_k, ndvi_0 : (n_fields,)
        Calibrated logistic steepness and midpoint, which also place the linear
        ramp when no explicit endpoints are given.
    lin_lo, lin_hi : float
        Explicit linear-ramp endpoints. Negative ``lin_lo`` (the default
        sentinel) selects the sigmoid-matched 10%-90% ramp instead.
    """
    n = ndvi.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        if mode == COVER_MODE_NONE:
            out[i] = 1.0
            continue

        if mode == COVER_MODE_KCB:
            v = fc[i]
        elif mode == COVER_MODE_SIGMOID:
            v = sigmoid[i]
        else:
            if lin_lo >= 0.0 and lin_hi > lin_lo:
                lo = lin_lo
                hi = lin_hi
            else:
                k = ndvi_k[i]
                if k < 1e-6:
                    k = 1e-6
                half_width = _LOGIT_90 / k
                lo = ndvi_0[i] - half_width
                hi = ndvi_0[i] + half_width
            v = (ndvi[i] - lo) / (hi - lo)

        if v > _FC_MAX:
            v = _FC_MAX
        elif v < 0.0:
            v = 0.0
        out[i] = v

    return out
