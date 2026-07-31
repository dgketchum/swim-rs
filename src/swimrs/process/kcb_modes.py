"""NDVI-to-Kcb formulations.

SWIM derives the basal crop coefficient from NDVI. Two forms are supported:

``sigmoid`` (0)
    ``Kcb = Kc_max / (1 + exp(-ndvi_k*(NDVI - ndvi_0)))`` -- the SWIM default,
    a two-parameter logistic with a calibrated steepness and midpoint.
``linear`` (1)
    ``Kcb = ndvi_beta*NDVI + ndvi_alpha`` -- the linear NDVI-Kcb relation used
    throughout the remote-sensing crop-coefficient literature and by SWIM's own
    predecessor (``model/obs_kcb_daily.py``, removed in ``d77bc84``), where it
    was the commented-out alternative sitting directly above the sigmoid.

The two forms carry *different* free parameters: ``(ndvi_k, ndvi_0)`` for the
logistic, ``(ndvi_alpha, ndvi_beta)`` for the linear relation. Both are
two-parameter, so neither arm of a comparison gets extra freedom, but the swap
has to be made in the PEST parameter set as well -- see
``PestBuilder.initial_parameter_dict``.

Why this exists
---------------
The cover-form experiment (``examples/4_Flux_Network/cover_form_experiment.py``)
varies only the transpiration cover weight ``fc_t`` in
``Kc_act = fc_t*Ks*Kcb + Ke``. Because ``fc`` is itself built from ``Kcb``,
every cover-weighted arm is quadratic in the logistic and even the ``none`` arm
keeps the sigmoid Kcb -- so none of those arms is the standard FAO-56 model.
Reaching that model needs the Kcb curve itself to change, which is what this
module adds. Combined with ``transpiration_cover_mode``, it completes the 2x2:

===================  =====================  ==========================
Kcb curve            cover weight           model
===================  =====================  ==========================
linear               ``none``               standard FAO-56 (literature)
linear               ``kcb``                cover-weighted linear
sigmoid              ``none``               logistic Kcb, unweighted
sigmoid              ``kcb``                SWIM default (Kcb-quadratic)
===================  =====================  ==========================

Clipping
--------
The linear form is clipped to ``[0, Kc_max]``. The historical implementation
applied no clip, so an unbounded ramp could drive Kcb negative at low NDVI or
past Kc_max at high NDVI; both are unphysical and the second would silently
break the ``fc`` and root-depth ratios, which assume ``Kcb <= Kc_max``. This is
a deliberate deviation from the legacy code, recorded here because it changes
what "the legacy linear model" means at the tails.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "KCB_MODE_CODES",
    "KCB_MODE_LINEAR",
    "KCB_MODE_NAMES",
    "KCB_MODE_SIGMOID",
    "KCB_LINEAR_PARAMS",
    "KCB_SIGMOID_PARAMS",
    "kcb_mode_parameters",
    "resolve_kcb_mode",
]

KCB_MODE_SIGMOID = 0
KCB_MODE_LINEAR = 1

KCB_MODE_CODES = {
    "sigmoid": KCB_MODE_SIGMOID,
    "linear": KCB_MODE_LINEAR,
}
KCB_MODE_NAMES = {code: name for name, code in KCB_MODE_CODES.items()}

# Free parameters carried by each form. Used by PestBuilder to swap the
# calibrated parameter set, and by the container/report tooling to label them.
KCB_SIGMOID_PARAMS = ("ndvi_k", "ndvi_0")
KCB_LINEAR_PARAMS = ("ndvi_alpha", "ndvi_beta")


def resolve_kcb_mode(mode=None) -> int:
    """Resolve a Kcb-mode name (or code) to its integer code.

    Parameters
    ----------
    mode : str | int | None
        ``sigmoid``/``linear`` or the corresponding code. ``None`` -> sigmoid,
        the historical default.

    Raises
    ------
    ValueError
        If *mode* is not a known name or code.
    TypeError
        If *mode* is neither a string nor an integer.
    """
    if mode is None:
        return KCB_MODE_SIGMOID

    if isinstance(mode, str):
        key = mode.strip().lower()
        if key not in KCB_MODE_CODES:
            raise ValueError(
                f"Unknown kcb NDVI mode {mode!r}. Expected one of {sorted(KCB_MODE_CODES)}."
            )
        return KCB_MODE_CODES[key]

    if isinstance(mode, int | np.integer) and not isinstance(mode, bool):
        code = int(mode)
        if code not in KCB_MODE_NAMES:
            raise ValueError(
                f"Unknown kcb NDVI mode code {code!r}. Expected one of {sorted(KCB_MODE_NAMES)}."
            )
        return code

    raise TypeError(f"kcb NDVI mode must be a str or int, got {type(mode)!r}")


def kcb_mode_parameters(mode=None) -> tuple[str, ...]:
    """Names of the calibrated NDVI-curve parameters used by *mode*."""
    if resolve_kcb_mode(mode) == KCB_MODE_LINEAR:
        return KCB_LINEAR_PARAMS
    return KCB_SIGMOID_PARAMS
