"""WP-C5 surface-capacity (REW/TEW) calibration tests.

Covers the two load-bearing pieces of reviving REW/TEW as SSM-gated calibratable
parameters (removed historically in 5f79a8c because ETf alone cannot identify them):

1. Gating — ``initial_parameter_dict`` exposes ``rew``/``tew`` ONLY when both
   ``ssm_calibration`` and ``ssm_surface_capacity`` are True, so every prior run
   (Ex5/6/7, e8cal, e8c1, e8c4) is byte-for-byte unchanged.
2. Property injection — ``load_pest_mult_properties`` routes the PEST
   ``p_rew_*`` / ``p_tew_*`` mult files into ``FieldProperties.rew`` / ``.tew``.
   Without this coupling the parameters would be perturbed by PEST but ignored by
   the model (zero sensitivity → silently culled). This is the failure mode the
   scoping flagged, so it gets an explicit test.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from swimrs.calibrate.pest_builder import PestBuilder
from swimrs.process.state import FieldProperties, load_pest_mult_properties


def _builder(*, ssm_calibration, ssm_surface_capacity):
    """Minimal PestBuilder for exercising initial_parameter_dict gating only."""
    b = PestBuilder.__new__(PestBuilder)
    b.params_file = "params.csv"
    b.config = SimpleNamespace(
        ssm_calibration=ssm_calibration,
        ssm_surface_capacity=ssm_surface_capacity,
        scheduler_calibration_params=[],
    )
    return b


def test_surface_capacity_gated_off_by_default():
    p = _builder(ssm_calibration=False, ssm_surface_capacity=False).initial_parameter_dict()
    assert "rew" not in p and "tew" not in p


def test_surface_capacity_requires_ssm_enabled():
    # ssm_surface_capacity alone (SSM off) must NOT expose rew/tew — the SMAP
    # observation is the identifying constraint.
    p = _builder(ssm_calibration=False, ssm_surface_capacity=True).initial_parameter_dict()
    assert "rew" not in p and "tew" not in p


def test_ssm_alone_does_not_free_surface_capacity():
    # SSM on but surface-capacity off (the e8c4 configuration) leaves rew/tew fixed.
    p = _builder(ssm_calibration=True, ssm_surface_capacity=False).initial_parameter_dict()
    assert "rew" not in p and "tew" not in p


def test_surface_capacity_params_present_when_enabled():
    p = _builder(ssm_calibration=True, ssm_surface_capacity=True).initial_parameter_dict()
    assert "rew" in p and "tew" in p
    assert p["rew"]["pargp"] == "rew" and p["tew"]["pargp"] == "tew"
    # Bounds/init reproduce the historical FAO-56 values (rew 2-6 mm, tew 6-29 mm).
    assert (p["rew"]["lower_bound"], p["rew"]["upper_bound"]) == (2.0, 6.0)
    assert (p["tew"]["lower_bound"], p["tew"]["upper_bound"]) == (6.0, 29.0)
    assert p["rew"]["initial_value"] == 3.0 and p["tew"]["initial_value"] == 18.0


def _write_mult(mult_dir, pest_name, fid, value):
    """Emulate a PEST++ per-field constant mult file (p_{name}_{fid}_0_constant.csv)."""
    pd.DataFrame({"1": [value]}, index=[0]).to_csv(mult_dir / f"p_{pest_name}_{fid}_0_constant.csv")


def test_rew_tew_mult_injection(tmp_path):
    """PEST rew/tew mult files must reach FieldProperties.rew/.tew (the coupling)."""
    fids = ["siteA", "siteB"]
    base = FieldProperties(n_fields=2)
    base.rew = np.array([3.0, 3.0])
    base.tew = np.array([18.0, 18.0])
    # Only siteA (index 0) is "calibrated" here.
    _write_mult(tmp_path, "rew", "siteA", 5.4)
    _write_mult(tmp_path, "tew", "siteA", 22.0)

    props = load_pest_mult_properties(str(tmp_path), fids, base)

    assert np.isclose(props.rew[0], 5.4)
    assert np.isclose(props.tew[0], 22.0)
    # siteB has no mult file -> keeps the base value (masked-assignment semantics).
    assert np.isclose(props.rew[1], 3.0)
    assert np.isclose(props.tew[1], 18.0)
