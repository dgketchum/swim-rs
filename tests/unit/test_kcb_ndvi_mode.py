"""NDVI-Kcb curve mode tests (standard-FAO-56 arm of the E1 cover-form study).

The cover-form experiment varied only the transpiration cover weight ``fc_t``.
Because ``fc`` is itself derived from ``Kcb``, every cover-weighted arm is
quadratic in the NDVI logistic and even the unweighted arm keeps the logistic
Kcb — so none of them is the standard FAO-56 model. Reaching that model needs
the Kcb curve to become linear in NDVI, which is what ``kcb_ndvi_mode`` adds.

Two pieces have to hold for the arm to mean anything:

1. Mode resolution is strict — an unknown name must fail loudly rather than
   silently falling back to the sigmoid and running the wrong physics.
2. ``PestBuilder`` swaps the *calibrated* parameters with the curve. If it
   kept perturbing ``ndvi_k``/``ndvi_0`` under a linear curve, those would
   have zero sensitivity (silently culled) and the linear curve would run at
   its fixed prior — an unfair comparison masquerading as a calibrated one.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from swimrs.calibrate.pest_builder import PestBuilder
from swimrs.process.kcb_modes import (
    KCB_MODE_LINEAR,
    KCB_MODE_SIGMOID,
    kcb_mode_parameters,
    resolve_kcb_mode,
)
from swimrs.process.kernels.crop_coefficient import kcb_affine, kcb_sigmoid
from swimrs.process.state import CalibrationParameters


def _builder(kcb_ndvi_mode):
    """Minimal PestBuilder for exercising initial_parameter_dict only."""
    b = PestBuilder.__new__(PestBuilder)
    b.params_file = "params.csv"
    b.config = SimpleNamespace(
        kcb_ndvi_mode=kcb_ndvi_mode,
        ssm_calibration=False,
        ssm_surface_capacity=False,
        scheduler_calibration_params=[],
    )
    return b


class TestResolveKcbMode:
    def test_default_is_sigmoid(self):
        assert resolve_kcb_mode(None) == KCB_MODE_SIGMOID

    @pytest.mark.parametrize(
        "name,code", [("sigmoid", KCB_MODE_SIGMOID), ("linear", KCB_MODE_LINEAR)]
    )
    def test_names_and_codes_round_trip(self, name, code):
        assert resolve_kcb_mode(name) == code
        assert resolve_kcb_mode(name.upper()) == code
        assert resolve_kcb_mode(code) == code

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown kcb NDVI mode"):
            resolve_kcb_mode("logistic")

    def test_unknown_code_raises(self):
        with pytest.raises(ValueError, match="Unknown kcb NDVI mode code"):
            resolve_kcb_mode(7)

    def test_bool_rejected(self):
        # bool is an int subclass; accepting it would let a stray toggle
        # silently select a curve.
        with pytest.raises(TypeError):
            resolve_kcb_mode(True)

    def test_mode_parameters(self):
        assert kcb_mode_parameters(None) == ("ndvi_k", "ndvi_0")
        assert kcb_mode_parameters("linear") == ("ndvi_alpha", "ndvi_beta")


class TestKcbAffineKernel:
    def test_matches_closed_form(self):
        ndvi = np.array([0.1, 0.4, 0.7])
        kc_max = np.full(3, 1.35)
        alpha = np.full(3, 0.2)
        beta = np.full(3, 1.25)
        np.testing.assert_allclose(
            kcb_affine(ndvi, kc_max, alpha, beta), np.array([0.325, 0.7, 1.075])
        )

    def test_clipped_both_ends(self):
        ndvi = np.array([-0.2, 0.95])
        kc_max = np.full(2, 1.0)
        out = kcb_affine(ndvi, kc_max, np.full(2, 0.1), np.full(2, 1.7))
        np.testing.assert_allclose(out, np.array([0.0, 1.0]))

    def test_two_parameter_like_the_sigmoid(self):
        """Both curves take exactly two free arrays — neither arm gets more freedom."""
        ndvi = np.linspace(0.05, 0.9, 12)
        kc_max = np.full(12, 1.35)
        sig = kcb_sigmoid(ndvi, kc_max, np.full(12, 10.0), np.full(12, 0.55))
        lin = kcb_affine(ndvi, kc_max, np.full(12, 0.2), np.full(12, 1.25))
        assert sig.shape == lin.shape
        # The forms are genuinely different, not a reparameterization
        assert not np.allclose(sig, lin, atol=0.05)


class TestCalibrationParameterDefaults:
    def test_legacy_linear_priors(self):
        p = CalibrationParameters(n_fields=3)
        np.testing.assert_allclose(p.ndvi_alpha, 0.2)
        np.testing.assert_allclose(p.ndvi_beta, 1.25)

    def test_copy_carries_linear_params(self):
        p = CalibrationParameters(n_fields=2)
        p.ndvi_alpha[:] = -0.3
        p.ndvi_beta[:] = 1.6
        c = p.copy()
        np.testing.assert_array_equal(c.ndvi_alpha, p.ndvi_alpha)
        np.testing.assert_array_equal(c.ndvi_beta, p.ndvi_beta)
        # Deep copy, not a view
        c.ndvi_alpha[:] = 0.0
        assert p.ndvi_alpha[0] == -0.3


class TestPestParameterSwap:
    def test_sigmoid_is_the_default_parameter_set(self):
        p = _builder(None).initial_parameter_dict()
        assert "ndvi_k" in p and "ndvi_0" in p
        assert "ndvi_alpha" not in p and "ndvi_beta" not in p

    def test_explicit_sigmoid_matches_default(self):
        assert list(_builder("sigmoid").initial_parameter_dict()) == list(
            _builder(None).initial_parameter_dict()
        )

    def test_linear_swaps_the_curve_parameters(self):
        p = _builder("linear").initial_parameter_dict()
        assert "ndvi_alpha" in p and "ndvi_beta" in p
        # The sigmoid parameters must be REMOVED, not merely joined: leaving
        # them in would give the linear arm four curve parameters, two of them
        # inert (zero sensitivity -> culled by PEST).
        assert "ndvi_k" not in p and "ndvi_0" not in p

    def test_parameter_count_is_preserved(self):
        assert len(_builder("linear").initial_parameter_dict()) == len(
            _builder(None).initial_parameter_dict()
        )

    def test_legacy_priors(self):
        """Bounds/init reproduce build_pp_files.initial_parameter_dict (pre-3ef4757)."""
        p = _builder("linear").initial_parameter_dict()
        assert p["ndvi_alpha"]["initial_value"] == 0.2
        assert (p["ndvi_alpha"]["lower_bound"], p["ndvi_alpha"]["upper_bound"]) == (-0.7, 1.5)
        assert p["ndvi_beta"]["initial_value"] == 1.25
        assert (p["ndvi_beta"]["lower_bound"], p["ndvi_beta"]["upper_bound"]) == (0.5, 1.7)
        assert p["ndvi_alpha"]["pargp"] == "ndvi_alpha"
        assert p["ndvi_beta"]["pargp"] == "ndvi_beta"


class TestRegularizationGroups:
    def test_sigmoid_groups_unchanged(self):
        b = _builder(None)
        b.config.prior_regularization_params = None
        groups = b._regularization_param_groups()
        assert "ndvi_k" in groups and "ndvi_0" in groups

    def test_linear_substitutes_curve_params(self):
        """Otherwise the linear arm silently loses NDVI-curve regularization."""
        b = _builder("linear")
        b.config.prior_regularization_params = None
        groups = b._regularization_param_groups()
        assert "ndvi_alpha" in groups and "ndvi_beta" in groups
        assert "ndvi_k" not in groups and "ndvi_0" not in groups

    def test_configured_list_also_substituted(self):
        b = _builder("linear")
        b.config.prior_regularization_params = ["aw", "ndvi_k", "ndvi_0", "mad"]
        groups = b._regularization_param_groups()
        assert groups[:2] == ["aw", "mad"]
        assert set(groups) == {"aw", "mad", "ndvi_alpha", "ndvi_beta"}
