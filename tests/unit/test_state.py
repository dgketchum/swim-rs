"""Unit tests for process package state containers.

Tests verify:
1. Container initialization with defaults
2. Container initialization from explicit values
3. Copy operations produce independent copies
4. Computed properties work correctly
5. Multiplier application for calibration parameters
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from swimrs.process.state import (
    WaterBalanceState,
    FieldProperties,
    CalibrationParameters,
)


class TestWaterBalanceState:
    """Tests for WaterBalanceState container."""

    def test_default_initialization(self):
        """State initializes with zero arrays by default."""
        state = WaterBalanceState(n_fields=5)

        assert state.n_fields == 5
        assert state.depl_root.shape == (5,)
        assert_array_equal(state.depl_root, np.zeros(5))
        assert_array_equal(state.swe, np.zeros(5))
        # Kr and Ks default to 1.0
        assert_array_equal(state.kr, np.ones(5))
        assert_array_equal(state.ks, np.ones(5))
        # Root depth defaults to minimum
        assert_array_equal(state.zr, np.full(5, 0.1))

    def test_explicit_initialization(self):
        """State accepts explicit array values."""
        depl = np.array([10.0, 20.0, 30.0])
        swe = np.array([5.0, 0.0, 15.0])

        state = WaterBalanceState(
            n_fields=3,
            depl_root=depl,
            swe=swe,
        )

        assert_array_equal(state.depl_root, depl)
        assert_array_equal(state.swe, swe)

    def test_from_spinup(self):
        """State can be created from spinup values."""
        n = 4
        depl_root = np.array([50.0, 40.0, 30.0, 20.0])
        swe = np.array([0.0, 10.0, 20.0, 5.0])
        kr = np.array([0.8, 0.9, 1.0, 0.7])
        ks = np.array([1.0, 0.9, 0.8, 0.6])
        zr = np.array([0.5, 0.6, 0.7, 0.4])

        state = WaterBalanceState.from_spinup(
            n_fields=n,
            depl_root=depl_root,
            swe=swe,
            kr=kr,
            ks=ks,
            zr=zr,
        )

        assert_array_equal(state.depl_root, depl_root)
        assert_array_equal(state.swe, swe)
        assert_array_equal(state.kr, kr)
        assert_array_equal(state.ks, ks)
        assert_array_equal(state.zr, zr)

    def test_from_spinup_copies_arrays(self):
        """from_spinup creates copies, not references."""
        depl_root = np.array([50.0, 40.0])
        swe = np.zeros(2)
        kr = np.ones(2)
        ks = np.ones(2)
        zr = np.full(2, 0.5)

        state = WaterBalanceState.from_spinup(
            n_fields=2,
            depl_root=depl_root,
            swe=swe,
            kr=kr,
            ks=ks,
            zr=zr,
        )

        # Modify original
        depl_root[0] = 999.0

        # State should be unaffected
        assert state.depl_root[0] == 50.0

    def test_copy(self):
        """copy() creates independent deep copy."""
        state1 = WaterBalanceState(n_fields=3)
        state1.depl_root = np.array([10.0, 20.0, 30.0])

        state2 = state1.copy()

        # Modify copy
        state2.depl_root[0] = 999.0

        # Original should be unaffected
        assert state1.depl_root[0] == 10.0


class TestFieldProperties:
    """Tests for FieldProperties container."""

    def test_default_initialization(self):
        """Properties initialize with reasonable defaults."""
        props = FieldProperties(n_fields=5)

        assert props.n_fields == 5
        assert props.awc.shape == (5,)
        # Check some defaults
        assert_array_equal(props.awc, np.full(5, 150.0))
        assert_array_equal(props.cn2, np.full(5, 75.0))
        assert_array_equal(props.zr_max, np.full(5, 1.0))
        assert_array_equal(props.irr_status, np.zeros(5, dtype=bool))

    def test_explicit_initialization(self):
        """Properties accept explicit values."""
        awc = np.array([100.0, 150.0, 200.0])
        cn2 = np.array([70.0, 75.0, 80.0])

        props = FieldProperties(
            n_fields=3,
            awc=awc,
            cn2=cn2,
        )

        assert_array_equal(props.awc, awc)
        assert_array_equal(props.cn2, cn2)

    def test_compute_taw(self):
        """TAW = AWC * Zr."""
        props = FieldProperties(n_fields=3)
        props.awc = np.array([100.0, 150.0, 200.0])  # mm/m

        zr = np.array([0.5, 1.0, 0.8])  # m

        taw = props.compute_taw(zr)

        expected = np.array([50.0, 150.0, 160.0])  # mm
        assert_array_almost_equal(taw, expected)

    def test_compute_raw(self):
        """RAW = p * TAW."""
        props = FieldProperties(n_fields=3)
        props.mad = np.array([0.4, 0.5, 0.6])

        taw = np.array([100.0, 150.0, 200.0])

        raw = props.compute_raw(taw)

        expected = np.array([40.0, 75.0, 120.0])
        assert_array_almost_equal(raw, expected)


class TestCalibrationParameters:
    """Tests for CalibrationParameters container."""

    def test_default_initialization(self):
        """Parameters initialize with reasonable defaults."""
        params = CalibrationParameters(n_fields=5)

        assert params.n_fields == 5
        assert_array_equal(params.kc_min, np.full(5, 0.15))
        assert_array_equal(params.ndvi_k, np.full(5, 7.0))
        assert_array_equal(params.swe_alpha, np.full(5, 0.5))

    def test_explicit_initialization(self):
        """Parameters accept explicit values."""
        kc_min = np.array([0.1, 0.15, 0.2])

        params = CalibrationParameters(
            n_fields=3,
            kc_min=kc_min,
        )

        assert_array_equal(params.kc_min, kc_min)

    def test_from_base_with_multipliers(self):
        """Multipliers are applied correctly."""
        base = CalibrationParameters(n_fields=3)
        base.ndvi_k = np.array([7.0, 7.0, 7.0])
        base.swe_alpha = np.array([0.5, 0.5, 0.5])

        multipliers = {
            "ndvi_k": np.array([1.2, 1.0, 0.8]),
            "swe_alpha": np.array([1.5, 1.5, 1.5]),
        }

        params = CalibrationParameters.from_base_with_multipliers(
            base, multipliers
        )

        assert_array_almost_equal(params.ndvi_k, [8.4, 7.0, 5.6])
        assert_array_almost_equal(params.swe_alpha, [0.75, 0.75, 0.75])
        # Unmultiplied param should be unchanged
        assert_array_equal(params.kc_min, base.kc_min)

    def test_from_base_with_multipliers_preserves_base(self):
        """Applying multipliers doesn't modify base."""
        base = CalibrationParameters(n_fields=2)
        base.ndvi_k = np.array([7.0, 7.0])

        multipliers = {"ndvi_k": np.array([2.0, 2.0])}

        _ = CalibrationParameters.from_base_with_multipliers(base, multipliers)

        # Base should be unchanged
        assert_array_equal(base.ndvi_k, [7.0, 7.0])

    def test_copy(self):
        """copy() creates independent deep copy."""
        params1 = CalibrationParameters(n_fields=3)
        params1.ndvi_k = np.array([7.0, 8.0, 9.0])

        params2 = params1.copy()

        # Modify copy
        params2.ndvi_k[0] = 999.0

        # Original should be unaffected
        assert params1.ndvi_k[0] == 7.0

    def test_unknown_multiplier_ignored(self):
        """Unknown multiplier names are silently ignored."""
        base = CalibrationParameters(n_fields=2)

        multipliers = {
            "unknown_param": np.array([2.0, 2.0]),
        }

        # Should not raise
        params = CalibrationParameters.from_base_with_multipliers(
            base, multipliers
        )

        # Params should be unchanged from base
        assert_array_equal(params.ndvi_k, base.ndvi_k)
