"""Tests for physics helper functions in swimrs.data_extraction.gridmet.gridmet.

Tests cover:
- air_pressure(): sea level, elevation dependence, method variants, array broadcast
- actual_vapor_pressure(): reference values, zero q, monotonicity
- wind_height_adjust(): height adjustment direction, reference height behavior
"""

import numpy as np
from numpy.testing import assert_allclose

from swimrs.data_extraction.gridmet.gridmet import (
    actual_vapor_pressure,
    air_pressure,
    wind_height_adjust,
)


class TestAirPressure:
    """Tests for air_pressure()."""

    def test_sea_level_approx_101_3(self):
        """Sea level elevation (0 m) produces ~101.3 kPa."""
        result = air_pressure(0.0)
        assert_allclose(result, 101.3, atol=0.1)

    def test_high_elevation_less_than_sea_level(self):
        """Pressure at 2000 m is less than at sea level."""
        p_sea = air_pressure(0.0)
        p_high = air_pressure(2000.0)
        assert float(p_high) < float(p_sea)

    def test_monotonically_decreasing_with_elevation(self):
        """Pressure decreases monotonically with elevation."""
        elevations = np.array([0, 500, 1000, 1500, 2000, 3000])
        pressures = air_pressure(elevations)
        for i in range(len(pressures) - 1):
            assert pressures[i] > pressures[i + 1]

    def test_asce_vs_refet_differ(self):
        """ASCE and RefET methods produce different values at nonzero elevation."""
        p_asce = air_pressure(1000.0, method="asce")
        p_refet = air_pressure(1000.0, method="refet")
        # They should be close but not identical
        assert not np.isclose(p_asce, p_refet, atol=1e-6)

    def test_array_broadcast(self):
        """air_pressure handles array input."""
        elevations = np.array([0.0, 500.0, 1000.0])
        result = air_pressure(elevations)
        assert result.shape == (3,)
        assert all(np.isfinite(result))

    def test_scalar_input_returns_array(self):
        """Scalar input returns ndarray."""
        result = air_pressure(0.0)
        assert isinstance(result, np.ndarray)


class TestActualVaporPressure:
    """Tests for actual_vapor_pressure()."""

    def test_reference_value(self):
        """Known reference: q=0.005, pair=101.3 -> ea approx 0.808 kPa."""
        ea = actual_vapor_pressure(0.005, 101.3)
        # ea = q * pair / (0.622 + 0.378 * q)
        expected = 0.005 * 101.3 / (0.622 + 0.378 * 0.005)
        assert_allclose(ea, expected, rtol=1e-4)

    def test_zero_q_returns_zero(self):
        """Zero specific humidity gives zero vapor pressure."""
        ea = actual_vapor_pressure(0.0, 101.3)
        assert_allclose(ea, 0.0, atol=1e-10)

    def test_monotonically_increasing_with_q(self):
        """Vapor pressure increases monotonically with specific humidity."""
        q_values = np.array([0.001, 0.003, 0.005, 0.010, 0.020])
        ea_values = actual_vapor_pressure(q_values, 101.3)
        for i in range(len(ea_values) - 1):
            assert ea_values[i] < ea_values[i + 1]

    def test_positive_inputs_give_positive_output(self):
        """Positive q and pair give positive vapor pressure."""
        ea = actual_vapor_pressure(0.01, 90.0)
        assert float(ea) > 0


class TestWindHeightAdjust:
    """Tests for wind_height_adjust()."""

    def test_10m_output_less_than_input(self):
        """Wind from zw=10 m adjusted to 2 m is less than input."""
        uz = 5.0
        u2 = wind_height_adjust(uz, 10.0)
        assert float(u2) < uz

    def test_higher_measurement_height_smaller_adjusted(self):
        """Higher measurement height produces smaller adjusted wind."""
        uz = 5.0
        u2_10 = wind_height_adjust(uz, 10.0)
        u2_20 = wind_height_adjust(uz, 20.0)
        assert float(u2_20) < float(u2_10)

    def test_2m_height_returns_identity(self):
        """At zw=2, adjustment should return approximately the input."""
        uz = 5.0
        u2 = wind_height_adjust(uz, 2.0)
        # uz * 4.87 / log(67.8*2 - 5.42) = uz * 4.87 / log(130.18)
        # = uz * 4.87 / 4.87 = uz  (approximately)
        assert_allclose(u2, uz, rtol=0.01)
