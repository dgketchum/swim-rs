"""Tests for swimrs.data_extraction.gridmet.gridmet pure logic functions."""

import numpy as np

from swimrs.data_extraction.gridmet.gridmet import (
    actual_vapor_pressure,
    air_pressure,
    wind_height_adjust,
)


class TestAirPressure:
    """Tests for air_pressure function."""

    def test_sea_level_returns_101_3(self):
        """Air pressure at sea level (0m) should be ~101.3 kPa."""
        result = air_pressure(0)
        assert np.isclose(result[0], 101.3, rtol=0.01)

    def test_higher_elevation_returns_lower_pressure(self):
        """Air pressure decreases with elevation."""
        sea_level = air_pressure(0)[0]
        high_elev = air_pressure(2000)[0]
        assert high_elev < sea_level

    def test_known_elevation_value_asce(self):
        """Test known value at 1000m using ASCE method."""
        # At 1000m, pressure should be approximately 89.9 kPa (ASCE formula)
        result = air_pressure(1000, method="asce")
        assert np.isclose(result[0], 89.9, rtol=0.02)

    def test_known_elevation_value_refet(self):
        """Test known value at 1000m using RefET method."""
        result = air_pressure(1000, method="refet")
        # RefET method uses slightly different exponent
        assert 85 < result[0] < 95

    def test_array_input(self):
        """air_pressure accepts array input."""
        elevations = [0, 500, 1000, 1500, 2000]
        result = air_pressure(elevations)
        assert len(result) == 5
        # Should be monotonically decreasing
        assert all(result[i] > result[i + 1] for i in range(4))

    def test_returns_ndarray(self):
        """air_pressure returns numpy ndarray."""
        result = air_pressure(500)
        assert isinstance(result, np.ndarray)

    def test_negative_elevation(self):
        """air_pressure handles below-sea-level elevation."""
        # Death Valley is ~-86m
        result = air_pressure(-86)
        sea_level = air_pressure(0)[0]
        assert result[0] > sea_level


class TestActualVaporPressure:
    """Tests for actual_vapor_pressure function."""

    def test_returns_positive_value(self):
        """Actual vapor pressure should be positive."""
        q = 0.01  # 10 g/kg specific humidity
        pair = 100.0  # kPa
        result = actual_vapor_pressure(q, pair)
        assert result[0] > 0

    def test_formula_correctness(self):
        """Test against manual calculation: ea = q * pair / (0.622 + 0.378 * q)."""
        q = 0.008  # 8 g/kg
        pair = 95.0  # kPa
        expected = q * pair / (0.622 + 0.378 * q)
        result = actual_vapor_pressure(q, pair)
        assert np.isclose(result[0], expected, rtol=1e-6)

    def test_higher_humidity_higher_pressure(self):
        """Higher specific humidity -> higher vapor pressure."""
        pair = 100.0
        low_q = actual_vapor_pressure(0.005, pair)[0]
        high_q = actual_vapor_pressure(0.015, pair)[0]
        assert high_q > low_q

    def test_array_input(self):
        """actual_vapor_pressure accepts array inputs."""
        q = np.array([0.005, 0.01, 0.015])
        pair = np.array([100, 95, 90])
        result = actual_vapor_pressure(q, pair)
        assert len(result) == 3

    def test_returns_ndarray(self):
        """actual_vapor_pressure returns numpy ndarray."""
        result = actual_vapor_pressure(0.01, 100)
        assert isinstance(result, np.ndarray)

    def test_zero_humidity_returns_zero(self):
        """Zero specific humidity returns zero vapor pressure."""
        result = actual_vapor_pressure(0, 100)
        assert result[0] == 0


class TestWindHeightAdjust:
    """Tests for wind_height_adjust function."""

    def test_2m_height_returns_same(self):
        """Wind at 2m measurement height returns approximately same value."""
        uz = 5.0  # m/s
        zw = 2.0  # measured at 2m
        result = wind_height_adjust(uz, zw)
        # At 2m, the adjustment should be close to 1
        # 4.87 / log(67.8 * 2 - 5.42) = 4.87 / log(130.18) = 4.87 / 4.87 = 1.0
        assert np.isclose(result, uz, rtol=0.01)

    def test_higher_measurement_reduces_speed(self):
        """Wind measured at higher height has larger adjustment (lower u2)."""
        uz = 5.0
        at_2m = wind_height_adjust(uz, 2.0)
        at_10m = wind_height_adjust(uz, 10.0)
        # Wind at 10m is higher, so u2 (adjusted to 2m) should be lower
        assert at_10m < at_2m

    def test_formula_correctness(self):
        """Test against manual calculation."""
        uz = 3.0
        zw = 10.0
        expected = uz * 4.87 / np.log(67.8 * zw - 5.42)
        result = wind_height_adjust(uz, zw)
        assert np.isclose(result, expected, rtol=1e-6)

    def test_array_input(self):
        """wind_height_adjust handles array input."""
        uz = np.array([2.0, 3.0, 4.0])
        zw = np.array([10, 10, 10])
        result = wind_height_adjust(uz, zw)
        assert len(result) == 3

    def test_scalar_input(self):
        """wind_height_adjust handles scalar input."""
        result = wind_height_adjust(5.0, 10.0)
        assert isinstance(result, (float, np.floating))

    def test_typical_station_height(self):
        """Test with typical 10m wind measurement station."""
        # 5 m/s at 10m should be reduced at 2m
        uz = 5.0
        zw = 10.0
        result = wind_height_adjust(uz, zw)
        # Result should be positive and less than original
        assert 0 < result < uz


class TestAirPressureEdgeCases:
    """Edge case tests for air_pressure."""

    def test_very_high_elevation(self):
        """Test at very high elevation (Mt. Everest ~8848m)."""
        result = air_pressure(8848)
        # Should be very low but positive
        assert 0 < result[0] < 40

    def test_float_input(self):
        """air_pressure handles float input."""
        result = air_pressure(1234.56)
        assert len(result) == 1

    def test_numpy_array_input(self):
        """air_pressure handles numpy array input directly."""
        elevs = np.array([100, 200, 300])
        result = air_pressure(elevs)
        assert result.shape == (3,)


class TestVaporPressureEdgeCases:
    """Edge case tests for actual_vapor_pressure."""

    def test_typical_range(self):
        """Test with typical atmospheric values."""
        # Typical specific humidity range: 0.001 to 0.025 kg/kg
        # Typical pressure: 80-101 kPa
        q = 0.012
        pair = 95
        result = actual_vapor_pressure(q, pair)
        # Result should be in reasonable range (0.5 to 4 kPa typical)
        assert 0.5 < result[0] < 4

    def test_broadcast_scalar_pair(self):
        """Test broadcasting scalar pair with array q."""
        q = np.array([0.005, 0.010, 0.015])
        pair = 100.0  # scalar
        result = actual_vapor_pressure(q, pair)
        assert len(result) == 3
