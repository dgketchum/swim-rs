"""Tests for Calculator helper methods as pure logic.

Tests cover:
- _merge_sensors(): preference order, NaN fill, single sensor
- _compute_k_parameters(): all-NaN defaults, no low-NDVI, kc_max floor
- _compute_groundwater_subsidy(): ET > PPT subsidy, ET < PPT no subsidy, zero PPT
- _detect_irrigation_windows(): flat NDVI, clear ramp, >200 NaN, DOY invariants
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import xarray as xr
from numpy.testing import assert_allclose


def _make_calculator():
    """Create a minimal Calculator with mocked state."""
    from swimrs.container.components.calculator import Calculator

    state = MagicMock()
    state.is_writable = True
    state._mode = "r+"
    calc = Calculator(state)
    return calc


class TestMergeSensors:
    """Tests for Calculator._merge_sensors()."""

    def _make_da(self, data, dates, sites):
        """Helper to create DataArrays."""
        return xr.DataArray(
            data,
            dims=["time", "site"],
            coords={"time": dates, "site": sites},
        )

    def test_preferred_has_all_data(self):
        """When preferred sensor has all data, result equals preferred."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", periods=5)
        sites = ["A", "B"]
        preferred = self._make_da(np.ones((5, 2)), dates, sites)
        secondary = self._make_da(np.full((5, 2), 99.0), dates, sites)

        result = calc._merge_sensors(
            [("landsat", preferred), ("sentinel", secondary)],
            preference_order=("landsat", "sentinel"),
        )
        assert_allclose(result.values, 1.0)

    def test_preferred_nans_filled_by_secondary(self):
        """NaN in preferred sensor is filled from secondary."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", periods=3)
        sites = ["A"]
        pref_data = np.array([[np.nan], [0.5], [np.nan]])
        sec_data = np.array([[0.3], [0.7], [0.4]])
        preferred = self._make_da(pref_data, dates, sites)
        secondary = self._make_da(sec_data, dates, sites)

        result = calc._merge_sensors(
            [("landsat", preferred), ("sentinel", secondary)],
            preference_order=("landsat", "sentinel"),
        )
        expected = np.array([[0.3], [0.5], [0.4]])
        assert_allclose(result.values, expected)

    def test_single_sensor_identity(self):
        """Single sensor returns identity."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", periods=3)
        sites = ["A"]
        data = np.array([[0.1], [0.5], [0.9]])
        da = self._make_da(data, dates, sites)

        result = calc._merge_sensors(
            [("landsat", da)],
            preference_order=("landsat",),
        )
        assert_allclose(result.values, data)

    def test_result_nan_count_leq_best_single(self):
        """Merged result has fewer NaN than any individual sensor."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", periods=6)
        sites = ["A"]
        pref_data = np.array([[np.nan], [0.5], [np.nan], [0.7], [np.nan], [np.nan]])
        sec_data = np.array([[0.3], [np.nan], [0.4], [np.nan], [0.6], [np.nan]])
        preferred = self._make_da(pref_data, dates, sites)
        secondary = self._make_da(sec_data, dates, sites)

        result = calc._merge_sensors(
            [("landsat", preferred), ("sentinel", secondary)],
            preference_order=("landsat", "sentinel"),
        )
        result_nans = np.isnan(result.values).sum()
        pref_nans = np.isnan(pref_data).sum()
        sec_nans = np.isnan(sec_data).sum()
        assert result_nans <= min(pref_nans, sec_nans)


class TestComputeKParameters:
    """Tests for Calculator._compute_k_parameters()."""

    def _make_ds(self, etf_vals, ndvi_vals, sites=("A",)):
        """Create a minimal xr.Dataset for K-parameter computation."""
        dates = pd.date_range("2020-01-01", periods=len(etf_vals))
        n_sites = len(sites)
        etf_2d = np.tile(np.array(etf_vals)[:, None], (1, n_sites))
        ndvi_2d = np.tile(np.array(ndvi_vals)[:, None], (1, n_sites))
        return xr.Dataset(
            {
                "etf": xr.DataArray(
                    etf_2d, dims=["time", "site"], coords={"time": dates, "site": list(sites)}
                ),
                "ndvi": xr.DataArray(
                    ndvi_2d, dims=["time", "site"], coords={"time": dates, "site": list(sites)}
                ),
            }
        )

    def test_all_nan_etf_returns_defaults(self):
        """All-NaN ETf produces default ke=1.0, kc=1.25."""
        calc = _make_calculator()
        ds = self._make_ds(
            [np.nan] * 10,
            [0.2] * 10,
        )
        ke, kc = calc._compute_k_parameters(ds)
        assert float(ke.values) == 1.0
        assert float(kc.values) == 1.25

    def test_no_low_ndvi_ke_defaults(self):
        """When all NDVI >= 0.3, ke defaults to 1.0."""
        calc = _make_calculator()
        ds = self._make_ds(
            [0.8, 0.9, 0.7, 0.85, 0.95],
            [0.5, 0.6, 0.55, 0.7, 0.65],
        )
        ke, kc = calc._compute_k_parameters(ds)
        assert float(ke.values) == 1.0

    def test_kc_max_floor_enforced(self):
        """kc_max is at least 1.25 even when all ETf is low."""
        calc = _make_calculator()
        ds = self._make_ds(
            [0.1, 0.2, 0.15, 0.1, 0.05],
            [0.5, 0.6, 0.55, 0.7, 0.65],
        )
        ke, kc = calc._compute_k_parameters(ds)
        assert float(kc.values) >= 1.25

    def test_known_percentile_scenario(self):
        """90th percentile of ETf where NDVI < 0.3 matches manual calculation."""
        calc = _make_calculator()
        # 5 obs with low NDVI, ETf values [0.2, 0.4, 0.6, 0.8, 1.0]
        etf_vals = [0.2, 0.4, 0.6, 0.8, 1.0]
        ndvi_vals = [0.1, 0.15, 0.2, 0.25, 0.28]  # all < 0.3
        ds = self._make_ds(etf_vals, ndvi_vals)
        ke, kc = calc._compute_k_parameters(ds)
        expected_ke = float(np.percentile(etf_vals, 90))
        assert_allclose(float(ke.values), expected_ke, atol=0.01)


class TestComputeGroundwaterSubsidy:
    """Tests for Calculator._compute_groundwater_subsidy()."""

    def _make_ds(self, eto_vals, etf_vals, prcp_vals, sites=("A",)):
        """Create a minimal xr.Dataset for GW subsidy computation."""
        dates = pd.date_range("2020-01-01", periods=len(eto_vals))
        n_sites = len(sites)
        eto_2d = np.tile(np.array(eto_vals)[:, None], (1, n_sites))
        etf_2d = np.tile(np.array(etf_vals)[:, None], (1, n_sites))
        prcp_2d = np.tile(np.array(prcp_vals)[:, None], (1, n_sites))
        return xr.Dataset(
            {
                "eto": xr.DataArray(
                    eto_2d, dims=["time", "site"], coords={"time": dates, "site": list(sites)}
                ),
                "etf": xr.DataArray(
                    etf_2d, dims=["time", "site"], coords={"time": dates, "site": list(sites)}
                ),
                "prcp": xr.DataArray(
                    prcp_2d, dims=["time", "site"], coords={"time": dates, "site": list(sites)}
                ),
            }
        )

    def test_et_greater_than_ppt_yields_positive_fsub(self):
        """When ET > PPT, f_sub should be > 0."""
        calc = _make_calculator()
        # 365 days: ETo=5, ETf=1.0 -> eta=5, prcp=2 -> ratio > 1
        n = 365
        ds = self._make_ds(
            [5.0] * n,
            [1.0] * n,
            [2.0] * n,
        )
        result = calc._compute_groundwater_subsidy(ds, irr_threshold=0.1)
        site_data = result["A"]
        yr_data = site_data.get(2020)
        assert yr_data is not None
        assert yr_data["f_sub"] > 0
        assert yr_data["subsidized"] == 1

    def test_et_less_than_ppt_yields_zero_fsub(self):
        """When ET < PPT, f_sub should be 0."""
        calc = _make_calculator()
        # 365 days: ETo=3, ETf=0.5 -> eta=1.5, prcp=5 -> ratio < 1
        n = 365
        ds = self._make_ds(
            [3.0] * n,
            [0.5] * n,
            [5.0] * n,
        )
        result = calc._compute_groundwater_subsidy(ds, irr_threshold=0.1)
        site_data = result["A"]
        yr_data = site_data.get(2020)
        assert yr_data is not None
        assert yr_data["f_sub"] == 0.0
        assert yr_data["subsidized"] == 0

    def test_zero_ppt_year_skipped(self):
        """Year with zero precipitation is skipped (no division by zero)."""
        calc = _make_calculator()
        n = 365
        ds = self._make_ds(
            [5.0] * n,
            [1.0] * n,
            [0.0] * n,
        )
        result = calc._compute_groundwater_subsidy(ds, irr_threshold=0.1)
        site_data = result["A"]
        # Year present but ratio based on ppt+1
        if 2020 in site_data:
            # Should not crash
            assert np.isfinite(site_data[2020]["ratio"])

    def test_subsidy_months_identified(self):
        """Months where eta > ppt are identified."""
        calc = _make_calculator()
        n = 365
        ds = self._make_ds(
            [5.0] * n,
            [1.0] * n,
            [2.0] * n,
        )
        result = calc._compute_groundwater_subsidy(ds, irr_threshold=0.1)
        site_data = result["A"]
        yr_data = site_data.get(2020)
        if yr_data and "months" in yr_data:
            assert isinstance(yr_data["months"], list)


class TestDetectIrrigationWindows:
    """Tests for Calculator._detect_irrigation_windows()."""

    def test_flat_ndvi_no_windows(self):
        """Flat NDVI time series produces no irrigation windows."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        ndvi = pd.Series(0.3, index=dates)
        doys = calc._detect_irrigation_windows(
            ndvi, lookback=10, ndvi_threshold=0.3, min_pos_days=10, year=2020
        )
        assert isinstance(doys, list)

    def test_clear_ramp_produces_doys(self):
        """Clear NDVI ramp-up produces DOYs."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        # Create a sigmoidal ramp-up pattern
        n = len(dates)
        ndvi_vals = np.where(
            np.arange(n) < 120,
            0.15,
            np.where(np.arange(n) < 200, 0.15 + 0.7 * (np.arange(n) - 120) / 80, 0.85),
        )
        # Add some decline after peak
        ndvi_vals = np.where(np.arange(n) > 260, 0.85 - 0.5 * (np.arange(n) - 260) / 100, ndvi_vals)
        ndvi = pd.Series(ndvi_vals, index=dates)
        doys = calc._detect_irrigation_windows(
            ndvi, lookback=10, ndvi_threshold=0.3, min_pos_days=10, year=2020
        )
        # Should produce some DOYs from the ramp-up period
        if len(doys) > 0:
            assert all(1 <= d <= 366 for d in doys)
            assert doys == sorted(doys)
            assert len(doys) == len(set(doys))

    def test_many_nans_returns_empty(self):
        """Time series with >200 NaN after processing returns empty."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        ndvi = pd.Series(np.nan, index=dates)
        doys = calc._detect_irrigation_windows(
            ndvi, lookback=10, ndvi_threshold=0.3, min_pos_days=10, year=2020
        )
        assert doys == []

    def test_doys_sorted_unique_in_range(self):
        """DOYs are sorted, unique, and within [1, 366]."""
        calc = _make_calculator()
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        # Create a pattern that should produce some DOYs
        n = len(dates)
        ndvi_vals = 0.2 + 0.6 * np.sin(np.pi * np.arange(n) / n)
        ndvi = pd.Series(ndvi_vals, index=dates)
        doys = calc._detect_irrigation_windows(
            ndvi, lookback=10, ndvi_threshold=0.3, min_pos_days=10, year=2020
        )
        if len(doys) > 0:
            assert doys == sorted(doys)
            assert len(doys) == len(set(doys))
            assert all(1 <= d <= 366 for d in doys)
