"""Tests for Exporter._build_switched_etf() mask switching logic.

Tests cover:
- Base mask preference order: inv_irr > no_mask > irr > None
- Irrigated year switches to irr mask
- Non-irrigated year keeps base mask
- fallow_years key skipped
- Missing field returns None
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from swimrs.container.components.exporter import Exporter


def _make_exporter():
    """Create a minimal Exporter with mocked state."""
    state = MagicMock()
    state.is_writable = True
    state._mode = "r+"
    return Exporter(state)


def _make_etf_da(values, dates, sites):
    """Create an ETf DataArray."""
    return xr.DataArray(
        values,
        dims=["time", "site"],
        coords={"time": dates, "site": sites},
    )


class TestBuildSwitchedEtf:
    """Tests for Exporter._build_switched_etf()."""

    @pytest.fixture
    def dates(self):
        return pd.date_range("2020-01-01", "2020-12-31", freq="D")

    @pytest.fixture
    def time_index(self, dates):
        return dates

    def test_inv_irr_preferred_over_irr(self, dates, time_index):
        """inv_irr is preferred as base mask over irr."""
        exp = _make_exporter()
        n = len(dates)
        etf_data = {
            "inv_irr": _make_etf_da(np.full((n, 1), 0.5), dates, ["A"]),
            "irr": _make_etf_da(np.full((n, 1), 0.9), dates, ["A"]),
        }
        irr_data = {}
        result = exp._build_switched_etf(
            "A", etf_data, irr_data, ("irr", "inv_irr"), 0.1, time_index
        )
        assert result is not None
        # Should be inv_irr values since field not irrigated
        assert np.allclose(result, 0.5)

    def test_no_mask_fallback(self, dates, time_index):
        """no_mask is used when inv_irr not available."""
        exp = _make_exporter()
        n = len(dates)
        etf_data = {
            "no_mask": _make_etf_da(np.full((n, 1), 0.6), dates, ["A"]),
            "irr": _make_etf_da(np.full((n, 1), 0.9), dates, ["A"]),
        }
        irr_data = {}
        result = exp._build_switched_etf(
            "A", etf_data, irr_data, ("irr", "no_mask"), 0.1, time_index
        )
        assert result is not None
        assert np.allclose(result, 0.6)

    def test_irr_only_fallback(self, dates, time_index):
        """irr is used when neither inv_irr nor no_mask available."""
        exp = _make_exporter()
        n = len(dates)
        etf_data = {
            "irr": _make_etf_da(np.full((n, 1), 0.8), dates, ["A"]),
        }
        irr_data = {}
        result = exp._build_switched_etf("A", etf_data, irr_data, ("irr",), 0.1, time_index)
        assert result is not None
        assert np.allclose(result, 0.8)

    def test_no_etf_data_returns_none(self, dates, time_index):
        """Empty etf_data returns None."""
        exp = _make_exporter()
        result = exp._build_switched_etf("A", {}, {}, ("irr", "inv_irr"), 0.1, time_index)
        assert result is None

    def test_irrigated_year_switches_to_irr(self, dates, time_index):
        """Irrigated year (f_irr >= threshold) uses irr mask values."""
        exp = _make_exporter()
        n = len(dates)
        inv_vals = np.full((n, 1), 0.5)
        irr_vals = np.full((n, 1), 0.9)
        etf_data = {
            "inv_irr": _make_etf_da(inv_vals, dates, ["A"]),
            "irr": _make_etf_da(irr_vals, dates, ["A"]),
        }
        irr_data = {
            "A": {
                "2020": {"f_irr": 0.8, "irr_doys": [100, 200]},
                "fallow_years": [],
            }
        }
        result = exp._build_switched_etf(
            "A", etf_data, irr_data, ("irr", "inv_irr"), 0.1, time_index
        )
        # All of 2020 should be switched to irr values
        assert np.allclose(result, 0.9)

    def test_non_irrigated_year_keeps_base(self, dates, time_index):
        """Non-irrigated year (f_irr < threshold) keeps base mask."""
        exp = _make_exporter()
        n = len(dates)
        etf_data = {
            "inv_irr": _make_etf_da(np.full((n, 1), 0.5), dates, ["A"]),
            "irr": _make_etf_da(np.full((n, 1), 0.9), dates, ["A"]),
        }
        irr_data = {
            "A": {
                "2020": {"f_irr": 0.05, "irr_doys": []},
                "fallow_years": [2020],
            }
        }
        result = exp._build_switched_etf(
            "A", etf_data, irr_data, ("irr", "inv_irr"), 0.1, time_index
        )
        assert np.allclose(result, 0.5)

    def test_fallow_years_key_skipped(self, dates, time_index):
        """fallow_years key does not crash the switcher."""
        exp = _make_exporter()
        n = len(dates)
        etf_data = {
            "inv_irr": _make_etf_da(np.full((n, 1), 0.5), dates, ["A"]),
            "irr": _make_etf_da(np.full((n, 1), 0.9), dates, ["A"]),
        }
        irr_data = {
            "A": {
                "fallow_years": [2020],
                "2020": {"f_irr": 0.0, "irr_doys": []},
            }
        }
        # Should not raise
        result = exp._build_switched_etf(
            "A", etf_data, irr_data, ("irr", "inv_irr"), 0.1, time_index
        )
        assert result is not None

    def test_missing_field_returns_none(self, dates, time_index):
        """Field not in DataArray returns None."""
        exp = _make_exporter()
        n = len(dates)
        etf_data = {
            "inv_irr": _make_etf_da(np.full((n, 1), 0.5), dates, ["B"]),
        }
        result = exp._build_switched_etf("A", etf_data, {}, ("irr", "inv_irr"), 0.1, time_index)
        assert result is None
