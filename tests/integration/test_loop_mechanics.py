"""Unit tests for process package day loop orchestration.

Tests verify:
1. DailyOutput initialization
2. step_day executes correctly
3. run_daily_loop integrates properly
4. Physical constraints maintained
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from swimrs.process.input import build_swim_input
from swimrs.process.loop import DailyOutput, run_daily_loop, step_day
from swimrs.process.state import (
    CalibrationParameters,
    FieldProperties,
    WaterBalanceState,
)


class TestDailyOutput:
    """Tests for DailyOutput container."""

    def test_initialization(self):
        """DailyOutput initializes with correct shapes."""
        output = DailyOutput(n_days=10, n_fields=5)

        assert output.eta.shape == (10, 5)
        assert output.etf.shape == (10, 5)
        assert output.kcb.shape == (10, 5)
        assert output.runoff.shape == (10, 5)

    def test_initialized_to_zeros(self):
        """Arrays are initialized to zeros."""
        output = DailyOutput(n_days=5, n_fields=3)

        assert_array_almost_equal(output.eta, np.zeros((5, 3)))
        assert_array_almost_equal(output.swe, np.zeros((5, 3)))


class TestStepDay:
    """Tests for step_day function."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple test setup with 3 fields."""
        n = 3
        state = WaterBalanceState(n_fields=n)
        state.depl_root = np.array([30.0, 50.0, 70.0])
        state.swe = np.array([0.0, 10.0, 0.0])
        state.zr = np.array([0.5, 0.6, 0.7])

        props = FieldProperties(n_fields=n)
        props.awc = np.array([150.0, 150.0, 150.0])
        props.cn2 = np.array([75.0, 75.0, 75.0])
        props.zr_max = np.array([1.0, 1.0, 1.0])
        props.irr_status = np.array([False, True, False])

        params = CalibrationParameters(n_fields=n)

        return state, props, params

    def test_step_day_executes(self, simple_setup):
        """step_day runs without error."""
        state, props, params = simple_setup
        n = 3

        ndvi = np.array([0.3, 0.5, 0.7])
        etr = np.array([5.0, 5.0, 5.0])
        prcp = np.array([0.0, 0.0, 10.0])
        tmin = np.array([10.0, 10.0, 10.0])
        tmax = np.array([25.0, 25.0, 25.0])
        srad = np.array([20.0, 20.0, 20.0])
        irr_flag = np.array([False, True, False])

        result = step_day(state, props, params, ndvi, etr, prcp, tmin, tmax, srad, irr_flag)

        assert "eta" in result
        assert "etf" in result
        assert "kcb" in result
        assert result["eta"].shape == (n,)

    def test_eta_positive(self, simple_setup):
        """ET is non-negative."""
        state, props, params = simple_setup

        ndvi = np.array([0.3, 0.5, 0.7])
        etr = np.array([5.0, 5.0, 5.0])
        prcp = np.array([0.0, 0.0, 0.0])
        tmin = np.array([10.0, 10.0, 10.0])
        tmax = np.array([25.0, 25.0, 25.0])
        srad = np.array([20.0, 20.0, 20.0])
        irr_flag = np.array([False, False, False])

        result = step_day(state, props, params, ndvi, etr, prcp, tmin, tmax, srad, irr_flag)

        assert np.all(result["eta"] >= 0)

    def test_eta_bounded_by_etr(self, simple_setup):
        """Actual ET bounded by reference ET."""
        state, props, params = simple_setup
        # Start with no water stress
        state.depl_root = np.array([0.0, 0.0, 0.0])

        ndvi = np.array([0.7, 0.7, 0.7])
        etr = np.array([5.0, 5.0, 5.0])
        prcp = np.array([0.0, 0.0, 0.0])
        tmin = np.array([10.0, 10.0, 10.0])
        tmax = np.array([25.0, 25.0, 25.0])
        srad = np.array([20.0, 20.0, 20.0])
        irr_flag = np.array([False, False, False])

        result = step_day(state, props, params, ndvi, etr, prcp, tmin, tmax, srad, irr_flag)

        # ETa <= Kc_max * ETr (with some margin for numerical issues)
        max_et = props.kc_max * etr
        assert np.all(result["eta"] <= max_et * 1.01)

    def test_snow_partitioning_cold(self, simple_setup):
        """Cold temps produce snow, not rain."""
        state, props, params = simple_setup

        ndvi = np.array([0.3, 0.5, 0.7])
        etr = np.array([1.0, 1.0, 1.0])
        prcp = np.array([10.0, 10.0, 10.0])
        tmin = np.array([-10.0, -10.0, -10.0])
        tmax = np.array([-5.0, -5.0, -5.0])
        srad = np.array([15.0, 15.0, 15.0])
        irr_flag = np.array([False, False, False])

        result = step_day(state, props, params, ndvi, etr, prcp, tmin, tmax, srad, irr_flag)

        # Should have accumulated SWE
        assert np.all(result["swe"] > 0)
        # Rain should be zero or near-zero
        assert np.all(result["rain"] < 1.0)

    def test_irrigation_triggers(self, simple_setup):
        """Irrigation triggers when depletion exceeds RAW."""
        state, props, params = simple_setup
        # Set high depletion to trigger irrigation
        state.depl_root = np.array([100.0, 100.0, 100.0])
        props.irr_status = np.array([True, True, True])

        ndvi = np.array([0.5, 0.5, 0.5])
        etr = np.array([5.0, 5.0, 5.0])
        prcp = np.array([0.0, 0.0, 0.0])
        tmin = np.array([15.0, 15.0, 15.0])
        tmax = np.array([25.0, 25.0, 25.0])
        srad = np.array([20.0, 20.0, 20.0])
        irr_flag = np.array([True, True, True])

        result = step_day(state, props, params, ndvi, etr, prcp, tmin, tmax, srad, irr_flag)

        # Should have irrigation
        assert np.all(result["irr_sim"] > 0)


@pytest.fixture
def sample_swim_input():
    """Create sample SwimInput for integration testing."""
    json_data = {
        "order": ["field_1", "field_2"],
        "props": {
            "field_1": {"awc": 0.15, "ksat": 10.0, "root_depth": 1.0, "irr": 0},
            "field_2": {"awc": 0.18, "ksat": 12.0, "root_depth": 0.8, "irr": 1},
        },
        "irr_data": {
            "field_1": {"2020": {"irr_doys": [], "irrigated": 0}},
            "field_2": {"2020": {"irr_doys": [95, 105, 115], "irrigated": 1}},
        },
        "gwsub_data": {},
        "kc_max": {"field_1": 1.15, "field_2": 1.20},
        "ke_max": {},
        "time_series": {},
        "missing": [],
    }

    # Create 30 days of data (April 2020)
    start = datetime(2020, 4, 1)
    for day in range(30):
        date = datetime(2020, 4, 1 + day)
        date_str = date.strftime("%Y-%m-%d")
        json_data["time_series"][date_str] = {
            "doy": 92 + day,
            "eto": [3.0 + day * 0.05, 3.2 + day * 0.05],
            "prcp": [0.0, 0.0] if day % 5 != 0 else [8.0, 10.0],
            "tmin": [8.0 + day * 0.1, 9.0 + day * 0.1],
            "tmax": [22.0 + day * 0.15, 23.0 + day * 0.15],
            "srad": [18.0 + day * 0.1, 19.0 + day * 0.1],
            "swe": [0.0, 0.0],
            "ndvi_irr": [0.25 + day * 0.015, 0.30 + day * 0.015],
        }

    return json_data


@pytest.mark.skip(
    reason="Uses deprecated JSON path API. build_swim_input() now requires "
    "a SwimContainer. See tests/parity/test_process_parity.py for the new pattern."
)
class TestRunDailyLoop:
    """Tests for run_daily_loop integration."""

    def test_run_daily_loop_executes(self, sample_swim_input):
        """run_daily_loop completes without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_swim_input, f)

            with build_swim_input(json_path, h5_path) as swim_input:
                output, final_state = run_daily_loop(swim_input)

                assert output.n_days == 30
                assert output.n_fields == 2
                assert final_state.n_fields == 2

    def test_output_shapes(self, sample_swim_input):
        """Output arrays have correct shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_swim_input, f)

            with build_swim_input(json_path, h5_path) as swim_input:
                output, _ = run_daily_loop(swim_input)

                assert output.eta.shape == (30, 2)
                assert output.kcb.shape == (30, 2)
                assert output.swe.shape == (30, 2)

    def test_eta_non_negative(self, sample_swim_input):
        """ET is non-negative throughout simulation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_swim_input, f)

            with build_swim_input(json_path, h5_path) as swim_input:
                output, _ = run_daily_loop(swim_input)

                assert np.all(output.eta >= 0)

    def test_etf_bounded(self, sample_swim_input):
        """ETf (ET fraction) is bounded [0, 1.5]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_swim_input, f)

            with build_swim_input(json_path, h5_path) as swim_input:
                output, _ = run_daily_loop(swim_input)

                assert np.all(output.etf >= 0)
                assert np.all(output.etf <= 1.5)

    def test_depletion_bounded(self, sample_swim_input):
        """Depletion stays bounded by TAW."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_swim_input, f)

            with build_swim_input(json_path, h5_path) as swim_input:
                output, _ = run_daily_loop(swim_input)

                # Depletion should not go negative
                assert np.all(output.depl_root >= -1e-6)

    def test_custom_parameters(self, sample_swim_input):
        """Custom parameters affect results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_swim_input, f)

            with build_swim_input(json_path, h5_path) as swim_input:
                # Run with default params
                output1, _ = run_daily_loop(swim_input)

                # Run with modified params (higher kc_max)
                params = swim_input.parameters.copy()
                params.kc_max = params.kc_max * 1.2

                output2, _ = run_daily_loop(swim_input, parameters=params)

                # Higher kc_max should produce higher ET on average
                mean_et1 = np.mean(output1.eta)
                mean_et2 = np.mean(output2.eta)

                assert mean_et2 > mean_et1
