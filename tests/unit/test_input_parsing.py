"""Unit tests for process package HDF5 input container.

Tests verify:
1. Building HDF5 from JSON works correctly
2. Loading from HDF5 restores all data
3. Time series access works
4. Multiplier application works

NOTE: These tests use the deprecated JSON-to-HDF5 API. The new API requires
a SwimContainer. See tests/parity/test_process_parity.py for examples using
the new container-based workflow with scripts/convert_legacy_input.py.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from swimrs.process.input import SwimInput, build_swim_input

# Mark tests that use deprecated JSON path API as skipped
pytestmark = pytest.mark.skip(
    reason="These tests use deprecated JSON path API. build_swim_input() now requires "
    "a SwimContainer. See tests/parity/test_process_parity.py for the new pattern."
)


@pytest.fixture
def sample_json_data():
    """Create sample prepped_input.json data structure."""
    return {
        "order": ["field_1", "field_2"],
        "props": {
            "field_1": {
                "awc": 0.15,  # Will be converted to mm/m (150)
                "ksat": 10.0,
                "root_depth": 1.0,
                "irr": 0,
            },
            "field_2": {
                "awc": 0.20,
                "ksat": 15.0,
                "root_depth": 0.8,
                "irr": 1,
            },
        },
        "irr_data": {
            "field_1": {
                "2020": {"irr_doys": [], "irrigated": 0},
            },
            "field_2": {
                "2020": {"irr_doys": [100, 110, 120], "irrigated": 1},
            },
        },
        "gwsub_data": {
            "field_2": True,
        },
        "kc_max": {
            "field_1": 1.15,
            "field_2": 1.20,
        },
        "ke_max": {},
        "time_series": {},
        "missing": [],
    }


@pytest.fixture
def sample_json_with_timeseries(sample_json_data):
    """Add time series data to sample JSON."""
    # Create 10 days of data
    start = datetime(2020, 4, 1)
    for day in range(10):
        date = start.replace(day=start.day + day)
        date_str = date.strftime("%Y-%m-%d")
        sample_json_data["time_series"][date_str] = {
            "doy": day + 92,  # April starts around DOY 92
            "eto": [3.0 + day * 0.1, 3.5 + day * 0.1],  # 2 fields
            "prcp": [0.0, 0.0] if day % 3 != 0 else [5.0, 8.0],
            "tmin": [5.0, 6.0],
            "tmax": [20.0, 22.0],
            "srad": [20.0, 21.0],
            "swe": [0.0, 0.0],
            # Split NDVI: ndvi_irr for irrigated fields, ndvi_inv_irr for non-irrigated
            "ndvi_irr": [0.3 + day * 0.02, 0.4 + day * 0.02],
            "ndvi_inv_irr": [0.3 + day * 0.02, 0.4 + day * 0.02],
        }
    return sample_json_data


class TestBuildSwimInput:
    """Tests for build_swim_input function."""

    def test_build_creates_h5_file(self, sample_json_with_timeseries):
        """build_swim_input creates an HDF5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_json_with_timeseries, f)

            swim_input = build_swim_input(json_path, h5_path)

            assert h5_path.exists()
            swim_input.close()

    def test_build_stores_config(self, sample_json_with_timeseries):
        """Config metadata is stored correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_json_with_timeseries, f)

            swim_input = build_swim_input(
                json_path, h5_path,
                runoff_process="cn",
                refet_type="eto",
            )

            assert swim_input.runoff_process == "cn"
            assert swim_input.refet_type == "eto"
            assert swim_input.n_fields == 2
            assert swim_input.fids == ["field_1", "field_2"]
            swim_input.close()

    def test_build_stores_properties(self, sample_json_with_timeseries):
        """Field properties are stored correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_json_with_timeseries, f)

            swim_input = build_swim_input(json_path, h5_path)

            props = swim_input.properties
            # AWC is converted from fraction to mm/m
            assert_array_almost_equal(props.awc, [150.0, 200.0])
            assert_array_almost_equal(props.ksat, [10.0, 15.0])
            assert_array_almost_equal(props.zr_max, [1.0, 0.8])
            assert_array_equal(props.irr_status, [False, True])
            assert_array_equal(props.gw_status, [False, True])
            swim_input.close()

    def test_build_stores_parameters(self, sample_json_with_timeseries):
        """Calibration parameters are stored correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_json_with_timeseries, f)

            swim_input = build_swim_input(json_path, h5_path)

            params = swim_input.parameters
            # kc_max uses FAO-56 default (1.0), matching legacy model tracker.py:77
            assert_array_almost_equal(params.kc_max, [1.0, 1.0])
            swim_input.close()


class TestSwimInputTimeSeries:
    """Tests for time series access."""

    def test_get_time_series_full(self, sample_json_with_timeseries):
        """Get full time series array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_json_with_timeseries, f)

            with build_swim_input(json_path, h5_path) as swim_input:
                ndvi = swim_input.get_time_series("ndvi")

                assert ndvi.shape == (10, 2)
                # Check first day values
                assert_array_almost_equal(ndvi[0, :], [0.3, 0.4])

    def test_get_time_series_single_day(self, sample_json_with_timeseries):
        """Get time series for a single day."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_json_with_timeseries, f)

            with build_swim_input(json_path, h5_path) as swim_input:
                ndvi_day0 = swim_input.get_time_series("ndvi", day_idx=0)

                assert ndvi_day0.shape == (2,)
                assert_array_almost_equal(ndvi_day0, [0.3, 0.4])

    def test_get_irr_flag(self, sample_json_with_timeseries):
        """Get irrigation flag array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_json_with_timeseries, f)

            with build_swim_input(json_path, h5_path) as swim_input:
                irr_flag = swim_input.get_irr_flag()

                assert irr_flag.shape == (10, 2)
                # field_1 has no irrigation
                assert not irr_flag[:, 0].any()
                # field_2 has irrigation on DOY 100, 110, 120
                # which may or may not be in our 10-day window


class TestSwimInputSpinup:
    """Tests for spinup state."""

    def test_default_spinup(self, sample_json_with_timeseries):
        """Default spinup uses zeros/ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_json_with_timeseries, f)

            with build_swim_input(json_path, h5_path) as swim_input:
                state = swim_input.spinup_state

                assert state.n_fields == 2
                assert_array_equal(state.depl_root, [0.0, 0.0])
                assert_array_equal(state.swe, [0.0, 0.0])
                assert_array_equal(state.kr, [1.0, 1.0])
                assert_array_equal(state.ks, [1.0, 1.0])

    def test_custom_spinup(self, sample_json_with_timeseries):
        """Custom spinup values are stored correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_json_with_timeseries, f)

            spinup = {
                "depl_root": np.array([50.0, 40.0]),
                "swe": np.array([10.0, 0.0]),
                "kr": np.array([0.8, 0.9]),
                "ks": np.array([0.7, 0.85]),
                "zr": np.array([0.5, 0.6]),
            }

            with build_swim_input(json_path, h5_path, spinup_state=spinup) as swim_input:
                state = swim_input.spinup_state

                assert_array_almost_equal(state.depl_root, [50.0, 40.0])
                assert_array_almost_equal(state.swe, [10.0, 0.0])
                assert_array_almost_equal(state.kr, [0.8, 0.9])


class TestSwimInputMultipliers:
    """Tests for PEST++ multiplier application."""

    def test_apply_multipliers_no_dir(self, sample_json_with_timeseries):
        """With no mult dir, returns copy of base params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_json_with_timeseries, f)

            with build_swim_input(json_path, h5_path) as swim_input:
                # Non-existent directory
                params = swim_input.apply_multipliers(Path(tmpdir) / "nonexistent")

                # Should return copy of base
                assert_array_almost_equal(params.kc_max, swim_input.parameters.kc_max)

    def test_apply_multipliers(self, sample_json_with_timeseries):
        """Multipliers are applied correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"
            mult_dir = Path(tmpdir) / "mult"
            mult_dir.mkdir()

            with open(json_path, "w") as f:
                json.dump(sample_json_with_timeseries, f)

            # Create a multiplier file
            mult_file = mult_dir / "p_ndvi_k_field_1_0_constant.csv"
            with open(mult_file, "w") as f:
                f.write(",0,sidx,idx_strs,pargp1,parval1_1,1\n")
                f.write("0,,(),idx0:nan,ndvi_k,1.5,7.0\n")

            with build_swim_input(json_path, h5_path) as swim_input:
                params = swim_input.apply_multipliers(mult_dir)

                # field_1 should have ndvi_k multiplied by 1.5
                # Default is 7.0, so result should be 10.5
                assert_array_almost_equal(params.ndvi_k, [10.5, 7.0])


class TestSwimInputReload:
    """Tests for loading from existing HDF5."""

    def test_reload_from_h5(self, sample_json_with_timeseries):
        """SwimInput can be loaded from existing HDF5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "input.json"
            h5_path = Path(tmpdir) / "swim_input.h5"

            with open(json_path, "w") as f:
                json.dump(sample_json_with_timeseries, f)

            # Build the HDF5
            swim_input = build_swim_input(json_path, h5_path)
            swim_input.close()

            # Reload from HDF5
            reloaded = SwimInput(h5_path=h5_path)

            assert reloaded.n_fields == 2
            assert reloaded.fids == ["field_1", "field_2"]
            assert_array_almost_equal(reloaded.properties.awc, [150.0, 200.0])
            reloaded.close()
