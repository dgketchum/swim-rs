"""Parity tests comparing loop.py (Python) vs loop_fast.py (Numba JIT).

These tests verify that run_daily_loop() and run_daily_loop_fast() produce
numerically identical results when given the same inputs.

Note: loop_fast.py does not compute irrigation tracking fields (et_irr,
dperc_irr, irr_frac_root, irr_frac_l3). Only the 14 common output fields
are compared.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Mark entire module as parity
pytestmark = [pytest.mark.parity, pytest.mark.slow]

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from swimrs.container import SwimContainer
from swimrs.process.input import build_swim_input
from swimrs.process.loop import run_daily_loop
from swimrs.process.loop_fast import run_daily_loop_fast

# Import the converter
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from convert_legacy_input import convert_to_container

# Path to multi-station fixture
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "multi_station"
GOLDEN_DIR = FIXTURE_DIR / "golden"
GOLDEN_INPUT = GOLDEN_DIR / "prepped_input.json"
GOLDEN_SPINUP = GOLDEN_DIR / "spinup.json"
CALIBRATED_PARAMS = GOLDEN_DIR / "calibrated_params.json"
SHAPEFILE = FIXTURE_DIR / "data" / "gis" / "multi_station.shp"

# Output fields that both implementations produce
COMMON_OUTPUT_FIELDS = [
    "eta",
    "etf",
    "kcb",
    "ke",
    "ks",
    "kr",
    "runoff",
    "rain",
    "melt",
    "swe",
    "depl_root",
    "dperc",
    "irr_sim",
    "gw_sim",
]

# Fields only produced by loop.py (not loop_fast.py)
LOOP_ONLY_FIELDS = [
    "et_irr",
    "dperc_irr",
    "irr_frac_root",
    "irr_frac_l3",
]

# Final state fields to compare
STATE_FIELDS = [
    "depl_root",
    "depl_ze",
    "swe",
    "albedo",
    "kr",
    "ks",
    "zr",
    "daw3",
    "taw3",
]


def fixture_available():
    """Check if multi-station fixture is available."""
    return GOLDEN_INPUT.exists() and SHAPEFILE.exists()


@pytest.fixture(scope="module")
def converted_container(tmp_path_factory):
    """Convert legacy JSON to SwimContainer once per module."""
    if not fixture_available():
        pytest.skip("Multi-station fixture not available")

    # Create container in a temp directory that persists for the module
    tmpdir = tmp_path_factory.mktemp("loop_parity")
    container_path = tmpdir / "converted.swim"

    convert_to_container(
        json_path=GOLDEN_INPUT,
        shapefile_path=SHAPEFILE,
        output_path=container_path,
        uid_column="site_id",
        met_source="gridmet",
        overwrite=True,
    )

    container = SwimContainer.open(str(container_path), mode="r")
    yield container
    container.close()


@pytest.fixture(scope="module")
def swim_input_and_outputs(converted_container):
    """Build SwimInput and run both loop implementations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "swim_input.h5"

        # Build HDF5 from converted container
        swim_input = build_swim_input(
            converted_container,
            h5_path,
            spinup_json_path=GOLDEN_SPINUP if GOLDEN_SPINUP.exists() else None,
            calibrated_params_path=CALIBRATED_PARAMS if CALIBRATED_PARAMS.exists() else None,
            start_date="2010-01-01",
            end_date="2011-12-31",
            etf_model="ssebop",
        )

        try:
            # Run Python implementation
            output_py, state_py = run_daily_loop(swim_input)

            # Run Numba JIT implementation
            output_fast, state_fast = run_daily_loop_fast(swim_input)

            # Return copies of the data since swim_input will be closed
            result = {
                "n_days": swim_input.n_days,
                "n_fields": swim_input.n_fields,
                "output_py": {
                    field: getattr(output_py, field).copy()
                    for field in COMMON_OUTPUT_FIELDS + LOOP_ONLY_FIELDS
                },
                "output_fast": {
                    field: getattr(output_fast, field).copy()
                    if getattr(output_fast, field) is not None
                    else None
                    for field in COMMON_OUTPUT_FIELDS + LOOP_ONLY_FIELDS
                },
                "state_py": {
                    field: getattr(state_py, field).copy()
                    if getattr(state_py, field) is not None
                    else None
                    for field in STATE_FIELDS
                },
                "state_fast": {
                    field: getattr(state_fast, field).copy()
                    if getattr(state_fast, field) is not None
                    else None
                    for field in STATE_FIELDS
                },
            }
        finally:
            swim_input.close()

        return result


class TestLoopOutputParity:
    """Tests comparing daily output arrays between implementations."""

    def test_output_shapes_match(self, swim_input_and_outputs):
        """Both implementations produce same output shapes."""
        data = swim_input_and_outputs
        n_days = data["n_days"]
        n_fields = data["n_fields"]

        for field in COMMON_OUTPUT_FIELDS:
            py_arr = data["output_py"][field]
            fast_arr = data["output_fast"][field]

            assert py_arr.shape == (n_days, n_fields), f"{field} shape mismatch (py)"
            assert fast_arr.shape == (n_days, n_fields), f"{field} shape mismatch (fast)"

    @pytest.mark.parametrize("field", COMMON_OUTPUT_FIELDS)
    def test_output_field_parity(self, swim_input_and_outputs, field):
        """Output field values match between implementations."""
        data = swim_input_and_outputs
        py_arr = data["output_py"][field]
        fast_arr = data["output_fast"][field]

        # Use tight tolerance - implementations should be numerically identical
        # Allow rtol=1e-10 for minor floating-point differences between
        # Python and Numba JIT compilation
        assert_allclose(
            fast_arr,
            py_arr,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Output field '{field}' differs between loop.py and loop_fast.py",
        )

    def test_eta_summary_statistics(self, swim_input_and_outputs):
        """ET summary statistics match closely."""
        data = swim_input_and_outputs
        eta_py = data["output_py"]["eta"]
        eta_fast = data["output_fast"]["eta"]

        # Compare means
        assert_allclose(
            np.mean(eta_fast),
            np.mean(eta_py),
            rtol=1e-10,
            err_msg="Mean ET differs",
        )

        # Compare totals per field
        assert_allclose(
            np.sum(eta_fast, axis=0),
            np.sum(eta_py, axis=0),
            rtol=1e-10,
            err_msg="Total ET per field differs",
        )

    def test_irrigation_totals_match(self, swim_input_and_outputs):
        """Total irrigation matches between implementations."""
        data = swim_input_and_outputs
        irr_py = data["output_py"]["irr_sim"]
        irr_fast = data["output_fast"]["irr_sim"]

        assert_allclose(
            np.sum(irr_fast, axis=0),
            np.sum(irr_py, axis=0),
            rtol=1e-10,
            err_msg="Total irrigation per field differs",
        )

    def test_water_balance_components_match(self, swim_input_and_outputs):
        """Water balance components match between implementations."""
        data = swim_input_and_outputs

        for component in ["rain", "melt", "runoff", "dperc"]:
            py_total = np.sum(data["output_py"][component], axis=0)
            fast_total = np.sum(data["output_fast"][component], axis=0)

            assert_allclose(
                fast_total,
                py_total,
                rtol=1e-10,
                err_msg=f"Total {component} per field differs",
            )


class TestLoopStateParity:
    """Tests comparing final state between implementations."""

    @pytest.mark.parametrize("field", STATE_FIELDS)
    def test_final_state_field_parity(self, swim_input_and_outputs, field):
        """Final state field values match between implementations."""
        data = swim_input_and_outputs
        py_state = data["state_py"][field]
        fast_state = data["state_fast"][field]

        if py_state is None and fast_state is None:
            return  # Both None is OK

        if py_state is None or fast_state is None:
            pytest.fail(f"State field '{field}' is None in one implementation but not the other")

        assert_allclose(
            fast_state,
            py_state,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Final state '{field}' differs between loop.py and loop_fast.py",
        )


class TestLoopOnlyFields:
    """Tests documenting fields only available in loop.py."""

    @pytest.mark.parametrize("field", LOOP_ONLY_FIELDS)
    def test_loop_only_field_not_in_fast(self, swim_input_and_outputs, field):
        """Irrigation tracking fields are not computed by loop_fast.py."""
        data = swim_input_and_outputs
        py_arr = data["output_py"][field]
        fast_arr = data["output_fast"][field]

        # loop.py should have these fields populated
        assert py_arr is not None, f"{field} should be populated in loop.py"

        # loop_fast.py leaves these as None (default initialization)
        # or zeros if DailyOutput.__post_init__ initializes them
        if fast_arr is not None:
            # If initialized to zeros, that's the expected behavior
            assert np.allclose(fast_arr, 0), (
                f"{field} in loop_fast.py should be zeros (not computed)"
            )


class TestNumericalStability:
    """Tests for numerical stability and edge cases."""

    def test_no_nans_in_output(self, swim_input_and_outputs):
        """Neither implementation produces NaN values."""
        data = swim_input_and_outputs

        for field in COMMON_OUTPUT_FIELDS:
            py_arr = data["output_py"][field]
            fast_arr = data["output_fast"][field]

            assert not np.any(np.isnan(py_arr)), f"NaN in loop.py output: {field}"
            assert not np.any(np.isnan(fast_arr)), f"NaN in loop_fast.py output: {field}"

    def test_no_infs_in_output(self, swim_input_and_outputs):
        """Neither implementation produces infinite values."""
        data = swim_input_and_outputs

        for field in COMMON_OUTPUT_FIELDS:
            py_arr = data["output_py"][field]
            fast_arr = data["output_fast"][field]

            assert not np.any(np.isinf(py_arr)), f"Inf in loop.py output: {field}"
            assert not np.any(np.isinf(fast_arr)), f"Inf in loop_fast.py output: {field}"

    def test_physical_bounds_consistent(self, swim_input_and_outputs):
        """Physical bounds are consistent between implementations."""
        data = swim_input_and_outputs

        # ET should be non-negative in both
        assert np.all(data["output_py"]["eta"] >= 0)
        assert np.all(data["output_fast"]["eta"] >= 0)

        # ETf should be bounded [0, ~1.5]
        assert np.all(data["output_py"]["etf"] >= 0)
        assert np.all(data["output_fast"]["etf"] >= 0)
        assert np.all(data["output_py"]["etf"] <= 2.0)
        assert np.all(data["output_fast"]["etf"] <= 2.0)

        # Coefficients bounded [0, 1]
        for coef in ["ks", "kr"]:
            assert np.all(data["output_py"][coef] >= 0)
            assert np.all(data["output_fast"][coef] >= 0)
            assert np.all(data["output_py"][coef] <= 1.0)
            assert np.all(data["output_fast"][coef] <= 1.0)

        # SWE non-negative
        assert np.all(data["output_py"]["swe"] >= 0)
        assert np.all(data["output_fast"]["swe"] >= 0)
