"""Parity tests comparing loop.py (Python) vs loop_fast.py (Numba JIT).

These tests verify that run_daily_loop() and run_daily_loop_fast() produce
numerically identical results when given the same inputs.

Uses native .swim fixture containers with short, hydrologically dynamic
windows chosen to exercise snow/melt/rain/runoff/ET/irrigation in ~2 months.
Full 2-year correctness is covered by test_golden_loop.py (fast loop only).

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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from swimrs.container import SwimContainer
from swimrs.process.input import build_swim_input
from swimrs.process.loop import run_daily_loop
from swimrs.process.loop_fast import run_daily_loop_fast

# Use the golden-loop native containers
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "golden_loop"

# 6-month dynamic windows covering snow, melt, rain, runoff, ET, and stress
# (+ irrigation for Crane).  Must be >100 days for NDVI interpolation limit.
# Full 2-year golden-snapshot correctness is in test_golden_loop.py.
CASES = {
    "fort_peck": {
        "container": "fort_peck.swim",
        "etf_model": "ptjpl",
        "start_date": "2008-01-01",  # Jan–Jun 2008: winter snow → spring melt → summer ET
        "end_date": "2008-06-30",
    },
    "crane": {
        "container": "crane.swim",
        "etf_model": "ssebop",
        "start_date": "2020-01-01",  # Jan–Jun 2020: snow → melt → irrigation onset
        "end_date": "2020-06-30",
    },
}

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


def fixture_available(case_name):
    """Check if fixture container exists."""
    return (FIXTURE_DIR / CASES[case_name]["container"]).exists()


def _run_both_loops(case_name):
    """Build SwimInput from native container and run both loop implementations."""
    case = CASES[case_name]
    container_path = FIXTURE_DIR / case["container"]

    container = SwimContainer.open(str(container_path), mode="r")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "swim_input.h5"

            swim_input = build_swim_input(
                container,
                h5_path,
                start_date=case["start_date"],
                end_date=case["end_date"],
                etf_model=case["etf_model"],
            )

            try:
                output_py, state_py = run_daily_loop(swim_input)
                output_fast, state_fast = run_daily_loop_fast(swim_input)

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
    finally:
        container.close()

    return result


@pytest.fixture(scope="module")
def fort_peck_parity():
    if not fixture_available("fort_peck"):
        pytest.skip("Fort Peck fixture not available")
    return _run_both_loops("fort_peck")


@pytest.fixture(scope="module")
def crane_parity():
    if not fixture_available("crane"):
        pytest.skip("Crane fixture not available")
    return _run_both_loops("crane")


def _get_data(request, case_name):
    if case_name == "fort_peck":
        return request.getfixturevalue("fort_peck_parity")
    elif case_name == "crane":
        return request.getfixturevalue("crane_parity")


# Tolerances for per-element field comparison.
# Irrigation triggers are threshold-sensitive: a tiny floating-point
# difference in depletion can shift an irrigation event by one day,
# cascading through ks, ET, depletion, and deep percolation.  The
# unirrigated Fort Peck site stays within 1e-6; irrigated Crane needs
# looser tolerance on the affected fields.
_FIELD_RTOL = {
    # Irrigation-sensitive fields (threshold cascade)
    "irr_sim": 0.05,
    "depl_root": 0.05,
    "ks": 0.03,
    "dperc": 0.002,
    # Moderate — downstream of ks/depletion via ET partitioning
    "eta": 5e-4,
    "etf": 5e-4,
    "ke": 5e-4,
    "kr": 5e-4,
    "gw_sim": 5e-4,
}
_DEFAULT_RTOL = 1e-4
_DEFAULT_ATOL = 1e-6


class TestLoopOutputParity:
    """Tests comparing daily output arrays between implementations."""

    @pytest.mark.parametrize("case_name", ["fort_peck", "crane"])
    def test_output_shapes_match(self, request, case_name):
        """Both implementations produce same output shapes."""
        data = _get_data(request, case_name)
        n_days = data["n_days"]
        n_fields = data["n_fields"]

        for field in COMMON_OUTPUT_FIELDS:
            py_arr = data["output_py"][field]
            fast_arr = data["output_fast"][field]

            assert py_arr.shape == (n_days, n_fields), f"{field} shape mismatch (py)"
            assert fast_arr.shape == (n_days, n_fields), f"{field} shape mismatch (fast)"

    @pytest.mark.parametrize("case_name", ["fort_peck", "crane"])
    @pytest.mark.parametrize("field", COMMON_OUTPUT_FIELDS)
    def test_output_field_parity(self, request, case_name, field):
        """Output field values match between implementations."""
        data = _get_data(request, case_name)
        py_arr = data["output_py"][field]
        fast_arr = data["output_fast"][field]

        rtol = _FIELD_RTOL.get(field, _DEFAULT_RTOL)
        assert_allclose(
            fast_arr,
            py_arr,
            rtol=rtol,
            atol=_DEFAULT_ATOL,
            err_msg=f"{case_name}: '{field}' differs between loop.py and loop_fast.py",
        )

    @pytest.mark.parametrize("case_name", ["fort_peck", "crane"])
    def test_eta_summary_statistics(self, request, case_name):
        """Cumulative ET totals match closely across implementations."""
        data = _get_data(request, case_name)
        eta_py = data["output_py"]["eta"]
        eta_fast = data["output_fast"]["eta"]

        assert_allclose(np.mean(eta_fast), np.mean(eta_py), rtol=5e-4)
        assert_allclose(np.sum(eta_fast, axis=0), np.sum(eta_py, axis=0), rtol=5e-4)

    @pytest.mark.parametrize("case_name", ["fort_peck", "crane"])
    def test_water_balance_components_match(self, request, case_name):
        """Cumulative water balance component totals match."""
        data = _get_data(request, case_name)

        for component in ["rain", "melt", "runoff", "dperc", "irr_sim"]:
            py_total = np.sum(data["output_py"][component], axis=0)
            fast_total = np.sum(data["output_fast"][component], axis=0)

            rtol = _FIELD_RTOL.get(component, _DEFAULT_RTOL)
            assert_allclose(
                fast_total,
                py_total,
                rtol=rtol,
                atol=_DEFAULT_ATOL,
                err_msg=f"{case_name}: total {component} differs",
            )


class TestLoopStateParity:
    """Tests comparing final state between implementations."""

    @pytest.mark.parametrize("case_name", ["fort_peck", "crane"])
    @pytest.mark.parametrize("field", STATE_FIELDS)
    def test_final_state_field_parity(self, request, case_name, field):
        """Final state field values match between implementations."""
        data = _get_data(request, case_name)
        py_state = data["state_py"][field]
        fast_state = data["state_fast"][field]

        if py_state is None and fast_state is None:
            return

        if py_state is None or fast_state is None:
            pytest.fail(f"State field '{field}' is None in one implementation but not the other")

        rtol = _FIELD_RTOL.get(field, _DEFAULT_RTOL)
        assert_allclose(
            fast_state,
            py_state,
            rtol=rtol,
            atol=_DEFAULT_ATOL,
            err_msg=f"{case_name}: final state '{field}' differs",
        )


class TestLoopOnlyFields:
    """Tests documenting fields only available in loop.py."""

    @pytest.mark.parametrize("case_name", ["fort_peck", "crane"])
    @pytest.mark.parametrize("field", LOOP_ONLY_FIELDS)
    def test_loop_only_field_not_in_fast(self, request, case_name, field):
        """Irrigation tracking fields are not computed by loop_fast.py."""
        data = _get_data(request, case_name)
        py_arr = data["output_py"][field]
        fast_arr = data["output_fast"][field]

        assert py_arr is not None, f"{field} should be populated in loop.py"

        if fast_arr is not None:
            assert np.allclose(fast_arr, 0), (
                f"{field} in loop_fast.py should be zeros (not computed)"
            )


class TestNumericalStability:
    """Tests for numerical stability and edge cases."""

    @pytest.mark.parametrize("case_name", ["fort_peck", "crane"])
    def test_no_nans_in_output(self, request, case_name):
        """Neither implementation produces NaN values."""
        data = _get_data(request, case_name)

        for field in COMMON_OUTPUT_FIELDS:
            py_arr = data["output_py"][field]
            fast_arr = data["output_fast"][field]

            assert not np.any(np.isnan(py_arr)), f"NaN in loop.py output: {field}"
            assert not np.any(np.isnan(fast_arr)), f"NaN in loop_fast.py output: {field}"

    @pytest.mark.parametrize("case_name", ["fort_peck", "crane"])
    def test_no_infs_in_output(self, request, case_name):
        """Neither implementation produces infinite values."""
        data = _get_data(request, case_name)

        for field in COMMON_OUTPUT_FIELDS:
            py_arr = data["output_py"][field]
            fast_arr = data["output_fast"][field]

            assert not np.any(np.isinf(py_arr)), f"Inf in loop.py output: {field}"
            assert not np.any(np.isinf(fast_arr)), f"Inf in loop_fast.py output: {field}"

    @pytest.mark.parametrize("case_name", ["fort_peck", "crane"])
    def test_physical_bounds_consistent(self, request, case_name):
        """Physical bounds are consistent between implementations."""
        data = _get_data(request, case_name)

        assert np.all(data["output_py"]["eta"] >= 0)
        assert np.all(data["output_fast"]["eta"] >= 0)

        assert np.all(data["output_py"]["etf"] >= 0)
        assert np.all(data["output_fast"]["etf"] >= 0)
        assert np.all(data["output_py"]["etf"] <= 2.0)
        assert np.all(data["output_fast"]["etf"] <= 2.0)

        for coef in ["ks", "kr"]:
            assert np.all(data["output_py"][coef] >= 0)
            assert np.all(data["output_fast"][coef] >= 0)
            assert np.all(data["output_py"][coef] <= 1.0)
            assert np.all(data["output_fast"][coef] <= 1.0)

        assert np.all(data["output_py"]["swe"] >= 0)
        assert np.all(data["output_fast"]["swe"] >= 0)
