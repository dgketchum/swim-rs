"""Tests for swimrs.calibrate.pest_cleanup module.

Tests cover:
- is_successful(): 5 criteria tested independently
- get_summary(): expected keys, phi arithmetic, iterations
- _get_recommendations(): issue pattern -> remedy mapping
"""

import pandas as pd
import pytest

from swimrs.calibrate.pest_cleanup import PestResults


@pytest.fixture
def pest_dir(tmp_path):
    """Create a minimal PEST directory structure."""
    pest = tmp_path / "pest"
    master = pest / "master"
    master.mkdir(parents=True)
    return pest, master


def _write_rec(master, project, content):
    """Write a .rec file in master."""
    (master / f"{project}.rec").write_text(content)


def _write_phi(master, project, phi_values):
    """Write a phi.meas.csv with given mean values."""
    df = pd.DataFrame({"mean": phi_values, "std": [0.1] * len(phi_values)})
    df.to_csv(master / f"{project}.phi.meas.csv", index=False)


def _write_par_files(master, project, n_iters):
    """Create par.csv files for iterations 0..n_iters."""
    for i in range(n_iters + 1):
        df = pd.DataFrame({"param_a": [1.0 + i * 0.1], "param_b": [2.0 - i * 0.05]})
        df.to_csv(master / f"{project}.{i}.par.csv")


def _write_pst(pest, project, noptmax):
    """Write a minimal .pst file."""
    (pest / f"{project}.pst").write_text(f"NOPTMAX {noptmax}\n")


class TestIsSuccessful:
    """Tests for PestResults.is_successful()."""

    def test_missing_rec_file(self, pest_dir):
        """Missing rec file returns failure."""
        pest, master = pest_dir
        results = PestResults(str(pest), "test_proj")
        success, issues = results.is_successful()
        assert success is False
        assert any("record file not found" in i.lower() for i in issues)

    @pytest.mark.parametrize(
        "pattern",
        [
            "FATAL ERROR in line 42",
            "Process terminated abnormally",
            "Traceback (most recent call last)",
            "PEST++ run failed",
        ],
    )
    def test_fatal_error_patterns(self, pest_dir, pattern):
        """Each fatal error pattern is detected."""
        pest, master = pest_dir
        _write_rec(master, "proj", f"Some preamble\n{pattern}\nMore text")
        _write_pst(pest, "proj", 5)
        _write_par_files(master, "proj", 5)
        _write_phi(master, "proj", [100.0, 50.0])

        results = PestResults(str(pest), "proj")
        success, issues = results.is_successful()
        assert any("fatal error" in i.lower() for i in issues)

    def test_missing_final_par_file(self, pest_dir):
        """Missing final par file triggers issue."""
        pest, master = pest_dir
        _write_rec(master, "proj", "analysis complete")
        _write_pst(pest, "proj", 5)
        # Only write par files 0..3, missing 5
        _write_par_files(master, "proj", 3)
        _write_phi(master, "proj", [100.0, 50.0])

        results = PestResults(str(pest), "proj")
        success, issues = results.is_successful()
        assert any("parameter file not found" in i.lower() for i in issues)

    def test_no_phi_improvement(self, pest_dir):
        """No phi improvement (final >= initial) triggers issue."""
        pest, master = pest_dir
        _write_rec(master, "proj", "analysis complete")
        _write_pst(pest, "proj", 3)
        _write_par_files(master, "proj", 3)
        _write_phi(master, "proj", [50.0, 60.0, 70.0])  # getting worse

        results = PestResults(str(pest), "proj")
        success, issues = results.is_successful()
        assert any("no phi improvement" in i.lower() for i in issues)

    def test_missing_completion_message(self, pest_dir):
        """Missing completion message triggers issue."""
        pest, master = pest_dir
        _write_rec(master, "proj", "Some output but no completion keywords")
        _write_pst(pest, "proj", 3)
        _write_par_files(master, "proj", 3)
        _write_phi(master, "proj", [100.0, 50.0])

        results = PestResults(str(pest), "proj")
        success, issues = results.is_successful()
        assert any("completion message" in i.lower() for i in issues)

    def test_all_pass_case(self, pest_dir):
        """All criteria pass returns success."""
        pest, master = pest_dir
        _write_rec(master, "proj", "Everything worked\nanalysis complete\n")
        _write_pst(pest, "proj", 3)
        _write_par_files(master, "proj", 3)
        _write_phi(master, "proj", [100.0, 80.0, 60.0, 40.0])

        results = PestResults(str(pest), "proj")
        success, issues = results.is_successful()
        assert success is True
        assert issues == []


class TestGetSummary:
    """Tests for PestResults.get_summary()."""

    def test_expected_keys(self, pest_dir):
        """Summary contains expected keys."""
        pest, master = pest_dir
        _write_rec(master, "proj", "analysis complete")
        _write_pst(pest, "proj", 3)
        _write_par_files(master, "proj", 3)
        _write_phi(master, "proj", [100.0, 80.0, 60.0, 40.0])

        results = PestResults(str(pest), "proj")
        summary = results.get_summary()

        assert "project" in summary
        assert "status" in summary
        assert "issues" in summary
        assert "noptmax" in summary
        assert "iterations_completed" in summary

    def test_phi_reduction_pct_arithmetic(self, pest_dir):
        """phi_reduction_pct = (initial - final) / initial * 100."""
        pest, master = pest_dir
        _write_rec(master, "proj", "analysis complete")
        _write_pst(pest, "proj", 3)
        _write_par_files(master, "proj", 3)
        _write_phi(master, "proj", [100.0, 80.0, 60.0, 40.0])

        results = PestResults(str(pest), "proj")
        summary = results.get_summary()

        expected_reduction = (100.0 - 40.0) / 100.0 * 100
        assert abs(summary["phi_reduction_pct"] - expected_reduction) < 0.1

    def test_iterations_completed(self, pest_dir):
        """iterations_completed = len(par_files) - 1."""
        pest, master = pest_dir
        _write_rec(master, "proj", "analysis complete")
        _write_pst(pest, "proj", 5)
        _write_par_files(master, "proj", 5)
        _write_phi(master, "proj", [100.0, 50.0])

        results = PestResults(str(pest), "proj")
        summary = results.get_summary()

        # 6 par files (0..5) -> 5 iterations completed
        assert summary["iterations_completed"] == 5


class TestGetRecommendations:
    """Tests for PestResults._get_recommendations()."""

    def test_record_file_not_found(self, pest_dir):
        """Record file not found maps to PATH recommendation."""
        pest, _ = pest_dir
        results = PestResults(str(pest), "proj")
        recs = results._get_recommendations(["Record file not found: /path/to/rec"])
        assert any("PATH" in r or "path" in r.lower() for r in recs)

    def test_fatal_error_maps_to_log_check(self, pest_dir):
        """Fatal error maps to check logs recommendation."""
        pest, _ = pest_dir
        results = PestResults(str(pest), "proj")
        recs = results._get_recommendations(
            ["Fatal error detected: 'FATAL ERROR' found in record file"]
        )
        assert any("panther_master" in r or "worker" in r.lower() for r in recs)

    def test_no_phi_improvement_maps_to_realization_advice(self, pest_dir):
        """No phi improvement maps to realization/bounds recommendation."""
        pest, _ = pest_dir
        results = PestResults(str(pest), "proj")
        recs = results._get_recommendations(["No phi improvement: 100.0 -> 120.0"])
        assert any("realization" in r.lower() or "bounds" in r.lower() for r in recs)

    def test_empty_issues_default_recommendations(self, pest_dir):
        """Empty issues list gives default recommendations."""
        pest, _ = pest_dir
        results = PestResults(str(pest), "proj")
        recs = results._get_recommendations([])
        assert len(recs) > 0

    def test_traceback_maps_to_forward_model(self, pest_dir):
        """Traceback pattern maps to Python error recommendation."""
        pest, _ = pest_dir
        results = PestResults(str(pest), "proj")
        # Use an issue that only contains "traceback" (not "fatal error")
        recs = results._get_recommendations(["Traceback found in output"])
        assert any("python" in r.lower() or "forward" in r.lower() for r in recs)
