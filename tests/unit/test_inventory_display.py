"""Tests for inventory display/formatting classes.

Tests cover:
- Coverage.status_symbol: all 4 DataStatus values
- Coverage.summary_line(): formatting for all statuses
- FieldValidationReport: arithmetic, summary, detailed_report, to_dataframe
- ValidationResult.summary(): READY vs NOT READY, truncation
"""

import pytest

from swimrs.container.inventory import (
    Coverage,
    DataStatus,
    FieldIssue,
    FieldValidationReport,
    ValidationResult,
)


class TestCoverageStatusSymbol:
    """Tests for Coverage.status_symbol property."""

    def test_complete_checkmark(self):
        """COMPLETE status returns checkmark."""
        cov = Coverage("test", DataStatus.COMPLETE, 10, 10, [])
        assert cov.status_symbol == "\u2713"

    def test_partial_half_circle(self):
        """PARTIAL status returns half circle."""
        cov = Coverage("test", DataStatus.PARTIAL, 5, 10, ["a", "b", "c", "d", "e"])
        assert cov.status_symbol == "\u25d0"

    def test_not_present_x_mark(self):
        """NOT_PRESENT status returns x mark."""
        cov = Coverage("test", DataStatus.NOT_PRESENT, 0, 10, list("abcdefghij"))
        assert cov.status_symbol == "\u2717"

    def test_error_exclamation(self):
        """ERROR status returns exclamation."""
        cov = Coverage("test", DataStatus.ERROR, 0, 10, [])
        assert cov.status_symbol == "!"


class TestCoverageSummaryLine:
    """Tests for Coverage.summary_line() method."""

    def test_not_present_includes_not_ingested(self):
        """NOT_PRESENT summary includes 'not ingested'."""
        cov = Coverage("my/path", DataStatus.NOT_PRESENT, 0, 10, list("abcdefghij"))
        line = cov.summary_line()
        assert "not ingested" in line

    def test_partial_with_date_range(self):
        """PARTIAL with date_range includes dates."""
        cov = Coverage(
            "my/path",
            DataStatus.PARTIAL,
            8,
            10,
            ["a", "b"],
            date_range=("2020-01-01", "2020-12-31"),
        )
        line = cov.summary_line()
        assert "2020-01-01" in line
        assert "2020-12-31" in line
        assert "8/10" in line

    def test_complete_omits_percent(self):
        """COMPLETE with 100% coverage omits percent."""
        cov = Coverage("my/path", DataStatus.COMPLETE, 10, 10, [])
        line = cov.summary_line()
        assert "10/10" in line

    def test_percent_complete_arithmetic(self):
        """percent_complete = 100 * present / total."""
        cov = Coverage("test", DataStatus.PARTIAL, 3, 10, list("abcdefg"))
        assert cov.percent_complete == pytest.approx(30.0)

    def test_percent_complete_zero_total(self):
        """percent_complete is 0 when total is 0."""
        cov = Coverage("test", DataStatus.NOT_PRESENT, 0, 0, [])
        assert cov.percent_complete == 0.0


class TestFieldValidationReport:
    """Tests for FieldValidationReport."""

    @pytest.fixture
    def report(self):
        """Create a sample report with issues."""
        issues_by_field = {
            "field_B": [
                FieldIssue("field_B", "low_awc", "AWC too low", value=0.01, threshold=0.05),
            ],
            "field_C": [
                FieldIssue("field_C", "low_awc", "AWC too low", value=0.02, threshold=0.05),
                FieldIssue("field_C", "missing_ksat", "Ksat not available"),
            ],
        }
        return FieldValidationReport(
            total_fields=3,
            valid_fields=["field_A"],
            invalid_fields=["field_B", "field_C"],
            issues_by_field=issues_by_field,
            issues_by_type={
                "low_awc": ["field_B", "field_C"],
                "missing_ksat": ["field_C"],
            },
            thresholds={"min_awc": 0.05},
        )

    def test_valid_count(self, report):
        assert report.valid_count == 1

    def test_invalid_count(self, report):
        assert report.invalid_count == 2

    def test_percent_valid(self, report):
        assert report.percent_valid == pytest.approx(100.0 / 3)

    def test_summary_header_and_totals(self, report):
        """summary() contains header and field counts."""
        s = report.summary()
        assert "FIELD VALIDATION REPORT" in s
        assert "Total fields:   3" in s
        assert "Valid fields:   1" in s
        assert "Invalid fields: 2" in s

    def test_summary_includes_issues_by_type(self, report):
        """summary() lists issues by type."""
        s = report.summary()
        assert "low_awc" in s
        assert "missing_ksat" in s

    def test_detailed_report_truncation(self):
        """detailed_report truncates at max_fields_per_issue."""
        n = 20
        invalid = [f"field_{i}" for i in range(n)]
        issues_by_field = {f: [FieldIssue(f, "bad_data", "problem")] for f in invalid}
        report = FieldValidationReport(
            total_fields=n,
            valid_fields=[],
            invalid_fields=invalid,
            issues_by_field=issues_by_field,
            issues_by_type={"bad_data": invalid},
            thresholds={},
        )
        detail = report.detailed_report(max_fields_per_issue=5)
        assert "... and 15 more" in detail

    def test_detailed_report_no_issues(self):
        """detailed_report with no issues says all passed."""
        report = FieldValidationReport(
            total_fields=3,
            valid_fields=["a", "b", "c"],
            invalid_fields=[],
            issues_by_field={},
            issues_by_type={},
            thresholds={},
        )
        detail = report.detailed_report()
        assert "All fields passed" in detail

    def test_to_dataframe_columns_and_rows(self, report):
        """to_dataframe has expected columns and row count."""
        df = report.to_dataframe()
        assert "field_uid" in df.columns
        assert "valid" in df.columns
        assert "issue_count" in df.columns
        assert len(df) == 3  # total_fields

    def test_to_dataframe_valid_fields_have_zero_issues(self, report):
        """Valid fields have issue_count=0 in dataframe."""
        df = report.to_dataframe()
        valid_rows = df[df["valid"].eq(True)]
        assert all(valid_rows["issue_count"] == 0)


class TestValidationResult:
    """Tests for ValidationResult.summary()."""

    def test_ready_summary(self):
        """READY status shows field count."""
        vr = ValidationResult(
            operation="calibration",
            ready=True,
            missing_data=[],
            incomplete_data=[],
            warnings=[],
            ready_fields=["A", "B", "C"],
            not_ready_fields=[],
        )
        s = vr.summary()
        assert "READY" in s
        assert "3 fields" in s

    def test_not_ready_summary(self):
        """NOT READY status is shown."""
        vr = ValidationResult(
            operation="calibration",
            ready=False,
            missing_data=["path/a", "path/b"],
            incomplete_data=[],
            warnings=["low coverage"],
            ready_fields=[],
            not_ready_fields=["A"],
        )
        s = vr.summary()
        assert "NOT READY" in s

    def test_missing_data_truncation_at_5(self):
        """Missing data list is truncated at 5."""
        paths = [f"path/{i}" for i in range(8)]
        vr = ValidationResult(
            operation="test",
            ready=False,
            missing_data=paths,
            incomplete_data=[],
            warnings=[],
            ready_fields=[],
            not_ready_fields=[],
        )
        s = vr.summary()
        assert "... and 3 more" in s

    def test_incomplete_data_percent(self):
        """Incomplete data shows percent complete."""
        partial = Coverage("test/path", DataStatus.PARTIAL, 7, 10, ["a", "b", "c"])
        vr = ValidationResult(
            operation="test",
            ready=False,
            missing_data=[],
            incomplete_data=[partial],
            warnings=["Some warning"],
            ready_fields=[],
            not_ready_fields=[],
        )
        s = vr.summary()
        assert "70%" in s
