"""Tests for swimrs.data_extraction.snodas.snodas module."""

import json
from pathlib import Path

import pandas as pd
import pytest

from swimrs.data_extraction.snodas.snodas import create_timeseries_json


class TestCreateTimeseriesJson:
    """Tests for create_timeseries_json function."""

    @pytest.fixture
    def snodas_csvs(self, tmp_path):
        """Create sample SNODAS CSV files."""
        # CSV 1: Jan 2020 data
        df1 = pd.DataFrame(
            {
                "FID": ["field_001", "field_002"],
                "2020-01-01": [0.05, 0.08],  # SWE in meters
                "2020-01-02": [0.06, 0.09],
            }
        )
        df1.to_csv(tmp_path / "snodas_202001.csv", index=False)

        # CSV 2: Feb 2020 data
        df2 = pd.DataFrame(
            {
                "FID": ["field_001", "field_002"],
                "2020-02-01": [0.10, 0.12],
                "2020-02-02": [0.11, 0.13],
            }
        )
        df2.to_csv(tmp_path / "snodas_202002.csv", index=False)

        return tmp_path

    def test_creates_json_file(self, snodas_csvs, tmp_path):
        """create_timeseries_json creates output JSON file."""
        json_out = tmp_path / "output" / "swe_timeseries.json"
        json_out.parent.mkdir(parents=True, exist_ok=True)

        create_timeseries_json(str(snodas_csvs), str(json_out))

        assert json_out.exists()

    def test_json_contains_all_features(self, snodas_csvs, tmp_path):
        """Output JSON contains all features from input CSVs."""
        json_out = tmp_path / "output" / "swe_timeseries.json"
        json_out.parent.mkdir(parents=True, exist_ok=True)

        create_timeseries_json(str(snodas_csvs), str(json_out))

        with open(json_out) as f:
            data = json.load(f)

        assert "field_001" in data
        assert "field_002" in data

    def test_converts_meters_to_mm(self, snodas_csvs, tmp_path):
        """Values are converted from meters to millimeters (*1000)."""
        json_out = tmp_path / "output" / "swe_timeseries.json"
        json_out.parent.mkdir(parents=True, exist_ok=True)

        create_timeseries_json(str(snodas_csvs), str(json_out))

        with open(json_out) as f:
            data = json.load(f)

        # Original value was 0.05 m, should be 50 mm
        field1_values = data["field_001"]
        jan1_entry = next(e for e in field1_values if e["date"] == "2020-01-01")
        assert jan1_entry["value"] == 50.0  # 0.05 * 1000

    def test_includes_date_and_value_keys(self, snodas_csvs, tmp_path):
        """Each entry has 'date' and 'value' keys."""
        json_out = tmp_path / "output" / "swe_timeseries.json"
        json_out.parent.mkdir(parents=True, exist_ok=True)

        create_timeseries_json(str(snodas_csvs), str(json_out))

        with open(json_out) as f:
            data = json.load(f)

        for fid, entries in data.items():
            for entry in entries:
                assert "date" in entry
                assert "value" in entry

    def test_aggregates_multiple_csvs(self, snodas_csvs, tmp_path):
        """Data from multiple CSVs is aggregated."""
        json_out = tmp_path / "output" / "swe_timeseries.json"
        json_out.parent.mkdir(parents=True, exist_ok=True)

        create_timeseries_json(str(snodas_csvs), str(json_out))

        with open(json_out) as f:
            data = json.load(f)

        # Each field should have 4 entries (2 dates x 2 files)
        assert len(data["field_001"]) == 4
        assert len(data["field_002"]) == 4

    def test_custom_feature_id(self, tmp_path):
        """create_timeseries_json respects custom feature_id column."""
        csv_dir = tmp_path / "csv_custom"
        csv_dir.mkdir()

        df = pd.DataFrame(
            {
                "SITE_ID": ["site_A", "site_B"],
                "2020-01-01": [0.02, 0.03],
            }
        )
        df.to_csv(csv_dir / "snodas.csv", index=False)

        json_out = tmp_path / "output.json"
        create_timeseries_json(str(csv_dir), str(json_out), feature_id="SITE_ID")

        with open(json_out) as f:
            data = json.load(f)

        assert "site_A" in data
        assert "site_B" in data

    def test_empty_directory(self, tmp_path):
        """create_timeseries_json handles empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        json_out = tmp_path / "empty_output.json"

        create_timeseries_json(str(empty_dir), str(json_out))

        with open(json_out) as f:
            data = json.load(f)

        assert data == {}


class TestCreateTimeseriesJsonValues:
    """Value-based tests for create_timeseries_json."""

    @pytest.fixture
    def precise_swe_csv(self, tmp_path):
        """Create CSV with precise SWE values for testing."""
        csv_dir = tmp_path / "precise"
        csv_dir.mkdir()

        df = pd.DataFrame(
            {
                "FID": ["test_field"],
                "2020-01-15": [0.123],  # 123 mm
                "2020-01-16": [0.0],    # 0 mm (no snow)
                "2020-01-17": [0.456],  # 456 mm
            }
        )
        df.to_csv(csv_dir / "snodas.csv", index=False)
        return csv_dir

    def test_zero_swe_preserved(self, precise_swe_csv, tmp_path):
        """Zero SWE values are preserved in output."""
        json_out = tmp_path / "output.json"
        create_timeseries_json(str(precise_swe_csv), str(json_out))

        with open(json_out) as f:
            data = json.load(f)

        values = {e["date"]: e["value"] for e in data["test_field"]}
        assert values["2020-01-16"] == 0.0

    def test_precise_conversion(self, precise_swe_csv, tmp_path):
        """Conversion from m to mm is precise."""
        json_out = tmp_path / "output.json"
        create_timeseries_json(str(precise_swe_csv), str(json_out))

        with open(json_out) as f:
            data = json.load(f)

        values = {e["date"]: e["value"] for e in data["test_field"]}
        assert values["2020-01-15"] == 123.0
        assert values["2020-01-17"] == 456.0
