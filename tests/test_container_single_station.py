"""
Single-station regression tests for SwimContainer.

Tests the container workflow using the S2 (Crane) flux station data.
Compares outputs against golden reference files with 1% tolerance.

Run with: pytest tests/test_container_single_station.py -v
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from conftest import (
    compare_json_with_tolerance,
    compare_scalars_with_tolerance,
    create_test_container,
    get_container_dynamics_values,
    load_golden_json,
)


# =============================================================================
# Test Constants
# =============================================================================

S2_UID = "S2"  # Field UID for Crane station
S2_UID_COLUMN = "site_id"  # Column name in shapefile
START_DATE = "2020-01-01"
END_DATE = "2022-12-31"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def s2_shapefile(s2_fixture_path) -> Path:
    """Path to S2 shapefile."""
    return s2_fixture_path / "data" / "gis" / "flux_footprint_s2.shp"


@pytest.fixture
def s2_golden_dir(s2_fixture_path) -> Path:
    """Path to S2 golden files directory."""
    return s2_fixture_path / "golden"


@pytest.fixture
def s2_input_dir(s2_fixture_path) -> Path:
    """Path to S2 input data directory."""
    return s2_fixture_path / "input"


@pytest.fixture
def s2_has_golden_files(s2_golden_dir) -> bool:
    """Check if golden files exist."""
    return (s2_golden_dir / "ke_max.json").exists()


@pytest.fixture
def s2_has_input_data(s2_input_dir) -> bool:
    """Check if input data exists."""
    return s2_input_dir.exists() and any(s2_input_dir.iterdir())


# =============================================================================
# Container Creation Tests
# =============================================================================

class TestContainerCreation:
    """Tests for SwimContainer creation with S2 data."""

    def test_container_creates_from_shapefile(self, s2_shapefile, tmp_path):
        """Container creates successfully from S2 shapefile."""
        if not s2_shapefile.exists():
            pytest.skip(f"S2 shapefile not found: {s2_shapefile}")

        from swimrs.container import SwimContainer

        container_path = tmp_path / "test.swim"
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(s2_shapefile),
            uid_column=S2_UID_COLUMN,
            start_date=START_DATE,
            end_date=END_DATE,
        )

        assert container.n_fields == 1
        assert S2_UID in container.field_uids
        assert container.start_date.strftime("%Y-%m-%d") == START_DATE
        assert container.end_date.strftime("%Y-%m-%d") == END_DATE

        container.close()

    def test_container_reopens(self, s2_shapefile, tmp_path):
        """Container can be closed and reopened."""
        if not s2_shapefile.exists():
            pytest.skip(f"S2 shapefile not found: {s2_shapefile}")

        from swimrs.container import SwimContainer

        container_path = tmp_path / "test.swim"

        # Create and close
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(s2_shapefile),
            uid_column=S2_UID_COLUMN,
            start_date=START_DATE,
            end_date=END_DATE,
        )
        container.save()
        container.close()

        # Reopen and verify
        container = SwimContainer.open(str(container_path), mode="r")
        assert container.n_fields == 1
        assert S2_UID in container.field_uids
        container.close()


# =============================================================================
# Data Ingestion Tests
# =============================================================================

class TestDataIngestion:
    """Tests for data ingestion into S2 container."""

    @pytest.mark.regression
    def test_ingest_ndvi(self, s2_shapefile, s2_input_dir, s2_has_input_data, tmp_path):
        """NDVI ingestion produces expected array shapes."""
        if not s2_shapefile.exists():
            pytest.skip(f"S2 shapefile not found: {s2_shapefile}")
        if not s2_has_input_data:
            pytest.skip("S2 input data not found - run generate_golden_files.py first")

        ndvi_dir = s2_input_dir / "ndvi"
        if not ndvi_dir.exists():
            pytest.skip(f"NDVI directory not found: {ndvi_dir}")

        from swimrs.container import SwimContainer

        container_path = tmp_path / "test.swim"
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(s2_shapefile),
            uid_column=S2_UID_COLUMN,
            start_date=START_DATE,
            end_date=END_DATE,
        )

        container.ingest.ndvi(
            source_dir=str(ndvi_dir),
            instrument="landsat",
            mask="irr",
        )

        # Verify NDVI array exists with correct shape
        ndvi_path = "remote_sensing/ndvi/landsat/irr"
        assert ndvi_path in container._state.root

        ndvi_arr = container._state.root[ndvi_path]
        assert ndvi_arr.shape[1] == 1  # One field

        # Check NDVI values are in valid range
        values = ndvi_arr[:, 0]
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            assert valid_values.min() >= -0.5
            assert valid_values.max() <= 1.0

        container.close()

    @pytest.mark.regression
    def test_ingest_etf(self, s2_shapefile, s2_input_dir, s2_has_input_data, tmp_path):
        """ETf ingestion produces expected array shapes."""
        if not s2_shapefile.exists():
            pytest.skip(f"S2 shapefile not found: {s2_shapefile}")
        if not s2_has_input_data:
            pytest.skip("S2 input data not found")

        etf_dir = s2_input_dir / "etf"
        if not etf_dir.exists():
            pytest.skip(f"ETf directory not found: {etf_dir}")

        from swimrs.container import SwimContainer

        container_path = tmp_path / "test.swim"
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(s2_shapefile),
            uid_column=S2_UID_COLUMN,
            start_date=START_DATE,
            end_date=END_DATE,
        )

        container.ingest.etf(
            source_dir=str(etf_dir),
            model="ssebop",
            instrument="landsat",
            mask="irr",
        )

        # Verify ETf array exists with correct shape
        etf_path = "remote_sensing/etf/landsat/ssebop/irr"
        assert etf_path in container._state.root

        etf_arr = container._state.root[etf_path]
        assert etf_arr.shape[1] == 1  # One field

        # Check ETf values are in valid range
        values = etf_arr[:, 0]
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            assert valid_values.min() >= 0.0
            assert valid_values.max() <= 2.0

        container.close()

    @pytest.mark.regression
    def test_ingest_meteorology(self, s2_shapefile, s2_input_dir, s2_has_input_data, tmp_path):
        """Meteorology ingestion produces expected values."""
        if not s2_shapefile.exists():
            pytest.skip(f"S2 shapefile not found: {s2_shapefile}")
        if not s2_has_input_data:
            pytest.skip("S2 input data not found")

        met_dir = s2_input_dir / "met"
        if not met_dir.exists():
            pytest.skip(f"Meteorology directory not found: {met_dir}")

        from swimrs.container import SwimContainer

        container_path = tmp_path / "test.swim"
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(s2_shapefile),
            uid_column=S2_UID_COLUMN,
            start_date=START_DATE,
            end_date=END_DATE,
        )

        container.ingest.gridmet(source_dir=str(met_dir))

        # Verify key meteorology variables exist
        for var in ["eto", "prcp", "tmin", "tmax"]:
            path = f"meteorology/gridmet/{var}"
            assert path in container._state.root, f"Missing {var}"

        # Check ETo values are reasonable
        eto_arr = container._state.root["meteorology/gridmet/eto"]
        eto_values = eto_arr[:, 0]
        valid_eto = eto_values[~np.isnan(eto_values)]
        if len(valid_eto) > 0:
            assert valid_eto.min() >= 0.0
            assert valid_eto.max() <= 20.0  # mm/day

        container.close()


# =============================================================================
# Dynamics Computation Tests
# =============================================================================

class TestDynamicsComputation:
    """Tests for dynamics computation against golden files."""

    @pytest.mark.regression
    def test_ke_max_matches_golden(
        self,
        s2_shapefile,
        s2_golden_dir,
        s2_has_golden_files,
        s2_input_dir,
        s2_has_input_data,
        tmp_path,
        tolerance,
    ):
        """ke_max matches golden file within tolerance."""
        if not s2_shapefile.exists():
            pytest.skip(f"S2 shapefile not found: {s2_shapefile}")
        if not s2_has_golden_files:
            pytest.skip("Golden files not found - run generate_golden_files.py first")
        if not s2_has_input_data:
            pytest.skip("S2 input data not found")

        from swimrs.container import SwimContainer

        # Load golden ke_max
        golden_ke = load_golden_json(s2_golden_dir, "ke_max")

        # Create and populate container
        container = _create_full_s2_container(s2_shapefile, s2_input_dir, tmp_path)

        # Compute dynamics
        container.compute.dynamics(
            etf_model="ssebop",
            masks=("irr",),
            instruments=("landsat",),
            use_lulc=True,
        )

        # Compare ke_max
        ke_path = "derived/dynamics/ke_max"
        assert ke_path in container._state.root

        actual_ke = container._state.root[ke_path][0]
        expected_ke = golden_ke.get(S2_UID, golden_ke.get("S2"))

        compare_scalars_with_tolerance(
            actual_ke, expected_ke,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            name="ke_max",
        )

        container.close()

    @pytest.mark.regression
    def test_kc_max_matches_golden(
        self,
        s2_shapefile,
        s2_golden_dir,
        s2_has_golden_files,
        s2_input_dir,
        s2_has_input_data,
        tmp_path,
        tolerance,
    ):
        """kc_max matches golden file within tolerance."""
        if not s2_shapefile.exists():
            pytest.skip(f"S2 shapefile not found: {s2_shapefile}")
        if not s2_has_golden_files:
            pytest.skip("Golden files not found")
        if not s2_has_input_data:
            pytest.skip("S2 input data not found")

        from swimrs.container import SwimContainer

        # Load golden kc_max
        golden_kc = load_golden_json(s2_golden_dir, "kc_max")

        # Create and populate container
        container = _create_full_s2_container(s2_shapefile, s2_input_dir, tmp_path)

        # Compute dynamics
        container.compute.dynamics(
            etf_model="ssebop",
            masks=("irr",),
            instruments=("landsat",),
            use_lulc=True,
        )

        # Compare kc_max
        kc_path = "derived/dynamics/kc_max"
        assert kc_path in container._state.root

        actual_kc = container._state.root[kc_path][0]
        expected_kc = golden_kc.get(S2_UID, golden_kc.get("S2"))

        compare_scalars_with_tolerance(
            actual_kc, expected_kc,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            name="kc_max",
        )

        container.close()

    @pytest.mark.regression
    def test_irrigation_classification_matches_golden(
        self,
        s2_shapefile,
        s2_golden_dir,
        s2_has_golden_files,
        s2_input_dir,
        s2_has_input_data,
        tmp_path,
        tolerance,
    ):
        """Irrigation classification matches golden file."""
        if not s2_shapefile.exists():
            pytest.skip(f"S2 shapefile not found: {s2_shapefile}")
        if not s2_has_golden_files:
            pytest.skip("Golden files not found")
        if not s2_has_input_data:
            pytest.skip("S2 input data not found")

        from swimrs.container import SwimContainer

        # Load golden irr_data
        golden_irr = load_golden_json(s2_golden_dir, "irr_data")

        # Create and populate container
        container = _create_full_s2_container(s2_shapefile, s2_input_dir, tmp_path)

        # Compute dynamics
        container.compute.dynamics(
            etf_model="ssebop",
            masks=("irr",),
            instruments=("landsat",),
            use_lulc=True,
        )

        # Get actual irr_data
        irr_path = "derived/dynamics/irr_data"
        assert irr_path in container._state.root

        irr_arr = container._state.root[irr_path]
        actual_irr_json = irr_arr[0]
        actual_irr = json.loads(actual_irr_json) if actual_irr_json else None

        expected_irr = golden_irr.get(S2_UID, golden_irr.get("S2"))

        # Compare structure and key values
        compare_json_with_tolerance(
            actual_irr, expected_irr,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
        )

        container.close()

    @pytest.mark.regression
    def test_groundwater_subsidy_matches_golden(
        self,
        s2_shapefile,
        s2_golden_dir,
        s2_has_golden_files,
        s2_input_dir,
        s2_has_input_data,
        tmp_path,
        tolerance,
    ):
        """Groundwater subsidy matches golden file."""
        if not s2_shapefile.exists():
            pytest.skip(f"S2 shapefile not found: {s2_shapefile}")
        if not s2_has_golden_files:
            pytest.skip("Golden files not found")
        if not s2_has_input_data:
            pytest.skip("S2 input data not found")

        from swimrs.container import SwimContainer

        # Load golden gwsub_data
        golden_gwsub = load_golden_json(s2_golden_dir, "gwsub_data")

        # Create and populate container
        container = _create_full_s2_container(s2_shapefile, s2_input_dir, tmp_path)

        # Compute dynamics
        container.compute.dynamics(
            etf_model="ssebop",
            masks=("irr",),
            instruments=("landsat",),
            use_lulc=True,
        )

        # Get actual gwsub_data
        gwsub_path = "derived/dynamics/gwsub_data"
        assert gwsub_path in container._state.root

        gwsub_arr = container._state.root[gwsub_path]
        actual_gwsub_json = gwsub_arr[0]
        actual_gwsub = json.loads(actual_gwsub_json) if actual_gwsub_json else None

        expected_gwsub = golden_gwsub.get(S2_UID, golden_gwsub.get("S2"))

        # Compare structure and key values
        compare_json_with_tolerance(
            actual_gwsub, expected_gwsub,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
        )

        container.close()


# =============================================================================
# Export Tests
# =============================================================================

class TestExport:
    """Tests for prepped_input.json export."""

    @pytest.mark.regression
    def test_prepped_input_export(
        self,
        s2_shapefile,
        s2_golden_dir,
        s2_has_golden_files,
        s2_input_dir,
        s2_has_input_data,
        tmp_path,
        tolerance,
    ):
        """prepped_input.json export matches golden structure."""
        if not s2_shapefile.exists():
            pytest.skip(f"S2 shapefile not found: {s2_shapefile}")
        if not s2_has_golden_files:
            pytest.skip("Golden files not found")
        if not s2_has_input_data:
            pytest.skip("S2 input data not found")

        from swimrs.container import SwimContainer

        # Create and populate container
        container = _create_full_s2_container(s2_shapefile, s2_input_dir, tmp_path)

        # Compute dynamics
        container.compute.dynamics(
            etf_model="ssebop",
            masks=("irr",),
            instruments=("landsat",),
            use_lulc=True,
        )

        # Export prepped_input.json
        output_path = tmp_path / "prepped_input.json"
        container.export.prepped_input_json(
            output_path=str(output_path),
            etf_model="ssebop",
            masks=("irr",),
        )

        assert output_path.exists()

        # Load and compare with golden
        golden_prepped_path = s2_golden_dir / "prepped_input.json"
        if golden_prepped_path.exists():
            # Compare first record (JSONL format)
            with open(output_path, 'r') as f:
                actual_first = json.loads(f.readline())

            with open(golden_prepped_path, 'r') as f:
                expected_first = json.loads(f.readline())

            # Compare structure
            assert set(actual_first.keys()) == set(expected_first.keys()), \
                "prepped_input.json has different keys"

            # Compare FID
            assert actual_first.get("fid") == expected_first.get("fid")

        container.close()


# =============================================================================
# Full Workflow Test
# =============================================================================

class TestFullWorkflow:
    """End-to-end workflow test."""

    @pytest.mark.regression
    @pytest.mark.slow
    def test_full_workflow_produces_consistent_results(
        self,
        s2_shapefile,
        s2_golden_dir,
        s2_has_golden_files,
        s2_input_dir,
        s2_has_input_data,
        tmp_path,
        tolerance,
    ):
        """Complete workflow produces results matching golden files."""
        if not s2_shapefile.exists():
            pytest.skip(f"S2 shapefile not found: {s2_shapefile}")
        if not s2_has_golden_files:
            pytest.skip("Golden files not found")
        if not s2_has_input_data:
            pytest.skip("S2 input data not found")

        from swimrs.container import SwimContainer

        # Create container
        container = _create_full_s2_container(s2_shapefile, s2_input_dir, tmp_path)

        # Run full workflow
        container.compute.dynamics(
            etf_model="ssebop",
            masks=("irr",),
            instruments=("landsat",),
            use_lulc=True,
        )

        # Get all dynamics values
        actual = get_container_dynamics_values(container)

        # Compare against all golden files
        for name in ["ke_max", "kc_max", "irr_data", "gwsub_data"]:
            golden_file = s2_golden_dir / f"{name}.json"
            if golden_file.exists():
                expected = load_golden_json(s2_golden_dir, name)
                expected_value = expected.get(S2_UID, expected.get("S2"))
                actual_value = actual[name][0] if isinstance(actual[name], list) else actual[name]

                if isinstance(actual_value, (int, float)):
                    compare_scalars_with_tolerance(
                        actual_value, expected_value,
                        rtol=tolerance["rtol"],
                        atol=tolerance["atol"],
                        name=name,
                    )
                else:
                    compare_json_with_tolerance(
                        actual_value, expected_value,
                        rtol=tolerance["rtol"],
                    )

        container.close()


# =============================================================================
# Helper Functions
# =============================================================================

def _create_full_s2_container(shapefile: Path, input_dir: Path, tmp_path: Path):
    """Create and populate S2 container with all input data."""
    from swimrs.container import SwimContainer

    container_path = tmp_path / "test.swim"
    container = SwimContainer.create(
        uri=str(container_path),
        fields_shapefile=str(shapefile),
        uid_column=S2_UID_COLUMN,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    # Ingest available data
    ndvi_dir = input_dir / "ndvi"
    if ndvi_dir.exists():
        container.ingest.ndvi(
            source_dir=str(ndvi_dir),
            instrument="landsat",
            mask="irr",
        )

    etf_dir = input_dir / "etf"
    if etf_dir.exists():
        container.ingest.etf(
            source_dir=str(etf_dir),
            model="ssebop",
            instrument="landsat",
            mask="irr",
        )

    met_dir = input_dir / "met"
    if met_dir.exists():
        container.ingest.gridmet(source_dir=str(met_dir))

    properties_json = input_dir / "properties" / "properties.json"
    if properties_json.exists():
        container.ingest.dynamics(dynamics_json=str(properties_json))

    return container


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
