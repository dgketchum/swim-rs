"""
Multi-station regression tests for SwimContainer.

Tests the container workflow using multiple flux stations:
- ALARC2_Smith6 (irrigated)
- MR (non-irrigated)
- S2 (irrigated)
- US-FPe (non-irrigated)

Compares outputs against golden reference files with 1% tolerance.

Run with: pytest tests/test_container_multi_station.py -v
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from conftest import (
    compare_json_with_tolerance,
    compare_scalars_with_tolerance,
    load_golden_json,
)


# =============================================================================
# Test Constants
# =============================================================================

STATION_UIDS = ["ALARC2_Smith6", "MR", "S2", "US-FPe"]
UID_COLUMN = "site_id"
START_DATE = "2020-01-01"
END_DATE = "2022-12-31"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def multi_station_shapefile(multi_station_fixture_path) -> Path:
    """Path to multi-station shapefile."""
    return multi_station_fixture_path / "data" / "gis" / "multi_station.shp"


@pytest.fixture
def multi_station_golden_dir(multi_station_fixture_path) -> Path:
    """Path to multi-station golden files directory."""
    return multi_station_fixture_path / "golden"


@pytest.fixture
def multi_station_input_dir(multi_station_fixture_path) -> Path:
    """Path to multi-station input data directory."""
    return multi_station_fixture_path / "input"


@pytest.fixture
def multi_station_has_golden_files(multi_station_golden_dir) -> bool:
    """Check if golden files exist."""
    return (multi_station_golden_dir / "ke_max.json").exists()


@pytest.fixture
def multi_station_has_input_data(multi_station_input_dir) -> bool:
    """Check if input data exists."""
    return multi_station_input_dir.exists() and any(multi_station_input_dir.iterdir())


# =============================================================================
# Container Creation Tests
# =============================================================================

class TestMultiStationContainerCreation:
    """Tests for multi-station SwimContainer creation."""

    def test_container_creates_with_multiple_fields(
        self, multi_station_shapefile, tmp_path
    ):
        """Container creates successfully with multiple fields."""
        if not multi_station_shapefile.exists():
            pytest.skip(f"Multi-station shapefile not found: {multi_station_shapefile}")

        from swimrs.container import SwimContainer

        container_path = tmp_path / "test.swim"
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(multi_station_shapefile),
            uid_column=UID_COLUMN,
            start_date=START_DATE,
            end_date=END_DATE,
        )

        assert container.n_fields == len(STATION_UIDS)
        for uid in STATION_UIDS:
            assert uid in container.field_uids

        container.close()

    def test_container_field_order_consistent(
        self, multi_station_shapefile, tmp_path
    ):
        """Container maintains consistent field ordering."""
        if not multi_station_shapefile.exists():
            pytest.skip(f"Multi-station shapefile not found: {multi_station_shapefile}")

        from swimrs.container import SwimContainer

        container_path = tmp_path / "test.swim"

        # Create first time
        container1 = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(multi_station_shapefile),
            uid_column=UID_COLUMN,
            start_date=START_DATE,
            end_date=END_DATE,
        )
        uids1 = list(container1.field_uids)
        container1.save()
        container1.close()

        # Reopen
        container2 = SwimContainer.open(str(container_path), mode="r")
        uids2 = list(container2.field_uids)
        container2.close()

        assert uids1 == uids2, "Field order not preserved after reopen"


# =============================================================================
# Per-Station Dynamics Tests
# =============================================================================

class TestPerStationDynamics:
    """Tests for per-station dynamics values against golden files."""

    @pytest.mark.regression
    @pytest.mark.parametrize("station_uid", STATION_UIDS)
    def test_ke_max_per_station(
        self,
        station_uid,
        multi_station_shapefile,
        multi_station_golden_dir,
        multi_station_has_golden_files,
        multi_station_input_dir,
        multi_station_has_input_data,
        tmp_path,
        tolerance,
    ):
        """ke_max matches golden file for each station."""
        if not multi_station_shapefile.exists():
            pytest.skip("Multi-station shapefile not found")
        if not multi_station_has_golden_files:
            pytest.skip("Golden files not found")
        if not multi_station_has_input_data:
            pytest.skip("Multi-station input data not found")

        from swimrs.container import SwimContainer

        # Load golden ke_max
        golden_ke = load_golden_json(multi_station_golden_dir, "ke_max")
        expected_ke = golden_ke.get(station_uid)
        if expected_ke is None:
            pytest.skip(f"No golden ke_max for station {station_uid}")

        # Create container
        container = _create_full_multi_station_container(
            multi_station_shapefile, multi_station_input_dir, tmp_path
        )

        # Compute dynamics
        container.compute.dynamics(
            etf_model="ssebop",
            masks=("irr", "inv_irr"),
            instruments=("landsat",),
            use_lulc=True,
        )

        # Get station index
        station_idx = container.field_uids.index(station_uid)

        # Compare ke_max
        ke_path = "derived/dynamics/ke_max"
        actual_ke = container._state.root[ke_path][station_idx]

        compare_scalars_with_tolerance(
            actual_ke, expected_ke,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            name=f"ke_max[{station_uid}]",
        )

        container.close()

    @pytest.mark.regression
    @pytest.mark.parametrize("station_uid", STATION_UIDS)
    def test_kc_max_per_station(
        self,
        station_uid,
        multi_station_shapefile,
        multi_station_golden_dir,
        multi_station_has_golden_files,
        multi_station_input_dir,
        multi_station_has_input_data,
        tmp_path,
        tolerance,
    ):
        """kc_max matches golden file for each station."""
        if not multi_station_shapefile.exists():
            pytest.skip("Multi-station shapefile not found")
        if not multi_station_has_golden_files:
            pytest.skip("Golden files not found")
        if not multi_station_has_input_data:
            pytest.skip("Multi-station input data not found")

        from swimrs.container import SwimContainer

        # Load golden kc_max
        golden_kc = load_golden_json(multi_station_golden_dir, "kc_max")
        expected_kc = golden_kc.get(station_uid)
        if expected_kc is None:
            pytest.skip(f"No golden kc_max for station {station_uid}")

        # Create container
        container = _create_full_multi_station_container(
            multi_station_shapefile, multi_station_input_dir, tmp_path
        )

        # Compute dynamics
        container.compute.dynamics(
            etf_model="ssebop",
            masks=("irr", "inv_irr"),
            instruments=("landsat",),
            use_lulc=True,
        )

        # Get station index
        station_idx = container.field_uids.index(station_uid)

        # Compare kc_max
        kc_path = "derived/dynamics/kc_max"
        actual_kc = container._state.root[kc_path][station_idx]

        compare_scalars_with_tolerance(
            actual_kc, expected_kc,
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
            name=f"kc_max[{station_uid}]",
        )

        container.close()

    @pytest.mark.regression
    @pytest.mark.parametrize("station_uid", STATION_UIDS)
    def test_irrigation_per_station(
        self,
        station_uid,
        multi_station_shapefile,
        multi_station_golden_dir,
        multi_station_has_golden_files,
        multi_station_input_dir,
        multi_station_has_input_data,
        tmp_path,
        tolerance,
    ):
        """Irrigation classification matches golden file for each station."""
        if not multi_station_shapefile.exists():
            pytest.skip("Multi-station shapefile not found")
        if not multi_station_has_golden_files:
            pytest.skip("Golden files not found")
        if not multi_station_has_input_data:
            pytest.skip("Multi-station input data not found")

        from swimrs.container import SwimContainer

        # Load golden irr_data
        golden_irr = load_golden_json(multi_station_golden_dir, "irr_data")
        expected_irr = golden_irr.get(station_uid)
        if expected_irr is None:
            pytest.skip(f"No golden irr_data for station {station_uid}")

        # Create container
        container = _create_full_multi_station_container(
            multi_station_shapefile, multi_station_input_dir, tmp_path
        )

        # Compute dynamics
        container.compute.dynamics(
            etf_model="ssebop",
            masks=("irr", "inv_irr"),
            instruments=("landsat",),
            use_lulc=True,
        )

        # Get station index
        station_idx = container.field_uids.index(station_uid)

        # Get actual irr_data
        irr_path = "derived/dynamics/irr_data"
        irr_arr = container._state.root[irr_path]
        actual_irr_json = irr_arr[station_idx]
        actual_irr = json.loads(actual_irr_json) if actual_irr_json else None

        compare_json_with_tolerance(
            actual_irr, expected_irr,
            rtol=tolerance["rtol"],
        )

        container.close()

    @pytest.mark.regression
    @pytest.mark.parametrize("station_uid", STATION_UIDS)
    def test_groundwater_per_station(
        self,
        station_uid,
        multi_station_shapefile,
        multi_station_golden_dir,
        multi_station_has_golden_files,
        multi_station_input_dir,
        multi_station_has_input_data,
        tmp_path,
        tolerance,
    ):
        """Groundwater subsidy matches golden file for each station."""
        if not multi_station_shapefile.exists():
            pytest.skip("Multi-station shapefile not found")
        if not multi_station_has_golden_files:
            pytest.skip("Golden files not found")
        if not multi_station_has_input_data:
            pytest.skip("Multi-station input data not found")

        from swimrs.container import SwimContainer

        # Load golden gwsub_data
        golden_gwsub = load_golden_json(multi_station_golden_dir, "gwsub_data")
        expected_gwsub = golden_gwsub.get(station_uid)
        if expected_gwsub is None:
            pytest.skip(f"No golden gwsub_data for station {station_uid}")

        # Create container
        container = _create_full_multi_station_container(
            multi_station_shapefile, multi_station_input_dir, tmp_path
        )

        # Compute dynamics
        container.compute.dynamics(
            etf_model="ssebop",
            masks=("irr", "inv_irr"),
            instruments=("landsat",),
            use_lulc=True,
        )

        # Get station index
        station_idx = container.field_uids.index(station_uid)

        # Get actual gwsub_data
        gwsub_path = "derived/dynamics/gwsub_data"
        gwsub_arr = container._state.root[gwsub_path]
        actual_gwsub_json = gwsub_arr[station_idx]
        actual_gwsub = json.loads(actual_gwsub_json) if actual_gwsub_json else None

        compare_json_with_tolerance(
            actual_gwsub, expected_gwsub,
            rtol=tolerance["rtol"],
        )

        container.close()


# =============================================================================
# Field Independence Tests
# =============================================================================

class TestFieldIndependence:
    """Tests that verify field computations are independent."""

    @pytest.mark.regression
    def test_field_values_independent(
        self,
        multi_station_shapefile,
        multi_station_golden_dir,
        multi_station_has_golden_files,
        multi_station_input_dir,
        multi_station_has_input_data,
        tmp_path,
        tolerance,
    ):
        """
        Each field's values match what would be computed individually.

        This ensures multi-station processing doesn't cross-contaminate
        between fields.
        """
        if not multi_station_shapefile.exists():
            pytest.skip("Multi-station shapefile not found")
        if not multi_station_has_golden_files:
            pytest.skip("Golden files not found")
        if not multi_station_has_input_data:
            pytest.skip("Multi-station input data not found")

        from swimrs.container import SwimContainer

        # Load all golden values
        golden_ke = load_golden_json(multi_station_golden_dir, "ke_max")
        golden_kc = load_golden_json(multi_station_golden_dir, "kc_max")

        # Create and compute multi-station container
        container = _create_full_multi_station_container(
            multi_station_shapefile, multi_station_input_dir, tmp_path
        )

        container.compute.dynamics(
            etf_model="ssebop",
            masks=("irr", "inv_irr"),
            instruments=("landsat",),
            use_lulc=True,
        )

        # Verify each field
        for i, uid in enumerate(container.field_uids):
            if uid not in golden_ke or uid not in golden_kc:
                continue

            # ke_max
            actual_ke = container._state.root["derived/dynamics/ke_max"][i]
            expected_ke = golden_ke[uid]
            compare_scalars_with_tolerance(
                actual_ke, expected_ke,
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                name=f"ke_max[{uid}]",
            )

            # kc_max
            actual_kc = container._state.root["derived/dynamics/kc_max"][i]
            expected_kc = golden_kc[uid]
            compare_scalars_with_tolerance(
                actual_kc, expected_kc,
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                name=f"kc_max[{uid}]",
            )

        container.close()


# =============================================================================
# Multi-Station Export Tests
# =============================================================================

class TestMultiStationExport:
    """Tests for multi-station export functionality."""

    @pytest.mark.regression
    def test_multi_station_export_produces_all_fields(
        self,
        multi_station_shapefile,
        multi_station_golden_dir,
        multi_station_has_golden_files,
        multi_station_input_dir,
        multi_station_has_input_data,
        tmp_path,
    ):
        """Multi-station export includes all fields."""
        if not multi_station_shapefile.exists():
            pytest.skip("Multi-station shapefile not found")
        if not multi_station_has_input_data:
            pytest.skip("Multi-station input data not found")

        from swimrs.container import SwimContainer

        # Create and compute
        container = _create_full_multi_station_container(
            multi_station_shapefile, multi_station_input_dir, tmp_path
        )

        container.compute.dynamics(
            etf_model="ssebop",
            masks=("irr", "inv_irr"),
            instruments=("landsat",),
            use_lulc=True,
        )

        # Export
        output_path = tmp_path / "prepped_input.json"
        container.export.prepped_input_json(
            output_path=str(output_path),
            etf_model="ssebop",
            masks=("irr", "inv_irr"),
        )

        assert output_path.exists()

        # Read JSONL and find the order section to verify all fields
        with open(output_path, 'r') as f:
            lines = f.readlines()

        # JSONL format has sections: props, irr_data, gwsub_data, ke_max, kc_max, order, time_series
        # Find the "order" section which lists all field UIDs
        exported_fids = set()
        for line in lines:
            record = json.loads(line)
            if "order" in record:
                exported_fids = set(record["order"])
                break
            # Also check for nested field data in props/ke_max sections
            for key in ["props", "ke_max", "kc_max"]:
                if key in record:
                    exported_fids.update(record[key].keys())
                    break

        assert len(exported_fids) == len(STATION_UIDS), \
            f"Expected {len(STATION_UIDS)} fields, got {len(exported_fids)}"

        for uid in STATION_UIDS:
            assert uid in exported_fids, f"Missing station {uid} in export"

        container.close()

    @pytest.mark.regression
    def test_multi_station_export_structure_matches_golden(
        self,
        multi_station_shapefile,
        multi_station_golden_dir,
        multi_station_has_golden_files,
        multi_station_input_dir,
        multi_station_has_input_data,
        tmp_path,
    ):
        """Multi-station export has correct structure."""
        if not multi_station_shapefile.exists():
            pytest.skip("Multi-station shapefile not found")
        if not multi_station_has_golden_files:
            pytest.skip("Golden files not found")
        if not multi_station_has_input_data:
            pytest.skip("Multi-station input data not found")

        from swimrs.container import SwimContainer

        # Create and compute
        container = _create_full_multi_station_container(
            multi_station_shapefile, multi_station_input_dir, tmp_path
        )

        container.compute.dynamics(
            etf_model="ssebop",
            masks=("irr", "inv_irr"),
            instruments=("landsat",),
            use_lulc=True,
        )

        # Export
        output_path = tmp_path / "prepped_input.json"
        container.export.prepped_input_json(
            output_path=str(output_path),
            etf_model="ssebop",
            masks=("irr", "inv_irr"),
        )

        # Load golden prepped_input
        golden_prepped_path = multi_station_golden_dir / "prepped_input.json"
        if not golden_prepped_path.exists():
            pytest.skip("Golden prepped_input.json not found")

        # Compare first record structure
        with open(output_path, 'r') as f:
            actual_first = json.loads(f.readline())

        with open(golden_prepped_path, 'r') as f:
            expected_first = json.loads(f.readline())

        assert set(actual_first.keys()) == set(expected_first.keys()), \
            "prepped_input.json has different structure"

        container.close()


# =============================================================================
# Full Workflow Test
# =============================================================================

class TestMultiStationFullWorkflow:
    """End-to-end multi-station workflow test."""

    @pytest.mark.regression
    @pytest.mark.slow
    def test_full_multi_station_workflow(
        self,
        multi_station_shapefile,
        multi_station_golden_dir,
        multi_station_has_golden_files,
        multi_station_input_dir,
        multi_station_has_input_data,
        tmp_path,
        tolerance,
    ):
        """Complete multi-station workflow produces results matching golden files."""
        if not multi_station_shapefile.exists():
            pytest.skip("Multi-station shapefile not found")
        if not multi_station_has_golden_files:
            pytest.skip("Golden files not found")
        if not multi_station_has_input_data:
            pytest.skip("Multi-station input data not found")

        from swimrs.container import SwimContainer

        # Create container
        container = _create_full_multi_station_container(
            multi_station_shapefile, multi_station_input_dir, tmp_path
        )

        # Run full workflow
        container.compute.dynamics(
            etf_model="ssebop",
            masks=("irr", "inv_irr"),
            instruments=("landsat",),
            use_lulc=True,
        )

        # Export
        output_path = tmp_path / "prepped_input.json"
        container.export.prepped_input_json(
            output_path=str(output_path),
            etf_model="ssebop",
            masks=("irr", "inv_irr"),
        )

        # Verify all golden values
        golden_ke = load_golden_json(multi_station_golden_dir, "ke_max")
        golden_kc = load_golden_json(multi_station_golden_dir, "kc_max")

        for i, uid in enumerate(container.field_uids):
            if uid in golden_ke:
                actual_ke = container._state.root["derived/dynamics/ke_max"][i]
                compare_scalars_with_tolerance(
                    actual_ke, golden_ke[uid],
                    rtol=tolerance["rtol"],
                    atol=tolerance["atol"],
                    name=f"ke_max[{uid}]",
                )

            if uid in golden_kc:
                actual_kc = container._state.root["derived/dynamics/kc_max"][i]
                compare_scalars_with_tolerance(
                    actual_kc, golden_kc[uid],
                    rtol=tolerance["rtol"],
                    atol=tolerance["atol"],
                    name=f"kc_max[{uid}]",
                )

        container.close()


# =============================================================================
# Helper Functions
# =============================================================================

def _create_full_multi_station_container(
    shapefile: Path, input_dir: Path, tmp_path: Path
):
    """Create and populate multi-station container with all input data."""
    from swimrs.container import SwimContainer

    container_path = tmp_path / "test.swim"
    container = SwimContainer.create(
        uri=str(container_path),
        fields_shapefile=str(shapefile),
        uid_column=UID_COLUMN,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    # Ingest available data - both irr and inv_irr masks
    ndvi_dir = input_dir / "ndvi"
    if ndvi_dir.exists():
        # Ingest irr mask first
        container.ingest.ndvi(
            source_dir=str(ndvi_dir),
            instrument="landsat",
            mask="irr",
        )
        # Then inv_irr mask (for non-irrigated stations like US-FPe)
        container.ingest.ndvi(
            source_dir=str(ndvi_dir),
            instrument="landsat",
            mask="inv_irr",
        )

    etf_dir = input_dir / "etf"
    if etf_dir.exists():
        # Ingest both masks for ETf
        container.ingest.etf(
            source_dir=str(etf_dir),
            model="ssebop",
            instrument="landsat",
            mask="irr",
        )
        container.ingest.etf(
            source_dir=str(etf_dir),
            model="ssebop",
            instrument="landsat",
            mask="inv_irr",
        )

    met_dir = input_dir / "met"
    if met_dir.exists():
        container.ingest.gridmet(source_dir=str(met_dir))

    # Ingest properties
    properties_dir = input_dir / "properties"
    if properties_dir.exists():
        lulc_csv = properties_dir / "lulc.csv"
        ssurgo_csv = properties_dir / "ssurgo.csv"
        irr_csv = properties_dir / "irr.csv"
        if lulc_csv.exists() or ssurgo_csv.exists():
            container.ingest.properties(
                lulc_csv=str(lulc_csv) if lulc_csv.exists() else None,
                soils_csv=str(ssurgo_csv) if ssurgo_csv.exists() else None,
                irrigation_csv=str(irr_csv) if irr_csv.exists() else None,
                uid_column="site_id",
                lulc_column="modis_lc",
                extra_lulc_column="glc10_lc",
            )

    return container


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
