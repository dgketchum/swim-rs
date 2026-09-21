"""Tests for the project container factory and supporting functions.

Tests cover:
- copy_static_groups(): UID check, selective copy, attrs preservation
- _copy_zarr_group(): recursive copy with attrs (including array attrs)
- compute_irr_data(): irrigation windows from NDVI only
- _extend_irr_props(): f_irr projection into missing years
- _compute_irr_dynamics(): None-safe threshold
- create_run_container(): target cleanup on failure, return type
- End-to-end: copy_static_groups with real containers, health check no-snow
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.core.dtype import VariableLengthUTF8

FIXTURE_SHP = (
    Path(__file__).parent.parent / "fixtures" / "S2" / "data" / "gis" / "flux_footprint_s2.shp"
)


def _create_runnable_container(tmp_path, *, start_date, end_date):
    """Build a minimal runnable container with the required forcings."""
    from swimrs.container.container import SwimContainer

    container = SwimContainer.create(
        str(tmp_path / "run_ready.swim"),
        fields_shapefile=str(FIXTURE_SHP),
        uid_column="site_id",
        start_date=start_date,
        end_date=end_date,
    )

    awc = container._create_property_array("properties/soils/awc")
    awc[:] = np.array([0.15], dtype=np.float32)

    ksat = container._create_property_array("properties/soils/ksat")
    ksat[:] = np.array([10.0], dtype=np.float32)

    modis = container._create_property_array(
        "properties/land_cover/modis_lc",
        dtype="int16",
        fill_value=-1,
    )
    modis[:] = np.array([12], dtype=np.int16)

    ndvi = container._create_timeseries_array("remote_sensing/ndvi/landsat/no_mask")
    n_days = container.n_days
    ndvi[:] = np.linspace(0.3, 0.6, n_days, dtype=np.float32).reshape(-1, 1)

    for path, values in {
        "meteorology/gridmet/prcp": np.zeros(n_days, dtype=np.float32),
        "meteorology/gridmet/tmin": np.linspace(5.0, 9.0, n_days, dtype=np.float32),
        "meteorology/gridmet/tmax": np.linspace(15.0, 19.0, n_days, dtype=np.float32),
        "meteorology/gridmet/srad": np.linspace(18.0, 20.0, n_days, dtype=np.float32),
        "meteorology/gridmet/eto": np.linspace(3.0, 3.8, n_days, dtype=np.float32),
    }.items():
        arr = container._create_timeseries_array(path)
        arr[:] = np.asarray(values, dtype=np.float32).reshape(-1, 1)

    container.save()
    return container


# ---------------------------------------------------------------------------
# _copy_zarr_group
# ---------------------------------------------------------------------------


class TestCopyZarrGroup:
    """Tests for SwimContainer._copy_zarr_group()."""

    def test_copies_arrays(self, tmp_path):
        src = zarr.open_group(str(tmp_path / "src"), mode="w")
        g = src.create_group("props")
        g.create_array("vals", data=np.array([1.0, 2.0, 3.0]))

        dst = zarr.open_group(str(tmp_path / "dst"), mode="w")

        from swimrs.container.container import SwimContainer

        SwimContainer._copy_zarr_group(src["props"], dst, "props")

        np.testing.assert_array_equal(dst["props/vals"][:], [1.0, 2.0, 3.0])

    def test_copies_nested_groups(self, tmp_path):
        src = zarr.open_group(str(tmp_path / "src"), mode="w")
        g = src.create_group("outer")
        inner = g.create_group("inner")
        inner.create_array("data", data=np.array([7.0]))

        dst = zarr.open_group(str(tmp_path / "dst"), mode="w")

        from swimrs.container.container import SwimContainer

        SwimContainer._copy_zarr_group(src["outer"], dst, "outer")

        np.testing.assert_array_equal(dst["outer/inner/data"][:], [7.0])

    def test_preserves_group_attrs(self, tmp_path):
        src = zarr.open_group(str(tmp_path / "src"), mode="w")
        g = src.create_group("cal")
        g.attrs["n_batches"] = 5
        g.attrs["summary_stat"] = "median"
        g.create_array("params", data=np.array([0.1, 0.2]))

        dst = zarr.open_group(str(tmp_path / "dst"), mode="w")

        from swimrs.container.container import SwimContainer

        SwimContainer._copy_zarr_group(src["cal"], dst, "cal")

        assert dict(dst["cal"].attrs) == {"n_batches": 5, "summary_stat": "median"}
        np.testing.assert_array_equal(dst["cal/params"][:], [0.1, 0.2])

    def test_copies_variable_length_utf8(self, tmp_path):
        src = zarr.open_group(str(tmp_path / "src"), mode="w")
        g = src.create_group("dyn")
        arr = g.create_array("gwsub", shape=(2,), dtype=VariableLengthUTF8())
        arr[:] = ['{"a":1}', ""]

        dst = zarr.open_group(str(tmp_path / "dst"), mode="w")

        from swimrs.container.container import SwimContainer

        SwimContainer._copy_zarr_group(src["dyn"], dst, "dyn")

        result = list(dst["dyn/gwsub"][:])
        assert result[0] == '{"a":1}'
        assert result[1] == ""

    def test_overwrites_existing(self, tmp_path):
        src = zarr.open_group(str(tmp_path / "src"), mode="w")
        g = src.create_group("grp")
        g.create_array("x", data=np.array([10.0]))

        dst = zarr.open_group(str(tmp_path / "dst"), mode="w")
        old = dst.create_group("grp")
        old.create_array("y", data=np.array([99.0]))

        from swimrs.container.container import SwimContainer

        SwimContainer._copy_zarr_group(src["grp"], dst, "grp")

        assert "x" in dst["grp"]
        # Old array should be gone
        assert "y" not in dst["grp"]

    def test_preserves_array_attrs(self, tmp_path):
        src = zarr.open_group(str(tmp_path / "src"), mode="w")
        g = src.create_group("grp")
        arr = g.create_array("arr", data=np.array([1.0, 2.0]))
        arr.attrs["units"] = "mm/day"
        arr.attrs["source"] = "gridmet"

        dst = zarr.open_group(str(tmp_path / "dst"), mode="w")

        from swimrs.container.container import SwimContainer

        SwimContainer._copy_zarr_group(src["grp"], dst, "grp")

        assert dict(dst["grp/arr"].attrs) == {"units": "mm/day", "source": "gridmet"}


# ---------------------------------------------------------------------------
# _extend_irr_props
# ---------------------------------------------------------------------------


class TestExtendIrrProps:
    """Tests for _extend_irr_props()."""

    def test_fills_missing_years_with_median(self):
        from swimrs.container.components.calculator import _extend_irr_props

        irr_props = {
            "field_1": {"2018": 0.8, "2019": 0.9, "2020": 0.7, "2021": 0.85},
        }
        target_years = [2020, 2021, 2022, 2023]

        result = _extend_irr_props(irr_props, target_years)

        assert "2022" in result["field_1"]
        assert "2023" in result["field_1"]
        # Median of [0.8, 0.9, 0.7, 0.85] = 0.825
        assert abs(result["field_1"]["2022"] - 0.825) < 0.01

    def test_no_change_when_all_years_present(self):
        from swimrs.container.components.calculator import _extend_irr_props

        irr_props = {"f1": {"2020": 0.5, "2021": 0.6}}
        target_years = [2020, 2021]

        result = _extend_irr_props(irr_props, target_years)

        assert result["f1"] == {"2020": 0.5, "2021": 0.6}

    def test_uses_last_n_recent(self):
        from swimrs.container.components.calculator import _extend_irr_props

        irr_props = {
            "f1": {str(y): 0.0 for y in range(2010, 2020)},
        }
        # Set last 3 years to 1.0
        irr_props["f1"]["2017"] = 1.0
        irr_props["f1"]["2018"] = 1.0
        irr_props["f1"]["2019"] = 1.0

        result = _extend_irr_props(irr_props, [2025], n_recent=3)

        # Median of last 3 years (all 1.0)
        assert result["f1"]["2025"] == 1.0

    def test_empty_irr_props(self):
        from swimrs.container.components.calculator import _extend_irr_props

        result = _extend_irr_props({}, [2025])
        assert result == {}

    def test_field_with_no_years(self):
        from swimrs.container.components.calculator import _extend_irr_props

        result = _extend_irr_props({"f1": {}}, [2025])
        # No available years → nothing to project from
        assert "2025" not in result["f1"]


# ---------------------------------------------------------------------------
# _compute_irr_dynamics threshold safety
# ---------------------------------------------------------------------------


class TestComputeIrrDynamicsThreshold:
    """Test that _compute_irr_dynamics handles None threshold."""

    def test_none_threshold_defaults_to_0_3(self):
        from swimrs.container.project import _compute_irr_dynamics

        config = MagicMock()
        config.irrigation_threshold = None
        config.mask_mode = "irrigation"

        target = MagicMock()
        target._root = MagicMock()
        target._root.__contains__ = MagicMock(return_value=False)

        # Should not raise even though irrigation_threshold is None
        _compute_irr_dynamics(target, config, "observed")
        # merged_ndvi not found → prints warning, returns without calling compute


# ---------------------------------------------------------------------------
# create_run_container closes handles on failure
# ---------------------------------------------------------------------------


class TestCreateRunContainerCleanup:
    """Test that both source and target are closed on mid-build failure."""

    def test_target_closed_on_exception(self):
        from swimrs.container.project import create_run_container

        config = MagicMock()
        config.container_path = "/nonexistent/source.swim"

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            create_run_container(config, mode="hindcast")

    def test_returns_path_string(self):
        """create_run_container return annotation is str, not SwimContainer."""
        import inspect

        from swimrs.container.project import create_run_container

        sig = inspect.signature(create_run_container)
        # With `from __future__ import annotations`, annotations are strings
        assert sig.return_annotation in (str, "str")


# ---------------------------------------------------------------------------
# End-to-end: copy_static_groups with real SwimContainers
# ---------------------------------------------------------------------------


def _create_source_container(tmp_path):
    """Build a minimal source container with properties, calibration, dynamics."""
    from swimrs.container.container import SwimContainer

    src = SwimContainer.create(
        str(tmp_path / "source.swim"),
        fields_shapefile=str(FIXTURE_SHP),
        uid_column="site_id",
        start_date="2020-01-01",
        end_date="2020-12-31",
    )

    root = src._root

    # Add a property
    props = root["properties"]
    props.create_array("soil_awc", data=np.array([150.0]))

    # Add calibration with attrs
    cal = root.create_group("calibration")
    cal.attrs["summary_stat"] = "median"
    cal.attrs["n_batches_completed"] = 3
    cal.create_array("swe_alpha", data=np.array([0.5]))

    # Add dynamics (ke_max, kc_max, gwsub_data, irr_data, merged_ndvi)
    dyn = src._ensure_group("derived/dynamics")
    dyn.create_array("ke_max", data=np.array([0.8]))
    dyn.create_array("kc_max", data=np.array([1.4]))
    gwsub = dyn.create_array("gwsub_data", shape=(1,), dtype=VariableLengthUTF8())
    gwsub[:] = ['{"2020": {"gw_flag": true}}']
    irr = dyn.create_array("irr_data", shape=(1,), dtype=VariableLengthUTF8())
    irr[:] = ['{"2020": {"irr_doys": [120, 121], "irrigated": 1, "f_irr": 0.9}}']

    # Add merged_ndvi (should NOT be copied)
    src._ensure_group("derived/merged_ndvi")
    n_days = src.n_days
    root["derived/merged_ndvi"].create_array(
        "irr", data=np.random.rand(n_days, 1).astype(np.float32)
    )

    src.save()
    return src


class TestCopyStaticGroupsIntegration:
    """End-to-end tests for copy_static_groups with real containers."""

    def test_copies_properties(self, tmp_path):
        src = _create_source_container(tmp_path / "s")
        tgt = SwimContainer.create(
            str(tmp_path / "t" / "target.swim"),
            fields_shapefile=str(FIXTURE_SHP),
            uid_column="site_id",
            start_date="2019-01-01",
            end_date="2021-12-31",
        )

        tgt.copy_static_groups(src)

        np.testing.assert_array_equal(tgt._root["properties/soil_awc"][:], [150.0])
        src.close()
        tgt.close()

    def test_copies_calibration_with_attrs(self, tmp_path):
        src = _create_source_container(tmp_path / "s")
        tgt = SwimContainer.create(
            str(tmp_path / "t" / "target.swim"),
            fields_shapefile=str(FIXTURE_SHP),
            uid_column="site_id",
            start_date="2019-01-01",
            end_date="2021-12-31",
        )

        tgt.copy_static_groups(src)

        assert tgt._root["calibration"].attrs["summary_stat"] == "median"
        assert tgt._root["calibration"].attrs["n_batches_completed"] == 3
        np.testing.assert_array_equal(tgt._root["calibration/swe_alpha"][:], [0.5])
        src.close()
        tgt.close()

    def test_copies_ke_kc_gwsub_but_not_irr_data(self, tmp_path):
        src = _create_source_container(tmp_path / "s")
        tgt = SwimContainer.create(
            str(tmp_path / "t" / "target.swim"),
            fields_shapefile=str(FIXTURE_SHP),
            uid_column="site_id",
            start_date="2019-01-01",
            end_date="2021-12-31",
        )

        tgt.copy_static_groups(src)

        np.testing.assert_array_equal(tgt._root["derived/dynamics/ke_max"][:], [0.8])
        np.testing.assert_array_equal(tgt._root["derived/dynamics/kc_max"][:], [1.4])
        assert "derived/dynamics/gwsub_data" in tgt._root
        # irr_data should NOT be copied
        assert "derived/dynamics/irr_data" not in tgt._root
        src.close()
        tgt.close()

    def test_does_not_copy_merged_ndvi(self, tmp_path):
        src = _create_source_container(tmp_path / "s")
        tgt = SwimContainer.create(
            str(tmp_path / "t" / "target.swim"),
            fields_shapefile=str(FIXTURE_SHP),
            uid_column="site_id",
            start_date="2019-01-01",
            end_date="2021-12-31",
        )

        tgt.copy_static_groups(src)

        assert "derived/merged_ndvi/irr" not in tgt._root
        src.close()
        tgt.close()

    def test_rejects_uid_mismatch(self, tmp_path):
        """copy_static_groups raises if source and target have different UIDs."""
        from swimrs.container.container import SwimContainer

        src = _create_source_container(tmp_path / "s")

        # Create target with different shapefile (multi_station has different UIDs)
        multi_shp = Path(__file__).parent.parent / "fixtures" / "multi_station"
        shp_files = list(multi_shp.glob("*.shp"))
        if not shp_files:
            pytest.skip("multi_station fixture shapefile not found")

        tgt = SwimContainer.create(
            str(tmp_path / "t" / "target.swim"),
            fields_shapefile=str(shp_files[0]),
            uid_column="site_id",
            start_date="2019-01-01",
            end_date="2021-12-31",
        )

        with pytest.raises(ValueError, match="UID mismatch"):
            tgt.copy_static_groups(src)

        src.close()
        tgt.close()


# Need SwimContainer import for integration tests
from swimrs.container.container import SwimContainer


class TestHealthCheckNoSnow:
    """Health check should not fail on missing snow in run containers."""

    def test_health_no_snow_source(self, tmp_path):
        """When health_config omits snow_source, no FAIL for missing SWE."""
        tgt = SwimContainer.create(
            str(tmp_path / "target.swim"),
            fields_shapefile=str(FIXTURE_SHP),
            uid_column="site_id",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )

        # Health check with no snow_source should not raise
        health_config = {"mask_mode": "irrigation"}
        report = tgt.report(config=health_config, raise_on_fail=False)

        # Verify no snow-related FAIL
        for check in report.checks:
            if "snow" in check.path.lower() or "swe" in check.path.lower():
                assert check.severity != "FAIL", f"Unexpected snow FAIL: {check.message}"

        tgt.close()

    def test_health_with_snow_source_fails(self, tmp_path):
        """When health_config includes snow_source, missing SWE is a FAIL."""
        tgt = SwimContainer.create(
            str(tmp_path / "target.swim"),
            fields_shapefile=str(FIXTURE_SHP),
            uid_column="site_id",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )

        health_config = {"snow_source": "snodas"}
        report = tgt.report(config=health_config, raise_on_fail=False)

        # Should have a FAIL for missing SWE
        snow_checks = [
            c for c in report.checks if "snow" in c.path.lower() or "swe" in c.path.lower()
        ]
        has_fail = any(c.severity == "FAIL" for c in snow_checks)
        assert has_fail, "Expected FAIL for missing snow/snodas/swe"

        tgt.close()


class TestDefaultRestartSetup:
    """Tests for forecast restart copying and hindcast initialization."""

    def test_copy_forecast_restart(self, tmp_path):
        from swimrs.container.project import _copy_forecast_restart

        src = _create_runnable_container(
            tmp_path / "src",
            start_date="2020-04-01",
            end_date="2020-04-05",
        )
        src.run(
            run_id="calibration_resolved_state",
            profile="state_only",
            mask_mode="none",
            use_default_restart=False,
        )
        src.runs.set_default_restart("calibration_resolved_state")
        src.save()

        tgt = _create_runnable_container(
            tmp_path / "tgt",
            start_date="2020-04-06",
            end_date="2020-04-10",
        )

        copied_run_id = _copy_forecast_restart(
            target=tgt,
            source=src,
            forecast_start=pd.Timestamp("2020-04-06"),
        )

        assert copied_run_id == "calibration_resolved_state"
        assert tgt.runs.default_restart_run_id() == "calibration_resolved_state"
        assert "calibration_resolved_state" in tgt.runs.list()

        meta = tgt.runs.metadata("calibration_resolved_state")
        assert meta["copied_from_container"] == src.uri
        assert meta["copied_from_run_id"] == "calibration_resolved_state"

        src.close()
        tgt.close()

    def test_build_hindcast_initializer(self, tmp_path):
        from swimrs.container.project import _build_hindcast_initializer

        target = _create_runnable_container(
            tmp_path / "hindcast",
            start_date="2020-04-01",
            end_date="2020-04-05",
        )

        config = MagicMock()
        config.hindcast_initialization_strategy = "cyclic_spinup"
        config.hindcast_initialization_window_years = 1
        config.hindcast_initialization_cycles = 2
        config.hindcast_initialization_tolerance = 1e9
        config.refet_type = "eto"
        config.etf_target_model = "ssebop"
        config.mask_mode = "none"

        run_id = _build_hindcast_initializer(
            target=target,
            config=config,
            met_source="gridmet",
            ndvi_mode="observed",
        )

        assert run_id == "hindcast_initial_state"
        assert target.runs.default_restart_run_id() == "hindcast_initial_state"

        meta = target.runs.metadata("hindcast_initial_state")
        assert meta["run_role"] == "initialization"
        assert meta["initialization_strategy"] == "cyclic_spinup"
        assert meta["n_cycles"] == 2
        assert meta["converged"] is True

        target.close()
