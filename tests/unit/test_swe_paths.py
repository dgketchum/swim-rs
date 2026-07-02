"""A3 regression tests: unified SWE path resolution.

Three call sites (obs exporter, PestBuilder, model input builder) previously
kept divergent SWE path lists. Example 6 stores SWE at meteorology/era5/swe,
which the exporter and input builder missed — the exporter wrote all-NaN SWE
obs files and every SWE observation was silently zero-weighted, so snow
parameters were never calibrated.

These tests lock the contract that all three sites resolve SWE through
schema.find_swe_path / schema.SWE_PATHS.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from swimrs.container.schema import SWE_PATHS, find_swe_path


class TestFindSwePath:
    def test_empty_root_returns_none(self):
        assert find_swe_path({}) is None

    @pytest.mark.parametrize("path", SWE_PATHS)
    def test_finds_each_canonical_path(self, path):
        root = {path: object()}
        assert find_swe_path(root) == path

    def test_snodas_has_priority(self):
        root = {p: object() for p in SWE_PATHS}
        assert find_swe_path(root) == "snow/snodas/swe"

    def test_era5_met_path_present(self):
        """meteorology/era5/swe (Ex6 layout) must be in the canonical list."""
        assert "meteorology/era5/swe" in SWE_PATHS


class TestNoListDrift:
    """Guard against the three call sites re-growing private path lists."""

    def test_exporter_uses_shared_helper(self):
        from swimrs.container.components import exporter

        assert exporter.find_swe_path is find_swe_path

    def test_pest_builder_uses_shared_helper(self):
        from swimrs.calibrate import pest_builder

        assert pest_builder.find_swe_path is find_swe_path
        assert pest_builder.SWE_PATHS is SWE_PATHS


class TestPestBuilderSweData:
    """PestBuilder._get_swe_data must find SWE at any canonical path."""

    def _make_builder(self, root, df):
        from swimrs.calibrate.pest_builder import PestBuilder

        builder = PestBuilder.__new__(PestBuilder)
        container = MagicMock()
        container.state.root = root
        container.query.dataframe.return_value = df
        builder._container = container
        builder.config = MagicMock(start_dt=None, end_dt=None)
        return builder, container

    def test_finds_era5_met_swe(self):
        dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
        df = pd.DataFrame({"siteA": np.arange(10.0)}, index=dates)
        builder, container = self._make_builder({"meteorology/era5/swe": object()}, df)

        result = builder._get_swe_data("siteA")

        container.query.dataframe.assert_called_once_with("meteorology/era5/swe", fields=["siteA"])
        assert np.allclose(result["swe"].values, np.arange(10.0))

    def test_raises_when_no_swe(self):
        builder, _ = self._make_builder({}, pd.DataFrame())
        with pytest.raises(ValueError, match="SWE data not found"):
            builder._get_swe_data("siteA")


class TestExporterSweExport:
    """Exporter.observations must export non-NaN SWE from meteorology/era5/swe."""

    def test_era5_met_swe_written(self, tmp_path):
        from swimrs.container.components.exporter import Exporter

        dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
        swe_da = xr.DataArray(
            np.arange(10.0).reshape(10, 1),
            dims=["time", "site"],
            coords={"time": dates, "site": ["siteA"]},
        )

        state = MagicMock()
        state.is_writable = True
        state._mode = "r+"
        state.field_uids = ["siteA"]
        state.get_time_slice.return_value = slice(None)
        state.time_index = dates
        state.root = {"meteorology/era5/swe": object()}

        def get_xarray(path, **kwargs):
            assert path == "meteorology/era5/swe"
            return swe_da

        state.get_xarray.side_effect = get_xarray

        exp = Exporter(state)
        with patch.object(exp, "_get_dynamics_dict", return_value={}):
            exp.observations(tmp_path, etf_model="ssebop", masks=("no_mask",))

        swe_values = np.loadtxt(tmp_path / "obs_swe_siteA.np")
        assert np.isfinite(swe_values).all()
        assert np.allclose(swe_values, np.arange(10.0))


class TestInputSwePathResolution:
    """_get_container_time_series must resolve swe_obs from any canonical path."""

    def test_paths_include_era5_met_swe(self):
        # Exercise just the path-resolution logic: find_swe_path against a
        # root that mimics the Ex6 layout.
        root = {
            "meteorology/era5/prcp": object(),
            "meteorology/era5/eto": object(),
            "meteorology/era5/swe": object(),
        }
        assert find_swe_path(root) == "meteorology/era5/swe"
