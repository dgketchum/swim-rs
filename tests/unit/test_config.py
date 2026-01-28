"""Tests for swimrs.swim.config module."""

import os

import pandas as pd
import pytest

from swimrs.swim.config import ProjectConfig


class TestProjectConfigInit:
    """Tests for ProjectConfig initialization."""

    def test_init_creates_instance(self):
        """ProjectConfig initializes with default None values."""
        config = ProjectConfig()
        assert config.project_name is None
        assert config.root_path is None
        assert config.fields_shapefile is None
        assert config.start_dt is None
        assert config.end_dt is None
        assert config.calibrate is None
        assert config.forecast is None

    def test_init_resolved_config_empty_dict(self):
        """ProjectConfig initializes with empty resolved_config dict."""
        config = ProjectConfig()
        assert config.resolved_config == {}


class TestProjectConfigReadConfig:
    """Tests for ProjectConfig.read_config method."""

    @pytest.fixture
    def minimal_toml(self, tmp_path):
        """Create a minimal valid TOML config file."""
        gis_dir = tmp_path / "gis"
        gis_dir.mkdir()
        shapefile = gis_dir / "fields.shp"
        shapefile.touch()

        toml_content = f"""
project = "test_project"
root = "{tmp_path}"

[paths]
fields_shapefile = "{shapefile}"

[ids]
feature_id = "FID"

[date_range]
start_date = "2020-01-01"
end_date = "2020-12-31"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)
        return toml_file

    def test_read_config_loads_project_name(self, minimal_toml):
        """read_config loads project name from TOML."""
        config = ProjectConfig()
        config.read_config(str(minimal_toml))
        assert config.project_name == "test_project"

    def test_read_config_loads_dates(self, minimal_toml):
        """read_config parses start and end dates."""
        config = ProjectConfig()
        config.read_config(str(minimal_toml))
        assert config.start_dt == pd.Timestamp("2020-01-01")
        assert config.end_dt == pd.Timestamp("2020-12-31")

    def test_read_config_loads_fields_shapefile(self, minimal_toml, tmp_path):
        """read_config loads fields shapefile path."""
        config = ProjectConfig()
        config.read_config(str(minimal_toml))
        assert "fields.shp" in config.fields_shapefile

    def test_read_config_sets_calibrate_false_by_default(self, minimal_toml):
        """read_config sets calibrate=False when not specified."""
        config = ProjectConfig()
        config.read_config(str(minimal_toml))
        assert config.calibrate is False

    def test_read_config_sets_forecast_false_by_default(self, minimal_toml):
        """read_config sets forecast=False when not specified."""
        config = ProjectConfig()
        config.read_config(str(minimal_toml))
        assert config.forecast is False

    def test_read_config_missing_fields_shapefile_raises(self, tmp_path):
        """read_config raises ValueError when fields_shapefile is missing."""
        toml_content = """
project = "test_project"

[ids]
feature_id = "FID"

[date_range]
start_date = "2020-01-01"
end_date = "2020-12-31"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        config = ProjectConfig()
        with pytest.raises(ValueError, match="fields_shapefile"):
            config.read_config(str(toml_file))

    def test_read_config_missing_feature_id_raises(self, tmp_path):
        """read_config raises ValueError when feature_id is missing."""
        gis_dir = tmp_path / "gis"
        gis_dir.mkdir()
        shapefile = gis_dir / "fields.shp"
        shapefile.touch()

        toml_content = f"""
project = "test_project"

[paths]
fields_shapefile = "{shapefile}"

[date_range]
start_date = "2020-01-01"
end_date = "2020-12-31"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        config = ProjectConfig()
        with pytest.raises(ValueError, match="feature_id"):
            config.read_config(str(toml_file))

    def test_read_config_missing_dates_raises(self, tmp_path):
        """read_config raises ValueError when dates are missing."""
        gis_dir = tmp_path / "gis"
        gis_dir.mkdir()
        shapefile = gis_dir / "fields.shp"
        shapefile.touch()

        toml_content = f"""
project = "test_project"

[paths]
fields_shapefile = "{shapefile}"

[ids]
feature_id = "FID"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        config = ProjectConfig()
        with pytest.raises(ValueError, match="start_date"):
            config.read_config(str(toml_file))

    def test_read_config_with_project_root_override(self, tmp_path):
        """read_config respects project_root_override parameter."""
        override_path = tmp_path / "override"
        override_path.mkdir()
        gis_dir = override_path / "gis"
        gis_dir.mkdir()
        shapefile = gis_dir / "fields.shp"
        shapefile.touch()

        toml_content = f"""
project = "test_project"
root = "/some/other/path"

[paths]
fields_shapefile = "{shapefile}"

[ids]
feature_id = "FID"

[date_range]
start_date = "2020-01-01"
end_date = "2020-12-31"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        config = ProjectConfig()
        config.read_config(str(toml_file), project_root_override=str(override_path))
        assert config.root_path == str(override_path)


class TestProjectConfigPathResolution:
    """Tests for path template resolution."""

    def test_resolve_paths_expands_root_variable(self):
        """_resolve_paths expands {root} in paths."""
        raw_config = {"paths": {"data": "{root}/data"}}
        base_vars = {"root": "/home/user/project", "project": "test"}

        result = ProjectConfig._resolve_paths(raw_config, base_vars)
        assert result["paths"]["data"] == "/home/user/project/data"

    def test_resolve_paths_expands_project_variable(self):
        """_resolve_paths expands {project} in paths."""
        raw_config = {"paths": {"output": "/data/{project}/output"}}
        base_vars = {"root": "/home/user", "project": "my_project"}

        result = ProjectConfig._resolve_paths(raw_config, base_vars)
        assert result["paths"]["output"] == "/data/my_project/output"

    def test_resolve_paths_expands_nested_templates(self):
        """_resolve_paths expands nested path templates."""
        raw_config = {"paths": {"data": "{root}/data", "gis": "{data}/gis"}}
        base_vars = {"root": "/project", "project": "test"}

        result = ProjectConfig._resolve_paths(raw_config, base_vars)
        assert result["paths"]["data"] == "/project/data"
        # After resolution, {data} should be available
        assert result["paths"]["gis"] == "/project/data/gis"

    def test_resolve_paths_expands_tilde(self):
        """_resolve_paths expands ~ in paths."""
        raw_config = {"paths": {"home_data": "~/data"}}
        base_vars = {"root": "/root", "project": "test"}

        result = ProjectConfig._resolve_paths(raw_config, base_vars)
        assert result["paths"]["home_data"].startswith(os.path.expanduser("~"))


class TestProjectConfigStr:
    """Tests for ProjectConfig.__str__ method."""

    def test_str_returns_formatted_string(self):
        """__str__ returns a formatted configuration summary."""
        config = ProjectConfig()
        config.project_name = "test_project"
        config.root_path = "/path/to/project"
        config.start_dt = pd.Timestamp("2020-01-01")
        config.end_dt = pd.Timestamp("2020-12-31")
        config.calibrate = False
        config.forecast = True

        result = str(config)

        assert "ProjectConfig:" in result
        assert "test_project" in result
        assert "2020-01-01" in result
        assert "Calibrate Mode: False" in result
        assert "Forecast Mode: True" in result


class TestProjectConfigSyncFromBucket:
    """Tests for ProjectConfig.sync_from_bucket method."""

    def test_sync_from_bucket_raises_without_bucket(self):
        """sync_from_bucket raises ValueError when bucket is not configured."""
        config = ProjectConfig()
        config.data_dir = "/some/path"
        config.project_name = "test"

        with pytest.raises(ValueError, match="ee_bucket not configured"):
            config.sync_from_bucket()

    def test_sync_from_bucket_raises_without_data_dir(self):
        """sync_from_bucket raises ValueError when data_dir is not configured."""
        config = ProjectConfig()
        config.ee_bucket = "my-bucket"
        config.project_name = "test"

        with pytest.raises(ValueError, match="data_dir not configured"):
            config.sync_from_bucket()

    def test_sync_from_bucket_raises_without_project_name(self):
        """sync_from_bucket raises ValueError when project_name is not configured."""
        config = ProjectConfig()
        config.ee_bucket = "my-bucket"
        config.data_dir = "/some/path"

        with pytest.raises(ValueError, match="project_name not configured"):
            config.sync_from_bucket()


class TestProjectConfigDataSources:
    """Tests for data source configuration."""

    @pytest.fixture
    def config_with_data_sources(self, tmp_path):
        """Create a config with data sources section."""
        gis_dir = tmp_path / "gis"
        gis_dir.mkdir()
        shapefile = gis_dir / "fields.shp"
        shapefile.touch()

        toml_content = f"""
project = "test_project"
root = "{tmp_path}"

[paths]
fields_shapefile = "{shapefile}"

[ids]
feature_id = "FID"

[date_range]
start_date = "2020-01-01"
end_date = "2020-12-31"

[data_sources]
met_source = "era5"
snow_source = "era5"
soil_source = "hwsd"
mask_mode = "none"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)
        return toml_file

    def test_read_config_loads_data_sources(self, config_with_data_sources):
        """read_config loads data source settings."""
        config = ProjectConfig()
        config.read_config(str(config_with_data_sources))

        assert config.met_source == "era5"
        assert config.snow_source == "era5"
        assert config.soil_source == "hwsd"
        assert config.mask_mode == "none"

    def test_read_config_defaults_data_sources(self, tmp_path):
        """read_config uses default data sources when not specified."""
        gis_dir = tmp_path / "gis"
        gis_dir.mkdir()
        shapefile = gis_dir / "fields.shp"
        shapefile.touch()

        toml_content = f"""
project = "test_project"
root = "{tmp_path}"

[paths]
fields_shapefile = "{shapefile}"

[ids]
feature_id = "FID"

[date_range]
start_date = "2020-01-01"
end_date = "2020-12-31"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        config = ProjectConfig()
        config.read_config(str(toml_file))

        assert config.met_source == "gridmet"
        assert config.snow_source == "snodas"
        assert config.soil_source == "ssurgo"
        assert config.mask_mode == "irrigation"


class TestProjectConfigMisc:
    """Tests for miscellaneous configuration options."""

    @pytest.fixture
    def config_with_misc(self, tmp_path):
        """Create a config with misc section."""
        gis_dir = tmp_path / "gis"
        gis_dir.mkdir()
        shapefile = gis_dir / "fields.shp"
        shapefile.touch()

        toml_content = f"""
project = "test_project"
root = "{tmp_path}"

[paths]
fields_shapefile = "{shapefile}"

[ids]
feature_id = "FID"

[date_range]
start_date = "2020-01-01"
end_date = "2020-12-31"

[misc]
runoff_process = "ier"
refet_type = "etr"
irrigation_threshold = 0.5
elev_units = "ft"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)
        return toml_file

    def test_read_config_loads_misc_settings(self, config_with_misc):
        """read_config loads misc settings."""
        config = ProjectConfig()
        config.read_config(str(config_with_misc))

        assert config.runoff_process == "ier"
        assert config.refet_type == "etr"
        assert config.irrigation_threshold == 0.5
        assert config.elev_units == "ft"

    def test_read_config_defaults_runoff_process_to_cn(self, tmp_path):
        """read_config defaults runoff_process to 'cn'."""
        gis_dir = tmp_path / "gis"
        gis_dir.mkdir()
        shapefile = gis_dir / "fields.shp"
        shapefile.touch()

        toml_content = f"""
project = "test_project"
root = "{tmp_path}"

[paths]
fields_shapefile = "{shapefile}"

[ids]
feature_id = "FID"

[date_range]
start_date = "2020-01-01"
end_date = "2020-12-31"
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        config = ProjectConfig()
        config.read_config(str(toml_file))

        assert config.runoff_process == "cn"
