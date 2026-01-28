import json
import os
import subprocess

import pandas as pd
import toml


class ProjectConfig:
    """Configuration manager for SWIM-RS projects.

    Loads and resolves project configuration from TOML files, handling path
    resolution, parameter validation, and mode-specific setup for calibration
    and forecasting workflows.

    Attributes:
        project_name: Name of the project from TOML config.
        root_path: Resolved root directory path.
        project_ws: Project workspace directory.
        fields_shapefile: Path to the fields geometry shapefile.
        start_dt: Simulation start date.
        end_dt: Simulation end date.
        calibrate: Whether calibration mode is enabled.
        forecast: Whether forecast mode is enabled.
        met_source: Meteorology data source ('gridmet' or 'era5').
        runoff_process: Runoff method ('cn' for Curve Number, 'ier' for infiltration-excess).

    Example:
        >>> config = ProjectConfig()
        >>> config.read_config("project.toml")
        >>> print(config.project_name)
        'my_project'
        >>> print(config.start_dt)
        Timestamp('2020-01-01 00:00:00')
    """

    def __init__(self):
        """Initialize ProjectConfig with default None values for all attributes."""
        super().__init__()
        # Metadata / resolution
        self.resolved_config = {}
        self.project_name = None
        self.root_path = None
        self.project_dir = None
        self.project_ws = None
        self.conf_file_path = None

        # Paths
        self.data_dir = None
        self.landsat_dir = None
        self.sentinel_dir = None
        self.met_dir = None
        self.gis_dir = None
        self.fields_shapefile = None
        self.gridmet_mapping_shp = None
        self.gridmet_centroids = None
        self.correction_tifs = None
        self.gridmet_factors = None
        self.properties_dir = None
        self.irr_csv = None
        self.ssurgo_csv = None
        self.lulc_csv = None
        self.properties_json = None
        self.snodas_in_dir = None
        self.snodas_out_json = None
        self.dynamics_data_json = None

        # EE
        self.ee_bucket = None

        # IDs
        self.feature_id_col = None
        self.gridmet_mapping_index_col = None
        self.gridmet_id_col = None  # Column name for GridMET cell ID (GFID)
        self.state_col = None

        # Runtime settings
        self.irrigation_threshold = None
        self.irr_threshold = None
        self.elev_units = None
        self.refet_type = None
        self.runoff_process = None  # "cn" | "ier"
        self.start_dt = None
        self.end_dt = None
        self.kc_proxy = None
        self.cover_proxy = None

        # Calibration / Forecast
        self.pest_run_dir = None
        self.etf_target_model = None
        self.etf_ensemble_members = None
        self.workers = None
        self.realizations = None
        self.calibration_dir = None
        self.obs_folder = None
        self.initial_values_csv = None
        self.spinup = None
        self.python_script = None
        self.forecast_parameters_csv = None

        # Data sources (new normalized config)
        self.met_source = None  # "gridmet" | "era5"
        self.snow_source = None  # "snodas" | "era5" | None
        self.soil_source = None  # "ssurgo" | "hwsd"
        self.mask_mode = None  # "irrigation" | "none"
        self.bucket_uri = None  # computed gs://{bucket}/{project}/

        # ERA5 config
        self.era5_params = None
        self.era5_param_mapping = None
        self.era5_extracts_dir = None

        # HWSD (international soils)
        self.hwsd_csv = None

        # Container path
        self.container_path = None

        # Ecostress
        self.ecostress_dir = None

        # Derived / mode flags
        self.calibrate = None
        self.forecast = None
        self.input_data = None
        self.plot_timeseries = None
        self.calibration_dir_override = None
        self.parameter_set_json = None
        self.parameter_list = None
        self.forecast_param_csv = None
        self.forecast_parameters = None
        self.forecast_parameter_groups = None

    def read_config(
        self,
        conf_file_path: str,
        project_root_override: str | None = None,
        calibrate: bool = False,
        forecast: bool = False,
        calibration_dir_override: str | None = None,
        parameter_set_json: str | None = None,
        forecast_param_csv: str | None = None,
    ) -> None:
        """Load and parse a TOML configuration file.

        Reads the configuration file, resolves path templates (e.g., {root}, {project}),
        validates required fields, and sets up mode-specific parameters for calibration
        or forecasting.

        Args:
            conf_file_path: Path to the TOML configuration file.
            project_root_override: Override the root path from the TOML file.
            calibrate: Enable calibration mode.
            forecast: Enable forecast mode.
            calibration_dir_override: Override the calibration directory path.
            parameter_set_json: Path to JSON file with parameter values.
            forecast_param_csv: Path to CSV file with forecast parameters.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If the configuration file doesn't exist.

        Example:
            >>> config = ProjectConfig()
            >>> config.read_config("project.toml", calibrate=True)
        """
        with open(conf_file_path) as f:
            raw_config = toml.load(f)

        # Store the config file path for later use (e.g., copying to pest directory)
        self.config_path = os.path.abspath(conf_file_path)

        self.calibrate = calibrate
        self.forecast = forecast
        self.project_name = raw_config.get("project")
        toml_root_path = raw_config.get("root")

        if project_root_override:
            self.root_path = os.path.expanduser(project_root_override)
        else:
            expanded_root = (
                os.path.expanduser(toml_root_path) if toml_root_path is not None else None
            )
            if expanded_root and not os.path.isabs(expanded_root):
                conf_dir = os.path.dirname(os.path.abspath(conf_file_path))
                self.root_path = os.path.normpath(os.path.join(conf_dir, expanded_root))
            else:
                self.root_path = expanded_root

        base_format_vars = {
            "root": self.root_path,
            "project": self.project_name,
        }

        self.resolved_config = self._resolve_paths(raw_config, base_format_vars)

        paths_conf = self.resolved_config.get("paths", {})
        ee_conf = self.resolved_config.get("earth_engine", {})
        ids_conf = self.resolved_config.get("ids", {})
        misc_conf = self.resolved_config.get("misc", {})
        date_range_conf = self.resolved_config.get("date_range", {})
        crop_coeff_conf = self.resolved_config.get("crop_coefficient", {})
        calib_toml_conf = self.resolved_config.get("calibration", {})
        forecast_toml_conf = self.resolved_config.get("forecast", {})
        era5land_conf = self.resolved_config.get("era5land", {})
        data_sources_conf = self.resolved_config.get("data_sources", {})

        # Nested sections (CONUS-specific and ERA5-specific)
        paths_conus_conf = paths_conf.get("conus", {})
        paths_era5_conf = paths_conf.get("era5", {})
        ids_conus_conf = ids_conf.get("conus", {})

        # Basic paths
        self.project_dir = paths_conf.get("project")
        self.data_dir = paths_conf.get("data")
        self.project_ws = paths_conf.get("project_workspace")
        self.conf_file_path = conf_file_path

        # Data roots
        self.landsat_dir = paths_conf.get("landsat")
        self.sentinel_dir = paths_conf.get("sentinel")
        # Met dir: check nested sections first, then flat
        self.met_dir = (
            paths_conus_conf.get("met") or paths_era5_conf.get("met") or paths_conf.get("met")
        )
        # ERA5 extracts dir (for international)
        self.era5_extracts_dir = paths_era5_conf.get("extracts")
        self.gis_dir = paths_conf.get("gis")

        # Field geometry and factors
        self.fields_shapefile = paths_conf.get("fields_shapefile")
        # Read from nested [paths.conus] with fallback to flat
        self.gridmet_mapping_shp = paths_conus_conf.get("gridmet_mapping") or paths_conf.get(
            "gridmet_mapping"
        )
        self.gridmet_centroids = paths_conus_conf.get("gridmet_centroids") or paths_conf.get(
            "gridmet_centroids"
        )
        self.correction_tifs = paths_conus_conf.get("correction_tifs") or paths_conf.get(
            "correction_tifs"
        )
        self.gridmet_factors = paths_conus_conf.get("gridmet_factors") or paths_conf.get(
            "gridmet_factors"
        )

        # Required field shapefile
        if not self.fields_shapefile:
            raise ValueError("Missing required paths.fields_shapefile in config TOML")

        # Properties, SNODAS, timeseries
        self.properties_dir = paths_conf.get("properties")
        # Read from nested [paths.conus] with fallback
        self.irr_csv = paths_conus_conf.get("irr") or paths_conf.get("irr")
        self.ssurgo_csv = paths_conus_conf.get("ssurgo") or paths_conf.get("ssurgo")
        self.lulc_csv = paths_conf.get("lulc")
        self.properties_json = paths_conf.get("properties_json")
        # HWSD for international
        self.hwsd_csv = paths_conf.get("hwsd")
        # SNODAS from nested [paths.conus]
        self.snodas_in_dir = paths_conus_conf.get("snodas_in") or paths_conf.get("snodas_in")
        self.snodas_out_json = paths_conus_conf.get("snodas_out") or paths_conf.get("snodas_out")
        self.dynamics_data_json = paths_conf.get("dynamics_data")

        # Earth Engine
        self.ee_bucket = ee_conf.get("bucket")
        # Compute bucket_uri
        if self.ee_bucket and self.project_name:
            self.bucket_uri = f"gs://{self.ee_bucket}/{self.project_name}/"

        # Data sources (new normalized config)
        self.met_source = data_sources_conf.get("met_source", "gridmet")
        self.snow_source = data_sources_conf.get("snow_source", "snodas")
        self.soil_source = data_sources_conf.get("soil_source", "ssurgo")
        self.mask_mode = data_sources_conf.get("mask_mode", "irrigation")

        # IDs
        self.feature_id_col = ids_conf.get("feature_id")
        # Read from nested [ids.conus] if available, else fall back to flat
        self.gridmet_mapping_index_col = ids_conus_conf.get("gridmet_join_id") or ids_conf.get(
            "gridmet_join_id"
        )
        self.gridmet_id_col = ids_conus_conf.get("gridmet_id") or ids_conf.get("gridmet_id", "GFID")
        self.state_col = ids_conus_conf.get("state_col") or ids_conf.get("state_col")

        # ERA5-Land config
        self.era5_params = era5land_conf.get("params", [])
        self.era5_param_mapping = era5land_conf.get("param_mapping", {})

        # Container path (for SwimContainer workflow)
        self.container_path = paths_conf.get("container")

        # Ecostress (international)
        self.ecostress_dir = paths_conf.get("ecostress")

        # Model settings
        self.irrigation_threshold = misc_conf.get("irrigation_threshold")
        self.irr_threshold = self.irrigation_threshold
        self.elev_units = misc_conf.get("elev_units", "m")
        self.refet_type = misc_conf.get("refet_type")
        # Runoff process selection: "cn" (Curve Number) or "ier" (infiltration-excess)
        self.runoff_process = misc_conf.get("runoff_process", "cn")

        # Dates
        sdt_str = date_range_conf.get("start_date")
        edt_str = date_range_conf.get("end_date")
        if sdt_str:
            self.start_dt = pd.to_datetime(sdt_str)
        if edt_str:
            self.end_dt = pd.to_datetime(edt_str)

        # Crop coefficients
        self.kc_proxy = crop_coeff_conf.get("kc_proxy")
        self.cover_proxy = crop_coeff_conf.get("cover_proxy")

        if self.project_ws:
            os.makedirs(self.project_ws, exist_ok=True)

        # Calibration
        self.pest_run_dir = calib_toml_conf.get("pest_run_dir")
        self.etf_target_model = calib_toml_conf.get("etf_target_model")
        self.etf_ensemble_members = calib_toml_conf.get("etf_ensemble_members")
        self.workers = calib_toml_conf.get("workers")
        self.realizations = calib_toml_conf.get("realizations")
        self.obs_folder = calib_toml_conf.get("obs_folder")
        self.calibration_dir = calib_toml_conf.get("calibration_dir")
        self.initial_values_csv = calib_toml_conf.get("initial_values_csv")
        self.spinup = calib_toml_conf.get("spinup")
        self.python_script = calib_toml_conf.get("python_script")

        # Forecast
        self.forecast_parameters_csv = forecast_toml_conf.get("forecast_parameters")

        # Overrides
        self.calibration_dir_override = calibration_dir_override
        self.parameter_set_json = parameter_set_json
        self.forecast_param_csv = forecast_param_csv

        if self.calibrate or calibration_dir_override:
            self.calibration_dir_override = calibration_dir_override
            self.read_calibration_parameters()

        # Validate required keys (global)
        missing = []
        if not self.fields_shapefile:
            missing.append("paths.fields_shapefile")
        if not self.feature_id_col:
            missing.append("ids.feature_id")
        if self.start_dt is None:
            missing.append("date_range.start_date")
        if self.end_dt is None:
            missing.append("date_range.end_date")
        if missing:
            raise ValueError("Missing required config keys: " + ", ".join(missing))

        # Mode-specific validation
        if self.calibrate:
            cal_missing = []
            if not self.pest_run_dir:
                cal_missing.append("calibration.pest_run_dir")
            if not self.calibration_dir:
                cal_missing.append("calibration.calibration_dir")
            if not self.initial_values_csv:
                cal_missing.append("calibration.initial_values_csv")
            if not self.etf_target_model:
                cal_missing.append("calibration.etf_target_model")
            if cal_missing:
                raise ValueError("Calibration config missing: " + ", ".join(cal_missing))
            self.read_calibration_parameters()

        if self.forecast:
            self.calibration_dir = None
            # Accept either forecast_parameters_csv or parameter_set_json
            if not (
                self.forecast_parameters_csv or self.parameter_set_json or self.forecast_param_csv
            ):
                raise ValueError(
                    "Forecast config missing: forecast.forecast_parameters (CSV) or parameter_set_json"
                )
            self.read_forecast_parameters()

    def read_calibration_parameters(self, sites: list[str] | None = None) -> None:
        """Load calibration parameter files from the calibration directory.

        Reads the initial values CSV and sets up the mapping from parameter
        names to their multiplier files in the calibration directory.

        Args:
            sites: Optional list of site IDs to filter parameters. If provided,
                only parameters containing these site IDs will be loaded.

        Raises:
            FileNotFoundError: If the initial values CSV doesn't exist.
        """
        self.calibrate = True

        if self.calibration_dir_override:
            self.calibration_dir = self.calibration_dir_override

        initial_values_csv_path = self.initial_values_csv
        if not os.path.isabs(initial_values_csv_path) and self.project_ws:
            initial_values_csv_path = os.path.join(self.project_ws, initial_values_csv_path)
        param_init = pd.read_csv(initial_values_csv_path, index_col=0)
        if sites:
            applicable_params = []
            for site in sites:
                idx = [i for i in param_init.index if site in i]
                applicable_params.extend(idx)
            param_init = param_init.loc[applicable_params]

        self.calibrated_parameters = param_init.index
        _files = list(param_init["mult_name"])
        self.calibration_files = {
            k: os.path.join(self.calibration_dir, f)
            for k, f in zip(self.calibrated_parameters, _files)
        }

    def read_forecast_parameters(self) -> None:
        """Load forecast parameters from CSV or JSON file.

        Reads parameter distributions from forecast_param_csv, forecast_parameters_csv,
        or parameter_set_json and computes mean values for each parameter.

        Sets:
            forecast_parameters: pandas Series of mean parameter values.
            parameter_list: List of parameter names.
            forecast_parameter_groups: Parameter groupings (if loaded from JSON).

        Raises:
            ValueError: If no forecast parameter source is configured.
        """
        self.calibration_dir = None

        if self.forecast_param_csv:
            parameter_dist_csv = self.forecast_param_csv

        elif self.forecast_parameters_csv:
            parameter_dist_csv = self.forecast_parameters_csv
            if not os.path.isabs(parameter_dist_csv) and self.project_ws:
                parameter_dist_csv = os.path.join(self.project_ws, parameter_dist_csv)
        else:
            parameter_dist_csv = None
        if parameter_dist_csv:
            param_dist = pd.read_csv(parameter_dist_csv, index_col=0)
            param_mean = param_dist.mean(axis=0)
            p_str = [
                "_".join(s.split(":")[1].split("_")[1:-1])
                if ":" in s and len(s.split(":")) > 1
                else s
                for s in list(param_mean.index)
            ]
            param_mean.index = p_str
            self.forecast_parameters = param_mean.copy()
            self.parameter_list = param_mean.index.to_list()

        elif self.parameter_set_json:
            with open(self.parameter_set_json) as f:
                param_arr = json.load(f)
            d = param_arr["fields"]
            self.forecast_parameter_groups = [list(v.keys()) for k, v in d.items()][0]
            k_list = []
            for main_key, val_dict in d.items():
                for sub_key in val_dict.keys():
                    k_list.append(f"{sub_key}_{main_key}")
            v_list = []
            for tup in [(i.split("_")[0], "_".join(i.split("_")[1:])) for i in k_list]:
                v_list.append(d[tup[1]][tup[0]])
            self.forecast_parameters = pd.Series(index=k_list, data=v_list)
            self.parameter_list = self.forecast_parameters.index.to_list()

    def __str__(self) -> str:
        return (
            f"ProjectConfig:\n"
            f"  Project Name: {self.project_name}\n"
            f"  Root Path: {self.root_path}\n"
            f"  Project Workspace: {self.project_ws}\n"
            f"  Data Directory: {self.data_dir}\n"
            f"  Fields Shapefile: {self.fields_shapefile}\n"
            f"  Start Date: {self.start_dt}\n"
            f"  End Date: {self.end_dt}\n"
            f"  Calibrate Mode: {self.calibrate}\n"
            f"  Forecast Mode: {self.forecast}"
        )

    def sync_from_bucket(
        self, dry_run: bool = False, subdirs: list[str] | None = None
    ) -> subprocess.CompletedProcess | list[subprocess.CompletedProcess]:
        """Sync Earth Engine exports from GCS bucket to local filesystem.

        Uses gsutil rsync to mirror the bucket structure to local data directory.
        Bucket structure: gs://{bucket}/{project}/remote_sensing/...
        Local structure:  {data_dir}/remote_sensing/...

        Parameters
        ----------
        dry_run : bool, optional
            If True, show what would be synced without making changes.
        subdirs : list of str, optional
            Specific subdirectories to sync (e.g., ['remote_sensing', 'properties']).
            If None, syncs the entire bucket prefix.

        Returns
        -------
        subprocess.CompletedProcess
            Result of the gsutil command.

        Raises
        ------
        ValueError
            If bucket or data_dir is not configured.
        subprocess.CalledProcessError
            If gsutil command fails.

        Example
        -------
        >>> cfg = ProjectConfig()
        >>> cfg.read_config("project.toml")
        >>> cfg.sync_from_bucket(dry_run=True)  # Preview
        >>> cfg.sync_from_bucket()  # Actually sync
        >>> cfg.sync_from_bucket(subdirs=['remote_sensing'])  # Sync only remote sensing
        """
        if not self.ee_bucket:
            raise ValueError("ee_bucket not configured in TOML [earth_engine] section")
        if not self.data_dir:
            raise ValueError("data_dir not configured")
        if not self.project_name:
            raise ValueError("project_name not configured")

        results = []

        if subdirs:
            # Sync specific subdirectories
            for subdir in subdirs:
                src = f"gs://{self.ee_bucket}/{self.project_name}/{subdir}/"
                dst = os.path.join(self.data_dir, subdir) + "/"

                os.makedirs(dst, exist_ok=True)

                cmd = ["gsutil", "-m", "rsync", "-r"]
                if dry_run:
                    cmd.append("-n")
                cmd.extend([src, dst])

                print(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True)
                results.append(result)
        else:
            # Sync entire project prefix
            src = f"gs://{self.ee_bucket}/{self.project_name}/"
            dst = self.data_dir + "/"

            os.makedirs(dst, exist_ok=True)

            cmd = ["gsutil", "-m", "rsync", "-r"]
            if dry_run:
                cmd.append("-n")
            cmd.extend([src, dst])

            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            results.append(result)

        return results[0] if len(results) == 1 else results

    @staticmethod
    def _resolve_paths(raw_config: dict, base_format_vars: dict) -> dict:
        config = json.loads(json.dumps(raw_config))
        format_vars = {
            k: (os.path.expanduser(v) if isinstance(v, str) else v)
            for k, v in base_format_vars.items()
        }

        def resolve_dict(d, format_vars, depth=0):
            """Recursively resolve format strings in a dict, returning count of resolved items."""
            if depth > 5:  # Prevent infinite recursion
                return 0
            resolved_count = 0
            for key, value in list(d.items()):
                if isinstance(value, dict):
                    # Recurse into nested dicts
                    resolved_count += resolve_dict(value, format_vars, depth + 1)
                elif isinstance(value, str) and ("{" in value or "}" in value):
                    template = value.replace(" ", "")
                    try:
                        formatted_value = template.format(**format_vars)
                        if formatted_value != d[key]:
                            d[key] = formatted_value
                            resolved_count += 1
                        if key not in format_vars or format_vars[key] != formatted_value:
                            format_vars[key] = formatted_value
                    except KeyError:
                        pass
            return resolved_count

        max_iterations = 10
        for i in range(max_iterations):
            newly_resolved_count = 0
            for section_name, section_content in list(config.items()):
                if isinstance(section_content, dict):
                    newly_resolved_count += resolve_dict(section_content, format_vars)
                elif isinstance(section_content, str) and (
                    "{" in section_content or "}" in section_content
                ):
                    template = section_content.replace("{{", "{").replace("}}", "}")
                    try:
                        formatted_value = template.format(**format_vars)
                        if formatted_value != config[section_name]:
                            config[section_name] = formatted_value
                            newly_resolved_count += 1
                        if (
                            section_name not in format_vars
                            or format_vars[section_name] != formatted_value
                        ):
                            format_vars[section_name] = formatted_value
                    except KeyError:
                        pass

            if newly_resolved_count == 0 and i > 0:
                break

        def expand_tildes(d):
            """Recursively expand ~ in paths."""
            for key, value in d.items():
                if isinstance(value, dict):
                    expand_tildes(value)
                elif isinstance(value, str) and value.startswith("~"):
                    d[key] = os.path.expanduser(value)

        for section_name, section_content in config.items():
            if isinstance(section_content, dict):
                expand_tildes(section_content)
            elif isinstance(section_content, str) and section_content.startswith("~"):
                config[section_name] = os.path.expanduser(section_content)
        return config


if __name__ == "__main__":
    pass
