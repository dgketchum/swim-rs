import os
import json
import toml
import pandas as pd
from pprint import pprint


class ProjectConfig:
    def __init__(self):
        super().__init__()
        self.forecast_parameters = None
        self.forecast_param_csv = None
        self.parameter_set_json = None
        self.parameter_dist_csv = None
        self.calibration_dir_override = None
        self.resolved_config = {}
        self.project_name = None
        self.root_path = None
        self.project_dir = None
        self.data_dir = None
        self.project_ws = None
        self.config_file_in_project = None
        self.landsat_dir = None
        self.landsat_ee_data_dir = None
        self.landsat_extracts_dir = None
        self.landsat_tables_dir = None
        self.landsat_extract_json = None
        self.landsat_etf_pqt = None
        self.landsat_ndvi_pqt = None
        self.sentinel_dir = None
        self.sentinel_extracts_dir = None
        self.sentinel_tables_dir = None
        self.sentinel_ee_data_dir = None
        self.sentinel_ndvi_pqt = None
        self.ecostress_dir = None
        self.ecostress_extracts_json = None
        self.ecostress_tables_dir = None
        self.ecostress_etf_pqt = None
        self.met_dir = None
        self.met_timeseries_dir = None
        self.era5_dir = None
        self.era5_extracts_dir = None
        self.era5_series_dir = None
        self.gis_dir = None
        self.gridmet_mapping_shp = None
        self.footprint_shapefile_shp = None
        self.properties_dir = None
        self.irr_csv = None
        self.ssurgo_csv = None
        self.lulc_csv = None
        self.modis_lulc_csv = None
        self.properties_json = None
        self.static_properties_json = None
        self.snodas_in_dir = None
        self.snodas_out_json = None
        self.remote_sensing_tables_dir = None
        self.rs_tables_dir = None
        self.joined_timeseries_dir = None
        self.dynamics_data_json = None
        self.station_metadata_csv = None
        self.station_file_csv = None
        self.results_dir = None
        self.ee_fields_flux3p = None
        self.ee_fields_flux = None
        self.ee_fields = None
        self.ee_bucket = None
        self.feature_id_col = None
        self.gridmet_mapping_index_col = None
        self.state_col = None
        self.irrigation_threshold = None
        self.workers = None
        self.realizations = None
        self.elev_units = None
        self.refet_type = None
        self.start_dt = None
        self.end_dt = None
        self.kc_proxy = None
        self.cover_proxy = None
        self.toml_calibration_dir = None
        self.toml_initial_values_csv = None
        self.forecast_parameters_csv = None
        self.era5land_params = None
        self.era5land_param_mapping = None
        self.alternative_root_config = {}

        self.correction_tifs = None
        self.gridmet_factors = None
        self.parameter_list = None
        self.spinup = None
        self.forecast_parameter_groups = None
        self.forecast = None

        self.etf_target_model = None
        self.etf_ensemble_members = None

        self.calibrate = None
        self.calibration_dir = None
        self.calibrated_parameters = None
        self.calibration_files = None
        self.obs_folder = None
        self.ts_quantity = int(1)
        self.calibration_groups = None
        self.static_keys = None
        self.initial_keys = None
        self.date_range = None
        self.use_individual_kcb = None
        self.ro_reinf_frac = None
        self.swb_mode = None
        self.rew_ceff = None
        self.evap_ceff = None
        self.winter_evap_limiter = None
        self.winter_end_day = None
        self.winter_start_day = None
        self.data_folder = None
        self.input_data = None
        self.plot_timeseries = None
        self.field_index = None
        self.irr_threshold = None

    def read_config(self, conf_file_path, project_root_override=None, calibrate=False, forecast=False,
                    calibration_dir_override=None, parameter_set_json=None, forecast_param_csv=None):
        with open(conf_file_path, 'r') as f:
            raw_config = toml.load(f)

        self.calibrate = calibrate
        self.forecast = forecast
        self.project_name = raw_config.get('project')
        toml_root_path = raw_config.get('root')

        if project_root_override:
            self.root_path = os.path.expanduser(project_root_override)
        else:
            self.root_path = os.path.expanduser(toml_root_path)

        base_format_vars = {
            'root': self.root_path,
            'project': self.project_name,
        }

        self.resolved_config = self._resolve_paths(raw_config, base_format_vars)

        paths_conf = self.resolved_config.get('paths', {})
        ee_conf = self.resolved_config.get('earth_engine', {})
        ids_conf = self.resolved_config.get('ids', {})
        misc_conf = self.resolved_config.get('misc', {})
        date_range_conf = self.resolved_config.get('date_range', {})
        crop_coeff_conf = self.resolved_config.get('crop_coefficient', {})
        calib_toml_conf = self.resolved_config.get('calibration', {})
        forecast_toml_conf = self.resolved_config.get('forecast', {})
        era5land_conf = self.resolved_config.get('era5land', {})

        self.project_dir = paths_conf.get('project')
        self.data_dir = paths_conf.get('data')
        self.project_ws = paths_conf.get('project_workspace', self.project_dir)
        self.config_file_in_project = paths_conf.get('config_file')
        self.landsat_dir = paths_conf.get('landsat')
        self.landsat_ee_data_dir = paths_conf.get('landsat_ee_data')
        self.landsat_extracts_dir = paths_conf.get('landsat_extracts', self.landsat_ee_data_dir)
        self.landsat_tables_dir = paths_conf.get('landsat_tables')
        self.landsat_extract_json = paths_conf.get('landsat_extract')
        self.landsat_etf_pqt = paths_conf.get('landsat_etf_pqt')
        self.landsat_ndvi_pqt = paths_conf.get('landsat_ndvi')
        self.sentinel_dir = paths_conf.get('sentinel')
        self.sentinel_extracts_dir = paths_conf.get('sentinel_extracts')
        self.sentinel_tables_dir = paths_conf.get('sentinel_tables')
        self.sentinel_ee_data_dir = paths_conf.get('sentinel_ee_data')
        self.sentinel_ndvi_pqt = paths_conf.get('sentinel_ndvi')
        self.ecostress_dir = paths_conf.get('ecostress')
        self.ecostress_extracts_json = paths_conf.get('ecostress_extracts')
        self.ecostress_tables_dir = paths_conf.get('ecostress_tables')
        self.ecostress_etf_pqt = paths_conf.get('ecostress_etf_pqt')
        self.met_dir = paths_conf.get('met')
        self.met_timeseries_dir = paths_conf.get('met_timeseries', self.met_dir)
        self.era5_dir = paths_conf.get('era5', self.met_timeseries_dir)
        self.era5_extracts_dir = paths_conf.get('era5_extracts')
        self.era5_series_dir = paths_conf.get('era5_series')
        self.gis_dir = paths_conf.get('gis')

        self.gridmet_mapping_shp = paths_conf.get('gridmet_mapping')
        self.correction_tifs = paths_conf.get('correction_tifs')
        self.gridmet_factors = paths_conf.get('gridmet_factors')

        self.footprint_shapefile_shp = paths_conf.get('footprint_shapefile')
        self.properties_dir = paths_conf.get('properties')
        self.irr_csv = paths_conf.get('irr')
        self.ssurgo_csv = paths_conf.get('ssurgo')
        self.lulc_csv = paths_conf.get('lulc')
        self.modis_lulc_csv = paths_conf.get('modis_lulc', self.lulc_csv)
        self.properties_json = paths_conf.get('properties_json')
        self.static_properties_json = paths_conf.get('static_properties', self.properties_json)
        self.snodas_in_dir = paths_conf.get('snodas_in')
        self.snodas_out_json = paths_conf.get('snodas_out')
        self.remote_sensing_tables_dir = paths_conf.get('remote_sensing_tables')
        self.rs_tables_dir = paths_conf.get('rs_tables', self.remote_sensing_tables_dir)
        self.joined_timeseries_dir = paths_conf.get('joined_timeseries')
        self.dynamics_data_json = paths_conf.get('dynamics_data')

        self.station_metadata_csv = paths_conf.get('station_metadata')
        self.station_file_csv = paths_conf.get('station_file', self.station_metadata_csv)
        self.results_dir = paths_conf.get('results')
        self.data_folder = self.data_dir
        self.input_data = paths_conf.get('prepped_input')
        self.plot_timeseries = self.joined_timeseries_dir

        self.ee_fields_flux3p = ee_conf.get('fields_flux3p')
        self.ee_fields_flux = ee_conf.get('fields_flux')
        self.ee_fields = ee_conf.get('fields', self.ee_fields_flux3p or self.ee_fields_flux)
        self.ee_bucket = ee_conf.get('bucket')

        self.feature_id_col = ids_conf.get('feature_id')
        self.gridmet_mapping_index_col = ids_conf.get('gridmet_mapping_index')
        self.state_col = ids_conf.get('state_col')
        self.field_index = self.feature_id_col
        self.irrigation_threshold = misc_conf.get('irrigation_threshold')

        self.etf_target_model = calib_toml_conf.get('etf_target_model')
        self.etf_ensemble_members = calib_toml_conf.get('etf_ensemble_members')
        self.workers = calib_toml_conf.get('workers')
        self.realizations = calib_toml_conf.get('realizations')
        self.obs_folder = calib_toml_conf.get('obs_folder')

        self.calibration_dir_override = calibration_dir_override
        self.parameter_set_json = parameter_set_json
        self.forecast_param_csv = forecast_param_csv

        self.elev_units = misc_conf.get('elev_units', 'm')
        self.refet_type = misc_conf.get('refet_type')
        self.irr_threshold = self.irrigation_threshold

        sdt_str = date_range_conf.get('start_date')
        edt_str = date_range_conf.get('end_date')
        if sdt_str:
            self.start_dt = pd.to_datetime(sdt_str)
        if edt_str:
            self.end_dt = pd.to_datetime(edt_str)

        self.kc_proxy = crop_coeff_conf.get('kc_proxy')
        self.cover_proxy = crop_coeff_conf.get('cover_proxy')
        self.era5land_params = era5land_conf.get('params')
        self.era5land_param_mapping = era5land_conf.get('param_mapping')

        if self.project_ws:
            os.makedirs(self.project_ws, exist_ok=True)

        self.toml_calibration_dir = calib_toml_conf.get('calibration_dir')
        self.toml_initial_values_csv = calib_toml_conf.get('initial_values_csv')
        self.forecast_parameters_csv = forecast_toml_conf.get('forecast_parameters')
        self.spinup = calib_toml_conf.get('spinup')

        if self.calibrate or calibration_dir_override:
            self.calibration_dir_override = calibration_dir_override
            self.read_calibration_parameters()

        elif self.forecast:
            self.calibration_dir = None
            self.read_forecast_parameters()

    def read_calibration_parameters(self):

            self.calibrate = True

            if self.calibration_dir_override:
                self.calibration_dir = self.calibration_dir_override
            elif self.toml_calibration_dir:
                self.calibration_dir = self.toml_calibration_dir

            initial_values_csv_path = self.toml_initial_values_csv
            if not os.path.isabs(initial_values_csv_path) and self.project_ws:
                initial_values_csv_path = os.path.join(self.project_ws, initial_values_csv_path)
            param_init = pd.read_csv(initial_values_csv_path, index_col=0)
            self.calibrated_parameters = param_init.index
            _files = list(param_init['mult_name'])
            self.calibration_files = {k: os.path.join(self.calibration_dir, f)
                                      for k, f in zip(self.calibrated_parameters, _files)}

    def read_forecast_parameters(self):

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
            p_str = ['_'.join(s.split(':')[1].split('_')[1:-1]) if ':' in s and len(s.split(':')) > 1 else s for s
                     in list(param_mean.index)]
            param_mean.index = p_str
            self.forecast_parameters = param_mean.copy()
            self.parameter_list = param_mean.index.to_list()

        elif self.parameter_set_json:
            with open(self.parameter_set_json, 'r') as f:
                param_arr = json.load(f)
            d = param_arr['fields']
            self.forecast_parameter_groups = [list(v.keys()) for k, v in d.items()][0]
            k_list = []
            for main_key, val_dict in d.items():
                for sub_key in val_dict.keys():
                    k_list.append(f'{sub_key}_{main_key}')
            v_list = []
            for tup in [(i.split('_')[0], '_'.join(i.split('_')[1:])) for i in k_list]:
                v_list.append(d[tup[1]][tup[0]])
            self.forecast_parameters = pd.Series(index=k_list, data=v_list)
            self.parameter_list = self.forecast_parameters.index.to_list()

    def __str__(self):
        return (
            f"ProjectConfig:\n"
            f"  Project Name: {self.project_name}\n"
            f"  Root Path: {self.root_path}\n"
            f"  Project Workspace: {self.project_ws}\n"
            f"  Data Directory: {self.data_dir}\n"
            f"  Shapefile: {self.footprint_shapefile_shp}\n"
            f"  Start Date: {self.start_dt}\n"
            f"  End Date: {self.end_dt}\n"
            f"  Calibrate Mode: {self.calibrate}\n"
            f"  Forecast Mode: {self.forecast}\n"
            f"  Alternative Root Config Loaded: {bool(self.alternative_root_config)}"
        )

    @staticmethod
    def _resolve_paths(raw_config, base_format_vars):
        config = json.loads(json.dumps(raw_config))
        format_vars = {k: (os.path.expanduser(v) if isinstance(v, str) else v)
                       for k, v in base_format_vars.items()}

        max_iterations = 10
        for i in range(max_iterations):
            newly_resolved_count = 0
            for section_name, section_content in list(config.items()):
                if isinstance(section_content, dict):
                    for key, value in list(section_content.items()):
                        if isinstance(value, str) and ('{' in value or '}' in value):
                            template = value.replace(' ', '')
                            try:
                                formatted_value = template.format(**format_vars)
                                if formatted_value != section_content[key]:
                                    section_content[key] = formatted_value
                                    newly_resolved_count += 1
                                if key not in format_vars or format_vars[key] != formatted_value:
                                    format_vars[key] = formatted_value
                            except KeyError as e:
                                pass
                elif isinstance(section_content, str) and ('{' in section_content or '}' in section_content):
                    template = section_content.replace('{{', '{').replace('}}', '}')
                    try:
                        formatted_value = template.format(**format_vars)
                        if formatted_value != config[section_name]:
                            config[section_name] = formatted_value
                            newly_resolved_count += 1
                        if section_name not in format_vars or format_vars[section_name] != formatted_value:
                            format_vars[section_name] = formatted_value
                    except KeyError:
                        pass

            if newly_resolved_count == 0 and i > 0:
                break

        for section_name, section_content in config.items():
            if isinstance(section_content, dict):
                for key, value in section_content.items():
                    if isinstance(value, str) and value.startswith('~'):
                        section_content[key] = os.path.expanduser(value)
            elif isinstance(section_content, str) and section_content.startswith('~'):
                config[section_name] = os.path.expanduser(section_content)
        return config


if __name__ == '__main__':
    pass
