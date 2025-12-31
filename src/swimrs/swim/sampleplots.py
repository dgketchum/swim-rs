import os
import json
import warnings

import numpy as np
import pandas as pd

from swimrs.model.tracker import TUNABLE_PARAMS


class SamplePlots:
    """A Container for input and output time series, historical, and static field information

    This should include some initial estimate of soil properties and historical
    estimate of irrigated status and crop type.

    """

    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None
        self.spinup = None

    def initialize_plot_data(self, config):
        f = config.input_data
        self.input = {}

        try:
            with open(f, 'r', encoding='utf-8') as fp:
                for line in fp:
                    self.input.update(json.loads(line))
        except json.decoder.JSONDecodeError:
            with open(f, 'r') as fp:
                self.input = json.load(fp)

    def initialize_spinup(self, config):

        if os.path.isfile(config.spinup):
            print(f'SPINUP: {config.spinup}')
            with open(config.spinup, 'r') as f:
                self.spinup = json.load(f)
        else:
            raise FileNotFoundError(f'Spinup file {config.spinup} not found')

    def input_to_dataframe(self, feature_id):

        idx = self.input['order'].index(feature_id)

        ts = self.input['time_series']
        dct = {k: [] for k in ts[list(ts.keys())[0]]}
        dates = []

        for dt in ts:
            doy_data = ts[dt]
            dates.append(dt)
            for k, v in doy_data.items():
                if k == 'doy':
                    dct['doy'].append(v)
                else:
                    # all other values are lists
                    dct[k].append(v[idx])

        df_ = pd.DataFrame().from_dict(dct)
        df_.index = pd.DatetimeIndex(dates)
        return df_

    def reconcile_with_parameters(self, config):
        """Reconcile plots data with config parameters, keeping only common fields.

        In forecast or calibration mode, this method identifies fields that exist
        in both the plots data and the parameter set, filters the plots data to
        only include those fields, and warns about any fields that are dropped.

        Parameters
        ----------
        config : ProjectConfig
            Configuration object with forecast_parameters or calibration_files.

        Returns
        -------
        tuple
            (common_fields, dropped_from_plots, dropped_from_params) where:
            - common_fields: list of field IDs present in both datasets
            - dropped_from_plots: fields in plots but not in parameters
            - dropped_from_params: fields in parameters but not in plots
        """
        if self.input is None:
            raise ValueError("Plots data not initialized. Call initialize_plot_data first.")

        plots_fields = set(self.input['order'])
        plots_fields_lower = {f.lower(): f for f in plots_fields}

        # Extract field IDs from parameters
        param_fields = set()
        if config.forecast and config.forecast_parameters is not None:
            for param_name in config.forecast_parameters.index:
                # Parameter names are like "aw_fieldid", "ndvi_k_fieldid"
                for tunable in TUNABLE_PARAMS:
                    if param_name.startswith(f"{tunable}_"):
                        fid = param_name[len(tunable) + 1:]
                        param_fields.add(fid)
                        break

        elif config.calibrate and config.calibration_files is not None:
            for param_name in config.calibration_files.keys():
                for tunable in TUNABLE_PARAMS:
                    if tunable in param_name:
                        fid = param_name.replace(f"{tunable}_", "")
                        param_fields.add(fid)
                        break

        if not param_fields:
            # No parameters to reconcile - return all plots fields
            return list(plots_fields), [], []

        # PEST++ lowercases field IDs, so we need case-insensitive matching
        param_fields_lower = {f.lower(): f for f in param_fields}

        # Find common fields (case-insensitive match)
        common_lower = set(plots_fields_lower.keys()) & set(param_fields_lower.keys())

        # Map back to original case from plots (plots are the source of truth for field IDs)
        common_fields = [plots_fields_lower[f] for f in common_lower]

        # Identify dropped fields
        dropped_from_plots = [f for f in plots_fields if f.lower() not in common_lower]
        dropped_from_params = [param_fields_lower[f] for f in param_fields_lower if f not in common_lower]

        if dropped_from_plots:
            warnings.warn(
                f"Dropping {len(dropped_from_plots)} field(s) from plots data "
                f"(no parameters found): {dropped_from_plots[:5]}"
                + (f"... and {len(dropped_from_plots) - 5} more" if len(dropped_from_plots) > 5 else "")
            )

        if dropped_from_params:
            warnings.warn(
                f"Ignoring {len(dropped_from_params)} parameter set(s) "
                f"(no plots data found): {dropped_from_params[:5]}"
                + (f"... and {len(dropped_from_params) - 5} more" if len(dropped_from_params) > 5 else "")
            )

        # Filter plots data to only include common fields
        if common_fields and len(common_fields) < len(plots_fields):
            self._filter_to_fields(common_fields)

        return common_fields, dropped_from_plots, dropped_from_params

    def _filter_to_fields(self, field_list):
        """Filter plots input data to only include specified fields.

        Parameters
        ----------
        field_list : list
            List of field IDs to keep.
        """
        if self.input is None:
            return

        original_order = self.input['order']
        field_set = set(field_list)

        # Get indices of fields to keep (preserving order from field_list)
        keep_indices = []
        new_order = []
        for fid in field_list:
            if fid in original_order:
                keep_indices.append(original_order.index(fid))
                new_order.append(fid)

        if not keep_indices:
            raise ValueError("No valid fields remaining after filtering")

        # Update order
        self.input['order'] = new_order

        # Filter time series data (each value is a list indexed by field)
        for dt, day_data in self.input['time_series'].items():
            for key, values in day_data.items():
                if key == 'doy':
                    continue  # doy is a scalar, not a list
                if isinstance(values, list):
                    day_data[key] = [values[i] for i in keep_indices]

        # Filter properties
        if 'props' in self.input:
            self.input['props'] = {k: v for k, v in self.input['props'].items() if k in field_set}

        # Filter ke_max and kc_max
        for key in ['ke_max', 'kc_max']:
            if key in self.input:
                self.input[key] = {k: v for k, v in self.input[key].items() if k in field_set}

        # Filter irr_data
        if 'irr_data' in self.input:
            self.input['irr_data'] = {k: v for k, v in self.input['irr_data'].items() if k in field_set}

        # Filter gwsub_data
        if 'gwsub_data' in self.input:
            self.input['gwsub_data'] = {k: v for k, v in self.input['gwsub_data'].items() if k in field_set}


if __name__ == '__main__':
    pass
