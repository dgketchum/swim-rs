import json

import geopandas as gpd
import pandas as pd


class SamplePlots:
    """A Container for input time series, historical, and static field information

    This should include some initial estimate of soil properties and historical
    estimate of irrigated status and crop type.

    """

    def __init__(self):
        super().__init__()
        self.fields_dict = None

    def initialize_plot_data(self, config, target=None):
        self.fields_dict = {}

        df = gpd.read_file(config.fields_path)
        df.index = df[config.field_index]

        for fid, row in df.iterrows():

            if target and target != fid:
                continue

            field = PlotData()

            field.field_id = str(fid)
            field.lat = row['LAT']
            field.lon = row['LON']
            field.geometry = row['geometry']

            self.fields_dict[field.field_id] = field

            field.set_input_timeseries(config)
            field.set_field_properties(config)
            field.set_irrigation_cuttings(config)


class PlotData:
    """ A container for individual sample plot information

    This should include input time series information, initial estimate of soil properties,
    and estimates of historical state (e.g., crop type and irrigation status).

    """

    def __init__(self):
        super().__init__()
        self.irrigation_data = None
        self.props = None
        self.refet = None
        self.crop_coeffs = None
        self.field_id = None
        self.field_id = None
        self.lat = None
        self.lon = None
        self.geometry = None
        self.input = None

    def set_input_timeseries(self, config):
        f = config.input_timeseries.format(self.field_id)
        df = pd.read_csv(f, parse_dates=True, index_col=0)
        df['doy'] = [int(dt.strftime('%j')) for dt in df.index]

        self.input = {}
        for dt, group in df.iterrows():
            date_str = '{}-{:02d}-{:02d}'.format(dt.year, dt.month, dt.day)
            self.input[date_str] = group.to_dict()

        if config.field_type == 'irrigated':
            self.refet = df['{}_mm'.format(config.refet_type)]
        elif config.field_type == 'unirrigated':
            self.refet = df['{}_mm_uncorr'.format(config.refet_type)]
        else:
            raise NotImplementedError('Uknown field type')

    def set_field_properties(self, config):
        f = config.field_properties
        with open(f, 'r') as fp:
            dct = json.load(fp)
        self.props = dct[str(self.field_id)]

    def set_irrigation_cuttings(self, config):
        f = config.irrigation_data
        with open(f, 'r') as fp:
            dct = json.load(fp)
        self.irrigation_data = dct[str(self.field_id)]


if __name__ == '__main__':
    pass
