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
        self.data = None
        self.fields = None
        self.cuttings = None
        self.field_props = None

    def initialize_plot_data(self, config):

        f = config.fields_path
        with open(f, 'r') as fp:
            self.fields = json.load(fp)

        f = config.field_properties
        with open(f, 'r') as fp:
            self.field_props = json.load(fp)

        f = config.irrigation_data
        with open(f, 'r') as fp:
            self.cuttings = json.load(fp)

        f = config.input_timeseries
        with open(f, 'r') as fp:
            self.data = json.load(fp)


if __name__ == '__main__':
    pass
