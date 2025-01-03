import json


class SamplePlots:
    """A Container for input time series, historical, and static field information

    This should include some initial estimate of soil properties and historical
    estimate of irrigated status and crop type.

    """

    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def initialize_plot_data(self, config):
        f = config.input_data
        with open(f, 'r') as fp:
            self.input = json.load(fp)


if __name__ == '__main__':
    pass
