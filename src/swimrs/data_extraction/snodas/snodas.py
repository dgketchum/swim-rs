import glob
import json

import pandas as pd
from tqdm import tqdm


def create_timeseries_json(directory, json_out, feature_id="FID"):
    """Aggregate SNODAS CSVs into per-feature daily SWE JSON.

    Expects CSV files where rows are features and columns are dates. Values are
    assumed to be SWE in meters and are converted to millimeters.

    Parameters
    - directory: folder containing SNODAS CSVs.
    - json_out: output JSON path to write nested dict {feature: [{date, value}, ...]}.
    - feature_id: column name used as index/feature ID in CSVs.
    """
    all_files = glob.glob(directory + "/*.csv")
    timeseries = {}
    for filename in tqdm(all_files, desc="Processing CSV SNODAS files"):
        df = pd.read_csv(filename, index_col=feature_id)
        for index, row in df.iterrows():
            fid = index
            if fid not in timeseries:
                timeseries[fid] = []
            for date, value in row.items():
                timeseries[fid].append({"date": date, "value": value * 1000.0})

    with open(json_out, "w") as f:
        json.dump(timeseries, f)
        print(f"wrote {len(timeseries)} points to {json_out}")


if __name__ == "__main__":
    pass

# ========================= EOF ====================================================================
