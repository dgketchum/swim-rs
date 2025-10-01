import pandas as pd
import numpy as np
from scipy.stats import percentileofscore


def sentinel_adjust_quantile_mapping(
        landsat_ndvi_df,
        sentinel_ndvi_df,
        window_days=5,
        min_pairs=20
):
    landsat_ndvi = landsat_ndvi_df.iloc[:, 0]
    sentinel_ndvi = sentinel_ndvi_df.iloc[:, 0]

    if landsat_ndvi.empty or sentinel_ndvi.empty:
        return sentinel_ndvi.copy()

    landsat_ndvi_sorted = landsat_ndvi.sort_index().dropna()
    sentinel_ndvi_sorted = sentinel_ndvi.sort_index().dropna()

    if landsat_ndvi_sorted.empty or sentinel_ndvi_sorted.empty:
        return sentinel_ndvi.copy()

    landsat_df = landsat_ndvi_sorted.rename('landsat_val').rename_axis('time').reset_index()
    sentinel_df = sentinel_ndvi_sorted.rename('sentinel_val').rename_axis('time').reset_index()

    paired_df = pd.merge_asof(
        left=landsat_df,
        right=sentinel_df,
        on='time',
        direction='nearest',
        tolerance=pd.Timedelta(days=window_days)
    )

    paired_df = paired_df.dropna(subset=['landsat_val', 'sentinel_val'])

    landsat_train_values = paired_df['landsat_val'].values
    sentinel_train_values = paired_df['sentinel_val'].values

    original_sentinel_to_adjust = sentinel_ndvi.dropna()

    adjusted_values_list = []

    for s_val in original_sentinel_to_adjust.values:
        s_val_percentile = percentileofscore(sentinel_train_values, s_val, kind='weak')
        s_val_percentile = np.clip(s_val_percentile, 0.0, 100.0)

        try:
            adjusted_s_val = np.percentile(landsat_train_values, s_val_percentile)
        except ValueError:
            adjusted_s_val = np.nan

        adjusted_values_list.append(adjusted_s_val)

    input_series_name = sentinel_ndvi.name

    if isinstance(input_series_name, tuple):
        name_list = list(input_series_name)
        if len(name_list) > 2 and isinstance(name_list[2], str) and 'ndvi' in name_list[2].lower():
            name_list[2] = name_list[2] + "_adjusted"
            series_name = tuple(name_list)
        else:
            name_list.append("adjusted")
            series_name = tuple(name_list)

    elif input_series_name:
        series_name = input_series_name + "_adjusted"

    else:
        series_name = "sentinel_ndvi_adjusted"

    adjusted_sentinel_series_nonan = pd.Series(
        adjusted_values_list,
    index = original_sentinel_to_adjust.index,
    name = series_name
    )

    final_adjusted_series = adjusted_sentinel_series_nonan.reindex(sentinel_ndvi.index)

    return final_adjusted_series


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
