"""Coverage-weighted zonal means, matching EE's weighted reduceRegions.

EE's default ee.Reducer.mean() in reduceRegions weights each pixel by the
fraction of it covered by the polygon; exactextract's 'mean' op does the
same. 'count' is the coverage-weighted count of unmasked cells — kept as a
QC column in the intermediate records (not in the final EE-format CSVs).
"""

import pandas as pd
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

from swimrs.data_extraction.mpc.grid import NODATA


def field_means(values, grid, gdf, feature_id, include_cols=None):
    """Mean NDVI per polygon over one scene window.

    `values` is a 2-D float array with NODATA where masked; `gdf` must be in
    `grid.crs`. Returns a DataFrame with the id columns plus [mean, count]
    (mean is NaN where no unmasked pixel touches the polygon).
    """
    include_cols = include_cols or [feature_id]
    xmin, ymin, xmax, ymax = grid.bounds
    src = NumPyRasterSource(
        values,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        nodata=NODATA,
        srs_wkt=grid.crs.to_wkt(),
    )
    result = exact_extract(
        src,
        gdf,
        ["mean", "count"],
        include_cols=include_cols,
        output="pandas",
    )
    out = {col: result[col].values for col in include_cols}
    out["mean"] = result["mean"].values
    out["count"] = result["count"].values
    return pd.DataFrame(out)
