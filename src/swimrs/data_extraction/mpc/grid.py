"""Raster grid description and windowed remote-COG reads."""

import time

import numpy as np
import planetary_computer as pc
import rasterio
from rasterio import windows
from rasterio.crs import CRS
from rasterio.transform import Affine

# Sentinel used instead of NaN so exactextract's nodata comparison is exact.
NODATA = -9999.0

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.jp2",
    "GDAL_CACHEMAX": 256,
}


class GridSpec:
    """A georeferenced regular grid: CRS + affine transform + shape."""

    def __init__(self, crs: CRS, transform: Affine, width: int, height: int):
        self.crs = crs
        self.transform = transform
        self.width = width
        self.height = height

    @property
    def bounds(self):
        """(xmin, ymin, xmax, ymax) — assumes north-up transform."""
        xmin = self.transform.c
        ymax = self.transform.f
        xmax = xmin + self.transform.a * self.width
        ymin = ymax + self.transform.e * self.height
        return xmin, ymin, xmax, ymax

    @property
    def shape(self):
        return self.height, self.width


def read_window(href, bounds, fill_value=0, retries=4, sign=True):
    """Sign + open + read one band windowed to `bounds` (native CRS).

    The href is signed immediately before every attempt so SAS tokens are
    always fresh on long runs. Returns (array, GridSpec). Off-edge portions
    of the window are filled with `fill_value` (band nodata).
    """
    last = None
    for attempt in range(retries):
        try:
            url = pc.sign(href) if sign else href
            with rasterio.Env(**GDAL_ENV), rasterio.open(url) as ds:
                win = windows.from_bounds(*bounds, transform=ds.transform)
                win = windows.Window(
                    int(np.floor(win.col_off)),
                    int(np.floor(win.row_off)),
                    int(np.ceil(win.width)),
                    int(np.ceil(win.height)),
                )
                arr = ds.read(1, window=win, boundless=True, fill_value=fill_value)
                transform = windows.transform(win, ds.transform)
                grid = GridSpec(ds.crs, transform, int(win.width), int(win.height))
                return arr, grid
        except Exception as exc:  # network / token blips
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to read {href} after {retries} tries: {last}")


def block_reduce_mean(arr, factor, nodata=0):
    """Downsample a 2-D array by integer `factor` using a nodata-aware mean.

    Used to bring Sentinel-2 10 m red onto the 20 m B8A/SCL grid. Trailing
    rows/cols that don't fill a block are trimmed (callers align windows to
    the coarse grid, so trims are empty in practice).
    """
    h = (arr.shape[0] // factor) * factor
    w = (arr.shape[1] // factor) * factor
    a = arr[:h, :w].astype(np.float64)
    a[a == nodata] = np.nan
    a = a.reshape(h // factor, factor, w // factor, factor)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(a, axis=(1, 3))
    return out
