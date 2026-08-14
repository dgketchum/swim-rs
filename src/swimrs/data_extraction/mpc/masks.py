"""Local IrrMapper rasters -> per-scene irr / inv_irr masks.

Replicates the EE semantics in scripts/nv_ndvi.py exactly:
- irr:     pixel classified irrigated THIS year (classification < 1) AND
           irrigated in >= min_years of the min-yr window (1987-2023).
- inv_irr: pixel classified anything-but-irrigated this year
           (classification > 0). The min-yr mask is NOT used.
- Years past IRR_MAX_YEAR (2023) reuse the 2023 classification.

Source rasters are user-provided annual classification GeoTIFFs named with a
4-digit year in the filename (e.g. irrmapper_NV_2003.tif), any CRS. The
4-class IrrMapper convention (0=irrigated, 1=dryland, 2=uncultivated,
3=wetland) is the default; `irrigated_value` covers collapsed binary exports.
Verify the convention against a known field before a production run.
"""

import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds

IRR_MAX_YEAR = 2023
MIN_YR_FILENAME = "irr_min_yr_mask.tif"


class IrrMapperMasks:
    def __init__(self, directory, irrigated_value=0, min_years=5, irr_max_year=IRR_MAX_YEAR):
        self.directory = Path(directory)
        self.irrigated_value = irrigated_value
        self.min_years = min_years
        self.irr_max_year = irr_max_year
        self.annual = {}
        for tif in sorted(self.directory.glob("*.tif")):
            if tif.name == MIN_YR_FILENAME:
                continue
            match = re.search(r"(19|20)\d{2}", tif.name)
            if match:
                self.annual[int(match.group(0))] = tif
        if not self.annual:
            raise FileNotFoundError(f"no annual IrrMapper rasters under {self.directory}")
        self.min_yr_path = self.directory / MIN_YR_FILENAME

    def classification(self, year, grid):
        """Annual classification resampled (nearest) onto `grid`.

        Returns an int array; pixels outside the raster get a fill value of
        255 (never a valid class), excluded by both mask types.
        """
        year = min(year, self.irr_max_year)
        if year not in self.annual:
            raise KeyError(f"no IrrMapper raster for {year} under {self.directory}")
        return self._reproject(self.annual[year], grid, fill=255)

    def mask_for(self, mask_type, year, grid):
        """Boolean include-mask on `grid` for 'irr' / 'inv_irr' / 'no_mask'."""
        if mask_type == "no_mask":
            return np.ones(grid.shape, dtype=bool)
        cls = self.classification(year, grid)
        if mask_type == "irr":
            return (cls == self.irrigated_value) & (self.min_yr_mask(grid) > 0)
        if mask_type == "inv_irr":
            return (cls != self.irrigated_value) & (cls != 255)
        raise ValueError(f"unknown mask_type: {mask_type}")

    def min_yr_mask(self, grid):
        """The >= min_years-of-record irrigated mask, resampled onto `grid`."""
        self.ensure_min_yr()
        return self._reproject(self.min_yr_path, grid, fill=0)

    def ensure_min_yr(self):
        """Build irr_min_yr_mask.tif from the annuals if not present.

        Counts years with classification == irrigated_value over the full
        annual record (needs 1987+ coverage to match the EE asset exactly;
        with fewer years the mask is more restrictive — flagged at build).
        """
        if self.min_yr_path.exists():
            return self.min_yr_path
        years = sorted(self.annual)
        if years[0] > 1987:
            print(
                f"WARNING: min-yr mask built from {years[0]}-{years[-1]}; "
                f"EE asset used 1987-2023 — provide the exported asset for exact parity"
            )
        count = None
        profile = None
        for year in years:
            with rasterio.open(self.annual[year]) as src:
                arr = src.read(1)
                if count is None:
                    count = np.zeros(arr.shape, dtype=np.uint8)
                    profile = src.profile.copy()
                count += (arr == self.irrigated_value).astype(np.uint8)
        mask = (count >= self.min_years).astype(np.uint8)
        profile.update(dtype="uint8", count=1, nodata=None, compress="lzw")
        with rasterio.open(self.min_yr_path, "w", **profile) as dst:
            dst.write(mask, 1)
        return self.min_yr_path

    def _reproject(self, path, grid, fill):
        with rasterio.open(path) as src:
            src_bounds = transform_bounds(grid.crs, src.crs, *grid.bounds)
            window = rasterio.windows.from_bounds(*src_bounds, transform=src.transform)
            window = rasterio.windows.Window(
                int(np.floor(window.col_off)),
                int(np.floor(window.row_off)),
                int(np.ceil(window.width)),
                int(np.ceil(window.height)),
            )
            window = window.crop(src.height, src.width) if _overlaps(window, src) else None
            dst = np.full(grid.shape, fill, dtype=np.int16)
            if window is None or window.width <= 0 or window.height <= 0:
                return dst
            data = src.read(1, window=window)
            reproject(
                source=data,
                destination=dst,
                src_transform=rasterio.windows.transform(window, src.transform),
                src_crs=src.crs,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                resampling=Resampling.nearest,
                src_nodata=src.nodata,
                dst_nodata=fill,
            )
            return dst


def _overlaps(window, src):
    return (
        window.col_off < src.width
        and window.row_off < src.height
        and window.col_off + window.width > 0
        and window.row_off + window.height > 0
    )
