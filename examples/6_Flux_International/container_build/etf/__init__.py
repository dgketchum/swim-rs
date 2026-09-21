"""
OpenET ETf zonal statistics export modules (International).

This package provides modules for exporting ET fraction (ETf) zonal statistics
from OpenET models to Google Cloud Storage as CSV tables. This version uses
ERA5LAND for meteorology and is suitable for international flux sites.

Modules
-------
ptjpl_export
    PT-JPL ET fraction zonal statistics export with ERA5LAND meteorology.
common
    Shared utilities and constants for export modules.

Example
-------
>>> from etf.ptjpl_export import export_ptjpl_zonal_stats
>>> export_ptjpl_zonal_stats(
...     shapefile='path/to/fields.shp',
...     bucket='my-gcs-bucket',
...     feature_id='sid',
...     start_yr=2020,
...     end_yr=2024,
... )
"""

from .ptjpl_export import export_ptjpl_zonal_stats

__all__ = [
    "export_ptjpl_zonal_stats",
]
