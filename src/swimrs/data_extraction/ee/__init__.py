"""
OpenET ETf zonal statistics export modules.

This package provides modules for exporting ET fraction (ETf) zonal statistics
from various OpenET models to Google Cloud Storage as CSV tables.

Modules
-------
ptjpl_export
    PT-JPL ET fraction zonal statistics export.
sims_export
    SIMS ET fraction zonal statistics export.
ssebop_export
    SSEBop ET fraction zonal statistics export.
geesebal_export
    geeSEBAL ET fraction zonal statistics export.
common
    Shared utilities and constants for all export modules.

Example
-------
>>> from swimrs.data_extraction.ee import export_ptjpl_zonal_stats
>>> export_ptjpl_zonal_stats(
...     shapefile='path/to/fields.shp',
...     bucket='my-gcs-bucket',
...     feature_id='site_id',
...     start_yr=2020,
...     end_yr=2024,
... )
"""

from swimrs.data_extraction.ee.geesebal_export import export_geesebal_zonal_stats
from swimrs.data_extraction.ee.ptjpl_export import export_ptjpl_zonal_stats
from swimrs.data_extraction.ee.sims_export import export_sims_zonal_stats
from swimrs.data_extraction.ee.ssebop_export import export_ssebop_zonal_stats

__all__ = [
    "export_ptjpl_zonal_stats",
    "export_sims_zonal_stats",
    "export_ssebop_zonal_stats",
    "export_geesebal_zonal_stats",
]
