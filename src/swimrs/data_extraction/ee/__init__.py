"""
OpenET ETf zonal statistics export modules.

This package provides modules for exporting ET fraction (ETf) zonal statistics
using the open-source OpenET Python packages (openet-ptjpl, openet-ssebop, etc.).

Primary Interface
-----------------
export_etf : Unified dispatcher for ETf extraction with sparse/clustered modes.

Individual Model Exports (for advanced use)
-------------------------------------------
export_ptjpl_zonal_stats : PT-JPL export (sparse mode only).
export_ssebop_zonal_stats : SSEBop export (sparse mode only).
export_sims_zonal_stats : SIMS export (sparse mode only).
export_geesebal_zonal_stats : geeSEBAL export (sparse mode only).

Example
-------
>>> from swimrs.data_extraction.ee import export_etf
>>> export_etf(
...     shapefile='path/to/fields.shp',
...     model='ptjpl',
...     feature_id='site_id',
...     start_yr=2020,
...     end_yr=2024,
...     clustered=True,  # efficient for clustered fields
...     dest='drive',
... )
"""

from swimrs.data_extraction.ee.etf_export import export_etf
from swimrs.data_extraction.ee.geesebal_export import export_geesebal_zonal_stats
from swimrs.data_extraction.ee.ptjpl_export import export_ptjpl_zonal_stats
from swimrs.data_extraction.ee.sims_export import export_sims_zonal_stats
from swimrs.data_extraction.ee.ssebop_export import export_ssebop_zonal_stats

__all__ = [
    "export_etf",
    "export_ptjpl_zonal_stats",
    "export_sims_zonal_stats",
    "export_ssebop_zonal_stats",
    "export_geesebal_zonal_stats",
]
