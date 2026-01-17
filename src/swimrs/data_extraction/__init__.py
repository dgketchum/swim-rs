"""
Data extraction module for SWIM-RS.

Provides tools for retrieving remote sensing and meteorological data
from various sources including Google Earth Engine and GridMET.

Subpackages:
    ee: Google Earth Engine export functions for OpenET products.
        - export_ptjpl_zonal_stats: PT-JPL ET model exports.
        - export_ssebop_zonal_stats: SSEBop ET model exports.
        - export_sims_zonal_stats: SIMS ET model exports.
        - export_geesebal_zonal_stats: geeSEBAL ET model exports.

    gridmet: GridMET meteorological data retrieval.
        - GridMet: Class for downloading daily meteorological data.

Note:
    Earth Engine functions require authenticated access to the
    Google Earth Engine API. Use `ee.Authenticate()` and `ee.Initialize()`
    before calling export functions.
"""

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
