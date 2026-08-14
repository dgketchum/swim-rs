"""Microsoft Planetary Computer (STAC) extraction — EE-free NDVI pipeline.

Produces per-field NDVI tables that are drop-in replacements for the Earth
Engine exports written by scripts/nv_ndvi.py / sid_ndvi.py, using Landsat
Collection 2 Level-2 and Sentinel-2 L2A assets served by the Planetary
Computer, with local IrrMapper rasters for irr/inv_irr masking.
"""
