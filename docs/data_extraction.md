# Data Extraction Guide

This guide covers extracting remote sensing and meteorological data for SWIM-RS using Google Earth Engine and GridMET/ERA5-Land.

## Overview

SWIM-RS requires several data inputs that are typically extracted from remote sources:

Some extraction pathways use optional OpenET Python implementations (and refetgee). If you plan to run those exporters, install:

```bash
pip install "swimrs[openet]"
```

| Data Type | Source | Function |
|-----------|--------|----------|
| NDVI | Landsat 8/9, Sentinel-2 | `clustered_sample_ndvi`, `sparse_sample_ndvi` |
| ET Fraction (ETf) | OpenET (SSEBop, PT-JPL, SIMS, geeSEBAL) | `clustered_sample_etf`, `sparse_sample_etf` |
| Meteorology | GridMET (CONUS), ERA5-Land (global) | `swim extract` CLI or THREDDS direct |
| Snow (SWE) | SNODAS (CONUS), ERA5 (global) | `swim extract` CLI |
| Properties | SSURGO/HWSD soils, CDL/NLCD land cover, LANID/IrrMapper irrigation | `swimrs.data_extraction.ee.ee_props` |

## CLI Extraction

The simplest approach uses the `swim extract` CLI command:

```bash
# CONUS workflow (GridMET + SNODAS + Landsat)
swim extract my_project.toml

# Add Sentinel-2 NDVI
swim extract my_project.toml --add-sentinel

# International workflow (ERA5-Land)
swim extract my_project.toml --international
```

Outputs are exported to Google Drive (default) or a Cloud Storage bucket (if `earth_engine.bucket` is set in the TOML).

## Python API: Clustered vs Sparse Sampling

For programmatic control, use the extraction functions directly.

### When to Use Each

| Function | Use Case |
|----------|----------|
| `clustered_sample_*` | Fields are geographically clustered (e.g., a single watershed or county). More efficient — groups nearby fields into fewer EE tasks. |
| `sparse_sample_*` | Fields are widely dispersed (e.g., flux towers across CONUS). Creates one task per field or small group. |

### Function Signatures

Both clustered functions accept:

- **`shapefile`**: Local shapefile path, EE FeatureCollection asset ID, or `ee.FeatureCollection` object
- **`feature_id`**: Column name for unique field identifiers
- **`state_col`**: Column with US state codes (enables IrrMapper west / LANID east mask selection)
- **`mask_type`**: `'irr'` (irrigated), `'inv_irr'` (non-irrigated), or `'none'`
- **`start_yr` / `end_yr`**: Year range for extraction
- **`select`**: List of feature IDs to limit extraction (optional)
- **`dest`**: `'drive'` or `'bucket'`
- **`drive_folder`**: Google Drive folder name (if `dest='drive'`)

### ETf-Specific Parameters

- **`model`**: ETf model — `'ssebop'`, `'ptjpl'`, `'sims'`, `'geesebal'`, or `'disalexi'`
- **`usgs_nhm`**: Use USGS NHM SSEBop (higher resolution) instead of OpenET SSEBop

### NDVI-Specific Parameters

- **`satellite`**: `'landsat'` or `'sentinel'`

## Example: Extract ETf and NDVI for Selected Fields

```python
from swimrs.data_extraction.ee.etf_export import clustered_sample_etf
from swimrs.data_extraction.ee.ndvi_export import clustered_sample_ndvi

select_fields = ['043_000130', '043_000128', '043_000161']
shapefile_path = 'examples/1_Boulder/data/gis/mt_sid_boulder.shp'

# Extract SSEBop ETf with irrigation mask
clustered_sample_etf(
    shapefile_path,
    mask_type='irr',
    start_yr=2004,
    end_yr=2023,
    feature_id='FID_1',
    state_col='STATE',
    select=select_fields,
    dest='drive',
    drive_folder='swim',
    drive_categorize=True,
    model='ssebop',
    usgs_nhm=True,
)

# Extract Landsat NDVI with same parameters
clustered_sample_ndvi(
    shapefile_path,
    mask_type='irr',
    start_yr=2004,
    end_yr=2023,
    feature_id='FID_1',
    state_col='STATE',
    select=select_fields,
    satellite='landsat',
    dest='drive',
    drive_folder='swim',
    drive_categorize=True,
)
```

## Example: Extract Multiple ETf Models for Ensemble

```python
from swimrs.data_extraction.ee.etf_export import clustered_sample_etf

for model in ['ssebop', 'ptjpl', 'sims']:
    clustered_sample_etf(
        'data/gis/flux_fields.shp',
        mask_type='irr',
        start_yr=2000,
        end_yr=2023,
        feature_id='site_id',
        state_col='state',
        dest='drive',
        drive_folder=f'swim_etf_{model}',
        model=model,
    )
```

## Monitoring EE Tasks

After submitting exports, monitor progress at: https://code.earthengine.google.com/tasks

Tasks are named with the field ID and date range for easy identification.

## Downloading Results

### From Google Drive

Use `gsutil` or the Drive web interface to download CSVs:

```bash
# Sync from Drive to local
# (requires gcloud auth and Drive API enabled)
rclone sync gdrive:swim data/remote_sensing/landsat/extracts/
```

### From Cloud Storage Bucket

```bash
gsutil -m rsync -r gs://your-bucket/swim/ data/remote_sensing/
```

## Meteorology Extraction

GridMET data is fetched directly from THREDDS (no EE required):

```python
from swimrs.data_extraction.gridmet.gridmet import get_gridmet

get_gridmet(
    shapefile='data/gis/fields.shp',
    output_dir='data/met_timeseries/gridmet',
    start_date='1987-01-01',
    end_date='2023-12-31',
    feature_id='site_id',
)
```

For ERA5-Land (international sites), use the `swim extract --international` CLI or the ERA5 extraction utilities.

## Tips

- **Rate limits**: EE has quotas. For large extractions (>1000 fields), batch over multiple days or use `select` to limit fields per run.
- **Mask selection**: The `state_col` parameter automatically selects IrrMapper (western US) or LANID (eastern US) for irrigation masks.
- **Drive vs Bucket**: Drive exports are simpler but have size limits. Use bucket exports for large projects.
- **Sentinel-2**: Only available from 2017+. For longer time series, use Landsat alone or fuse both in the container.
