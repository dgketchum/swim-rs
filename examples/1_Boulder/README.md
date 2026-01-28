# 1_Boulder Tutorial

A complete end-to-end SWIM-RS workflow demonstrating the SwimContainer API for data management, extraction, and model execution.

## Overview

This tutorial walks through all stages of a SWIM-RS project using a small study area near Boulder, Montana. It showcases the `SwimContainer` API which provides a unified interface for:

- **Data ingestion**: Remote sensing, meteorology, snow, and static properties
- **Computation**: Fused NDVI, irrigation dynamics, crop coefficients
- **Export**: Model-ready JSON files with full provenance tracking

## Notebooks

Run these notebooks in order:

| Notebook | Description |
|----------|-------------|
| `01_create_container.ipynb` | Explore the shapefile and create a SwimContainer |
| `02_extract_data.ipynb` | Extract data from Earth Engine and GridMET (optional) |
| `03_ingest_data.ipynb` | Ingest extracted or pre-built data into the container |
| `04_compute_and_export.ipynb` | Compute dynamics and export model inputs |
| `05_run_model.ipynb` | Run the SWIM model and visualize outputs |

## Two Paths

### With Earth Engine Access

Run all 5 notebooks in sequence. Notebook 02 extracts data from Google Earth Engine and GridMET THREDDS.

### Without Earth Engine Access

1. Run `01_create_container.ipynb` to create the container
2. Skip `02_extract_data.ipynb`
3. Run `03_ingest_data.ipynb` with `USE_PREBUILT = True` to use pre-extracted data
4. Continue with notebooks 04 and 05

## Configuration

- **1_Boulder.toml**: Project configuration with paths, date range, and settings
- **Container**: `data/1_Boulder.swim` - single file containing all project data

## Data Sources

- **NDVI**: Landsat 8/9 from Earth Engine
- **ETf**: SSEBop fraction of reference ET from Earth Engine
- **Meteorology**: GridMET via THREDDS (with bias correction)
- **Snow**: SNODAS SWE from Earth Engine
- **Properties**: SSURGO soils, NLCD land cover, IrrMapper/LANID irrigation fractions

## Study Area

The shapefile (`data/gis/mt_sid_boulder.shp`) contains agricultural fields from the Montana Statewide Irrigation Dataset, clipped to an area near Boulder, Montana. Fields are in Albers Equal Area projection (EPSG:5071).

## Outputs

- `data/1_Boulder.swim`: Container with all ingested data
- `data/prepped_input.json`: Model-ready input file
- `data/model_output/`: CSV files with daily model results
