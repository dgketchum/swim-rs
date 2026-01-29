# How-To Guide: From Shapefile to Calibrated Model

This guide walks through the complete SWIM-RS workflow starting with just a shapefile of fields or polygons.

## What You Get

After running SWIM-RS, you receive daily time series for each field including:

- **Evapotranspiration** (ET, mm/day) — calibrated to satellite observations
- **Snow water equivalent** (SWE, mm) — accumulation and melt
- **Soil moisture** (root zone depletion, mm)
- **Deep percolation** (groundwater recharge, mm/day)
- **Runoff** (mm/day)
- **Irrigation** (simulated applied water, mm/day)
- **Crop coefficients** (Kcb, Ke, Ks)

All outputs are calibrated against satellite ET fraction (ETf) from OpenET models, ensuring consistency with remote sensing observations while providing complete daily coverage and physically-based partitioning.

## Prerequisites

1. **SWIM-RS installed** — see [Installation Guide](installation.md)
2. **Earth Engine account** — sign up at https://earthengine.google.com/
3. **EE authenticated** — run `earthengine authenticate` and complete OAuth

Verify your setup:

```bash
source .venv/bin/activate
swim --help
python -c "import ee; ee.Initialize(); print('EE OK')"
```

## Overview

The SWIM-RS workflow has four main steps:

```
swim extract → swim prep → swim calibrate → swim evaluate
```

| Step | What it does | Time |
|------|--------------|------|
| `extract` | Exports NDVI, ETf, met, properties from EE/GridMET | Hours (EE queue) |
| `prep` | Builds `.swim` container, computes dynamics | Minutes |
| `calibrate` | Runs PEST++ IES parameter estimation | Minutes to hours |
| `evaluate` | Runs calibrated model, writes output CSVs | Seconds |

## Step 1: Prepare Your Shapefile

Your shapefile needs:

- **Unique ID column** — each polygon must have a unique identifier (e.g., `site_id`, `field_id`)
- **Valid geometries** — no self-intersections or null geometries
- **Projected or WGS84** — SWIM-RS will reproject as needed

Optional but recommended:
- **State column** — US state codes (e.g., `MT`, `OR`) for automatic irrigation mask selection

Example shapefile structure:

| site_id | state | geometry |
|---------|-------|----------|
| field_001 | MT | POLYGON(...) |
| field_002 | MT | POLYGON(...) |

## Step 2: Create Project Directory

Set up the standard directory structure:

```bash
mkdir -p my_project/data/gis
cp /path/to/my_fields.shp my_project/data/gis/
cp /path/to/my_fields.shx my_project/data/gis/
cp /path/to/my_fields.dbf my_project/data/gis/
cp /path/to/my_fields.prj my_project/data/gis/
```

## Step 3: Create the TOML Config

Copy the template and customize it:

```bash
cp /path/to/swim-rs/docs/template.toml my_project/my_project.toml
```

Edit the TOML to set your project name, shapefile path, date range, and other options. The template includes comments explaining each setting. Key fields to customize:

- `project` — your project name
- `fields_shapefile` — path to your shapefile
- `feature_id` — the unique ID column in your shapefile
- `start_date` / `end_date` — your study period
- `etf_target_model` — which OpenET model to calibrate against

See [template.toml](template.toml) for the full annotated template and [Config Schema Reference](#config-schema-reference) below for detailed documentation of all options.

## Step 4: Extract Data from Earth Engine

Run the extraction command:

```bash
cd my_project
swim extract my_project.toml
```

This exports to Google Drive by default. To use a GCS bucket:

```bash
swim extract my_project.toml --export bucket --bucket my-gcs-bucket
```

### What gets extracted

| Data | Source | Destination |
|------|--------|-------------|
| Landsat NDVI | EE Landsat 8/9 | `data/remote_sensing/landsat/` |
| ET fraction | OpenET (PT-JPL default) | `data/remote_sensing/etf/` |
| Meteorology | GridMET THREDDS | `data/met_timeseries/gridmet/` |
| Snow (SWE) | SNODAS | `data/snow/snodas/` |
| Soils | SSURGO | `data/properties/` |
| Land cover | CDL/NLCD | `data/properties/` |
| Irrigation | LANID/IrrMapper | `data/properties/` |

### Monitor EE tasks

Check progress at: https://code.earthengine.google.com/tasks

### Download from Drive

Once tasks complete, download the exported CSVs:

```bash
# Using rclone (configure gdrive remote first)
rclone sync gdrive:swim/landsat data/remote_sensing/landsat/extracts/
rclone sync gdrive:swim/etf data/remote_sensing/etf/extracts/
rclone sync gdrive:swim/snodas data/snow/snodas/extracts/
```

Or download manually from Drive and place in the appropriate directories.

### Optional: Add Sentinel-2

For higher temporal resolution NDVI (2017+):

```bash
swim extract my_project.toml --add-sentinel
```

### Optional: Multiple ETf models

For ensemble calibration:

```bash
swim extract my_project.toml --etf-models ssebop,ptjpl,sims
```

## Step 5: Build the Container

Once extraction is complete, build the `.swim` container:

```bash
swim prep my_project.toml
```

This:
1. Creates `data/my_project.swim` (Zarr-based container)
2. Ingests all extracted data with provenance tracking
3. Computes merged NDVI and crop dynamics
4. Exports model-ready inputs (`prepped_input.json`, `swim_input.h5`)

### Prep options

```bash
# Include Sentinel-2 NDVI
swim prep my_project.toml --add-sentinel

# Overwrite existing container
swim prep my_project.toml --overwrite

# International mode (no irrigation masks)
swim prep my_project.toml --international

# Limit to specific sites for testing
swim prep my_project.toml --sites field_001,field_002
```

### Inspect the container

```bash
swim inspect data/my_project.swim --detailed
```

## Step 6: Calibrate the Model

Run PEST++ IES calibration:

```bash
swim calibrate my_project.toml
```

Calibration adjusts soil and vegetation parameters so that modeled ET matches satellite-observed ETf on clear-sky days. The calibrated model then fills gaps and provides physically consistent partitioning of ET into evaporation and transpiration.

### Single model vs ensemble calibration

**Single model** (default): Calibrate against one ETf model.

```toml
[calibration]
etf_target_model = "ptjpl"
```

**Ensemble mean**: Calibrate against the average of all available ETf models in the container.

```toml
[calibration]
etf_target_model = "ensemble"
```

**Uncertainty-weighted**: Calibrate against one model, but use multiple models to weight observations. Observations where models agree get higher weight; observations where models diverge get lower weight.

```toml
[calibration]
etf_target_model = "ptjpl"
etf_ensemble_members = ["ssebop", "sims", "geesebal"]
```

For ensemble or uncertainty-weighted calibration, extract multiple ETf models during Step 4:

```bash
swim extract my_project.toml --etf-models ssebop,ptjpl,sims,geesebal
```

### Calibration options

```bash
# More realizations for better uncertainty quantification
swim calibrate my_project.toml --realizations 300

# More parallel workers (faster calibration)
swim calibrate my_project.toml --workers 12

# Both
swim calibrate my_project.toml --workers 12 --realizations 300
```

### What calibration produces

- `data/pestrun/spinup.json` — initial state from spinup run
- `data/pestrun/params.csv` — calibrated parameters
- `data/pestrun/pest/` — full PEST++ project files

## Step 7: Run the Calibrated Model

Generate the output time series:

```bash
swim evaluate my_project.toml
```

### Output files

For each field, SWIM-RS writes a CSV with daily values:

| Column | Description | Units |
|--------|-------------|-------|
| `date` | Date | YYYY-MM-DD |
| `eta` | Actual evapotranspiration | mm/day |
| `etf` | ET fraction (ET/ETref) | - |
| `kcb` | Basal crop coefficient | - |
| `ke` | Evaporation coefficient | - |
| `ks` | Stress coefficient | - |
| `swe` | Snow water equivalent | mm |
| `rain` | Rainfall (liquid precip) | mm/day |
| `snow` | Snowfall (solid precip) | mm/day |
| `melt` | Snowmelt | mm/day |
| `runoff` | Surface runoff | mm/day |
| `dperc` | Deep percolation | mm/day |
| `depl_root` | Root zone depletion | mm |
| `irr_sim` | Simulated irrigation | mm/day |
| `ndvi` | NDVI (interpolated) | - |

Output location: project directory or `--out-dir` if specified.

## International Workflows

For sites outside CONUS, use ERA5-Land and HWSD:

```toml
[data_sources]
met_source = "era5"
snow_source = "era5"
soil_source = "hwsd"
mask_mode = "none"  # no irrigation masking
```

Extract with:

```bash
swim extract my_project.toml --international
```

## Troubleshooting

### EE quota exceeded

Split extraction across multiple days or limit sites:

```bash
swim extract my_project.toml --sites field_001,field_002,field_003
```

### Missing data for some fields

Check container coverage:

```bash
swim inspect data/my_project.swim
```

Re-extract missing data with `--overwrite` for specific components:

```bash
swim extract my_project.toml --no-met --no-properties  # only RS data
```

### Calibration not converging

- Increase realizations: `--realizations 500`
- Check ETf target quality in the container
- Try a different ETf model: change `etf_target_model` in TOML

### Memory issues

Limit parallel workers:

```bash
swim calibrate my_project.toml --workers 2
```

## Config Schema Reference

Each project uses a TOML with a small, consistent set of keys. These are the required and common optional entries.

### Top level
- `project` (string): Project identifier, used in output filenames
- `root` (string): Root directory for resolving paths (usually `"."`)

### `[paths]` (required)
- `project_workspace` (string): Usually `{root}` or `{root}/{project}`
- `data` (string): Data directory under workspace
- `container` (string): Path to the `.swim` container file
- `landsat`, `sentinel` (strings): Remote sensing directories
- `met` (string): GridMET/ERA5 time series directory
- `gis` (string): GIS files directory
- `fields_shapefile` (string, **REQUIRED**): Path to your input shapefile
- `gridmet_mapping` (string): Shapefile for precomputed GridMET cell mapping
- `gridmet_factors` (string): JSON written by mapping step
- `correction_tifs` (string): Folder of monthly ETo/ETr bias correction rasters
- `properties` (string): Directory for properties CSV/JSON
- `irr`, `ssurgo`, `lulc`, `properties_json` (strings): Property file paths
- `snodas_in` (string): SNODAS extract directory

### `[earth_engine]` (optional)
- `bucket` (string): GCS bucket for exports (defaults to Google Drive if omitted)

### `[ids]` (required)
- `feature_id` (string, **REQUIRED**): Unique field identifier column in your shapefile
- `gridmet_join_id` (string): ID column in `gridmet_mapping` shapefile
- `state_col` (string): US state codes column for irrigation mask selection

### `[data_sources]` (required)
- `met_source` (string): `"gridmet"` (CONUS) or `"era5"` (global)
- `snow_source` (string): `"snodas"` (CONUS) or `"era5"` (global)
- `soil_source` (string): `"ssurgo"` (CONUS) or `"hwsd"` (global)
- `mask_mode` (string): `"irrigation"` or `"none"`

### `[misc]` (required)
- `irrigation_threshold` (float): Fraction of field area irrigated to trigger irrigation mode (0.0–1.0)
- `elev_units` (string): Elevation units in shapefile (`"m"` or `"ft"`)
- `refet_type` (string): Reference ET type — `"eto"` (grass) or `"etr"` (alfalfa)
- `runoff_process` (string): `"cn"` (Curve Number) or `"ier"` (infiltration-excess)

### `[date_range]` (required)
- `start_date`, `end_date` (string): YYYY-MM-DD format

### `[crop_coefficient]` (required)
- `kc_proxy` (string): Usually `"etf"`
- `cover_proxy` (string): Usually `"ndvi"`

### `[calibration]` (required for `swim calibrate`)
- `pest_run_dir` (string): Directory for PEST++ files
- `etf_target_model` (string): Calibration target — `"ptjpl"`, `"ssebop"`, `"sims"`, `"geesebal"`, `"eemetric"`, `"disalexi"`, or `"ensemble"` (mean of all available models)
- `etf_ensemble_members` (array of strings, optional): Additional models for uncertainty-weighted calibration. When provided, observation weights are computed from inter-model spread: observations where models agree get higher weight, observations where models diverge get lower weight.
- `workers` (int): Parallel workers for calibration
- `realizations` (int): PEST++ IES ensemble size
- `calibration_dir`, `obs_folder`, `initial_values_csv`, `spinup` (strings): PEST file paths
- `python_script` (string, optional): Custom forward model script

### `[forecast]` (optional)
- `forecast_parameters` (string): Path to calibrated parameter CSV for forecasting

### Notes

- All paths support `{variable}` substitution from earlier definitions
- Shapefile is the canonical source; EE exporters convert it to a FeatureCollection automatically
- See [template.toml](template.toml) for a complete working example

## Next Steps

- Explore the [Examples](../README.md#examples) for more detailed workflows
- Read [Algorithm Description](algorithm_description.md) for model physics
- See [Container Architecture](container_architecture.md) for data structure details
- Check [CLI Cheatsheet](swim_cli_cheatsheet.md) for quick reference
