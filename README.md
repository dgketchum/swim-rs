# SWIM-RS

Soil Water balance Inverse Modeling using Remote Sensing

SWIM-RS combines the desirable qualities of remote sensing and hydrological modeling into an easy-to-use, rigorous
water use modeling system. The model is a modified FAO-56 soil water balance that uses tunable parameters 
sensitive to remote sensing detections of crop and non-crop behavior. The drivers are commonly
available remote sensing (Landsat and Sentinel satellite) and gridded meteorological data (from GridMET or ERA5-Land). 
Calibration is conducted on a field-by-field basis using state-of-the-art inverse modeling software (PEST++ Iterative Ensemble Smoother; IES). The
calibration procedure uses a custom SWIM-RS module that allows setup and optimization to run in the background, 
saving users from having to orchestrate a complicated calibration procedure.

SWIM-RS makes the most of the data it is given, using inverse modeling to glean as much useful information from
the data it is provided as possible. It can ingest irrigation masks, but will infer irrigation (or groundwater subsidy)
without them.

SWIM-RS is modular. This early version of the software simulates irrigation in an eager and simple fashion, but is 
designed such that an irrigation scenario or pre-determined shedule can easily be used.

SWIM-RS uses modern Python software, reducing the cognitive load on users to keep track of project files, monitor project
state, and investigate how and when results were produced. It has a set of easy to use tools that automatically
record the provenance of data, provide an easy command line interface to check the project inventory, status,
and results, and keep the file system simple. 

SWIM-RS is fast. The core modeling logic uses a fully enclosed, NUMBA-based just-in-time compiler, giving machine
language speed but maintaining algorithm exposure in simple Python functions. A 40-year model run on a single field
takes less than a second. 

**Modern workflow (container-first)**

```
swim extract (EE + met) → swim prep (build .swim container) → build_swim_input (HDF5) → run_daily_loop / calibrate
```

The legacy `prep/` and `model/` packages are deprecated; the container (`src/swimrs/container`) plus process engine (`src/swimrs/process`) are the canonical path. `prepped_input.json` is retained for compatibility (see `DEPRECATION_PLAN.md`).

## Quick start (shipped data, no EE required)

Use the Fort Peck example included in the repo:

```bash
pip install -e .

# Ingest shipped data into a container and export model inputs
swim prep examples/2_Fort_Peck/2_Fort_Peck.toml

# Debug run and per-site CSVs
swim evaluate examples/2_Fort_Peck/2_Fort_Peck.toml --out-dir /tmp/swim_fp
```

To refresh data from Earth Engine and GridMET, authenticate EE and run `swim extract <config.toml> --add-sentinel` before `swim prep`.

## Why swim-rs

- Unified data container: Zarr-backed `.swim` files with provenance, coverage, and validation.
- High-performance physics: numba kernels and typed state via `swimrs.process`.
- PEST++ IES integration: parameter estimation with spinup/localization helpers.
- Remote sensing + met: Landsat/Sentinel NDVI, OpenET ETf, GridMET/ERA5-Land, SNODAS.
- Forecast/analysis: NDVI analog forecasts; metrics vs flux/OpenET; Plotly visualizations.

## Scientific and Software Innovations

### Scientific Innovation

- **Unified Framework Combining Remote Sensing Coverage with Hydrologic Process Fidelity**
  Bridges the spatial coverage and observational power of satellite data with the temporal and physical rigor of process-based soil water balance modeling — enabling field-scale estimation of ET, soil moisture, and irrigation dynamics.

- **Seamless Fusion of Remote Sensing and Water Balance Simulation**
  Dynamically integrates NDVI (Landsat, Sentinel-2) and ET fraction (OpenET/PT-JPL, SSEBop, SIMS) into a FAO-56 dual crop coefficient framework — allowing remote sensing to inform both transpiration and evaporation components on a daily basis.

- **OpenET Ensemble Integration for Calibration and Benchmarking**
  Uses multi-model OpenET ensembles (PT-JPL, SSEBop, SIMS, geeSEBAL) for calibration targets and evaluation metrics, supporting robust comparison across algorithms and against ground truth.

- **Flexible Workflows: Irrigation Classification or LULC-Only Modes**
  Supports both irrigation-mask workflows (e.g., via LANID/IrrMapper in CONUS) and non-mask modes suitable for international contexts using only land cover and NDVI time series.

- **Physically-Bounded Kernels for Hydrologic Processes**
  Implements all major components — snowmelt, infiltration, runoff (curve number and infiltration excess), root dynamics, evaporation, and transpiration — using physically consistent and testable numerical kernels.

- **Validation Across 160+ Flux Tower Sites**
  Field-scale results benchmarked against observed ET from eddy covariance towers, with daily and monthly comparisons to SSEBop and OpenET ensembles. RMSE and R² logged per site/month with automatic diagnostics.

### Modern Scientific Software Architecture

- **Container-Based Data Management with Built-In Provenance**
  All inputs (remote sensing, met, soils, snow, derived features) are stored in a single Zarr-based `.swim` file, with audit logging, coverage tracking, and spatial indexing — enabling traceable and reproducible modeling workflows.

- **Portable HDF5 Inputs for Simulation and Calibration**
  A single HDF5 file (`swim_input.h5`) is generated per project or worker and contains all data needed to run simulations or calibrations independently of the original container or filesystem.

- **Fast, Modular Simulation Engine Using Numba JIT**
  Implements core model kernels as Numba-accelerated functions, achieving 5–10× speedups over standard NumPy code and enabling daily simulations over years for thousands of fields.

- **End-to-End Integration with PEST++ IES**
  Includes built-in support for spinup, control file generation, localization, and parameter bounds for Iterative Ensemble Smoother–based calibration, directly from container inputs.

- **CLI-Driven Workflow with TOML Configs**
  Unified command-line interface (`swim`) supports data extraction, container building, simulation, calibration, and evaluation — all driven by compact and versioned TOML config files.

- **Full Audit Trail and Inventory**
  Every ingest and compute step records metadata, input source, time, and affected fields. Combined with inventory validation and xarray-backed data views, this supports complete transparency from source to output.

- **Structured, Extensible, and Tested Codebase**
  Test-driven design with explicit deprecation handling, parity checks, and typed dataclasses for state, parameters, and properties. Easily extensible for future snow, ET, or runoff modules or input sources.

## Repository map

- `src/swimrs/container` — Zarr container (ingest/compute/export/query)
- `src/swimrs/process` — Simulation engine, HDF5 `SwimInput`, daily loop
- `src/swimrs/cli.py` — `swim` CLI (extract, prep, calibrate, evaluate, inspect)
- `src/swimrs/calibrate` — PEST++ builders/runners
- `src/swimrs/data_extraction` — Earth Engine + meteorology utilities
- `src/swimrs/swim` — config parsing and legacy helpers
- Deprecated: `src/swimrs/prep`, `src/swimrs/model` (see `DEPRECATION_PLAN.md`)
- `examples/` — configs, notebooks, and small data snippets
  - `examples/4_Flux_Network/README.md` — full CONUS flux network workflow
  - `examples/5_Flux_Ensemble/README.md` — cropland flux ensemble workflow

## CLI overview (container-first)

- `swim extract <config.toml>` — Earth Engine + GridMET/ERA5 exports (Drive or bucket)
- `swim prep <config.toml>` — ingest into `.swim`, compute dynamics, export model inputs
- `swim calibrate <config.toml>` — build/run PEST++ IES (requires container)
- `swim evaluate <config.toml>` — debug run, per-site CSVs, optional metrics vs flux/OpenET
- `swim inspect <container.swim>` — container coverage/provenance report

Common flags: `--out-dir` (override project root), `--sites` (restrict IDs), `--workers` (parallel steps), `--add-sentinel`, `--use-lulc-irr` / `--international` (no-mask workflows).

## What it does

- Data extraction (EE): Landsat/Sentinel NDVI, ET fraction from OpenET/USGS-NHM, CDL/LANID irrigation, SSURGO/HWSD, ERA5-Land daily variables.
- Meteorology: GridMET or ERA5-Land daily forcing; optional bias corrections.
- Container compute: merged NDVI, irrigation windows, groundwater subsidy, crop dynamics.
- Modeling: daily SWB with snow, runoff (CN or infiltration-excess), NDVI→Kcb, dynamic Ke/Ks, irrigation scheduling, root growth.
- Calibration: PEST++ IES via pyemu; writes obs/preds and manages worker/master runs.
- Forecasting/analysis/viz: NDVI analog forecasts; metrics vs flux and OpenET; Plotly visualizations.

## Config Schema

Each project uses a TOML with a small, consistent set of keys. These are the required and common optional entries.

- Top level
  - `project` (string): project identifier
  - `root` (string): root directory for resolving paths

- `[paths]` (required)
  - `project_workspace` (string): usually `{root}/{project}`
  - `data` (string): data directory under workspace
  - `landsat`, `landsat_ee_data`, `landsat_tables` (strings)
  - `sentinel`, `sentinel_ee_data`, `sentinel_tables` (strings)
  - `met` (string): GridMET time series directory
  - `gis` (string)
  - `fields_shapefile` (string, REQUIRED): path to a shapefile of fields/polygons
  - `gridmet_mapping` (string, optional): shapefile for precomputed GFID mapping
  - `correction_tifs` (string): folder of monthly correction rasters (ETo/ETr)
  - `gridmet_factors` (string): JSON written by mapping step
  - `properties` (string): directory for properties CSV/JSON
  - `irr`, `ssurgo`, `lulc`, `properties_json` (strings)
  - `snodas_in`, `snodas_out` (strings): SNODAS CSV dir and JSON output
  - `remote_sensing_tables` (string)
  - `joined_timeseries` (string)
  - `dynamics_data` (string)
  - `prepped_input` (string)

- `[earth_engine]` (optional)
  - `fields` (string): EE FeatureCollection asset path (optional; shapefile is used by default)
  - `bucket` (string): optional default Cloud Storage bucket for exports

- `[ids]` (required)
  - `feature_id` (string): field/site identifier column name
  - `gridmet_join_id` (string): ID in `gridmet_mapping` shapefile
  - `state_col` (string, optional): used by irrigation mask logic

- `[misc]` (required)
  - `irrigation_threshold` (float)
  - `elev_units` (string)
  - `refet_type` (string): `eto` or `etr`
  - `runoff_process` (string): `cn` or `ier` (enables NLDAS hours)

- `[date_range]` (required)
  - `start_date`, `end_date` (YYYY-MM-DD)

- `[crop_coefficient]` (required)
  - `kc_proxy` (string)
  - `cover_proxy` (string)

- `[calibration]` (optional, required for calibrate)
  - `pest_run_dir` (string)
  - `etf_target_model` (string)
  - `etf_ensemble_members` (array of strings, optional)
  - `workers` (int), `realizations` (int)
  - `calibration_dir`, `obs_folder`, `initial_values_csv`, `spinup`
  - `python_script` (string, optional): path to a custom forward runner script. Defaults to the packaged script if omitted. Can be overridden via `swim calibrate --python-script`.

- `[forecast]` (optional)
  - `forecast_parameters` (string)

Notes
- Shapefile is the canonical source; when exporting via Earth Engine, the CLI and exporters convert the shapefile to a FeatureCollection under the hood. Providing an EE fields asset is optional and not required for default workflows.

## Configuration

`swimrs.swim.config.ProjectConfig` loads a TOML and resolves path templates (with iterative substitution), sets:
- Paths: project workspace, data subfolders, EE assets, met/RS tables, results, etc.
- IDs/fields: feature IDs, mapping columns.
- Date range, refET type, elevation units, workers/realizations.
- Calibration/forecast parameter sources (CSV/JSON).

See the `examples/` subdirectories for example TOMLs and end-to-end runs.

## Licensing

This software is offered under a dual-license model designed to support both non-commercial research/personal use and commercial applications.

1) Non-Commercial Use — Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

You are free to share/adapt under:
- Attribution — credit, link to license, indicate changes.
- NonCommercial — no commercial use.
- ShareAlike — distribute contributions under the same license.

A copy of the CC BY-NC 4.0 license text is included in `LICENSE_CC-BY-NC-4.0.txt` and online at https://creativecommons.org/licenses/by-nc/4.0/.

Commercial use requires a separate commercial license agreement. Contact: David Ketchum — dgketchum@gmail.com.

## Dependencies

The software uses third-party libraries subject to their own licenses (e.g., Apache-2.0 in the upstream codebase). Ensure compliance with these dependencies per their terms.

---

Disclaimer: The licensing information is for informational purposes only. Consult the full license texts and seek legal advice for specific questions about licensing or compliance.

### EE Exporter Tips

- Prefer `clustered_sample_etf` / `clustered_sample_ndvi` when your fields are geographically clustered; use `sparse_sample_*` when fields are widely dispersed.
- Both clustered functions accept a local shapefile path, an EE FeatureCollection asset ID, or an `ee.FeatureCollection` object; set `feature_id` to the identifier column. If your shapefile includes a state column (e.g., `STATE`), pass `state_col` so the exporters apply IrrMapper (west) or LANID (east) masks appropriately.
- To limit extraction to a few fields, pass `select=[...]` with the desired feature IDs.

Example (Python):

```python
from swimrs.data_extraction.ee.etf_export import clustered_sample_etf
from swimrs.data_extraction.ee.ndvi_export import clustered_sample_ndvi

select_fields = ['043_000130', '043_000128', '043_000161']
shapefile_path = 'examples/1_Boulder/data/gis/mt_sid_boulder.shp'

clustered_sample_etf(
    shapefile_path,
    mask_type='irr', start_yr=2004, end_yr=2023,
    feature_id='FID_1', state_col='STATE', select=select_fields,
    dest='drive', drive_folder='swim', drive_categorize=True,
    model='ssebop', usgs_nhm=True,
)

clustered_sample_ndvi(
    shapefile_path,
    mask_type='irr', start_yr=2004, end_yr=2023,
    feature_id='FID_1', state_col='STATE', select=select_fields,
    satellite='landsat', dest='drive', drive_folder='swim', drive_categorize=True,
)
```
