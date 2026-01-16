# swim-rs

Soil Water balance Inverse Modeling using Remote Sensing

swim-rs is an end-to-end toolkit for building field-scale soil water balance models with satellite remote sensing and meteorological forcing. It extracts inputs (NDVI, ET fraction, GridMET/ERA5-Land, SNODAS), prepares field properties, computes irrigation/gw-subsidy dynamics, runs a daily water balance model, calibrates parameters with PEST++ IES, and supports forecasting, analysis, and visualization.

## Quick Start

Run the full workflow from a project TOML using the `swim` CLI (see the Command-Line Interface section for details).

1) Install (editable recommended during development):
   - `pip install -e .`

2) Extract data (Earth Engine + GridMET):
   - `swim extract examples/5_Flux_Ensemble/5_Flux_Ensemble.toml --add-sentinel`

3) Prepare inputs:
   - `swim prep examples/5_Flux_Ensemble/5_Flux_Ensemble.toml --add-sentinel`

4) Calibrate (PEST++ IES):
   - `swim calibrate examples/5_Flux_Ensemble/5_Flux_Ensemble.toml --workers 8 --realizations 300`

5) Evaluate and compute metrics (optional):
   - `swim evaluate examples/5_Flux_Ensemble/5_Flux_Ensemble.toml --flux-dir <data>/daily_flux_files --openet-dir <data>/openet_flux --metrics-out <results>/metrics`

Notes
- Earth Engine: ensure you are authenticated and the configured bucket/assets are accessible.
- PEST++: `pestpp-ies` must be on PATH.
- Metrics: OpenET daily/monthly CSVs should exist under the `--openet-dir` directories.

## What It Does

- Data extraction (Google Earth Engine): Landsat/Sentinel NDVI, ET fraction from OpenET/USGS-NHM, CDL/LANID irrigation, SSURGO/HWSD, ERA5-Land daily variables.
- GridMET/NLDAS integration: daily refET, temperature, precip, radiation; monthly correction factors for ETo/ETr.
- Preprocessing: per-field Parquet panels, property tables, irrigation windows, groundwater subsidy, NDVI quantile mapping (Sentinel→Landsat).
- Modeling: daily SWB with snow, runoff (infiltration-excess or SCS Curve Number), NDVI→Kcb sigmoid, dynamic Ke/Ks, irrigation scheduling, root growth.
- Calibration: PEST++ IES via pyemu; writes obs/preds and manages worker/master runs.
- Forecasting/analysis/viz: NDVI analog forecasts with probabilities; metrics vs flux towers; interactive time-series plots.

## Repository Structure

- `src/swimrs/` — Python package source
  - `swim/` — core config and input container
    - `config.py` — parses TOML, resolves paths, sets modes (calibrate/forecast)
    - `sampleplots.py` — loads/serves line-delimited model input JSON
  - `data_extraction/` — Earth Engine and meteorological data routines
    - `ee/` — NDVI/ETf exporters, irrigation/landcover/soils queries, helpers
    - `gridmet/` — GridMET mapping, factor building, daily downloads (optional NLDAS-2)
    - `snodas/` — SWE CSV → JSON converter
  - `prep/` — preprocessing & feature engineering
  - `model/` — daily SWB components and loop
  - `calibrate/` — PEST++ builders/runners
  - `forecast/` — NDVI analog prediction and plotting
  - `analysis/` — metrics vs flux and OpenET
  - `viz/` — Plotly visualizations
- `examples/` — example projects, configs, notebooks, small data snippets

## Typical Workflow

1. Configure a project TOML (paths, dates, Earth Engine assets, calibration/forecast sections).
2. Extract remote sensing & met data:
   - EE exporters in `data_extraction/ee/` for NDVI/ETf/properties.
   - Build GridMET correction factors and daily series in `data_extraction/gridmet/`.
   - Optional SNODAS and ERA5-Land extractions.
3. Build per-field RS Parquet: `prep/remote_sensing.py`.
4. Build field properties: `prep/field_properties.py`.
5. Join daily time-series: `prep/field_timeseries.py`.
6. Derive dynamics (irrigation windows, gw subsidy, ke/kc): `prep/dynamics.py`.
7. Assemble `prepped_input.json`: `prep/prep_plots.py`.
8. Run the model loop or calibrate:
   - Direct run: `model/obs_field_cycle.py` (returns ETf/SWE or debug DataFrames).
   - Calibration: `calibrate/pest_builder.py` + `run_pest.py` (PEST++ IES via pyemu).
9. Optional forecasting: `forecast/ndvi_forecast.py` (analog-based NDVI forecast with KDE probabilities).
10. Analyze & visualize: `analysis/metrics.py`, `viz/*`.

## Command-Line Interface

Install the package (editable or standard) and use the `swim` CLI to run the end-to-end workflow from a project TOML.

- Global flags (all subcommands)
  - `--workers` (default 6): parallelism for steps that support it.
  - `--out-dir`: overrides the project root; defaults to the directory containing the TOML.
  - `--sites`: comma-separated site IDs to restrict processing.

- Extract data
  - `swim extract <config.toml> [--add-sentinel] [--etf-models ssebop,ptjpl,...] [--no-snodas] [--no-properties] [--no-rs] [--no-gridmet] [--export drive|bucket] [--bucket BUCKET] [--drive-categorize] [--file-prefix PATH] [--sites ...] [--workers N] [--out-dir PATH]`
  - Exports SNODAS, CDL/irrigation/soils/landcover, NDVI (Landsat and optional Sentinel-2), optional ETF models, and GridMET time series.
  - By default exports to Google Drive (`--export drive`). Use `--drive-categorize` to place outputs into per-category Drive folders (e.g., `swim_properties`, `swim_ndvi`, `swim_etf`). To export to Cloud Storage, use `--export bucket --bucket <name>`. Use `--file-prefix` (default `swim`) to organize outputs under a project path in the bucket (e.g., `--file-prefix swim/projects/flux`).

- Prepare inputs
  - `swim prep <config.toml> [--add-sentinel] [--sites ...] [--workers N] [--out-dir PATH]`
  - Converts EE extracts to Parquet, joins per-station panels, writes properties and SNODAS JSON, builds daily time series and dynamics, and writes `prepped_input.json` and observation arrays.

- Calibrate with PEST++ IES
  - `swim calibrate <config.toml> [--realizations N] [--workers N] [--out-dir PATH]`
  - Builds a PEST++ project, runs spinup (noptmax=0), then IES (noptmax=3). Uses `pestpp-ies` on PATH.

- Evaluate model and compute metrics
  - `swim evaluate <config.toml> [--sites ...] [--forecast-params CSV] [--spinup JSON] [--flux-dir DIR] [--openet-dir DIR] [--metrics-out DIR] [--out-dir PATH]`
  - Runs the model (debug detail) and writes per-site CSVs. When `--flux-dir` and `--openet-dir` are provided, computes daily/overpass/monthly metrics vs flux and OpenET, writing `metrics_by_site.json` and `metrics_monthly.csv` to `--metrics-out` (defaults to `--out-dir`).

Examples
- Extract and prep for the Flux Ensemble example (with Sentinel-2 NDVI):
  - `swim extract examples/5_Flux_Ensemble/5_Flux_Ensemble.toml --add-sentinel`
  - `swim prep examples/5_Flux_Ensemble/5_Flux_Ensemble.toml --add-sentinel`

- Calibrate with 8 workers and 300 realizations:
  - `swim calibrate examples/5_Flux_Ensemble/5_Flux_Ensemble.toml --workers 8 --realizations 300`

- Evaluate and write metrics (assuming OpenET/flux dirs under the data path):
  - `swim evaluate examples/5_Flux_Ensemble/5_Flux_Ensemble.toml --flux-dir <data>/daily_flux_files --openet-dir <data>/openet_flux --metrics-out <results>/metrics`

Prerequisites
- Earth Engine: authenticated account and access to configured assets/bucket.
- PEST++: `pestpp-ies` on PATH and `pyemu` installed.
- OpenET comparison (optional): OpenET daily/monthly CSVs available in the directories passed to `--openet-dir`.

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
