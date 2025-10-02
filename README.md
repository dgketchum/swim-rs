# swim-rs

Soil Water balance Inverse Modeling using Remote Sensing

swim-rs is an end-to-end toolkit for building field-scale soil water balance models with satellite remote sensing and meteorological forcing. It extracts inputs (NDVI, ET fraction, GridMET/ERA5-Land, SNODAS), prepares field properties, computes irrigation/gw-subsidy dynamics, runs a daily water balance model, calibrates parameters with PEST++ IES, and supports forecasting, analysis, and visualization.

## What It Does

- Data extraction (Google Earth Engine): Landsat/Sentinel NDVI, ET fraction from OpenET/USGS-NHM, CDL/LANID irrigation, SSURGO/HWSD, ERA5-Land daily variables.
- GridMET/NLDAS integration: daily refET, temperature, precip, radiation; monthly correction factors for ETo/ETr.
- Preprocessing: per-field Parquet panels, property tables, irrigation windows, groundwater subsidy, NDVI quantile mapping (Sentinel→Landsat).
- Modeling: daily SWB with snow, runoff (infiltration-excess or SCS Curve Number), NDVI→Kcb sigmoid, dynamic Ke/Ks, irrigation scheduling, root growth.
- Calibration: PEST++ IES via pyemu; writes obs/preds and manages worker/master runs.
- Forecasting/analysis/viz: NDVI analog forecasts with probabilities; metrics vs flux towers; interactive time-series plots.

## Repository Structure

- `swim/` — core config and data container.
  - `config.py` — parses TOML, resolves paths, sets modes (calibrate/forecast) and parameter inputs.
  - `sampleplots.py` — loads/serves large line-delimited model input JSON.
- `data_extraction/` — Earth Engine and met data routines.
  - `ee/` — NDVI/ETf exporters, irrigation/landcover/soils queries, helpers.
  - `gridmet/` — GridMET mapping, factor building, daily downloads (optional NLDAS-2).
  - `snodas/` — SWE CSV → JSON converter.
- `prep/` — preprocessing & feature engineering.
  - `remote_sensing.py` — build per-field RS panels (Parquet), instrument harmonization.
  - `field_properties.py` — landcover/soils/area/irrigation property tables.
  - `field_timeseries.py` — join met + RS (+SWE) into unified per-field daily panels.
  - `dynamics.py` — irrigation windows, groundwater subsidy, ke/kc maxima.
  - `prep_plots.py` — assemble final model input JSON for runs/calibration.
- `model/` — daily SWB components and loop.
  - `obs_kcb_daily.py` (Kcb from NDVI), `compute_snow.py`, `runoff.py`, `k_dynamics.py`, `grow_root.py`.
  - `tracker.py` (state/parameters), `day_data.py` (daily inputs), `obs_field_cycle.py` (driver).
- `calibrate/` — PEST++ builders/runners (`pest_builder.py`, `run_pest.py`, `custom_forward_run.py`).
- `forecast/` — NDVI analog prediction and plotting (`ndvi_forecast.py`).
- `analysis/` — metrics vs flux and OpenET (`metrics.py`).
- `viz/` — Plotly visualizations.
- `tutorials/` — example projects, configs, notebooks, data snippets.

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

## Configuration

`swim.config.ProjectConfig` loads a TOML and resolves path templates (with iterative substitution), sets:
- Paths: project workspace, data subfolders, EE assets, met/RS tables, results, etc.
- IDs/fields: feature IDs, mapping columns.
- Date range, refET type, elevation units, workers/realizations.
- Calibration/forecast parameter sources (CSV/JSON).

See the `tutorials/` subdirectories for example TOMLs and end-to-end runs.

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
