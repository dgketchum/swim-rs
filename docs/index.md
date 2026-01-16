# SWIM-RS

**S**oil **W**ater **I**nverse **M**odeling with **R**emote **S**ensing

SWIM-RS is an end-to-end toolkit for building field-scale soil water balance
models with satellite remote sensing and meteorological forcing. It extracts
inputs (NDVI, ET fraction, GridMET/ERA5-Land, SNODAS), prepares field
properties, computes irrigation and groundwater dynamics, runs a daily water
balance model, and calibrates parameters with PEST++ IES.

## Documentation

- [Algorithm Description](algorithm_description.md) — Detailed walkthrough of
  the FAO-56 dual crop coefficient model and all physics components
- [Process Package](process_architecture.md) — Architecture of the daily
  simulation loop, dataclasses, and Numba kernels
- [Container Package](container_architecture.md) — Zarr-based data container
  for unified project data management
- [CLI Cheat Sheet](swim_cli_cheatsheet.md) — Quick reference for command-line
  usage

## Quick Start

```bash
# Install
pip install -e .

# Extract data (Earth Engine + GridMET)
swim extract project.toml --add-sentinel

# Prepare inputs
swim prep project.toml --add-sentinel

# Calibrate (PEST++ IES)
swim calibrate project.toml --workers 8 --realizations 300
```

## Key Features

- **Data extraction**: Landsat/Sentinel NDVI, OpenET ET fraction, CDL/LANID
  irrigation mapping, SSURGO soils
- **Meteorology**: GridMET or ERA5-Land daily forcing with optional bias
  correction
- **Modeling**: Daily soil water balance with snow dynamics, SCS runoff,
  NDVI-driven crop coefficients, irrigation scheduling
- **Calibration**: PEST++ IES integration via pyemu for parameter estimation
- **Analysis**: Metrics vs flux towers, NDVI analog forecasting

## Links

- [GitHub Repository](https://github.com/dgketchum/swim-rs)
- [License](https://github.com/dgketchum/swim-rs/blob/main/LICENSE_CC-BY-NC-4.0.txt)
  (CC-BY-NC-4.0)
