# 2_Fort_Peck: Single-Site Calibration Tutorial

This example demonstrates the complete SWIM-RS calibration workflow for a single flux tower site at Fort Peck, Montana (US-FPe).

## Overview

The tutorial covers:
1. Running an uncalibrated model with default parameters
2. Calibrating model parameters using PEST++ with remote sensing observations
3. Running the calibrated model and validating against flux tower data

## Workflow

Run the notebooks in order:

| Notebook | Description |
|----------|-------------|
| `01_uncalibrated_model.ipynb` | Load data, run uncalibrated model, compare with flux observations |
| `02_calibration.ipynb` | Set up and run PEST++ calibration using SSEBop ETf and SNODAS SWE |
| `03_calibrated_model.ipynb` | Run calibrated model, visualize parameter evolution, validate improvement |

## Configuration

- **Config file:** `2_Fort_Peck.toml`
- **PEST++ worker script:** `custom_forward_run.py`
- **ETf source:** SSEBop only (no ensemble members)

## Data

Pre-built data is provided in `data/`:

| File | Description |
|------|-------------|
| `prepped_input.zip` | Model input data (JSON format) |
| `US-FPe_daily_data.zip` | Flux tower validation data from Volk et al. |
| `gis/flux_fields.shp` | 150m buffer around US-FPe flux tower |

## Expected Results

| Metric | Uncalibrated | Calibrated |
|--------|--------------|------------|
| RMSE | ~0.26 | ~0.16 |
| R² | ~-0.63 | ~0.36 |

Calibration reduces RMSE by >40% and produces better daily ET estimates than SSEBop alone.

## Requirements

- Python environment with SWIM-RS installed
- PEST++ (`pestpp-ies`) for calibration

## References

This example is based on the flux footprint study by Volk et al.:
- Paper: https://www.sciencedirect.com/science/article/pii/S0168192323000011
- Data: https://www.sciencedirect.com/science/article/pii/S2352340923003931
