# 3_Crane: Irrigated Site Calibration Tutorial

This example demonstrates the complete SWIM-RS calibration workflow for an irrigated alfalfa site at Crane, Oregon (S2).

## Overview

The tutorial covers:
1. Running an uncalibrated model with default parameters
2. Calibrating model parameters using PEST++ with remote sensing observations
3. Running the calibrated model and evaluating improvement

## Workflow

Run the notebooks in order:

| Notebook | Description |
|----------|-------------|
| `01_uncalibrated_model.ipynb` | Load data, run uncalibrated model, compare with SSEBop |
| `02_calibration.ipynb` | Set up and run PEST++ calibration using SSEBop ETf and SNODAS SWE |
| `03_calibrated_model.ipynb` | Run calibrated model, visualize parameter evolution, evaluate improvement |

## Configuration

- **Config file:** `3_Crane.toml`
- **PEST++ worker script:** `custom_forward_run.py`
- **ETf source:** SSEBop only (no ensemble members)
- **Date range:** 2003-01-01 to 2007-12-31

## Site Details

| Property | Value |
|----------|-------|
| Site ID | S2 |
| Location | Crane, Oregon |
| Crop | Irrigated alfalfa |
| Irrigation | Active since ~1996 (per IrrMapper) |

## Data

Pre-built data is provided in `data/`:

| File | Description |
|------|-------------|
| `prepped_input.zip` | Model input data (JSON format) |

## Key Differences from Fort Peck

| Aspect | 3_Crane | 2_Fort_Peck |
|--------|---------|-------------|
| Land use | Irrigated alfalfa | Unirrigated grassland |
| SWB mode | CN (curve number) | IER |
| Date range | 2003-2007 | 1987-2022 |

## Expected Results

The uncalibrated model underestimates irrigation and shows poor agreement with SSEBop. After calibration:
- RMSE reduced by ~50%
- Model learns site-specific irrigation patterns and crop coefficients

## Requirements

- Python environment with SWIM-RS installed
- PEST++ (`pestpp-ies`) for calibration
