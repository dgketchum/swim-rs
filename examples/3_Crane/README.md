# 3_Crane: Irrigated Site Calibration Tutorial

This example demonstrates the complete SWIM-RS calibration workflow for an irrigated alfalfa site at Crane, Oregon (S2).

## OpenET Ensemble Members

This example uses the **open source OpenET ensemble members** for ETf observations:
- **SIMS** (Satellite Irrigation Management Support)
- **geeSEBAL** (Google Earth Engine Surface Energy Balance Algorithm for Land)
- **PT-JPL** (Priestley-Taylor Jet Propulsion Laboratory)
- **SSEBop** (Operational Simplified Surface Energy Balance)

To re-extract the remote sensing data from Google Earth Engine, install SWIM-RS with the OpenET optional dependencies:

```bash
pip install swimrs[openet]
```

Pre-extracted data is provided in `data/` so you can run the tutorials without these dependencies.

## Overview

The tutorial covers:
1. Running an uncalibrated model with default parameters
2. Calibrating model parameters using PEST++ with remote sensing observations
3. Running the calibrated model and evaluating improvement

## Workflow

Run the notebooks in order:

| Notebook | Description |
|----------|-------------|
| `01_uncalibrated_model.ipynb` | Load data, run uncalibrated model, compare with OpenET ensemble |
| `02_calibration.ipynb` | Set up and run PEST++ calibration using OpenET ETf and SNODAS SWE |
| `03_calibrated_model.ipynb` | Run calibrated model, visualize parameter evolution, evaluate improvement |

## Configuration

- **Config file:** `3_Crane.toml`
- **PEST++ worker script:** `custom_forward_run.py`
- **ETf source:** OpenET ensemble (SIMS, geeSEBAL, PT-JPL, SSEBop)
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

The uncalibrated model underestimates irrigation and shows poor agreement with the OpenET ensemble. After calibration:
- RMSE reduced by ~50%
- Model learns site-specific irrigation patterns and crop coefficients

## Requirements

- Python environment with SWIM-RS installed
- PEST++ (`pestpp-ies`) for calibration
