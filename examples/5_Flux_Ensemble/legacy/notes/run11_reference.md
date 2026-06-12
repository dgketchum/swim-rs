# Run 11 — Reference Calibration (1995–2025)

**Commits:** `a0b6615` (irr_flag fix), `901baec` (kc_max/irr_rate), `b4972d1` (1995–2025 config)

**Validation status:** As of March 31, 2026, Example 5 headline validation is
governed by `VALIDATION_POLICY.md`. Older February 19, 2026 note tables were
generated before the paired-comparison policy and are deprecated.

## Overview

Run 11 is the reference calibration for Example 5. It calibrates SWIM against
unmasked Landsat ETf from a 6-model OpenET ensemble (SSEBop, PT-JPL, SIMS,
geeSEBAL, eeMETRIC, DisALEXI) at 60 US cropland flux tower sites over the full
1995–2025 period. For validation reporting, `MB_Pch` is excluded, leaving a
59-site evaluation candidate cohort.

## Configuration

| Setting | Value |
|---------|-------|
| Period | 1995-01-01 to 2025-12-31 (11,323 days) |
| Fields | 60 US cropland flux stations |
| mask_mode | none (unmasked NDVI and ETf) |
| ETf target | ensemble (computed mean of 6 Landsat models) |
| ETf members | ssebop, sims, geesebal, eemetric, ptjpl, disalexi |
| Parameters | 8 per site: aw, ndvi_k, ndvi_0, mad, kr_alpha, ks_alpha, swe_alpha, swe_beta |
| Realizations | 200 |
| noptmax | 3 |
| Workers | 40 |
| kc_max floor | 1.35 |
| max_irr_rate | 100 mm/day |
| refet_type | eto |
| runoff_process | cn |
| snow_source | snodas |
| soil_source | ssurgo |

## Data Sources

| Dataset | Instrument | Period | Mask | Source |
|---------|-----------|--------|------|--------|
| NDVI | Landsat + Sentinel (fused) | 1995–2025 | no_mask | EE extractions |
| ETf | Landsat (6 models) | 2016–2025 | no_mask | EE asset extractions |
| Meteorology | GridMET | 1995–2025 | — | gridMET archive |
| Snow | SNODAS | 2004–2025 | — | NOHRSC extractions |
| Soils | SSURGO | static | — | NRCS |
| Land cover | CDL/LANID | annual | — | USDA/Xie et al. |

ETf observations begin in 2016; SWIM is calibrated against ETf only where
observations exist but runs the full water balance from 1995. NDVI coverage
extends to 1995 via Landsat 5/7. SNODAS begins in 2004; earlier years have no
snow observations (SWE initialized to zero).

## Calibration

PEST++ IES with 200-member ensemble, 3 iterations. The irrigation flag is
derived from NDVI phenology: days within NDVI-detected growing season windows
(from `_detect_irrigation_windows`) plus a runtime fallback where NDVI > 0.3
enables irrigation regardless of detected windows. The `no_mask` NDVI variable
feeds the fallback, ensuring irrigation is not artificially suppressed at sites
where no irrigation mask is applied.

### Phi Convergence

| Iteration | Mean Phi |
|-----------|----------|
| 0 (prior) | 38,373 |
| 1 | 12,981 |
| 2 | 9,874 |
| 3 | 9,349 |

Total runtime: 108.6 min (884 model runs, 40 workers, ~1.15 min/run).

## Validation Policy

Canonical Example 5 validation now follows these rules:

- Use the Run 11 parameter file explicitly:
  `/data/ssd1/swim/5_Flux_Ensemble/results/run11_full_period/5_Flux_Ensemble.3.par.csv`
- Use the project container explicitly:
  `/data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble.swim`
- Use Volk 3x3 ET as the headline ET benchmark.
- Use paired SWIM-vs-ensemble comparisons for headline performance claims.
- Treat older pre-fairness tables and any uncommitted "independent" summaries
  as diagnostic-only, not canonical.

See `VALIDATION_POLICY.md` for the full rule set.

## Results — Canonical Paired Benchmark (March 31, 2026 rerun)

### Daily ET vs Flux Tower

58 paired sites with finite SWIM and ensemble metrics.

| Model | R² mean | R² median | RMSE mean | Bias mean |
|-------|---------|-----------|-----------|-----------|
| SWIM | 0.392 | 0.652 | 1.277 | -0.151 |
| ensemble | 0.323 | 0.567 | 1.382 | -0.099 |

The paired rerun lowers the headline SWIM R² substantially relative to the
older February 19, 2026 summaries because SWIM is no longer scored on a broader
date set than the benchmark.

### Monthly ET vs Flux Tower

33 paired sites with finite SWIM and ensemble metrics.

| Model | R² mean | R² median | RMSE mean (mm/mo) | Bias mean (mm/mo) |
|-------|---------|-----------|--------------------|--------------------|
| SWIM | 0.814 | 0.845 | 21.451 | -0.774 |
| ensemble | 0.799 | 0.859 | 21.224 | -5.877 |

At the monthly scale, SWIM remains slightly higher on mean R², while the
ensemble retains a slightly higher median R² and lower RMSE. SWIM remains much
less negatively biased than the ensemble.

## Deprecated Historical Tables

The older February 19, 2026 tables that reported daily SWIM mean R² = 0.654 and
monthly SWIM mean R² = 0.808 were generated before the March 31, 2026 paired
comparison policy. They should not be used as the current Example 5 validation
benchmark.

## Files

| File | Purpose |
|------|---------|
| `5_Flux_Ensemble.toml` | Project config (1995–2025, mask_mode=none, 6-model ensemble) |
| `calibrate.py` | PEST++ IES calibration driver (Run 11, `run11_full_period`) |
| `container_prep.py` | Build the .swim container from extracted data |
| `data_extract.py` | Earth Engine NDVI/ETf extraction + GridMET/SNODAS download |
| `evaluate.py` | Run calibrated model, compare to flux towers and Volk 3×3 OpenET |
| `etf_asset_extract.py` | Extract ETf from OpenET EE asset collections |
| `copy_openet_assets.py` | Copy OpenET EE assets to project bucket |
| `setup_shapefile.py` | Filter master flux station shapefile to cropland sites |
| `notes/VALIDATION_POLICY.md` | Canonical Example 5 comparison policy |

## Reproducing

```bash
# 1. Extract remote sensing data (requires EE auth)
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/data_extract.py

# 2. Extract ETf from OpenET asset collections (requires EE auth)
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/etf_asset_extract.py

# 3. Build container
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/container_prep.py

# 4. Calibrate (~110 min with 40 workers)
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/calibrate.py

# 5. Canonical paired daily benchmark
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/evaluate.py --par-csv /data/ssd1/swim/5_Flux_Ensemble/results/run11_full_period/5_Flux_Ensemble.3.par.csv --container /data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble.swim --openet-source volk

# 6. Canonical paired monthly benchmark
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/evaluate.py --par-csv /data/ssd1/swim/5_Flux_Ensemble/results/run11_full_period/5_Flux_Ensemble.3.par.csv --container /data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble.swim --monthly
```
