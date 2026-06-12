# OPTIMIZATION: 60-Site Calibration Experiments

Systematic comparison of calibration configurations for 60 CONUS flux tower cropland sites.
All cases use the corrected obs pipeline (real ETf targets from container, not SWIM baseline).

## Data Inventory

### DIY ETf (image-level, computed with open-source OpenET packages)

| Model | Sites | Years | Notes |
|-------|-------|-------|-------|
| ptjpl | 60/60 | 1987-2024 | 38 yrs, uniform |
| sims | 60/60 | 1987-2024 | 38 yrs, uniform |
| ssebop | 60/60 | 1987-2022 | 36 yrs, missing 2023-2024 |
| geesebal | 1/60 | — | S2 only; excluded from ensemble |

### EE ETf (pre-computed OpenET EE assets, `_p3` suffix)

| Model | Sites | Years | Notes |
|-------|-------|-------|-------|
| ssebop | 59/60 | 2016-2024 | MB_Pch missing |
| ptjpl | 59/60 | 2016-2024 | MB_Pch missing |
| sims | 59/60 | 2016-2024 | MB_Pch missing |
| geesebal | 59/60 | 2016-2024 | MB_Pch missing |

### Ancillary

- Landsat NDVI: 60/60 (1987-2024)
- Sentinel NDVI: 60/60 (2017-2024)
- SNODAS SWE: 240 monthly files (all sites)
- GridMET: 43 cells
- Flux tower validation: 59/60 (MB_Pch missing)

## Cases

### Case A: 60-site DIY, full period of record

- **ETf source**: DIY (ptjpl + sims + ssebop, 3-model ensemble)
- **Period**: 1987-2024 (38 yr)
- **Sites**: 60
- **Realizations**: 200
- **Workers**: 40
- **PDC removal**: yes
- **Purpose**: Baseline with maximum temporal depth. Longest NDVI + ETf record.

### Case B: 60-site EE, 2016-2024

- **ETf source**: EE assets (ptjpl + sims + ssebop + geesebal, 4-model ensemble)
- **Period**: 2016-2024 (9 yr)
- **Sites**: 60
- **Realizations**: 200
- **Workers**: 40
- **PDC removal**: yes
- **Purpose**: Compare EE assets vs DIY. Test whether 4-model ensemble improves over 3-model.

### Case C: 60-site DIY, 2016-2024

- **ETf source**: DIY (ptjpl + sims + ssebop, 3-model ensemble)
- **Period**: 2016-2024 (9 yr)
- **Sites**: 60
- **Realizations**: 200
- **Workers**: 40
- **PDC removal**: yes
- **Purpose**: Isolate period-of-record effect (Case A vs C) from ETf source effect (Case B vs C).

## Results

### ET vs Flux Tower (mm/day), median statistics

#### Case A: Period of Record, 3 Models (SIMS, PT-JPL, SSEBop)

60 sites, 1987-2024. DIY image-level ETf. Complete (155 min, 3 IES iterations).

| Metric | SWIM | SIMS | PT-JPL | SSEBop | Ensemble Mean |
|--------|------|------|--------|--------|---------------|
| R² | 0.599 | -0.105 | 0.442 | 0.317 | 0.346 |
| r | 0.817 | 0.788 | 0.809 | 0.790 | 0.822 |
| RMSE (mm/day) | 1.223 | 1.725 | 1.724 | 1.576 | 1.538 |
| Bias (mm/day) | -0.410 | -1.200 | -0.848 | -0.688 | -0.888 |

SWIM R² > Ensemble Mean R² at 44/60 sites.

#### Case B: 2016-2024, 4 Models (geeSEBAL, SIMS, PT-JPL, SSEBop)

33 sites with ensemble coverage, 2016-2024. EE asset ETf. Complete (37 min, 3 IES iterations).

| Metric | SWIM | geeSEBAL | SIMS | PT-JPL | SSEBop | Ensemble Mean |
|--------|------|----------|------|--------|--------|---------------|
| R² | 0.576 | 0.411 | 0.520 | 0.543 | 0.448 | 0.596 |
| r | 0.795 | 0.798 | 0.818 | 0.806 | 0.828 | 0.847 |
| RMSE (mm/day) | 1.286 | 1.282 | 1.281 | 1.277 | 1.326 | 1.122 |
| Bias (mm/day) | -0.332 | -0.242 | -0.581 | -0.252 | -0.351 | -0.333 |

### ETf vs OpenET ETf (at Landsat dates), median statistics

#### Case A

| Model | Fields | R² | RMSE | Bias |
|-------|--------|----|------|------|
| PT-JPL | 40 | -0.079 | 0.270 | 0.075 |
| SSEBop | 59 | 0.089 | 0.268 | 0.008 |
| SIMS | 60 | 0.315 | 0.222 | 0.047 |

#### Case B

| Model | Fields | R² | RMSE | Bias |
|-------|--------|----|------|------|
| geeSEBAL | 59 | 0.044 | 0.332 | 0.039 |
| PT-JPL | 59 | 0.086 | 0.300 | -0.028 |
| SSEBop | 59 | 0.172 | 0.275 | 0.075 |
| SIMS | 59 | 0.200 | 0.235 | 0.103 |

### Notes

- **Case A** covers 38 years with 60 sites; SWIM strongly outperforms the 3-model ensemble (median R² 0.599 vs 0.346) because sparse pre-2013 ETf degrades raw OpenET interpolation.
- **Case B** covers 9 years with 33 sites having ensemble coverage; the 4-model EE ensemble is competitive with SWIM (median R² 0.596 vs 0.576) thanks to dense Landsat 8/9 coverage.
- Both cases show negative bias (underestimation) across all models; Case B has lower bias overall.
- Individual OpenET models improve substantially from Case A to B: SIMS median R² goes from -0.105 to 0.520, SSEBop from 0.317 to 0.448.

### Case C

Status: pending
