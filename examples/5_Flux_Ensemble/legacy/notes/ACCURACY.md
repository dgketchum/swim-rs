# Example 5 Accuracy Tracking

## Pre-LULC Baseline: 60-Site, 2016-2024

Production baseline using DIY ETf, PEST++ IES with 4-model ensemble ETf target.
Par.csv: `results/case_b_60site_ee_9yr/5_Flux_Ensemble.3.par.csv`

44 fields had flux tower overlap; 32 of those had OpenET ensemble coverage (Volk source).

### Daily ET (mm/day) — median metrics, all 44 fields with flux data

| Model | n | Median R² | Median RMSE | Median Bias |
|---|---|---|---|---|
| SWIM | 44 | 0.579 | 1.345 | -0.294 |
| geesebal | 32 | 0.535 | 1.409 | -0.631 |
| ptjpl | 32 | 0.632 | 1.145 | -0.219 |
| ssebop | 32 | 0.650 | 1.212 | -0.266 |
| sims | 21 | 0.728 | 1.064 | -0.413 |
| ensemble | 32 | 0.770 | 1.046 | -0.411 |

### Daily ET (mm/day) — median metrics, 32 fields with ensemble coverage

| Model | n | Median R² | Median RMSE | Median Bias |
|---|---|---|---|---|
| SWIM | 32 | 0.640 | 1.248 | -0.201 |
| geesebal | 32 | 0.535 | 1.409 | -0.631 |
| ptjpl | 32 | 0.632 | 1.145 | -0.219 |
| ssebop | 32 | 0.650 | 1.212 | -0.266 |
| sims | 21 | 0.728 | 1.064 | -0.413 |
| ensemble | 32 | 0.770 | 1.046 | -0.411 |

### Per-field win rate: SWIM vs each model (daily R²)

| Comparison | n | SWIM wins |
|---|---|---|
| SWIM vs geesebal | 32 | 62% |
| SWIM vs ptjpl | 32 | 50% |
| SWIM vs ssebop | 32 | 47% |
| SWIM vs sims | 21 | 29% |
| SWIM vs ensemble | 32 | 34% |

## Post-LULC Fix: 60-Site, 2016-2024

*Placeholder — rerun with LULC-driven root_depth and perennial flags.*

## Unmasked OpenET Evaluation

*Placeholder — results after no_mask ETf extraction and ingestion.*

OpenET models are designed to run on unmasked data. For a fair accuracy comparison,
evaluate.py now loads no_mask ETf when available, falling back to irr/inv_irr masked
data. This section will be populated once no_mask ETf CSVs are extracted via
`etf_asset_extract.py` and ingested into the container.
