# Irrigation vs. Ks Stress Investigation

## Context

Run 9 (kc_max floor 1.35, max_irr_rate 100 mm/d) showed only modest improvement over Run 8
(R² mean 0.404→0.436, bias -0.185→-0.142). The soil water balance diagnostic on worst-error
days showed irrigation increased 45% but Ks barely moved (0.449→0.455) — depletion remained
~148 mm on worst days. This investigation asks: is the NDVI-derived irrigation flag preventing
irrigation on days when real irrigation is occurring?

## Root Depth Source

Yang et al. 2016, "Global estimation of effective plant rooting depth: Implications for
hydrological modeling" (doi:10.1002/2016WR019392). Uses MODIS IGBP LULC codes.
Cropland (code 12): mean=0.55 m, max=1.12 m. All 59 Example 5 sites use zr_max=1.12 m
(defined in `src/swimrs/container/schema.py`).

## Irrigation Trigger Mechanism

The daily loop (`loop_fast.py:376`) requires two conditions for new irrigation:

```
needs_irrigation = (irr_flag > 0.5) AND (depl_after_et > RAW)
```

Where:
- `irr_flag`: binary flag from NDVI-derived irrigation windows per year
- `depl_after_et`: root zone depletion after ET extraction
- `RAW = MAD * TAW`: readily available water

Even with uncapped irrigation rate (100 mm/d), if irr_flag=0 on a given day, **no irrigation
occurs regardless of depletion level**.

## How irr_flag Is Built

Two sources (`input.py:1077-1144`):

1. **Slope-based window detection** (`calculator.py:997`): Smooths NDVI with 32-day rolling
   mean, finds consecutive positive-slope periods >= min_pos_days, extends by lookback and
   until NDVI drops below threshold. Stored as `irr_doys` per field per year.

2. **NDVI threshold fallback** (`input.py:1103-1142`): For irrigated years, if NDVI > 0.3
   on a day not already flagged, set irr_flag=1. Designed to fill gaps where slope-based
   detection misses plateau and decline phases (e.g., between alfalfa cuttings).

## Bug: NDVI Fallback Dead in no_mask Mode

The fallback at `input.py:1111` searches for NDVI variables in this order:

```python
for var in ["ndvi_irr", "ndvi_inv_irr"]:
```

In no_mask mode (Runs 8/9), the NDVI variable is `ndvi_no_mask` — **not checked**. The
fallback is completely inactive, and irr_flag relies solely on the slope-based window
detection.

**Fix**: Add `"ndvi_no_mask"` to the lookup:

```python
for var in ["ndvi_no_mask", "ndvi_irr", "ndvi_inv_irr"]:
```

## Diagnostic Results (Pre-Fix, Run 9)

Across 38 irrigated fields:

- **57,004 of 74,518 stress-days (76.5%) are blocked** — depl > RAW but irr_flag=0
- On irr_flag=1 days: mean Ks = 0.984 (no stress)
- On irr_flag=0 days: mean Ks = 0.903 (8% reduction, significant stressed tail)

Two site patterns:

| Pattern | Example Sites | Blocking Rate | Mean Ks (blocked) | Notes |
|---------|---------------|---------------|-------------------|-------|
| Real stress | US-Bi1/2, Tw2/3, Ne1/2/3, S2, BAR/RIP/SLM, cotton/soy | 85-99% | 0.51-0.81 | Under-predicting ET |
| False alarm | AZ/CA irrigated (ALARC, JPL, UA, LYS, Almond) | 48-72% | 1.000 | depl barely exceeds RAW, Ks damping prevents stress |

Monthly distribution peaks in Aug-Oct (growing season peak/late season) — exactly when
irrigation demand is highest and under-prediction is worst.

## Implications

The slope-based irrigation window misses significant portions of the growing season. For
sites like Mead NE (US-Ne1/2/3) and the CA wetlands (US-Bi1/2, Tw2/3), 85-99% of days
needing irrigation are blocked. This directly explains the persistent under-prediction on
high-ETo summer days: the model depletes the root zone, Ks drops, and irrigation cannot
replenish because irr_flag=0.

Fixing the NDVI fallback to include `ndvi_no_mask` should substantially expand the irrigation
window and reduce stress-related under-prediction. A re-calibration after the fix is needed
to quantify the improvement.

## Run 10 Results (Post-Fix)

Run 10: irr_flag NDVI fallback fix + kc_max=1.35 + max_irr_rate=100.
Phi progression: 23,482 → 13,325 → 10,007 → 9,313 (best of any run).

### Aggregate (vs Volk, 32 overlapping fields)

| Run | Change | R² mean | R² med | r mean | RMSE mean | Bias mean |
|-----|--------|---------|--------|--------|-----------|-----------|
| 8 | baseline | 0.404 | 0.600 | 0.745 | 1.530 | -0.185 |
| 9 | kc_max+irr_rate | 0.436 | 0.604 | 0.745 | 1.526 | -0.142 |
| **10** | **irr_flag fix** | **0.663** | **0.691** | **0.857** | **1.167** | **+0.189** |

### SWIM vs OpenET models (Run 10, 32 fields)

| Model | R² mean | R² med | r mean | RMSE mean | Bias mean |
|-------|---------|--------|--------|-----------|-----------|
| **swim** | **0.663** | **0.691** | **0.857** | **1.167** | **+0.189** |
| geesebal | 0.411 | 0.535 | 0.787 | 1.551 | -0.685 |
| ptjpl | 0.622 | 0.632 | 0.832 | 1.303 | -0.177 |
| ssebop | 0.608 | 0.650 | 0.860 | 1.268 | -0.214 |
| sims | 0.682 | 0.728 | 0.878 | 1.052 | -0.324 |
| eemetric | 0.547 | 0.613 | 0.811 | 1.364 | -0.118 |
| disalexi | 0.594 | 0.736 | 0.842 | 1.333 | -0.428 |
| ensemble | 0.720 | 0.781 | 0.890 | 1.088 | -0.352 |

SWIM is now competitive with the best individual models (SIMS R²=0.682) and within
0.06 R² of the ensemble. Bias flipped from negative to positive — the irrigation gate
was the dominant error source.

### Sites where SWIM beats ensemble (12 of 32, 37.5%)

| Site | R²_swim | R²_ens | Type |
|------|---------|--------|------|
| US-Tw3 | 0.812 | 0.745 | CA wetland |
| US-KLS | 0.360 | 0.100 | KS grassland |
| US-Ne1 | 0.771 | 0.743 | NE irrigated maize |
| US-Ro1 | 0.623 | 0.551 | IL rainfed |
| US-Ro5 | 0.818 | 0.593 | IL rainfed |
| US-IB1 | 0.795 | 0.727 | IL rainfed |
| LYS_NW | 0.692 | 0.647 | NE lysimeter |
| LYS_SW | 0.721 | 0.575 | NE lysimeter |
| manilacotton | 0.579 | 0.429 | MS cotton |
| RIP760 | 0.888 | 0.803 | MT riparian |
| Almond_Low | 0.811 | 0.809 | CA almond |
| Almond_Med | 0.814 | 0.773 | CA almond |

Up from 7/32 in Run 9. New wins: US-Tw3, LYS_NW, RIP760, Almond_Low, Almond_Med.

### Biggest site-level improvements (Run 9 → 10)

| Site | R²_run9 | R²_run10 | Delta |
|------|---------|----------|-------|
| Almond_Med | -0.633 | 0.814 | +1.447 |
| Almond_High | -0.251 | 0.815 | +1.066 |
| JPL1_Smith5 | -0.372 | 0.737 | +1.109 |
| UA3_JV108 | -0.567 | 0.684 | +1.251 |
| ALARC2_Smith6 | -0.072 | 0.719 | +0.791 |
| RIP760 | 0.382 | 0.888 | +0.506 |
| US-Bi1 | 0.462 | 0.782 | +0.320 |
| Almond_Low | 0.563 | 0.811 | +0.248 |

### Remaining weaknesses

- **Bias flipped positive (+0.189)**: kc_max=1.35 floor may be too high now that
  irrigation is unconstrained, or PEST over-irrigates to compensate for historical
  under-prediction. A re-calibration starting from Run 10 params with kc_max=1.25
  could test this.
- **JPL1_JV114** (R²=-3.022) and **UA3_KN15** (R²=-2.531): still catastrophic.
  These AZ sites may have data quality issues or require site-specific treatment.
- **US-xSL** (0.194) and **Ellendale** (0.187): persistent poor fits at these
  rainfed sites — not irrigation-related.
