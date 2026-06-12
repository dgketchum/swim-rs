# Overpass vs Non-Overpass Split Evaluation

**Date:** 2026-04-09

## Question

Does SWIM add value beyond satellite interpolation on days without a direct
Landsat overpass?

## Method

The canonical Ex5 daily evaluation (Run 11 parameters, Volk 3x3 OpenET ensemble)
was split into two subsets per site:

- **Overpass days**: dates where Volk has a non-null ensemble ET value (direct
  Landsat retrieval). On these days, the OpenET benchmark is an actual satellite
  observation.
- **Non-overpass days**: all other paired dates. On these days, the OpenET
  benchmark is linearly interpolated between captures.

Both SWIM and the interpolated OpenET ensemble are scored against
energy-balance-corrected flux tower ET (ET_corr) on each subset independently.
The same paired-comparison rule applies: flux, SWIM, and ensemble must all be
finite on a given day for it to count.

Script: `examples/5_Flux_Ensemble/overpass_split_evaluation.py`

## Results

### Overpass days (45 sites, median 78 days/site)

| Model | med R² | med RMSE (mm/d) | med bias (mm/d) |
|---|---|---|---|
| SWIM | 0.678 | 1.171 | -0.082 |
| Ensemble | 0.745 | 1.084 | -0.406 |

**SWIM win rate: 31% (14/45)**

The OpenET ensemble outperforms SWIM on overpass days. This is expected — the
ensemble has a direct satellite observation while SWIM is running a process model.
The ensemble's lower RMSE (1.08 vs 1.17) reflects the information advantage of
six independent retrievals on capture days.

### Non-overpass days (58 sites, median 922 days/site)

| Model | med R² | med RMSE (mm/d) | med bias (mm/d) |
|---|---|---|---|
| SWIM | 0.648 | 1.134 | -0.001 |
| Ensemble | 0.550 | 1.197 | -0.075 |

**SWIM win rate: 74% (43/58)**

SWIM decisively outperforms the interpolated ensemble between overpasses. The
median R² gap is +0.10 in SWIM's favor, RMSE is 5% lower, and SWIM is
essentially unbiased (median -0.001 mm/day).

## Interpretation

- Overpass days are ~8% of the daily record (median 78 of ~1000 paired days).
  Non-overpass days are ~92%.
- The canonical daily headline (SWIM median R² 0.65 vs ensemble 0.57) is
  dominated by the non-overpass majority, where SWIM's process model adds
  substantial value over linear interpolation.
- On the ~8% of days with a direct satellite observation, the ensemble wins —
  its direct retrieval carries more information than SWIM's model prediction.
- SWIM's bias advantage is strongest on non-overpass days (median -0.001 vs
  -0.075), consistent with the process model tracking daily meteorological
  forcing while interpolation accumulates drift between captures.

## Implications for the manuscript

This result supports the central claim that inverse modeling adds value to
satellite ET products — not by improving on the satellite observation itself,
but by providing physically constrained daily estimates between overpasses that
are superior to temporal interpolation. The framing should be:

- SWIM does not replace satellite ET; it complements it
- The value is in the ~92% of days without an overpass
- On capture days, the satellite product is the better estimate
- A hybrid approach (satellite on overpass days, SWIM between) would likely
  outperform either alone, though this has not been tested

## Reproduction

```bash
cd examples/5_Flux_Ensemble
uv run python overpass_split_evaluation.py \
  --par-csv /data/ssd1/swim/5_Flux_Ensemble/results/run11_full_period/5_Flux_Ensemble.3.par.csv
```

Output: `/data/ssd1/swim/5_Flux_Ensemble/results/overpass_split_metrics.csv`
