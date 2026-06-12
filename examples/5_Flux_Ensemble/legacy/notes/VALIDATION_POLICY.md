# Example 5 Validation Policy

**Status date:** March 31, 2026

This note defines the canonical comparison policy for Example 5. It exists
because several older notes and CSV summaries were generated with different
denominators, which produced materially different SWIM performance numbers.

## ETf Masking: no\_mask Only

Both calibration and validation use **no\_mask** (full footprint) ETf
exclusively. The TOML sets `mask_mode = "none"`, and the evaluator loads
per-model ETf from `remote_sensing/etf/landsat/{model}/no_mask`. The
irr/inv\_irr mask-switched paths are not used in the canonical pipeline.

This is consistent with Example 4, which also calibrates and validates with
`mask_mode = "none"`.

## Canonical Cohorts

- **Calibration configuration:** 60 cropland flux sites in Run 11.
- **Excluded from validation outputs:** `MB_Pch`.
- **Evaluation candidate cohort:** 59 sites after exclusion.
- **Canonical daily paired cohort:** 58 sites with finite SWIM and Volk
  ensemble metrics on identical valid days.
- **Canonical monthly paired cohort:** 33 sites with finite SWIM and Volk
  ensemble metrics on identical valid months.

## Canonical Comparison Rules

1. Use the Run 11 parameter file explicitly:
   `/data/ssd1/swim/5_Flux_Ensemble/results/run11_full_period/5_Flux_Ensemble.3.par.csv`
2. Use the project container explicitly:
   `/data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble.swim`
3. Use Volk 3x3 OpenET ET as the headline ET benchmark.
4. Use paired comparisons for headline performance claims:
   - Daily: SWIM and the ensemble are scored on the exact same valid days
     within each site.
   - Monthly: SWIM and the ensemble are scored on the exact same valid months
     within each site, after requiring at least 20 valid daily flux values in a
     month.
5. Aggregate only across sites with finite SWIM and ensemble metrics in the
   paired output.
6. Report the site count and the evaluation date with every table or figure
   derived from Example 5 validation outputs.

## Diagnostic-Only Comparisons

These are useful for troubleshooting, but they are not the canonical Example 5
headline benchmark:

- Historical pre-fairness summaries generated before March 31, 2026.
- Any SWIM-vs-flux full-record or "independent" summaries from experimental or
  uncommitted evaluator branches.
- Any run that relies on automatic `par.csv` discovery in
  `/data/ssd1/swim/5_Flux_Ensemble/results/`, because that directory also
  contains older top-level parameter files that are not the Run 11 reference.
- Per-model comparisons beyond the ensemble when they are not accompanied by an
  explicitly matched SWIM denominator for that same model and time scale.

## Canonical Commands

Run these commands from the repository root:

```bash
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/evaluate.py --par-csv /data/ssd1/swim/5_Flux_Ensemble/results/run11_full_period/5_Flux_Ensemble.3.par.csv --container /data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble.swim --openet-source volk
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/evaluate.py --par-csv /data/ssd1/swim/5_Flux_Ensemble/results/run11_full_period/5_Flux_Ensemble.3.par.csv --container /data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble.swim --monthly
```

## Current Canonical Snapshot

These values come from the March 31, 2026 rerun with the commands above.

### Daily ET vs Flux Tower

58 paired sites with finite SWIM and ensemble metrics.

| Model | R² mean | R² median | RMSE mean | Bias mean |
|-------|---------|-----------|-----------|-----------|
| SWIM | 0.392 | 0.652 | 1.277 | -0.151 |
| Ensemble | 0.323 | 0.567 | 1.382 | -0.099 |

### Monthly ET vs Flux Tower

33 paired sites with finite SWIM and ensemble metrics.

| Model | R² mean | R² median | RMSE mean (mm/mo) | Bias mean (mm/mo) |
|-------|---------|-----------|--------------------|--------------------|
| SWIM | 0.814 | 0.845 | 21.451 | -0.774 |
| Ensemble | 0.799 | 0.859 | 21.224 | -5.877 |

## Deprecated Outputs

The February 19, 2026 validation tables in older Example 5 notes and
performance summaries are retained only as historical context. They were
generated before the paired-comparison policy above and should not be cited as
the current Example 5 benchmark.
