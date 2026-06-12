# Error Structure (what is failing)

- In the full current run (`/data/ssd1/swim/5_Flux_Ensemble/results/evaluation_metrics.csv`), SWIM is competitive on median (`r2_swim=0.54` vs `r2_ensemble=0.509`) but both have heavy bad tails.
- In the flux-study-style 9-site EE subsets, OpenET ensemble is clearly better:
  - `case_d_9site_ee`: median `r2_swim=0.385` vs `r2_ensemble=0.813`
  - `case_e_9site_ee_ls_ndvi`: median `r2_swim=0.609` vs `r2_ensemble=0.813`
- SWIM errors are structured, not random:
  - High-ET sites are where SWIM falls behind most (large negative bias and lower R²).
  - Biggest misses are concentrated at specific towers (e.g., `SLM001`, `US-Bi2` in `case_e`).

# Why SWIM can’t quite match OpenET ensemble in this setup

## 1. Training signal mismatch / incomplete member coverage

- Container ETf coverage is uneven: SIMS is near-complete, PTJPL partial, SSEBOP very sparse, and other models absent in this run.
- So SWIM is often calibrated to a partial proxy of “ensemble,” while evaluation compares to a richer OpenET ensemble signal.
- Relevant code path: `examples/5_Flux_Ensemble/evaluate.py`, `src/swimrs/container/components/exporter.py`.

## 2. Large observation pruning during calibration

- PDC/conflict handling removes about half of ETf constraints (from log parsing: ~23,700 removed vs ~22,837 kept).
- That reduces identifiability and can leave the model fitting an easier subset.
- Relevant behavior: `src/swimrs/calibrate/pest_builder.py`, log `/tmp/v21_pipeline.log`.

## 3. Weighting geometry over-focuses certain groups

- ETf weights (`obsval/(std+0.1)`) plus conflict dropping can concentrate influence in high-weight site/classes, reducing generalization to towers that define the flux benchmark.
- Relevant code: `src/swimrs/calibrate/pest_builder.py`.

## 4. Structural ET ceiling in high-demand periods

- ET is capped by `kc_max * ETr` and water availability; `kc_max` is pre-derived (not calibrated in this workflow).
- Towers with advective/high-demand conditions expose this ceiling and create persistent underestimation.
- Relevant code: `src/swimrs/container/components/calculator.py`, `src/swimrs/process/loop_fast.py`.

## 5. Input coverage issues affect the stronger case jump

- The D→E jump (adding Landsat NDVI context) strongly improved weak sites, indicating input representativeness is a first-order limiter.
- Build logs show missing Landsat NDVI ingest in the current pipeline run.
- Relevant log: `/tmp/v21_pipeline.log`.

# High-leverage ideas to improve performance

## 1. Enforce ensemble parity

- Calibrate with the exact same OpenET member set/temporal handling used in evaluation (`volk` interpolation path), not a partial surrogate.
- Goal: remove train/eval target mismatch.

## 2. Reduce destructive conflict dropping

- Replace hard removal with robust reweighting (Huber/Tukey-style or capped leverage) so constraints are downweighted, not discarded.
- Goal: keep information content while handling inconsistency.

## 3. Rebalance ETf weights

- Cap per-site/group contribution and normalize across site classes so a few high-weight groups don’t dominate.
- Goal: improve transfer to flux-study tower mix.

## 4. Add controlled ET flexibility for high-ET towers

- Introduce a calibratable multiplier or regime-specific adjustment around `kc_max` (bounded, physically constrained).
- Goal: reduce systematic underestimation in advective/high-demand windows.

## 5. Tighten input completeness checks before calibration

- Fail fast if key NDVI or ETf sources are missing, instead of proceeding with silent sparsity.
- Goal: avoid running expensive calibrations on degraded forcing/observation stacks.

## 6. Calibrate/evaluate with stratified objective

- Split objective by ET regime (low/medium/high ET days) and site climate class.
- Goal: stop good performance in easy regimes from masking high-ET failure.
