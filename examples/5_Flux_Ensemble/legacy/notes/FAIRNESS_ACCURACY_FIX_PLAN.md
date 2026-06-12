# Example 5 Fairness and Accuracy Fix Plan

## Scope

This plan covers Example 5 evaluation, the OpenET benchmark comparisons, and
the paper figures derived from Example 5 outputs.

## Problems To Fix

1. The code uses a matched-site cohort, but most comparisons are not matched on
   identical valid days or months inside each site.
2. `MB_Pch` is described as excluded in some notes, but it is still present in
   the shapefile and saved evaluation outputs.
3. Several figure scripts start from a matched site list but then recompute
   SWIM and OpenET metrics on different observation sets.
4. Example 5 notes, README text, and paper performance text disagree on
   whether the cohort contains 59 or 60 sites.
5. The evaluator does not expose whether a result is a strict paired
   comparison or an independent full-record validation.

## Implementation Plan

### 1. Freeze the Example 5 exclusion and cohort policy

Files:
- `examples/5_Flux_Ensemble/evaluate.py`
- `examples/5_Flux_Ensemble/README.md`
- `examples/5_Flux_Ensemble/notes/openet_etf_source.md`
- `examples/5_Flux_Ensemble/notes/run11_reference.md`
- `paper/performance/example_5_flux_ensemble.md`

Steps:
1. Decide the canonical excluded-site set for Example 5.
2. Enforce it in evaluation code before any metric calculations.
3. Print and save:
   - excluded sites
   - calibration/evaluation cohort size
   - matched-site cohort size
4. Reconcile all Example 5 notes and paper text to the same cohort definition.

Preferred direction:
- Treat `MB_Pch` as excluded if that is the intended policy.
- Separate "calibration cohort" from "benchmark-matched cohort" in the docs.

### 2. Refactor daily evaluation to strict paired observations

Files:
- `examples/5_Flux_Ensemble/evaluate.py`

Steps:
1. Add an evaluation-mode flag with values:
   - `paired`
   - `independent`
2. Define `paired` as: SWIM and the comparator are scored on the exact same
   valid daily observations within a site.
3. Define `independent` as: each model is scored against flux on its own full
   valid record, with separate denominator columns.
4. For each site and comparator, build a daily frame with:
   - `flux`
   - `swim`
   - comparator model ET
5. Compute SWIM and comparator metrics on the exact same valid dates for that
   site/model pair.
6. For ensemble comparisons, define one explicit paired date index used by both
   `r2_swim` and `r2_ensemble`.
7. Save count columns such as:
   - `evaluation_mode`
   - `n_days_paired_ensemble`
   - `n_days_swim_available`
   - `n_days_ensemble_available`
8. For per-model comparisons, save per-model paired counts as well.

Why this matters:
- Current outputs allow SWIM to be scored on dates where the comparator has no
  support, which is not a fair benchmark.
- `independent` mode is still useful, but it needs to be explicit and auditable.

### 3. Refactor monthly evaluation to strict paired months

Files:
- `examples/5_Flux_Ensemble/evaluate.py`

Steps:
1. Build monthly totals for flux, SWIM, and each OpenET model first.
2. In `paired` mode, for each site/model pair, compute metrics only on the
   identical valid month
   set shared by:
   - flux
   - SWIM
   - comparator
3. In `independent` mode, compute each model on its own full valid month set,
   still saving month counts explicitly.
4. Define one ensemble-paired monthly index and use it for both SWIM and
   ensemble metrics.
5. Save per-row counts for:
   - `evaluation_mode`
   - paired months
   - SWIM-only months
   - comparator-only months
6. Ensure the main monthly comparison CSV contains only comparable rows or
   labels non-comparable rows clearly.

### 4. Split paired comparison outputs from diagnostics

Files:
- `examples/5_Flux_Ensemble/evaluate.py`

Steps:
1. Produce mode-specific daily comparison outputs.
2. Produce mode-specific monthly comparison outputs.
3. Produce optional diagnostic outputs listing:
   - unmatched sites
   - dropped dates/months
   - coverage gaps by site and model
4. Include exact denominators and selected mode in the aggregate console
   summary.

Tradeoff:
- More output files, but much cleaner semantics for figures and manuscript
  tables.

### 5. Rewire Example 5 figures to consume paired metrics

Files:
- `paper/figures/fig4_accuracy.py`
- `paper/figures/fig5_monthly_models.py`
- `paper/figures/fig8_cumulative.py`

Steps:
1. Figure 4:
   - Keep pooled triplets for scatter panels.
   - Recompute site-level deltas from one paired date set per site so panel (c)
     is fair.
2. Figure 5:
   - Read only paired monthly outputs for the fairness figure.
   - Use the same paired monthly denominator for all models shown in the panel.
3. Figure 8:
   - For cumulative curves, decide whether to show:
     - sparse observed dates only, or
     - interpolated curves clearly labeled as such.
   - Do not zero-fill missing ensemble or flux days.
   - For seasonal/annual bias, compute SWIM and ensemble totals on the same
     paired dates within a year.

### 6. Normalize narrative counts across docs

Files:
- `examples/5_Flux_Ensemble/README.md`
- `examples/5_Flux_Ensemble/notes/openet_etf_source.md`
- `examples/5_Flux_Ensemble/notes/run11_reference.md`
- `paper/performance/example_5_flux_ensemble.md`

Steps:
1. Standardize the following terms:
   - total calibration cohort
   - excluded-site count
   - daily matched cohort
   - monthly matched cohort
   - independent full-record cohort
2. Add exact rules for how a site qualifies for daily and monthly paired
   comparison.
3. Report exact `n` and evaluation mode in every table and figure caption
   source note.

## Validation Plan

1. Confirm excluded sites, especially `MB_Pch`, are absent from paired outputs.
2. Confirm every `paired` daily row uses the same paired day count for SWIM and
   ensemble.
3. Confirm every `paired` monthly row uses the same paired month count for SWIM
   and ensemble.
4. Confirm every `independent` row retains separate per-model counts.
5. Recompute cohort sizes from CSVs and verify they match README/performance
   claims exactly.
6. Spot-check the largest previous coverage-gap sites:
   - `US-Dk1`
   - `US-Bo1`
   - `US-ARM`
   - `SLM001`

## Deliverables

1. Refactored Example 5 evaluator with strict paired daily and monthly metrics.
2. Mode-specific outputs for `paired` and `independent` evaluation.
3. Diagnostics outputs for unmatched sites and coverage gaps.
4. Updated paper figures that consume the paired outputs directly.
5. Consistent Example 5 and paper text with one canonical cohort/exclusion
   story.
