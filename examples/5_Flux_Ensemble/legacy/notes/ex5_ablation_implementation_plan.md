# Ex5 Ablation Implementation Plan

Date: 2026-04-03
Scope: Study 2 in `examples/ablation_plan.md` and the current Ex5 workflow in `examples/5_Flux_Ensemble`

## Goal

Implement a controlled Example 5 experiment to test whether using ensemble-member ETf spread as the basis for PEST++ IES observation weighting improves calibration and validation skill relative to a fixed-spread control.

The headline claim is narrow:

- hold the ETf target values constant as the 6-member ensemble mean
- hold sites, dates, realizations, workers, PDC policy, and evaluation policy constant
- change only how ETf capture dates are weighted in calibration

## What Ex5 Does Today

Current Ex5 is already very close to the spread-weighted case:

- `5_Flux_Ensemble.toml` uses `etf_target_model = "ensemble"`
- `ensemble_source = "computed"`
- six configured members are listed in `etf_ensemble_members`
- `calibrate.py` always passes those members into `PestBuilder.build_pest(...)`
- `PestBuilder._write_etf_obs()` applies:

```text
weight = obsval / (std + 0.1)
```

when `members` are provided, and falls back to:

```text
weight = obsval / 0.33
```

when they are not.

This means:

- E1 largely exists already
- E2 is not exposed as an explicit run mode
- the ablation needs implementation control and diagnostics more than a new calibration engine

## Current Code Mismatches With The Study 2 Spec

Three details need to be corrected or made explicit before the experiment is scientifically clean.

### 1. No explicit weighting mode exists

Right now Ex5 implicitly chooses spread-based weighting by passing `cfg.etf_ensemble_members` into `build_pest()`. There is no explicit run switch for:

- `spread`
- `fixed_sd`

Without that switch, E1 vs E2 is not reproducible as a named experiment.

### 2. Spread is currently computed from members plus the ensemble target

`PestBuilder._write_etf_obs()` currently builds `members_and_target = members + [target]`.

For Ex5 that means the spread term is computed from:

- six member ETf series
- plus the ensemble mean target itself

That is not the intended Study 2 definition. The spread basis should be the member ETf spread only. The target mean should not also be counted as a pseudo-member.

### 3. The computed ensemble target is discovered from container contents, not frozen from config

`_get_etf_data(model="ensemble")` currently uses `_discover_etf_models()` rather than the explicit config list. That is risky for an ablation because the target mean can drift if the container contents change.

For the weighting experiment, the target mean must come from one frozen, explicit member list:

- `["ssebop", "sims", "geesebal", "eemetric", "ptjpl", "disalexi"]`

## Important Terminology Fix

The current Study 2 table calls E2 "fixed/uniform" with formula `obsval / 0.33`.

That is not truly uniform weighting. It is better described as:

- fixed-uncertainty weighting
- or magnitude-only weighting

because `obsval / 0.33` still gives larger ETf values larger weights.

This is actually the right control for the main scientific question. It removes spread information while preserving the current magnitude-weighting structure. A truly flat weight would confound two changes at once:

1. removing spread information
2. removing magnitude weighting

Recommendation:

- headline comparison uses `spread` vs `fixed_sd`
- if desired, add a later sensitivity run with fully uniform ETf weights, but not as the primary Study 2 control

## Recommended Experiment Definition

Use one container, one cohort, one period, one ensemble target, and two weighting modes.

### Headline runs

| ID | ETf target | Weighting mode | Formula | Realizations | Purpose |
|---|---|---|---|---|---|
| E1 | Computed 6-member ensemble mean | `spread` | `obsval / (member_std + 0.1)` | 200 | Spread-aware primary run |
| E2 | Computed 6-member ensemble mean | `fixed_sd` | `obsval / 0.33` | 200 | Spread-free control |

### Constants across both runs

- same Ex5 container
- same 60-site calibration cohort
- same validation exclusion policy (`MB_Pch`)
- same calibration period
- same evaluation period
- same ETf target values
- same realizations
- same workers
- same PEST++ options
- same PDC policy
- same mask mode

### Recommended eligibility rule

To keep the comparison clean, both E1 and E2 should use the same ETf observation eligibility mask.

Recommended default:

- observation is eligible if ensemble mean ETf is finite and at least 2 member ETf values are present on that capture date

Then:

- E1 uses the member standard deviation in the denominator
- E2 uses the fixed `0.33` denominator

This avoids a confound where E1 silently drops dates with undefined spread while E2 keeps them.

## Recommended Implementation Approach

This should be a small Ex5-specific extension, not a new framework.

### 1. Add explicit config controls

Extend `ProjectConfig` and Ex5 config handling with:

- `etf_weighting_mode = "spread" | "fixed_sd"`
- `etf_weighting_fixed_sd = 0.33`
- `etf_weighting_spread_floor = 0.1`
- `etf_weighting_min_members = 2`
- optional `etf_weighting_members`

Recommended default behavior:

- if `etf_weighting_members` is unset, use `etf_ensemble_members`
- if `etf_target_model != "ensemble"`, ignore spread weighting and use existing single-model behavior

### 2. Refactor `PestBuilder` weighting logic

Separate three concepts that are currently coupled:

1. target ETf values
2. member spread diagnostics
3. final PEST observation weights

Recommended internal changes:

- add a helper to resolve the exact configured ensemble member list
- compute the ensemble target mean from that explicit list, not from container auto-discovery
- compute spread from member ETf only, not `members + target`
- apply weights based on `etf_weighting_mode`
- preserve one common eligibility mask for both modes

Recommended helper structure:

- `_resolve_target_members()`
- `_get_computed_ensemble_target(fid, members)`
- `_compute_member_spread(fid, members)`
- `_compute_etf_weights(obsval, member_std, member_count, mode, fixed_sd, spread_floor, min_members)`

### 3. Expose the ablation at the Ex5 script level

Extend `examples/5_Flux_Ensemble/calibrate.py` so it can run either weighting mode without hand-editing the main TOML.

Recommended CLI additions:

- `--etf-weighting-mode spread|fixed_sd`
- `--results-tag e1_spread|e2_fixed_sd`
- optional `--fixed-sd`
- optional `--spread-floor`
- optional `--min-members`

Recommended output roots:

- `results/ablation_e1_spread/`
- `results/ablation_e2_fixed_sd/`

Do not create separate containers for this experiment. The target ETf data are identical; only the weighting changes.

### 4. Add a small ablation runner

Add one thin orchestration script, for example:

- `examples/5_Flux_Ensemble/run_weighting_ablation.py`

Responsibilities:

- materialize E1 and E2 config snapshots
- launch both calibrations
- record runtime metadata
- run the canonical evaluation for each result set
- call a summarizer to produce paired-delta outputs

## Required Diagnostics

The weighting experiment needs more than final ET skill metrics. It must prove that only weights changed.

### A. Weight audit

Write a per-observation audit file for each run, for example:

- `etf_weight_audit.csv`

Columns should include:

- `fid`
- `date`
- `obsval`
- `member_count`
- `member_mean`
- `member_std`
- `weight_mode`
- `weight_pre_pdc`
- `weight_final`
- `eligible`

This file is the core ablation proof artifact.

### B. Weight summary by site

Write:

- `etf_weight_summary_by_site.csv`

Metrics:

- nonzero ETf obs
- total ETf weight
- share of total ETf weight
- mean weight
- max weight

This matters because existing Ex5 docs already show site-level ETf weight concentration is a real risk.

### C. Phi diagnostics

Parse and summarize:

- `*.phi.meas.csv`

Write:

- `phi_summary.csv`

Metrics:

- initial phi
- final phi
- per-iteration phi reduction
- wall time to final iteration

### D. Posterior parameter spread

Parse the full `.par.csv` realization table for each run and compute:

- posterior mean by parameter and site
- posterior std by parameter and site
- posterior std by LULC or crop class

Write:

- `posterior_parameter_summary.csv`
- `posterior_parameter_by_class.csv`

This supports the Study 2 secondary claim about information efficiency.

## Evaluation Plan

Reuse the canonical Ex5 evaluation path rather than creating a new benchmark definition.

### Per-run outputs

For each run, write:

- daily metrics
- monthly metrics
- ETf capture-date metrics

using the current Ex5 evaluation code and current validation policy.

### Paired comparison outputs

Add one summarizer, for example:

- `examples/5_Flux_Ensemble/summarize_weighting_ablation.py`

Required outputs:

- `paired_site_deltas_daily.csv`
- `paired_site_deltas_monthly.csv`
- `paired_site_deltas_etf.csv`
- `ablation_summary.csv`

Each paired table should include:

- `fid`
- E1 metric
- E2 metric
- delta
- win indicator
- paired observation count

### Spread-stratified performance

This is a secondary diagnostic, but it is the most direct mechanism test.

Recommended implementation:

- classify each capture date by member spread quartile using the same member-std table used for weighting
- evaluate E1 and E2 on those same capture dates
- report whether E1 gains are concentrated in higher-spread quartiles

Important constraint:

- do this first at ETf capture dates
- do not start with full daily ET quartiles, because daily spread outside capture dates would require interpolation and would confound the mechanism test

## Container and Data Preconditions

This ablation depends on all six member ETf series being present and aligned.

Before running E1/E2, audit:

- `remote_sensing/etf/landsat/ssebop/no_mask`
- `remote_sensing/etf/landsat/sims/no_mask`
- `remote_sensing/etf/landsat/geesebal/no_mask`
- `remote_sensing/etf/landsat/eemetric/no_mask`
- `remote_sensing/etf/landsat/ptjpl/no_mask`
- `remote_sensing/etf/landsat/disalexi/no_mask`
- Landsat NDVI `no_mask`
- GridMET key variables

Required audit outputs:

- member ETf capture count by site
- member overlap count by site and date
- dates with `< 2` member observations

The weighting experiment should fail early if configured members are missing from the container.

## Validation Plan

### Dry run

Run a small subset first, for example 3 to 5 representative sites, with reduced realizations.

The dry run must verify:

- E1 and E2 export identical ETf obs values
- E1 and E2 use the same eligible capture dates
- only weights differ
- phi files are written correctly
- the summary scripts generate paired outputs cleanly

### Full run acceptance checks

Before interpreting results, confirm:

1. Same container fingerprint for E1 and E2
2. Same target member list for both runs
3. Same ETf obs count and same `obsval` series
4. Same evaluation cohort
5. Same evaluation period bounds
6. Same realizations and workers
7. Same PDC settings
8. Different ETf weights in the expected direction

## Recommended Output Layout

Keep ablation artifacts grouped under Ex5 results.

Suggested layout:

- `results/ablation_e1_spread/`
- `results/ablation_e2_fixed_sd/`
- `results/ablation_summary/`

Per-run contents:

- `config_snapshot.toml`
- `runtime.json`
- `etf_weight_audit.csv`
- `etf_weight_summary_by_site.csv`
- `phi_summary.csv`
- `evaluation_daily.csv`
- `evaluation_monthly.csv`
- `evaluation_etf.csv`

Shared summary contents:

- `ablation_summary.csv`
- `paired_site_deltas_daily.csv`
- `paired_site_deltas_monthly.csv`
- `paired_site_deltas_etf.csv`
- `posterior_parameter_summary.csv`
- `spread_quartile_summary.csv`

## Execution Order

1. Freeze terminology and formulas
2. Add explicit weighting config controls
3. Refactor `PestBuilder` so target mean and member spread are separated correctly
4. Add weight audit outputs
5. Extend `calibrate.py` with explicit weighting mode overrides
6. Add the ablation runner
7. Add paired summarization and spread-quartile diagnostics
8. Dry-run on a small site subset
9. Run full E1 and E2
10. Summarize and review before manuscript interpretation

## Key Tradeoffs

### Keep one container

Pros:

- simplest implementation
- guarantees the target data are identical
- avoids unnecessary rebuild time

Cons:

- requires stronger audit discipline to prove target values stayed fixed

### Keep E2 as fixed-SD rather than truly uniform

Pros:

- isolates the value of spread information specifically
- preserves current magnitude-weighting structure

Cons:

- the manuscript must not call it "uniform" without clarification

### Use computed ensemble mean from explicit members

Pros:

- clean scientific definition
- same member set drives both target mean and spread basis

Cons:

- slightly stricter than current container auto-discovery behavior

## Open Decisions

1. Should the manuscript rename E2 from "uniform" to "fixed-uncertainty" or "magnitude-only"?
2. Should the shared eligibility mask require at least 2 members, or a stricter threshold such as 4?
3. Should the Study 2 headline run remain `ensemble_source = "computed"` even if other Ex5 analyses later compare against precomputed OpenET ensemble products?
4. Do you want a third sensitivity run with truly flat ETf weights, or should the implementation stay focused on the two-run E1/E2 ablation only?

## Bottom Line

The clean implementation is:

- one Ex5 container
- one explicit six-member ensemble mean target
- one new weighting mode switch
- one fix so spread is computed from members only
- one paired evaluation and diagnostics layer

That is enough to answer the Study 2 question without changing the broader Ex5 workflow.
