# RSE Manuscript Plan: Examples 4 and 5

Last updated: 2026-02-22
Scope: `examples/4_Flux_Network` + `examples/5_Flux_Ensemble`

## 1) Why this paper can be strong

The current results already suggest a publishable core story:

1. SWIM can outperform single-model ET products when calibration targets are sparse in time (long period-of-record settings).
2. OpenET ensemble targets can be very strong in modern Landsat-dense eras, but SWIM remains competitive and often less biased.
3. Performance is not uniform; it depends on target density, land cover, and calibration design choices.

The manuscript should turn these observations into quantified, controlled evidence.

## 2) What is already established in existing docs

From Example 4 (`1987-2024`, 160 sites, all land covers, SSEBop target):

- Monthly skill is strong in major classes after calibration (overall monthly median R2 around 0.58 to 0.64 depending on version in `ACCURACY.md`).
- SWIM wins most monthly site-level comparisons vs SSEBop in cropland/grassland/evergreen classes.
- Weakest class remains wetland/riparian, with likely missing groundwater subsidy mechanism.

From Example 5 (cropland-focused, ensemble target experiments):

- Case A/B comparisons indicate strong period-of-record sensitivity: long sparse periods can make interpolated OpenET daily ET look worse, changing relative ranking.
- Ensemble comparison shows high benchmark performance in 2016+ dense-observation windows.
- Correct observation pipeline materially improved SWIM performance (Cases E-H vs J), reducing risk that prior conclusions were pipeline artifacts.

## 3) Manuscript contribution targets

Primary contribution:

- A quantitative "cost-benefit frontier" for SWIM calibration as a function of ETf target record length and target architecture (single-model vs ensemble).

Secondary contribution:

- Clear identification of where ensemble ETf constraints add real value and where SWIM process constraints recover skill/bias advantages.

## 4) Hypotheses to test

H1. Record-length hypothesis:

- There is an elbow in performance gain vs calibration cost; beyond a certain period-of-record length, additional years increase runtime more than they improve flux-validated ET skill.

H2. Ensemble-value hypothesis:

- Ensemble ETf calibration targets improve robustness (lower variance across sites, better tail behavior) compared with single-model targets, especially in 2016+ periods.

H3. Process-value hypothesis:

- SWIM provides greatest relative value in conditions where ETf sampling/interpolation is sparse or inconsistent; ensemble has greatest relative value in dense-observation periods.

H4. Stratified-performance hypothesis:

- Relative SWIM vs ensemble performance is strongly stratified by site class (crop type, climate aridity, irrigation regime), not just by overall median metrics.

## 5) Controlled experiment matrix

Use only experiments that can be compared on matched site cohorts and matched evaluation periods.

### Factor A: Period of record

- A1: long period (max available, e.g., 1987+ where valid)
- A2: intermediate period (e.g., 2005+ or 2010+)
- A3: modern period (2016-2024)

### Factor B: Calibration target type

- B1: single-model target (SSEBop baseline)
- B2: computed ensemble target (existing Ex5 approach)
- B3: precomputed OpenET ensemble target (where available and valid)

### Factor C: Site cohort

- C1: full Ex4 all-land-cover cohort for single-target narrative
- C2: harmonized cropland-only cohort present in both Ex4/Ex5 windows
- C3: high-quality overlap cohort (sites with full ensemble coverage and flux overlap)

Minimum comparison set for headline claims:

- Compare A1/A2/A3 under B1 and B2 on C2.
- Compare B1/B2/B3 in A3 on C3.

## 6) Cost accounting plan (core novelty)

For each run, log:

- Wall-clock runtime (total and per IES iteration)
- CPU-hours (`workers x runtime`)
- Number of calibrated parameters and observations
- Effective ETf observation count after filtering/conflict drops
- Storage IO footprint (container read/write volume proxy)

Report:

- Accuracy-cost Pareto curves (`R2`, RMSE, bias vs CPU-hours)
- Marginal gain per added year of record
- Marginal gain per added target complexity (single -> 3/4/6-member ensemble)

## 7) Evaluation metrics and statistics

Core metrics:

- Daily and monthly `R2`, RMSE, bias
- Site-level win rate (SWIM vs baseline model/ensemble)
- Tail metrics (worst-decile site performance)

Statistical evidence:

- Paired site-level deltas for all model comparisons
- Bootstrap confidence intervals for medians and win rates
- Sensitivity checks for coverage mismatch (n-common-site reporting in every table)

Required reporting rule:

- Every comparative number must include exact `n` (sites and site-days/months), period bounds, and target definition.

## 8) Figures and tables for RSE submission

Figure 1: Study design schematic

- Ex4 vs Ex5 cohorts, period windows, target types, and calibration/evaluation flow.

Figure 2: Accuracy vs cost frontier

- Scatter/line of median site skill vs CPU-hours for each experiment family.

Figure 3: Record-length effect

- Delta skill and delta bias from A3 -> A2 -> A1, with uncertainty bars.

Figure 4: Ensemble-value effect

- B1 vs B2 vs B3 paired site deltas for A3/C3, plus win-rate summary.

Figure 5: Stratified performance

- Faceted by climate/crop/irrigation class to show where SWIM gains and where ensemble gains.

Table 1: Harmonized experiment matrix and metadata.
Table 2: Main daily/monthly metrics with CIs.
Table 3: Cost metrics and marginal efficiency (`delta R2 per 100 CPU-hours`).

## 9) Guardrails to protect inference quality

- No mixed-cohort comparisons in headline claims.
- No claims from runs with known pipeline defects.
- Keep calibration objective and evaluation definitions fixed within each comparison family.
- Report both median and distribution tails; avoid mean-only summaries.
- Explicitly flag regimes where SWIM underperforms (for credibility and discussion depth).

## 10) Execution order

1. Freeze comparable cohorts and period windows (C2/C3 + A-levels).
2. Recompute harmonized summary tables from existing runs where possible.
3. Run only missing calibrations needed to complete the minimum comparison set.
4. Build cost logs and accuracy-cost frontier.
5. Produce figure/table package and claim-evidence checklist.

## 11) Claim-evidence checklist (must pass before writing final manuscript text)

Claim C1: "SWIM value depends on period-of-record density."

- Required evidence: controlled A1/A2/A3 comparison on fixed cohort with cost and CI.

Claim C2: "Ensemble ETf targets add value."

- Required evidence: controlled B1/B2/B3 comparison at A3/C3 with paired deltas and uncertainty.

Claim C3: "SWIM remains competitive by bias/robustness."

- Required evidence: absolute bias and worst-decile metrics vs ensemble and single models.

Claim C4: "Findings generalize beyond one site or one crop."

- Required evidence: stratified results across at least two independent site classes.

## 12) Immediate next actions in repo

1. Build one harmonized results index CSV covering Ex4 + Ex5 runs with: run_id, cohort, period, target type, n, metrics, cost.
2. Draft one script/notebook-free reporting path (Python module) to generate the core tables/figures from that index.
3. Update `examples/4_Flux_Network/PLAN.md` and `examples/5_Flux_Ensemble/PLAN.md` with links to this plan so workflow notes and manuscript strategy are separated.

## 13) Execution checklist

- [ ] Freeze C2 and C3 site cohorts and write exact site lists to a versioned CSV.
- [ ] Freeze A1/A2/A3 period windows and record absolute start/end dates for each run.
- [ ] Confirm B1/B2/B3 target definitions and ensemble member lists in config for each run.
- [ ] Build harmonized run registry (`run_id`, cohort, period, target, `n`, status, notes).
- [ ] Mark existing runs that can be reused without rerun (pipeline-valid only).
- [ ] Execute only missing runs needed for minimum comparison set (A-levels x B-levels on C2/C3).
- [ ] Capture cost logs for every run (wall time, workers, CPU-hours, obs counts after filtering).
- [ ] Build daily and monthly paired-delta tables for all headline model comparisons.
- [ ] Compute bootstrap confidence intervals for medians and win rates.
- [ ] Generate coverage diagnostics (`n` by site, model, period) and enforce common-cohort reporting.
- [ ] Produce Figure 1 study schematic.
- [ ] Produce Figure 2 accuracy-cost frontier.
- [ ] Produce Figure 3 record-length effect.
- [ ] Produce Figure 4 ensemble-value effect.
- [ ] Produce Figure 5 stratified performance.
- [ ] Produce Tables 1-3 and verify all include period bounds and sample counts.
- [ ] Complete claim-evidence gate for C1-C4 before drafting manuscript results text.
