# SWIM-RS Validation Policy

This document defines the canonical validation framework for all SWIM-RS
examples. The reported values live in `paper/data/final/` and in the experiment
READMEs; this document states the rules that produce them.

## Naming Convention

Publication-facing prose distinguishes tutorials from experiments. Repo
directories keep their example numbers.

| Paper-facing label | Repo example | Directory | Role |
|---|---|---|---|
| `T1` | Example 1 | `examples/1_Boulder` | Tutorial |
| `T2` | Example 2 | `examples/2_Fort_Peck` | Tutorial |
| `T3` | Example 3 | `examples/3_Crane` | Tutorial |
| None | Example 4 | `examples/4_Flux_Network` | Supporting CONUS flux-network workflow |
| `E0` | Example 6 | `examples/6_Flux_International` | Vegetation-formulation experiment (37 sites outside the E1 cohort) |
| `E1` | Example 5 | `examples/5_Flux_Ensemble` | CONUS flux-ensemble experiment |
| `E2` | Example 6 | `examples/6_Flux_International` | International transfer experiment |
| `E3` | Example 7 | `examples/7_Applied_Water` | Applied-water experiment |

---

## Common Framework

These rules apply to Tutorials T2 and T3, the supporting repo Example 4
workflow, and Experiments E0-E3 (repo Examples 5-7). Tutorial T1 is a
container workflow tutorial and is outside the scope of this validation
policy.

### Publication Data Horizon

For publication-track Experiments E0-E3 (repo Examples 5-7), the default
container data horizon is **through 2025-12-31** for NDVI and ETf.

- This is the default publication policy for E0-E3.
- A shorter or otherwise different analysis window is allowed only when it is
  explicitly declared in the experiment TOML and the companion experiment
  plan.

In other words:

- **Container policy:** carry NDVI and ETf through `2025-12-31`.
- **Experiment policy:** a run may use a narrower period, but that narrower
  period must be intentional and documented.

### Flux Data Role

Flux tower ET is used **exclusively for validation**. No flux data may enter
calibration targets, parameter estimation, or model inputs. Calibration
targets are satellite-derived ETf (and optionally SWE from SNODAS). The
suggested manuscript framing is: "We evaluate SWIM as a field-specific
inverse model calibrated to satellite ETf constraints, with independent flux
data used exclusively for validation."

### Flux Data Version (Repo Examples 4 and 5)

**All validation for repo Examples 4 and 5 must use the Volk v2.1 paired flux
dataset** (`daily_flux_files_2pt1`). Volk provided this dataset directly. It
replaces the per-site `ET_corr` CSVs for all publication metrics. Key
properties:

- 151 sites; 10 of the 161 sites in the per-site CSVs are absent from v2.1.
- The `Closed` column provides the energy-balance-corrected ET reference.
- Closed ET is systematically lower than the per-site `ET_corr` at many sites
  because of updated energy balance closure corrections.
- The record ends in about May 2022, as the per-site CSVs do. Stricter QAQC
  leaves fewer valid days per site.

The v2.1 data is specified via the `[validation] flux_dir` field in each
experiment's TOML configuration file. The canonical path is:

```
{root}/5_Flux_Ensemble/data/daily_flux_files_2pt1
```

**Do not use the original `daily_flux_files` directory for publication
metrics.** It serves diagnostic comparison only.

Experiment E2 is not affected. It uses the multi-network QAQC archive
(AmeriFlux, FLUXNET, ICOS, OzFlux) under
`{root}/6_Flux_International/data/flux_qaqc/`.

### Site Minimum Data Requirements

A site is included in the validation cohort only if its flux record meets
**all** of the following thresholds:

1. **At least 90 valid daily flux observations** (ET or ET_corr finite).
2. **At least 3 qualifying months**, where a qualifying month has at least
   20 valid daily flux observations.

Sites that fail either criterion are excluded from all headline tables and
figures. They may appear in diagnostic or supplementary outputs if labeled
as below-threshold.

### Paired Comparison Methodology

All headline model-vs-model comparisons use **paired evaluation**:

- **Daily:** SWIM and the reference model (SSEBop, OpenET ensemble, or
  per-model ET) are scored on the **exact same valid days** within each
  site. A day is valid only if flux ET, SWIM ET, and reference ET are all
  finite.
- **Monthly:** SWIM and the reference model are scored on the **exact same
  valid months** within each site. The default gate is at least 20 valid
  daily flux observations with finite SWIM and reference totals. A
  benchmark-specific gate applies when the reference is supplied only as a
  full-calendar-month total. Experiment E1 follows the Volk et al. (2024)
  monthly protocol: tower ET is gap-filled as in flux-data-qaqc (tower EToF
  against raw gridMET ETo, IQR outlier filter, 7-day centered mean, linear
  interpolation, times ETo on missing days), a month is a total when more than
  80% of its days carry ET, and it is admitted with at most five filled days;
  SWIM is summed over the full calendar month. No site minimum applies at the
  monthly scale. Pooled rows use every paired site, and station-weighted rows
  and per-site metrics require at least three paired months. The within-E1
  transfer comparison (`within_e1_transfer.py`) uses a 28-valid-day
  full-month gate.

Unpaired or "independent" summaries (where SWIM and the reference are scored
on different day/month sets) are diagnostic only and must not be cited as
headline benchmarks.

### Aggregation and Reporting

- **Aggregate only** across sites with finite metrics for both SWIM and the
  reference model in the paired output.
- **Report with every table or figure:** site count (n), time period,
  evaluation date, and data source.
- **Median over mean** for per-site efficiency (NSE): the mean NSE is
  dominated by a few catastrophic sites with extremely negative values. Report
  both, but use the median as the primary summary statistic for heterogeneous
  cohorts.
- **Land cover stratification** (where applicable): report per-class and
  all-site aggregates side by side so that weak classes are visible rather
  than hidden in the average.

### Metrics

Standard metrics for all examples:

| Metric | Daily units | Monthly units |
|--------|-------------|---------------|
| NSE (Nash–Sutcliffe efficiency) | dimensionless | dimensionless |
| KGE (Kling–Gupta efficiency, 2009) | dimensionless | dimensionless |
| RMSE | mm/day | mm/month |
| MBE (mean bias error, model − observed) | mm/day | mm/month |
| Pearson r | dimensionless | dimensionless |

**NSE labeling.** Per-site NSE is the Nash–Sutcliffe efficiency (1 − SSE/SST,
computed by `sklearn.metrics.r2_score` and stored under the `r2` column in the
evaluation code); it can be negative. Pooled/concatenated tables (e.g. the
Example 5 grouped evaluator and E2 `r²_pool` tables) report squared Pearson
correlation (r²), which is bounded [0, 1]. The two are not comparable. Do not
mix them in a single table without labeling. Win rates (fraction of sites where
SWIM beats the reference) are **not a headline metric** because they collapse
effect size to a sign. Report the paired median metrics above instead.

### Site Exclusion List

`MB_Pch` is excluded from validation wherever it appears. The source inventory
contains two deployment identifiers (`MB_Pch_121315` and `MB_Pch_2014`) that
share one footprint. No deployment-to-flux mapping exists that defines an
unambiguous canonical validation record. The
Volk v2.1 closure-corrected flux archive also contains no canonical `MB_Pch`
file. Together, the unresolved deployment identity and absent canonical flux
file prevent an attributable validation comparison. This is a
performance-independent validation-provenance exclusion; the site may remain
in satellite calibration when its model inputs pass the applicable
completeness gates.

Additional per-example exclusions may be defined in the example-specific
sections below. Any exclusion must state the reason.

### Parameter Files

Always specify the parameter file and container path explicitly in
evaluation commands. Never rely on automatic discovery of `.par.csv` files
in results directories, because multiple iteration files may coexist.

### Outputs Excluded from Citation

Evaluation outputs that do not apply the paired-comparison methodology and the
site minimum data requirements above must not be cited as benchmarks.

---

## Tutorial T2 (Repo Example 2): Fort Peck

### Scope

Single-site tutorial. One unirrigated grassland flux tower (US-FPe) in
eastern Montana.

### Configuration

| Setting | Value |
|---------|-------|
| Sites | 1 (US-FPe) |
| Period | 1987-01-01 to 2022-12-31 |
| Calibration target | PT-JPL Landsat ETf (single model) |
| Meteorology | GridMET |
| Soils | SSURGO |
| mask_mode | irrigation |
| runoff_process | cn |
| PEST++ IES | 200 realizations, 3 iterations |

### Validation Reference

SWIM ET is compared against:
- **PT-JPL ET** (interpolated ETf x ETo from Landsat capture dates)
- **Flux tower ET** (energy-balance-corrected `ET_corr` from US-FPe)

Flux data source: `data/US-FPe_daily_data.csv` (Volk et al.).

### Nuances

- **Single site**: win rates and aggregate statistics are not applicable.
  Report site-level R2, RMSE, and bias only.
- **Two comparison modes**: (a) Landsat capture dates only (sparse), where
  both SWIM and PT-JPL have values; (b) full time series, where PT-JPL is
  linearly interpolated between capture dates and SWIM provides daily
  output. Both modes must report n (number of comparison days).
- **Calibration improvement**: the primary narrative is uncalibrated vs
  calibrated SWIM performance, not SWIM vs PT-JPL head-to-head.
- **mask_mode = irrigation**: unlike Examples 4 and 5, this example uses
  irrigation masking. This is acceptable for a single-site grassland tutorial
  where the distinction has minimal impact.
- **Minimum data threshold**: US-FPe has a multi-decade flux record and
  easily exceeds the 90-day / 3-month minimum.

### Evaluation Workflow

Evaluation is performed in notebook `03_calibrated_model.ipynb`. There is
no standalone `evaluate.py` script for this example.

---

## Tutorial T3 (Repo Example 3): Crane

### Scope

Single-site tutorial. One irrigated alfalfa flux tower (S2) near Crane,
Oregon.

### Configuration

| Setting | Value |
|---------|-------|
| Sites | 1 (S2) |
| Period | 1987-01-01 to 2022-12-31 |
| Calibration target | Ensemble mean of 4 OpenET models |
| Ensemble members | PT-JPL, SIMS, SSEBop, geeSEBAL |
| Meteorology | GridMET |
| Soils | SSURGO |
| mask_mode | irrigation |
| runoff_process | cn |
| PEST++ IES | 20 realizations, 3 iterations |

### Validation Reference

SWIM ET is compared against:
- **OpenET ensemble ET** (mean of 4 models, interpolated ETf x ETo)
- **Flux tower ET** (energy-balance-corrected `ET_corr` from S2)

Flux data source: `data/S2_daily_data.csv`.

### Nuances

- **Single site**: same as Tutorial T2. Report site-level metrics only, no
  win rates.
- **Two comparison modes**: (a) Landsat capture dates only; (b) full time
  series with interpolated OpenET. Both modes must report n.
- **Reduced ensemble**: uses 4 of 6 OpenET models (no eeMETRIC or
  DisALEXI), unlike Experiment E1 which uses all 6. This is a tutorial
  limitation, not a methodological choice.
- **Low realization count**: 20 realizations (vs 200 in Examples 4/5) for
  tutorial speed. Uncertainty estimates from this run are illustrative only.
- **mask_mode = irrigation**: same caveat as Example 2. Acceptable for a
  single irrigated site where the mask correctly identifies irrigation
  status.
- **Calibration improvement**: primary narrative is uncalibrated vs
  calibrated, same as Tutorial T2.
- **Minimum data threshold**: S2 has sufficient flux coverage to exceed the
  90-day / 3-month minimum, but the evaluation period (overlapping with
  Landsat captures 2003-2007) is shorter than Example 2.

### Evaluation Workflow

Evaluation is performed in notebook `03_calibrated_model.ipynb`. There is
no standalone `evaluate.py` script for this example.

---

## Supporting Workflow (Repo Example 4): Flux Network

### Scope

Multi-site CONUS evaluation. 160 US flux tower sites across 6 land cover
classes, calibrated against SSEBop ETf.

### Configuration

| Setting | Value |
|---------|-------|
| Sites | 160 (6 LULC classes) |
| Period | 1987-01-01 to 2025-12-31 (publication default) |
| Calibration target | SSEBop NHM ETf (no_mask) |
| Parameters | 8 per site: aw, ndvi_k, ndvi_0, mad, kr_alpha, ks_alpha, swe_alpha, swe_beta |
| PEST++ IES | 200 realizations, 3 iterations |
| Meteorology | GridMET (with static TIF-based ETo correction) |
| Snow | SNODAS |
| Soils | SSURGO |
| mask_mode | none |
| runoff_process | cn |
| refet_type | eto |

### ETf Masking: no_mask Only

Both calibration and validation use **no_mask** (full footprint) SSEBop ETf
exclusively. The TOML sets `mask_mode = "none"`, and the evaluator loads
ETf from `remote_sensing/etf/landsat/ssebop/no_mask`. The irr/inv_irr
mask-switched paths are retained in the container for diagnostic use but
are not part of the canonical pipeline.

### Publication Window Rule

For Example 4 containers, NDVI and SSEBop NHM ETf should be carried through
`2025-12-31`. If a specific run uses a shorter window, that shorter window
must be declared in the Example 4 TOML and in the governing experiment
plan.

### Canonical Cohorts

- **Container:** 160 flux sites.
- **Validation-only exclusion:** `MB_Pch`, for the deployment-identity and
  canonical-record issue documented in the site exclusion list above.
- **Evaluation candidate cohort:** 159 sites after exclusion. Of these, 151
  have Volk v2.1 flux; the remainder are dropped as "no flux data."
- **Daily paired cohort:** 124 sites that pass the site-minimum filter
  (90 valid days, 3 qualifying months) and have finite SWIM and SSEBop metrics
  on identical valid days.
- **Monthly paired cohort:** 109 sites scored on identical valid months (20
  days/month minimum) that hold at least 10 paired months, the metric floor
  (`MIN_OBS_FOR_METRICS`). Four sites (`US-Ro6`, `US-xUN`, `US-NC4`, `BPLV`)
  have only 6–9 paired months and are excluded.

### Validation Reference

- **Headline benchmark:** SSEBop NHM no_mask ET (interpolated ETf x ETo
  from container-stored Landsat SSEBop ETf at full footprint).
- **Flux tower ET:** Volk v2.1 energy-balance-corrected daily ET (151 of
  160 sites present in v2.1; see Flux Data Version policy above).

### Land Cover Stratification

| Class | Configured sites | Daily eval (paired) |
|-------|------------------|---------------------|
| Croplands | 60 | 45 |
| Grasslands | 30 | 21 |
| Shrublands | 29 | 24 |
| Evergreen Forests | 18 | 14 |
| Mixed Forests | 14 | 12 |
| Wetland/Riparian | 9 | 8 |
| **All** | **160** | **124** |

Counts are the `lc_class` field in `data/gis/flux_fields.shp`, the
authoritative cohort source.
Report per-class and all-site aggregates side by side.

### Known Limitations

Two distinct failure modes appear in the Example 4 reference run (see
`examples/4_Flux_Network/notes/E1_RESULTS.md`). Efficiency is reported as NSE.
The mechanisms below are qualified interpretations, not results demonstrated
by state or event diagnostics:

- **Closed-canopy forests (bias-driven):** Evergreen and Mixed Forests
  correlate well (Pearson r 0.61-0.83) but SWIM systematically **over-predicts**
  ET (MBE +0.59 to +0.82 mm/day). Evergreen median NSE is negative (-0.13) from
  this over-prediction (|MBE|/RMSE 0.42-0.63). This is consistent with dense
  canopy holding NDVI, and therefore Kcb, high year-round, so the soil water
  balance draws its store down little. It is also consistent with partial
  inheritance from the SSEBop target, which also over-predicts forest.
- **Shrublands (weak temporal correspondence):** median NSE is negative (-0.22)
  despite near-zero MBE (-0.08 mm/day). The weakness is low correlation
  (r 0.50), consistent with sparse pulse-driven arid ET that a monotone
  NDVI→Kcb mapping reproduces poorly. SSEBop is also negative here (-0.08).
- **Wetland/Riparian:** over-predict (MBE +0.41 mm/day) but retain positive
  daily median NSE (0.574) and KGE (0.677) on only eight sites; monthly, SWIM
  narrowly trails SSEBop. The evaluation does **not** demonstrate a wetland
  failure through negative efficiency. The one-dimensional-balance domain caveat
  stands a priori (positive bias plus known omitted processes), not via
  negative NSE.
- **Heavy tails:** mean NSE is near zero or negative for both models (SWIM
  daily 0.098 / monthly 0.193; SSEBop monthly -0.344) due to a handful of
  catastrophic sites. **Median NSE is the informative aggregate.**

### Diagnostic-Only Comparisons

- Unpaired outputs (see Outputs Excluded from Citation).
- Runs relying on automatic `par.csv` discovery.
- ETf-only comparisons (`--etf` flag).
- Mask-switched (irr/inv_irr) variants.

### Commands and Inputs

The evaluation commands and input locations are in the Setup, Workflow, and
reference-run sections of `examples/4_Flux_Network/README.md`. Every location
derives from the TOML `root` and is relocated with `SWIM_ROOT`.

---

## Experiment E1 (Repo Example 5): Flux Ensemble

### Scope

Multi-site CONUS cropland evaluation. 60 cropland flux tower sites
calibrated against the 6-model OpenET ensemble mean.

### Configuration

| Setting | Value |
|---------|-------|
| Sites | 60 cropland flux sites |
| Period | 1995-01-01 to 2025-12-31 |
| Calibration target | Ensemble mean of 6 OpenET Landsat models |
| Ensemble members | SSEBop, PT-JPL, SIMS, geeSEBAL, eeMETRIC, DisALEXI |
| ETf observation period | 2016-2025 |
| Parameters | 8 per site |
| PEST++ IES | 200 realizations, 3 iterations, 40 workers |
| Runtime | ~109 min |
| Meteorology | GridMET |
| Snow | SNODAS (2004+) |
| Soils | SSURGO |
| mask_mode | none |
| kc_max floor | 1.35 |
| max_irr_rate | 100 mm/day |
| runoff_process | cn |
| refet_type | eto |

### ETf Masking: no_mask Only

As in the supporting Example 4 workflow, calibration and validation use full-footprint ETf
from `remote_sensing/etf/landsat/{model}/no_mask`.

### Publication Window Rule

For publication-track E1 containers, NDVI and all ensemble-member ETf
inputs should be carried through `2025-12-31`. If a specific experiment uses
a shorter window, that shorter window must be declared in the Example 5 TOML
and in the governing experiment plan.

### Canonical Cohorts

- **Calibration configuration:** 60 cropland flux sites, simple-mean ensemble,
  ETo-corrected, and spread-weighted.
- **Validation-only exclusion:** `MB_Pch`, for the deployment-identity and
  canonical-record issue documented in the site exclusion list above.
- **Evaluation candidates:** 59 sites after the provenance exclusion; 45 pass
  the flux site minimum of 90 valid days and three 20-day months.
- **Daily paired cohort:** 45 sites and 59,516 paired site-days with finite
  SWIM, OpenET ensemble, and flux ET.
- **Primary monthly paired cohort (Volk et al. 2024 protocol):** 51 sites and
  1,548 site-months enter the pooled rows, every site with at least one paired
  month (a gap-filled flux total with at most five filled days, a
  full-calendar-month SWIM total, and a finite OpenET ensemble total). The
  station-weighted rows and the per-site metrics use the 42 sites with at
  least three paired months (1,534 site-months); nine sites with one or two
  paired months are pooled-only, and eight sites with no OpenET monthly file
  are excluded.
- **Within-domain transfer cohort:** 45 daily and 31 monthly sites on common
  support across the two transfer paths, local calibration, generic defaults,
  and flux ET. The monthly comparison uses the 28-valid-day full-month gate
  and does not require Volk benchmark support.

### Validation Reference

- **Headline benchmark:** OpenET ensemble ET from the Volk et al. v2.1 3 x 3
  extraction.
- **Per-model benchmarks:** SSEBop, PT-JPL, SIMS, geeSEBAL, eeMETRIC,
  DisALEXI (each scored on its own paired day/month set with SWIM).
- **Flux tower ET:** Volk v2.1 energy-balance-corrected daily ET from
  cropland stations (see Flux Data Version policy above).

Daily OpenET benchmark ET is reconstructed from the retrieval dates. Capture ET
is divided by same-day OpenET bias-corrected ETo. ETf is then reconstructed
with the OpenET-core temporal support convention, and daily ET is the
reconstructed ETf multiplied by daily ETo. Direct interpolation of ET is
prohibited. The split into retrieval and between-retrieval dates derives from
the separately extracted Volk benchmark, not from SWIM-RS calibration
captures. The within-model, non-retrieval-minus-retrieval support contrast is a
separate diagnostic and must not be presented as a SWIM-OpenET performance
contrast.

### SWIM-OpenET Benchmark Aggregation

This subsection is scoped to the head-to-head SWIM-OpenET benchmark in repo
Example 5. It does not replace aggregation rules for other experiments. The
grouped output follows the aggregation logic of Volk et al. (2024), while
retaining both pooled and station-weighted views of the manuscript's primary
metric triad (KGE, RMSE, and MBE):

1. **Common support remains mandatory.** For each site and timescale, flux ET,
   SWIM ET, and OpenET ET must be finite on the exact same dates or months.
   Let `n_i` be that common-support count for retained site `i`. The two models
   must never be scored on independently filtered observations.
2. **The pooled output contains six metrics.** Concatenate the paired
   observations across all retained sites and concatenate each model's
   corresponding estimates in the same order. Calculate pooled KGE, RMSE,
   signed MBE, Pearson `r`, Pearson `r^2`, and the least-squares slope forced
   through the origin. Here `r^2` is squared Pearson correlation, never NSE.
3. **The station-weighted output contains the primary triad.** Calculate KGE,
   RMSE, and signed MBE at each site first, then report each grouped value as
   `sum(sqrt(n_i) * metric_i) / sum(sqrt(n_i))`. The square-root weighting
   limits domination by the longest records without giving a short record the
   same influence as a long record. Preserve the sign of MBE (modeled minus
   observed).
4. **Name the estimands precisely.** Use *pooled KGE/RMSE/MBE/r/r-squared/slope*
   for statistics calculated on concatenated observations, and
   *sqrt(n)-weighted KGE/RMSE/MBE* for weighted site statistics. Do not describe
   a mean or median of site metrics as pooled.
5. **Both grouped views are emitted by default.** Every grouped result must
   state its aggregation, number of sites, total paired observations,
   timescale, units, benchmark source/version, and (where applicable) the
   `sqrt(n_i)` weight rule. KGE, RMSE, and MBE remain the primary manuscript
   metrics. For the reader-facing E1 SWIM-OpenET intercomparison, lead with the
   `sqrt(n_i)`-weighted KGE, RMSE, and MBE because these aggregate station-level
   metrics and most closely follow Volk et al.'s treatment of station error
   metrics. Report pooled KGE, RMSE, and MBE as the complementary cohort-level
   view; pooled `r`, `r^2`, and slope provide compact diagnostic context. When
   pooled and weighted results support different conclusions, state that
   difference explicitly rather than selecting one estimand as the general
   result. NSE, MAE, per-site medians/IQRs, and model win rates are not part of
   the default grouped output.
6. **Site-effect summaries are secondary and non-default.** The per-site
   metric table remains a required diagnostic and an input to the weighted
   errors. A median paired site effect, its bootstrap interval, or a site-win
   count is produced only when explicitly requested and must be labeled as a
   site-level diagnostic rather than the cohort headline. This optionality
   applies to the *site-effect summary*, not to observational pairing: the
   common-support pairing in item 1 is always required.
7. **Uncertainty preserves the sampling unit.** When confidence intervals are
   reported for pooled or weighted metrics and SWIM-OpenET contrasts, resample
   whole sites, retain every paired observation belonging to each sampled
   site, and use the same bootstrap site draw for both models.

Volk et al. pooled observations for regression statistics and used
`sqrt(n_i)`-weighted station-level MBE, MAE, and RMSE. KGE was not one of their
metrics. Emitting pooled and weighted KGE, RMSE, and MBE here exposes both
the complementary cohort-level view and the primary station-metric view with
moderated record-length weighting, without expanding the manuscript's primary
metric set; pooled `r`, `r^2`, and zero-intercept slope reproduce the associated
regression diagnostics. See [Volk et al. (2024), Nature Water,
doi:10.1038/s44221-023-00181-7](https://doi.org/10.1038/s44221-023-00181-7).

The frozen reporting source for this comparison is
`paper/data/final/e1_openet_benchmark/`; its `MANIFEST.json` defines artifact
roles, precedence, hashes, cohort sizes, and validation status. Tables,
manuscript values, and benchmark figures must read the grouped products there.
The `e2_*` file names under `paper/data/final/` hold supporting E1 evidence.
They keep these names because the frozen hashes key on them. Within that set,
`e2_primary_performance_summary.csv` contains secondary site-median summaries
and is not the headline source.

### Ensemble-Derived Weighting

E1 uses `weight = obsval / (std + 0.1)`, where `std` is the per-timestep
standard deviation across the available ensemble-member ETf values.
Observations where models agree strongly receive higher weight. The controlled
fixed-SD comparison uses `weight = obsval / 0.33`; it is fixed-denominator,
not uniform, weighting. The matched arms have identical initial parameter
ensembles, target values, eligible dates, active observations, model
configuration, and PEST++ settings. Cross-arm phi is not compared because
changing weights changes the objective scale.

### Calibration Target: Simple Mean

The publication calibration target is the **simple per-overpass member
mean** (nanmean) of the 6 ensemble members, not the MAD-filtered
ensemble. The weights above derive from the member standard deviation
and are therefore identical under either target.

No MAD-filtered arm shares the primary E1 configuration, so reader-facing
evidence makes no claim about the effect of the target statistic. Such a claim
requires a matched rerun. **Policy: E1 and the mean ensembles in E2 calibrate
to the simple member mean.**

### Diagnostic-Only Comparisons

- Unpaired summaries (see Outputs Excluded from Citation).
- SWIM-vs-flux "independent" summaries from experimental branches.
- Runs relying on automatic `par.csv` discovery in
  `{root}/5_Flux_Ensemble/results/`.
- Per-model comparisons without an explicitly matched SWIM denominator.
- Any result that does not use the primary configuration and the explicit
  parameter and container paths given in the experiment README.

### Commands and Inputs

The evaluation commands and input locations are in the Inputs and Workflow
sections of `examples/5_Flux_Ensemble/README.md`. Every location derives from
the TOML `root` and is relocated with `SWIM_ROOT`. Evaluation commands write
to explicit scratch directories. A complete bundle is reviewed before it is
promoted to a durable run archive.

---

## Experiments E0 and E2 (Repo Example 6): Flux International

### Scope

Multi-site international cropland evaluation. **66 flux tower sites** spanning
the Americas, Europe, and Oceania, calibrated against a Landsat SSEBop +
PT-JPL ensemble mean ETf, with a two-stage `annual_2yr` water-balance
irrigation classifier (retain threshold plus demand-season test). Meteorology
is ERA5-Land and soils are HWSD. ECOSTRESS ETf is not a calibration target.

### Configuration

| Setting | Value |
|---------|-------|
| Sites | 66 international cropland flux sites |
| Period | 2013-01-01 to 2025-12-31 |
| Calibration target | Landsat ensemble mean (SSEBop + PT-JPL) ETf |
| Irrigation | two-stage `annual_2yr` `use_lulc` classifier (water balance only, no override) |
| PEST++ IES | 200 realizations, 3 iterations, 20 workers, 2 batches |
| Meteorology | ERA5-Land |
| Soils | HWSD |
| Shapefile | `flux_crop_pub_66_150m.shp` |
| mask_mode | none |
| runoff_process | cn |
| refet_type | eto |
| TOML | `6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr.toml` |

### ETf and NDVI Masking: no_mask Only

Experiment E2 uses `mask_mode = "none"` and the international workflow ingests
NDVI and ETf under `no_mask` only. There is no canonical irrigation-mask
switching workflow for E2 publication runs.

### Publication Window Rule

For publication-track Experiment E2 containers, NDVI and ETf should be carried
through `2025-12-31`.

- This applies to Landsat NDVI, Sentinel NDVI, Landsat ETf, and ECOSTRESS
  ETf where those products are part of the container.
- A shorter or otherwise different period is allowed only when it is
  explicitly declared in the experiment TOML and the companion experiment
  plan.

### Validation Reference

- **Headline benchmark:** SWIM ET vs flux tower ET (energy-balance-corrected
  ET_corr from multi-network QAQC archive: AmeriFlux, FLUXNET, ICOS, OzFlux).
- **RS diagnostic benchmark:** native Landsat ETf (ensemble, SSEBop, PT-JPL
  individually), linearly interpolated to daily, multiplied by ERA5-Land ETo.
  Both SWIM and RS ETa are scored against flux on identical paired days.
- Calibration parameters are loaded from the container (ingested by
  batch_runner); no external `.par.csv` is required.

### Known Limitations

- **No OpenET reference:** international sites lack OpenET coverage, so
  the SWIM-vs-OpenET head-to-head is not applicable.
- **ERA5 ETo bias:** ETo from ERA5-Land is systematically biased relative to
  station observations. No station-based correction is applied. All
  publication runs use **raw ERA5-Land ETo**.
- **Multi-network flux data:** QAQC archive spans four networks with
  varying data quality conventions.
- **Site minimum data threshold** applies as in the common framework
  (90 days, 3 qualifying months). The paper's E2 pool is the 47 cohort sites
  with closure-corrected flux ET; sites lacking sufficient flux data are
  excluded automatically.

### Terminology

- **"all days"**: every day with flux. The RS benchmark is the native target
  ETf linearly interpolated between satellite captures, × daily ETo.
- **"overpass" / "capture dates"**: native satellite retrieval dates only.

Reported metrics use **all days**. A capture-date-only pool is a separate mode.

### Commands and Inputs

The evaluation commands and input locations are in the Inputs and Workflow
sections of `examples/6_Flux_International/README.md`. Every location derives
from the TOML `root` and is relocated with `SWIM_ROOT`.
