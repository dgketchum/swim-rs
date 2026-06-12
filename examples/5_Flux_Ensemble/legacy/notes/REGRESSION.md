# Regression: 60-Site EE Calibration (Feb 2026)

## Summary

The latest 60-site calibration (case_b_60site_ee_9yr, EE ETf, 2016-2024, 200 realizations,
3 iterations) shows a severe SWIM performance regression relative to the 9-site Case J
baseline established during the obs-pipeline fix.

| Run | Sites | SWIM R2 mean | SWIM R2 median | Ens R2 mean | Ens R2 median |
|-----|-------|-------------|----------------|-------------|---------------|
| Case J (9-site, EE) | 9 | 0.584 | 0.641 | 0.653 | -- |
| Case B baseline (60-site, Volk eval) | 32 | -- | 0.640 | -- | 0.770 |
| **Current (60-site, EE, 4-member)** | **34** | **0.180** | **0.419** | **0.424** | **0.579** |

SWIM median R2 dropped from 0.64 to 0.42. Several sites that performed well in Case J
now have negative R2: BAR012 (-0.11), RIP760 (-0.22), SLM001 (0.26 down from 0.19 but
still poor). Meanwhile sites like US-Ro5 (0.79), S2 (0.80), US-IB1 (0.72) perform fine,
indicating the problem is site-dependent rather than a global model failure.

## Phi Breakdown

From the build log:

- Total observations: 394,560
- Valid: 193,246
- ETf nonzero weight (after PDC drop): 8,593
- SWE nonzero weight: 13,334
- Initial actual phi: 4646, final actual phi: 1380

The top-10 phi groups are all ETf (no SWE group appears). ETf dominates phi despite SWE
having more nonzero-weight observations, because SWE weights are tiny (~0.0006 per obs)
while ETf weights are large (~5-8 per obs). This means SWE is not the problem -- ETf
fitting is what's failing.

## ETf Weight Distribution Across Sites

The magnitude-weighted scheme (`weight = obsval / (std + 0.1)`) produces extreme site-level
imbalance:

| Site | ETf w>0 | w_sum | % of total |
|------|---------|-------|------------|
| Almond_Low | 355 | 1324 | ~10% |
| Almond_Med | 337 | 1166 | ~9% |
| RIP760 | 297 | 854 | ~7% |
| SLM001 | 262 | 710 | ~6% |
| ... | ... | ... | ... |
| US-ARM | 153 | 397 | ~3% |
| US-Ne1 | (not in top 25) | <389 | <3% |

Sites with more Landsat captures and higher ETf values receive disproportionate weight.
This is one potential contributor but unlikely the sole cause of the regression, since
the 9-site Case J used the same weighting scheme and performed well.

## Potential Problems

The regression emerged when scaling from 9 to 60 sites. Multiple factors could contribute:

### 1. Data pipeline / container build

- **Are all 60 sites getting valid ETf ingested?** The container build ingests EE ETf for
  6 models x 2 masks x 60 sites. Any silent failures (missing CSVs, date parsing errors,
  field ID mismatches) would leave sites with zero or sparse ETf, yet they'd still have
  parameters being calibrated.

- **NDVI coverage**: Case D showed that missing Landsat NDVI caused negative R2 at 3 of 9
  sites. With 60 sites, how many have adequate Landsat NDVI? Sentinel-only NDVI sites may
  be systematically underperforming. The fused NDVI quantile mapping may also behave
  differently with more sites.

- **ETf date alignment**: The EE asset extracts use scene-level dates. Are these aligning
  correctly with the container's daily time axis for all 60 sites? Off-by-one or timezone
  issues could place ETf values on wrong days.

- **Mask switching**: The irr/inv_irr mask selection depends on irrigation classification.
  For 60 sites spanning diverse geographies, are the irrigation fractions correctly computed?
  A site misclassified as irrigated would get the wrong ETf mask.

### 2. Observation export and PEST build

- **Ensemble mean computation**: With `etf_target_model="ensemble"` and `ensemble_source=
  "computed"`, the exporter averages across all models found under `remote_sensing/etf/
  landsat/`. If "ensemble" was accidentally ingested (from a prior run), it would be
  included in the average, double-counting. Need to verify what models are actually
  discovered.

- **Observation file correctness**: The obs pipeline bug (Cases E-H, where SWIM was fitting
  itself) was fixed for Case J. But the fix involved restructuring `build_pest()` to call
  `_export_observations()` first. Did scaling to 60 sites introduce any new failure mode?
  Are the obs numpy files actually being written with real ETf values for all 60 sites?

- **PDC (prior-data conflict) drops**: 598 observations were dropped as conflicted. Are
  these concentrated at specific sites, effectively removing entire sites from calibration?

- **Localizer construction**: The localizer restricts parameter-observation correlations.
  With 60 sites x 8 parameters = 480 parameters, the localizer matrix is much larger. Is
  it correctly constructed? Are site names being parsed correctly from observation names?

### 3. Weighting scheme

- **No site normalization**: As documented above, weight is not normalized across sites.
  The optimizer can improve phi most efficiently by fitting high-weight sites, potentially
  at the expense of low-weight sites that share parameter groups.

- **Magnitude weighting bias**: High-ETf observations (summer, irrigated) receive more
  weight than low-ETf observations (winter, dormant). This biases the calibration toward
  getting peak-season ET right while potentially distorting the seasonal cycle.

- **Inter-model std computation**: The `etf_std` used for uncertainty weighting comes from
  the 4 ensemble members + the target. With only 4 members (ptjpl missing from the no_mask
  data), low-std dates get very high weight. Are there dates with artificially low std
  that dominate the objective?

### 4. Parameter interactions at scale

- **Parameter space**: 480 parameters (8 per site x 60 sites) with 200 realizations. The
  ratio of parameters to realizations is 2.4:1. PEST++ IES can handle this with
  localization, but the effective degrees of freedom may be insufficient.

- **Shared parameter priors**: The initial parameter values for `mad` and `ndvi_0` are
  conditioned on irrigation fraction. With 60 diverse sites, are the priors appropriate?
  International sites (UA*, JPL*, ALARC*) may have very different characteristics.

- **swe_alpha/swe_beta stuck at bounds**: The parameter change summary shows swe_alpha with
  251 realizations at upper bound and swe_beta with 1040 at upper bound. These snow
  parameters may be poorly constrained for snow-free sites, consuming degrees of freedom.

## Recommended Investigation

A systematic audit of the full pipeline from raw data to final phi:

1. **Container audit**: For each of the 60 sites, verify ETf observation count, NDVI count,
   and meteorology coverage. Flag sites with <50 ETf captures or zero Landsat NDVI.

2. **Observation file audit**: Read the obs numpy files from `pestrun/obs/` and verify they
   contain real ETf values (not model output or all-NaN). Compare obs file values against
   container data for a sample of sites.

3. **Weight audit**: Compute per-site weight sums and phi contributions. Identify sites that
   are effectively unconstrained (near-zero total weight) or dominant (>5% of total weight).

4. **Site-level diagnostics**: For the worst-performing sites (BAR012, RIP760, JPL1_JV114,
   UA3_KN15), examine the ETf time series, NDVI coverage, parameter trajectories, and
   residual patterns to identify whether the problem is data quality, model structure, or
   calibration.

5. **Comparison with Case J sites**: The 9 Case J sites are a subset of the 60. Compare
   their performance in the current run vs Case J. If they degraded, the problem is
   calibration interaction (other sites pulling shared information). If they held, the
   problem is specific to the new sites.

## Files

- Par CSV: `results/case_b_60site_ee_9yr/5_Flux_Ensemble.3.par.csv`
- Phi: `results/case_b_60site_ee_9yr/5_Flux_Ensemble.phi.meas.csv`
- PDC: `results/case_b_60site_ee_9yr/5_Flux_Ensemble.pdc.csv`
- Build log: `/tmp/v21_pipeline.log` (lines 5115-5150, 10267-10870)
- Evaluation: `results/evaluation_metrics.csv`, `results/evaluation_etf_metrics.csv`
- Container: `data/5_Flux_Ensemble.swim`
