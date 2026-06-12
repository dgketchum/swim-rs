# Example 5 Regression Deep Dive and Fix Options

Date: 2026-02-17
Scope: Root-cause analysis of severe SWIM accuracy degradation in the latest Example 5 run, with evidence-backed implications and alternative fix paths.

## Executive Summary

The performance collapse is most likely a combined effect of:

1. Missing Landsat NDVI ingestion in the rebuilt container (high confidence, high impact).
2. Very aggressive ETf prior-data conflict (PDC) pruning before final calibration (high confidence, high impact).
3. Existing between-station ETf weighting imbalance amplifying the above (high confidence, medium impact).

The "precomputed ensemble" path was not active in the degraded run, so it is not the direct cause here. Also, the prior interpretation of `et_ensemble_mad` as "uncertainty only" was incorrect. OpenET documentation indicates `et_ensemble_mad` is an ensemble ET estimate derived using MAD-based outlier filtering.

## What Was Audited

- Example 5 markdown docs:
  - `README.md`, `PLAN.md`, `INVESTIGATE.md`, `REGRESSION.md`, `ACCURACY.md`, `ETF_PLAN.md`, `OPTIMIZATION.md`
- Runtime logs and run artifacts:
  - `/tmp/v21_pipeline.log`
  - `/data/ssd1/swim/5_Flux_Ensemble/results/case_b_60site_ee_9yr/*`
  - `/data/ssd1/swim/5_Flux_Ensemble/results/evaluation_metrics.csv`
  - `/data/ssd1/swim/5_Flux_Ensemble/results/case_b_post2016_metrics.csv`
- Live container inventory and time series:
  - `/data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble.swim`
- Relevant implementation:
  - `examples/5_Flux_Ensemble/calibrate.py`
  - `examples/5_Flux_Ensemble/container_prep.py`
  - `examples/5_Flux_Ensemble/etf_asset_extract.py`
  - `src/swimrs/calibrate/pest_builder.py`
  - `src/swimrs/container/components/exporter.py`
  - `src/swimrs/container/components/ingestor.py`

No code or data were modified during this audit.

## Findings by Hypothesis

## 1) "Precomputed ensemble caused the regression"

Status: Not active in the failing run.

Evidence:
- Config is set to computed ensemble:
  - `examples/5_Flux_Ensemble/5_Flux_Ensemble.toml`:
    - `etf_target_model = "ensemble"`
    - `ensemble_source = "computed"`
- Runtime ingest in `/tmp/v21_pipeline.log` shows ETf ingestion for:
  - `ssebop`, `sims`, `geesebal`, `eemetric`
  - No `ensemble` ingestion in this run.
- Container confirms ETf models present are those 4 only.

Conclusion:
- Precomputed ensemble was not used as target in the degraded run and is not the direct root cause.

Important clarification:
- `examples/5_Flux_Ensemble/etf_asset_extract.py` maps `model == "ensemble"` to:
  - `select("et_ensemble_mad").divide(10000).rename("etf")`
- `ETF_PLAN.md` documents the same.
- OpenET docs describe `et_ensemble_mad` as an ET estimate produced after MAD-based filtering of member models.
- EE asset inspection for `projects/openet/assets/ensemble/conus/gridmet/landsat/v2_1` confirms only `et_ensemble_mad` and `et_ensemble_mad_count` bands are present.
- Therefore, using `et_ensemble_mad` as the precomputed ensemble target is conceptually valid for this workflow.

Remaining caveat:
- Keep band semantics and scaling documented and verified whenever switching between OpenET collections/products (for example ET vs ETf representations across products), but this is not the identified cause of the present regression.

## 2) "Between-station weighting misuse caused disproportionate fitting"

Status: True contributor, likely amplifier rather than sole trigger.

Evidence:
- ETf weights use magnitude weighting:
  - `src/swimrs/calibrate/pest_builder.py`:
    - `weight = obsval / (std + 0.1)`
- Reconstructed pre-PDC ETf weights from current container + obs files:
  - Top 5 sites carry about 18.0% of total ETf weight.
  - Top 10 sites carry about 30.2%.
- After PDC removal (using logged "Removed ... leaving ..." values):
  - Top 5 carry about 20.5%.
  - Top 10 carry about 32.7%.

Interpretation:
- Weight concentration is real.
- Concentration increases after PDC pruning.
- This can systematically bias updates toward high-capture/high-magnitude sites.

## 3) "Data pipeline / container mechanics caused degradation"

Status: Strong evidence this is a primary root cause.

Evidence from current run log:
- `/tmp/v21_pipeline.log` immediately reports no Landsat NDVI CSVs:
  - `no_csv_files` for:
    - `.../landsat/extracts/ndvi/irr`
    - `.../landsat/extracts/ndvi/inv_irr`
- Those directories are indeed empty on disk.
- Sentinel NDVI files are present and ingested.
- Container path inventory shows sentinel NDVI and merged NDVI only; no Landsat NDVI datasets.

Why this matters:
- This exactly matches the previously documented failure mode:
  - `INVESTIGATE.md` states Case D was degraded under sentinel-only NDVI.
  - Adding Landsat NDVI in Case E substantially recovered performance at key sites (notably RIP760 and BAR012).

Current behavior consistency:
- In this degraded run, RIP760 and BAR012 are again major SWIM failures while ensemble remains good at those sites.
- Pattern is consistent with NDVI driver mismatch and not with a broad ET data collapse.

## 4) "PDC mechanics / conflict handling caused degradation"

Status: Strong evidence this is a second primary root cause.

Evidence:
- `calibrate.py` performs a PDC pre-pass with:
  - `noptmax = -1`, `reals = 5` (very small ensemble) before main run.
- Build diagnostics from `/tmp/v21_pipeline.log`:
  - Before conflict dropping:
    - ETf valid = 17,926
    - ETf nonzero weight = 17,926
  - After applying conflict file:
    - ETf nonzero weight = 8,593
- This is about a 52% drop in ETf-constraining observations.
- Logged per-site removal indicates broad and often severe pruning:
  - Mean removed fraction is about 51%.
  - Several sites exceed 70% removal.

Additional note:
- Final saved `case_b_60site_ee_9yr/5_Flux_Ensemble.pdc.csv` has 598 rows and mostly SWE rows.
- The large ETf pruning occurred earlier through the temporary PDC file generated by the pre-pass and applied during rebuild.

Interpretation:
- Conflict pruning appears too aggressive and likely unstable with `reals=5` for a 480-parameter, 60-site system.
- This strips much of the ETf signal before optimization.

## 4a) PEST++ manual guidance on PDC and implications for this workflow

Key points from `../pestpp/documentation/pestpp_users_manual.md`:
- PESTPP-IES already performs prior-data conflict detection after evaluating the prior ensemble (or restart).
- `ies_drop_conflicts` controls whether conflicted non-zero-weight observations are removed from upgrades.
- Default for `ies_drop_conflicts` is `false`.
- Manual characterizes dropping conflicted observations as a "draconian" but sometimes useful bias-variance tradeoff.

How current Example 5 workflow differs:
- `src/swimrs/calibrate/pest_builder.py` forces `ies_drop_conflicts = "true"`.
- `examples/5_Flux_Ensemble/calibrate.py` adds a custom pre-pass (`noptmax=-1`, `reals=5`) and then manually zeros weights using the generated conflict file before the final run.

Implication:
- We are combining built-in PEST++ dropping with an additional custom pre-pruning step.
- The custom pre-pass uses a very small ensemble (`reals=5`) and appears to over-prune ETf constraints before the main calibration.
- This strongly suggests the custom pre-pass is low value in its current form and may be a major source of avoidable degradation.

## 5) "Something else"

Observed secondary factors:
- ETf coverage exists for all 60 sites when considering irr + inv_irr masks together; this is not an ETf-missing-data collapse.
- ETf models remain competitive/stable while SWIM drops:
  - Ensemble median R2 changed only slightly relative to prior baseline.
  - SWIM median R2 dropped markedly.

Interpretation:
- Problem is concentrated in SWIM calibration inputs/constraints (NDVI + weight/conflict pipeline), not in raw ETf model quality alone.

## Quantified Performance Change

Using common evaluated fields (`evaluation_metrics.csv` vs `case_b_post2016_metrics.csv`):

- SWIM mean R2: 0.305 -> 0.201
- SWIM median R2: 0.584 -> 0.377
- Ensemble mean R2: 0.416 -> 0.424 (stable)
- Ensemble median R2: 0.596 -> 0.579 (small change)

For fields with ensemble overlap:
- Current run: SWIM mean/median R2 = 0.180 / 0.419
- Prior run: SWIM mean/median R2 = 0.294 / 0.576

Largest SWIM drops include RIP760 and BAR012, consistent with prior known sensitivity to Landsat NDVI availability.

## Implications

## Scientific/interpretation implications

- Current calibration may underrepresent true ETf constraints due to over-pruned observations.
- Site-level conclusions (especially for degraded fields) are likely confounded by pipeline artifacts.
- Comparisons claiming structural SWIM weakness versus ensemble are likely inflated in this run.

## Operational/reproducibility implications

- Rebuild success currently depends on hidden data presence assumptions (Landsat NDVI extracts expected but not guaranteed).
- A run can complete "successfully" while silently missing critical NDVI inputs.
- PDC behavior is sensitive to pre-pass setup and can dramatically change objective composition.

## Future risk implications

- The largest forward-looking risk in this workflow is not `et_ensemble_mad` itself; it is brittle conflict-handling strategy (custom pre-pass plus built-in dropping) and silent input-data gaps (for example missing Landsat NDVI directories).

## Alternative Fix Paths (No-Code and Code Options)

## Option A: Data-only recovery (lowest risk, fastest to validate)

Actions:
- Restore Landsat NDVI extracts in:
  - `data/remote_sensing/landsat/extracts/ndvi/irr`
  - `data/remote_sensing/landsat/extracts/ndvi/inv_irr`
- Rebuild container.
- Re-run calibration/evaluation unchanged.

Expected impact:
- High probability of major recovery on degraded sites (especially RIP760, BAR012-like behavior).

Pros:
- No algorithm changes.
- Directly addresses known prior failure mode.

Cons:
- Does not address potential over-pruning from PDC.

## Option B: Keep current data, remove custom PDC pre-pass (isolation test)

Actions:
- Run with `pdc_remove=False` in `calibrate.py` so the custom `noptmax=-1, reals=5` pre-pass is skipped.
- Keep built-in PEST++ conflict handling as configured (`ies_drop_conflicts=true`) for this test.

Expected impact:
- Should reveal how much degradation is due specifically to the custom pre-pruning step.

Pros:
- Fast sensitivity test.
- Retains standard in-run PEST++ conflict handling.

Cons:
- May retain more conflicted observations than the current two-stage process.
- Could change convergence behavior and phi trajectory.

## Option C: Explicitly test built-in PDC on vs off (policy decision)

Alternatives:
- Case C1: `ies_drop_conflicts=true` (built-in dropping on).
- Case C2: `ies_drop_conflicts=false` (built-in dropping off).
- Compare fit quality, bias, and stability on the same container and seed setup.

Expected impact:
- Establishes whether conflict dropping helps this problem or is net harmful.

Pros:
- Directly aligns with documented PEST++ options and avoids hidden behavior.

Cons:
- Requires at least two full calibration runs for clean comparison.

## Option D: Rebalance ETf weights across sites (structural improvement)

Alternatives:
- Normalize ETf weight totals by site.
- Use capped weights or robust loss on high-leverage dates.
- Blend magnitude and inverse-variance weighting with site scaling.

Expected impact:
- Reduces dominance of high-capture/high-magnitude sites.
- Improves fairness and stability across heterogeneous stations.

Pros:
- Addresses long-standing imbalance.

Cons:
- Changes objective definition and may shift published comparability.

## Option E: Document and lock precomputed-ensemble semantics

Actions:
- Keep `et_ensemble_mad` usage but document its meaning clearly (MAD-filtered ensemble ET estimate).
- Add a short runtime log message indicating which ensemble source is being used (`computed` vs `openet`) and which band is read.
- Add a quick assertion/diagnostic check of value ranges and non-NaN counts after ingestion.

Expected impact:
- Prevents future confusion and accidental workflow drift.

Pros:
- Low-risk hardening and better transparency.

Cons:
- Does not, by itself, improve current calibration behavior.

## Option F: Add guardrails and fail-fast diagnostics (recommended regardless)

Potential checks to enforce before calibration:
- Landsat NDVI exists and nonzero counts exceed threshold.
- ETf nonzero-weight ratio after PDC cannot fall below a configured floor.
- Emit and persist per-site weight-share and removed-fraction summaries.
- Abort run on known-invalid ensemble variable selection.

Expected impact:
- Prevents expensive runs with invalid or weak inputs.

## Recommended sequence

1. Option A first (restore Landsat NDVI, rerun) to remove known primary failure mode.
2. Option B immediate A/B test (custom pre-pass off vs current behavior) on same rebuilt container.
3. Option C explicit built-in PDC sensitivity (`ies_drop_conflicts` on/off) once Option B impact is known.
4. If needed, adopt Option D based on sensitivity outcomes.
5. Implement Option E and F as hardening.

## Suggested Validation Criteria for Next Run

- Pre-calibration diagnostics:
  - Landsat NDVI present for all expected sites.
  - ETf nonzero-weight count remains near full capture set before and after conflict handling.
  - No site loses an extreme fraction of ETf weight without explicit override.
- Post-calibration metrics:
  - SWIM recovery at previously affected sites (RIP760, BAR012, SLM001) relative to current degraded run.
  - Aggregate SWIM median R2 returns toward prior post-2016 baseline.
  - Ensemble performance remains stable, indicating fair comparison recovered.

## Bottom Line

The strongest evidence points to a pipeline/regression interaction, not a fundamental SWIM model collapse:

- Missing Landsat NDVI is a direct, known high-impact failure mode and is present in this run.
- PDC pre-pruning removed about half of ETf constraints before optimization.
- Existing weighting concentration then magnifies who still drives calibration.

Precomputed ensemble mode is not the cause here. The dominant actionable issues are NDVI data completeness and the current two-stage conflict dropping strategy.
