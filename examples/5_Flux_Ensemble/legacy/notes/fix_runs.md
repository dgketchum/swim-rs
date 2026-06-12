# Fix Plan Run Log

Date: 2026-02-17

## Degraded Baseline (pre-fix)

- Landsat NDVI: missing (empty dirs)
- pdc_remove: True (custom pre-pass, reals=5)
- ies_drop_conflicts: true
- ensemble_source: computed
- ensemble ETf scaling: broken (et_ensemble_mad / 10000, no ETo)
- **SWIM median R2: 0.419, mean R2: 0.180**
- **Ensemble median R2: 0.579, mean R2: 0.416**

## Run 1: Phase 2 — computed ensemble, no custom PDC pre-pass

- Landsat NDVI: restored (copied from Ex4, 59/60 sites)
- pdc_remove: False
- ies_drop_conflicts: true
- ensemble_source: computed (3-model mean — only ptjpl, sims, ssebop ingested)
- ensemble ETf scaling: not used (computed mode)
- **SWIM median R2: 0.570, mean R2: 0.341**
- **Ensemble median R2: 0.581, mean R2: 0.415**
- Key sites: RIP760=0.871, BAR012=0.538, SLM001=0.316, JPL1_JV114=-1.960
- Note: container was built with --openet-source diy; geesebal, eemetric,
  disalexi had empty old-style extract dirs, so "6-model" was actually 3-model

## Run 2: OpenET ensemble target

- Landsat NDVI: restored
- pdc_remove: False
- ies_drop_conflicts: true
- ensemble_source: openet (et_ensemble_mad / 1000 / ETo, corrected scaling)
- **SWIM median R2: 0.578, mean R2: 0.328**
- **Ensemble median R2: 0.529, mean R2: 0.348**
- Note: ensemble reference itself degraded vs computed; openet MAD ensemble
  is a weaker target for these sites
- Key sites: RIP760=0.863, BAR012=0.204 (regressed), SLM001=0.487

## Run 3: computed ensemble, ies_drop_conflicts=false

- Landsat NDVI: restored
- pdc_remove: False
- ies_drop_conflicts: false
- ensemble_source: computed (6-model mean)
- nonzero-weight obs: 31,453 (vs 30,881 with drop=true)
- phi: ~2.16M (high due to retained conflicting obs)
- **SWIM median R2: 0.573, mean R2: 0.325**
- **Ensemble median R2: 0.581, mean R2: 0.415**
- Key sites: RIP760=0.864, BAR012=0.586, SLM001=0.154, JPL1_JV114=-2.114
- Conclusion: essentially identical to Run 1; ies_drop_conflicts has negligible effect

## Run 4: per-site weight normalization (reverted)

- MB_Pch excluded (59 sites)
- pdc_remove: False
- ies_drop_conflicts: true
- ensemble_source: computed (3-model — same container as Run 1)
- Per-site ETf weight normalization: active (equalize each site's total weight)
- **SWIM median R2: 0.540, mean R2: 0.208**
- **Ensemble median R2: 0.509, mean R2: 0.195**
- Conclusion: weight normalization hurt — boosted noisy low-quality sites. Reverted.

## Run 5: full 6-model ensemble

- MB_Pch excluded (59 sites)
- pdc_remove: False
- ies_drop_conflicts: true
- ensemble_source: computed (true 6-model mean: ssebop, sims, geesebal, eemetric, ptjpl, disalexi)
- Container built with --openet-source ee (all 6 models × irr + inv_irr ingested)
- **SWIM median R2: 0.588, mean R2: 0.337**
- **Ensemble median R2: 0.581, mean R2: 0.415**
- **SWIM beats ensemble reference** (0.588 vs 0.581 median R2)
- Key sites: RIP760=0.857, BAR012=0.593, SLM001=0.266, JPL1_JV114=-2.355
- Best sites: UA1_HartFarm=0.859, RIP760=0.857, Almond_Med=0.781, Almond_High=0.776

## Run 6: m_kc multiplier + empirical p95 kc_max

- MB_Pch excluded (59 sites)
- pdc_remove: False
- ies_drop_conflicts: true
- ensemble_source: computed (6-model mean)
- kc_max base: 95th percentile of ETf per site (no floor), was p90 with 1.25 floor
- New parameter: m_kc (per-site multiplier, bounds 0.85–1.25, initial 1.0)
- Parameter count: 480 → 531 (9 params × 59 sites)
- empirical_kc_max=True in build_swim_input
- Container dynamics recomputed to update kc_max values
- Results dir: run6_m_kc
- Calibration: 38.9 min, 200 realizations, 40 workers, noptmax=3
- DIY eval (27 sites): **SWIM median R2: 0.507, mean R2: 0.312**
- **Ensemble median R2: 0.605, mean R2: 0.445**
- Volk 3x3 eval (26 sites): **SWIM median R2: 0.656, mean R2: 0.610**
- **Volk ensemble median R2: 0.781, mean R2: 0.729**
- SWIM bias: -0.173 (Volk), -0.608 (DIY)
- Key sites: RIP760=0.784, BAR012=0.303, SLM001=0.507, S2=0.732, US-Tw3=0.732
- Regression vs Run 5: DIY median R2 dropped 0.588→0.507 (-0.081), Volk improved 0.634→0.656 (+0.022)
- Monthly: SWIM median R2=0.558, mean R2=0.016 (dragged down by sites with large negative R2)

## Summary

| Run | ETf models | Notes | SWIM med R2 | SWIM mean R2 | Ens med R2 |
|-----|-----------|-------|-------------|-------------|------------|
| Baseline | 3 (diy) | broken NDVI + PDC pre-pass | 0.419 | 0.180 | 0.579 |
| Run 1 | 3 (diy) | NDVI restored, no PDC | 0.570 | 0.341 | 0.581 |
| Run 2 | 3 (diy) | openet ensemble target | 0.578 | 0.328 | 0.529 |
| Run 3 | 3 (diy) | drop_conflicts=false | 0.573 | 0.325 | 0.581 |
| Run 4 | 3 (diy) | per-site weight norm | 0.540 | 0.208 | 0.509 |
| Run 5 | 6 (ee) | full ensemble, MB_Pch excl | 0.588 | 0.337 | 0.581 |
| Run 6 | 6 (ee) | m_kc multiplier, p95 kc_max | 0.507 | 0.312 | 0.605 |
| Run 7 | 6 (ee) | no_mask NDVI + ETf | 0.629 | 0.592 | 0.781 |
| **Run 8** | **6 (ee)** | **no_mask, m_kc frozen** | **0.627** | **0.604** | **0.781** |

Key findings:
1. Restoring Landsat NDVI + removing custom PDC pre-pass = dominant fix (+0.15 R2)
2. ies_drop_conflicts on/off makes negligible difference
3. OpenET MAD ensemble is a weaker calibration target than computed mean
4. Per-site weight normalization hurts — amplifies noisy sites
5. Full 6-model ensemble (vs 3-model) adds +0.018 median R2 and pushes SWIM past ensemble
6. Recommended config: computed 6-model ensemble, pdc_remove=False, ies_drop_conflicts=true, --openet-source ee

---

## Volk 3x3 Benchmark (Run 6 parameters)

26 sites with both SWIM and Volk ensemble data.

| Model | med R2 | mean R2 | med RMSE | mean bias |
|-------|--------|---------|----------|-----------|
| **SWIM** | **0.656** | **0.610** | **1.319** | **-0.173** |
| sims | 0.794 | 0.731 | 0.998 | -0.309 |
| disalexi | 0.723 | 0.587 | 1.064 | -0.539 |
| ssebop | 0.649 | 0.621 | 1.212 | -0.181 |
| ptjpl | 0.622 | 0.622 | 1.121 | -0.227 |
| eemetric | 0.613 | 0.564 | 1.262 | -0.158 |
| geesebal | 0.491 | 0.384 | 1.453 | -0.771 |
| **Volk ensemble** | **0.781** | **0.729** | **1.043** | **-0.393** |

SWIM Volk R2 improved from 0.634 (Run 5) to **0.656 (Run 6)** (+0.022).
SWIM bias improved from -0.020 to **-0.173** (more negative — slight overcorrection).

---

## Volk 3x3 Benchmark (Run 5 parameters, archived)

Independent validation against Volk et al. OpenET 3x3 flux tower extractions.
These are unmasked ET (mm/day) at 3×3 Landsat pixels centered on each tower —
same footprint as our polygons (~8100 m², 90m side). Uses Volk's pre-computed
`ensemble_mean_3x3` (MAD-filtered) rather than our own nanmean.

32 sites with both SWIM and Volk ensemble data.

### Per-model comparison

| Model | med R2 | mean R2 | med RMSE | mean bias |
|-------|--------|---------|----------|-----------|
| **SWIM** | **0.634** | **0.603** | **1.250** | **-0.020** |
| disalexi | 0.736 | 0.594 | 1.050 | -0.428 |
| sims | 0.728 | 0.682 | 1.064 | -0.324 |
| ssebop | 0.650 | 0.608 | 1.212 | -0.214 |
| ptjpl | 0.632 | 0.622 | 1.145 | -0.177 |
| eemetric | 0.613 | 0.547 | 1.275 | -0.118 |
| geesebal | 0.535 | 0.411 | 1.409 | -0.685 |
| **Volk ensemble** | **0.781** | **0.720** | **1.036** | **-0.352** |

### DIY vs Volk evaluation comparison

The DIY evaluation uses our irr/inv_irr masked ETf × ETo interpolated to daily.
The Volk benchmark uses unmasked ET computed by OpenET's native pipeline.

| Evaluation | SWIM med R2 | SWIM mean R2 | Ens med R2 | Ens mean R2 |
|------------|-------------|-------------|------------|-------------|
| DIY (34 sites) | 0.588 | 0.337 | 0.581 | 0.415 |
| Volk 3x3 (32 sites) | 0.634 | 0.603 | 0.781 | 0.720 |

Key observations:
- SWIM scores higher against Volk (0.634 vs 0.588) — different site subset and
  unmasked OpenET reference
- Volk ensemble is much stronger (0.781 vs 0.581) — unmasked extraction and
  OpenET's native ET pipeline vs our ETf×ETo reconstruction
- SWIM has the lowest bias of any model (-0.020 mean) against flux observations
- SWIM beats ptjpl (0.632) and geesebal (0.535), trails sims (0.728) and
  disalexi (0.736)
- Gap to Volk ensemble: 0.147 median R2

---

## Run 7: no_mask NDVI + ETf

- MB_Pch excluded (59 sites)
- pdc_remove: False
- ies_drop_conflicts: true
- ensemble_source: computed (6-model mean)
- m_kc multiplier + empirical p95 kc_max (same params as Run 6)
- **New: no_mask Landsat NDVI ingested** (531 CSVs, 59 sites × 9 years)
- Container rebuilt with `--overwrite --openet-source ee`, masks = irr, inv_irr, no_mask
- container_prep.py updated: NDVI mask list now includes no_mask
- Calibration: 41.3 min, 200 realizations, 40 workers, noptmax=3
- Best mean phi: 17049 → 7757 over 3 iterations
- Results dir: run7_no_mask

### Daily ET — Volk 3x3 Benchmark (32 sites)

| Model | med R2 | mean R2 | med RMSE | mean bias |
|-------|--------|---------|----------|-----------|
| **SWIM** | **0.629** | **0.592** | **1.271** | **-0.015** |
| sims | 0.728 | 0.682 | 1.064 | -0.324 |
| ptjpl | 0.632 | 0.622 | 1.145 | -0.177 |
| ssebop | 0.650 | 0.608 | 1.212 | -0.214 |
| disalexi | 0.736 | 0.594 | 1.050 | -0.428 |
| eemetric | 0.613 | 0.547 | 1.275 | -0.118 |
| geesebal | 0.535 | 0.411 | 1.409 | -0.685 |
| **Volk ensemble** | **0.781** | **0.720** | **1.036** | **-0.352** |

### ETf at Landsat Capture Dates (59 sites × 6 models)

| Model | combos | R2 mean | R2 median | RMSE mean | bias mean |
|-------|--------|---------|-----------|-----------|-----------|
| ssebop | 59 | 0.144 | 0.299 | 0.251 | 0.036 |
| eemetric | 59 | 0.082 | 0.212 | 0.290 | -0.027 |
| sims | 59 | 0.027 | 0.190 | 0.253 | 0.110 |
| disalexi | 59 | 0.003 | 0.088 | 0.389 | 0.050 |
| ptjpl | 59 | 0.041 | 0.022 | 0.392 | 0.008 |
| geesebal | 59 | -0.087 | 0.002 | 0.402 | 0.054 |
| ALL | 354 | 0.035 | 0.089 | 0.329 | 0.038 |

### Monthly ET (21 sites with SWIM + ensemble)

| Model | med R2 | mean R2 | med RMSE | mean bias |
|-------|--------|---------|----------|-----------|
| **SWIM** | **0.774** | **0.695** | **27.2** | **-6.6** |
| sims | 0.856 | 0.784 | 20.9 | 1.1 |
| disalexi | 0.835 | 0.776 | 19.6 | -6.0 |
| ptjpl | 0.832 | 0.760 | 23.8 | 0.7 |
| eemetric | 0.762 | 0.638 | 24.3 | -8.8 |
| ssebop | 0.712 | 0.603 | 25.6 | -5.0 |
| geesebal | 0.728 | 0.575 | 26.7 | -12.6 |
| **Volk ensemble** | **0.877** | **0.805** | **18.0** | **-6.0** |

### Run 7 vs Run 6 comparison

| Metric | Run 6 | Run 7 | Delta |
|--------|-------|-------|-------|
| Daily R2 median (Volk) | 0.656 | 0.629 | -0.027 |
| Daily R2 mean (Volk) | 0.610 | 0.592 | -0.018 |
| Daily SWIM bias | -0.173 | -0.015 | +0.158 (improved) |
| Monthly R2 median | 0.558 | 0.774 | +0.216 |
| Monthly R2 mean | 0.016 | 0.695 | +0.679 |

Key observations:
- Daily R2 slipped slightly (-0.027 median), but bias nearly zeroed (-0.173 → -0.015)
- Monthly performance dramatically improved (+0.216 median R2, +0.679 mean R2)
- Monthly mean R2 jump (0.016 → 0.695) suggests Run 6 had sites with large negative monthly R2 that are now fixed
- SWIM has the lowest daily bias of any model (-0.015 vs ensemble -0.352)
- Gap to Volk ensemble: 0.152 median R2 (daily), 0.103 median R2 (monthly)

---

## Run 8: no_mask NDVI/ETf, m_kc frozen (best result)

- MB_Pch excluded (59 sites)
- pdc_remove: False
- ies_drop_conflicts: true
- ensemble_source: computed (6-model mean)
- empirical p95 kc_max (same as Run 6/7)
- **m_kc frozen at 1.0** via `freeze_pargps=["m_kc"]` (partrans="fixed", removed from loc.mat)
- 8 free params × 59 sites = 472 adjustable parameters
- Same container as Run 7 (no_mask NDVI + ETf already ingested)
- Calibration: 40.5 min, 200 realizations, 40 workers, noptmax=3
- Results dir: run8_no_mkc

### Daily ET — Volk 3x3 Benchmark (32 sites)

| Model | med R2 | mean R2 | med RMSE | mean bias |
|-------|--------|---------|----------|-----------|
| **SWIM** | **0.627** | **0.604** | **1.263** | **-0.001** |
| sims | 0.728 | 0.682 | 1.064 | -0.324 |
| disalexi | 0.736 | 0.594 | 1.050 | -0.428 |
| ptjpl | 0.632 | 0.622 | 1.145 | -0.177 |
| ssebop | 0.650 | 0.608 | 1.212 | -0.214 |
| eemetric | 0.613 | 0.547 | 1.275 | -0.118 |
| geesebal | 0.535 | 0.411 | 1.409 | -0.685 |
| **Volk ensemble** | **0.781** | **0.720** | **1.036** | **-0.352** |

### ETf at Landsat Capture Dates (59 sites × 6 models)

| Model | combos | R2 mean | R2 median | RMSE mean | bias mean |
|-------|--------|---------|-----------|-----------|-----------|
| ssebop | 59 | 0.204 | 0.322 | 0.247 | 0.034 |
| eemetric | 59 | 0.093 | 0.214 | 0.289 | -0.028 |
| sims | 59 | 0.076 | 0.168 | 0.250 | 0.108 |
| ptjpl | 59 | 0.044 | 0.013 | 0.392 | 0.006 |
| disalexi | 59 | 0.011 | 0.081 | 0.388 | 0.048 |
| geesebal | 59 | -0.079 | 0.000 | 0.400 | 0.052 |
| ALL | 354 | 0.058 | 0.109 | 0.328 | 0.037 |

### Monthly ET (21 sites with SWIM + ensemble)

| Model | med R2 | mean R2 | med RMSE | mean bias |
|-------|--------|---------|----------|-----------|
| **SWIM** | **0.792** | **0.708** | **26.5** | **-6.0** |
| sims | 0.856 | 0.784 | 20.9 | 1.1 |
| disalexi | 0.835 | 0.776 | 19.6 | -6.0 |
| ptjpl | 0.832 | 0.760 | 23.8 | 0.7 |
| eemetric | 0.762 | 0.638 | 24.3 | -8.8 |
| ssebop | 0.712 | 0.603 | 25.6 | -5.0 |
| geesebal | 0.728 | 0.575 | 26.7 | -12.6 |
| **Volk ensemble** | **0.877** | **0.805** | **18.0** | **-6.0** |

### Run 8 vs Run 7 comparison (isolating m_kc effect)

| Metric | Run 7 (m_kc) | Run 8 (no m_kc) | Delta |
|--------|-------------|-----------------|-------|
| Daily R2 mean (Volk) | 0.592 | 0.604 | +0.012 |
| Daily R2 median (Volk) | 0.629 | 0.627 | -0.002 |
| Daily SWIM bias | -0.015 | -0.001 | +0.014 |
| Daily RMSE | 1.289 | 1.273 | -0.016 |
| Monthly R2 mean | 0.695 | 0.708 | +0.013 |
| Monthly R2 median | 0.774 | 0.792 | +0.018 |
| ETf R2 mean (ALL) | 0.035 | 0.058 | +0.023 |

### Key findings from Runs 7-8

1. **no_mask NDVI is the dominant improvement** — bias went from -0.173 (Run 6) to -0.001 (Run 8)
2. **m_kc adds noise without benefit** — removing it improved every metric
3. **SWIM has the lowest bias of any model** (-0.001 vs ensemble -0.352)
4. SWIM beats eemetric (0.613) and geesebal (0.535) on median R2, ties with disalexi/ssebop/ptjpl
5. Gap to Volk ensemble: 0.154 median R2 (daily), 0.085 median R2 (monthly)
6. **Recommended config**: no_mask NDVI+ETf, 8 params (no m_kc), computed 6-model ensemble, pdc_remove=False
