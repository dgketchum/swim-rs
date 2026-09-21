# Experiment E3 (Repo Example 6): Flux International

Publication-facing label: `E3`. Repo path remains
`examples/6_Flux_International`.

This is the international publication experiment set for SWIM-RS. The current
publication-track run is the long-period-of-record Landsat SSEBop + PT-JPL
ensemble calibration with the two-stage `annual_2yr` water-balance irrigation
classifier, on the **66-site cropland cohort** (Experiment A, Landsat ensemble).
Experiment B (ECOSTRESS) is forthcoming.

## Current Publication Track

- Canonical configuration:
  `6_Flux_International_LSEnsemble_POR_annual2yr.toml`
- Current canonical results note:
  `notes/E3_RESULTS.md` (run detail: `notes/MODEGATE_RESULTS.md`;
  irrigation design: `notes/IRRIGATION_CLASSIFICATION.md`)
- Shared validation policy:
  `examples/VALIDATION_POLICY.md`
- Forthcoming: Experiment B (ECOSTRESS). A prior 75-cohort combined
  Landsat+ECOSTRESS ablation (`6_Flux_International_TripleETf_POR.toml`) is
  documented in `notes/E3_RESULTS.md`.

## Canonical Workflow

**1. Build the publication shapefile** (66-site cohort)
```bash
uv run python /home/dgketchum/code/swim-rs/examples/6_Flux_International/container_build/shapefile.py --gis-dir /data/ssd1/swim/6_Flux_International/data/gis
```

**2. Write the curated cropland gate and build/recompute the POR container**
```bash
uv run python /home/dgketchum/code/swim-rs/examples/6_Flux_International/container_build/landcover_crop.py
uv run python /home/dgketchum/code/swim-rs/examples/6_Flux_International/container_build/container_prep_ls_ensemble_por.py --config /home/dgketchum/code/swim-rs/examples/6_Flux_International/6_Flux_International_LSEnsemble_POR_annual2yr.toml --recompute-dynamics
```

**3. Calibrate** (batch IES; always via `uv run`)
```bash
uv run python -m swimrs.calibrate.batch_runner --config /home/dgketchum/code/swim-rs/examples/6_Flux_International/6_Flux_International_LSEnsemble_POR_annual2yr.toml --action calibrate-all --reals 200 --noptmax 3 --workers 20 --batch-size 50 --exclude-uncovered
```

**4. Evaluate**
```bash
uv run python /home/dgketchum/code/swim-rs/examples/6_Flux_International/evaluate.py --config /home/dgketchum/code/swim-rs/examples/6_Flux_International/6_Flux_International_LSEnsemble_POR_annual2yr.toml
```

## Notes

- `legacy/` holds older non-POR experiments, SSEBop-only workflows, merged
  ETf diagnostics, and historical comparison scripts.
- `notes/CURRENT_FINDINGS.md` is historical diagnostic context, not the
  canonical publication note.
