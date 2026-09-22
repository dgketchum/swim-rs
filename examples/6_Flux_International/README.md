# Example 6: International Flux Towers (paper Experiments E0 and E2)

Sixty-six cropland flux-tower sites on five continents, 2008–2025, forced with
ERA5-Land, HWSD v2.0 soils, and FROM-GLC10 land cover, with a two-member
Landsat ET-fraction target (ESPA SSEBop on the grass reference basis, PT-JPL).
Calibration runs on all 66 sites. The paper evaluates the 47 sites with
closure-corrected flux ET.

This directory produces:

| Paper artifact | Analysis | Output |
|---|---|---|
| Table 3, Fig. 2 (E0 vegetation formulation) | `pooled_arm_compare.py` on the 37 sites outside the E1 cohort | `results/e0_disjoint/*/disjoint37/` |
| Table 5, Table S8, §3.3 (E2 performance, transfer) | `closure_pool_summary.py` | `results/<run>/archive/6_evaluation/closure_pool/` |
| Fig. 5b (parameter transfer into E2) | `transfer_ex5_params.py` re-cut in the closure-pool summary | `closure_pool/transfer_refresh_summary.csv` |
| Table S2 transferred parameter sets | `transfer/build_ex5_irrigation_stratified_params.py` | `paper/data/final/e2_run22_transfer_vectors_by_irrigation.json` |
| Table S9 (E1 vs E2 product parity) | `product_parity/e1_e2_product_parity.py` | `--out-dir` |
| Supplement S9.1 (28-day month rule) | `awc_recal/monthly_28day_sensitivity.py` | `closure_pool/monthly_28day_sensitivity_*` |
| Frozen source of record for all of the above except S2/S9 | `promote_final.py` | `paper/data/final/e2_closure_pool/` + `MANIFEST.json` |

The paper's tables and figures read `paper/data/final/e2_closure_pool/`, not
the results root. `promote_final.py` copies the results-root outputs into that
package and writes its `MANIFEST.json` (sha256 and source of every file, run
SHAs, headline values); with no flags it checks the package against the
results root byte for byte.

**Status (2026-09-22).** The HWSD available-water-capacity units defect
(`notes/HANDOFF_HWSD_AWC_UNITS_RECAL.md`) invalidated every earlier E2 run.
The recalibration via `awc_recal/run_awc_recal_chain.sh` is complete and the
package is frozen from it; the superseded runs are under
`results/superseded_awc320_20260921/`.

## Layout

```
6_Flux_International/
├── 6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr.toml   canonical E2 configuration
├── ..._GrassBasis_POR_annual2yr_fao56_sig.toml                     E0 arm: unscaled sigmoid Kcb
├── ..._GrassBasis_POR_annual2yr_fao56.toml                         E0 arm: unscaled linear Kcb
├── 6_Flux_International_LSEnsemble_POR_annual2yr.toml              parent of the three above
├── 6_Flux_International_LSEnsemble_POR.toml                        container-build configuration
├── 6_Flux_International.toml                                       extraction configuration
├── container_build/            raw data → validated container (not needed if you have the container)
├── ex6_paths.py                every on-disk location, derived from the TOML `root`
├── objective_audit.py          independent reconstruction of the ETf weights (gate G9)
├── archive_prelaunch.py        RUN_POLICY Cats 1-3 capture and launch-time hash verify
├── archive_postcalibration.py  RUN_POLICY Cats 4-5: merged posterior, bounds, phi history
├── evaluation_summary.py       RUN_POLICY Cat 6: paired metrics, groups, reproduction checks
├── closure_pool_summary.py     47-site closure-corrected pool re-cut (Table 5, S8, §3.3)
├── promote_final.py            results root → paper/data/final/e2_closure_pool/ (check by default, --write)
├── evaluate.py                 daily, monthly, and ETf evaluation against flux ET
├── derived_metrics.py          shared: per-member benchmarks, uncalibrated baseline, decompositions
├── pooled_metrics.py           shared: concatenated-pool and √n-weighted metrics
├── transfer/                   freeze the E1 (Run 22) transfer vectors and the per-site class mapping
├── transfer_ex5_params.py      score the E1 vectors on the E2 cohort without recalibration
├── pooled_arm_compare.py       E0 two-arm pooled comparison
├── e0_disjoint/                E0 site lists and the arms runner
├── product_parity/             Table S9
├── awc_recal/                  the current recalibration chain and its gates
└── legacy/                     untracked; retired scripts
```

`derived_metrics.py` and `pooled_metrics.py` are imported by the scripts
above, not run on their own except where the chain calls them.

## Inputs

| Input | Location | Configured by |
|---|---|---|
| Calibrated container | `{root}/6_Flux_International/data/6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim` | `[paths] container` in the canonical TOML |
| Cohort shapefile (66 sites, flux source per site) | `{root}/6_Flux_International/data/gis/flux_crop_pub_66_150m.shp` | `[paths] fields_shapefile` |
| Flux truth (QAQC daily, closure-corrected `ET_corr`) | `{flux_dir}/{network}/{sid}_daily_data.csv` | `[validation] flux_dir` in the TOML |
| E1 posterior for transfer | `paper/data/final/e2_run22_transfer_vector*.json` | `--params`, `--params-by-site` |

`root` is set in the TOML (`root = "/data/ssd1/swim"`). Change that one line
to relocate every derived path.

## Workflow

`awc_recal/run_awc_recal_chain.sh` runs every step below unattended and is the
authoritative order. The commands here are the same ones, from the repository
root, with `EX6=examples/6_Flux_International`,
`CFG=$EX6/6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr.toml`,
`RUN=6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr`, and
`ARCHIVE={root}/6_Flux_International/results/$RUN/archive`.

**1. Validate the container** (mandatory before any calibration)

```bash
uv run python $EX6/container_build/e2_refooting/phase8_container_health.py --config $CFG --out-dir <qa-dir>
```

**2. Calibrate** (PEST++ IES, 200 realizations, 3 iterations, batches of 50;
always via `uv run`)

```bash
uv run python -m swimrs.calibrate.batch_runner --config $CFG --action calibrate-all --reals 200 --noptmax 3 --workers 20 --batch-size 50 --exclude-uncovered
```

The chain wraps this with the objective audit and the run-policy captures
(`objective_audit.py`, `archive_prelaunch.py`
before, `archive_postcalibration.py` after). The merged posterior lands
at `$ARCHIVE/4_pest_outputs/merged/merged_posterior.csv`.

**3. Evaluate**

```bash
uv run python $EX6/evaluate.py --config $CFG --par-csv $ARCHIVE/4_pest_outputs/merged/merged_posterior.csv
uv run python $EX6/evaluate.py --config $CFG --par-csv $ARCHIVE/4_pest_outputs/merged/merged_posterior.csv --monthly
uv run python $EX6/evaluate.py --config $CFG --par-csv $ARCHIVE/4_pest_outputs/merged/merged_posterior.csv --etf
uv run python $EX6/pooled_metrics.py --results-dir {root}/6_Flux_International/results/$RUN
uv run python $EX6/derived_metrics.py --config $CFG --uncalibrated --out {root}/6_Flux_International/results/$RUN/derived
```

**4. Transfer the E1 parameter sets** (no recalibration; vectors frozen upstream)

```bash
uv run python $EX6/transfer/build_e2_irrigation_mapping.py --container <container> --out-dir <results>/transfer_refresh
uv run python $EX6/transfer_ex5_params.py --config $CFG --params paper/data/final/e2_run22_transfer_vector.json --params-by-site <results>/transfer_refresh/e3_irrigation_stratified_param_mapping.json --container <container> --e2-results-dir <results> --out <transfer-out>
```

**5. Summaries** (Table 5, S8, Fig. 5b; read-only on the archive)

```bash
uv run python $EX6/evaluation_summary.py --config $CFG --run-name $RUN
uv run python $EX6/closure_pool_summary.py --config $CFG --run-name $RUN
uv run python $EX6/awc_recal/monthly_28day_sensitivity.py --run-name $RUN
```

**6. E0 formulation arms** (Table 3, Fig. 2)

```bash
bash $EX6/e0_disjoint/run_e0_arms.sh
```

Calibrates the two unscaled arms with the same method as step 2, evaluates
them, and runs `pooled_arm_compare.py` for each pair of arms on the 37-site
disjoint set and the 47-site pool.

**7. Product parity** (Table S9)

```bash
uv run python $EX6/product_parity/e1_e2_product_parity.py --out-dir <out>
```

**8. Promote and freeze the paper evidence**

```bash
uv run python $EX6/promote_final.py --config $CFG            # byte-for-byte check; --write replaces the package
```

Copies the step 4–6 outputs (`closure_pool/`, `transfer/`, `e0_disjoint/`) into
`paper/data/final/e2_closure_pool/` and regenerates its `MANIFEST.json`. The
manifest's prose fields (reason for the freeze, gate note, retired files) are
carried over from the existing manifest; edit them there when the reason for a
re-freeze changes.

## Rebuilding the container

Only needed without the shipped container. Everything is under
`container_build/` and runs in this order: `data_extract.py` (Earth Engine,
ERA5-Land), the ESPA SSEBop chain under `espa/`, the reference-ET sidecar and
grass-basis conversion under `e2_refooting/` (phases 1–7),
`shapefile.py`, `landcover_crop.py`, then

```bash
uv run python $EX6/container_build/container_prep_ls_ensemble_por.py --config $EX6/6_Flux_International_LSEnsemble_POR_annual2yr.toml --recompute-dynamics
```

followed by step 1 above. Earth Engine and ESPA steps need credentials and
quota and are not part of reproduction from the container.

## Tests

```bash
uv run pytest tests/unit -q -k "objective_audit or evaluation_summary or closure_pool or container_health or stratified_transfer or e2_irrigation_mapping or ex6_flux_source or e2_promote_final"
```

## Rules

- Irrigation status comes from the internal water-balance classifier only.
  Flux data never configures a model input.
- Transfer vectors are frozen from the E1 posterior before any E2 evaluation.
- Every run archives per `examples/RUN_POLICY.md` categories 1–6 before a
  number is reported.
