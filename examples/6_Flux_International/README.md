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
| Table 5, Table S9, §3.3 (E2 performance, transfer) | `closure_pool_summary.py` | `results/<run>/archive/6_evaluation/closure_pool/` |
| Fig. 5b (parameter transfer into E2) | `transfer_ex5_params.py` re-cut in the closure-pool summary | `closure_pool/transfer_refresh_summary.csv` |
| Table S2 transferred parameter sets | `transfer/build_ex5_irrigation_stratified_params.py` | `paper/data/final/e2_run23_transfer_vectors_by_irrigation.json` |
| Table S10 (E1 vs E2 product parity) | `product_parity/e1_e2_product_parity.py` | `--out-dir` |
| Supplement S9.1 (28-day month rule) | `awc_recal/monthly_28day_sensitivity.py` | `closure_pool/monthly_28day_sensitivity_*` |
| Frozen source of record for all of the above except S2/S10 | `promote_final.py` | `paper/data/final/e2_closure_pool/` + `MANIFEST.json` |

The paper's tables and figures read `paper/data/final/e2_closure_pool/`, not
the results root. `promote_final.py` copies the results-root outputs into that
package and writes its `MANIFEST.json` (sha256 and source of every file, run
SHAs, headline values); with no flags it checks the package against the
results root byte for byte.

**Status (2026-09-22).** The HWSD available-water-capacity units defect
(mm/m stored as m/m) invalidated every earlier E2 run.
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
├── closure_pool_summary.py     47-site closure-corrected pool re-cut (Table 5, S9, §3.3)
├── promote_final.py            results root → paper/data/final/e2_closure_pool/ (check by default, --write)
├── evaluate.py                 daily, monthly, and ETf evaluation against flux ET
├── derived_metrics.py          shared: per-member benchmarks, uncalibrated baseline, decompositions
├── pooled_metrics.py           shared: concatenated-pool and √n-weighted metrics
├── transfer/                   freeze the E1 (Run 23) transfer vectors and the per-site class mapping
├── transfer_ex5_params.py      score the E1 vectors on the E2 cohort without recalibration
├── pooled_arm_compare.py       E0 two-arm pooled comparison
├── e0_disjoint/                E0 site lists and the arms runner
├── product_parity/             Table S10
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
| Flux truth (QAQC daily, closure-corrected `ET_corr`) | `{root}/6_Flux_International/data/flux_qaqc/{network}/{sid}_daily_data.csv` (the 66 cohort files ship in `swim_data_e2.tar.gz`; on the authoring machine a symlink to the flux-data-qaqc archive) | `[validation] flux_dir = "{data}/flux_qaqc"` |
| E1 posterior for transfer | `paper/data/final/e2_run23_transfer_vector*.json` | `--params`, `--params-by-site` |

`{root}` is the TOML `root` (`/data/ssd1/swim` on the authoring machine);
set `SWIM_ROOT` to relocate every derived path without editing the TOML.

## Workflow

Commands run from the repository root with

```bash
export SWIM_ROOT=/path/to/swim                 # the data root (the TOML root, /data/ssd1/swim, on the authoring machine)
EX6=examples/6_Flux_International
CFG=$EX6/6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr.toml
RUN=6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr
RESULTS=$SWIM_ROOT/6_Flux_International/results
ARCHIVE=$RESULTS/$RUN/archive
```

**With the data packages** (`../README.md`, "Reproducing the paper"): skip
steps 1 and 2 and the calibrations inside step 6. Run step 3, step 4 into a
scratch `<transfer-out>`, step 5, the three `pooled_arm_compare.py` pairs of
step 6 on the shipped arm posteriors, step 7, and step 8, whose default check
mode reports `match` or `match~` for every file and an unchanged
`MANIFEST.json` (build stamps, machine paths, and the hashes of the inputs
that steps 3 and 5 rewrote are not compared). Steps 3 and 5 write into the
shipped run directory under `$RESULTS/$RUN/`; keep a copy of the package if
you want the shipped files back.

**1. Validate the container** (mandatory before any calibration)

```bash
uv run python $EX6/container_build/e2_refooting/phase8_container_health.py --config $CFG --out-dir <qa-dir>
```

**2. Calibrate** (PEST++ IES, 200 realizations, 3 iterations, batches of 50;
always via `uv run`; `awc_recal/run_awc_recal_chain.sh` runs steps 1–5 and 8
unattended and is the authoritative order)

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
uv run python $EX6/pooled_metrics.py --results-dir $RESULTS/$RUN
uv run python $EX6/derived_metrics.py --config $CFG --uncalibrated --out $RESULTS/$RUN/derived
```

Two uncalibrated-baseline files exist. Table 5 and Table S9 quote
`closure_pool/uncalibrated_baseline_summary.csv`, scored by
`transfer_ex5_params.py` on the retrieval-gated paired days
(`rs_gated_paired_days`, the same mask as the calibrated rows); the
`derived_metrics.py --uncalibrated` output
(`uncalibrated_baseline_summary_flux_swim_mask.csv`, every day with finite
flux and SWIM ET) is a diagnostic and is not quoted.

**4. Transfer the E1 parameter sets** (no recalibration; vectors frozen upstream)

```bash
CONTAINER=$SWIM_ROOT/6_Flux_International/data/6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim
uv run python $EX6/transfer/build_e2_irrigation_mapping.py --container $CONTAINER --out-dir $RESULTS/$RUN/transfer_refresh --allow-unexpected
uv run python $EX6/transfer_ex5_params.py --config $CFG --params paper/data/final/e2_run23_transfer_vector.json --params-by-site $RESULTS/$RUN/transfer_refresh/e3_irrigation_stratified_param_mapping.json --container $CONTAINER --e2-results-dir $RESULTS/$RUN --out <transfer-out>
```

`--allow-unexpected` reproduces the shipped mapping (its metadata records the
flag); `--e2-results-dir` is the run directory that step 3 filled, not the
results root.

**5. Summaries** (Table 5, S9, Fig. 5b; read-only on the archive)

```bash
uv run python $EX6/evaluation_summary.py --config $CFG --run-name $RUN --skip-baseline-forward
uv run python $EX6/closure_pool_summary.py --config $CFG --run-name $RUN
uv run python $EX6/awc_recal/monthly_28day_sensitivity.py --run-name $RUN
```

`evaluation_summary.py` also compares the run against the superseded
classifier-recal baseline (its evaluator outputs ship in the e2 package;
`--skip-baseline-forward` skips the baseline's own forward run, which needs
its container and PEST++ archive, neither of which ships, and is not part of
the frozen outputs). When
the baseline is absent the comparison is skipped and its two summary files are
not written.

**6. E0 formulation arms** (Table 3, Fig. 2)

```bash
bash $EX6/e0_disjoint/run_e0_arms.sh
```

Calibrates the two unscaled arms (`${RUN}_fao56`, `${RUN}_fao56_sig`) with
the same method as step 2, evaluates them, and runs `pooled_arm_compare.py`
for each pair of arms on the 37-site disjoint set and the 47-site pool. The
comparison alone, on the shipped arm posteriors, is the last block of the
script: the six invocations below, which are what the data packages need.

```bash
A=$ARCHIVE/4_pest_outputs/merged/merged_posterior.csv; SIG=$RESULTS/${RUN}_fao56_sig/archive/4_pest_outputs/merged/merged_posterior.csv; LIN=$RESULTS/${RUN}_fao56/archive/4_pest_outputs/merged/merged_posterior.csv
for SET in disjoint37:disjoint_sites.txt pool47:closure_pool_sites.txt; do TAG=${SET%%:*}; SITES=$EX6/e0_disjoint/${SET##*:}; uv run python $EX6/pooled_arm_compare.py --a-name grassbasis --a-config $CFG --a-par $A --b-name fao56_sig --b-config $EX6/${RUN}_fao56_sig.toml --b-par $SIG --sites-file $SITES --out-dir $RESULTS/e0_disjoint/grassbasis_vs_fao56_sig/$TAG; uv run python $EX6/pooled_arm_compare.py --a-name grassbasis --a-config $CFG --a-par $A --b-name fao56 --b-config $EX6/${RUN}_fao56.toml --b-par $LIN --sites-file $SITES --out-dir $RESULTS/e0_disjoint/grassbasis_vs_fao56/$TAG; uv run python $EX6/pooled_arm_compare.py --a-name fao56_sig --a-config $EX6/${RUN}_fao56_sig.toml --a-par $SIG --b-name fao56 --b-config $EX6/${RUN}_fao56.toml --b-par $LIN --sites-file $SITES --out-dir $RESULTS/e0_disjoint/fao56_sig_vs_fao56/$TAG; done
```

The six outputs (three pairs × `disjoint37`, `pool47`) land under
`$RESULTS/e0_disjoint/` and are checked by step 8. Each invocation prints
`GATE PASS` when arm A wins the pooled metrics and `GATE FAIL`, with exit
status 1, when it does not: the two `grassbasis` pairs pass and the
`fao56_sig` vs `fao56` pair fails, because the sigmoid arm scores below the
linear one (Table 3). The exit status is the comparison's verdict, not an
error.

**7. Product parity** (Table S10)

```bash
uv run python $EX6/product_parity/e1_e2_product_parity.py --out-dir <out>
```

**8. Promote and freeze the paper evidence**

```bash
uv run python $EX6/promote_final.py --config $CFG            # check; --write replaces the package
```

The check reports `match` (byte-equal) or `match~` (equal within
floating-point round-off; a rebuild on other hardware differs in the last
digit of a few pooled statistics) for every file, `differs` otherwise.

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
