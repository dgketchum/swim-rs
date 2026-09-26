# Example 5: CONUS Flux Ensemble (paper Experiment E1)

Sixty cropland flux-tower sites in the conterminous United States, forced with
GridMET, calibrated to the per-capture mean of six unmasked OpenET v2.1 Landsat
ET-fraction members (SSEBop, PT-JPL, SIMS, geeSEBAL, eeMETRIC, DisALEXI) with
the intermodel standard deviation as the observation-weighting denominator.
Eight parameters per site, PEST++ IES, 200 realizations, three iterations.
Flux-tower ET is validation only: it never configures inputs, supplies a
target, or enters a transferred parameter set.

This directory produces:

| Paper artifact | Analysis | Output |
|---|---|---|
| Table 4, Tables S3 and S5, Fig. 3 (E1 daily and monthly benchmark vs OpenET) | `evaluate.py` | `paper/data/final/e1_openet_benchmark/{daily,monthly}/` |
| Table S4 (retrieval-date vs between-retrieval support) | `overpass_decomposition.py` on the daily evaluator bundle | `paper/data/final/e1_openet_benchmark/temporal/` |
| Table S6, Fig. 4d (spread vs fixed observation weighting) | `run_weighting_ablation.py`, rescored by `rebuild_e1_benchmark_evidence.py` | `paper/data/final/e2_weighting_ablation_*.csv` |
| Fig. 4a–c (member spread vs error; conditioned-ensemble spread) | `spread_error.py`, `conditioned_ensemble_uncertainty.py` | `paper/data/final/e2_spread_error_*.csv`; `results/run22/conditioned_ensemble_uncertainty/` |
| Table S7, Fig. 5a (within-E1 parameter transfer) | `within_e1_transfer.py`, promoted by `promote_final.py` | `paper/data/final/e2_irrigation_stratified_*.csv`, `e2_within_transfer_*.csv` |
| Fig. 5b, Table S2 (E1 vectors transferred into E2) | Example 6 `transfer/` reads the Run 22 posterior | `paper/data/final/e2_run22_transfer_vector*.json` |

**Status (2026-09-21).** The published calibration is Run 22 (recalibrated
2026-07-02). The benchmark package was frozen on 2026-09-01; its
`MANIFEST.json` pins the sha256 of every input and of the analysis code, and
`rebuild_e1_benchmark_evidence.py --verify` checks the rest of the `e2_*`
files against `e2_evidence_metadata.json`. The E0 vegetation-formulation
experiment (Table 3, Fig. 2) is run in Example 6 on the 37 sites outside this
cohort; the earlier 60-site arms are retired to the untracked `legacy/`.

Two identifiers predate the paper numbering and are kept because frozen
hashes and the figure builder key on them: `run22` is the archive tag of the
E1 calibration, and `e2_*` filenames and `results/within_e2_transfer*` output
directories are the legacy namespace for E1 evidence. Neither is a paper
experiment label; the mapping lives in
[`../VALIDATION_POLICY.md`](../VALIDATION_POLICY.md).

## Layout

```
5_Flux_Ensemble/
├── 5_Flux_Ensemble.toml                    canonical E1 configuration (`root` sets every path)
├── ex5_paths.py                            shared: run dir, container, posterior, log derived from the TOML
├── container_build/                        raw data → base container → run container (skip if you have the container)
│   ├── data_extract.py                     Earth Engine, GridMET, OpenET member and reference-ET extraction
│   ├── container_prep.py                   base container (meteorology, NDVI, properties, flux fields)
│   └── build_container.py                  run container: six-member target, corrected ETo/ETr, dynamics
├── calibrate.py                            PEST++ IES wrapper; archives the trajectory
├── archive_run.py                          run-policy archive for a finished calibration
├── evaluate.py                             daily and monthly benchmark vs flux ET and OpenET
├── overpass_decomposition.py               strict consumer of the daily evaluator bundle
├── spread_error.py                         member spread vs acquisition-date ETf error
├── conditioned_ensemble_uncertainty.py     retrieval spread vs conditioned-parameter spread
├── run_weighting_ablation.py               spread vs fixed-SD weighting (two recalibrations)
├── within_e1_transfer.py                   leave-region-out / leave-one-site-out transfer, pooled and irrigation-stratified
├── promote_final.py                        tracked producer for the hand-promoted e2_* supporting files
├── rebuild_e1_benchmark_evidence.py        rebuild or --verify the frozen e2_* evidence
├── data/                                   extracted inputs (gitignored except gis/, see Inputs)
├── legacy/                                 untracked; retired E0 arms and scripts
└── notes/                                  untracked; working notes
```

## Inputs

| Input | Location | Configured by |
|---|---|---|
| Base container | `{root}/5_Flux_Ensemble/data/5_Flux_Ensemble.swim` | `[paths] container` |
| Run 22 container | `{root}/5_Flux_Ensemble/data/5_Flux_Ensemble_run22.swim` | `container_build/build_container.py --run run22` |
| Run 22 posterior | `{root}/5_Flux_Ensemble/results/run22/5_Flux_Ensemble.3.par.csv` | `calibrate.py --results-tag run22` |
| Cohort shapefile (60 sites) | `data/gis/flux_fields.shp` (tracked) and `{root}/5_Flux_Ensemble/data/gis/` | `[paths] fields_shapefile` |
| Flux truth, Volk v2.1 closure-corrected daily ET | `{root}/5_Flux_Ensemble/data/daily_flux_files_2pt1/` | `[validation] flux_dir` |
| OpenET benchmark, 3 x 3 MAD-filtered v2.1 ensemble at the towers | `{root}/5_Flux_Ensemble/data/openet_flux_2pt1/{daily,monthly}_data/` | `OPENET_SOURCE_DIRNAME` in `evaluate.py` |
| Paired Volk delivery tables | `data/flux_2pt1/{daily,monthly}_2pt1_paired_data.csv` | sha256 pinned in the benchmark manifest |
| OpenET bias-corrected GridMET ETo and ETr | `data/openet_refet/openet_{eto,etr}.csv` | `container_build/build_container.py`; sha256 pinned |
| Six OpenET ETf member tables | `data/etf_v21_openet_eto/*_etf_no_mask.csv` | `container_build/data_extract.py --steps etf_v21` |

`root` is set in the TOML (`root = "/data/ssd1/swim"`). Change that one line to
relocate every derived path; the scripts resolve run directories, containers,
and the posterior through `ex5_paths.py`, so no path is hard-coded elsewhere.
Everything under `data/` except `gis/` is gitignored; the Volk flux and OpenET
deliveries are subject to their source data policy and must be supplied
separately. File presence is not a completeness test.

## Workflow

Commands run from the repository root with `EX5=examples/5_Flux_Ensemble`,
`CFG=$EX5/5_Flux_Ensemble.toml`, `DATA={root}/5_Flux_Ensemble/data`, and
`RUN22={root}/5_Flux_Ensemble/results/run22`. Steps 1 and 2 are expensive;
steps 3 onward reproduce the paper from the calibrated container and posterior.

**1. Build the run container** (only if you do not have
`5_Flux_Ensemble_run22.swim`; see Rebuilding the container below)

**2. Calibrate** (PEST++ IES; hours on a workstation)

```bash
uv run python $EX5/calibrate.py --config $CFG --container $DATA/5_Flux_Ensemble_run22.swim --results-tag run22 --keep-pestrun
uv run python $EX5/archive_run.py --results-tag run22 --container $DATA/5_Flux_Ensemble_run22.swim
```

`archive_run.py` writes the run-policy archive under `$RUN22/archive/`
(problem definition, parameter bounds, input health, posterior summary).

**3. Evaluate against flux ET and OpenET** (Table 4, S3, S5, Fig. 3)

```bash
uv run python $EX5/evaluate.py --config $CFG --par-csv $RUN22/5_Flux_Ensemble.3.par.csv --container $DATA/5_Flux_Ensemble_run22.swim --openet-source volk --output-dir <daily-dir> --bootstrap-reps 10000 --bootstrap-seed 42 --quiet-sites
uv run python $EX5/evaluate.py --config $CFG --par-csv $RUN22/5_Flux_Ensemble.3.par.csv --container $DATA/5_Flux_Ensemble_run22.swim --monthly --output-dir <monthly-dir> --bootstrap-reps 10000 --bootstrap-seed 42 --quiet-sites
```

Daily: capture-date OpenET ET is divided by same-day bias-corrected ETo,
reconstructed with the OpenET-core 32-day support, and multiplied back. Monthly
uses the independently extracted full-month product against flux ET gap-filled
and totaled by the Volk et al. (2024) rules (raw GridMET ETo × smoothed EToF fill,
>80% of days observed, ≤5 filled days); pooled rows take every paired site and
station-weighted rows take sites with ≥3 paired months. Each run writes
`evaluation_grouped_{daily,monthly}_metrics.csv` (pooled and √n-weighted
estimates with bootstrap intervals), `_contrasts.csv` (paired SWIM minus
OpenET), `_metadata.json` (inputs, hashes, record contract),
`evaluation_paired_daily_records.csv`, per-site `evaluation_metrics.csv`, and
the exclusion ledger.

**3b. Promote the monthly bundle** (re-freeze under RUN_POLICY)

```bash
uv run python $EX5/promote_e1_monthly.py --source-dir <monthly-dir>            # check only
uv run python $EX5/promote_e1_monthly.py --source-dir <monthly-dir> --write    # promote
```

The gate refuses a bundle whose sidecar git sha is not HEAD or whose analysis
files (`evaluate.py`, `benchmark.py`, `flux_utils.py`) have uncommitted changes,
whose sidecar hashes do not match, or whose 18 grouped estimates are not
replicated (tolerance 1e-6) from `archive/6_evaluation/site_daily_timeseries`
plus the Volk flux and OpenET monthly files. On `--write` it moves the prior
monthly package to `paper/data/final/superseded_e1_monthly_*/`, copies the new
bundle to `paper/data/final/e1_openet_benchmark/monthly/`, rewrites
`MANIFEST.json`, and refreshes `archive/6_evaluation/`.

**3c. Freeze the irrigation-class split** (Table S11)

```bash
uv run python $EX5/e1_class_split.py            # gates + report
uv run python $EX5/e1_class_split.py --write    # freeze under e1_openet_benchmark/class_split/
```

Partitions the frozen daily record, the Volk-protocol monthly record replicated
from the archive, and the common temporal cohort by the E2 fold class of each
site (`paper/data/final/e2_irrigation_stratified_fold_mad_domain.csv`) and
freezes pooled and √n-weighted KGE/RMSE/MBE per class with bootstrap intervals
on the SWIM minus OpenET contrast. The reunited classes must reproduce every
frozen grouped estimate before anything is written; `--write` refuses an
existing `class_split/` directory.

**4. Decompose by retrieval support** (Table S4)

```bash
uv run python $EX5/overpass_decomposition.py --evaluator-output-dir <daily-dir> --output-dir <temporal-dir> --bootstrap-reps 10000 --seed 42
```

Stale or hash-mismatched parent artifacts are hard errors.

**5. Spread–error and conditioned-ensemble uncertainty** (Fig. 4a–c)

```bash
uv run python $EX5/spread_error.py --config $CFG
uv run python $EX5/conditioned_ensemble_uncertainty.py
```

Outputs land in `$RUN22/spread_error/` and
`$RUN22/conditioned_ensemble_uncertainty/`.

**6. Weighting ablation** (Table S6, Fig. 4d; two further calibrations)

```bash
uv run python $EX5/container_build/build_container.py --run run22ablation --source $DATA/5_Flux_Ensemble.swim --mad
uv run python $EX5/run_weighting_ablation.py --tag run22 --container $DATA/5_Flux_Ensemble_run22ablation.swim
```

Arm outputs are `results/ablation_run22_e1_spread/`,
`results/ablation_run22_e2_fixed_sd/`, and `results/ablation_run22_summary/`
(`e1`/`e2` here are arm ids, not paper experiments). `--summary-only`
regenerates the summaries without recalibrating.

**7. Within-E1 parameter transfer** (Table S7, Fig. 5a)

```bash
uv run python $EX5/within_e1_transfer.py
```

The posterior and container default to Run 22 and the output to
`results/within_e2_transfer_irrigation_stratified/`, which holds both the
pooled (`loro`, `loso`) and the irrigation-stratified (`loro_strat`,
`loso_strat`) arms beside the local and default references. The stratified
arms carry two class vectors so a rainfed site never receives an irrigated
`mad`. The sibling `results/within_e2_transfer/` is an earlier run of the same
script before the stratified arms existed; its pooled columns equal the
stratified run's to floating-point round-off, and `promote_final.py` copies
the frozen `e2_within_transfer_*` files from it.

**8. Promote and freeze the paper evidence**

```bash
uv run python $EX5/promote_final.py            # byte-for-byte check of the promoted files; --write replaces them
uv run python $EX5/rebuild_e1_benchmark_evidence.py --output-dir <scratch-dir> --verify
```

`promote_final.py` produces the `e2_spread_error_*`, `e2_within_transfer_*`,
`e2_irrigation_stratified_transfer_summary.csv`, and
`e2_irrigation_stratified_fold_mad_domain.csv` files from steps 5 and 7.
`rebuild_e1_benchmark_evidence.py` rebuilds the `e2_primary_*`,
`e2_benchmark_*`, `e2_temporal_*` files, rescores the ablation onto the frozen
benchmark record, and writes `e2_evidence_metadata.json`; `--verify` compares
against the frozen hashes. The headline `e1_openet_benchmark/` package is
promoted from the step 3 and 4 output directories and described by its
`MANIFEST.json`, which names the similarly named legacy products that must not
be substituted for it.

## Rebuilding the container

Only needed without `5_Flux_Ensemble_run22.swim`. `data_extract.py` contacts
Earth Engine and downloads GridMET; run it with approval, not to inspect
existing evidence.

```bash
uv run python $EX5/container_build/data_extract.py                       # or --steps etf_v21,refet --sites US-Bi1,US-Ne1
uv run python $EX5/container_build/container_prep.py --overwrite --getinfo
uv run python $EX5/container_build/build_container.py --run run22 --source $DATA/5_Flux_Ensemble.swim --mad
```

After `container_prep.py`, every site must have seasonal NDVI and finite
meteorology (ETo, precipitation, radiation, tmax, tmin); ETf is deliberately
absent from the base container. After `build_container.py`, every site must
have at least one finite ETf capture per member. Report an incomplete site;
do not drop or fill it.

## Tests

```bash
uv run pytest tests/unit -v -k "e1_paired_record_temporal or e2_grouped_benchmark_metrics or overpass_decomposition or benchmark_regression or rebuild_verify or conditioned_ensemble or weighting_ablation or archive_eto_export or promote_final"
```

`test_e2_grouped_benchmark_metrics.py` keeps a legacy filename; it exercises
the current E1 evaluator.
