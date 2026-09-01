# Example 5: CONUS Flux Ensemble

Example 5 supports two paper experiments on the same 60-site CONUS cropland
cohort:

- **E0** compares three vegetation formulations.
- **E1** calibrates SWIM-RS to a six-member OpenET ETf ensemble and evaluates
  daily ET reconstruction, observation weighting, uncertainty, and parameter
  transfer.

Flux-tower ET is validation-only. It never configures model inputs, supplies a
calibration target, or contributes to transferred parameter sets.

## Start Here

The primary E1 configuration is `5_Flux_Ensemble.toml`. The checked-in paths
assume the project workspace is
`/data/ssd1/swim/5_Flux_Ensemble`; use explicit config, container, parameter,
and output paths for any result intended for review or publication.

Install the project environment from the repository root:

```bash
uv sync --all-extras
```

The canonical workflow is:

1. Extract or obtain the required inputs.
2. Build and validate a base container.
3. Build an E1 run container with the six-member target and corrected
   reference ET.
4. Calibrate with PEST++ IES.
5. Evaluate against flux ET and the separately extracted OpenET benchmark.
6. Derive the retrieval-date decomposition from the evaluator-owned paired
   records.

Steps 1 and 4 contact external services or launch expensive computation. They
should not be run merely to inspect or reproduce existing evidence.

## Naming and Provenance

The scientific labels are E0 and E1. Three older identifiers remain only to
preserve archived paths and hashes:

| Identifier | Meaning |
|---|---|
| `run22` | Internal archive tag for the primary E1 calibration |
| `e2_*` | Legacy filename namespace in the frozen E1 evidence package |
| `within_e2_transfer.py` | Legacy script filename for the within-E1 transfer analysis |

Do not use those identifiers as paper experiment labels. The current
experiment mapping and comparison rules are defined in
[`../VALIDATION_POLICY.md`](../VALIDATION_POLICY.md).

## Scientific Configuration

SWIM-RS is calibrated at 60 cropland flux sites against the simple per-capture
mean of six unmasked OpenET v2.1 Landsat ETf members: SSEBop, PT-JPL, SIMS,
geeSEBAL, eeMETRIC, and DisALEXI. ET-denominated members are divided by OpenET
bias-corrected GridMET grass-reference ET (ETo) before they enter the target.
The intermodel sample standard deviation supplies the observation-weighting
denominator. Eight parameters are estimated per site with 200 realizations and
three PEST++ IES iterations.

The E1 benchmark uses the separately extracted, 3 x 3,
MAD-filtered OpenET v2.1 ensemble supplied with the Volk flux comparison. For
daily evaluation, capture-date ET is divided by same-day OpenET
bias-corrected ETo, ETf is reconstructed with the OpenET-core 32-day temporal
support behavior, and daily ET is recovered with the same ETo. The monthly
benchmark is an independently extracted full-month product; it is not a sum
of reconstructed daily values.

Headline SWIM-OpenET output uses exact common support and reports:

- pooled KGE, RMSE, MBE, Pearson r, Pearson r-squared, and zero-intercept
  slope;
- square-root-record-length-weighted site KGE, RMSE, and MBE; and
- whole-site bootstrap intervals and paired SWIM-minus-OpenET contrasts.

Per-site tables remain required diagnostics. Median site effects are emitted
only with `--site-effect-summary` and are not the default cohort headline.

## Data Boundary

Large, restricted, or generated inputs are intentionally not committed.

| Location | Contents | Versioned |
|---|---|---|
| `examples/data/` | Shared flux-footprint geometry and station metadata used to generate the 60-site shapefile | Yes |
| `examples/5_Flux_Ensemble/data/etf_v21_openet_eto/` | Six locally extracted OpenET ETf member tables and extraction summaries | No |
| `examples/5_Flux_Ensemble/data/openet_refet/` | Locally extracted OpenET bias-corrected ETo and ETr | No |
| `{workspace}/data/` | Meteorology, properties, flux records, shapefiles, and SWIM containers | No |
| `{workspace}/results/` | PEST++ outputs, evaluation bundles, and run archives | No |

The Volk v2.1 closure-corrected flux files must be supplied separately under
the configured `daily_flux_files_2pt1` directory and remain subject to their
source data policy. File presence is not a completeness test.

## Primary Workflow

The commands below assume the checked-in canonical filesystem paths. Earth
Engine extraction and calibration require explicit operator approval.

### 1. Extract Inputs

Run all current extraction steps:

```bash
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/data_extract.py
```

Or run a checkpointed subset:

```bash
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/data_extract.py --steps etf_v21,refet --sites US-Bi1,US-Ne1
```

The default path uses synchronous Earth Engine retrievals. The
`ndvi_bucket`, `snodas_bucket`, and `properties_bucket` steps are
compatibility paths and are not part of the current E1 workflow.

### 2. Build and Validate the Base Container

```bash
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/container_prep.py --overwrite --getinfo
```

At this intermediate stage, verify every configured site has seasonal NDVI
coverage and finite meteorology, especially ETo, precipitation, solar
radiation, and maximum and minimum temperature. With `--getinfo`, ETf is
intentionally absent from the base container and the base is not
calibration-ready.

### 3. Build the Run Container

```bash
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/build_container.py --run <tag> --source <base-container>
```

This non-clobbering step copies the base container, ingests the six OpenET ETf
members and corrected ETo/ETr, constructs the simple-mean ensemble target, and
recomputes dynamics. `--mad` builds a diagnostic target and is not the primary
E1 configuration.

Before calibration, inspect the run-container validation summary and verify
that every site has at least one finite ETf capture for every target member,
seasonal NDVI coverage, and no all-null meteorological variable. Report and
resolve any incomplete site rather than dropping or filling it silently.

### 4. Calibrate

```bash
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/calibrate.py --config /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/5_Flux_Ensemble.toml --container <run-container> --results-tag <tag> --keep-pestrun
```

Calibration is an expensive inverse run. `calibrate.py` archives the raw
PEST++ trajectory before cleanup; `--keep-pestrun` retains the working
directories as well.

### 5. Evaluate E1

Canonical daily benchmark:

```bash
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/evaluate.py --config /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/5_Flux_Ensemble.toml --par-csv /data/ssd1/swim/5_Flux_Ensemble/results/run22/5_Flux_Ensemble.3.par.csv --container /data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble_run22.swim --openet-source volk --output-dir <daily-output-dir> --bootstrap-reps 10000 --bootstrap-seed 42 --quiet-sites
```

Canonical monthly benchmark:

```bash
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/evaluate.py --config /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/5_Flux_Ensemble.toml --par-csv /data/ssd1/swim/5_Flux_Ensemble/results/run22/5_Flux_Ensemble.3.par.csv --container /data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble_run22.swim --monthly --output-dir <monthly-output-dir> --bootstrap-reps 10000 --bootstrap-seed 42 --quiet-sites
```

The daily evaluator writes the following authoritative bundle:

| Artifact | Role |
|---|---|
| `evaluation_grouped_daily_metrics.csv` | Pooled and weighted estimates with intervals |
| `evaluation_grouped_daily_contrasts.csv` | Paired SWIM-minus-OpenET contrasts |
| `evaluation_grouped_daily_metadata.json` | Inputs, configuration, hashes, and record contract |
| `evaluation_paired_daily_records.csv` | Canonical paired observations and retrieval-support class |
| `evaluation_metrics.csv` | Secondary per-site metrics |
| `evaluation_sites_excluded.csv` | Exclusion and eligibility ledger |

Monthly output follows the same `evaluation_grouped_monthly_*` convention and
adds `evaluation_monthly_metrics.csv`. Automatic parameter discovery and the
`diy` OpenET source are diagnostic conveniences, not publication provenance.

### 6. Decompose Retrieval Support

`overpass_decomposition.py` is a strict consumer of the complete daily
evaluator bundle. It does not reconstruct OpenET independently.

```bash
uv run python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/overpass_decomposition.py --evaluator-output-dir <daily-output-dir> --output-dir <temporal-output-dir> --bootstrap-reps 10000 --seed 42
```

Missing, stale, or hash-mismatched parent artifacts are hard errors. Use
`--legacy-site-products` only when reproducing the former compatibility
tables.

## E0 Vegetation-Formulation Arms

| Configuration | Vegetation response | Transpiration term |
|---|---|---|
| `5_Flux_Ensemble.toml` | Sigmoid Kcb-NDVI | `fc * Ks * Kcb` |
| `5_Flux_Ensemble_fao56_sig.toml` | Sigmoid Kcb-NDVI | `Ks * Kcb` |
| `5_Flux_Ensemble_fao56.toml` | Linear Kcb-NDVI | `Ks * Kcb` |

Each arm is calibrated independently on the same cohort and satellite target.
`pooled_arm_compare.py` compares two arms on one identical flux-observation
mask; run it for the two prespecified contrasts. It exits nonzero when its
configuration-identity gate fails.

## Supporting and Maintenance Analyses

| Script | Purpose | Execution class |
|---|---|---|
| `archive_run.py` | Complete the provenance, input-health, posterior, and evaluation archive for a calibrated run | Post-processing; writes archive |
| `spread_error.py` | Relate OpenET member spread to acquisition-date ETf error | Read-only analysis |
| `conditioned_ensemble_uncertainty.py` | Compare retrieval spread with conditioned-parameter ensemble spread | Read-only inputs; writes dedicated results |
| `run_weighting_ablation.py` | Compare spread and fixed-denominator observation weighting | Expensive calibration unless `--summary-only` |
| `within_e2_transfer.py` | Within-E1 leave-region-out and leave-one-site-out parameter transfer | Forward runs; legacy filename |
| `rebuild_e1_benchmark_evidence.py` | Rebuild or verify the frozen legacy-named E1 evidence package | Evidence maintenance; not the normal evaluator |

The primary evaluator now implements the pooled and square-root-record-length
weighted OpenET comparison directly. The former `volk_replication.py` workflow
is superseded and must not be used.

## Repository File Map

| File | Role |
|---|---|
| `5_Flux_Ensemble.toml` | Primary E1 and cover-scaled E0 configuration |
| `5_Flux_Ensemble_fao56_sig.toml` | E0 unscaled-sigmoid configuration |
| `5_Flux_Ensemble_fao56.toml` | E0 unscaled-linear configuration |
| `data_extract.py` | Earth Engine and GridMET extraction entry point |
| `container_prep.py` | Base-container builder |
| `build_container.py` | Six-member target and reference-ET run-container builder |
| `calibrate.py` | PEST++ IES calibration wrapper |
| `evaluate.py` | Canonical E1 flux/OpenET evaluator |
| `overpass_decomposition.py` | Retrieval versus between-retrieval consumer |
| `pooled_arm_compare.py` | E0 paired formulation comparator |
| `archive_run.py` | Run archive materializer |

## Validation

Run the complete repository suite before committing logic changes:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
```

Focused tests for the E1 benchmark contract are under
`tests/unit/test_e2_grouped_benchmark_metrics.py`,
`tests/unit/test_e1_paired_record_temporal.py`,
`tests/unit/test_overpass_decomposition.py`,
`tests/unit/test_benchmark_regression.py`, and
`tests/unit/test_rebuild_verify.py`. The first filename retains a legacy paper
number; its tests exercise the current E1 evaluator.
