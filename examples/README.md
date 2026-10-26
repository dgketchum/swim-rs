# SWIM-RS Examples

Two naming layers are in use. Repo example numbers are the stable filesystem and
script paths; paper labels (`T1`–`T3`, `E0`–`E3`) are what the manuscript and
reviewer-facing prose use. Use repo paths in commands.

| Paper label | Repo example | Directory | Role |
|---|---|---|---|
| `T1` | Example 1 | `examples/1_Boulder` | Tutorial: container build, extraction, ingestion, model run |
| `T2` | Example 2 | `examples/2_Fort_Peck` | Tutorial: single-site calibration |
| `T3` | Example 3 | `examples/3_Crane` | Tutorial: irrigated single-site calibration |
| `E0` | Example 6 (37 sites outside the E1 cohort) | `examples/6_Flux_International` | Vegetation-formulation comparison (Table 3, Fig. 2) |
| `E1` | Example 5 | `examples/5_Flux_Ensemble` | CONUS cropland benchmark against the OpenET ensemble (Table 4, Figs. 3–4, 5a) |
| `E2` | Example 6 | `examples/6_Flux_International` | International period-of-record benchmark and parameter transfer (Table 5, Fig. 5b) |
| `E3` | Example 7 | `examples/7_Applied_Water` | Metered applied-water validation (Table 6, Fig. 6) |
| — | Example 4 | `examples/4_Flux_Network` | CONUS all-land-cover SSEBop benchmark; not in the paper |
| — | Examples 8, 9 | `examples/8_Soil_Moisture`, `examples/9_Bushland` | Soil-moisture and lysimeter checks built on Example 5; not in the paper |

Some frozen file names carry a label other than the paper's: the
`paper/data/final/e2_*` files belong to E1 (Example 5), and `e3_*` / `e4_*`
names belong to E2 (Example 6) and E3 (Example 7). The names are kept because
the frozen hashes and the figure builder key on them; each experiment README
has a naming note.

## Reproducing the paper

The paper's findings are reproducible from a clone of this repository plus six
data packages (built by `scripts/package_reproduction_data.py`; their
`MANIFEST.json` records the commit and the contents of every archive). Until
submission the packages are at `gs://wudr/swim/reprod/` (a private bucket; ask
for access); at submission they move to a Zenodo record and this section will
cite its DOI. Nothing else is required: calibration and Earth Engine extraction
are not re-run, and PEST++ binaries are not needed.

| Package | Unpack into | Content |
|---|---|---|
| `swim_data_e1.tar.gz` | data root | Example 5 containers, flux and OpenET deliveries, the calibration archive and analysis outputs |
| `swim_data_e2.tar.gz` | data root | Example 6 containers (three formulation arms), flux QAQC files, run archives, transfer and E0 outputs |
| `swim_data_e3.tar.gz` | data root | Example 7 containers, inputs, calibration archive, scored outputs |
| `swim_repo_local.tar.gz` | clone root | untracked repo-local inputs: `examples/5_Flux_Ensemble/data`, `examples/7_Applied_Water/data`, `data/cartographic`, the ESPA control gate `data/idwr_wmis/espa_control_irrmapper.csv` |
| `swim_reference_final.tar.gz` | clone root | the frozen `paper/data/final/` (untracked in git) |
| `swim_e3_truth_sources.tar.gz` | clone root, optional | the raw E3 metered-truth pulls, for re-running the Example 7 truth build offline |

The reproduction has four steps: install, unpack, recompute the evidence of
each experiment into scratch directories, and compare. Set these four
variables once per shell; every command below uses them and nothing else needs
editing:

```bash
export SWIM_REPO=/path/to/swim-rs          # the clone; every command runs from here
export SWIM_ROOT=/path/to/swim             # the data root: the three swim_data_* packages unpack into it
export SWIM_PACKAGES=/path/to/packages     # where the downloaded tarballs sit
export SWIM_SCRATCH=/path/to/scratch       # an empty directory, outside the clone and the data root, for rebuilt evidence
```

`SWIM_ROOT` is the one the code reads: the test suite, every script, and the
figure builder relocate the TOML `root` to it, and a bare
`PermissionError: /data/ssd1/swim` means it is not set in the current shell.
The other three are shell conveniences for the commands in this section. Put
the four lines in a file you can `source`, and in PyCharm add `SWIM_ROOT` to
the environment variables of the Run/Debug configuration template and of the
pytest template; every `uv run python <script>` line below is then that script
run from the IDE with the same arguments.

### 1. Install

Clone the repository at the commit named in `MANIFEST.json` (branch `dev`
until submission) into `$SWIM_REPO` and install it (Python 3.13; `uv.lock` is
authoritative):

```bash
git clone git@github.com:dgketchum/swim-rs.git $SWIM_REPO
uv sync --all-extras                        # from $SWIM_REPO
```

Every `{root}`-derived path in the project TOMLs, the `ex*_paths.py` helpers,
and the figure builder follows `SWIM_ROOT` (`project_root_override` on
`ProjectConfig.read_config` wins when a script passes one).

### 2. Unpack

The bucket is private: ask for access for your Google account. `gsutil` needs
an interactive `gcloud auth login`; with application-default credentials alone
it returns 401, but the JSON API accepts the same credentials:

```bash
mkdir -p $SWIM_PACKAGES $SWIM_ROOT $SWIM_SCRATCH
gsutil -m cp "gs://wudr/swim/reprod/*.tar.gz" gs://wudr/swim/reprod/MANIFEST.json $SWIM_PACKAGES/   # after gcloud auth login
# or, with application-default credentials only:
TOKEN=$(gcloud auth application-default print-access-token); for f in MANIFEST.json swim_data_e1.tar.gz swim_data_e2.tar.gz swim_data_e3.tar.gz swim_repo_local.tar.gz swim_reference_final.tar.gz swim_e3_truth_sources.tar.gz; do curl -sfL -H "Authorization: Bearer $TOKEN" -o $SWIM_PACKAGES/$f "https://storage.googleapis.com/storage/v1/b/wudr/o/swim%2Freprod%2F$f?alt=media"; done
for p in e1 e2 e3; do tar -xzf $SWIM_PACKAGES/swim_data_$p.tar.gz -C $SWIM_ROOT; done
tar -xzf $SWIM_PACKAGES/swim_repo_local.tar.gz -C $SWIM_REPO
tar -xzf $SWIM_PACKAGES/swim_reference_final.tar.gz -C $SWIM_REPO
tar -xzf $SWIM_PACKAGES/swim_e3_truth_sources.tar.gz -C $SWIM_REPO   # optional: only to re-run the Example 7 truth build
```

A truncated download fails at `tar`; there is nothing else to verify. Then run
the test suite, which rebuilds the figure package from the shipped data and
compares it with the frozen one:

```bash
uv run pytest tests/ -v
```

### 3. Recompute the evidence

The commands below are the evaluation steps of the three experiment READMEs
(`5_Flux_Ensemble/README.md`, `6_Flux_International/README.md`,
`7_Applied_Water/README.md`), which describe what each one computes and the
files it writes. The READMEs also hold the container-build and calibration
steps, which the packages make unnecessary. Where a script takes an output
option, write under `$SWIM_SCRATCH` and compare with the shipped copy; a few
scripts write only into the shipped run directory, and the comparison for
those is the frozen-package check in step 4. For the CSV comparisons, `cmp`
is silent when the files are identical; a rebuild on other hardware may
differ in the last digit of a few statistics, which `diff` shows. A cohort,
sign, or file-set difference is a failure.

**E1 (Example 5).** Daily and monthly benchmark, retrieval-support
decomposition, spread–error, conditioned-ensemble spread, weighting-ablation
summary, within-E1 transfer.

```bash
EX5=examples/5_Flux_Ensemble; CFG=$EX5/5_Flux_Ensemble.toml; DATA=$SWIM_ROOT/5_Flux_Ensemble/data; RUN23=$SWIM_ROOT/5_Flux_Ensemble/results/run23; S=$SWIM_SCRATCH/e1
uv run python $EX5/evaluate.py --config $CFG --par-csv $RUN23/5_Flux_Ensemble.3.par.csv --container $DATA/5_Flux_Ensemble_run23.swim --openet-source volk --output-dir $S/daily --bootstrap-reps 10000 --bootstrap-seed 42 --quiet-sites
uv run python $EX5/evaluate.py --config $CFG --par-csv $RUN23/5_Flux_Ensemble.3.par.csv --container $DATA/5_Flux_Ensemble_run23.swim --monthly --output-dir $S/monthly --bootstrap-reps 10000 --bootstrap-seed 42 --quiet-sites
for f in evaluation_grouped_daily_metrics.csv evaluation_grouped_daily_contrasts.csv evaluation_paired_daily_records.csv; do cmp $S/daily/$f $RUN23/archive/6_evaluation/$f; done
for f in evaluation_grouped_monthly_metrics.csv evaluation_grouped_monthly_contrasts.csv; do cmp $S/monthly/$f $RUN23/archive/6_evaluation/$f; done
uv run python $EX5/overpass_decomposition.py --evaluator-output-dir $S/daily --output-dir $S/temporal --bootstrap-reps 10000 --seed 42
for f in evaluation_temporal_grouped_metrics.csv evaluation_temporal_grouped_contrasts.csv evaluation_temporal_interactions.csv; do cmp $S/temporal/$f $RUN23/temporal_decomposition/$f; done
uv run python $EX5/spread_error.py --config $CFG --out-dir $S/spread_error
for f in spread_error_summary.csv spread_error_quintiles.csv spread_error_persite.csv; do cmp $S/spread_error/$f $RUN23/spread_error/$f; done
uv run python $EX5/conditioned_ensemble_uncertainty.py                                  # rewrites $RUN23/conditioned_ensemble_uncertainty/ in place
uv run python $EX5/run_weighting_ablation.py --tag run23 --container $DATA/5_Flux_Ensemble_run23ablation.swim --summary-only   # rewrites results/ablation_run23_summary/ in place
uv run python $EX5/within_e1_transfer.py --out $S/within_e1_transfer
for f in summary_metrics.csv region_breakdown.csv win_rates.csv persite_daily.csv persite_monthly.csv; do cmp $S/within_e1_transfer/$f $RUN23/within_e1_transfer/$f; done
uv run python $EX5/e1_period_split.py                                                   # gate: replicates the frozen daily estimates, prints the §3.2 split
```

**E0 and E2 (Example 6).** Daily, monthly, and ETf evaluation of the
published calibration, the E1 parameter transfer, the closure-pool summaries,
the E0 formulation comparison, and product parity.

```bash
EX6=examples/6_Flux_International; CFG=$EX6/6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr.toml; RUN=6_Flux_International_LSEnsemble_GrassBasis_POR_annual2yr; RESULTS=$SWIM_ROOT/6_Flux_International/results; ARCHIVE=$RESULTS/$RUN/archive; S=$SWIM_SCRATCH/e2
POST=$ARCHIVE/4_pest_outputs/merged/merged_posterior.csv; CONTAINER=$SWIM_ROOT/6_Flux_International/data/6_Flux_International_ls_ensemble_grassbasis_por_annual2yr.swim
uv run python $EX6/evaluate.py --config $CFG --par-csv $POST                            # these five write into $RESULTS/$RUN/
uv run python $EX6/evaluate.py --config $CFG --par-csv $POST --monthly
uv run python $EX6/evaluate.py --config $CFG --par-csv $POST --etf
uv run python $EX6/pooled_metrics.py --results-dir $RESULTS/$RUN
uv run python $EX6/derived_metrics.py --config $CFG --uncalibrated --out $RESULTS/$RUN/derived
uv run python $EX6/transfer/build_e2_irrigation_mapping.py --container $CONTAINER --out-dir $RESULTS/$RUN/transfer_refresh --allow-unexpected
uv run python $EX6/transfer_ex5_params.py --config $CFG --params paper/data/final/e2_run23_transfer_vector.json --params-by-site $RESULTS/$RUN/transfer_refresh/e3_irrigation_stratified_param_mapping.json --container $CONTAINER --e2-results-dir $RESULTS/$RUN --out $S/transfer
uv run python $EX6/evaluation_summary.py --config $CFG --run-name $RUN                   # writes $ARCHIVE/6_evaluation/
uv run python $EX6/closure_pool_summary.py --config $CFG --run-name $RUN
uv run python $EX6/awc_recal/monthly_28day_sensitivity.py --run-name $RUN
A=$POST; SIG=$RESULTS/${RUN}_fao56_sig/archive/4_pest_outputs/merged/merged_posterior.csv; LIN=$RESULTS/${RUN}_fao56/archive/4_pest_outputs/merged/merged_posterior.csv
for SET in disjoint37:disjoint_sites.txt pool47:closure_pool_sites.txt; do TAG=${SET%%:*}; SITES=$EX6/e0_disjoint/${SET##*:}; uv run python $EX6/pooled_arm_compare.py --a-name grassbasis --a-config $CFG --a-par $A --b-name fao56_sig --b-config $EX6/${RUN}_fao56_sig.toml --b-par $SIG --sites-file $SITES --out-dir $RESULTS/e0_disjoint/grassbasis_vs_fao56_sig/$TAG; uv run python $EX6/pooled_arm_compare.py --a-name grassbasis --a-config $CFG --a-par $A --b-name fao56 --b-config $EX6/${RUN}_fao56.toml --b-par $LIN --sites-file $SITES --out-dir $RESULTS/e0_disjoint/grassbasis_vs_fao56/$TAG; uv run python $EX6/pooled_arm_compare.py --a-name fao56_sig --a-config $EX6/${RUN}_fao56_sig.toml --a-par $SIG --b-name fao56 --b-config $EX6/${RUN}_fao56.toml --b-par $LIN --sites-file $SITES --out-dir $RESULTS/e0_disjoint/fao56_sig_vs_fao56/$TAG; done
uv run python $EX6/product_parity/e1_e2_product_parity.py --out-dir $S/product_parity
uv run python $EX6/promote_final.py --config $CFG                                        # check: match / match~ for every file
```

The six `pooled_arm_compare.py` invocations print `GATE PASS` when arm A wins
the pooled metrics and `GATE FAIL`, with exit status 1, when it does not: the
two `grassbasis` pairs pass and the `fao56_sig` vs `fao56` pair fails, because
the sigmoid arm scores below the linear one (Table 3). The exit status is the
comparison's verdict, not an error. The `promote_final.py` check reports
`match` (byte-equal) or `match~` (equal within floating-point round-off) for
every file of `paper/data/final/e2_closure_pool/` against the run directory
the commands above refilled; build stamps, machine paths, and the hashes of
inputs the steps rewrote are not compared.

**E3 (Example 7).** The two parameter paths scored on the metered fields, and
the Table 6 statistics.

```bash
EX7=examples/7_Applied_Water; E7=$SWIM_ROOT/7_Applied_Water; S=$SWIM_SCRATCH/e3
uv run python $EX7/evaluate_applied_water.py --container $E7/data/7_Applied_Water_e7cal.swim --params-json $E7/results/e7cal/archive/4_pest_outputs/merged/merged_posterior.json --label calibrated --out $S/calibrated
uv run python $EX7/build_applied_irrigation_mapping.py --verify-keys $E7/results/applied_calibrated/per_field_year.csv --out-dir $S/mapping
cmp $S/mapping/e4_irrigation_stratified_param_mapping.json paper/data/final/e4_irrigation_stratified_param_mapping.json
uv run python $EX7/evaluate_applied_water.py --container $E7/data/7_Applied_Water_e7cal.swim --params-json paper/data/final/e4_irrigation_stratified_param_mapping.json --label transfer_run23_by_irrigation --out $S/transfer
for a in calibrated:applied_calibrated transfer:applied_transfer_run23_by_irrigation; do for f in per_field_year.csv summary_metrics.csv negative_controls.json; do cmp $S/${a%%:*}/$f $E7/results/${a##*:}/$f; done; done
uv run python $EX7/e3_period_split.py                                                   # gate: replicates the frozen Table 6 values, prints the §3.4 split
```

The evaluator scores every metered field it finds (100 fields, 730
field-years); Table 6 is computed in step 4 after restricting to the 50 SLV
fields and 404 field-years.

### 4. Compare

```bash
uv run python examples/5_Flux_Ensemble/rebuild_e1_benchmark_evidence.py --output-dir paper/data/final --verify   # G-VALUES: PASS
uv run python examples/5_Flux_Ensemble/promote_final.py                                                           # match / match~ for every file
uv run python scripts/figures/build_figure_data.py --all --out $SWIM_SCRATCH/figures --compare                   # 0 table(s) differ
tar -dzf $SWIM_PACKAGES/swim_reference_final.tar.gz -C $SWIM_REPO                                                 # silent when identical
```

`--verify` re-derives the E1 benchmark statistics from the frozen record; the
Example 5 `promote_final.py` check compares the shipped spread–error and
transfer outputs with the frozen `e2_*` files; `--compare` rebuilds every
figure table into a scratch directory (seeded from the frozen package) and
reports any table whose content the rebuild does not reproduce; the `tar`
check confirms that nothing under `paper/data/final/` was rewritten. Content
means the records and values: the comparison allows floating-point round-off,
ignores build stamps and the hashes of inputs that step 3 rewrote, and
compares recorded paths below the data root and the clone. Record every
deviation and guess in a findings log alongside the comparison.

The `promote_*.py` scripts in the experiment directories are the authoring
workflow that froze `paper/data/final/`. Their default mode is the check used
above; `--write` replaces the frozen package with the scratch bundle and is
never part of a reproduction.

The packages drop the PEST++ ensemble files (`*.obs.csv`, `loc.mat`, `*.rei`,
`*.jcb`, worker directories) that no evaluation or figure script reads; the
one exception, the posterior observation ensemble read by
`conditioned_ensemble_uncertainty.py`, is kept. The full archives stay on the
authoring machine under the run policy.

## From runs to the paper

Reproduction runs through three layers.

1. **Experiment directories.** Each of `5_Flux_Ensemble`, `6_Flux_International`,
   and `7_Applied_Water` has a README whose producer table names, for every table
   and figure it feeds, the analysis script and the archived output under
   `results/<run>/archive/` or `results/applied_*/`. Every on-disk location is
   derived from `root` in the project TOML through the example's `ex*_paths.py`
   module. Calibration and Earth Engine steps are compute-gated and documented, not
   re-run for the paper.
2. **Frozen data, `paper/data/final/`.** The paper's source of record. Example 5
   and Example 6 promote their archive outputs with a `promote_final.py` that
   compares by default and writes on request (`e1_openet_benchmark/`,
   `e2_closure_pool/` with a hashed `MANIFEST.json`, and the loose `e2_*` E1
   files). `scripts/figures/build_figure_data.py` reads the archives and frozen
   files and writes `figures/` with `fig_manifest.json` (`--all`, or
   `--only <name>` per table). Two builders read the existing package, so a
   scratch `--out` directory is seeded with a copy of the frozen `figures/`
   before the build; `--compare` then reports every table the rebuild does
   not reproduce.
3. **Typeset tables and figures.** The manuscript's tables and figure
   proofs are typeset from layer 2 outside this repository; no experiment
   script writes them, and they are not part of the reproduction.

Tables with no producer script are hand-authored and are checked against the
configuration, not regenerated: **Table 1** and **Table 2** (domains, inputs, and
calibration constraints, from the project TOMLs and the experiment READMEs) and
**Table S1** (fixed and derived field properties, from the forward-model code and
data sources). **Table S2** is mixed: its bounds columns come from the prior
bounds in `src/swimrs/calibrate/pest_builder.py` (the class-specific `mad`
bounds, 0.10–0.30 irrigated and 0.30–0.80 rainfed, are set per site by the
calibration's irrigation status and restated in Example 6
`transfer/build_ex5_irrigation_stratified_params.py`) and its transferred-value
columns from `paper/data/final/e2_run23_transfer_vectors_by_irrigation.json`,
which Example 6 `transfer/` produces.

## Policies

- `examples/VALIDATION_POLICY.md`: flux, meter, and soil-moisture data are
  validation only and never configure inputs.
- `examples/RUN_POLICY.md`: what every calibration run archives, and the
  provenance a reported number must carry.
