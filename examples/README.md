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

Paper numbering changed during preparation, so some frozen file names carry the
old labels: `paper/data/final/e2_*` files belong to E1 (Example 5), and `e3_*` /
`e4_*` names belong to E2 (Example 6) and E3 (Example 7). Each experiment README
has a naming note.

## Reproducing the paper

The paper's findings are reproducible from a clone of this repository plus six
data packages (built by `scripts/package_reproduction_data.py`; their
`MANIFEST.json` records the commit and the sha256 of every archive). Until
submission the packages are at `gs://wudr/swim/reprod/` (a private bucket; ask
for access); at submission they move to a Zenodo record and this section will
cite its DOI. Nothing else is required: calibration and Earth Engine extraction
are not re-run, and PEST++ binaries are not needed.

| Package | Unpack into | Content |
|---|---|---|
| `swim_data_e1.tar.gz` | data root | Example 5 containers, flux and OpenET deliveries, Run 23 archive and analysis outputs |
| `swim_data_e2.tar.gz` | data root | Example 6 containers (three formulation arms), flux QAQC files, run archives, transfer and E0 outputs, the superseded baseline's evaluator outputs |
| `swim_data_e3.tar.gz` | data root | Example 7 containers, inputs, calibration archive, scored outputs |
| `swim_repo_local.tar.gz` | clone root | untracked repo-local inputs: `examples/5_Flux_Ensemble/data`, `examples/7_Applied_Water/data`, `data/cartographic` |
| `swim_reference_final.tar.gz` | clone root | the frozen `paper/data/final/` (untracked in git) |
| `swim_e3_truth_sources.tar.gz` | clone root, optional | the raw E3 metered-truth pulls, for re-running the Example 7 truth build offline |

1. Clone the repository at the commit named in `MANIFEST.json` (branch `dev`
   until submission), then `uv sync --all-extras` and `uv run pytest tests/ -v`
   (Python 3.13; `uv.lock` is authoritative).
2. Pick a data root and unpack the packages:

   ```bash
   gcloud auth login                                       # the bucket is private; application-default credentials alone give 401
   gsutil -m cp "gs://wudr/swim/reprod/*" /path/to/packages/
   mkdir -p /path/to/swim && for p in e1 e2 e3; do tar -xzf /path/to/packages/swim_data_$p.tar.gz -C /path/to/swim; done
   tar -xzf /path/to/packages/swim_repo_local.tar.gz -C /path/to/swim-rs      # the clone root
   tar -xzf /path/to/packages/swim_reference_final.tar.gz -C /path/to/swim-rs
   ```

   The reference package has to be in place before step 4: the check modes
   compare what they recompute against it. Before step 4 also confirm that
   the shipped inputs are the ones the frozen figure tables were built from:

   ```bash
   uv run python scripts/figures/build_figure_data.py --check    # "checked N recorded sources; 0 changed or missing"
   ```

   Run it first: the experiment steps rewrite some of those inputs under the
   shipped run directories (equal within round-off, but no longer the same
   bytes), after which `--check` reports them as changed.
3. Point the configuration at the data root without editing any TOML:

   ```bash
   export SWIM_ROOT=/path/to/swim
   ```

   Every `{root}`-derived path in the project TOMLs, the `ex*_paths.py`
   helpers, and the figure builder follows it (`project_root_override` on
   `ProjectConfig.read_config` still wins when a script passes one).
4. Run the three experiment READMEs. Each Workflow section opens with a
   "With the data packages" paragraph that names the steps to run and the
   steps to skip (container builds, extraction, calibration):
   `5_Flux_Ensemble/README.md`, `6_Flux_International/README.md`,
   `7_Applied_Water/README.md`. The scripts recompute the `paper/data/final/`
   evidence from the shipped containers and run archives; the promotion
   scripts' default check modes report any file that differs from what is on
   disk.
5. The check modes are the comparison: every `promote_*.py` default run must
   report `match` (byte-equal) or `match~` (equal within floating-point
   round-off, which a rebuild on other hardware produces in the last digit of
   a few statistics) for every file and an unchanged manifest, and
   `rebuild_e1_benchmark_evidence.py --verify` must print `G-VALUES: PASS`.
   Scripts that write fresh outputs (for example the Example 7 evaluation)
   are compared against the shipped `results/` copies: the same records and
   columns, values equal within round-off. The figure builder runs into a
   scratch `--out` directory (seeded from the frozen package) with
   `--compare`, which must report `0 table(s) differ`. The sidecars and
   manifests the packages ship record the authoring machine's paths; the
   check modes compare them below the data root and the clone, so no editing
   is needed. When everything has run, confirm that nothing under
   `paper/data/final/` was rewritten:

   ```bash
   tar -dzf /path/to/packages/swim_reference_final.tar.gz -C /path/to/swim-rs   # silent when identical
   ```

   A cohort, sign, or file-set difference is a failure. Record every deviation
   and guess in a findings log alongside the comparison.

The packages drop the PEST++ ensemble files (`*.obs.csv`, `loc.mat`, `*.rei`,
`*.jcb`, worker directories) that no evaluation or figure script reads; the
one exception, the Run 23 posterior observation ensemble read by
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
   files and writes `figures/` with `fig_manifest.json`. Two builders read
   the previous package, so a scratch `--out` directory is seeded with a copy
   of the frozen `figures/` before the build; `--compare` then reports every
   table the rebuild does not reproduce.
3. **Typeset tables and figures.** The manuscript's tables and figure
   proofs are typeset from layer 2 outside this repository; no experiment
   script writes them, and they are not part of the reproduction.

Tables with no producer script are hand-authored and are checked against the
configuration, not regenerated: **Table 1** and **Table 2** (domains, inputs, and
calibration constraints, from the project TOMLs and the experiment READMEs) and
**Table S1** (fixed and derived field properties, from the forward-model code and
data sources). **Table S2** is mixed: its bounds columns come from the calibration
TOML and its transferred-value columns from
`paper/data/final/e2_run23_transfer_vectors_by_irrigation.json`, which Example 6
`transfer/` produces.

## Policies

- `examples/VALIDATION_POLICY.md`: flux, meter, and soil-moisture data are
  validation only and never configure inputs.
- `examples/RUN_POLICY.md`: what every calibration run archives, and the
  provenance a reported number must carry.
