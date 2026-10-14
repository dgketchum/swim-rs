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
   files and writes `figures/` with `fig_manifest.json`; `figures/BUILD_STATUS.md`
   records each rebuild.
3. **Typeset tables and figure proofs.** `paper/tables/*.tex` and
   `paper/figures/proofs/` are produced from layer 2 by the dedicated table and
   figure agent (`paper/notes/tables_figures_handoff.md`), never by an experiment
   script.

Tables with no producer script are hand-authored and are checked against the
configuration, not regenerated: **Table 1** and **Table 2** (domains, inputs, and
calibration constraints, from the project TOMLs and the experiment READMEs) and
**Table S1** (fixed and derived field properties, from the forward-model code and
data sources). **Table S2** is mixed: its bounds columns come from the calibration
TOML and its transferred-value columns from
`paper/data/final/e2_run22_transfer_vectors_by_irrigation.json`, which Example 6
`transfer/` produces.

## Policies

- `examples/VALIDATION_POLICY.md`: flux, meter, and soil-moisture data are
  validation only and never configure inputs.
- `examples/RUN_POLICY.md`: what every calibration run archives, and the
  provenance a reported number must carry.
- `notes/calibration_guidance.md`: how to launch and monitor PEST++ runs safely.
