# Example 7: Metered Applied Water (paper Experiment E3)

Fifty sprinkler-irrigated fields in Colorado's San Luis Valley (SLV), each
served by one metered groundwater well, 2000–2024 model window with metered
pumping 2011–2021. SWIM-RS simulates irrigation internally from the satellite
ET-fraction record; the metered volume is revealed only at scoring, never used
as a model input. Two parameter paths are scored on the same 408 field-years:
calibration at each field (E1 method, six-member OpenET v2.1 ETf target) and
the E1-derived irrigated parameter set applied without local calibration.

The build covers 110 fields: the 50 SLV fields, 50 metered Idaho Eastern Snake
Plain Aquifer (ESPA) fields, and 10 ESPA rainfed negative controls. The paper
reports SLV only; ESPA records are upstream diversions that include conveyance
losses outside the field, so they are kept as supporting output and never
quoted. Every consumer filters on the `SLV_` site-id prefix.

**Naming.** Paper E3 is this directory. It was planned as "E4" before the paper
dropped an experiment, and the frozen files, the transfer-mapping metadata, and
one test still carry the `e4_` prefix. Those names are kept; the numbering in
the paper is E3.

This directory produces:

| Paper artifact | Analysis | Output |
|---|---|---|
| Table 6, Fig. 6, and the §3.4 bootstrap sentence | `evaluate_applied_water.py`, once per parameter path | `results/applied_calibrated/per_field_year.csv`, `results/applied_transfer_run22_by_irrigation/per_field_year.csv` |
| (same, last hop) | `scripts/figures/build_figure_data.py --only fig06 --only fig06_bootstrap`: SLV re-cut, the nine statistics, 10,000 whole-field resamples (seed 42) | `paper/data/final/figures/fig06_{field_years,field_summaries,bootstrap_effects}.csv` |
| Fig. 1 SLV panel | `build_figure_data.py --only fig01` reads the e7cal container geometry and the HUC8 context layer | `paper/data/final/figures/fig01_*` |
| Supplement S5.2 (field selection rules) | `select_fields.py` constants and `select_slv()` | `metered_truth.csv`, `notes/selection_qc.md` |
| Transfer-arm per-field parameters | `build_applied_irrigation_mapping.py` | `paper/data/final/e4_irrigation_stratified_param_mapping{,_metadata}.json` |
| Tables 1 and 2 E3 rows | hand-authored from the TOML and this README | — |

Table 6 is typed from `fig06_bootstrap_effects.csv` (`value_local_calibration`,
`value_e1_irrigated_transfer`, and the `ci95_*` columns for the bootstrap
sentence). The figure builder and the table images are owned by the dedicated
figure and table agents; this directory ends at the two `per_field_year.csv`
files and the frozen mapping JSON.

## Layout

```
7_Applied_Water/
├── 7_Applied_Water.toml                 project config; `root` sets every on-disk location
├── ex7_paths.py                         workspace, container, results, and archive paths derived from the TOML
├── select_fields.py                     SLV + ESPA cohort selection → fields shapefile + withheld metered_truth.csv
├── espa_control_irrmapper.py            IrrMapper never-irrigated gate for the ESPA control pool (Earth Engine)
├── cohort_irrmapper.py                  writes `irr_mean` onto the cohort for QGIS inspection (Earth Engine)
├── irrmapper_timeseries.py              per-field annual IrrMapper fraction, cohort QC (Earth Engine)
├── data_extract.py                      Earth Engine extraction (Example 5 extractors, repointed)
├── container_prep.py                    base container from the extracts (Example 5 wrapper)
├── build_container.py                   run container: six-member ETf target, OpenET refET, dynamics
├── archive_prelaunch.py                 RUN_POLICY Cats 1–2 before calibration
├── archive_postcalibration.py           RUN_POLICY Cats 4–5: merged posterior, bounds, phi history
├── build_applied_irrigation_mapping.py  irrigated/rainfed class → per-field parameter mapping for the transfer arm
├── evaluate_applied_water.py            annual simulated irrigation vs metered depth, per parameter path
├── transfer_applied_water.py            supporting: the pooled cropland-median vector, not in the paper
├── field_accuracy.py                    supporting: 110-field between/within-field decomposition
├── paired_field_bootstrap.py            supporting: 110-field basin-stratified paired bootstrap
├── data/                                untracked: metered_truth.csv, ETf and refET CSVs from data_extract.py
└── notes/                               untracked: plan, as-built note, results write-ups
```

Every script reads its locations from the TOML, either through `ProjectConfig`
or through `ex7_paths.py`, which interpolates the same `{root}`-based templates
without creating directories. The only absolute path left in the example is the
external IDWR 2015 irrigated-lands shapefile in `select_fields.py`.

## Inputs

| Input | Location | Built by |
|---|---|---|
| SLV metered pumping and parcels | `data/co_slv_wells/{co_slv_well_year_applied_depth.parquet, co_slv_irrigated_parcels.fgb, co_slv_parcel_well_links.parquet}` | `data/co_slv_wells/{pull_structures,pull_divrec,build_parcels,build_wells}.py` (CDSS HydroBase REST, RGDSS irrigated lands; see that README) |
| ESPA metered diversions | `data/idwr_wmis/idwr_wmis_applied_water.fgb`, `pou_polygons.fgb` | `data/idwr_wmis/{download_wmis,download_pou,build_fgb,download_pou_geom}.py` (IDWR WMIS; see that README) |
| ESPA field polygons | `/nas/irrmapper/raw_field_polygons/ID/ESPA/2015_Irrigated_Lands_*.shp` | IDWR 2015 irrigated-lands inventory |
| ESPA control gate | `data/idwr_wmis/espa_control_irrmapper.csv` | `espa_control_irrmapper.py` |
| Cohort shapefile and truth table | `{root}/7_Applied_Water/data/gis/applied_water_fields.shp`, `examples/7_Applied_Water/data/metered_truth.csv` | `select_fields.py` |
| Meteorology, soils, snow, irrigation status, NDVI, ETf | `{root}/7_Applied_Water/data/` and `examples/7_Applied_Water/data/{etf_v21_openet_eto,openet_refet}/` | `data_extract.py` (GridMET, SSURGO, SNODAS, IrrMapper, OpenET v2.1) |
| Transfer-arm parameter vectors | `paper/data/final/e2_run22_transfer_vectors_by_irrigation.json` | Example 5 (paper E1, Run 22) |
| Containers | `{root}/7_Applied_Water/data/7_Applied_Water.swim` (base), `7_Applied_Water_e7cal.swim` (run) | `container_prep.py`, `build_container.py` |

`metered_truth.csv` holds `site_id, year, metered_depth_mm, metered_volume_af,
acres, method, source`. SLV depth is pumped acre-feet over the served parcel
acres; ESPA depth is flow-metered diversion over the place-of-use acres. It is
read by `evaluate_applied_water.py` at scoring and, for `site_id` and `source`
only, by the mapping builder.

## Workflow

```bash
EX7=/home/dgketchum/code/swim-rs/examples/7_Applied_Water
CFG=$EX7/7_Applied_Water.toml
E7=/data/ssd1/swim/7_Applied_Water          # the TOML `root` + project
```

**1. Ground truth** (once; open state APIs, no credentials)

```bash
uv run python data/co_slv_wells/pull_structures.py && uv run python data/co_slv_wells/pull_divrec.py && uv run python data/co_slv_wells/build_parcels.py && uv run python data/co_slv_wells/build_wells.py
uv run python data/idwr_wmis/download_wmis.py && uv run python data/idwr_wmis/download_pou.py && uv run python data/idwr_wmis/build_fgb.py && uv run python data/idwr_wmis/download_pou_geom.py
uv run python $EX7/espa_control_irrmapper.py          # Earth Engine
```

**2. Select the cohort and write the withheld truth table**

```bash
uv run python $EX7/select_fields.py
```

Selection rules are the module constants (40-acre floor, common field crops,
200–1,200 mm applied depth, Polsby–Popper ≥ 0.60, five qualifying years, area
CV ≤ 0.15, service-area agreement within 5%, the 50 fields nearest the median
area). `cohort_irrmapper.py` and `irrmapper_timeseries.py` are optional Earth
Engine QC passes over the written cohort.

**3. Extract inputs** (Earth Engine, quota-gated)

```bash
uv run python $EX7/data_extract.py
```

**4. Build the containers**, then run the completeness check in `CLAUDE.md`

```bash
uv run python $EX7/container_prep.py --overwrite --getinfo
uv run python $EX7/build_container.py --run e7cal
```

Calibration and evaluation take `--container $E7/data/7_Applied_Water_e7cal.swim`.
The base container has no ensemble ETf target.

**5. Calibrate** (compute-gated; read `examples/RUN_POLICY.md` first)

```bash
uv run python $EX7/archive_prelaunch.py --config $CFG --container $E7/data/7_Applied_Water_e7cal.swim --run-name e7cal --command "<the batch_runner line below>"
uv run python -m swimrs.calibrate.batch_runner --config $CFG --container $E7/data/7_Applied_Water_e7cal.swim --action calibrate-all --reals 200 --noptmax 3 --workers 20 --batch-size 50 --exclude-uncovered
uv run python $EX7/archive_postcalibration.py --container $E7/data/7_Applied_Water_e7cal.swim --pestrun $E7/pestrun --run-name e7cal --noptmax 3
```

Three GFID-grouped batches (49/48/13 fields), 200 realizations, three IES
iterations, eight parameters per field. The post-calibration archive writes
`results/e7cal/archive/4_pest_outputs/merged/merged_posterior.json`, the
evaluator's input.

**6. Score the calibrated path**

```bash
uv run python $EX7/evaluate_applied_water.py --container $E7/data/7_Applied_Water_e7cal.swim --params-json $E7/results/e7cal/archive/4_pest_outputs/merged/merged_posterior.json --label calibrated
```

**7. Score the transferred irrigated set**

```bash
uv run python $EX7/build_applied_irrigation_mapping.py --verify-keys $E7/results/applied_calibrated/per_field_year.csv
uv run python $EX7/evaluate_applied_water.py --container $E7/data/7_Applied_Water_e7cal.swim --params-json paper/data/final/e4_irrigation_stratified_param_mapping.json --label transfer_run22_by_irrigation
```

The mapping builder assigns the irrigated vector to every metered field and the
rainfed vector to the ten controls, from the site-id prefix and the IrrMapper
gate only; it writes the two `e4_*` JSONs into `paper/data/final/` by default
(`--out-dir` elsewhere). The evaluator writes `per_field_year.csv`,
`summary_metrics.csv`, `negative_controls.json`, and a scatter under
`results/applied_<label>/`.

**8. Paper statistics and figures** (figure agent)

```bash
uv run python scripts/figures/build_figure_data.py --only fig06 --only fig06_bootstrap --only fig01
```

The builder hardcodes the two `results/applied_*` directories and the e7cal
container, restricts to `SLV_`, asserts 50 fields and 408 field-years under both
paths with identical metered depths, and writes the `fig06_*` tables from which
Table 6 and Fig. 6 are typed and drawn.

**9. Supporting analyses** (110 fields; not in the paper)

```bash
uv run python $EX7/transfer_applied_water.py                                  # pooled cropland-median vector → results/applied_transfer
uv run python $EX7/field_accuracy.py --label calibrated                       # between/within-field decomposition per basin
uv run python $EX7/paired_field_bootstrap.py                                  # basin-stratified paired bootstrap → results/applied_local_vs_transfer_run22
```

`results/applied_transfer_run22/` (pooled Run 22 vector) and
`paper/data/final/e4_irrigation_stratified_transfer_summary.csv` (pooled vs
stratified on all 110 fields) are earlier supporting outputs with no tracked
producer; neither is cited.

## Results on disk

| Directory | Content | Status |
|---|---|---|
| `results/e7cal/archive/` | RUN_POLICY Cats 1–2, 4–6 for the batch calibration | canonical |
| `results/applied_calibrated/` | step 6 | canonical; Table 6 / Fig. 6 local arm |
| `results/applied_transfer_run22_by_irrigation/` | step 7 | canonical; Table 6 / Fig. 6 transfer arm |
| `results/applied_transfer/`, `applied_transfer_run22/`, `applied_local_vs_transfer_run22/`, `COMPARISON.md` | step 9 and earlier 110-field cuts | supporting |
| `results/e7cal/archive_superseded_singlerun_20260706/` | pre-launch capture of the abandoned single-run attempt | superseded |

## Tests

```bash
uv run pytest tests/unit -q -k "ex7_paths or applied_irrigation_mapping or e4_paired_field_bootstrap"
```

## Rules

- Metered pumping is validation only. No script reads a metered value before
  `evaluate_applied_water.py`; the mapping builder records that it read only the
  site roster.
- Irrigation status comes from IrrMapper and the model's own scheduler. The
  parameter class fixes the transfer vector per field; it never forces irrigation.
- `summary_metrics.csv` reports `r2` as Nash–Sutcliffe against the 1:1 line, not
  a regression R². The within-field anomaly `bias_pct` is a zero-over-zero
  statistic; do not table it.
- The ETf and refET CSVs live beside the scripts in `data/` (Example 5
  convention); the per-scene remote sensing and meteorology live under the TOML
  `root`.
