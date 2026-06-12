# Example 5: Flux Ensemble — Publication Run Plan

## Goal

Clean, reproducible workflow on `/data/ssd1/swim/5_Flux_Ensemble/` for ~60 cropland
flux stations.  Rebuild container from scratch, run fresh PEST++ IES calibration,
evaluate SWIM against flux towers and all 6 OpenET models, produce publication-quality
figures and summary tables.

## Manuscript strategy

For the cross-example manuscript experiment design and publication claim framework, see:

- [`notes/RSE_EX4_EX5_PLAN.md`](notes/RSE_EX4_EX5_PLAN.md)

---

## Progress

### DONE

- [x] **CLAUDE.md**: Added `ls` commands as pre-approved
- [x] **Data migration (ssd2 → ssd1)**:
  - `remote_sensing/landsat/` (NDVI + SSEBop/PTJPL/SIMS/geeSEBAL ETf)
  - `remote_sensing/sentinel/` (NDVI extracts copied from ssd2)
  - `meteorology/gridmet/` (290 parquet files)
  - `snow/snodas/extracts/` (240 CSV files)
  - `properties/` (SSURGO, landcover, irrigation CSVs)
  - `bias_correction_tif/` (24 TIFs)
  - `gis/` (shapefiles, gridmet centroids/mapping)
  - `station_metadata.csv` (from canonical `examples/data/`)
- [x] **Pruned ssd1 data**: Removed 83,969 non-cropland CSV extracts + 6 stale
  `calibration_*` property files.  ssd1 now contains only the 60 cropland sites.
- [x] **geeSEBAL ETf**: Promoted from `openet/geesebal_etf/` to top-level
  `geesebal_etf/` (528 files/mask, cropland-only).  Source on ssd2 has full
  1,995 files/mask from OpenET composite assets.
- [x] **TOML updates**:
  - `gridmet_mapping`/`gridmet_factors` → `flux_fields_gfid` (filename fix)
  - `etf_target_model = "ensemble"` (was `"ssebop"`)
  - `etf_ensemble_members = ["ptjpl", "sims", "geesebal"]` (added geeSEBAL)
- [x] **container_prep.py fixes**:
  - `create_container(path=...)` → `create_container(uri=...)` (API mismatch)
  - `irrigation_csv=` → `irr_csv=` (kwarg name fix)
  - Removed stale `min_pairs`/`window_days` kwargs from `fused_ndvi()` call
  - Removed dead `export_model_inputs()` (used non-existent `prepped_input_json`)
  - Added `build_shapefile()` from canonical repo data as Step 0
  - Added `workers=cfg.workers` to all `ndvi()`/`etf()` calls (parallel CSV reading)
- [x] **Example 6 fix**: Removed stale `fused_ndvi()` kwargs (`instrument1`,
  `instrument2`, `min_pairs`, `window_days`)
- [x] **Ingestor parallelism** (`src/swimrs/container/components/ingestor.py`):
  - Extracted `_parse_single_csv()` as module-level pure function
  - Added `workers` param to `ndvi()`, `etf()`, `_parse_ee_remote_sensing_csvs()`
  - Uses `ProcessPoolExecutor` when `workers > 1`
  - Extracted `_filter_files_by_mask()` as static method
- [x] **Bulk zarr write** (`_write_timeseries`):
  - Replaced per-column `pd.to_numeric` + per-field zarr write with bulk
    `np.float64` cast and single `arr[:, col_indices] = values` slice
- [x] **Ensemble ETf in calculator** (`src/swimrs/container/components/calculator.py`):
  - Added `_load_ensemble_etf()` method — discovers all ETf models in the container,
    loads each, returns the per-date/per-field mean via `xr.concat + .mean(dim="model")`
  - `_load_dynamics_dataset()` now handles `etf_model="ensemble"` by calling this
  - `pest_builder.py` already had equivalent logic; now the calculator matches
- [x] **Container build**: Complete. 60 fields, all pass `validate_for_forward_run()`
  and `validate_for_calibration()`.  Contains:
  - Meteorology: 10 GridMET variables (eto, etr, eto_corr, etr_corr, prcp, tmin, tmax, srad, u2, ea)
  - NDVI: Landsat (irr/inv_irr) + Sentinel (irr/inv_irr)
  - ETf: 4 models (SSEBop, PTJPL, SIMS, geeSEBAL) × 2 masks (irr/inv_irr)
  - Merged NDVI: irr (49,614 obs) + inv_irr (30,928 obs) across 60 fields
  - SNODAS SWE
  - Properties: soils (awc, clay, sand, ksat), land cover, irrigation
  - Dynamics: kc_max, ke_max, irr_data, gwsub_data (all 60 fields populated)
- [x] **calibrate_group.py rewrite**: Clean 148-line script, no more dead
  `non_crop_sites` filter bug.  Supports `debug_fields` for single-site testing.
- [x] **pest_builder.py bug fixes** (2 bugs found during Ex5 calibration):
  - `spinup()` now creates `self.pest_dir` before writing `swim_input.h5`
    (was failing with `FileNotFoundError` on fresh runs)
  - `spinup()` now writes observation baseline files (`obs/obs_etf_{fid}.np`,
    `obs/obs_swe_{fid}.np`) to `self.obs_dir` from spinup model output.
    These are required by `pyemu.PstFrom.add_observations()` and were missing
    on fresh runs (only existed from prior runs in Ex3).
- [x] **S2 test calibration**: Successfully ran PEST++ IES for S2 only
  (20 realizations, 3 iterations, PDC removal).  Phi: 3278 → 713 → 704 → 703.
  Calibration machinery works end-to-end.
- [x] **All tests pass**: 557 passed, ruff clean

### SUPERSEDED — FOSS Re-extraction

The FOSS re-extraction approach (`reextract_stale_data.py`) is kept for users who
want to run their own ET models, but is no longer the primary ETf pipeline for Ex5.
It was abandoned after hitting the 3000 EE task queue limit on geeSEBAL inv_irr.

### DEFERRED — GCS GeoTIFF Export (Reproducibility)

`copy_openet_assets.py` exports per-site GeoTIFF chips from OpenET v2.1 source
collections to `gs://wudr/openet/etf/{model}/`.  This is for reproducibility only
(archiving the raw rasters).  Killed 2026-02-16 after 15 hours; per-task EE overhead
dominates (~0.5 EECU compute but ~72K tasks total across 7 models × 60 sites × ~1200
scenes).  Needs refactoring before resuming: batch by WRS path/row instead of
per-site, pre-query scene list once instead of 60 `getInfo()` calls.

### REMOVED — Stale Pre-v2.1 ETf Extracts

CSVs on ssd1 under `.../extracts/openet/` dated from April 2025 (pre-v2.1).
Removed from ssd1 working dir; originals preserved on ssd2 at
`/data/ssd2/swim/5_Flux_Ensemble/data/landsat/extracts/openet/`.

### IN PROGRESS — Extract v2.1 ETf Zonal Stats

Extract per-field zonal-mean ETf directly from official OpenET v2.1 source
collections.  `etf_asset_extract.py` updated to read from source collections
(not cached copies), with per-model band selection and normalization:

| Model | Source collection | Band | Normalization |
|-------|-------------------|------|---------------|
| ssebop | `.../ssebop/conus/gridmet/landsat/v2_1` | `et_fraction` | ÷ 10000 |
| sims | `.../sims/conus/gridmet/landsat/v2_1` | `et_fraction` | ÷ 10000 |
| eemetric | `.../eemetric/conus/gridmet/landsat/v2_1` | `et_fraction` | ÷ 10000 |
| geesebal | `.../geesebal/conus/gridmet/landsat/v2_1` | `et` | ÷ 1000 ÷ ETo |
| ptjpl | `.../ptjpl/conus/nldas2/landsat/v2_1` | `et` | ÷ 1000 ÷ ETo |
| disalexi | `.../disalexi/conus/cfsr/landsat/v2_1` | `et` | ÷ 1000 ÷ ETo |
| ensemble | `.../ensemble/conus/gridmet/landsat/v2_1` | `et_ensemble_mad` | ÷ 10000 |

**Masks:** irr + inv_irr (for calibration).  no_mask deferred.

**Extraction status:**

| Model | irr | inv_irr | Status |
|-------|-----|---------|--------|
| ssebop | — | — | Pending |
| sims | — | — | Pending |
| eemetric | — | — | Pending |
| geesebal | — | — | Pending |
| ptjpl | — | — | Pending |
| disalexi | — | — | Pending |
| ensemble | — | — | Pending |

### TODO — After v2.1 Extraction

1. **Sync from GCS** → local ssd1

2. **Rebuild container** with all 6 individual models + ensemble for evaluation:
   ```bash
   python examples/5_Flux_Ensemble/container_prep.py --overwrite
   ```

3. **Re-calibrate** (200 realizations, 20 workers, noptmax=3):
   ```bash
   python examples/5_Flux_Ensemble/calibrate_group.py
   ```

4. **Evaluate** — SWIM vs all 6 individual models + pre-computed ensemble

5. **Publication figures** — `publication_figures.py`

### TODO — Later (low priority)

6. **Refactor `copy_openet_assets.py`** for GCS archival (reproducibility):
   batch by WRS path/row, single scene-list query, reduce task count ~10x.

7. **Extract no_mask ETf** for fair unmasked OpenET comparison.

---

## Key Files

| File | Status | Notes |
|------|--------|-------|
| `5_Flux_Ensemble.toml` | Updated | ensemble target, 4 ETf members, path fixes |
| `container_prep.py` | Fixed | 6 API fixes, shapefile + parallel ingestion |
| `calibrate_group.py` | Rewritten | Clean script, debug_fields support, old bugs gone |
| `copy_openet_assets.py` | Deferred | Per-site GCS export too slow; needs batching refactor |
| `etf_asset_extract.py` | Done | v2.1 irr/inv_irr CSVs extracted for all 7 models |
| `reextract_stale_data.py` | Superseded | FOSS approach kept for reference, not primary pipeline |
| `evaluate_group.py` | Unchecked | May have stale API calls |
| `openet_evaluation.py` | Unchecked | May have stale API calls |
| `publication_figures.py` | TODO | New module for journal figures |
| `pest_builder.py` | Fixed | spinup creates obs files + pest_dir for fresh runs |
| `ingestor.py` | Updated | Parallel CSV reading, bulk zarr write |
| `calculator.py` | Updated | Ensemble ETf mean support |

## Data Locations

- **Working directory**: `/data/ssd1/swim/5_Flux_Ensemble/`
- **Container**: `/data/ssd1/swim/5_Flux_Ensemble/data/5_Flux_Ensemble.swim`
- **Canonical repo data**: `examples/data/` (footprints, metadata)
- **Source extracts (archive)**: `/data/ssd2/swim/5_Flux_Ensemble/data/`
- **GCS bucket**: `gs://wudr/5_Flux_Ensemble/` (re-extraction target)
- **EE task monitor**: https://code.earthengine.google.com/tasks

## Commits Pending

The `pest_builder.py` fixes need to be committed after pre-validation.
Files changed across sessions:
- `src/swimrs/calibrate/pest_builder.py` (spinup creates pest_dir + obs baseline files)
- `examples/5_Flux_Ensemble/reextract_stale_data.py` (added `--no-clear` resume flag)
