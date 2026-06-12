# Plan: Update Example 5 ETf extraction to OpenET v2.1 assets

## Context

OpenET now publishes ETf data at v2.1 asset paths. We use a two-step workflow:

1. **Copy** (clip only): Read from official OpenET v2.1 collections,
   clip raw integer images to 4km buffers around sites, write to user-owned
   cache at `projects/ee-dgketchum/assets/openet_etf/v2_1/{model}`
2. **Extract** (sparse zonal stats): Read from cached collection, compute
   per-field zonal means over flux footprints, export CSVs.
   Band selection and scaling happens at extraction time, not copy time.

**Source:** `projects/openet/assets/{model}/conus/{met}/landsat/v2_1`
**Cache:** `projects/ee-dgketchum/assets/openet_etf/v2_1/{model}`

**All 7 models available:** `ssebop`, `sims`, `eemetric`, `geesebal`, `ensemble`, `ptjpl`, `disalexi`

Band/scaling (applied at extraction, NOT during copy):
- `ssebop`, `sims`, `eemetric`: `et_fraction` / 10000
- `geesebal`, `disalexi`, `ptjpl`: `et` / 1000 / ETo (ref ET in mm, no scaling)
- `ensemble`: `et_ensemble_mad` / 10000

Met source varies by model: gridmet (most), nldas2 (ptjpl), cfsr (disalexi).

## Changes

### 1. `examples/5_Flux_Ensemble/copy_openet_assets.py` — DONE

- [x] v2_1 paths for all 7 models (ssebop, sims, geesebal, eemetric, ensemble, ptjpl, disalexi)
- [x] `DEST_ROOT = "projects/ee-dgketchum/assets/openet_etf/v2_1"`
- [x] Simplified to raw clip — no band selection, no scaling, no `_normalize_etf()`
- [x] Images copied as-is with `system:time_start`, `system:index`, `SPACECRAFT_ID` preserved
- [x] Cleared old (incorrectly scaled) images from all destination collections
- [x] disalexi copy running (2016-2025), test confirmed working

### 2. `examples/5_Flux_Ensemble/etf_asset_extract.py` — TODO

- Update `CACHED_ROOT` to `"projects/ee-dgketchum/assets/openet_etf/v2_1"`
- Update `ASSET_PATHS` to all 7 models under new root
- Cached images are raw integers — extraction must apply per-model band/scaling:
  - `ssebop`, `sims`, `eemetric`: `.select("et_fraction").divide(10000)`
  - `geesebal`, `disalexi`, `ptjpl`: `.select("et").divide(1000)` then divide by ETo
  - `ensemble`: `.select("et_ensemble_mad").divide(10000)`
- Update argparse choices and docstrings

### 3. `src/swimrs/data_extraction/ee/etf_export.py`

- Add `eemetric`, `ensemble` to the allowed model set (line 316 validation)
- Add lazy import for `openet.eemetric` in `_get_openet_module()`
- Add `eemetric` handling in `_compute_etf_image()` (same pattern as ssebop)
- Update `ASSET_COLLECTION_ROOT` to `"projects/ee-dgketchum/assets/openet_etf/v2_1"`

### 4. `examples/5_Flux_Ensemble/data_extract.py`

- Update `extract_openet_etf_assets()` default models to the 5 available
- Update `__main__` block models list
- Add `eemetric` to FOSS `model_exporters` in `extract_openet_etf()`

### 5. `src/swimrs/container/schema.py`

- Add `ENSEMBLE = "ensemble"` to `ETModel` enum so the pre-computed OpenET
  ensemble can be ingested as a stored model

### 6. `examples/5_Flux_Ensemble/5_Flux_Ensemble.toml`

- Update `etf_ensemble_members` to the 4 available individual models:
  ```toml
  etf_ensemble_members = ["ssebop", "sims", "geesebal", "eemetric"]
  ```

## Files modified

| File | Change |
|------|--------|
| `examples/5_Flux_Ensemble/copy_openet_assets.py` | **DONE** — raw clip for all 7 models, no scaling |
| `examples/5_Flux_Ensemble/etf_asset_extract.py` | TODO — per-model band/scaling at extraction time |
| `examples/5_Flux_Ensemble/data_extract.py` | 5-model lists |
| `src/swimrs/data_extraction/ee/etf_export.py` | +eemetric/ensemble in validation, FOSS eemetric |
| `src/swimrs/container/schema.py` | ENSEMBLE enum value |
| `examples/5_Flux_Ensemble/5_Flux_Ensemble.toml` | 4-member ensemble list |

## Verification

1. `ruff check --fix . && ruff format .`
2. `pytest tests/ -v`
3. Manual (EE auth): `python copy_openet_assets.py --shapefile <shp> --model ensemble --start-yr 2020 --end-yr 2020`
4. Manual (EE auth): `python etf_asset_extract.py --shapefile <shp> --model ensemble --start-yr 2020 --end-yr 2020`
