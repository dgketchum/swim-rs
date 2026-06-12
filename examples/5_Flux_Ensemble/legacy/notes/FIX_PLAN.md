# Example 5 Regression Recovery Plan

Date: 2026-02-17
Baseline: FIX.md root-cause analysis

## Problem Summary

60-site EE calibration shows severe SWIM regression (median R2: 0.419 vs 0.640 baseline).
Three root causes identified in FIX.md:

1. **Missing Landsat NDVI** — empty irr/inv_irr dirs, only Sentinel ingested (known Case D failure mode)
2. **Custom PDC pre-pass** — `reals=5` for 480 params strips 52% of ETf obs before calibration
3. **Ensemble ETf scaling bug** — `et_ensemble_mad` is ET (mm x 1000), not ETf; code divides
   by 10000 instead of dividing by 1000 then by ETo (dormant in failing run, but wrong for
   future `ensemble_source=openet` runs)

Phases are ordered by risk and dependency. Each phase produces a measurable result
before proceeding to the next.

---

## Phase 1: Restore Landsat NDVI (data-only, no code changes)

### What

Copy Landsat NDVI CSVs from Example 4 to Example 5 for matching sites and years.

### Source / destination

```
Source irr:     /data/ssd1/swim/4_Flux_Network/data/remote_sensing/landsat/extracts/ndvi/irr/
Source inv_irr: /data/ssd1/swim/4_Flux_Network/data/remote_sensing/landsat/extracts/ndvi/inv_irr/
Dest irr:       /data/ssd1/swim/5_Flux_Ensemble/data/remote_sensing/landsat/extracts/ndvi/irr/
Dest inv_irr:   /data/ssd1/swim/5_Flux_Ensemble/data/remote_sensing/landsat/extracts/ndvi/inv_irr/
```

### Site filter

Copy only CSVs whose filename contains a site_id present in the Ex5 shapefile
(`/data/ssd1/swim/5_Flux_Ensemble/data/gis/flux_fields.shp`, 60 sites).

Ex4 NDVI filenames follow the pattern: `ndvi_{SITE_ID}_{mask}_{YEAR}.csv`

### Year filter

Keep only files for years 2016-2024 (matching Ex5 study period).

### Known gap

MB_Pch is not in Example 4. It has Sentinel NDVI only and shows 0 evaluation
days in the current run. Accept this gap for now.

### Expected file count

59 sites x 9 years x 2 masks = ~1,062 files per mask dir. Some site-years may
have multiple CSVs due to path/row overlap in Ex4 naming; copy all matching files.

### Command

Write a Python copy script to `commands.sh` that:
1. Reads Ex5 site_id list from shapefile
2. Globs Ex4 NDVI irr/inv_irr dirs
3. Copies matching files (site_id in filename, year 2016-2024) to Ex5 dirs
4. Reports count of files copied and any missing sites

### Validation

After copy, confirm:
- Both `irr/` and `inv_irr/` dirs are non-empty
- At least 59 unique site_ids represented
- Years span 2016-2024

---

## Phase 2: Rebuild container and calibrate without custom PDC pre-pass

### What

Rebuild the .swim container with restored NDVI, then calibrate with the custom
PDC pre-pass disabled. This tests both fixes simultaneously.

### Code change

`examples/5_Flux_Ensemble/calibrate.py` line 150:

```python
# Before:
    run_pest_sequence(
        cfg,
        results,
        pdc_remove=True,       # custom pre-pass ON
        debug_fields=DEBUG_FIELDS,
    )

# After:
    run_pest_sequence(
        cfg,
        results,
        pdc_remove=False,      # custom pre-pass OFF
        debug_fields=DEBUG_FIELDS,
    )
```

This skips the `noptmax=-1, reals=5` dry-run and the subsequent rebuild with
`conflicted_obs=temp_pdc`. The built-in PEST++ `ies_drop_conflicts=true`
(`pest_builder.py` line 856) still handles conflicts during the main run.

### What pdc_remove controls (calibrate.py lines 77-103)

When `pdc_remove=True`:
1. `write_control_settings(noptmax=-1, reals=5)` — tiny ensemble, eval-only
2. `dry_run(exe_)` — generates `{project}.pdc.csv` listing conflicted obs
3. Rebuilds PestBuilder with `conflicted_obs=temp_pdc` — zeros those obs weights
4. Then proceeds to actual calibration with reduced observation set

When `pdc_remove=False`:
1. `write_control_settings(noptmax=0)` — single dry-run for diagnostics only
2. `dry_run(exe_)` — no PDC file generated (noptmax=0 skips conflict detection)
3. Proceeds directly to calibration with full observation set

### Container rebuild command

```bash
python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/container_prep.py \
    --overwrite --openet-source ee
```

### Calibration command

```bash
python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/calibrate.py
```

Results dir: `/data/ssd1/swim/5_Flux_Ensemble/results/case_b_60site_ee_9yr/`

### Success criteria

- SWIM median R2 >= 0.55 (substantial recovery toward 0.640 baseline)
- No site drops below R2 = -0.5 that was previously positive
- ETf nonzero-weight count stays near 17,926 (no massive pruning)
- Ensemble median R2 remains stable (~0.58)

---

## Phase 3: PDC sensitivity test (conditional — only if Phase 2 < 0.55)

### What

Test whether the built-in `ies_drop_conflicts` is itself harmful by running
with it disabled.

### Code change

`src/swimrs/calibrate/pest_builder.py` line 856:

```python
# Test A (current default):
pst.pestpp_options["ies_drop_conflicts"] = "true"

# Test B (disable):
pst.pestpp_options["ies_drop_conflicts"] = "false"
```

### Protocol

1. Phase 2 already produced results with `ies_drop_conflicts=true`
2. Run calibration with `ies_drop_conflicts=false` on same container
3. Compare: SWIM R2, phi trajectory, per-site metrics

### Results dirs

```
/data/ssd1/swim/5_Flux_Ensemble/results/case_b_pdc_true/   (from Phase 2)
/data/ssd1/swim/5_Flux_Ensemble/results/case_b_pdc_false/
```

---

## Phase 4: Fix ensemble ETf extraction scaling

### Problem

`etf_asset_extract.py` lines 58-59 treat `et_ensemble_mad` as ETf (divide by 10000),
but it is actually ET in mm x 1000 — same representation as geesebal/ptjpl/disalexi.
This means ensemble ETf extractions produce wrong values whenever `--model ensemble`
is used.

Current code:
```python
if model == "ensemble":
    return src_image.select("et_ensemble_mad").divide(10000).rename("etf")
```

### Fix

Remove the `if model == "ensemble"` special case. Add `"ensemble"` to `_ET_MODELS`
and handle its different band name (`"et_ensemble_mad"` instead of `"et"`).

```python
# Module level (line 43):
_ET_MODELS = {"geesebal", "ptjpl", "disalexi", "ensemble"}

# New band name mapping:
_ET_BAND = {"ensemble": "et_ensemble_mad"}  # all others use "et"

# _normalize_etf() replacement (lines 50-75):
def _normalize_etf(model, img_id):
    src_path = OPENET_V21[model]
    src_image = ee.Image(f"{src_path}/{img_id}")

    if model in _ET_MODELS:
        band = _ET_BAND.get(model, "et")
        et_mm = src_image.select(band).divide(1000)
        date = src_image.date()
        eto = (
            ee.ImageCollection(_GRIDMET)
            .filterDate(date, date.advance(1, "day"))
            .first()
            .select(_ETO_BAND)
        )
        return et_mm.divide(eto).rename("etf")

    # ssebop, sims, eemetric: et_fraction is integer x 10000
    return src_image.select("et_fraction").divide(10000).rename("etf")
```

Also update the module docstring (line 10):
```
- ensemble: ``et_ensemble_mad / 1000 / ETo``  (GridMET daily ETo)
```

### Re-extraction (requires EE, user confirmation)

After code fix, re-extract ensemble ETf for all 60 sites:

```bash
python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/etf_asset_extract.py \
    --shapefile /data/ssd1/swim/5_Flux_Ensemble/data/gis/flux_fields.shp \
    --model ensemble --mask irr --start-yr 2016 --end-yr 2024 \
    --bucket wudr --dest bucket

python /home/dgketchum/code/swim-rs/examples/5_Flux_Ensemble/etf_asset_extract.py \
    --shapefile /data/ssd1/swim/5_Flux_Ensemble/data/gis/flux_fields.shp \
    --model ensemble --mask inv_irr --start-yr 2016 --end-yr 2024 \
    --bucket wudr --dest bucket
```

This only matters when `ensemble_source = "openet"` is used. The current failing
run uses `ensemble_source = "computed"`, so this fix is forward-looking but should
be done before any openet-ensemble run.

---

## Phase 5: Guardrails

### NDVI presence check

Add to `calibrate.py` before `run_pest_sequence()`:

```python
container = SwimContainer.open(container_path, mode="r")
for mask in ("irr", "inv_irr"):
    key = f"remote_sensing/ndvi/landsat/{mask}"
    assert key in container._root, f"Missing Landsat NDVI ({mask}) in container"
container.close()
```

### PDC observation count logging

Add to `PestBuilder.build_pest()` or the PDC application path in calibrate.py:

```python
pre_count = (pst.observation_data["weight"] > 0).sum()
# ... apply PDC ...
post_count = (pst.observation_data["weight"] > 0).sum()
print(f"PDC: {pre_count} -> {post_count} nonzero-weight obs "
      f"({100*(1 - post_count/pre_count):.1f}% removed)")
```

### Per-site weight summary

Log top-10 sites by weight share and warn if any site loses >70% of ETf
observations after conflict handling.

---

## Execution Order

| Step | Phase | Blocks on | EE needed | Approx time |
|------|-------|-----------|-----------|-------------|
| 1    | Phase 1: Copy NDVI | nothing | no | minutes |
| 2    | Phase 4: Fix ensemble scaling code | nothing | no | minutes |
| 3    | Phase 2: Rebuild container | Step 1 | no | ~30 min |
| 4    | Phase 2: Calibrate (pdc_remove=False) | Step 3 | no | ~2 hr |
| 5    | Phase 2: Evaluate | Step 4 | no | minutes |
| 6    | Phase 3: PDC sensitivity (conditional) | Step 5 | no | ~2 hr |
| 7    | Phase 4: Re-extract ensemble ETf | Step 2 | yes | ~1 hr EE |
| 8    | Phase 5: Add guardrails | nothing | no | minutes |

Steps 1 and 2 can run in parallel. Steps 2 and 8 are independent of the main
pipeline and can be done anytime. The critical path is 1 -> 3 -> 4 -> 5.
