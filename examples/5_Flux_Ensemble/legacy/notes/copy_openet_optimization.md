# Optimization Plan: copy_openet_assets.py

## Problem

The GCS GeoTIFF archival export (`copy_openet_assets.py`) was killed on 2026-02-16
after 15 hours.  Root cause: per-site export creates one EE task per (site, scene)
pair, producing massive duplication when sites share Landsat path/rows.

**Evidence from partial run (ensemble model, 4 CA Delta sites):**
- 4,312 TIFs exported, but only 1,407 unique scenes — 3.06x duplication
- 4 co-located sites share 4 WRS path/rows × 3 sensors = 12 footprints

**Projected full run (7 models × 60 sites):**
- Per model: 60 sites × ~1,078 scenes/site ≈ 64,680 per-site exports
- Deduped: ~14,000 unique scenes per model (60 sites across ~40 path/rows)
- 7 models: **~450K tasks (old)** → **~98K tasks (deduped)** — 4.6x reduction

## Solution: Per-Scene Export with Multi-Site Clipping

### Phase 1: Scene Discovery (unchanged)

Query `aggregate_histogram("system:index")` per site (60 `getInfo()` calls per
model, ~2 min total).  Build a dedup mapping:

```
scene_to_sites: dict[str, set[str]]
  "lc08_044033_20200101" → {"US-Bi1", "US-Bi2", "US-Tw3"}
  "lc08_038032_20200103" → {"US-MC1"}
```

### Phase 2: Export (restructured)

For each unique scene:
1. Compute the shapely union of all overlapping sites' 4km buffers
2. Convert to `ee.Geometry`
3. Export **one** GeoTIFF per scene, clipped to the multi-site union

```
gs://wudr/openet/etf/{model}/{scene_id}.tif
```

Each TIF is small (~8km × 8km per site, spatially indexed), fully georeferenced,
and contains data around all sites it covers.

### File Naming Change

Old: `{site_id}_{scene_id}.tif` (one TIF per site×scene)
New: `{scene_id}.tif` (one TIF per unique scene)

The 4,312 existing TIFs with old naming will be ignored by the new skip logic.
Clean up with `gsutil -m rm gs://wudr/openet/etf/ensemble/US-*` after confirming
new exports work.

## EE Project Split

Run with two EE projects to double the effective queue (3000 tasks each):

| Project | Models | Est. Unique Scenes |
|---------|--------|--------------------|
| `ee-hoylman` | ssebop, sims, eemetric, geesebal | ~56K |
| `ee-dgketchum` | ptjpl, disalexi, ensemble | ~42K |

### Instructions for ee-hoylman

1. **Clone repo and set up environment:**
   ```bash
   git clone <repo-url>
   cd swim-rs
   uv sync --all-extras
   ```

2. **EE authentication** (one-time):
   ```bash
   earthengine authenticate --project ee-hoylman
   ```

3. **Get the shapefile** from `/data/ssd1/swim/5_Flux_Ensemble/data/gis/flux_fields.shp`
   (or regenerate with `python examples/5_Flux_Ensemble/setup_shapefile.py`).

4. **Run one model at a time** (each takes ~4-8 hours):
   ```bash
   nohup python examples/5_Flux_Ensemble/copy_openet_assets.py \
       --shapefile /path/to/flux_fields.shp \
       --models ssebop \
       --project ee-hoylman \
       > nohup_copy_ssebop.out 2>&1 &
   ```

5. **Single-site test first** to verify everything works:
   ```bash
   python examples/5_Flux_Ensemble/copy_openet_assets.py \
       --shapefile /path/to/flux_fields.shp \
       --models ssebop \
       --project ee-hoylman \
       --sites S2 \
       --single-test
   ```

6. **Monitor:** EE task dashboard at https://code.earthengine.google.com/tasks
   or tail the nohup log.

7. **Resume:** The script skips scenes already in GCS and tasks already pending
   in EE.  Safe to restart after interruption.

8. **After ssebop completes**, run `sims`, then `eemetric`, then `geesebal`.

## Estimated Runtime

- ~14,000 tasks per model
- EE queue: 3000 concurrent, ~30s completion each → ~100 tasks/min
- Per model: ~2.3 hours of active export + queue management overhead
- 4 models per project: ~12-16 hours total (sequential, with resume support)
- Two projects in parallel: ~8 hours wall time for all 7 models

## Future Optimization (not needed now)

Replace 60 per-site `getInfo()` calls with a single collection query + client-side
WRS2 spatial join.  This saves ~2 min per model but doesn't affect the dominant
cost (export task count).
