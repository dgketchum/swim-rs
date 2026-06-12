# Fix copy_openet_assets.py — Per-Site Export

## What Went Wrong

### The Union Geometry Problem

`_build_union_geometry()` (line 39) loads all 60 flux sites, buffers each by 4km,
then calls `unary_union()` to dissolve them into **one polygon**.  That polygon spans
the spatial extent of all 60 sites across CONUS — roughly 3000 km east-west by
2000 km north-south.

Every Landsat scene that intersects *any* site is then:
1. Clipped to this giant union polygon (line 238)
2. Exported at 30m in EPSG:5070 (line 254)

Even though `.clip()` masks pixels outside the polygon, the **bounding box** of the
export region is the full union extent.  EE's `Export.image.toAsset()` exports a
rectangular grid covering the `region` parameter's bounding envelope.  So each
exported image is a 30m raster spanning ~3000×2000 km — roughly 100,000 × 67,000
pixels per band.  At uint16 (2 bytes), that's ~13 GB uncompressed per band.  Even
with compression and masking, exports land at 50MB+ and consume 35,000 EECU.

### Why It Looks Like It Works

The code runs without error because:
- `filterBounds(union_geom)` correctly finds scenes overlapping the union
- `.clip(union_geom)` correctly masks pixels outside the polygon
- `Export.image.toAsset()` accepts the geometry

But EE exports the full bounding rectangle, not just the polygon interior.
Masked pixels still consume storage and compute.

## The Fix: Per-Site Export

### Target Architecture

Instead of one export per scene (clipped to union of all sites), do one export
per **site × scene** (clipped to that site's 4km buffer).

**Expected image size per site:**
- 4km buffer → 8km × 8km bounding box
- At 30m: ~267 × 267 = ~71,289 pixels per band
- uint16, 1–3 bands: ~140–430 KB per image
- EECU: trivial (seconds, not hours)

### Asset Organization

Current (broken):
```
openet_etf/v2_1/{model}/{scene_id}
```

Proposed — one sub-collection per site:
```
openet_etf/v2_1/{model}/{site_id}/{scene_id}
```

This keeps images small and lets downstream extraction find all scenes for a given
site via `ee.ImageCollection(f".../{model}/{site_id}")`.

### Code Changes to `copy_openet_assets.py`

1. **Remove `_build_union_geometry()`** — no longer needed.

2. **Add `_build_site_geometries()`** — returns a dict of `{site_id: ee.Geometry}`
   where each geometry is a single site's 4km buffer in EPSG:4326.

3. **Restructure `copy_openet_to_assets()` main loop:**
   ```
   for site_id, site_geom in sites.items():
       dst = f"{DEST_ROOT}/{model}/{site_id}"
       ensure_collection(dst)
       existing = list_existing(dst)
       for year in range(start_yr, end_yr + 1):
           scenes = discover_scenes(src, year, site_geom)
           for scene_id in scenes:
               if scene_id in existing: skip
               clip + export to dst/scene_id
   ```

4. **Adjust `_ensure_asset_exists()`** — create nested folder + collection
   (`openet_etf/v2_1/{model}/{site_id}`).

5. **Adjust `_list_existing_images()`** — list per-site collection.

6. **Export params** — `region=site_geom` (tiny 8km box), `scale=30`,
   `crs="EPSG:5070"`, `maxPixels=1e9` (way more than needed).

### Proof-of-Concept Test

Before scaling to all 60 sites × 7 models:

1. Pick one site (e.g., `S2`) and one model (e.g., `ssebop`)
2. Pick one year (e.g., 2020)
3. Export a single image
4. Verify:
   - Image size (should be <500 KB)
   - EECU (should be <100)
   - Band names and properties preserved
   - Can load and inspect with `ee.Image(asset_id).getInfo()`

### CLI Changes

Add `--single-test` flag that:
- Processes only the first site
- Processes only the first matching scene
- Prints image info (size, bands, region) before and after export
- Waits for the export to complete and reports EECU

### Scaling Strategy (after proof-of-concept)

- 60 sites × ~20 scenes/site/year × 9 years × 7 models ≈ 75,600 exports
- At <500 KB each: ~38 GB total (manageable)
- EE queue limit: 3000 tasks — batch by model, resume via skip logic
- Each export completes in seconds → throughput limited by task submission rate
