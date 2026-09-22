# Example 4: CONUS Flux Network (not in the paper)

Field-scale SWIM runs across 160 CONUS flux stations (all land cover types), using
SSEBop NHM as the sole ETf calibration target. Default project date range: 1987-2025.
Specific experiments may intentionally use shorter windows.

This example was the paper's first CONUS benchmark and was removed from the
manuscript on 2026-08-22; the paper's CONUS experiment is Example 5
(`examples/5_Flux_Ensemble`, paper E1). The `julyphysics` run below is kept as run
history and as an all-land-cover reference for the SSEBop-only target.

## Data sources

- **ETf**: USGS SSEBop NHM (`projects/usgs-gee-nhm-ssebop/assets/ssebop/landsat/c02`)
- **NDVI**: Landsat + Sentinel (fused)
- **Meteorology**: GridMET (ETo, ETr, prcp, tmin, tmax, srad, u2, ea, bias-corrected)
- **Snow**: SNODAS SWE
- **Soils/properties**: SSURGO, CDL, LANID irrigation masks

## Setup

Every on-disk location is `root` from `4_Flux_Network.toml` plus the project name
(`{root}/4_Flux_Network/`); the flux files under `data/` are symlinks to the flux
archive.

```bash
# Install dependencies
uv sync --all-extras
```

## Workflow

### Step 1: Setup shapefile
```bash
python setup_shapefile.py
```
Creates `data/gis/flux_fields.shp` from canonical repo data (all land cover types, no filter).

### Step 2: Extract data (if not already present)
```bash
python data_extract.py --extract nhm     # SSEBop NHM ETf only
python data_extract.py --extract ndvi    # NDVI only
python data_extract.py --extract all     # everything
```

### Step 3: Build container
```bash
python container_prep.py --overwrite
python container_prep.py --overwrite --sites US-ARM,US-Ne1  # subset
```

### Step 4: Run single site
```bash
python run.py --site US-ARM
```

### Step 5: Calibrate with PEST++
```bash
python calibrate.py
```

### Step 6: Evaluate
```bash
python evaluate.py --sites US-ARM
python evaluate.py                   # all sites
python evaluate.py --etf             # ETf comparison at capture dates
```

## Reference run: `julyphysics`

`julyphysics` (2026-07-16) is a PEST++ IES recalibration under the source-exclusive
irrigation/gwsub physics (container dynamics recomputed so the tightened irrigation
windows are baked in). Full results, the land-cover-stratified tables, and the
failure-mode analysis are in [`notes/E1_RESULTS.md`](notes/E1_RESULTS.md) (written
while this example still carried the E1 label); the RUN_POLICY archive is at
`results/julyphysics/archive/`.

- Container: `data/4_Flux_Network_julyphysics.swim`
- Posterior parameters: `results/julyphysics/4_Flux_Network.3.par.csv`
- Cohort: 160 configured → 124 daily / 109 monthly (finite metrics, ≥10 paired months).
- Headline (all sites, medians): daily NSE **0.460** vs SSEBop 0.403, MBE
  **+0.196** mm/day, KGE 0.634; monthly NSE **0.611** vs 0.531, MBE +4.123 mm/mo.
- Croplands: daily NSE **0.667** / KGE 0.734 / MBE +0.247; monthly NSE 0.843 /
  KGE 0.837. Forest and wetland sites carry the largest class-median MBE (+0.666)
  and amplify the all-site wet bias, but the positive bias is not exclusive to them
  (crop+grass +0.196; +0.096 excluding forest/wetland). See `notes/E1_RESULTS.md`.

```bash
EX4=examples/4_Flux_Network
E4=/data/ssd1/swim/4_Flux_Network          # the TOML `root` + project
# Daily + monthly benchmarks (write to the tagged results dir)
uv run python $EX4/evaluate.py --par-csv $E4/results/julyphysics/4_Flux_Network.3.par.csv --container $E4/data/4_Flux_Network_julyphysics.swim --out-dir $E4/results/julyphysics
uv run python $EX4/evaluate.py --par-csv $E4/results/julyphysics/4_Flux_Network.3.par.csv --container $E4/data/4_Flux_Network_julyphysics.swim --out-dir $E4/results/julyphysics --monthly
```

## Quick test (single site end-to-end)
```bash
python setup_shapefile.py
python container_prep.py --overwrite --sites US-ARM
python run.py --site US-ARM
```
