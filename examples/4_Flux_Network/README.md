# 4_Flux_Network example

Field-scale SWIM runs across a network of ~160 flux stations (CONUS), using the container-first workflow. Data sources: Landsat NDVI/ETf (SSEBop target), GridMET met, SNODAS, SSURGO soils, CDL/LANID irrigation masks. This is a full, reproducible workflow (extract → container build → calibration → evaluation) used in analysis, not a minimal teaching demo.

## What this example does
- Build a `.swim` container from the flux station shapefile (`fields_shapefile`), ingest met/RS/snow/properties, compute fused NDVI and dynamics.
- Run SWIM for selected sites (`run.py`, `evaluate.py`) and compare against flux and SSEBop (see `ssebop_evaluation.py`).
- Model inputs are generated on-the-fly from the container via `build_swim_input()` (container→HDF5).

## Data expectations
- Shapefile: `data/gis/flux_stations.shp` (copy/symlink from `examples/gis/flux_stations.shp`).
- Remote sensing extracts: under `data/remote_sensing/landsat/extracts/...` (and `sentinel/...` if present).
- GridMET: `data/meteorology/gridmet` (optionally bias corrections in `data/bias_correction_tif`).
- Properties: `data/properties/4_Flux_Network_*` CSVs (ssurgo, landcover, irrigation).
- Snow: `data/snow/snodas/extracts`.
- Flux: `data/daily_flux_files` for evaluation.

## Workflow

### Prerequisites
- Conda environment: `conda activate swim`
- For calibration: `conda install conda-forge::pestpp`
- For data extraction: Earth Engine authentication

### Step 1: Setup shapefile
```bash
python setup_shapefile.py
```

### Step 2: Extract data (optional if data already present)
```bash
python data_extract.py
# or: swim extract 4_Flux_Network.toml
```

### Step 3: Build container
```bash
python container_prep.py --overwrite

# Limit to specific sites:
python container_prep.py --overwrite --sites US-ARM,US-Ne1
```

### Step 4: Run single site
```bash
python run.py --site US-ARM
# Output: {project_ws}/testrun/US-ARM/US-ARM.csv
```

### Step 5: Evaluate against flux/SSEBop
```bash
# Single site:
python evaluate.py --sites US-ARM --gap-tolerance 5

# All sites:
python evaluate.py --output-dir results --gap-tolerance 5
```

### Step 6: Calibrate with PEST++
```bash
# Basic calibration:
python calibrate_group.py

# With PDC (prior-data conflict) detection:
python calibrate_group.py --pdc-remove

# Limit sites and workers:
python calibrate_group.py --sites US-ARM,US-Ne1 --workers 4
```

### Quick test (single site end-to-end)
```bash
python setup_shapefile.py
python container_prep.py --overwrite --sites US-ARM
python run.py --site US-ARM
python evaluate.py --sites US-ARM
```
