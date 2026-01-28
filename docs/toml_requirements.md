# SWIM-RS Project TOML Requirements

Minimal settings needed to run the container-first extract → prep → calibrate workflow for a site, with Fort Peck (Example 2) as a concrete template. The `.swim` container is the canonical artifact; `prepped_input.json` is legacy/compatibility only. Inline comments highlight options and which entries are optional.

> Tip: Paths use `{root}` / `{project_workspace}` / `{data}` substitution. By default, `--out-dir` sets `{root}` to the directory containing the TOML.

## Minimal Template (Fort Peck)
```toml
project = "2_Fort_Peck"
root = "."  # override with --out-dir to relocate outputs

[paths]
project_workspace = "{root}"          # base for all derived paths
data = "{project_workspace}/data"     # data root (container, inputs, outputs)

# Container
container = "{data}/{project}.swim"   # SwimContainer path

[data_sources]
met_source = "gridmet"   # gridmet | era5
snow_source = "snodas"   # snodas | era5 (set to era5 when met_source=era5 to skip SNODAS)
mask_mode = "irrigation" # irrigation | none

# Remote sensing (Landsat required; Sentinel optional)
remote_sensing = "{data}/remote_sensing"
landsat = "{remote_sensing}/landsat"
landsat_ee_data = "{landsat}/extracts"   # EE CSVs/exports for Landsat
# sentinel = "{remote_sensing}/sentinel" # optional
# sentinel_ee_data = "{sentinel}/extracts"

# Meteorology (choose one primary source)
# met_source is set under [data_sources]; met points to the local directory for ingested met data.
met = "{data}/met_timeseries/gridmet"    # gridmet | era5 (ERA5-Land) local path
# met = "{data}/met_timeseries/era5"     # uncomment if using ERA5-Land

[ids]
feature_id = "site_id"  # column in shapefile with unique IDs
gridmet_join_id = "GFID"  # column in mapping shapefile for GridMET cell ID
gridmet_id = "GFID"       # column name in met files for GridMET cell ID
state_col = "state"       # state code (for irrigation masking)

[paths.gis]
fields_shapefile = "{paths.data}/gis/flux_fields.shp"     # required: polygons with feature IDs
gridmet_mapping = "{paths.data}/gis/flux_fields_gfid.shp"   # UID→GFID join (written by mapping step; required for GridMET)
# gridmet_centroids = "{paths.data}/gis/gridmet_centroids.shp" # optional: use provided GridMET centroids to map fields to shared GFIDs
# correction_tifs   = "{paths.data}/gis/bias_correction_tif"   # optional: directory with monthly ETo/ETr bias correction rasters
# gridmet_factors   = "{paths.data}/gis/flux_fields_gfid.json" # optional: sampled correction factors JSON (written if correction_tifs set)

[paths.properties]
properties_dir = "{paths.data}/properties"
irr = "{properties_dir}/{project}_irr.csv"       # irrigation CSV (optional if no mask)
ssurgo = "{properties_dir}/{project}_ssurgo.csv"    # soils CSV
lulc = "{properties_dir}/{project}_landcover.csv" # land cover CSV
properties_json = "{properties_dir}/{project}_properties.json"

[paths.snow]
snodas_in = "{paths.data}/snow/snodas/extracts"  # SNODAS extracts (optional if no snow)

[paths.outputs]
prepped_input = "{paths.data}/prepped_input.json"  # Deprecated/compatibility; container+HDF5 is preferred.

[earth_engine]
bucket = "wudr"  # only needed for EE bucket exports; drive exports don’t require this

[misc]
irrigation_threshold = 0.3
elev_units = "m"
refet_type = "eto"   # eto | etr
runoff_process = "cn"      # cn (SCS Curve Number) | ier (Infiltration excess runoff exceeding Ksat rate)

[date_range]
start_date = "1987-01-01"
end_date = "2022-12-31"

[crop_coefficient]
kc_proxy = "etf"   # etf is the only ET proxy currently available for calibration
cover_proxy = "ndvi"  # ndvi is the only basal crop coefficient (Kcb) currently available to drive transpiration

[calibration]
pest_run_dir = "{project_workspace}/data/pestrun"
etf_target_model = "ptjpl"      # ETf model for calibration (ptjpl in Fort Peck)
workers = 20            # calibration workers
realizations = 200           # PEST++ IES realizations
calibration_dir = "{pest_run_dir}/pest/mult"
obs_folder = "{pest_run_dir}/obs"
initial_values_csv = "{pest_run_dir}/params.csv"
spinup = "{pest_run_dir}/spinup.json"
python_script = "{project_workspace}/custom_forward_run.py"  # forward runner (can use default)

[forecast]
forecast_parameters = "{pest_run_dir}/pest/archive/{project}.3.par.csv"  # optional for forecast runs
```

## What’s Required vs Optional (Minimal Working Example)
- Required:
  - `project`, `root`
  - `paths.project_workspace`, `paths.data`, `paths.container`
  - `paths.gis.fields_shapefile`, `paths.gis.gridmet_mapping`
  - `paths.properties.ssurgo`, `paths.properties.lulc`, `paths.properties.properties_json`
  - `met` (one primary met source path)
  - `ids.feature_id`, `ids.gridmet_join_id`, `ids.gridmet_id`
  - `date_range` start/end
  - `crop_coefficient` proxies
  - `calibration.etf_target_model`, `pest_run_dir`, `initial_values_csv`, `spinup`
- Optional but common:
  - Sentinel paths (only if extracting Sentinel)
  - `earth_engine.bucket` (only for bucket exports; not needed for Drive)
  - `paths.snow.snodas_in` (if using SNODAS)
  - `forecast` block (only for forecast runs)
  - `state_col` (used for irrigation masking; keep if available)
  - `gridmet_centroids` (convenience for mapping/diagnostics)

## Notes
- Remote sensing: Landsat is the baseline; Sentinel entries can be commented out if not used.
- Met source: Fort Peck uses GridMET; ERA5-Land is supported by pointing `met` to the ERA5 directory and setting matching grid mapping if needed.
- Calibration: The custom forward script can be omitted if using the packaged default; keep paths consistent under `pest_run_dir`.
- EE auth: Needed only for fresh extracts. Shipped examples include data, so prep/calibrate/evaluate can run without EE. 
