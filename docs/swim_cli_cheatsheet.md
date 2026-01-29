# SWIM CLI Cheatsheet

Fast reference for the `swim` CLI. Default order: extract → prep (.swim) → calibrate/evaluate. Run `swim --help` or `swim <command> --help` for full detail.

## Example: Fort Peck (Example 2)
```
# Data are shipped in the repo for this example; you can skip extract unless you need fresh EE/GridMET pulls.
# If refreshing, authenticate EE first (https://earthengine.google.com/), e.g.:
#   earthengine authenticate
# CLI docs: https://developers.google.com/earth-engine/guides/command_line
# Then run:
#   swim extract examples/2_Fort_Peck/2_Fort_Peck.toml --etf-models ptjpl

# 2) Prep model inputs (build .swim and export HDF5/json)
swim prep examples/2_Fort_Peck/2_Fort_Peck.toml

# 3) Calibrate with defaults
swim calibrate examples/2_Fort_Peck/2_Fort_Peck.toml

# 4) Evaluate (debug CSVs)
swim evaluate examples/2_Fort_Peck/2_Fort_Peck.toml --metrics-out /tmp/fort_peck_metrics
```

## Common Flags
- `config`: Path to project TOML (required for extract/prep/calibrate/evaluate).
- `--out-dir`: Override output root (defaults to directory containing the TOML).
- `--sites siteA,siteB`: Restrict to specific site IDs.
- `--workers N`: Parallel workers for multi-site steps (dynamics, calibration).

## Extract (Earth Engine + GridMET/ERA5)
```
swim extract path/to/project.toml \
  --export drive|bucket \
  --bucket my-bucket \        # required when --export=bucket
  --file-prefix swim/runs \   # bucket key prefix
  --add-sentinel \            # include Sentinel-2 NDVI
  --etf-models ssebop,ptjpl \
  --no-snodas --no-properties --no-rs --no-met      # skip pieces as needed
  --use-gridmet-centroids \     # optional: map GridMET via provided centroids
  --gridmet-correction          # optional: apply ETo/ETr bias correction rasters
```
Outputs: EE exports (Drive or bucket), GridMET parquet, properties extracts.

## Prep (Container ingest → prepped_input.json)
```
swim prep path/to/project.toml \
  --add-sentinel            # ingest Sentinel-2 NDVI if available
  --overwrite               # overwrite existing container data
  --landsat-only-ndvi       # skip Sentinel and export Landsat-only NDVI
  --use-lulc-irr            # use LULC-based irrigation detection (no masks)
  --international           # alias for LULC mode with no-mask NDVI/ETf
  --no-ndvi --no-etf --no-met --no-snow  # skip parts as needed
```
Does: create/open the `.swim` container, ingest properties/NDVI/ETf/met/SNODAS, compute merged NDVI and dynamics, and export model-ready inputs (HDF5/JSON).

## Calibrate (PEST++ IES)
```
swim calibrate path/to/project.toml \
  --realizations 300 \      # override config
  --python-script custom_forward.py
```
Does: build PEST project, run spinup (noptmax=0), run IES (default noptmax=3). Uses the `.swim` container as the data source. Outputs PEST files and calibrated params.

## Evaluate (Debug run + CSV)
```
swim evaluate path/to/project.toml \
  --forecast-params params.csv \
  --spinup spinup.json \
  --flux-dir flux_csvs/ \
  --openet-dir openet_exports/ \
  --metrics-out metrics/
```
Does: run model in debug mode, write per-site CSVs, optional metrics vs flux/OpenET.

## Inspect (Container status)
```
swim inspect path/to/container.swim --detailed
```
Shows inventory, coverage, provenance; `--detailed` includes provenance log.

## Minimal Single-Site Smoke (prep + run)
```
swim prep examples/1_Boulder/1_Boulder.toml --sites US-Bo1 --out-dir /tmp/swim_demo
swim evaluate examples/1_Boulder/1_Boulder.toml --sites US-Bo1 --out-dir /tmp/swim_demo
```

## Tips
- Authenticate EE before `extract` (`earthengine authenticate`).
- For bucket exports, set `--bucket` or `config.ee_bucket`.
- Met source is set in `[data_sources]` (`met_source = "gridmet"` or `"era5"`); `--no-met` skips met downloads.
- Use `--sites` to shorten turnaround while debugging.
- Use `--workers` to speed multi-site dynamics/calibration on multi-core machines.
