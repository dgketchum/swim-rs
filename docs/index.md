# SWIM-RS

**S**oil **W**ater **I**nverse **M**odeling with **R**emote **S**ensing

swim-rs ingests remote sensing (NDVI/ETf), meteorology, and properties into a unified container, computes irrigation/groundwater dynamics, runs a daily soil water balance, and supports calibration (PEST++ IES), forecasting, analysis, and visualization.

## Modern workflow

```
swim extract (EE + met) → swim prep (build .swim container) → build_swim_input (HDF5) → run_daily_loop / calibrate
```

`prep/` and `model/` are deprecated; the container (`swimrs.container`) and process engine (`swimrs.process`) are canonical. `prepped_input.json` remains for compatibility (see `DEPRECATION_PLAN.md`).

## Quick start (shipped data)

```bash
pip install -e .
swim prep examples/2_Fort_Peck/2_Fort_Peck.toml
swim evaluate examples/2_Fort_Peck/2_Fort_Peck.toml --out-dir /tmp/swim_fp
```

For fresh EE/GridMET pulls, run `swim extract <config.toml> --add-sentinel` first (EE auth required).

## Highlights

- Zarr-backed `.swim` containers with provenance and coverage checks.
- Numba-powered physics via `swimrs.process`.
- PEST++ IES calibration with spinup/localization helpers.
- Remote sensing + met: Landsat/Sentinel NDVI, OpenET ETf, GridMET/ERA5-Land, SNODAS.
- Forecast/analysis: NDVI analog forecasts, metrics vs flux/OpenET, Plotly viz.

## Documentation

- Architecture: container / process
- CLI: extract, prep, calibrate, evaluate, inspect
- API reference: per-package pages with legacy section
- Config templates: TOML requirements, flags for mask/LULC/met

## Links

- GitHub: https://github.com/dgketchum/swim-rs
- License: CC-BY-NC-4.0 (see `LICENSE_CC-BY-NC-4.0.txt`)
