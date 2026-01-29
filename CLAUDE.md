# Project: SWIM-RS

**S**oil **W**ater **I**nverse **M**odeling with **R**emote **S**ensing

## Environment

This project uses **uv** for dependency management.

### First-time setup
```bash
uv sync --all-extras
```

### Running commands

Activate the venv:
```bash
source .venv/bin/activate
python <script.py>
pytest tests/ -v
```

Or use `uv run` without activating:
```bash
uv run python <script.py>
uv run pytest tests/ -v
```

### Adding dependencies
```bash
uv add <package>           # add to main dependencies
uv add --dev <package>     # add to dev dependencies
```

Do NOT use conda, pip, or virtualenv — this project uses uv exclusively.

## Running Tests
```bash
pytest tests/ -v
```

## Code Quality
```bash
ruff check .
```

Check touched code for cruft beyond what ruff does automatically. This includes:
- Unused imports and functions that ruff might miss (e.g., only called from tests)
- Dependencies in `pyproject.toml` that are no longer used
- Dead code paths after refactoring

## Tech Stack

### Geospatial
geopandas, rasterio, rioxarray, fiona, pyproj, shapely, rasterstats, pyogrio

### Remote Sensing / Evapotranspiration
earthengine-api, openet-core, openet-ssebop, openet-ptjpl, openet-sims, openet-disalexi, openet-geesebal, openet-landsat-lai

### Scientific Computing
numpy, scipy, scikit-learn, pandas, xarray

### Climate/Hydrology Data
netCDF4, zarr, pynldas2

### Inverse Modeling
pyemu (PEST parameter estimation)

## Key Dependencies

- **swimrs 0.1.0:** This project's core package (editable install via `uv sync`)

## Project Context

Inverse modeling framework that:
- Retrieves soil water parameters from remote sensing observations
- Uses OpenET algorithms for evapotranspiration estimation
- Leverages Google Earth Engine for satellite data access
- Employs PEST/pyemu for parameter estimation and uncertainty analysis

## Configuration: runoff_process

The `runoff_process` setting in the TOML `[misc]` section controls the soil water balance runoff method:

- **`cn`** (default): Curve Number runoff method. Uses daily precipitation from GridMET only.
- **`ier`**: Infiltration-Excess Runoff method. Requires hourly precipitation from NLDAS-2 via `pynldas2`.

The `pynldas2` dependency is only used when `runoff_process = "ier"`. All current examples use `runoff_process = "cn"`.

## Cautions

- Don't run Earth Engine scripts without confirmation (API quotas)
- Don't execute calibration/inverse runs without confirmation (computationally expensive)
- Don't `git add` .md files unless specifically requested (keep them untracked)
- **Never use flux tower data to configure model inputs** - flux data is for validation only, not for deriving parameters like groundwater subsidy (f_sub). Model inputs must come from remote sensing (ETf models, NDVI) and ancillary data (land cover, soils, meteorology)

## Git

- Commit messages should be very terse (one short line)
- Don't ever include a Co-Authored-By line
