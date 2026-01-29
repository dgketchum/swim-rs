# Test Fixtures

This directory contains test fixture data for SwimContainer regression tests.

## Structure

```
fixtures/
├── S2/                        # Single-station (Crane) fixture
│   ├── S2.toml               # Configuration file
│   ├── data/
│   │   └── gis/
│   │       └── flux_footprint_s2.shp   # Field shapefile
│   ├── golden/                # Reference outputs (generated)
│   │   ├── ke_max.json
│   │   ├── kc_max.json
│   │   ├── irr_data.json
│   │   ├── gwsub_data.json
│   │   └── prepped_input.json
│   └── input/                 # Input data
│       ├── ndvi/             # Landsat NDVI CSV exports
│       ├── etf/              # SSEBop ETf CSV exports
│       ├── met/              # GridMET parquet files
│       └── properties/       # Properties JSON
│
└── multi_station/             # Multi-station fixture
    ├── multi_station.toml    # Configuration file
    ├── data/
    │   └── gis/
    │       └── multi_station.shp       # Multi-field shapefile
    ├── golden/                # Reference outputs (generated)
    └── input/                 # Input data (same structure as S2)
```

## Setting Up Fixtures

### 1. S2 Single-Station Fixture

The S2 shapefile already exists. To set up the rest:

1. Copy NDVI CSV exports for S2 to `S2/input/ndvi/`
2. Copy ETf CSV exports for S2 to `S2/input/etf/`
3. Copy GridMET parquet files for S2 to `S2/input/met/`
4. Copy properties.json for S2 to `S2/input/properties/`

### 2. Multi-Station Fixture

For ALARC2_Smith6, MR, and US_FPe:

1. Create combined shapefile at `multi_station/data/gis/multi_station.shp`
2. Copy NDVI CSV exports to `multi_station/input/ndvi/`
3. Copy ETf CSV exports to `multi_station/input/etf/`
4. Copy GridMET parquet files to `multi_station/input/met/`
5. Copy properties.json to `multi_station/input/properties/`

### 3. Generate Golden Files

After input data is in place, run the golden file generator:

```bash
# For S2
python scripts/generate_golden_files.py s2 \
    --shapefile tests/fixtures/S2/data/gis/flux_footprint_s2.shp \
    --uid-column FID \
    --ndvi-dir tests/fixtures/S2/input/ndvi \
    --etf-dir tests/fixtures/S2/input/etf \
    --met-dir tests/fixtures/S2/input/met \
    --properties-json tests/fixtures/S2/input/properties/properties.json \
    --start-date 2020-01-01 \
    --end-date 2022-12-31 \
    --output-dir tests/fixtures/S2/golden

# For multi-station
python scripts/generate_golden_files.py multi \
    --shapefile tests/fixtures/multi_station/data/gis/multi_station.shp \
    --uid-column site_id \
    --ndvi-dir tests/fixtures/multi_station/input/ndvi \
    --etf-dir tests/fixtures/multi_station/input/etf \
    --met-dir tests/fixtures/multi_station/input/met \
    --properties-json tests/fixtures/multi_station/input/properties/properties.json \
    --start-date 2020-01-01 \
    --end-date 2022-12-31 \
    --output-dir tests/fixtures/multi_station/golden
```

## Running Tests

After fixtures are set up:

```bash
# Run all tests
pytest tests/test_container_single_station.py tests/test_container_multi_station.py -v

# Run only regression tests
pytest tests/ -m regression -v

# Skip slow tests
pytest tests/ -m "not slow" -v
```

## Updating Golden Files

When intentional changes are made to the algorithm:

1. Verify the changes are correct
2. Re-run the golden file generator
3. Commit the updated golden files with documentation

## File Size Guidelines

Target: < 50MB total for CI compatibility

- Date range: 2020-2022 (3 years)
- Remote sensing: Sparse observations (every 8-16 days)
- Compress CSVs if needed
