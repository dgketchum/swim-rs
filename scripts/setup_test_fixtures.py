#!/usr/bin/env python3
"""
Set up test fixture data by copying required files from /data/ssd2/swim/.

This script copies a subset of data for testing:
- S2 station for single-station tests
- ALARC2_Smith6, MR, US-FPe for multi-station tests

Usage:
    python scripts/setup_test_fixtures.py

The script will create the following structure:
    tests/fixtures/S2/
        data/gis/flux_footprint_s2.shp  (extracted from larger shapefile)
        input/ndvi/                     (2020-2022 files)
        input/etf/
        input/met/
    tests/fixtures/multi_station/
        data/gis/multi_station.shp      (3 stations)
        input/ndvi/
        input/etf/
        input/met/
"""

import shutil
import subprocess
from pathlib import Path

# Source data paths
DATA_ROOT = Path("/data/ssd2/swim/5_Flux_Ensemble/data")
FLUX_NETWORK_DATA = Path("/data/ssd2/swim/4_Flux_Network/data")

# Target paths
FIXTURES_ROOT = Path("/home/dgketchum/code/swim-rs/tests/fixtures")

# Date range for test data
YEARS = ["2020", "2021", "2022"]

# Station configurations
S2_CONFIG = {
    "station_id": "S2",
    "shapefile_fid": "S2",
}

# Note: ALARC2_Smith6 is truncated to ALARC2_Smi in shapefile due to DBF 10-char limit
MULTI_STATIONS = {
    "ALARC2_Smith6": {"shapefile_fid": "ALARC2_Smi"},  # Truncated in shapefile
    "MR": {"shapefile_fid": "MR"},
}

# US-FPe is in 4_Flux_Network, not 5_Flux_Ensemble
US_FPE_CONFIG = {
    "station_id": "US-FPe",
    "data_root": FLUX_NETWORK_DATA,
}


def setup_s2_fixture():
    """Set up S2 single-station fixture."""
    print("Setting up S2 fixture...")

    s2_root = FIXTURES_ROOT / "S2"
    gis_dir = s2_root / "data" / "gis"
    ndvi_dir = s2_root / "input" / "ndvi"
    etf_dir = s2_root / "input" / "etf"
    met_dir = s2_root / "input" / "met"

    # Create directories
    for d in [gis_dir, ndvi_dir, etf_dir, met_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Extract S2 from shapefile using ogr2ogr
    src_shp = DATA_ROOT / "gis" / "flux_fields.shp"
    dst_shp = gis_dir / "flux_footprint_s2.shp"

    if not dst_shp.exists():
        cmd = ["ogr2ogr", "-where", "FID = 'S2'", str(dst_shp), str(src_shp)]
        print("  Extracting S2 from shapefile...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
        else:
            print(f"  Created {dst_shp}")

    # Copy NDVI files
    src_ndvi = DATA_ROOT / "landsat" / "extracts" / "ndvi" / "irr"
    for year in YEARS:
        src_file = src_ndvi / f"ndvi_S2_irr_{year}.csv"
        if src_file.exists():
            dst_file = ndvi_dir / src_file.name
            if not dst_file.exists():
                shutil.copy(src_file, dst_file)
                print(f"  Copied {src_file.name}")

    # Copy ETf files
    src_etf = DATA_ROOT / "landsat" / "extracts" / "ssebop_etf" / "irr"
    for year in YEARS:
        src_file = src_etf / f"ssebop_etf_S2_irr_{year}.csv"
        if src_file.exists():
            dst_file = etf_dir / src_file.name
            if not dst_file.exists():
                shutil.copy(src_file, dst_file)
                print(f"  Copied {src_file.name}")

    # Copy meteorology - need to find the GFID for S2
    # For now, we'll skip this as it requires knowing the GFID mapping
    print("  Note: Meteorology data requires GFID mapping - skipping for now")

    print("S2 fixture setup complete.")


def setup_multi_station_fixture():
    """Set up multi-station fixture (ALARC2_Smith6, MR, US-FPe)."""
    print("Setting up multi-station fixture...")

    multi_root = FIXTURES_ROOT / "multi_station"
    gis_dir = multi_root / "data" / "gis"
    ndvi_dir = multi_root / "input" / "ndvi"
    etf_dir = multi_root / "input" / "etf"
    met_dir = multi_root / "input" / "met"

    # Create directories
    for d in [gis_dir, ndvi_dir, etf_dir, met_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Extract stations from shapefile
    src_shp = DATA_ROOT / "gis" / "flux_fields.shp"
    dst_shp = gis_dir / "multi_station.shp"

    # Build WHERE clause for multiple stations
    fids = ["S2", "MR", "ALARC2_Smi"]  # Using truncated name
    where_clause = " OR ".join([f"FID = '{fid}'" for fid in fids])

    if not dst_shp.exists():
        cmd = ["ogr2ogr", "-where", where_clause, str(dst_shp), str(src_shp)]
        print("  Extracting stations from shapefile...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
        else:
            print(f"  Created {dst_shp}")

    # Copy NDVI and ETf files for each station
    stations = ["S2", "MR", "ALARC2_Smith6"]

    src_ndvi = DATA_ROOT / "landsat" / "extracts" / "ndvi" / "irr"
    src_etf = DATA_ROOT / "landsat" / "extracts" / "ssebop_etf" / "irr"

    for station in stations:
        for year in YEARS:
            # NDVI
            src_file = src_ndvi / f"ndvi_{station}_irr_{year}.csv"
            if src_file.exists():
                dst_file = ndvi_dir / src_file.name
                if not dst_file.exists():
                    shutil.copy(src_file, dst_file)
                    print(f"  Copied {src_file.name}")

            # ETf
            src_file = src_etf / f"ssebop_etf_{station}_irr_{year}.csv"
            if src_file.exists():
                dst_file = etf_dir / src_file.name
                if not dst_file.exists():
                    shutil.copy(src_file, dst_file)
                    print(f"  Copied {src_file.name}")

    print("Multi-station fixture setup complete.")


def check_fixture_sizes():
    """Report total fixture sizes."""
    print("\nFixture sizes:")

    for fixture in ["S2", "multi_station"]:
        fixture_path = FIXTURES_ROOT / fixture
        if fixture_path.exists():
            total_size = sum(f.stat().st_size for f in fixture_path.rglob("*") if f.is_file())
            print(f"  {fixture}: {total_size / 1024 / 1024:.2f} MB")


def main():
    print("Setting up test fixtures...")
    print(f"Source: {DATA_ROOT}")
    print(f"Target: {FIXTURES_ROOT}")
    print(f"Years: {YEARS}")
    print()

    setup_s2_fixture()
    print()
    setup_multi_station_fixture()
    print()
    check_fixture_sizes()

    print("\nNext steps:")
    print("1. Set up meteorology data (requires GFID mapping)")
    print("2. Run generate_golden_files.py to create reference outputs")
    print("3. Run pytest to verify tests work")


if __name__ == "__main__":
    main()
