#!/usr/bin/env python
"""Generate golden-loop regression test fixtures.

For each site (Fort Peck, Crane), this script:
1. Creates a SwimContainer with a 2-year date range
2. Ingests source data (properties, gridmet, snodas, etf, ndvi)
3. Computes merged_ndvi and dynamics
4. Saves the container to tests/fixtures/golden_loop/{site}.swim/
5. Runs build_swim_input() + run_daily_loop_fast()
6. Saves output arrays to tests/fixtures/golden_loop/{site}_golden.npz

Usage
-----
    python scripts/generate_golden_loop.py
"""

import os
import sys
import tempfile

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from swimrs.container import SwimContainer
from swimrs.process.input import build_swim_input
from swimrs.process.loop_fast import run_daily_loop_fast

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures", "golden_loop")

OUTPUT_FIELDS = [
    "eta",
    "etf",
    "kcb",
    "ke",
    "ks",
    "kr",
    "runoff",
    "rain",
    "melt",
    "swe",
    "depl_root",
    "dperc",
    "irr_sim",
    "gw_sim",
]

CASES = [
    {
        "name": "fort_peck",
        "example_dir": os.path.join(PROJECT_ROOT, "examples", "2_Fort_Peck"),
        "project_name": "2_Fort_Peck",
        "shapefile": os.path.join("data", "gis", "flux_fields.shp"),
        "uid_column": "site_id",
        "start_date": "2007-01-01",
        "end_date": "2008-12-31",
        "etf_model": "ptjpl",
        "etf_dir_name": "ptjpl_etf",
        "select_fields": ["US-FPe"],
    },
    {
        "name": "crane",
        "example_dir": os.path.join(PROJECT_ROOT, "examples", "3_Crane"),
        "project_name": "3_Crane",
        "shapefile": os.path.join("data", "gis", "flux_fields.shp"),
        "uid_column": "site_id",
        "start_date": "2020-01-01",
        "end_date": "2021-12-31",
        "etf_model": "ssebop",
        "etf_dir_name": "ssebop_etf",
        "select_fields": ["S2"],
    },
]


def generate_case(case):
    """Build fixture container and golden arrays for one site."""
    name = case["name"]
    example_dir = case["example_dir"]
    data_dir = os.path.join(example_dir, "data")
    uid_col = case["uid_column"]
    select_fields = case["select_fields"]

    container_path = os.path.join(OUTPUT_DIR, f"{name}.swim")
    golden_path = os.path.join(OUTPUT_DIR, f"{name}_golden.npz")

    print(f"\n{'=' * 60}")
    print(f"Generating: {name}")
    print(f"{'=' * 60}")

    # --- Create container ---
    shapefile = os.path.join(example_dir, case["shapefile"])
    print(f"  Shapefile: {shapefile}")
    print(f"  Date range: {case['start_date']} to {case['end_date']}")

    container = SwimContainer.create(
        container_path,
        fields_shapefile=shapefile,
        uid_column=uid_col,
        start_date=case["start_date"],
        end_date=case["end_date"],
        project_name=case["project_name"],
        overwrite=True,
    )

    try:
        # --- Ingest properties ---
        props_dir = os.path.join(data_dir, "properties")
        pname = case["project_name"]
        container.ingest.properties(
            lulc_csv=os.path.join(props_dir, f"{pname}_landcover.csv"),
            soils_csv=os.path.join(props_dir, f"{pname}_ssurgo.csv"),
            irr_csv=os.path.join(props_dir, f"{pname}_irr.csv"),
            uid_column=uid_col,
            overwrite=True,
        )

        # --- Ingest GridMET ---
        if name == "fort_peck":
            met_dir = os.path.join(data_dir, "met_timeseries", "gridmet")
        else:
            met_dir = os.path.join(data_dir, "met")
        grid_shp = os.path.join(data_dir, "gis", "flux_fields_gfid.shp")

        container.ingest.gridmet(
            source_dir=met_dir,
            grid_shapefile=grid_shp,
            uid_column=uid_col,
            grid_column="GFID",
            include_corrected=True,
            overwrite=True,
        )

        # --- Ingest SNODAS ---
        snodas_dir = os.path.join(data_dir, "snow", "snodas", "extracts")
        container.ingest.snodas(
            source_dir=snodas_dir,
            uid_column=uid_col,
            fields=select_fields,
            overwrite=True,
        )

        # --- Ingest ETf ---
        etf_base = os.path.join(
            data_dir, "remote_sensing", "landsat", "extracts", case["etf_dir_name"]
        )
        for mask in ["irr", "inv_irr"]:
            mask_dir = os.path.join(etf_base, mask)
            if os.path.exists(mask_dir):
                container.ingest.etf(
                    source_dir=mask_dir,
                    uid_column=uid_col,
                    model=case["etf_model"],
                    mask=mask,
                    instrument="landsat",
                    fields=select_fields,
                    overwrite=True,
                )

        # --- Ingest NDVI ---
        ndvi_base = os.path.join(data_dir, "remote_sensing", "landsat", "extracts", "ndvi")
        for mask in ["irr", "inv_irr"]:
            mask_dir = os.path.join(ndvi_base, mask)
            if os.path.exists(mask_dir):
                container.ingest.ndvi(
                    source_dir=mask_dir,
                    uid_column=uid_col,
                    instrument="landsat",
                    mask=mask,
                    fields=select_fields,
                    overwrite=True,
                )

        # --- Compute merged NDVI and dynamics ---
        container.compute.merged_ndvi(
            masks=("irr", "inv_irr"),
            instruments=("landsat",),
            overwrite=True,
        )
        container.compute.dynamics(
            etf_model=case["etf_model"],
            masks=("irr", "inv_irr"),
            instruments=("landsat",),
            use_mask=True,
            use_lulc=False,
            fields=select_fields,
            overwrite=True,
        )

        container.save()

        # --- Run simulation ---
        print("  Running simulation...")
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "swim_input.h5")
            swim_input = build_swim_input(
                container,
                h5_path,
                start_date=case["start_date"],
                end_date=case["end_date"],
                etf_model=case["etf_model"],
            )
            try:
                output, _ = run_daily_loop_fast(swim_input)
            finally:
                swim_input.close()

        # --- Save golden arrays ---
        arrays = {field: getattr(output, field) for field in OUTPUT_FIELDS}
        np.savez_compressed(golden_path, **arrays)
        print(f"  Saved golden arrays to {golden_path}")

        # Print sizes
        container_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(container_path)
            for f in fns
        )
        golden_size = os.path.getsize(golden_path)
        print(f"  Container size: {container_size / 1024:.0f} KB")
        print(f"  Golden NPZ size: {golden_size / 1024:.0f} KB")

    finally:
        container.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for case in CASES:
        generate_case(case)

    print(f"\nDone! Fixtures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
