"""SSEBop NHM ETf extraction from the USGS Earth Engine asset.

Extracts ET fraction zonal statistics from the pre-computed USGS NHM SSEBop
Landsat Collection 2 asset.  Supports irrigation masking (IrrMapper for
western states, LANID for eastern states) and a ``no_mask`` mode that clips
to the full site footprint.

Usage:
    python ssebop_etf.py --shapefile <path> --mask {irr,inv_irr,no_mask} \\
        [--sites SITE1,SITE2] [--bucket BUCKET] [--start-yr 2000] [--end-yr 2024]
"""

import os

import ee
from tqdm import tqdm

from swimrs.data_extraction.ee.common import (
    build_feature_collection,
    clip_and_apply_irrigation_mask,
    export_table,
    get_irrigation_mask,
    load_shapefile,
    parse_scene_name,
    setup_irrigation_masks,
)

NHM_SSEBOP = "projects/usgs-gee-nhm-ssebop/assets/ssebop/landsat/c02"


def extract_ssebop_nhm_etf(
    shapefile,
    mask_type="no_mask",
    check_dir=None,
    feature_id="FID",
    select=None,
    start_yr=2000,
    end_yr=2024,
    state_col="state",
    dest="bucket",
    bucket=None,
    file_prefix="swim",
):
    """Export per-field SSEBop NHM ETf zonal statistics.

    Iterates features row-by-row from a local shapefile, discovers scenes
    from the NHM SSEBop asset, applies irrigation masking, and exports
    mean ETf via ``reduceRegions`` at 30 m.

    Parameters
    ----------
    shapefile : str
        Path to polygon shapefile with feature IDs.
    mask_type : str
        Irrigation masking: 'no_mask', 'irr', or 'inv_irr'.
    check_dir : str, optional
        Skip exports if CSV already exists at ``<check_dir>/<desc>.csv``.
    feature_id : str
        Column name for feature identifier.
    select : list[str], optional
        Subset of feature IDs to process.
    start_yr, end_yr : int
        Inclusive year range.
    state_col : str
        Column with state abbreviation for mask source selection.
    dest : str
        Export destination: 'drive' or 'bucket'.
    bucket : str, optional
        GCS bucket name (required if ``dest='bucket'``).
    file_prefix : str
        Bucket path prefix.
    """
    if dest == "bucket" and not bucket:
        raise ValueError("bucket is required when dest='bucket'")

    df = load_shapefile(shapefile, feature_id)

    if select is not None:
        df = df[df.index.isin(select)]

    print(f"SSEBop NHM ETf: {len(df)} fields, mask={mask_type}, {start_yr}-{end_yr}")

    irr_coll, irr_min_yr_mask, lanid, east_fc = setup_irrigation_masks()

    skipped, exported = 0, 0

    for fid, row in tqdm(df.iterrows(), desc="SSEBop NHM ETf", total=len(df)):
        if row["geometry"].geom_type not in ("Polygon", "MultiPolygon"):
            continue

        polygon = ee.Geometry(row.geometry.__geo_interface__)

        for year in range(start_yr, end_yr + 1):
            # Get irrigation mask if needed
            if mask_type in ("irr", "inv_irr"):
                state = row.get(state_col, None) if state_col in row.index else None
                irr, irr_mask = get_irrigation_mask(
                    year, state, irr_coll, irr_min_yr_mask, lanid, east_fc
                )
            else:
                irr, irr_mask = None, None

            # Discover scenes for this field-year
            etf_coll = (
                ee.ImageCollection(NHM_SSEBOP)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .filterBounds(polygon)
            )
            etf_scenes = etf_coll.aggregate_histogram("system:index").getInfo()

            if not etf_scenes:
                continue

            scene_ids = sorted(etf_scenes.keys(), key=lambda s: s.split("_")[-1])

            desc = f"ssebop_etf_{fid}_{mask_type}_{year}"

            if check_dir:
                check_path = os.path.join(check_dir, f"{desc}.csv")
                if os.path.exists(check_path):
                    skipped += 1
                    continue

            fn_prefix = (
                f"{file_prefix}/remote_sensing/landsat/extracts/ssebop_etf/{mask_type}/{desc}"
            )

            first, bands = True, None
            selectors = [feature_id]

            for img_id in scene_ids:
                _name = parse_scene_name(img_id)
                selectors.append(_name)

                etf_img = (
                    ee.Image(f"{NHM_SSEBOP}/{img_id}")
                    .select("et_fraction")
                    .divide(10000)
                    .rename(_name)
                )

                etf_img = clip_and_apply_irrigation_mask(
                    etf_img,
                    polygon,
                    mask_type,
                    irr=irr,
                    irr_mask=irr_mask,
                )

                if first:
                    bands = etf_img
                    first = False
                else:
                    bands = bands.addBands([etf_img])

            if bands is None:
                continue

            fc = build_feature_collection(polygon, fid, feature_id)
            data = bands.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)

            export_table(
                data=data,
                desc=desc,
                selectors=selectors,
                dest=dest,
                bucket=bucket,
                fn_prefix=fn_prefix,
            )
            exported += 1

    print(f"SSEBop NHM ETf: exported {exported}, skipped {skipped}")


if __name__ == "__main__":
    import argparse

    from swimrs.data_extraction.ee.ee_utils import is_authorized

    parser = argparse.ArgumentParser(description="Extract SSEBop NHM ETf")
    parser.add_argument("--shapefile", required=True, help="Path to polygon shapefile")
    parser.add_argument(
        "--mask",
        choices=["irr", "inv_irr", "no_mask"],
        default="no_mask",
        help="Mask type (default: no_mask)",
    )
    parser.add_argument("--sites", type=str, default=None, help="Comma-separated site IDs")
    parser.add_argument("--start-yr", type=int, default=2000)
    parser.add_argument("--end-yr", type=int, default=2025)
    parser.add_argument("--feature-id", type=str, default="FID")
    parser.add_argument("--state-col", type=str, default="state")
    parser.add_argument("--bucket", type=str, default=None)
    parser.add_argument("--dest", choices=["drive", "bucket"], default="bucket")
    parser.add_argument("--file-prefix", type=str, default="swim")
    args = parser.parse_args()

    is_authorized()

    sites = [s.strip() for s in args.sites.split(",")] if args.sites else None

    extract_ssebop_nhm_etf(
        shapefile=args.shapefile,
        mask_type=args.mask,
        feature_id=args.feature_id,
        select=sites,
        start_yr=args.start_yr,
        end_yr=args.end_yr,
        state_col=args.state_col,
        dest=args.dest,
        bucket=args.bucket,
        file_prefix=args.file_prefix,
    )
