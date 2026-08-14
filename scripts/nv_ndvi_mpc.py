# MPC (Planetary Computer) NDVI extraction — EE-free replacement for
# nv_ndvi.py. Same partitioning (FIPS + a/b/c... chunks of <=900 fields),
# same feature ID (OPENET_ID), same CSV contract, writes
# {out_dir}/{label}/ndvi/{mask}/ndvi_{mask}_{year}.csv (Landsat) or
# {label}/ndvi_s2/{mask}/ndvi_s2_{mask}_{year}.csv (Sentinel-2), with
# optional gsutil upload into gs://wudr/nv/. Requires local IrrMapper
# annual rasters (--masks-dir) for irr/inv_irr.

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd

from swimrs.data_extraction.mpc import export, runner, stac

FEATURE_ID = "OPENET_ID"
SHAPEFILE = (
    "/nas/Nevada/dri_field_pts/fields_gis/Nevada_Agricultural_Field_Boundaries_20250214/"
    "Nevada_Agricultural_Field_Boundaries_20250214_5071_GFID.shp"
)
RECORDS_DIR = "/data/ssd1/swim/nv/mpc_records"
OUT_DIR = "/data/ssd1/swim/nv/mpc_csv"


def main():
    parser = argparse.ArgumentParser(description="Nevada NDVI extraction via Planetary Computer")
    parser.add_argument("--shapefile", default=SHAPEFILE)
    parser.add_argument("--feature-id", default=FEATURE_ID)
    parser.add_argument("--fips", default=None, help="Comma-separated FIPS codes")
    parser.add_argument(
        "--partitions", default=None, help="Comma-separated partition labels; overrides --fips"
    )
    parser.add_argument("--start-yr", type=int, default=1995)
    parser.add_argument("--end-yr", type=int, default=2025)
    parser.add_argument("--years", default=None, help="Comma-separated years (overrides start/end)")
    parser.add_argument("--mask-types", default="irr,inv_irr")
    parser.add_argument("--instrument", choices=["landsat", "sentinel"], default="landsat")
    parser.add_argument("--masks-dir", default=None, help="Dir of annual IrrMapper GeoTIFFs")
    parser.add_argument(
        "--irrigated-value", type=int, default=0, help="Raster value meaning irrigated"
    )
    parser.add_argument("--records-dir", default=RECORDS_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--bucket", default="wudr")
    parser.add_argument("--file-prefix", default="nv")
    parser.add_argument(
        "--fill-gaps",
        action="store_true",
        help="Skip (partition, mask, year) files already in the bucket",
    )
    parser.add_argument("--upload", action="store_true", help="gsutil cp assembled CSVs to bucket")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--assemble-only", action="store_true", help="Skip extraction; pivot existing records"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mask_types = [m.strip() for m in args.mask_types.split(",")]
    if args.masks_dir is None and set(mask_types) - {"no_mask"}:
        sys.exit("--masks-dir is required for irr/inv_irr masks")

    gdf = gpd.read_file(args.shapefile, engine="fiona")
    dupes = gdf.loc[gdf[args.feature_id].duplicated(keep=False), args.feature_id].tolist()
    if dupes:
        print(f"WARNING: duplicated {args.feature_id} values (kept as-is): {sorted(set(dupes))}")
    gdf["_row"] = range(len(gdf))

    partitions = export.partition_fields(gdf, args.feature_id)
    if args.partitions:
        selected = {p.strip() for p in args.partitions.split(",")}
        partitions = [(lbl, fids) for lbl, fids in partitions if lbl in selected]
    elif args.fips:
        selected = {f.strip() for f in args.fips.split(",")}
        partitions = [
            (lbl, fids) for lbl, fids in partitions if lbl.rstrip(export.CHUNK_SUFFIXES) in selected
        ]

    years = (
        [int(y) for y in args.years.split(",")]
        if args.years
        else list(range(args.start_yr, args.end_yr + 1))
    )

    existing = (
        export.list_bucket_files(args.bucket, args.file_prefix, args.instrument)
        if args.fill_gaps
        else set()
    )
    missing, per_year = export.build_targets(partitions, mask_types, years, existing)
    print(
        f"{len(partitions)} partitions x {len(mask_types)} masks x {len(years)} years: "
        f"{len(missing)} files to produce ({len(existing)} already in bucket)"
    )
    if args.dry_run:
        for label, fids in partitions:
            need = sorted({(m, y) for lbl, m, y in missing if lbl == label})
            print(f"  {label}: {len(fids)} fields, {len(need)} files")
        sys.exit(0)
    if not missing:
        print("nothing to do")
        sys.exit(0)

    fields_4326 = gdf[[args.feature_id, "_row", "geometry"]].to_crs(4326)
    config = {
        "feature_id": args.feature_id,
        "masks_dir": args.masks_dir,
        "irrigated_value": args.irrigated_value,
    }
    runner.write_worker_inputs(args.records_dir, fields_4326, per_year, config)

    if not args.assemble_only:
        extract(args, per_year, fields_4326)

    assemble_and_upload(args, missing, partitions, gdf)


def extract(args, per_year, fields_4326):
    catalog = stac.open_catalog()
    search = stac.search_landsat if args.instrument == "landsat" else stac.search_sentinel2
    tasks = []
    for year in sorted(per_year):
        ids = set(per_year[year]["fields"])
        bbox = fields_4326[fields_4326[args.feature_id].isin(ids)].total_bounds
        items = search(catalog, bbox, year)
        fresh = 0
        for item in items:
            scene_id = (
                stac.landsat_scene_id(item)
                if args.instrument == "landsat"
                else stac.sentinel_scene_id(item)
            )
            if not runner.records_path(args.records_dir, args.instrument, year, scene_id).exists():
                tasks.append({"item": item, "instrument": args.instrument, "year": year})
                fresh += 1
        print(f"  {year}: {len(items)} scenes, {fresh} to extract")
    if not tasks:
        print("all scene records present")
        return

    t0 = time.time()
    done = failed = 0
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=runner.init_worker, initargs=(args.records_dir,)
    ) as pool:
        futures = {pool.submit(runner.process_scene, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                scene_id, n_fields, status = future.result()
                done += 1
            except Exception as exc:
                task = futures[future]
                print(f"  FAILED {task['year']} {task['item'].get('id')}: {exc}", flush=True)
                failed += 1
                continue
            if done % 50 == 0 or done == len(tasks):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(tasks)} scenes ({rate:.1f}/s), {failed} failed", flush=True)
    if failed:
        print(f"WARNING: {failed} scene(s) failed — rerun to retry (records are idempotent)")


def assemble_and_upload(args, missing, partitions, gdf):
    rows_by_label = {
        label: [
            (row, fid) for row, fid in zip(gdf["_row"], gdf[args.feature_id]) if fid in set(fids)
        ]
        for label, fids in partitions
    }
    n = 0
    for label, mask, year in sorted(missing):
        out_path = export.csv_path(args.out_dir, args.instrument, label, mask, year)
        export.assemble_csv(
            args.records_dir,
            args.instrument,
            year,
            mask,
            rows_by_label[label],
            args.feature_id,
            out_path,
        )
        n += 1
        if args.upload:
            subdir, stem = export.LAYOUT[args.instrument]
            dest = (
                f"gs://{args.bucket}/{args.file_prefix}/{label}/{subdir}/{mask}/"
                f"{stem}_{mask}_{year}.csv"
            )
            subprocess.run(["gsutil", "-q", "cp", str(out_path), dest], check=True)
    print(f"assembled {n} CSVs under {args.out_dir}" + (" and uploaded" if args.upload else ""))


if __name__ == "__main__":
    main()

# ========================= EOF ====================================================================
