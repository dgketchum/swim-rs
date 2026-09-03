"""Download and verify GridMET meteorology parquets for NWI extraction units.

One parquet per unique GridMET cell ({GFID}.parquet) is written to a shared
store so cells straddling partition boundaries (e.g. 32019a/b/c) are fetched
once and reused statewide.

Why this script exists rather than calling ``download_gridmet`` directly:

* ``download_gridmet`` looks up cell coordinates with
  ``fields[gridmet_id_col] == int(g_fid)``, which silently no-matches against
  the DRI-supplied ``GRIDMET_ID`` column (STRING dtype) in nwi_fields.shp.
* It also needs ``LAT``/``LON`` columns, which nwi_fields.shp does not carry.

So we build our own mapping input: GFID is derived from each field centroid
with the south-origin gridMET formula

    lat = 25.066667 + (GFID // 1386) / 24
    lon = -124.766667 + (GFID % 1386) / 24

and LAT/LON are then regenerated from that formula so the downloader queries
exact cell centers rather than field centroids. The derived GFIDs are
cross-checked against ``int(GRIDMET_ID)`` and disagreements are reported.

THREDDS ``#fillmismatch`` failures are swallowed by the downloader (per-variable
``OSError`` and a per-cell bare ``except``), so nothing raises when a cell comes
back short or holed. The completeness sweep is therefore mandatory, and failing
parquets are deleted and re-downloaded rather than patched.

Usage:
    uv run python scripts/nwi_gridmet.py --fips 32019 \
        --dest /project/handily/swim/data/nwi_gridmet \
        --start 1985-01-01 --end 2025-12-31 --workers 6
"""

import argparse
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box as shapely_box

# gridMET CONUS grid geometry (south-origin, 1/24 degree, 1386 columns).
GRIDMET_LAT0 = 25.066667
GRIDMET_LON0 = -124.766667
GRIDMET_RES = 1.0 / 24.0
GRIDMET_NCOLS = 1386

DEFAULT_SHAPEFILE = "/project/handily/swim/data/nwi_gis/nwi_fields.shp"
DEFAULT_DEST = "/project/handily/swim/data/nwi_gridmet"
DEFAULT_DATA_ROOT = "/project/handily/swim/data"
FEATURE_ID = "NWI_ID"

# Column set written by download_gridmet when gridmet_factors is None
# (no eto_corr/etr_corr); matches the 32009 pilot parquets.
EXPECTED_COLS = ["tmin", "tmax", "eto", "etr", "prcp", "srad", "u2", "ea", "elev"]


def gfid_to_latlon(gfid):
    """Cell-center lat/lon for a gridMET cell id (vectorized)."""
    gfid = np.asarray(gfid, dtype=np.int64)
    lat = GRIDMET_LAT0 + (gfid // GRIDMET_NCOLS) * GRIDMET_RES
    lon = GRIDMET_LON0 + (gfid % GRIDMET_NCOLS) * GRIDMET_RES
    return lat, lon


def latlon_to_gfid(lat, lon):
    """Nearest gridMET cell id for point coordinates (vectorized)."""
    row = np.rint((np.asarray(lat) - GRIDMET_LAT0) / GRIDMET_RES).astype(np.int64)
    col = np.rint((np.asarray(lon) - GRIDMET_LON0) / GRIDMET_RES).astype(np.int64)
    return row * GRIDMET_NCOLS + col


def select_fields(shapefile, fips=None, partitions=None, data_root=DEFAULT_DATA_ROOT):
    """Subset the statewide fields shapefile by FIPS code(s) or partition label(s).

    With neither filter, every field in the shapefile is returned — the
    statewide case.
    """
    gdf = gpd.read_file(shapefile, engine="fiona")
    if not fips and not partitions:
        return gdf
    if fips:
        codes = {str(f).strip() for f in fips}
        sub = gdf[gdf["FIPS"].astype(str).isin(codes)].copy()
        if sub.empty:
            raise SystemExit(f"no fields for FIPS {sorted(codes)} in {shapefile}")
        return sub

    uids = set()
    for label in partitions:
        prop_dir = Path(data_root) / label / "properties"
        found = sorted(prop_dir.glob("*.csv"))
        if not found:
            raise SystemExit(f"no properties CSV under {prop_dir} to resolve partition {label}")
        for csv in found:
            uids.update(pd.read_csv(csv, usecols=[FEATURE_ID])[FEATURE_ID].astype(str))
    sub = gdf[gdf[FEATURE_ID].astype(str).isin(uids)].copy()
    if sub.empty:
        raise SystemExit(f"no fields matched partitions {partitions}")
    return sub


def build_mapping(fields):
    """Return (per-field frame, unique-cell frame) and print the DRI cross-check."""
    wgs = fields.geometry.centroid.to_crs("EPSG:4326")
    lat_c, lon_c = wgs.y.values, wgs.x.values
    gfid = latlon_to_gfid(lat_c, lon_c)

    per_field = pd.DataFrame(
        {
            FEATURE_ID: fields[FEATURE_ID].astype(str).values,
            "GFID": gfid,
            "CENT_LAT": lat_c,
            "CENT_LON": lon_c,
        }
    )

    if "GRIDMET_ID" in fields.columns:
        dri = pd.to_numeric(fields["GRIDMET_ID"], errors="coerce").values
        per_field["DRI_GFID"] = dri
        ok = np.isfinite(dri)
        agree = int((per_field.loc[ok, "GFID"].values == dri[ok].astype(np.int64)).sum())
        n_ok = int(ok.sum())
        rate = 100.0 * agree / n_ok if n_ok else float("nan")
        print(f"GFID cross-check: {agree}/{n_ok} agree with DRI GRIDMET_ID ({rate:.2f}%)")
        if agree != n_ok:
            bad = per_field.loc[ok].copy()
            bad = bad[bad["GFID"].values != dri[ok].astype(np.int64)]
            print(f"  {len(bad)} mismatches; first 20:")
            print(bad.head(20).to_string(index=False))
        if n_ok != len(per_field):
            print(f"  WARNING: {len(per_field) - n_ok} fields have unparseable GRIDMET_ID")

    cells = pd.DataFrame({"GFID": np.unique(per_field["GFID"].values)})
    cells["LAT"], cells["LON"] = gfid_to_latlon(cells["GFID"].values)
    print(f"{len(per_field)} fields -> {len(cells)} unique GridMET cells")
    return per_field, cells


def intersecting_cells(fields):
    """Every gridMET cell whose 4 km footprint intersects a field polygon.

    ``build_mapping`` assigns one cell per field from its centroid, which is
    what the model uses for forcing. This is the superset: a 4 km cell is
    coarse enough that a field near a cell edge overlaps neighbours the
    centroid rule never names. Fetching the superset costs little and makes
    the store robust to a future change in the field-to-cell rule (e.g.
    area-weighted rather than centroid).
    """
    wgs = fields.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = wgs.total_bounds
    # One cell of slack each way so an edge-touching field cannot fall outside
    # the candidate block.
    c0 = int(np.floor((minx - GRIDMET_LON0) / GRIDMET_RES)) - 1
    c1 = int(np.ceil((maxx - GRIDMET_LON0) / GRIDMET_RES)) + 1
    r0 = int(np.floor((miny - GRIDMET_LAT0) / GRIDMET_RES)) - 1
    r1 = int(np.ceil((maxy - GRIDMET_LAT0) / GRIDMET_RES)) + 1

    half = GRIDMET_RES / 2.0
    gfids, boxes = [], []
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            lon = GRIDMET_LON0 + col * GRIDMET_RES
            lat = GRIDMET_LAT0 + row * GRIDMET_RES
            gfids.append(row * GRIDMET_NCOLS + col)
            boxes.append(shapely_box(lon - half, lat - half, lon + half, lat + half))

    grid = gpd.GeoDataFrame({"GFID": gfids}, geometry=boxes, crs="EPSG:4326")
    hit = gpd.sjoin(grid, wgs[["geometry"]], predicate="intersects", how="inner")
    cells = pd.DataFrame({"GFID": np.unique(hit["GFID"].to_numpy())})
    cells["LAT"], cells["LON"] = gfid_to_latlon(cells["GFID"].values)
    print(f"{len(grid)} candidate cells over the field bbox -> {len(cells)} intersect a field")
    return cells


def write_cell_input(cells, path):
    """Write the downloader's input shapefile: GFID (int64) + exact cell-center LAT/LON."""
    gdf = gpd.GeoDataFrame(
        {
            "GFID": cells["GFID"].astype("int64"),
            "LAT": cells["LAT"].astype(float),
            "LON": cells["LON"].astype(float),
        },
        geometry=gpd.points_from_xy(cells["LON"], cells["LAT"]),
        crs="EPSG:4326",
    )
    gdf.to_file(path, engine="fiona")
    return path


def copy_existing(gfids, dest, sources):
    """Copy already-downloaded parquets from precedent stores instead of refetching."""
    copied = []
    for gfid in gfids:
        target = Path(dest) / f"{gfid}.parquet"
        if target.exists():
            continue
        for src_dir in sources:
            src = Path(src_dir) / f"{gfid}.parquet"
            if src.exists():
                shutil.copy2(src, target)
                copied.append(int(gfid))
                break
    return copied


def _download_chunk(shp_path, dest, start, end, chunk):
    from swimrs.data_extraction.gridmet.gridmet import download_gridmet

    download_gridmet(
        shp_path,
        None,
        dest,
        start=start,
        end=end,
        overwrite=False,
        target_fields=[str(g) for g in chunk],
        feature_id="GFID",
        gridmet_id_col="GFID",
    )
    return len(chunk)


def download_cells(shp_path, dest, start, end, gfids, workers):
    """Fetch the listed cells, fanned out over ``workers`` processes."""
    gfids = list(gfids)
    if not gfids:
        return
    workers = max(1, min(workers, len(gfids)))
    chunks = [list(c) for c in np.array_split(np.array(gfids), workers) if len(c)]
    if workers == 1:
        _download_chunk(shp_path, dest, start, end, chunks[0])
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_chunk, shp_path, dest, start, end, c) for c in chunks]
        for fut in as_completed(futures):
            fut.result()


def sweep(gfids, dest, start, end):
    """Per-cell completeness check against the expected daily POR and column set."""
    expected_index = pd.date_range(start, end, freq="D")
    n_expected = len(expected_index)
    rows = []
    for gfid in gfids:
        path = Path(dest) / f"{gfid}.parquet"
        rec = {
            "GFID": int(gfid),
            "exists": path.exists(),
            "n_rows": 0,
            "n_expected": n_expected,
            "first_date": "",
            "last_date": "",
            "missing_dates": n_expected,
            "n_cols": 0,
            "missing_cols": ",".join(EXPECTED_COLS),
            "extra_cols": "",
            "n_nan": -1,
            "bytes": 0,
            "status": "MISSING",
            "reason": "file absent",
        }
        if not path.exists():
            rows.append(rec)
            continue
        rec["bytes"] = path.stat().st_size
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - corrupt file is a sweep failure, not a crash
            rec["status"] = "FAIL"
            rec["reason"] = f"unreadable: {exc}"
            rows.append(rec)
            continue

        idx = pd.DatetimeIndex(df.index)
        missing = expected_index.difference(idx)
        rec["n_rows"] = len(df)
        rec["first_date"] = str(idx.min().date()) if len(idx) else ""
        rec["last_date"] = str(idx.max().date()) if len(idx) else ""
        rec["missing_dates"] = len(missing)
        rec["n_cols"] = df.shape[1]
        rec["missing_cols"] = ",".join(c for c in EXPECTED_COLS if c not in df.columns)
        rec["extra_cols"] = ",".join(c for c in df.columns if c not in EXPECTED_COLS)
        rec["n_nan"] = int(df.isna().to_numpy().sum())

        reasons = []
        if len(df) != n_expected:
            reasons.append(f"rows {len(df)} != {n_expected}")
        if len(missing):
            reasons.append(f"{len(missing)} missing dates (first {missing[0].date()})")
        if idx.has_duplicates:
            reasons.append("duplicate dates")
        if rec["missing_cols"]:
            reasons.append(f"missing cols {rec['missing_cols']}")
        if rec["extra_cols"]:
            reasons.append(f"unexpected cols {rec['extra_cols']}")
        if rec["n_nan"]:
            reasons.append(f"{rec['n_nan']} NaN cells")
        rec["status"] = "PASS" if not reasons else "FAIL"
        rec["reason"] = "; ".join(reasons)
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shapefile", default=DEFAULT_SHAPEFILE)
    ap.add_argument("--fips", help="comma-separated county FIPS codes, e.g. 32019")
    ap.add_argument("--partitions", help="comma-separated partition labels, e.g. 32019a,32019b")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--start", default="1985-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--attempts", type=int, default=3, help="download passes before giving up")
    ap.add_argument("--sweep-only", action="store_true", help="verify, do not download")
    ap.add_argument(
        "--statewide",
        action="store_true",
        help="every field in the shapefile; mutually exclusive with --fips/--partitions",
    )
    ap.add_argument(
        "--intersecting",
        action="store_true",
        help=(
            "fetch every cell whose 4 km footprint intersects a field, not just the "
            "centroid-assigned cell for each field"
        ),
    )
    ap.add_argument(
        "--copy-from",
        default="/project/handily/swim/data/32009/met_timeseries/gridmet",
        help="comma-separated precedent parquet dirs to copy from before downloading",
    )
    ap.add_argument("--sweep-csv", default=None, help="path for the per-GFID sweep table")
    ap.add_argument("--mapping-csv", default=None, help="optional path for the field->GFID table")
    args = ap.parse_args()

    if args.statewide and (args.fips or args.partitions):
        raise SystemExit("--statewide is mutually exclusive with --fips/--partitions")
    if not args.statewide and not args.fips and not args.partitions:
        raise SystemExit("one of --statewide, --fips or --partitions is required")

    t0 = time.time()
    fips = [s for s in (args.fips or "").split(",") if s]
    partitions = [s for s in (args.partitions or "").split(",") if s]
    fields = select_fields(args.shapefile, fips, partitions, args.data_root)
    # per_field is always centroid-based: that is the model's field-to-cell
    # assignment, and the DRI cross-check is defined against it. --intersecting
    # only widens the set of cells fetched.
    per_field, cells = build_mapping(fields)
    if args.intersecting:
        centroid_cells = set(cells["GFID"].astype(int))
        cells = intersecting_cells(fields)
        wider = set(cells["GFID"].astype(int))
        missing = centroid_cells - wider
        print(
            f"--intersecting: {len(wider)} cells "
            f"({len(wider) - len(centroid_cells):+d} vs centroid assignment)"
        )
        if missing:
            # A centroid-assigned cell that no field polygon intersects means the
            # two rules disagree about a field, which is worth failing on.
            raise SystemExit(
                f"{len(missing)} centroid-assigned cell(s) absent from the intersecting "
                f"set: {sorted(missing)[:20]}"
            )
    gfids = cells["GFID"].astype(int).tolist()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    if args.mapping_csv:
        per_field.to_csv(args.mapping_csv, index=False)
        print(f"wrote {args.mapping_csv}")

    if not args.sweep_only:
        sources = [s for s in (args.copy_from or "").split(",") if s and Path(s).is_dir()]
        copied = copy_existing(gfids, dest, sources)
        print(f"copied {len(copied)} parquets from precedent stores: {sorted(copied)}")

        tmpdir = tempfile.mkdtemp(prefix="nwi_gridmet_")
        shp = write_cell_input(cells, str(Path(tmpdir) / "gridmet_cells.shp"))

        for attempt in range(1, args.attempts + 1):
            todo = [g for g in gfids if not (dest / f"{g}.parquet").exists()]
            print(f"--- attempt {attempt}/{args.attempts}: {len(todo)} cells to fetch", flush=True)
            if todo:
                download_cells(shp, str(dest), args.start, args.end, todo, args.workers)
            table = sweep(gfids, dest, args.start, args.end)
            bad = table[table["status"] != "PASS"]
            print(f"attempt {attempt}: {len(table) - len(bad)} pass / {len(bad)} fail", flush=True)
            if bad.empty:
                break
            if attempt < args.attempts:
                for gfid in bad["GFID"]:
                    p = dest / f"{gfid}.parquet"
                    if p.exists():
                        p.unlink()
                        print(
                            f"deleted failing {p.name}: "
                            f"{bad.loc[bad['GFID'] == gfid, 'reason'].iloc[0]}",
                            flush=True,
                        )
        shutil.rmtree(tmpdir, ignore_errors=True)

    table = sweep(gfids, dest, args.start, args.end)
    out = args.sweep_csv or str(dest / "gridmet_sweep.csv")
    table.to_csv(out, index=False)
    bad = table[table["status"] != "PASS"]
    total_bytes = int(table["bytes"].sum())
    print(f"\nsweep: {len(table) - len(bad)} PASS / {len(bad)} FAIL -> {out}")
    print(f"store bytes: {total_bytes} ({total_bytes / 1e6:.1f} MB)")
    print(f"wall time: {time.time() - t0:.1f} s")
    if not bad.empty:
        print("FAILING CELLS:")
        print(
            bad[["GFID", "status", "n_rows", "missing_dates", "n_nan", "reason"]].to_string(
                index=False
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
