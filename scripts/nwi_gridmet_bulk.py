"""Bulk gridMET acquisition from the yearly NetCDF files, one request per variable-year.

``nwi_gridmet.py`` drives ``download_gridmet``, which works a point at a time:
for every cell it opens the CONUS OPeNDAP aggregation once per variable and
slices out a single pixel. That is ~10 remote dataset opens per cell, so a
statewide run of 3,555 cells is ~34,000 round trips against
thredds.northwestknowledge.net. Measured on zephyr at 12 workers it sustained
about 0.6 cells/min -- roughly 90 hours -- and the ``#fillmismatch`` I/O
failures that make the sweep-and-refetch loop necessary scale with the request
count. During that run the THREDDS service degraded far enough that even
single dataset opens hung past 110 s.

This script drops OPeNDAP entirely and pulls the same data as plain files from
https://www.northwestknowledge.net/metdata/, which serves gridMET as one
NetCDF per variable per year. Cost becomes 8 variables x 41 years = 328 file
downloads (~39 GB, measured at ~23 MB/s) regardless of how many cells are
wanted, because every cell in the state is inside every file. Cells are sliced
out locally.

The output is interchangeable with the point path, and that is checked rather
than asserted: ``--verify-only`` assembles cells that ``download_gridmet``
already wrote and diffs every column. Spot checks during development found
exact agreement -- ``max|diff| = 0.0`` for eto over 1985 and for elevation
across 12 cells -- which is expected, since the OPeNDAP aggregations are these
same files aggregated.

Decoding and unit steps are lifted from ``download_gridmet`` deliberately,
including the ones that look like bugs:

* ``u2`` is gridMET ``vs``, wind speed at 10 m, assigned with no height
  adjustment. Reproduced as-is: this script exists to produce the same
  parquets faster, not to change the meteorology under calibrated runs.
* Nearest-neighbour selection mirrors ``.sel(method="nearest")`` by argmin
  over the coordinate arrays, against the same formula-derived cell centers.
* ``ea`` is derived from specific humidity and elevation-based air pressure
  exactly as the point path does.

Note the yearly files name their variable differently from the aggregations
(``potential_evapotranspiration`` rather than
``daily_mean_reference_evapotranspiration_grass``), so the variable is
identified by its dimensions instead of by name.

Usage:
    uv run python scripts/nwi_gridmet_bulk.py --statewide --intersecting \
        --dest /project/handily/swim/data/nwi_gridmet \
        --start 1985-01-01 --end 2025-12-31
"""

import argparse
import shutil
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nwi_gridmet import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_DEST,
    DEFAULT_SHAPEFILE,
    build_mapping,
    intersecting_cells,
    select_fields,
    sweep,
)

BASE = "https://www.northwestknowledge.net/metdata"
# Yearly grids live under metdata/data/, the static elevation grid under
# metdata/elev/ -- different subdirectories, not a shared prefix.
DATA_BASE = f"{BASE}/data"
ELEV_URL = f"{BASE}/elev/metdata_elevationdata.nc"

# gridMET variable -> output column, matching CLIMATE_COLS['col'] in
# download_gridmet. 'q' is an intermediate: specific humidity becomes 'ea'.
VARIABLES = {
    "tmmn": "tmin",
    "tmmx": "tmax",
    "pet": "eto",
    "etr": "etr",
    "pr": "prcp",
    "srad": "srad",
    "vs": "u2",
    "sph": "q",
}

OUT_COLS = ["tmin", "tmax", "eto", "etr", "prcp", "srad", "u2", "ea", "elev"]


def _download(url, path, attempts=4, backoff=8):
    last = None
    for i in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(path, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            return path
        except Exception as exc:  # noqa: BLE001 - urllib raises a wide family
            last = exc
            if path.exists():
                path.unlink()
            if i < attempts:
                time.sleep(backoff * i)
    raise RuntimeError(f"download failed after {attempts} attempts: {url}: {last}")


def _data_var(ds):
    """The one variable on (day, lat, lon); the yearly files name it per-product."""
    for name, da in ds.data_vars.items():
        if {"lat", "lon"}.issubset(set(da.dims)):
            return name
    raise RuntimeError(f"no gridded variable among {list(ds.data_vars)}")


def _nearest_indices(coord, targets):
    """Index of the nearest coordinate value per target, as .sel(method='nearest') does."""
    coord = np.asarray(coord, dtype=float)
    return np.array([int(np.abs(coord - t).argmin()) for t in targets])


def fetch_year(var, year, lats, lons, tmpdir):
    """Download one variable-year and return (days, [n_cells, n_days]) for the target cells.

    Runs in a worker process: takes plain arrays, returns plain arrays.
    """
    url = f"{DATA_BASE}/{var}_{year}.nc"
    path = Path(tmpdir) / f"{var}_{year}.nc"
    _download(url, path)
    try:
        with xr.open_dataset(path) as ds:
            name = _data_var(ds)
            lat_idx = _nearest_indices(ds["lat"].values, lats)
            lon_idx = _nearest_indices(ds["lon"].values, lons)
            lat_lo, lat_hi = int(lat_idx.min()), int(lat_idx.max())
            lon_lo, lon_hi = int(lon_idx.min()), int(lon_idx.max())
            # Read only the window covering the targets, not the CONUS grid:
            # a full year at 585x1386 float64 is 2.4 GB, the Nevada window ~70 MB.
            block = (
                ds[name].isel(lat=slice(lat_lo, lat_hi + 1), lon=slice(lon_lo, lon_hi + 1)).values
            )
            days = pd.to_datetime(ds[ds[name].dims[0]].values)
        vals = block[:, lat_idx - lat_lo, lon_idx - lon_lo].T.astype(np.float64)
        return var, np.asarray(days), vals
    finally:
        if path.exists():
            path.unlink()


def fetch_elevation(lats, lons, tmpdir):
    path = Path(tmpdir) / "metdata_elevationdata.nc"
    _download(ELEV_URL, path)
    try:
        with xr.open_dataset(path) as ds:
            arr = np.asarray(ds["elevation"].values).squeeze()
            lat_idx = _nearest_indices(ds["lat"].values, lats)
            lon_idx = _nearest_indices(ds["lon"].values, lons)
        return arr[lat_idx, lon_idx].astype(np.float64)
    finally:
        if path.exists():
            path.unlink()


def assemble(cells, start, end, workers=8, tmpdir=None):
    """Fetch every variable-year in parallel; return {GFID: DataFrame} in point-path order."""
    from swimrs.data_extraction.gridmet.gridmet import actual_vapor_pressure, air_pressure

    lats = cells["LAT"].to_numpy(dtype=float)
    lons = cells["LON"].to_numpy(dtype=float)
    dates = pd.date_range(start, end, freq="D")
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)

    blocks = {col: np.full((len(cells), len(dates)), np.nan) for col in VARIABLES.values()}
    filled = {col: np.zeros(len(dates), dtype=bool) for col in VARIABLES.values()}

    tasks = [(v, y) for v in VARIABLES for y in years]
    done, total = 0, len(tasks)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        elev_fut = ex.submit(fetch_elevation, lats, lons, tmpdir)
        futures = {ex.submit(fetch_year, v, y, lats, lons, tmpdir): (v, y) for v, y in tasks}
        for fut in as_completed(futures):
            var, year = futures[fut]
            v, days, vals = fut.result()
            col = VARIABLES[v]
            keep = (days >= dates[0]) & (days <= dates[-1])
            pos = np.searchsorted(dates, days[keep])
            blocks[col][:, pos] = vals[:, keep]
            filled[col][pos] = True
            done += 1
            rate = done / max(time.time() - t0, 1e-9)
            print(
                f"  [{done}/{total}] {var} {year} ({(total - done) / rate / 60:.0f} min left)",
                flush=True,
            )
        elev = elev_fut.result()

    for col, mask in filled.items():
        if not mask.all():
            raise RuntimeError(f"{col}: {int((~mask).sum())} day(s) never filled")

    # air_pressure depends only on elevation, so it is one value per cell.
    pair = air_pressure(elev)
    frames = {}
    for i, gfid in enumerate(cells["GFID"].astype(int).to_numpy()):
        df = pd.DataFrame(index=dates)
        df["tmin"] = blocks["tmin"][i] - 273.15
        df["tmax"] = blocks["tmax"][i] - 273.15
        df["eto"] = blocks["eto"][i]
        df["etr"] = blocks["etr"][i]
        df["prcp"] = blocks["prcp"][i]
        df["srad"] = blocks["srad"][i]
        df["u2"] = blocks["u2"][i]
        df["ea"] = actual_vapor_pressure(blocks["q"][i], np.full(len(dates), pair[i]))
        df["elev"] = elev[i]
        frames[int(gfid)] = df[OUT_COLS].astype("float64")
    return frames


def verify(frames, dest):
    """Diff assembled cells against parquets the point path already wrote."""
    dest = Path(dest)
    checked, worst = 0, {}
    for gfid, df in frames.items():
        path = dest / f"{gfid}.parquet"
        if not path.exists():
            continue
        ref = pd.read_parquet(path)
        # Compare on shared dates only, so a short-range probe can be checked
        # against a full-POR reference without refetching 41 years.
        shared = ref.index.intersection(df.index)
        if len(shared) == 0:
            print(f"  {gfid}: no overlapping dates with {path.name}")
            return False, {}
        a, b = ref.loc[shared], df.loc[shared]
        for col in OUT_COLS:
            d = float(np.nanmax(np.abs(a[col].to_numpy() - b[col].to_numpy())))
            worst[col] = max(worst.get(col, 0.0), d)
        checked += 1
    if not checked:
        print("  no overlapping parquets to verify against")
        return False, {}
    print(f"  verified {checked} cell(s) against the point path:")
    for col in OUT_COLS:
        print(f"    {col:>5}: max|diff| = {worst[col]:.3e}")
    return max(worst.values()) < 1e-9, worst


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shapefile", default=DEFAULT_SHAPEFILE)
    ap.add_argument("--fips")
    ap.add_argument("--partitions")
    ap.add_argument("--statewide", action="store_true")
    ap.add_argument("--intersecting", action="store_true")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--start", default="1985-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--workers", type=int, default=8, help="concurrent variable-year downloads")
    ap.add_argument("--tmpdir", default=None, help="scratch for downloaded NetCDF files")
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="assemble and diff against existing parquets, write nothing",
    )
    ap.add_argument("--overwrite", action="store_true", help="rewrite cells already present")
    ap.add_argument("--sweep-csv", default=None)
    args = ap.parse_args()

    if args.statewide and (args.fips or args.partitions):
        raise SystemExit("--statewide is mutually exclusive with --fips/--partitions")
    if not args.statewide and not args.fips and not args.partitions:
        raise SystemExit("one of --statewide, --fips or --partitions is required")

    t0 = time.time()
    fips = [s for s in (args.fips or "").split(",") if s]
    partitions = [s for s in (args.partitions or "").split(",") if s]
    fields = select_fields(args.shapefile, fips, partitions, args.data_root)
    _, cells = build_mapping(fields)
    if args.intersecting:
        centroid_cells = set(cells["GFID"].astype(int))
        cells = intersecting_cells(fields)
        missing = centroid_cells - set(cells["GFID"].astype(int))
        if missing:
            raise SystemExit(f"{len(missing)} centroid-assigned cell(s) outside intersecting set")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Every cell is inside every downloaded file, so unlike the point path there
    # is nothing to save by skipping cells already on disk -- and rebuilding them
    # is what lets --verify-only check this path against the point path.
    print(f"assembling {len(cells)} cells, {args.start} -> {args.end}", flush=True)

    tmpdir = args.tmpdir or tempfile.mkdtemp(prefix="nwi_gridmet_bulk_")
    Path(tmpdir).mkdir(parents=True, exist_ok=True)
    try:
        frames = assemble(cells, args.start, args.end, workers=args.workers, tmpdir=tmpdir)
    finally:
        if not args.tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    if args.verify_only:
        ok, _ = verify(frames, dest)
        print(f"\nverdict: {'MATCH' if ok else 'MISMATCH'}")
        print(f"wall time: {time.time() - t0:.1f} s")
        return 0 if ok else 1

    existing = {int(p.stem) for p in dest.glob("*.parquet") if p.stem.isdigit()}
    written, skipped = 0, 0
    for gfid, df in frames.items():
        if gfid in existing and not args.overwrite:
            skipped += 1
            continue
        df.to_parquet(dest / f"{gfid}.parquet")
        written += 1
    print(f"wrote {written} parquets, left {skipped} existing in place", flush=True)

    gfids = sorted(frames)
    table = sweep(gfids, dest, args.start, args.end)
    out = args.sweep_csv or str(dest / "gridmet_sweep_bulk.csv")
    table.to_csv(out, index=False)
    bad = table[table["status"] != "PASS"]
    print(f"sweep: {len(table) - len(bad)} PASS / {len(bad)} FAIL -> {out}")
    print(f"wall time: {time.time() - t0:.1f} s")
    if not bad.empty:
        print(bad[["GFID", "status", "n_rows", "missing_dates", "n_nan", "reason"]].to_string())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
