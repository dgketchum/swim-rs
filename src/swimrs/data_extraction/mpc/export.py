"""Partitioning, gap accounting, and EE-contract CSV assembly.

Output contract (drop-in for the EE exports consumed by
container.components.ingestor._parse_single_csv):
- one CSV per (partition, mask, year)
- header: {feature_id},{scene_id},... where a Landsat scene_id's last
  '_'-token is YYYYMMDD and a Sentinel column starts with YYYYMMDD
- one row per field (duplicate feature ids keep duplicate rows, matching
  the EE pull), empty cell where no unmasked pixel touched the field
"""

import math
import re
import subprocess
from pathlib import Path

import pandas as pd

CHUNK_SIZE = 900
CHUNK_SUFFIXES = "abcdefghijklmnopqrstuvwxyz"

# instrument -> (bucket subdir, csv stem)
LAYOUT = {"landsat": ("ndvi", "ndvi"), "sentinel": ("ndvi_s2", "ndvi_s2")}


def partition_fields(gdf, feature_id, fips_col="FIPS", chunk_size=CHUNK_SIZE):
    """FIPS + a/b/c... chunks of ceil(n/chunk_size) — the gs://wudr/nv layout.

    Returns [(label, [feature ids]), ...] like ('32001a', [...]), ('32009', [...]).
    """
    partitions = []
    for fips, grp in gdf.groupby(fips_col):
        fids = grp[feature_id].tolist()
        n_chunks = math.ceil(len(fids) / chunk_size)
        if n_chunks > 1:
            for ci, chunk in enumerate(_chunk_list(fids, n_chunks)):
                partitions.append((f"{fips}{CHUNK_SUFFIXES[ci]}", chunk))
        else:
            partitions.append((str(fips), fids))
    return partitions


def list_bucket_files(bucket, file_prefix, instrument="landsat"):
    """Existing (partition, mask, year) tuples already in the bucket."""
    subdir, stem = LAYOUT[instrument]
    pattern = f"gs://{bucket}/{file_prefix}/*/{subdir}/*/*.csv"
    proc = subprocess.run(["gsutil", "ls", pattern], capture_output=True, text=True)
    existing = set()
    rx = re.compile(
        rf"gs://{bucket}/{file_prefix}/(?P<part>[^/]+)/{subdir}/(?P<mask>[^/]+)/"
        rf"{stem}_(?P=mask)_(?P<year>\d{{4}})\.csv$"
    )
    for line in proc.stdout.splitlines():
        match = rx.match(line.strip())
        if match:
            existing.add((match.group("part"), match.group("mask"), int(match.group("year"))))
    return existing


def build_targets(partitions, mask_types, years, existing=None):
    """Missing (partition, mask, year) set and the per-year work summary.

    Returns (missing, per_year) where per_year maps year -> dict with the
    union of field ids and mask types needed for that year.
    """
    existing = existing or set()
    missing = set()
    for label, _ in partitions:
        for mask in mask_types:
            for year in years:
                if (label, mask, year) not in existing:
                    missing.add((label, mask, year))
    fields_by_label = dict(partitions)
    per_year = {}
    for label, mask, year in missing:
        entry = per_year.setdefault(year, {"masks": set(), "fields": set()})
        entry["masks"].add(mask)
        entry["fields"].update(fields_by_label[label])
    return missing, per_year


def assemble_csv(records_dir, instrument, year, mask, field_rows, feature_id, out_path):
    """Pivot per-scene records into one EE-contract CSV.

    `field_rows` is the ordered [(row_key, feature_id_value), ...] for the
    partition — row_key is the unique original row index, so duplicate
    feature ids stay distinct rows exactly as in the EE export.
    """
    year_dir = Path(records_dir) / instrument / str(year)
    frames = []
    for parquet in sorted(year_dir.glob("*.parquet")):
        df = pd.read_parquet(parquet)
        if not df.empty:
            frames.append(df[df["mask"] == mask])
    row_keys = [rk for rk, _ in field_rows]
    if frames:
        records = pd.concat(frames, ignore_index=True)
        records = records[records["_row"].isin(set(row_keys))]
        wide = records.pivot_table(index="_row", columns="scene_id", values="mean", aggfunc="first")
        wide = wide[sorted(wide.columns, key=_column_date_key)]
    else:
        wide = pd.DataFrame()
    wide = wide.reindex(row_keys)
    wide.insert(0, feature_id, [fid for _, fid in field_rows])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, index=False)
    return out_path


def csv_path(out_dir, instrument, partition, mask, year):
    subdir, stem = LAYOUT[instrument]
    return Path(out_dir) / partition / subdir / mask / f"{stem}_{mask}_{year}.csv"


def _column_date_key(col):
    """Chronological sort key: Landsat date is the last token, S2 the first."""
    parts = col.split("_")
    token = parts[-1] if parts[-1][:8].isdigit() and len(parts[-1]) == 8 else parts[0][:8]
    return (token, col)


def _chunk_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]
