# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "pyarrow",
# ]
# ///
"""Build an ESPA-ready Landsat product ID CSV from local Example 6 extracts.

This script discovers Landsat scenes from the headers of local NDVI and/or
PT-JPL ETf extract CSVs, then joins those scene keys to a downloaded USGS
bulk metadata table to recover orderable Collection 2 Level-2 product IDs.

The local extracts only preserve scene keys like:
    LT05_226085_19870114

Those keys are enough to recover:
  - sensor
  - WRS path/row
  - acquisition date

They are not enough to reconstruct a full product identifier such as:
    LT05_L2SP_226085_19870114_20200918_02_T1

For that reason, this script requires an authoritative USGS bulk metadata
table in CSV or Parquet format and joins against it instead of guessing.

Example:
    uv run examples/6_Flux_International/build_espa_order_csv.py \
        --site US-Ro4 \
        --start-date 2018-01-01 \
        --end-date 2025-12-31 \
        --output /tmp/us_ro4_espa.csv

This script restricts export size by limiting the scene list to the requested
site(s) and date range. To keep each ESPA order spatially small, pair it with
the Example 6 extent builder and submit a chip subset in ESPA's
"Customization Options".
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

SCENE_KEY_RE = re.compile(r"^(LT04|LT05|LE07|LC08|LC09)_(\d{6})_(\d{8})$")
PRODUCT_ID_RE = re.compile(
    r"^(LT04|LT05|LE07|LC08|LC09)_(L2SP|L2SR)_(\d{6})_(\d{8})_(\d{8})_(\d{2})_(RT|T1|T2)$"
)

DEFAULT_DATA_ROOT = Path("/data/ssd1/swim/6_Flux_International/data")
DEFAULT_METADATA = Path("/data/ssd1/swim/landsat_metadata/LANDSAT_C2_L2_product_ids.parquet")
DEFAULT_NDVI_DIR = (
    DEFAULT_DATA_ROOT / "remote_sensing" / "landsat" / "extracts" / "ndvi" / "no_mask"
)
DEFAULT_PTJPL_DIR = (
    DEFAULT_DATA_ROOT / "remote_sensing" / "landsat" / "extracts" / "ptjpl_etf" / "no_mask"
)


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _parse_scene_key(scene_key: str) -> dict[str, str]:
    match = SCENE_KEY_RE.match(scene_key)
    if not match:
        raise ValueError(f"Invalid local scene key: {scene_key}")
    sensor, pathrow, acquired = match.groups()
    return {
        "scene_key": scene_key,
        "sensor": sensor,
        "pathrow": pathrow,
        "acquired": acquired,
        "path": pathrow[:3],
        "row": pathrow[3:],
    }


def _parse_product_id(product_id: str) -> dict[str, str] | None:
    match = PRODUCT_ID_RE.match(product_id)
    if not match:
        return None
    sensor, processing_level, pathrow, acquired, generated, collection, tier = match.groups()
    return {
        "product_id": product_id,
        "sensor": sensor,
        "processing_level": processing_level,
        "pathrow": pathrow,
        "acquired": acquired,
        "generated": generated,
        "collection": collection,
        "tier": tier,
        "path": pathrow[:3],
        "row": pathrow[3:],
    }


def _find_product_id_column(df: pd.DataFrame) -> str:
    exact_candidates = [
        "landsat_product_identifier_l2",
        "landsat_product_id_l2",
        "landsat_product_identifier",
        "landsat_product_id",
        "product_id",
        "display_id",
    ]
    normalized = {_normalize_column_name(col): col for col in df.columns}
    for candidate in exact_candidates:
        if candidate in normalized:
            return normalized[candidate]

    matching_cols: list[str] = []
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_string_dtype(series) and series.dtype != object:
            continue
        sample = series.dropna().astype(str).head(200)
        if not sample.empty and sample.str.match(PRODUCT_ID_RE).any():
            matching_cols.append(col)

    if len(matching_cols) == 1:
        return matching_cols[0]
    if len(matching_cols) > 1:
        raise ValueError(f"Multiple possible product ID columns found: {matching_cols}")
    raise ValueError(
        "Could not identify a Landsat product ID column in metadata. "
        f"Available columns: {list(df.columns)}"
    )


def _load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    suffix = metadata_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(metadata_path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(metadata_path, low_memory=False)
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")

    product_col = _find_product_id_column(df)
    product_df = df[[product_col]].rename(columns={product_col: "product_id"}).copy()
    product_df["product_id"] = product_df["product_id"].astype(str).str.strip()
    product_df = product_df[product_df["product_id"].str.match(PRODUCT_ID_RE, na=False)].copy()
    if product_df.empty:
        raise ValueError("No Collection 2 Level-2 product IDs found in metadata file")

    parsed = product_df["product_id"].map(_parse_product_id)
    parsed_df = pd.DataFrame([row for row in parsed if row is not None])
    if parsed_df.empty:
        raise ValueError("Failed to parse any Level-2 product IDs from metadata")

    level_rank = {"L2SP": 0, "L2SR": 1}
    tier_rank = {"T1": 0, "T2": 1, "RT": 2}
    parsed_df["level_rank"] = parsed_df["processing_level"].map(level_rank).fillna(9).astype(int)
    parsed_df["tier_rank"] = parsed_df["tier"].map(tier_rank).fillna(9).astype(int)
    parsed_df["generated_dt"] = pd.to_datetime(
        parsed_df["generated"], format="%Y%m%d", errors="coerce"
    )

    parsed_df = parsed_df.sort_values(
        by=["sensor", "pathrow", "acquired", "level_rank", "tier_rank", "generated_dt"],
        ascending=[True, True, True, True, True, False],
    ).reset_index(drop=True)
    return parsed_df


def _read_header(path: Path) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def _scene_keys_from_header(header: list[str]) -> list[str]:
    return [cell.strip() for cell in header if SCENE_KEY_RE.match(cell.strip())]


def _collect_files_for_site(site: str, source_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    patterns = [
        f"ndvi_{site}_no_mask_*.csv",
        f"ptjpl_etf_{site}_no_mask_*.csv",
    ]
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for pattern in patterns:
            files.extend(sorted(source_dir.glob(pattern)))
    return sorted(set(files))


def _collect_scene_inventory(sites: list[str], source_dirs: list[Path]) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    scene_sites: defaultdict[str, set[str]] = defaultdict(set)
    scene_files: defaultdict[str, set[str]] = defaultdict(set)

    for site in sites:
        files = _collect_files_for_site(site, source_dirs)
        for path in files:
            header = _read_header(path)
            for scene_key in _scene_keys_from_header(header):
                scene_sites[scene_key].add(site)
                scene_files[scene_key].add(path.name)

    for scene_key, scene_site_set in sorted(scene_sites.items()):
        parsed = _parse_scene_key(scene_key)
        parsed["sites"] = ",".join(sorted(scene_site_set))
        parsed["site_count"] = len(scene_site_set)
        parsed["source_files"] = ",".join(sorted(scene_files[scene_key]))
        records.append(parsed)

    if not records:
        raise ValueError("No Landsat scene keys found for requested site(s)")

    return (
        pd.DataFrame(records).sort_values(["sensor", "pathrow", "acquired"]).reset_index(drop=True)
    )


def _filter_scene_inventory(
    scene_df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if not start_date and not end_date:
        return scene_df

    filtered = scene_df.copy()
    filtered["acquired_dt"] = pd.to_datetime(filtered["acquired"], format="%Y%m%d", errors="coerce")

    if filtered["acquired_dt"].isna().any():
        bad_count = int(filtered["acquired_dt"].isna().sum())
        raise ValueError(f"Failed to parse acquisition date for {bad_count} local scene keys")

    if start_date:
        start_ts = pd.to_datetime(start_date)
        filtered = filtered[filtered["acquired_dt"] >= start_ts]
    if end_date:
        end_ts = pd.to_datetime(end_date)
        filtered = filtered[filtered["acquired_dt"] <= end_ts]

    filtered = filtered.drop(columns=["acquired_dt"]).reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No Landsat scene keys remain after applying the requested date range")
    return filtered


def _match_scene_inventory(
    scene_df: pd.DataFrame, metadata_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = scene_df.merge(
        metadata_df,
        how="left",
        on=["sensor", "pathrow", "acquired", "path", "row"],
        suffixes=("", "_meta"),
    )

    matched = merged.dropna(subset=["product_id"]).copy()
    if not matched.empty:
        matched = matched.sort_values(
            by=["scene_key", "level_rank", "tier_rank", "generated_dt"],
            ascending=[True, True, True, False],
        )
        matched["match_rank"] = matched.groupby("scene_key").cumcount()
        matched["candidate_count"] = matched.groupby("scene_key")["product_id"].transform("count")
        best = matched[matched["match_rank"] == 0].copy()
    else:
        best = matched.copy()

    unmatched_keys = set(scene_df["scene_key"]) - set(best["scene_key"])
    unmatched = scene_df[scene_df["scene_key"].isin(unmatched_keys)].copy()
    return best.reset_index(drop=True), unmatched.reset_index(drop=True)


def build_espa_order(
    sites: list[str],
    metadata_path: Path,
    output_csv: Path,
    audit_csv: Path | None = None,
    unmatched_csv: Path | None = None,
    source_dirs: list[Path] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    source_dirs = source_dirs or [DEFAULT_NDVI_DIR, DEFAULT_PTJPL_DIR]

    scene_df = _collect_scene_inventory(sites, source_dirs)
    scene_df = _filter_scene_inventory(scene_df, start_date=start_date, end_date=end_date)
    metadata_df = _load_metadata(metadata_path)
    best, unmatched = _match_scene_inventory(scene_df, metadata_df)

    if best.empty:
        raise ValueError("No local scene keys could be matched to Level-2 metadata")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_csv = audit_csv or output_csv.with_name(f"{output_csv.stem}_matches.csv")
    unmatched_csv = unmatched_csv or output_csv.with_name(f"{output_csv.stem}_unmatched.csv")

    product_list = (
        best[["product_id"]]
        .drop_duplicates()
        .rename(columns={"product_id": "landsat_product_identifier_l2"})
    )
    product_list.to_csv(output_csv, index=False)

    audit_cols = [
        "scene_key",
        "sites",
        "site_count",
        "product_id",
        "processing_level",
        "tier",
        "generated",
        "candidate_count",
        "source_files",
    ]
    best[audit_cols].sort_values(["scene_key"]).to_csv(audit_csv, index=False)
    unmatched.sort_values(["scene_key"]).to_csv(unmatched_csv, index=False)

    print(f"Sites: {', '.join(sites)}")
    if start_date or end_date:
        print(f"Date range: {start_date or 'start'} to {end_date or 'end'}")
    print(f"Local scene keys: {len(scene_df)}")
    print(f"Matched scene keys: {len(best)}")
    print(f"Unmatched scene keys: {len(unmatched)}")
    print(f"Unique Level-2 product IDs: {len(product_list)}")
    print(f"ESPA CSV: {output_csv}")
    print(f"Match audit: {audit_csv}")
    print(f"Unmatched audit: {unmatched_csv}")

    if not unmatched.empty:
        print("WARNING: unmatched scene keys were written to the audit CSV")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ESPA product ID CSV from Example 6 scene headers"
    )
    parser.add_argument("--site", action="append", required=True, help="Flux site ID (repeatable)")
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA),
        help=f"USGS bulk metadata file (CSV or Parquet), default: {DEFAULT_METADATA}",
    )
    parser.add_argument("--output", required=True, help="Output ESPA CSV")
    parser.add_argument("--audit", default=None, help="Detailed match audit CSV")
    parser.add_argument("--unmatched", default=None, help="Unmatched scene audit CSV")
    parser.add_argument(
        "--start-date", default=None, help="Inclusive acquisition start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", default=None, help="Inclusive acquisition end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--ndvi-dir", default=str(DEFAULT_NDVI_DIR), help="Local Landsat NDVI extract dir"
    )
    parser.add_argument(
        "--ptjpl-dir",
        default=str(DEFAULT_PTJPL_DIR),
        help="Local Landsat PT-JPL ETf extract dir",
    )
    args = parser.parse_args()

    build_espa_order(
        sites=sorted(set(args.site)),
        metadata_path=Path(args.metadata),
        output_csv=Path(args.output),
        audit_csv=Path(args.audit) if args.audit else None,
        unmatched_csv=Path(args.unmatched) if args.unmatched else None,
        source_dirs=[Path(args.ndvi_dir), Path(args.ptjpl_dir)],
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()
