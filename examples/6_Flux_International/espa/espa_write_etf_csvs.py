# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
# ]
# ///
"""Convert extracted per-scene ETF observations into ingest-ready annual CSVs.

Reads the per-site-year ETF JSON files and writes annual CSVs matching the
format expected by the container ingestor. Output goes into the Landsat
extracts tree so the legacy SSEBop container builder finds it when
``etf_target_model = "ssebop"``.

Output format:
    sid,ETF_YYYYMMDD,ETF_YYYYMMDD,...
    US-Ro4,0.85,0.92,...

Filename convention:
    ssebop_etf_{site}_no_mask_{year}.csv

Default output directory:
    {landsat_dir}/extracts/ssebop_etf/no_mask/

Example:
    uv run examples/6_Flux_International/espa/espa_write_etf_csvs.py \
        --manifest /data/ssd1/swim/6_Flux_International/data/remote_sensing/espa/espa_manifest.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_CSV_DIR = Path(
    "/data/ssd1/swim/6_Flux_International/data/remote_sensing/landsat/extracts/ssebop_etf/no_mask"
)


def _date_key_to_column(date_key: str) -> str:
    """Convert '2020-07-15' to 'ETF_20200715'."""
    return "ETF_" + date_key.replace("-", "")


def write_csvs(manifest_path: Path, output_dir: Path | None = None) -> None:
    manifest = pd.read_csv(manifest_path, dtype=str)
    if "csv_status" not in manifest.columns:
        manifest["csv_status"] = ""
    json_dir = manifest_path.parent / "extracts" / "etf_json"
    csv_dir = output_dir or DEFAULT_CSV_DIR
    csv_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*_etf.json")) if json_dir.exists() else []
    if not json_files:
        print("No ETF JSON files found.")
        return

    print(f"Processing {len(json_files)} ETF JSON files...")
    written = 0

    skipped = 0
    for jf in json_files:
        # Parse site and year from filename: {site}_{year}_etf.json
        stem = jf.stem  # e.g. US-Ro4_2020_etf
        parts = stem.rsplit("_", 2)
        if len(parts) < 3:
            print(f"  SKIP {jf.name}: unexpected filename format")
            continue
        # Handle site IDs with underscores by taking last two parts as year and "etf"
        year = parts[-2]
        site = "_".join(parts[:-2])

        csv_path = csv_dir / f"ssebop_etf_{site}_no_mask_{year}.csv"

        # Only rewrite if JSON is newer than existing CSV
        if csv_path.exists() and csv_path.stat().st_mtime >= jf.stat().st_mtime:
            skipped += 1
            continue

        with open(jf) as f:
            data = json.load(f)

        site_data = data.get(site, {})
        if not site_data:
            site_data = next(iter(data.values()), {})

        if not site_data:
            continue

        # Build row: sid + ETF columns sorted by date
        row = {"sid": site}
        for date_key, stats in sorted(site_data.items()):
            mean_val = stats.get("mean")
            if mean_val is not None:
                col = _date_key_to_column(date_key)
                row[col] = round(mean_val, 6)

        etf_cols = sorted(k for k in row if k.startswith("ETF_"))
        if not etf_cols:
            continue

        columns = ["sid"] + etf_cols
        df = pd.DataFrame([row], columns=columns)
        df.to_csv(csv_path, index=False)
        written += 1

        # Update manifest csv_status for this site-year
        match = (manifest["site"] == site) & (manifest["year"] == year)
        if match.any():
            manifest.loc[match, "csv_status"] = "written"

    # Also mark skipped (already up-to-date) rows
    for jf in json_files:
        stem = jf.stem
        parts = stem.rsplit("_", 2)
        if len(parts) < 3:
            continue
        year = parts[-2]
        site = "_".join(parts[:-2])
        csv_path = csv_dir / f"ssebop_etf_{site}_no_mask_{year}.csv"
        if csv_path.exists():
            match = (manifest["site"] == site) & (manifest["year"] == year)
            if match.any() and manifest.loc[match, "csv_status"].iloc[0] != "written":
                manifest.loc[match, "csv_status"] = "written"

    manifest.to_csv(manifest_path, index=False)
    print(f"\nWrote {written} CSV files, {skipped} up-to-date ({csv_dir})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write ingest-ready ETF CSVs")
    parser.add_argument("--manifest", required=True, help="Path to espa_manifest.csv")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory for CSVs (default: {DEFAULT_CSV_DIR})",
    )
    args = parser.parse_args()
    write_csvs(
        Path(args.manifest),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
