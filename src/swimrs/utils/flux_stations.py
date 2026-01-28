"""
Flux Stations Master Shapefile Utilities
=========================================

Create and manage a master shapefile of flux stations with standardized columns.
Combines footprint geometries with station metadata.

Usage
-----
    # Create master shapefile
    python -m swimrs.utils.flux_stations create \\
        --footprints /path/to/footprints.shp \\
        --metadata /path/to/metadata.csv \\
        --output examples/gis/flux_stations.shp

    # List stations
    python -m swimrs.utils.flux_stations list --master examples/gis/flux_stations.shp

    # Extract specific stations
    python -m swimrs.utils.flux_stations extract \\
        --master examples/gis/flux_stations.shp \\
        --sites US-FPe S2 \\
        --output examples/2_Fort_Peck/data/gis/flux_fields.shp

    # Filter by classification
    python -m swimrs.utils.flux_stations filter \\
        --master examples/gis/flux_stations.shp \\
        --classification Croplands \\
        --output examples/5_Flux_Ensemble/data/gis/flux_fields.shp
"""

import argparse
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Columns to retain from metadata (in order)
# Note: 'classification' is renamed to 'lc_class' for shapefile 10-char limit
METADATA_COLUMNS = ["site_id", "classification", "state", "source", "record", "lat", "lon", "elev"]

# Column name mapping for shapefile format (10 char max)
SHAPEFILE_COLUMN_MAP = {
    "classification": "lc_class",
}
SHAPEFILE_COLUMN_MAP_REVERSE = {v: k for k, v in SHAPEFILE_COLUMN_MAP.items()}


def _write_provenance(output_shp, command, sources, extra_info=None):
    """Write provenance file alongside output shapefile.

    Parameters
    ----------
    output_shp : Path
        Path to the output shapefile.
    command : str
        The command used to create the shapefile.
    sources : dict
        Dictionary of source name -> path mappings.
    extra_info : dict, optional
        Additional information to include in provenance.
    """
    output_shp = Path(output_shp)
    provenance_file = output_shp.with_name("shapefile_provenance.txt")

    lines = [
        "SHAPEFILE PROVENANCE",
        "=" * 60,
        f"Output: {output_shp.name}",
        f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Sources:",
    ]

    for name, path in sources.items():
        lines.append(f"  {name}: {path}")

    if extra_info:
        lines.append("")
        lines.append("Details:")
        for key, value in extra_info.items():
            lines.append(f"  {key}: {value}")

    lines.extend(
        [
            "",
            "Command:",
            f"  {command}",
            "",
            "=" * 60,
        ]
    )

    with open(provenance_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Provenance written to: {provenance_file}")


def _normalize_columns(gdf):
    """Rename shapefile columns back to standard names."""
    return gdf.rename(columns=SHAPEFILE_COLUMN_MAP_REVERSE)


def _shp_columns(gdf):
    """Rename columns for shapefile 10-char limit."""
    return gdf.rename(columns=SHAPEFILE_COLUMN_MAP)


def _get_classification_col(gdf):
    """Get the classification column name (handles both standard and shapefile names)."""
    if "classification" in gdf.columns:
        return "classification"
    elif "lc_class" in gdf.columns:
        return "lc_class"
    else:
        raise KeyError("No classification column found (tried 'classification' and 'lc_class')")


def create_master_shapefile(footprints_shp, metadata_csv, output_shp, overwrite=False):
    """Create master flux stations shapefile from footprints and metadata.

    Parameters
    ----------
    footprints_shp : str or Path
        Path to shapefile with station footprint geometries.
        Must have 'site_id' column.
    metadata_csv : str or Path
        Path to CSV with station metadata.
        Must have 'site_id' and columns in METADATA_COLUMNS.
    output_shp : str or Path
        Output path for master shapefile.
    overwrite : bool
        If True, overwrite existing output file.

    Returns
    -------
    geopandas.GeoDataFrame
        Master shapefile GeoDataFrame.
    """
    output_shp = Path(output_shp)

    if output_shp.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_shp}. Use overwrite=True to replace.")

    # Load footprints
    footprints = gpd.read_file(footprints_shp)
    if "site_id" not in footprints.columns:
        raise ValueError(
            f"Footprints shapefile must have 'site_id' column. Found: {list(footprints.columns)}"
        )

    # Load metadata
    metadata = pd.read_csv(metadata_csv)
    if "site_id" not in metadata.columns:
        raise ValueError(
            f"Metadata CSV must have 'site_id' column. Found: {list(metadata.columns)}"
        )

    # Check for required columns
    missing = [c for c in METADATA_COLUMNS if c not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {missing}")

    # Select only needed columns from metadata
    metadata = metadata[METADATA_COLUMNS].copy()

    # Merge on site_id (inner join keeps only common stations)
    merged = footprints.merge(metadata, on="site_id", how="inner", suffixes=("_fp", "_meta"))

    # Handle duplicate 'state' column (prefer metadata version)
    if "state_fp" in merged.columns and "state_meta" in merged.columns:
        merged["state"] = merged["state_meta"]
        merged.drop(columns=["state_fp", "state_meta"], inplace=True)

    # Keep only the final columns we want
    final_columns = METADATA_COLUMNS + ["geometry"]
    merged = merged[[c for c in final_columns if c in merged.columns]]

    # Ensure CRS is set (use footprints CRS)
    if merged.crs is None:
        merged.set_crs(footprints.crs, inplace=True)

    # Create output directory if needed
    output_shp.parent.mkdir(parents=True, exist_ok=True)

    # Rename columns for shapefile 10-char limit
    merged = merged.rename(columns=SHAPEFILE_COLUMN_MAP)

    # Write shapefile
    merged.to_file(output_shp)

    print(f"Created master shapefile: {output_shp}")
    print(f"  Stations: {len(merged)}")
    print(f"  Columns: {list(merged.columns)}")
    print(f"  CRS: {merged.crs}")

    # Write provenance
    command = f"python -m swimrs.utils.flux_stations create --footprints {footprints_shp} --metadata {metadata_csv} --output {output_shp}"
    if overwrite:
        command += " --overwrite"
    _write_provenance(
        output_shp,
        command,
        sources={
            "footprints": str(Path(footprints_shp).resolve()),
            "metadata": str(Path(metadata_csv).resolve()),
        },
        extra_info={
            "stations_count": len(merged),
            "crs": str(merged.crs),
        },
    )

    return merged


def extract_stations(master_shp, site_ids, output_shp, overwrite=False):
    """Extract specific stations from master shapefile.

    Parameters
    ----------
    master_shp : str or Path
        Path to master flux stations shapefile.
    site_ids : list of str
        Station IDs to extract.
    output_shp : str or Path
        Output path for extracted shapefile.
    overwrite : bool
        If True, overwrite existing output file.

    Returns
    -------
    geopandas.GeoDataFrame
        Extracted stations GeoDataFrame.
    """
    output_shp = Path(output_shp)

    if output_shp.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_shp}. Use overwrite=True to replace.")

    # Load master shapefile
    master = gpd.read_file(master_shp)

    # Filter to requested sites
    extracted = master[master["site_id"].isin(site_ids)].copy()

    if len(extracted) == 0:
        available = master["site_id"].tolist()
        raise ValueError(
            f"No matching stations found. Requested: {site_ids}. Available: {available[:10]}..."
        )

    # Check for missing sites
    found = set(extracted["site_id"].tolist())
    missing = set(site_ids) - found
    if missing:
        print(f"  Warning: {len(missing)} site(s) not found: {list(missing)}")

    # Create output directory if needed
    output_shp.parent.mkdir(parents=True, exist_ok=True)

    # Write shapefile
    extracted.to_file(output_shp)

    print(f"Extracted {len(extracted)} station(s) to: {output_shp}")
    for sid in extracted["site_id"]:
        print(f"  - {sid}")

    # Write provenance
    sites_str = " ".join(site_ids)
    command = f"python -m swimrs.utils.flux_stations extract --master {master_shp} --sites {sites_str} --output {output_shp}"
    if overwrite:
        command += " --overwrite"
    _write_provenance(
        output_shp,
        command,
        sources={
            "master": str(Path(master_shp).resolve()),
        },
        extra_info={
            "sites_requested": ", ".join(site_ids),
            "sites_extracted": ", ".join(extracted["site_id"].tolist()),
            "stations_count": len(extracted),
        },
    )

    return extracted


def filter_by_classification(master_shp, classification, output_shp, overwrite=False):
    """Filter master shapefile by land cover classification.

    Parameters
    ----------
    master_shp : str or Path
        Path to master flux stations shapefile.
    classification : str
        Land cover classification to filter (e.g., 'Croplands', 'Grasslands').
    output_shp : str or Path
        Output path for filtered shapefile.
    overwrite : bool
        If True, overwrite existing output file.

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered stations GeoDataFrame.
    """
    output_shp = Path(output_shp)

    if output_shp.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_shp}. Use overwrite=True to replace.")

    # Load master shapefile
    master = gpd.read_file(master_shp)

    # Filter by classification (handle both column names)
    cls_col = _get_classification_col(master)
    filtered = master[master[cls_col] == classification].copy()

    if len(filtered) == 0:
        available = master[cls_col].unique().tolist()
        raise ValueError(
            f"No stations with classification '{classification}'. Available: {available}"
        )

    # Create output directory if needed
    output_shp.parent.mkdir(parents=True, exist_ok=True)

    # Write shapefile
    filtered.to_file(output_shp)

    print(f"Filtered {len(filtered)} {classification} station(s) to: {output_shp}")

    # Write provenance
    command = f"python -m swimrs.utils.flux_stations filter --master {master_shp} --classification {classification} --output {output_shp}"
    if overwrite:
        command += " --overwrite"
    _write_provenance(
        output_shp,
        command,
        sources={
            "master": str(Path(master_shp).resolve()),
        },
        extra_info={
            "classification": classification,
            "sites_extracted": ", ".join(filtered["site_id"].tolist()),
            "stations_count": len(filtered),
        },
    )

    return filtered


def list_stations(master_shp, classification=None):
    """List stations in master shapefile.

    Parameters
    ----------
    master_shp : str or Path
        Path to master flux stations shapefile.
    classification : str, optional
        Filter to specific classification.

    Returns
    -------
    pandas.DataFrame
        Station information.
    """
    master = gpd.read_file(master_shp)

    if classification:
        cls_col = _get_classification_col(master)
        master = master[master[cls_col] == classification]

    # Drop geometry for display
    info = master.drop(columns=["geometry"])

    return info


def print_summary(master_shp):
    """Print summary of master shapefile."""
    master = gpd.read_file(master_shp)

    print(f"\n=== Master Flux Stations: {master_shp} ===")
    print(f"Total stations: {len(master)}")
    print(f"CRS: {master.crs}")
    print(f"\nColumns: {list(master.columns)}")

    cls_col = _get_classification_col(master)
    print(f"\nClassification counts (column: {cls_col}):")
    for cls, count in master[cls_col].value_counts().items():
        print(f"  {cls}: {count}")

    print("\nState counts:")
    for state, count in master["state"].value_counts().head(10).items():
        print(f"  {state}: {count}")
    if len(master["state"].unique()) > 10:
        print(f"  ... and {len(master['state'].unique()) - 10} more states")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Manage flux stations master shapefile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create master shapefile")
    create_parser.add_argument("--footprints", required=True, help="Footprints shapefile")
    create_parser.add_argument("--metadata", required=True, help="Metadata CSV")
    create_parser.add_argument("--output", required=True, help="Output shapefile path")
    create_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing")

    # List command
    list_parser = subparsers.add_parser("list", help="List stations")
    list_parser.add_argument("--master", required=True, help="Master shapefile")
    list_parser.add_argument("--classification", help="Filter by classification")
    list_parser.add_argument("--verbose", "-v", action="store_true", help="Show all columns")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract specific stations")
    extract_parser.add_argument("--master", required=True, help="Master shapefile")
    extract_parser.add_argument("--sites", nargs="+", required=True, help="Site IDs to extract")
    extract_parser.add_argument("--output", required=True, help="Output shapefile path")
    extract_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing")

    # Filter command
    filter_parser = subparsers.add_parser("filter", help="Filter by classification")
    filter_parser.add_argument("--master", required=True, help="Master shapefile")
    filter_parser.add_argument("--classification", required=True, help="Classification to filter")
    filter_parser.add_argument("--output", required=True, help="Output shapefile path")
    filter_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing")

    args = parser.parse_args()

    if args.command == "create":
        create_master_shapefile(
            args.footprints, args.metadata, args.output, overwrite=args.overwrite
        )

    elif args.command == "list":
        if args.verbose:
            info = list_stations(args.master, args.classification)
            print(info.to_string())
        else:
            print_summary(args.master)

    elif args.command == "extract":
        extract_stations(args.master, args.sites, args.output, overwrite=args.overwrite)

    elif args.command == "filter":
        filter_by_classification(
            args.master, args.classification, args.output, overwrite=args.overwrite
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
