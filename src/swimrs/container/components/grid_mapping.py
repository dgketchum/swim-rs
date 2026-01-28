"""Grid-to-field mapping utilities for coarse-resolution datasets.

This module provides a generalized pattern for handling gridded meteorological
data where multiple fields map to the same grid cell or weather station.

Supported use cases:
- GridMET (4km grid cells, GFID)
- ERA5-Land (9km grid cells)
- Weather station data (AgriMet, etc.)
- Any future gridded/point data sources
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


class GridMapping:
    """
    Manages mapping between field UIDs and grid cell/station IDs.

    This class handles the relationship where multiple fields may share
    the same grid cell or weather station for meteorological data.

    Example:
        # From a shapefile with UID and GFID columns
        mapping = GridMapping.from_shapefile(
            "fields_gridmet.shp",
            uid_column="FID",
            grid_column="GFID",
            source_name="gridmet"
        )

        # Get all fields sharing grid cell 12345
        fields = mapping.get_uids_for_grid(12345)

        # Get which grid cell a field belongs to
        gfid = mapping.get_grid_id("043_000130")
    """

    def __init__(
        self,
        uid_to_grid: dict[str, int | str],
        source_name: str = "grid",
    ):
        """
        Initialize GridMapping from a UID to grid ID dictionary.

        Args:
            uid_to_grid: Mapping of field UID -> grid/station ID
            source_name: Name of the grid source (for logging/display)
        """
        self.uid_to_grid = uid_to_grid
        self.source_name = source_name

        # Build reverse mapping: grid_id -> [uids]
        self.grid_to_uids: dict[int | str, list[str]] = defaultdict(list)
        for uid, grid_id in uid_to_grid.items():
            self.grid_to_uids[grid_id].append(uid)

    @classmethod
    def from_shapefile(
        cls,
        shapefile: str | Path,
        uid_column: str,
        grid_column: str,
        source_name: str = "grid",
    ) -> GridMapping:
        """
        Create mapping from a shapefile with UID and grid ID columns.

        Args:
            shapefile: Path to shapefile containing the mapping
            uid_column: Column name containing field UIDs
            grid_column: Column name containing grid/station IDs
            source_name: Name for logging/display

        Returns:
            GridMapping instance

        Raises:
            FileNotFoundError: If shapefile doesn't exist
            KeyError: If required columns are missing
        """
        import geopandas as gpd

        shapefile = Path(shapefile)
        if not shapefile.exists():
            raise FileNotFoundError(f"Shapefile not found: {shapefile}")

        gdf = gpd.read_file(shapefile)

        if uid_column not in gdf.columns:
            raise KeyError(
                f"UID column '{uid_column}' not found in shapefile. "
                f"Available columns: {list(gdf.columns)}"
            )
        if grid_column not in gdf.columns:
            raise KeyError(
                f"Grid column '{grid_column}' not found in shapefile. "
                f"Available columns: {list(gdf.columns)}"
            )

        uid_to_grid = {}
        for _, row in gdf.iterrows():
            uid = str(row[uid_column])
            grid_id = row[grid_column]
            if pd.notna(grid_id):
                # Convert float to int if it's a whole number
                if isinstance(grid_id, float) and grid_id.is_integer():
                    grid_id = int(grid_id)
                uid_to_grid[uid] = grid_id

        return cls(uid_to_grid, source_name)

    @classmethod
    def from_json(
        cls,
        json_path: str | Path,
        source_name: str = "grid",
    ) -> GridMapping:
        """
        Create mapping from a JSON file.

        Expected format: {"uid1": grid_id1, "uid2": grid_id2, ...}

        Args:
            json_path: Path to JSON file
            source_name: Name for logging/display

        Returns:
            GridMapping instance
        """
        import json

        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path) as f:
            uid_to_grid = json.load(f)

        return cls(uid_to_grid, source_name)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        uid_column: str,
        grid_column: str,
        source_name: str = "grid",
    ) -> GridMapping:
        """
        Create mapping from a pandas DataFrame.

        Args:
            df: DataFrame containing the mapping
            uid_column: Column name containing field UIDs
            grid_column: Column name containing grid/station IDs
            source_name: Name for logging/display

        Returns:
            GridMapping instance
        """
        uid_to_grid = {}
        for _, row in df.iterrows():
            uid = str(row[uid_column])
            grid_id = row[grid_column]
            if pd.notna(grid_id):
                if isinstance(grid_id, float) and grid_id.is_integer():
                    grid_id = int(grid_id)
                uid_to_grid[uid] = grid_id

        return cls(uid_to_grid, source_name)

    def get_grid_id(self, uid: str) -> int | str | None:
        """
        Get grid ID for a field UID.

        Args:
            uid: Field UID

        Returns:
            Grid/station ID, or None if not mapped
        """
        return self.uid_to_grid.get(uid)

    def get_uids_for_grid(self, grid_id: int | str) -> list[str]:
        """
        Get all field UIDs mapped to a grid cell.

        Args:
            grid_id: Grid cell or station ID

        Returns:
            List of field UIDs (empty if none mapped)
        """
        return self.grid_to_uids.get(grid_id, [])

    def filter_to_valid_uids(self, valid_uids: set[str]) -> GridMapping:
        """
        Return new mapping containing only the specified UIDs.

        Useful for filtering to only UIDs present in a container.

        Args:
            valid_uids: Set of UIDs to keep

        Returns:
            New GridMapping with filtered UIDs
        """
        filtered = {uid: gid for uid, gid in self.uid_to_grid.items() if uid in valid_uids}
        return GridMapping(filtered, self.source_name)

    @property
    def unique_grid_ids(self) -> list[int | str]:
        """List of unique grid/station IDs in the mapping."""
        return list(self.grid_to_uids.keys())

    @property
    def n_fields(self) -> int:
        """Number of fields in the mapping."""
        return len(self.uid_to_grid)

    @property
    def n_grid_cells(self) -> int:
        """Number of unique grid cells in the mapping."""
        return len(self.grid_to_uids)

    def __len__(self) -> int:
        """Return number of field mappings."""
        return len(self.uid_to_grid)

    def __contains__(self, uid: str) -> bool:
        """Check if a UID is in the mapping."""
        return uid in self.uid_to_grid

    def __repr__(self) -> str:
        return (
            f"GridMapping(source={self.source_name!r}, "
            f"fields={self.n_fields}, "
            f"grid_cells={self.n_grid_cells})"
        )
