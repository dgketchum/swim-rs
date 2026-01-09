"""
Query component for data access and status queries.

Provides a clean API for querying container data and status.
Usage: container.query.status() instead of container.status()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd

from .base import Component

if TYPE_CHECKING:
    import xarray as xr
    from swimrs.container.state import ContainerState
    from swimrs.container.inventory import FieldValidationReport, ValidationResult


class Query(Component):
    """
    Component for querying container data and status.

    Provides methods for accessing data in various formats (xarray, pandas,
    geopandas), checking container status, and validating data completeness.

    Example:
        container.query.status()
        df = container.query.dataframe("remote_sensing/ndvi/landsat/irr")
        gdf = container.query.geodataframe()
    """

    def __init__(self, state: "ContainerState", container=None):
        """
        Initialize the Query component.

        Args:
            state: ContainerState instance
            container: Optional reference to parent SwimContainer
        """
        super().__init__(state)
        self._container = container

    def status(self, detailed: bool = False) -> str:
        """
        Get formatted status report of container contents.

        Shows data coverage, completeness, and readiness for operations.

        Args:
            detailed: If True, show detailed per-field statistics

        Returns:
            Formatted status string
        """
        import numpy as np

        lines = []
        lines.append("=" * 60)
        lines.append("CONTAINER STATUS")
        lines.append("=" * 60)

        # Basic metadata
        lines.append(f"  URI: {self._state._provider.uri}")
        lines.append(f"  Fields: {self._state.n_fields}")
        time_idx = self._state._time_index
        if time_idx is not None and len(time_idx) > 0:
            lines.append(f"  Date range: {time_idx[0].date()} to {time_idx[-1].date()}")
        lines.append(f"  Days: {self._state.n_days}")
        lines.append("")

        # List all paths in the zarr
        lines.append("DATA PATHS:")
        lines.append("-" * 40)

        def _walk_group(group, prefix=""):
            """Recursively list paths in zarr group."""
            paths = []
            try:
                for key in sorted(group.keys()):
                    child = group[key]
                    full_path = f"{prefix}{key}" if prefix else key
                    import zarr
                    if isinstance(child, zarr.Array):
                        # Get array stats
                        shape = child.shape
                        dtype = child.dtype
                        try:
                            arr = child[:]
                            if np.issubdtype(arr.dtype, np.floating):
                                valid_count = np.sum(~np.isnan(arr))
                                total = arr.size
                                pct = 100.0 * valid_count / total if total > 0 else 0
                                stats = f"shape={shape}, {pct:.1f}% valid"
                            elif arr.dtype.kind in ('U', 'O') or 'str' in str(dtype).lower():
                                # String array - count non-empty
                                non_empty = sum(1 for v in arr.flat if v and str(v).strip())
                                total = arr.size
                                pct = 100.0 * non_empty / total if total > 0 else 0
                                stats = f"shape={shape}, {pct:.1f}% non-empty"
                            else:
                                stats = f"shape={shape}, dtype={dtype}"
                        except Exception:
                            stats = f"shape={shape}"
                        paths.append((full_path, stats))
                    else:
                        paths.extend(_walk_group(child, f"{full_path}/"))
            except Exception:
                pass
            return paths

        paths = _walk_group(self._state.root)

        # Group by top-level category
        categories = {}
        for path, stats in paths:
            parts = path.split("/")
            cat = parts[0]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((path, stats))

        for cat in sorted(categories.keys()):
            lines.append(f"\n  {cat}/")
            for path, stats in sorted(categories[cat]):
                subpath = path[len(cat) + 1:] if "/" in path else path
                lines.append(f"    {subpath}: {stats}")

        # Check key meteorology paths
        lines.append("")
        lines.append("METEOROLOGY CHECK:")
        lines.append("-" * 40)
        met_paths = [
            ("meteorology/gridmet/eto", "GridMET ETo"),
            ("meteorology/gridmet/prcp", "GridMET Precip"),
            ("meteorology/era5/eto", "ERA5 ETo"),
            ("meteorology/era5/prcp", "ERA5 Precip"),
            ("meteorology/era5/precip", "ERA5 Precip (alt)"),
            ("meteorology/snodas/swe", "SNODAS SWE"),
            ("meteorology/era5/swe", "ERA5 SWE"),
        ]
        for path, label in met_paths:
            try:
                arr = self._state.root[path]
                valid = np.sum(~np.isnan(arr[:])) if np.issubdtype(arr.dtype, np.floating) else arr.size
                total = arr.size
                pct = 100.0 * valid / total if total > 0 else 0
                lines.append(f"  ✓ {label}: {arr.shape}, {pct:.1f}% valid")
            except KeyError:
                lines.append(f"  ✗ {label}: NOT FOUND")

        if detailed:
            lines.append("")
            lines.append("PROVENANCE LOG:")
            lines.append("-" * 40)
            lines.append(self._state.provenance.summary())

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def validate(
        self,
        operation: str = "calibration",
        model: str = "ssebop",
        mask: str = "irr",
        met_source: str = "gridmet",
        snow_source: str = "snodas",
        instrument: str = "landsat",
    ) -> "ValidationResult":
        """
        Validate container readiness for an operation.

        Args:
            operation: "calibration" or "forward_run"
            model: ET model to check
            mask: Mask type to check
            met_source: Meteorology source to check
            snow_source: Snow source to check
            instrument: NDVI instrument to check

        Returns:
            ValidationResult with readiness status and details
        """
        raise NotImplementedError(
            "Validation not yet implemented. "
            "Implement logic to validate container readiness for operations."
        )

    def validate_fields(
        self,
        min_area_m2: float = 0,
        require_awc: bool = True,
        require_ksat: bool = False,
        require_lulc: bool = True,
        require_ndvi: bool = True,
        require_etf: bool = True,
        require_meteorology: bool = True,
        min_ndvi_obs: int = 10,
        min_etf_obs: int = 5,
        etf_model: str = "ssebop",
        mask: str = "irr",
        instrument: str = "landsat",
        met_source: str = "gridmet",
    ) -> "FieldValidationReport":
        """
        Validate individual fields against criteria.

        Args:
            min_area_m2: Minimum field area
            require_awc: Require available water capacity
            require_ksat: Require saturated hydraulic conductivity
            require_lulc: Require land use/land cover
            require_ndvi: Require NDVI observations
            require_etf: Require ETf observations
            require_meteorology: Require meteorology data
            min_ndvi_obs: Minimum NDVI observation count
            min_etf_obs: Minimum ETf observation count
            etf_model: ET model for ETf validation
            mask: Mask type for validation
            instrument: NDVI instrument for validation
            met_source: Meteorology source for validation

        Returns:
            FieldValidationReport with per-field validation results
        """
        raise NotImplementedError(
            "Field validation not yet implemented. "
            "Implement logic to validate fields against data requirements."
        )

    def valid_fields(self, **kwargs) -> List[str]:
        """
        Get list of valid field UIDs based on validation criteria.

        Args:
            **kwargs: Arguments passed to validate_fields()

        Returns:
            List of valid field UIDs
        """
        report = self.validate_fields(**kwargs)
        return report.valid_fields

    def xarray(
        self,
        path: str,
        fields: Optional[List[str]] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        name: Optional[str] = None,
    ) -> "xr.DataArray":
        """
        Get data as labeled xarray DataArray.

        Args:
            path: Zarr path (e.g., "remote_sensing/ndvi/landsat/irr")
            fields: Optional field UIDs to include
            start_date: Optional start date
            end_date: Optional end date
            name: Optional name for DataArray

        Returns:
            xr.DataArray with 'time' and 'site' coordinates
        """
        return self._state.get_xarray(
            path, fields=fields, start_date=start_date, end_date=end_date, name=name
        )

    def dataset(
        self,
        paths: Optional[Dict[str, str]] = None,
        fields: Optional[List[str]] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> "xr.Dataset":
        """
        Get multiple variables as xarray Dataset.

        Args:
            paths: Mapping of var names to zarr paths
            fields: Optional field UIDs to include
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            xr.Dataset with requested variables
        """
        return self._state.get_dataset(
            paths=paths, fields=fields, start_date=start_date, end_date=end_date
        )

    def dataframe(
        self,
        path: str,
        fields: Optional[List[str]] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """
        Get data as pandas DataFrame.

        Args:
            path: Zarr path
            fields: Optional field UIDs
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with DatetimeIndex and field columns
        """
        da = self.xarray(path, fields=fields, start_date=start_date, end_date=end_date)
        return da.to_pandas()

    def geodataframe(self):
        """
        Get field geometries as GeoDataFrame.

        Returns:
            gpd.GeoDataFrame with field geometries and properties
        """
        raise NotImplementedError(
            "GeoDataFrame query not yet implemented. "
            "Implement logic to reconstruct GeoDataFrame from zarr geometry storage."
        )

    def field_timeseries(
        self,
        uid: str,
        parameters: List[str] = None,
    ) -> pd.DataFrame:
        """
        Get all time series for a single field.

        Args:
            uid: Field UID
            parameters: Optional list of parameters to include

        Returns:
            DataFrame with DatetimeIndex and parameter columns
        """
        raise NotImplementedError(
            "Field timeseries query not yet implemented. "
            "Implement logic to assemble all time series for a single field."
        )

    def dynamics(self, uid: str) -> Dict[str, Any]:
        """
        Get computed dynamics for a field.

        Args:
            uid: Field UID

        Returns:
            Dict with ke_max, kc_max, irr (per-year), gwsub (per-year)
        """
        raise NotImplementedError(
            "Dynamics query not yet implemented. "
            "Implement logic to retrieve computed dynamics from zarr."
        )

    def inventory(self):
        """
        Get the container inventory tracker.

        Returns:
            Inventory instance for coverage analysis
        """
        return self._state.inventory
