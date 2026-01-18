"""HDF5-based input data container for SWIM-RS.

Provides portable data packaging for PEST++ worker distribution.
The SwimInput class can:
- Build HDF5 from prepped_input.json
- Load from existing HDF5 (for workers)
- Apply PEST++ multipliers from CSV files
- Provide lazy access to time series data
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd

from swimrs.process.state import (
    CalibrationParameters,
    FieldProperties,
    WaterBalanceState,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["SwimInput", "build_swim_input"]


@dataclass
class SwimInput:
    """HDF5-backed input data container for soil water balance modeling.

    This container provides all data needed for a SWIM-RS simulation run.
    It can be built from JSON input files and saved to HDF5 for distribution
    to PEST++ workers, or loaded directly from HDF5.

    Attributes
    ----------
    h5_path : Path
        Path to the HDF5 file
    start_date : datetime
        Simulation start date
    end_date : datetime
        Simulation end date
    n_days : int
        Number of simulation days
    n_fields : int
        Number of fields/pixels
    fids : list[str]
        Field identifiers
    runoff_process : str
        Runoff mode ('cn' Curve Number or 'ier' infiltration-excess)
    refet_type : str
        Reference ET type ('eto' or 'etr')
    properties : FieldProperties
        Static field properties
    parameters : CalibrationParameters
        Calibration parameters (base values)
    spinup_state : WaterBalanceState
        Initial state from spinup
    """

    h5_path: Path
    start_date: datetime = field(default=None)
    end_date: datetime = field(default=None)
    n_days: int = field(default=0)
    n_fields: int = field(default=0)
    fids: list = field(default_factory=list)
    runoff_process: str = field(default="cn")
    refet_type: str = field(default="eto")
    properties: FieldProperties = field(default=None)
    parameters: CalibrationParameters = field(default=None)
    spinup_state: WaterBalanceState = field(default=None)
    _h5_file: h5py.File = field(default=None, repr=False)
    _gwsub_years: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Open HDF5 file and load metadata if path exists."""
        if self.h5_path is not None and Path(self.h5_path).exists():
            self._load_from_h5()

    def _load_from_h5(self):
        """Load metadata and static arrays from HDF5."""
        self._h5_file = h5py.File(self.h5_path, "r")
        h5 = self._h5_file

        # Load config
        config = json.loads(h5.attrs["config"])
        self.start_date = datetime.fromisoformat(config["start_date"])
        self.end_date = datetime.fromisoformat(config["end_date"])
        self.runoff_process = config.get("runoff_process", "cn")
        self.refet_type = config.get("refet_type", "eto")

        # Load field info
        self.fids = [fid.decode() if isinstance(fid, bytes) else fid
                     for fid in h5["fields/fids"][:]]
        self.n_fields = len(self.fids)
        self.n_days = (self.end_date - self.start_date).days + 1

        # Load properties
        self.properties = self._load_properties(h5)

        # Load parameters
        self.parameters = self._load_parameters(h5)

        # Load spinup state
        self.spinup_state = self._load_spinup(h5)

        # Load year-specific groundwater subsidy data
        self._gwsub_years = self._load_gwsub_years(h5)

    def _load_gwsub_years(self, h5: h5py.File) -> dict[int, NDArray[np.float64]]:
        """Load year-specific f_sub data from HDF5.

        Returns
        -------
        dict[int, NDArray[np.float64]]
            Mapping of year -> f_sub array (n_fields,)
        """
        result = {}
        if "gwsub" not in h5:
            return result

        gwsub_group = h5["gwsub"]
        for year_str in gwsub_group.keys():
            try:
                year = int(year_str)
                result[year] = gwsub_group[year_str][:]
            except (ValueError, TypeError):
                continue

        return result

    def _load_properties(self, h5: h5py.File) -> FieldProperties:
        """Load field properties from HDF5."""
        props = h5["properties"]
        return FieldProperties(
            n_fields=self.n_fields,
            fids=np.array(self.fids),
            awc=props["awc"][:],
            ksat=props["ksat"][:],
            rew=props["rew"][:],
            tew=props["tew"][:],
            cn2=props["cn2"][:],
            zr_max=props["zr_max"][:],
            zr_min=props["zr_min"][:],
            p_depletion=props["p_depletion"][:],
            irr_status=props["irr_status"][:].astype(bool),
            perennial=props["perennial"][:].astype(bool),
            gw_status=props["gw_status"][:].astype(bool),
            ke_max=props["ke_max"][:] if "ke_max" in props else None,
            f_sub=props["f_sub"][:] if "f_sub" in props else None,
        )

    def _load_parameters(self, h5: h5py.File) -> CalibrationParameters:
        """Load calibration parameters from HDF5."""
        params = h5["parameters"]
        return CalibrationParameters(
            n_fields=self.n_fields,
            kc_max=params["kc_max"][:],
            kc_min=params["kc_min"][:],
            ndvi_k=params["ndvi_k"][:],
            ndvi_0=params["ndvi_0"][:],
            swe_alpha=params["swe_alpha"][:],
            swe_beta=params["swe_beta"][:],
            kr_damp=params["kr_damp"][:],
            ks_damp=params["ks_damp"][:],
            max_irr_rate=params["max_irr_rate"][:],
        )

    def _load_spinup(self, h5: h5py.File) -> WaterBalanceState:
        """Load spinup state from HDF5."""
        spinup = h5["spinup"]
        return WaterBalanceState.from_spinup(
            n_fields=self.n_fields,
            depl_root=spinup["depl_root"][:],
            swe=spinup["swe"][:],
            kr=spinup["kr"][:],
            ks=spinup["ks"][:],
            zr=spinup["zr"][:],
            daw3=spinup["daw3"][:] if "daw3" in spinup else None,
            taw3=spinup["taw3"][:] if "taw3" in spinup else None,
            depl_ze=spinup["depl_ze"][:] if "depl_ze" in spinup else None,
            albedo=spinup["albedo"][:] if "albedo" in spinup else None,
            s=spinup["s"][:] if "s" in spinup else None,
            s1=spinup["s1"][:] if "s1" in spinup else None,
            s2=spinup["s2"][:] if "s2" in spinup else None,
            s3=spinup["s3"][:] if "s3" in spinup else None,
            s4=spinup["s4"][:] if "s4" in spinup else None,
        )

    def close(self):
        """Close the HDF5 file."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_time_series(
        self,
        variable: str,
        day_idx: int | None = None,
    ) -> NDArray[np.float64]:
        """Get time series data for a variable.

        Parameters
        ----------
        variable : str
            Variable name (e.g., 'ndvi', 'prcp', 'tmin', 'tmax', 'srad', 'etr')
        day_idx : int, optional
            If provided, return only data for this day index.
            Otherwise return full time series (n_days, n_fields).

        Returns
        -------
        NDArray[np.float64]
            Time series data. Shape (n_fields,) if day_idx provided,
            otherwise (n_days, n_fields).
        """
        if self._h5_file is None:
            raise RuntimeError("HDF5 file not open")

        ts = self._h5_file["time_series"]
        if variable not in ts:
            raise KeyError(f"Variable '{variable}' not in time_series")

        if day_idx is not None:
            return ts[variable][day_idx, :]
        return ts[variable][:]

    def get_irr_flag(self, day_idx: int | None = None) -> NDArray[np.bool_]:
        """Get irrigation flag data.

        Parameters
        ----------
        day_idx : int, optional
            If provided, return only data for this day index.

        Returns
        -------
        NDArray[np.bool_]
            Irrigation flag array. Shape (n_fields,) if day_idx provided,
            otherwise (n_days, n_fields).
        """
        if self._h5_file is None:
            raise RuntimeError("HDF5 file not open")

        irr = self._h5_file["irrigation/irr_flag"]
        if day_idx is not None:
            return irr[day_idx, :].astype(bool)
        return irr[:].astype(bool)

    def has_hourly_precip(self) -> bool:
        """Check if hourly precipitation data is available.

        Returns
        -------
        bool
            True if prcp_hr dataset exists in time_series group
        """
        if self._h5_file is None:
            return False
        return "prcp_hr" in self._h5_file["time_series"]

    def get_hourly_precip(self, day_idx: int) -> NDArray[np.float64] | None:
        """Get hourly precipitation for a specific day.

        Parameters
        ----------
        day_idx : int
            Day index

        Returns
        -------
        NDArray[np.float64] | None
            Hourly precip array of shape (24, n_fields), transposed from
            storage format (n_fields, 24). Returns None if not available.
        """
        if self._h5_file is None:
            raise RuntimeError("HDF5 file not open")

        if "prcp_hr" not in self._h5_file["time_series"]:
            return None

        # Storage format is (n_days, n_fields, 24)
        # Return (24, n_fields) for infiltration_excess kernel
        prcp_hr = self._h5_file["time_series/prcp_hr"][day_idx, :, :]
        return prcp_hr.T  # Transpose to (24, n_fields)

    def get_date(self, day_idx: int) -> datetime:
        """Get date for a given day index."""
        from datetime import timedelta
        return self.start_date + timedelta(days=day_idx)

    def get_day_idx(self, date: datetime) -> int:
        """Get day index for a given date."""
        return (date - self.start_date).days

    def get_f_sub_for_year(self, year: int) -> NDArray[np.float64]:
        """Get year-specific groundwater subsidy fraction.

        Parameters
        ----------
        year : int
            The year for which to get f_sub values

        Returns
        -------
        NDArray[np.float64]
            f_sub array of shape (n_fields,) for the specified year.
            Falls back to static properties.f_sub if year-specific data
            is not available.
        """
        if self._gwsub_years and year in self._gwsub_years:
            return self._gwsub_years[year]
        # Fall back to static f_sub from properties
        if self.properties is not None and self.properties.f_sub is not None:
            return self.properties.f_sub
        return np.zeros(self.n_fields, dtype=np.float64)

    def has_year_specific_gwsub(self) -> bool:
        """Check if year-specific groundwater subsidy data is available."""
        return bool(self._gwsub_years)

    def apply_multipliers(
        self,
        mult_dir: Path | str,
    ) -> CalibrationParameters:
        """Apply PEST++ multipliers to create adjusted parameters.

        Parameters
        ----------
        mult_dir : Path | str
            Directory containing multiplier CSV files.
            Files should be named: p_{param}_{fid}_0_constant.csv

        Returns
        -------
        CalibrationParameters
            New parameters with multipliers applied
        """
        mult_dir = Path(mult_dir)
        if not mult_dir.exists():
            return self.parameters.copy()

        multipliers: dict[str, NDArray[np.float64]] = {}

        # Mapping from PEST param names to CalibrationParameters attributes
        param_map = {
            "aw": None,  # Handled separately as property
            "ndvi_k": "ndvi_k",
            "ndvi_0": "ndvi_0",
            "swe_alpha": "swe_alpha",
            "swe_beta": "swe_beta",
            "kr_alpha": "kr_damp",
            "ks_alpha": "ks_damp",
            "mad": None,  # Handled as p_depletion in properties
        }

        for mult_file in mult_dir.glob("p_*_constant.csv"):
            # Parse filename: p_{param}_{fid}_0_constant.csv
            parts = mult_file.stem.split("_")
            if len(parts) < 4:
                continue

            # Extract param name (may have underscores)
            # Find fid by looking for match in self.fids
            param_name = None
            fid = None
            for _i, potential_fid in enumerate(self.fids):
                # Check if filename contains this fid
                fid_idx = mult_file.stem.find(f"_{potential_fid}_")
                if fid_idx > 0:
                    param_name = mult_file.stem[2:fid_idx]  # Skip "p_"
                    fid = potential_fid
                    break

            if param_name is None or fid is None:
                continue

            # Read multiplier value (column 6, 0-indexed col 5)
            try:
                df = pd.read_csv(mult_file, header=0)
                mult_val = float(df.iloc[0, 5])
            except (IndexError, ValueError):
                continue

            # Map to parameter name
            attr_name = param_map.get(param_name)
            if attr_name is None:
                continue

            # Initialize array if needed
            if attr_name not in multipliers:
                multipliers[attr_name] = np.ones(self.n_fields, dtype=np.float64)

            # Find field index
            try:
                fid_idx = self.fids.index(fid)
                multipliers[attr_name][fid_idx] = mult_val
            except ValueError:
                continue

        return CalibrationParameters.from_base_with_multipliers(
            self.parameters, multipliers
        )


def build_swim_input(
    json_path: Path | str,
    output_h5: Path | str,
    spinup_state: dict[str, NDArray[np.float64]] | None = None,
    spinup_json_path: Path | str | None = None,
    calibrated_params_path: Path | str | None = None,
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
    runoff_process: str = "cn",
    refet_type: str = "eto",
) -> SwimInput:
    """Build HDF5 input file from prepped_input.json.

    Parameters
    ----------
    json_path : Path | str
        Path to prepped_input.json
    output_h5 : Path | str
        Path for output HDF5 file
    spinup_state : dict, optional
        Dictionary with spinup arrays: depl_root, swe, kr, ks, zr.
        If not provided, uses default initialization.
    spinup_json_path : Path | str, optional
        Path to spinup JSON file (like old model's spinup.json).
        Takes precedence over spinup_state if both provided.
    calibrated_params_path : Path | str, optional
        Path to JSON file with calibrated parameters per field.
        Format: {fid: {param_name: value, ...}, ...}
    start_date : str | datetime, optional
        Override start date (default: from time_series)
    end_date : str | datetime, optional
        Override end date (default: from time_series)
    runoff_process : str
        Runoff mode: 'cn' (curve number) or 'ier' (infiltration excess)
    refet_type : str
        Reference ET: 'eto' (grass) or 'etr' (alfalfa)

    Returns
    -------
    SwimInput
        Loaded SwimInput container
    """
    json_path = Path(json_path)
    output_h5 = Path(output_h5)

    with open(json_path) as f:
        data = json.load(f)

    # Get field order and count
    fids = data["order"]
    n_fields = len(fids)

    # Get date range from time_series
    ts_dates = sorted(data["time_series"].keys())
    if start_date is None:
        start_date = datetime.fromisoformat(ts_dates[0])
    elif isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date)

    if end_date is None:
        end_date = datetime.fromisoformat(ts_dates[-1])
    elif isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date)

    n_days = (end_date - start_date).days + 1

    # Create HDF5 file
    with h5py.File(output_h5, "w") as h5:
        # Config attributes
        config = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "runoff_process": runoff_process,
            "refet_type": refet_type,
        }
        h5.attrs["config"] = json.dumps(config)

        # Field IDs
        h5.create_group("fields")
        h5.create_dataset(
            "fields/fids",
            data=np.array(fids, dtype="S64"),
        )

        # Load calibrated parameters if provided
        calibrated_params = None
        if calibrated_params_path is not None:
            calibrated_params = _load_calibrated_params(calibrated_params_path, fids)

        # Properties (with optional calibrated aw/mad overrides)
        _write_properties(h5, data, fids, n_fields, calibrated_params)

        # Parameters
        _write_parameters(h5, data, fids, n_fields, calibrated_params)

        # Time series
        _write_time_series(h5, data, fids, n_fields, n_days, start_date, refet_type)

        # Irrigation flags
        _write_irrigation(h5, data, fids, n_fields, n_days, start_date)

        # Load spinup from JSON file if provided
        if spinup_json_path is not None:
            spinup_state = _load_spinup_json(spinup_json_path, fids)

        # Spinup state
        _write_spinup(h5, n_fields, spinup_state)

    # Return loaded container
    return SwimInput(h5_path=output_h5)


def _write_properties(
    h5: h5py.File,
    data: dict,
    fids: list[str],
    n_fields: int,
    calibrated_params: dict[str, NDArray[np.float64]] | None = None,
):
    """Write static field properties to HDF5.

    Parameters
    ----------
    calibrated_params : dict, optional
        If provided, uses calibrated 'aw' for awc, 'mad' for p_depletion,
        and 'f_sub' for groundwater subsidy fraction.

    Notes
    -----
    ke_max and f_sub are observation-derived properties (not tunable params):
    - ke_max: 90th percentile of ETf where NDVI < 0.3 (bare soil evaporation cap)
    - f_sub: derived from ETa/PPT ratio where f_sub = (ratio - 1) / ratio
    """
    props = h5.create_group("properties")

    # Extract from JSON props
    json_props = data["props"]

    # awc: use calibrated 'aw' if provided, otherwise compute from soil awc
    if calibrated_params is not None and "aw" in calibrated_params:
        awc = calibrated_params["aw"]
    else:
        awc = np.array([json_props[fid].get("awc", 0.15) * 1000 for fid in fids])  # mm/m

    ksat = np.array([json_props[fid].get("ksat", 10.0) for fid in fids])

    # Perennial status: derived from lulc_code (same logic as tracker.py)
    # crops = [12, 14], all other codes 1-17 are perennial
    crops_codes = {12, 14}
    def _get_perennial(fid):
        lulc_code = json_props[fid].get("lulc_code", 0)
        return lulc_code not in crops_codes and 1 <= lulc_code <= 17
    perennial = np.array([_get_perennial(fid) for fid in fids])

    # Apply zr_mult multiplier to root_depth if present
    zr_max = np.array([
        json_props[fid].get("root_depth", 1.0) * json_props[fid].get("zr_mult", 1.0)
        for fid in fids
    ])
    # For perennials, zr_min = zr_max (roots don't shrink); for crops, zr_min = 0.1
    zr_min = np.where(perennial, zr_max, 0.1)

    # Default values for properties not in JSON
    # Match legacy model defaults (tracker.py lines 99-100)
    rew = np.full(n_fields, 3.0)
    tew = np.full(n_fields, 18.0)

    # Derive curve number (CN2) from clay content (same as tracker.py)
    # Hydrologic group based on clay percentage:
    #   Coarse (A, clay < 15%): CN2 = 67
    #   Medium (B, 15-30%):     CN2 = 77
    #   Fine (C/D, > 30%):      CN2 = 85
    clay = np.array([json_props[fid].get("clay", 20.0) for fid in fids])
    cn2 = np.where(clay < 15.0, 67.0, np.where(clay > 30.0, 85.0, 77.0))

    # p_depletion: use calibrated 'mad' if provided, otherwise default
    if calibrated_params is not None and "mad" in calibrated_params:
        p_depletion = calibrated_params["mad"]
    else:
        p_depletion = np.full(n_fields, 0.5)

    # Irrigation status from props - can be a dict {year: fraction} or a single value
    def _get_irr_status(fid):
        irr = json_props[fid].get("irr", 0)
        if isinstance(irr, dict):
            # If any year has irrigation fraction > 0, consider irrigated
            return any(v > 0 for v in irr.values())
        return irr > 0
    irr_status = np.array([_get_irr_status(fid) for fid in fids])

    # Groundwater status from gwsub_data
    gw_data = data.get("gwsub_data", {})
    gw_status = np.array([fid in gw_data and bool(gw_data[fid]) for fid in fids])

    # ke_max: derived from observations (90th percentile ETf where NDVI < 0.3)
    # This is a property, not a calibration parameter
    json_ke_max = data.get("ke_max", {})
    ke_max = np.array([json_ke_max.get(fid, 1.0) for fid in fids])

    # f_sub: derived from ETa/PPT ratio, use calibrated if provided
    # f_sub = (ratio - 1) / ratio where ratio > 1
    if calibrated_params is not None and "f_sub" in calibrated_params:
        f_sub = calibrated_params["f_sub"]
    else:
        # Default: derive from gwsub_data if available
        # gwsub_data structure: {fid: {year: {"f_sub": float, ...}}} or {fid: bool}
        gwsub = data.get("gwsub_data", {})
        f_sub_values = []
        for fid in fids:
            fid_gw = gwsub.get(fid, {})
            if isinstance(fid_gw, dict) and fid_gw:
                # Average f_sub across years for this field
                yearly_fsub = [yr_data.get("f_sub", 0.0) for yr_data in fid_gw.values()
                               if isinstance(yr_data, dict)]
                if yearly_fsub:
                    f_sub_values.append(np.mean(yearly_fsub))
                else:
                    f_sub_values.append(0.0)
            else:
                f_sub_values.append(0.0)
        f_sub = np.array(f_sub_values)

    props.create_dataset("awc", data=awc)
    props.create_dataset("ksat", data=ksat)
    props.create_dataset("rew", data=rew)
    props.create_dataset("tew", data=tew)
    props.create_dataset("cn2", data=cn2)
    props.create_dataset("zr_max", data=zr_max)
    props.create_dataset("zr_min", data=zr_min)
    props.create_dataset("p_depletion", data=p_depletion)
    props.create_dataset("irr_status", data=irr_status.astype(np.uint8))
    props.create_dataset("perennial", data=perennial.astype(np.uint8))
    props.create_dataset("gw_status", data=gw_status.astype(np.uint8))
    props.create_dataset("ke_max", data=ke_max)
    props.create_dataset("f_sub", data=f_sub)

    # Write year-specific groundwater subsidy data
    _write_gwsub_years(h5, data, fids, n_fields, calibrated_params)


def _write_gwsub_years(
    h5: h5py.File,
    data: dict,
    fids: list[str],
    n_fields: int,
    calibrated_params: dict[str, NDArray[np.float64]] | None = None,
):
    """Write year-specific groundwater subsidy (f_sub) data to HDF5.

    Creates a /gwsub group with one dataset per year containing f_sub values.
    This enables year-specific groundwater subsidy handling in the simulation.

    Parameters
    ----------
    h5 : h5py.File
        Open HDF5 file handle
    data : dict
        JSON input data containing gwsub_data
    fids : list[str]
        Field IDs in order
    n_fields : int
        Number of fields
    calibrated_params : dict, optional
        Calibrated parameters (not used here, for consistency)
    """
    gwsub_data = data.get("gwsub_data", {})
    if not gwsub_data:
        return

    # Collect all years from gwsub_data
    all_years = set()
    for fid in fids:
        fid_gw = gwsub_data.get(fid, {})
        if isinstance(fid_gw, dict):
            for year_str in fid_gw.keys():
                try:
                    all_years.add(int(year_str))
                except ValueError:
                    continue

    if not all_years:
        return

    gwsub_group = h5.create_group("gwsub")

    for year in sorted(all_years):
        year_str = str(year)
        f_sub_year = np.zeros(n_fields, dtype=np.float64)

        for i, fid in enumerate(fids):
            fid_gw = gwsub_data.get(fid, {})
            if isinstance(fid_gw, dict) and year_str in fid_gw:
                year_data = fid_gw[year_str]
                if isinstance(year_data, dict):
                    f_sub_year[i] = year_data.get("f_sub", 0.0)

        gwsub_group.create_dataset(year_str, data=f_sub_year)


def _write_parameters(
    h5: h5py.File,
    data: dict,
    fids: list[str],
    n_fields: int,
    calibrated_params: dict[str, NDArray[np.float64]] | None = None,
):
    """Write calibration parameters to HDF5.

    Parameters
    ----------
    calibrated_params : dict, optional
        Dictionary with parameter arrays from calibration file.
        Keys: 'aw', 'ndvi_k', 'ndvi_0', 'ks_damp', 'kr_damp', 'mad',
              'swe_alpha', 'swe_beta'
    """
    params_group = h5.create_group("parameters")

    # kc_max for model is typically 1.0-1.25, not the observational max ETf in JSON
    kc_max = np.full(n_fields, 1.0)
    kc_min = np.full(n_fields, 0.15)

    # Default calibration parameter values
    ndvi_k = np.full(n_fields, 7.0)
    ndvi_0 = np.full(n_fields, 0.4)
    swe_alpha = np.full(n_fields, 0.5)
    swe_beta = np.full(n_fields, 2.0)
    kr_damp = np.full(n_fields, 0.2)
    ks_damp = np.full(n_fields, 0.2)
    max_irr_rate = np.full(n_fields, 25.0)

    # Override with calibrated values if provided
    if calibrated_params is not None:
        if "ndvi_k" in calibrated_params:
            ndvi_k = calibrated_params["ndvi_k"]
        if "ndvi_0" in calibrated_params:
            ndvi_0 = calibrated_params["ndvi_0"]
        if "swe_alpha" in calibrated_params:
            swe_alpha = calibrated_params["swe_alpha"]
        if "swe_beta" in calibrated_params:
            swe_beta = calibrated_params["swe_beta"]
        if "kr_damp" in calibrated_params:
            kr_damp = calibrated_params["kr_damp"]
        if "ks_damp" in calibrated_params:
            ks_damp = calibrated_params["ks_damp"]

    params_group.create_dataset("kc_max", data=kc_max)
    params_group.create_dataset("kc_min", data=kc_min)
    params_group.create_dataset("ndvi_k", data=ndvi_k)
    params_group.create_dataset("ndvi_0", data=ndvi_0)
    params_group.create_dataset("swe_alpha", data=swe_alpha)
    params_group.create_dataset("swe_beta", data=swe_beta)
    params_group.create_dataset("kr_damp", data=kr_damp)
    params_group.create_dataset("ks_damp", data=ks_damp)
    params_group.create_dataset("max_irr_rate", data=max_irr_rate)


def _write_time_series(
    h5: h5py.File,
    data: dict,
    fids: list[str],
    n_fields: int,
    n_days: int,
    start_date: datetime,
    refet_type: str,
):
    """Write time series data to HDF5."""
    ts_group = h5.create_group("time_series")
    ts_data = data["time_series"]
    json_props = data["props"]

    # Build year-specific irrigation status for NDVI selection
    # Returns dict: {year: np.array of bool for each field}
    def _build_year_irr_status():
        result = {}
        # Determine years from start/end date
        years = list(range(start_date.year, start_date.year + (n_days // 365) + 2))
        for year in years:
            year_status = []
            for fid in fids:
                irr = json_props[fid].get("irr", 0)
                if isinstance(irr, dict):
                    # Get irrigation fraction for this specific year
                    frac = irr.get(str(year), 0)
                    year_status.append(frac > 0)
                else:
                    year_status.append(irr > 0)
            result[year] = np.array(year_status)
        return result

    year_irr_status = _build_year_irr_status()

    # Variables to extract (simple ones without irr/non-irr variants)
    variables = {
        "prcp": "prcp",
        "tmin": "tmin",
        "tmax": "tmax",
        "srad": "srad",
        "swe_obs": "swe",  # Observed SWE (e.g., SNODAS)
    }

    # Add reference ET based on type
    if refet_type == "eto":
        variables["etr"] = "eto"  # Use grass reference
    else:
        variables["etr"] = "etr"  # Use alfalfa reference

    for out_name, json_name in variables.items():
        arr = np.zeros((n_days, n_fields), dtype=np.float64)

        for day_idx in range(n_days):
            date = start_date + pd.Timedelta(days=day_idx)
            date_str = date.strftime("%Y-%m-%d")

            if date_str in ts_data and json_name in ts_data[date_str]:
                values = ts_data[date_str][json_name]
                if len(values) >= n_fields:
                    arr[day_idx, :] = values[:n_fields]
                else:
                    arr[day_idx, : len(values)] = values

        ts_group.create_dataset(out_name, data=arr, compression="gzip")

    # Handle NDVI - check for single 'ndvi' field or separate irr/inv_irr fields
    ndvi_arr = np.zeros((n_days, n_fields), dtype=np.float64)

    # Check which NDVI format is used (sample first few dates)
    sample_dates = [d for d in sorted(ts_data.keys())[:10]]
    has_single_ndvi = any("ndvi" in ts_data.get(d, {}) for d in sample_dates)
    has_split_ndvi = any("ndvi_irr" in ts_data.get(d, {}) for d in sample_dates)

    for day_idx in range(n_days):
        date = start_date + pd.Timedelta(days=day_idx)
        date_str = date.strftime("%Y-%m-%d")

        if date_str in ts_data:
            day_data = ts_data[date_str]

            if has_single_ndvi and "ndvi" in day_data:
                # Single ndvi field (e.g., S2 fixture format)
                ndvi_vals = day_data["ndvi"]
                if len(ndvi_vals) >= n_fields:
                    ndvi_arr[day_idx, :] = ndvi_vals[:n_fields]
                else:
                    ndvi_arr[day_idx, : len(ndvi_vals)] = ndvi_vals
            elif has_split_ndvi:
                # Separate ndvi_irr / ndvi_inv_irr fields (Fort Peck format)
                ndvi_irr = day_data.get("ndvi_irr", [np.nan] * n_fields)
                ndvi_inv = day_data.get("ndvi_inv_irr", [np.nan] * n_fields)

                # Get year-specific irrigation status
                year = date.year
                irr_status_year = year_irr_status.get(year, np.zeros(n_fields, dtype=bool))

                for fid_idx in range(n_fields):
                    if irr_status_year[fid_idx]:
                        # Irrigated field this year - use ndvi_irr
                        if fid_idx < len(ndvi_irr):
                            ndvi_arr[day_idx, fid_idx] = ndvi_irr[fid_idx]
                    else:
                        # Non-irrigated field this year - use ndvi_inv_irr
                        if fid_idx < len(ndvi_inv):
                            ndvi_arr[day_idx, fid_idx] = ndvi_inv[fid_idx]

    ts_group.create_dataset("ndvi", data=ndvi_arr, compression="gzip")

    # Add hourly precip for IER mode
    if any("prcp_hr_00" in ts_data.get(d, {}) for d in list(ts_data.keys())[:10]):
        prcp_hr = np.zeros((n_days, n_fields, 24), dtype=np.float64)

        for day_idx in range(n_days):
            date = start_date + pd.Timedelta(days=day_idx)
            date_str = date.strftime("%Y-%m-%d")

            if date_str in ts_data:
                for hr in range(24):
                    hr_key = f"prcp_hr_{hr:02d}"
                    if hr_key in ts_data[date_str]:
                        values = ts_data[date_str][hr_key]
                        if len(values) >= n_fields:
                            prcp_hr[day_idx, :, hr] = values[:n_fields]
                        else:
                            prcp_hr[day_idx, : len(values), hr] = values

        ts_group.create_dataset("prcp_hr", data=prcp_hr, compression="gzip")


def _write_irrigation(
    h5: h5py.File,
    data: dict,
    fids: list[str],
    n_fields: int,
    n_days: int,
    start_date: datetime,
):
    """Write irrigation flag array to HDF5."""
    irr_group = h5.create_group("irrigation")
    irr_data = data.get("irr_data", {})

    # Build daily irrigation flag array
    irr_flag = np.zeros((n_days, n_fields), dtype=np.uint8)

    for fid_idx, fid in enumerate(fids):
        if fid not in irr_data:
            continue

        fid_irr = irr_data[fid]
        for year_str, year_data in fid_irr.items():
            if not isinstance(year_data, dict):
                continue

            irr_doys = year_data.get("irr_doys", [])
            if not irr_doys:
                continue

            year = int(year_str)

            for doy in irr_doys:
                # Convert DOY to date
                try:
                    date = datetime(year, 1, 1) + pd.Timedelta(days=doy - 1)
                    day_idx = (date - start_date).days
                    if 0 <= day_idx < n_days:
                        irr_flag[day_idx, fid_idx] = 1
                except (ValueError, OverflowError):
                    continue

    irr_group.create_dataset("irr_flag", data=irr_flag, compression="gzip")


def _load_spinup_json(
    spinup_path: Path | str,
    fids: list[str],
) -> dict[str, NDArray[np.float64]]:
    """Load spinup state from JSON file (like old model's spinup.json).

    Parameters
    ----------
    spinup_path : Path | str
        Path to spinup JSON file
    fids : list[str]
        Field IDs in order

    Returns
    -------
    dict
        Dictionary with spinup arrays: depl_root, swe, kr, ks, zr, etc.
    """
    spinup_path = Path(spinup_path)
    n_fields = len(fids)

    with open(spinup_path) as f:
        spinup_data = json.load(f)

    # Initialize arrays with defaults
    # Default S value from CN2=75 => S=84.7 mm
    default_s = 84.7
    spinup_state = {
        "depl_root": np.zeros(n_fields),
        "swe": np.zeros(n_fields),
        "kr": np.ones(n_fields),
        "ks": np.ones(n_fields),
        "zr": np.full(n_fields, 0.1),
        "daw3": np.zeros(n_fields),
        "taw3": np.zeros(n_fields),
        "depl_ze": np.zeros(n_fields),
        "albedo": np.full(n_fields, 0.45),
        "s": np.full(n_fields, default_s),
        "s1": np.full(n_fields, default_s),
        "s2": np.full(n_fields, default_s),
        "s3": np.full(n_fields, default_s),
        "s4": np.full(n_fields, default_s),
    }

    # Map spinup JSON keys to our state keys
    key_map = {
        "depl_root": "depl_root",
        "swe": "swe",
        "kr": "kr",
        "ks": "ks",
        "zr": "zr",
        "daw3": "daw3",
        "taw3": "taw3",
        "depl_ze": "depl_ze",
        "albedo": "albedo",
        "s": "s",
        "s1": "s1",
        "s2": "s2",
        "s3": "s3",
        "s4": "s4",
    }

    # Fill arrays from spinup JSON
    for fid_idx, fid in enumerate(fids):
        if fid in spinup_data:
            field_spinup = spinup_data[fid]
            for json_key, state_key in key_map.items():
                if json_key in field_spinup:
                    spinup_state[state_key][fid_idx] = field_spinup[json_key]

    return spinup_state


def _write_spinup(
    h5: h5py.File,
    n_fields: int,
    spinup_state: dict[str, NDArray[np.float64]] | None,
):
    """Write spinup state to HDF5."""
    spinup = h5.create_group("spinup")

    if spinup_state is not None:
        spinup.create_dataset("depl_root", data=spinup_state.get(
            "depl_root", np.zeros(n_fields)))
        spinup.create_dataset("swe", data=spinup_state.get(
            "swe", np.zeros(n_fields)))
        spinup.create_dataset("kr", data=spinup_state.get(
            "kr", np.ones(n_fields)))
        spinup.create_dataset("ks", data=spinup_state.get(
            "ks", np.ones(n_fields)))
        spinup.create_dataset("zr", data=spinup_state.get(
            "zr", np.full(n_fields, 0.1)))

        # Optional spinup arrays
        if "daw3" in spinup_state:
            spinup.create_dataset("daw3", data=spinup_state["daw3"])
        if "taw3" in spinup_state:
            spinup.create_dataset("taw3", data=spinup_state["taw3"])
        if "depl_ze" in spinup_state:
            spinup.create_dataset("depl_ze", data=spinup_state["depl_ze"])
        if "albedo" in spinup_state:
            spinup.create_dataset("albedo", data=spinup_state["albedo"])
        # S history for smoothed CN runoff
        if "s" in spinup_state:
            spinup.create_dataset("s", data=spinup_state["s"])
        if "s1" in spinup_state:
            spinup.create_dataset("s1", data=spinup_state["s1"])
        if "s2" in spinup_state:
            spinup.create_dataset("s2", data=spinup_state["s2"])
        if "s3" in spinup_state:
            spinup.create_dataset("s3", data=spinup_state["s3"])
        if "s4" in spinup_state:
            spinup.create_dataset("s4", data=spinup_state["s4"])
    else:
        # Default initialization
        spinup.create_dataset("depl_root", data=np.zeros(n_fields))
        spinup.create_dataset("swe", data=np.zeros(n_fields))
        spinup.create_dataset("kr", data=np.ones(n_fields))
        spinup.create_dataset("ks", data=np.ones(n_fields))
        spinup.create_dataset("zr", data=np.full(n_fields, 0.1))


def _load_calibrated_params(
    params_path: Path | str,
    fids: list[str],
) -> dict[str, NDArray[np.float64]]:
    """Load calibrated parameters from JSON file.

    Parameters
    ----------
    params_path : Path | str
        Path to calibrated parameters JSON file.
        Format: {fid: {param_name: value, ...}, ...}
    fids : list[str]
        Field IDs in order

    Returns
    -------
    dict
        Dictionary with parameter arrays indexed by parameter name.
        Keys: ndvi_k, ndvi_0, ks_damp, kr_damp, swe_alpha, swe_beta, aw, mad
    """
    params_path = Path(params_path)
    n_fields = len(fids)

    with open(params_path) as f:
        params_data = json.load(f)

    # Map from calibration file names to internal names
    name_map = {
        "ks_alpha": "ks_damp",
        "kr_alpha": "kr_damp",
        "ndvi_k": "ndvi_k",
        "ndvi_0": "ndvi_0",
        "swe_alpha": "swe_alpha",
        "swe_beta": "swe_beta",
        "aw": "aw",
        "mad": "mad",
        "f_sub": "f_sub",
    }

    # Initialize result with empty arrays
    result: dict[str, NDArray[np.float64]] = {}
    for internal_name in name_map.values():
        result[internal_name] = np.zeros(n_fields, dtype=np.float64)

    # Fill arrays from calibration data
    for fid_idx, fid in enumerate(fids):
        if fid not in params_data:
            continue

        field_params = params_data[fid]
        for file_name, internal_name in name_map.items():
            if file_name in field_params:
                result[internal_name][fid_idx] = field_params[file_name]

    return result
