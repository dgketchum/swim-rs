"""Centralized unit documentation for SWIM-RS.

This module is intentionally lightweight: it provides a *single place* to see
what units SWIM-RS expects internally, plus a small registry of common external
sources (notably Google Earth Engine datasets) and how we convert them.

Why this exists
---------------
SWIM-RS pulls meteorology/snow/remote-sensing data from multiple sources
(GridMET, ERA5-Land via Earth Engine, SNODAS, etc.). Source datasets often use
different units (e.g., Kelvin vs Celsius; meters vs millimeters; energy vs
power), and conversions were historically scattered across extraction/ingest
code. This file is the place to document:

- Canonical/internal units the *process model* expects
- Source dataset/band native units and the conversion performed

The intent is that code performing conversions references this registry so the
unit choices are auditable and consistent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UnitSpec:
    """Document a variable's units and conversion.

    Fields are documentation-first (strings), so this module stays dependency-
    free (no Earth Engine types, no pint).
    """

    native_units: str
    canonical_units: str
    conversion: str
    notes: str = ""
    reference: str = ""


# -----------------------------------------------------------------------------
# Canonical units used by the process model (SwimInput + kernels)
# -----------------------------------------------------------------------------

PROCESS_CANONICAL_UNITS: dict[str, str] = {
    # Meteorology time series
    "tmin": "C",
    "tmax": "C",
    "prcp": "mm/day",  # daily total
    "prcp_hr": "mm/hr",  # hourly intensity, 24 values per day
    "etr": "mm/day",  # reference ET used by model; see SwimInput.config['refet_type']
    "srad": "W/m^2",  # daily mean downward shortwave radiation
    # Snow
    "swe": "mm",
    # Remote sensing
    "ndvi": "unitless",
    "etf": "unitless",
    # Properties
    "awc": "mm/m",
    "ksat": "mm/day",  # converted to mm/hr internally for IER runoff
}


# -----------------------------------------------------------------------------
# Google Earth Engine (GEE) dataset/band unit documentation
# -----------------------------------------------------------------------------

# NOTE: Network access is restricted in many dev environments. Keep the catalog
# references as URLs so they can be verified externally.

GEE_ERA5_LAND_HOURLY_DATASET = "ECMWF/ERA5_LAND/HOURLY"
GEE_ERA5_LAND_HOURLY_CATALOG = (
    "https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY"
)

ERA5_LAND_HOURLY_UNITS: dict[str, UnitSpec] = {
    # Used in src/swimrs/data_extraction/ee/ee_era5.py
    "temperature_2m": UnitSpec(
        native_units="K",
        canonical_units="C",
        conversion="C = K - 273.15",
        reference=GEE_ERA5_LAND_HOURLY_CATALOG,
    ),
    "total_precipitation_hourly": UnitSpec(
        native_units="m",
        canonical_units="mm/hr",
        conversion="mm = m * 1000",
        notes="Hourly accumulation depth; daily total is sum(hourly_mm).",
        reference=GEE_ERA5_LAND_HOURLY_CATALOG,
    ),
    "surface_solar_radiation_downwards_hourly": UnitSpec(
        native_units="J/m^2",
        canonical_units="W/m^2 (daily mean)",
        conversion="W/m^2 = (sum_hourly_J_per_m2 / 86400)",
        notes="We store daily mean flux (power) rather than daily energy.",
        reference=GEE_ERA5_LAND_HOURLY_CATALOG,
    ),
    "snow_depth_water_equivalent": UnitSpec(
        native_units="m",
        canonical_units="mm",
        conversion="mm = m * 1000",
        reference=GEE_ERA5_LAND_HOURLY_CATALOG,
    ),
}


GEE_SNODAS_DAILY_DATASET = (
    "projects/earthengine-legacy/assets/projects/climate-engine/snodas/daily"
)

SNODAS_DAILY_UNITS: dict[str, UnitSpec] = {
    # Used in src/swimrs/data_extraction/ee/snodas_export.py and
    # src/swimrs/container/components/ingestor.py::_load_snodas_extracts
    "SWE": UnitSpec(
        native_units="m",
        canonical_units="mm",
        conversion="mm = m * 1000",
        notes="This is the convention assumed by SWIM-RS SNODAS extract handling.",
    )
}

