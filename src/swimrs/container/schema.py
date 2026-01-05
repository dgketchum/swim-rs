"""
Schema definitions for SWIM data container.

Defines the structure of data that can be stored in a SwimContainer,
including instruments, parameters, masks, and models.

Also provides centralized constants migrated from swimrs.prep, including:
- Rooting depth tables by LULC code
- Required meteorology parameters
- Unit mappings
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple


class Instrument(str, Enum):
    """Remote sensing instruments."""
    LANDSAT = "landsat"
    SENTINEL = "sentinel"
    ECOSTRESS = "ecostress"
    COMBINED = "combined"  # Fused multi-instrument product


class MaskType(str, Enum):
    """Irrigation mask types."""
    IRR = "irr"
    INV_IRR = "inv_irr"
    NO_MASK = "no_mask"


class ETModel(str, Enum):
    """ET fraction models."""
    SSEBOP = "ssebop"
    PTJPL = "ptjpl"
    SIMS = "sims"
    EEMETRIC = "eemetric"
    DISALEXI = "disalexi"
    GEESEBAL = "geesebal"


class MetSource(str, Enum):
    """Meteorology data sources."""
    GRIDMET = "gridmet"
    ERA5 = "era5"
    NLDAS = "nldas"


class SnowSource(str, Enum):
    """Snow data sources."""
    SNODAS = "snodas"
    ERA5 = "era5"


class SoilSource(str, Enum):
    """Soil data sources."""
    SSURGO = "ssurgo"  # US Soil Survey
    HWSD = "hwsd"      # Harmonized World Soil Database (international)


class Parameter(str, Enum):
    """Data parameters/variables."""
    # Remote sensing
    NDVI = "ndvi"
    ETF = "etf"
    LST = "lst"  # Land surface temperature (ECOSTRESS)

    # Meteorology - base variables
    ETO = "eto"
    ETR = "etr"    # Reference ET tall (ASCE standardized)
    PRCP = "prcp"
    TMIN = "tmin"
    TMAX = "tmax"
    TMEAN = "tmean"
    SRAD = "srad"
    VPD = "vpd"
    EA = "ea"      # Vapor pressure (kPa)
    U2 = "u2"      # Wind speed at 2m (m/s)
    WIND = "wind"  # Alias for wind speed
    RHMIN = "rhmin"
    RHMAX = "rhmax"
    ELEV = "elev"  # Elevation (m)

    # Meteorology - bias-corrected variables
    ETO_CORR = "eto_corr"
    ETR_CORR = "etr_corr"

    # Snow
    SWE = "swe"


class PropertyType(str, Enum):
    """Static property types."""
    # Soils
    AWC = "awc"
    CLAY = "clay"
    SAND = "sand"
    KSAT = "ksat"

    # Land cover
    MODIS_LC = "modis_lc"
    CDL = "cdl"
    GLC10 = "glc10"

    # Irrigation
    LANID = "lanid"
    IRRMAPPER = "irrmapper"

    # Location
    LAT = "lat"
    LON = "lon"
    ELEVATION = "elevation"
    STATE = "state"


@dataclass
class ParameterSpec:
    """Specification for a data parameter."""
    name: str
    dtype: str = "float32"
    valid_range: tuple = (None, None)
    units: str = ""
    description: str = ""
    required_for_model: bool = False


@dataclass
class DataGroupSpec:
    """Specification for a data group (e.g., remote_sensing/ndvi)."""
    path: str
    parameters: List[str]
    instruments: List[str] = field(default_factory=list)
    masks: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class RootingDepthSpec:
    """
    Specification for rooting depth by land cover class.

    Reference: "Global estimation of effective plant rooting depth:
    Implications for hydrological modeling"
    https://doi.org/10.1002/2016WR019392

    LULC codes follow MODIS Land Cover Type 1 (IGBP classification):
    https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1#bands
    """
    mean_depth: float       # Mean effective rooting depth (m)
    max_depth: float        # Maximum effective rooting depth (m)
    zr_multiplier: int      # Multiplier for rootzone calculation
    description: str        # Land cover class description


# ================================================================================
# Centralized Constants (migrated from swimrs.prep)
# ================================================================================

# Rooting depth specifications by MODIS IGBP Land Cover Type 1 code
# Keys are integer LULC codes (1-16, note 15 is missing - bare/sparse vegetation)
ROOTING_DEPTH_BY_LULC: Dict[int, RootingDepthSpec] = {
    1: RootingDepthSpec(
        mean_depth=0.43, max_depth=1.34, zr_multiplier=5,
        description="Evergreen Needleleaf Forests: dominated by evergreen conifer trees (canopy >2m). Tree cover >60%."
    ),
    2: RootingDepthSpec(
        mean_depth=3.14, max_depth=7.99, zr_multiplier=2,
        description="Evergreen Broadleaf Forests: dominated by evergreen broadleaf and palmate trees (canopy >2m). Tree cover >60%."
    ),
    3: RootingDepthSpec(
        mean_depth=0.38, max_depth=0.84, zr_multiplier=5,
        description="Deciduous Needleleaf Forests: dominated by deciduous needleleaf (larch) trees (canopy >2m). Tree cover >60%."
    ),
    4: RootingDepthSpec(
        mean_depth=1.07, max_depth=2.09, zr_multiplier=5,
        description="Deciduous Broadleaf Forests: dominated by deciduous broadleaf trees (canopy >2m). Tree cover >60%."
    ),
    5: RootingDepthSpec(
        mean_depth=0.54, max_depth=1.94, zr_multiplier=5,
        description="Mixed Forests: dominated by neither deciduous nor evergreen (40-60% of each) tree type (canopy >2m). Tree cover >60%."
    ),
    6: RootingDepthSpec(
        mean_depth=0.37, max_depth=1.12, zr_multiplier=3,
        description="Closed Shrublands: dominated by woody perennials (1-2m height) >60% cover."
    ),
    7: RootingDepthSpec(
        mean_depth=0.37, max_depth=1.12, zr_multiplier=3,
        description="Open Shrublands: dominated by woody perennials (1-2m height) 10-60% cover."
    ),
    8: RootingDepthSpec(
        mean_depth=0.80, max_depth=2.28, zr_multiplier=3,
        description="Woody Savannas: tree cover 30-60% (canopy >2m)."
    ),
    9: RootingDepthSpec(
        mean_depth=0.80, max_depth=2.28, zr_multiplier=3,
        description="Savannas: tree cover 10-30% (canopy >2m)."
    ),
    10: RootingDepthSpec(
        mean_depth=0.51, max_depth=1.18, zr_multiplier=3,
        description="Grasslands: dominated by herbaceous annuals (<2m)."
    ),
    11: RootingDepthSpec(
        mean_depth=0.37, max_depth=1.12, zr_multiplier=3,
        description="Wetlands"
    ),
    12: RootingDepthSpec(
        mean_depth=0.55, max_depth=1.12, zr_multiplier=3,
        description="Cropland, same depth as shrublands"
    ),
    13: RootingDepthSpec(
        mean_depth=0.55, max_depth=1.12, zr_multiplier=3,
        description="Developed"
    ),
    14: RootingDepthSpec(
        mean_depth=0.55, max_depth=1.12, zr_multiplier=1,
        description="Cropland/Natural Mosaic, same depth as shrublands"
    ),
    16: RootingDepthSpec(
        mean_depth=0.41, max_depth=1.43, zr_multiplier=5,
        description="Desert vegetation."
    ),
}

# Required meteorology parameters for model input
REQUIRED_MET_BASE: List[str] = ["tmin", "tmax", "srad", "swe", "prcp", "nld_ppt_d"]
REQUIRED_MET_HOURLY: List[str] = [f"prcp_hr_{i:02d}" for i in range(24)]
REQUIRED_MET_IRR: List[str] = ["eto_corr"]  # Required for irrigated fields
REQUIRED_MET_UNIRR: List[str] = ["eto"]     # Required for non-irrigated fields

# Parameters that can have NaN values in model input
ACCEPT_NAN_PARAMS: List[str] = REQUIRED_MET_IRR + REQUIRED_MET_UNIRR + ["swe"]

# Parameter to unit mapping
ACCEPTED_UNITS_MAP: Dict[str, str] = {
    "elev": "m",
    "tmin": "c",
    "tmax": "c",
    "etr": "mm",
    "etr_corr": "mm",
    "eto": "mm",
    "eto_corr": "mm",
    "prcp": "mm",
    "srad": "wm2",
    "u2": "ms",
    "ea": "kpa",
    "et_fraction": "unitless",
    "nld_ppt_d": "mm",
    "centroid_lat": "degrees",
    "centroid_lon": "degrees",
    "swe": "mm",
    **{f"prcp_hr_{i:02d}": "mm" for i in range(24)},
}

# Column MultiIndex structure for parquet time series files
COLUMN_MULTIINDEX: List[str] = ["site", "instrument", "parameter", "units", "algorithm", "mask"]


def get_rooting_depth(lulc_code: int, use_max: bool = True) -> Tuple[float, int]:
    """
    Get rooting depth and zr_multiplier for a given LULC code.

    Args:
        lulc_code: MODIS IGBP land cover type code (1-16)
        use_max: If True, return max rooting depth; otherwise mean

    Returns:
        Tuple of (rooting_depth_m, zr_multiplier)

    Example:
        >>> depth, mult = get_rooting_depth(12)  # Cropland
        >>> print(f"Depth: {depth}m, Multiplier: {mult}")
        Depth: 1.12m, Multiplier: 3
    """
    if lulc_code not in ROOTING_DEPTH_BY_LULC:
        # Default to cropland if unknown
        lulc_code = 12

    spec = ROOTING_DEPTH_BY_LULC[lulc_code]
    depth = spec.max_depth if use_max else spec.mean_depth
    return depth, spec.zr_multiplier


class SwimSchema:
    """
    Schema definition for SWIM data container.

    Defines the complete structure of data that can be stored,
    including valid combinations of instruments, parameters, masks, etc.
    """

    VERSION = "1.0"

    # Parameter specifications
    PARAMETERS = {
        Parameter.NDVI: ParameterSpec(
            name="ndvi",
            dtype="float32",
            valid_range=(0.0, 1.0),
            units="",
            description="Normalized Difference Vegetation Index",
            required_for_model=True,
        ),
        Parameter.ETF: ParameterSpec(
            name="etf",
            dtype="float32",
            valid_range=(0.0, 2.0),
            units="",
            description="ET fraction (actual ET / reference ET)",
            required_for_model=True,
        ),
        Parameter.LST: ParameterSpec(
            name="lst",
            dtype="float32",
            valid_range=(200.0, 350.0),
            units="K",
            description="Land Surface Temperature",
        ),
        Parameter.ETO: ParameterSpec(
            name="eto",
            dtype="float32",
            valid_range=(0.0, 20.0),
            units="mm/day",
            description="Reference evapotranspiration",
            required_for_model=True,
        ),
        Parameter.PRCP: ParameterSpec(
            name="prcp",
            dtype="float32",
            valid_range=(0.0, 500.0),
            units="mm/day",
            description="Precipitation",
            required_for_model=True,
        ),
        Parameter.TMIN: ParameterSpec(
            name="tmin",
            dtype="float32",
            valid_range=(-50.0, 50.0),
            units="C",
            description="Minimum temperature",
            required_for_model=True,
        ),
        Parameter.TMAX: ParameterSpec(
            name="tmax",
            dtype="float32",
            valid_range=(-50.0, 60.0),
            units="C",
            description="Maximum temperature",
            required_for_model=True,
        ),
        Parameter.SRAD: ParameterSpec(
            name="srad",
            dtype="float32",
            valid_range=(0.0, 40.0),
            units="MJ/m2/day",
            description="Solar radiation",
        ),
        Parameter.SWE: ParameterSpec(
            name="swe",
            dtype="float32",
            valid_range=(0.0, 5000.0),
            units="mm",
            description="Snow water equivalent",
        ),
        Parameter.U2: ParameterSpec(
            name="u2",
            dtype="float32",
            valid_range=(0.0, 50.0),
            units="m/s",
            description="Wind speed at 2m height",
        ),
        Parameter.ELEV: ParameterSpec(
            name="elev",
            dtype="float32",
            valid_range=(-500.0, 9000.0),
            units="m",
            description="Elevation above sea level",
        ),
        Parameter.ETO_CORR: ParameterSpec(
            name="eto_corr",
            dtype="float32",
            valid_range=(0.0, 20.0),
            units="mm/day",
            description="Bias-corrected reference evapotranspiration (grass)",
        ),
        Parameter.ETR_CORR: ParameterSpec(
            name="etr_corr",
            dtype="float32",
            valid_range=(0.0, 25.0),
            units="mm/day",
            description="Bias-corrected reference evapotranspiration (alfalfa)",
        ),
    }

    # Valid combinations
    REMOTE_SENSING_STRUCTURE = {
        "ndvi": {
            "instruments": [Instrument.LANDSAT, Instrument.SENTINEL, Instrument.ECOSTRESS],
            "masks": [MaskType.IRR, MaskType.INV_IRR, MaskType.NO_MASK],
            "models": [],  # NDVI has no model dimension
        },
        "etf": {
            "instruments": [Instrument.LANDSAT, Instrument.ECOSTRESS],
            "masks": [MaskType.IRR, MaskType.INV_IRR, MaskType.NO_MASK],
            "models": list(ETModel),
        },
        "lst": {
            "instruments": [Instrument.ECOSTRESS],
            "masks": [MaskType.NO_MASK],
            "models": [],
        },
    }

    METEOROLOGY_STRUCTURE = {
        "sources": [MetSource.GRIDMET, MetSource.ERA5, MetSource.NLDAS],
        "variables": [
            Parameter.ETO, Parameter.ETR, Parameter.PRCP,
            Parameter.TMIN, Parameter.TMAX, Parameter.TMEAN,
            Parameter.SRAD, Parameter.VPD, Parameter.EA, Parameter.U2,
            Parameter.ELEV,
            # Bias-corrected versions (e.g., GridMET corrections)
            Parameter.ETO_CORR, Parameter.ETR_CORR,
        ],
    }

    SNOW_STRUCTURE = {
        "sources": [SnowSource.SNODAS, SnowSource.ERA5],
        "variables": [Parameter.SWE],
    }

    PROPERTIES_STRUCTURE = {
        "soils": ["awc", "clay", "sand", "ksat", "rock_ite", "rew", "source"],
        "land_cover": ["modis_lc", "cdl", "glc10", "lulc_code"],
        "vegetation": ["rooting_depth"],
        "irrigation": ["lanid", "irrmapper", "irr"],
        "location": ["lat", "lon", "elevation", "state", "area_m2"],
    }

    DERIVED_STRUCTURE = {
        "dynamics": ["ke_max", "kc_max", "irr_data", "gwsub_data"],
        "combined_ndvi": ["ndvi"],  # Fused multi-instrument NDVI
    }

    @classmethod
    def get_zarr_path(cls, category: str, parameter: str,
                      instrument: str = None, mask: str = None,
                      model: str = None, source: str = None) -> str:
        """
        Generate the Zarr array path for a given data specification.

        Examples:
            remote_sensing/ndvi/landsat/irr
            remote_sensing/etf/landsat/ssebop/irr
            meteorology/gridmet/eto
            properties/soils/awc
        """
        parts = [category]

        if category == "remote_sensing":
            parts.append(parameter)
            if instrument:
                parts.append(instrument)
            if model:
                parts.append(model)
            if mask:
                parts.append(mask)
        elif category == "meteorology":
            if source:
                parts.append(source)
            parts.append(parameter)
        elif category == "properties":
            # e.g., properties/soils/awc
            parts.append(parameter)
        elif category == "derived":
            parts.append(parameter)
        elif category == "snow":
            parts.append(parameter)

        return "/".join(parts)

    @classmethod
    def validate_path(cls, path: str) -> bool:
        """Validate that a data path conforms to the schema."""
        parts = path.strip("/").split("/")
        if not parts:
            return False

        category = parts[0]

        if category == "remote_sensing":
            if len(parts) < 3:
                return False
            param = parts[1]
            if param not in cls.REMOTE_SENSING_STRUCTURE:
                return False
            spec = cls.REMOTE_SENSING_STRUCTURE[param]
            instrument = parts[2] if len(parts) > 2 else None
            if instrument and instrument not in [i.value for i in spec["instruments"]]:
                return False
            return True

        elif category == "meteorology":
            if len(parts) < 3:
                return False
            source = parts[1]
            if source not in [s.value for s in cls.METEOROLOGY_STRUCTURE["sources"]]:
                return False
            return True

        elif category == "properties":
            return len(parts) >= 2

        elif category in ("derived", "snow", "geometry"):
            return len(parts) >= 1

        return False

    @classmethod
    def list_all_paths(cls) -> List[str]:
        """List all valid data paths in the schema."""
        paths = []

        # Remote sensing
        for param, spec in cls.REMOTE_SENSING_STRUCTURE.items():
            for inst in spec["instruments"]:
                for mask in spec["masks"]:
                    if spec["models"]:
                        for model in spec["models"]:
                            paths.append(f"remote_sensing/{param}/{inst.value}/{model.value}/{mask.value}")
                    else:
                        paths.append(f"remote_sensing/{param}/{inst.value}/{mask.value}")

        # Meteorology
        for source in cls.METEOROLOGY_STRUCTURE["sources"]:
            for var in cls.METEOROLOGY_STRUCTURE["variables"]:
                paths.append(f"meteorology/{source.value}/{var.value}")

        # Properties
        for group, props in cls.PROPERTIES_STRUCTURE.items():
            for prop in props:
                paths.append(f"properties/{group}/{prop}")

        # Derived
        for group, items in cls.DERIVED_STRUCTURE.items():
            for item in items:
                paths.append(f"derived/{group}/{item}")

        # Snow
        for source in cls.SNOW_STRUCTURE["sources"]:
            for var in cls.SNOW_STRUCTURE["variables"]:
                paths.append(f"snow/{source.value}/{var.value}")

        return paths

    @classmethod
    def required_for_calibration(cls, model: str = "ssebop", mask: str = "irr",
                                  met_source: str = "gridmet",
                                  snow_source: str = "snodas",
                                  instrument: str = "landsat") -> List[str]:
        """
        List data paths required to run calibration.

        Args:
            model: ET model to calibrate against (default: ssebop)
            mask: Mask type to use (irr, inv_irr, or no_mask)
            met_source: Meteorology source (gridmet, era5, nldas)
            snow_source: Snow data source (snodas, era5)
            instrument: Remote sensing instrument (landsat, ecostress)
        """
        # For calibration we typically need both irr and inv_irr masks
        # to handle irrigated vs non-irrigated analysis
        required = [
            f"remote_sensing/ndvi/{instrument}/irr",
            f"remote_sensing/ndvi/{instrument}/inv_irr",
            f"remote_sensing/etf/{instrument}/{model}/irr",
            f"remote_sensing/etf/{instrument}/{model}/inv_irr",
            f"meteorology/{met_source}/eto",
            f"meteorology/{met_source}/prcp",
            f"meteorology/{met_source}/tmin",
            f"meteorology/{met_source}/tmax",
            "properties/soils/awc",
            "properties/irrigation/irr",
            "derived/dynamics/ke_max",
            "derived/dynamics/kc_max",
            "derived/dynamics/irr_data",
        ]
        return required

    @classmethod
    def required_for_forward_run(cls, model: str = "ssebop", mask: str = "irr",
                                  met_source: str = "gridmet",
                                  instrument: str = "landsat") -> List[str]:
        """
        List data paths required to run forward model (uncalibrated).

        Args:
            model: ET model to use (default: ssebop)
            mask: Mask type - can be 'irr', 'inv_irr', or 'no_mask'
            met_source: Meteorology source (gridmet, era5, nldas)
            instrument: Remote sensing instrument (landsat, ecostress)
        """
        required = [
            f"remote_sensing/ndvi/{instrument}/{mask}",
            f"remote_sensing/etf/{instrument}/{model}/{mask}",
            f"meteorology/{met_source}/eto",
            f"meteorology/{met_source}/prcp",
            f"meteorology/{met_source}/tmin",
            f"meteorology/{met_source}/tmax",
            "properties/soils/awc",
        ]
        return required
