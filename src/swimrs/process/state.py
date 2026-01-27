"""Typed state containers for SWIM-RS water balance modeling.

Provides dataclass-based containers for:
- WaterBalanceState: Mutable daily state arrays
- FieldProperties: Static soil/crop properties
- CalibrationParameters: PEST++ calibration parameters

All containers use numpy arrays with shape (n_fields,) for vectorized
computation with numba kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "WaterBalanceState",
    "FieldProperties",
    "CalibrationParameters",
    "load_pest_mult_properties",
]


@dataclass
class WaterBalanceState:
    """Mutable state arrays for daily water balance computation.

    All arrays have shape (n_fields,). State is updated in-place
    during the daily step loop.

    Attributes
    ----------
    n_fields : int
        Number of fields/pixels being modeled
    depl_root : NDArray[np.float64]
        Root zone depletion (mm), 0 = field capacity
    depl_ze : NDArray[np.float64]
        Surface layer depletion (mm) for evaporation
    daw3 : NDArray[np.float64]
        Available water in layer 3 below root zone (mm)
    taw3 : NDArray[np.float64]
        Total available water capacity in layer 3 (mm)
    swe : NDArray[np.float64]
        Snow water equivalent (mm)
    albedo : NDArray[np.float64]
        Snow albedo (dimensionless, 0-1)
    zr : NDArray[np.float64]
        Current root depth (m)
    kr : NDArray[np.float64]
        Evaporation reduction coefficient (damped)
    ks : NDArray[np.float64]
        Water stress coefficient (damped)
    irr_continue : NDArray[np.float64]
        Irrigation continuation flag (1.0 if continuing from previous day)
    next_day_irr : NDArray[np.float64]
        Carryover irrigation amount for next day (mm)
    s : NDArray[np.float64]
        Current day S retention parameter (mm) for CN runoff
    s1, s2, s3, s4 : NDArray[np.float64]
        S history from 1-4 days ago (mm), used for smoothed runoff
        on irrigated fields
    irr_frac_root : NDArray[np.float64]
        Fraction of root zone water that originated from irrigation [0, 1].
        Used for consumptive use accounting.
    irr_frac_l3 : NDArray[np.float64]
        Fraction of layer 3 water that originated from irrigation [0, 1].
        Used for deep percolation accounting.
    """

    n_fields: int
    depl_root: NDArray[np.float64] = field(default=None)
    depl_ze: NDArray[np.float64] = field(default=None)
    daw3: NDArray[np.float64] = field(default=None)
    taw3: NDArray[np.float64] = field(default=None)
    swe: NDArray[np.float64] = field(default=None)
    albedo: NDArray[np.float64] = field(default=None)
    zr: NDArray[np.float64] = field(default=None)
    kr: NDArray[np.float64] = field(default=None)
    ks: NDArray[np.float64] = field(default=None)
    irr_continue: NDArray[np.float64] = field(default=None)
    next_day_irr: NDArray[np.float64] = field(default=None)
    # S history for smoothed CN runoff (irrigated fields)
    s: NDArray[np.float64] = field(default=None)      # Current day S retention
    s1: NDArray[np.float64] = field(default=None)     # 1 day ago
    s2: NDArray[np.float64] = field(default=None)     # 2 days ago
    s3: NDArray[np.float64] = field(default=None)     # 3 days ago
    s4: NDArray[np.float64] = field(default=None)     # 4 days ago
    # Irrigation fraction tracking for consumptive use accounting
    irr_frac_root: NDArray[np.float64] = field(default=None)  # Root zone [0, 1]
    irr_frac_l3: NDArray[np.float64] = field(default=None)  # Layer 3 [0, 1]

    def __post_init__(self):
        """Initialize arrays with zeros if not provided."""
        n = self.n_fields
        if self.depl_root is None:
            self.depl_root = np.zeros(n, dtype=np.float64)
        if self.depl_ze is None:
            self.depl_ze = np.zeros(n, dtype=np.float64)
        if self.daw3 is None:
            self.daw3 = np.zeros(n, dtype=np.float64)
        if self.taw3 is None:
            self.taw3 = np.zeros(n, dtype=np.float64)
        if self.swe is None:
            self.swe = np.zeros(n, dtype=np.float64)
        if self.albedo is None:
            self.albedo = np.full(n, 0.23, dtype=np.float64)  # Dry soil albedo
        if self.zr is None:
            self.zr = np.full(n, 0.1, dtype=np.float64)  # Minimum root depth
        if self.kr is None:
            self.kr = np.ones(n, dtype=np.float64)
        if self.ks is None:
            self.ks = np.ones(n, dtype=np.float64)
        if self.irr_continue is None:
            self.irr_continue = np.zeros(n, dtype=np.float64)
        if self.next_day_irr is None:
            self.next_day_irr = np.zeros(n, dtype=np.float64)
        # S history: initialize with default S value from CN2=75 => S=84.7 mm
        default_s = 84.7
        if self.s is None:
            self.s = np.full(n, default_s, dtype=np.float64)
        if self.s1 is None:
            self.s1 = np.full(n, default_s, dtype=np.float64)
        if self.s2 is None:
            self.s2 = np.full(n, default_s, dtype=np.float64)
        if self.s3 is None:
            self.s3 = np.full(n, default_s, dtype=np.float64)
        if self.s4 is None:
            self.s4 = np.full(n, default_s, dtype=np.float64)
        # Irrigation fraction tracking: default to 0.0 (no irrigation water)
        # Proper initialization based on irr_status is done in from_spinup()
        if self.irr_frac_root is None:
            self.irr_frac_root = np.zeros(n, dtype=np.float64)
        if self.irr_frac_l3 is None:
            self.irr_frac_l3 = np.zeros(n, dtype=np.float64)

    @classmethod
    def from_spinup(
        cls,
        n_fields: int,
        depl_root: NDArray[np.float64],
        swe: NDArray[np.float64],
        kr: NDArray[np.float64],
        ks: NDArray[np.float64],
        zr: NDArray[np.float64],
        daw3: NDArray[np.float64] | None = None,
        taw3: NDArray[np.float64] | None = None,
        depl_ze: NDArray[np.float64] | None = None,
        albedo: NDArray[np.float64] | None = None,
        s: NDArray[np.float64] | None = None,
        s1: NDArray[np.float64] | None = None,
        s2: NDArray[np.float64] | None = None,
        s3: NDArray[np.float64] | None = None,
        s4: NDArray[np.float64] | None = None,
        irr_frac_root: NDArray[np.float64] | None = None,
        irr_frac_l3: NDArray[np.float64] | None = None,
        irr_status: NDArray[np.bool_] | None = None,
    ) -> WaterBalanceState:
        """Create state from spinup values.

        Parameters
        ----------
        n_fields : int
            Number of fields
        depl_root : NDArray[np.float64]
            Initial root zone depletion (mm)
        swe : NDArray[np.float64]
            Initial snow water equivalent (mm)
        kr : NDArray[np.float64]
            Initial evaporation reduction coefficient
        ks : NDArray[np.float64]
            Initial water stress coefficient
        zr : NDArray[np.float64]
            Initial root depth (m)
        daw3 : NDArray[np.float64], optional
            Initial layer 3 available water (mm)
        taw3 : NDArray[np.float64], optional
            Initial layer 3 total capacity (mm)
        depl_ze : NDArray[np.float64], optional
            Initial surface layer depletion (mm)
        albedo : NDArray[np.float64], optional
            Initial snow albedo
        s : NDArray[np.float64], optional
            Current S retention (mm)
        s1, s2, s3, s4 : NDArray[np.float64], optional
            S history from 1-4 days ago (mm)
        irr_frac_root : NDArray[np.float64], optional
            Irrigation fraction in root zone [0, 1]. If not provided,
            initialized based on irr_status (0.5 if irrigated, 0.0 if not).
        irr_frac_l3 : NDArray[np.float64], optional
            Irrigation fraction in layer 3 [0, 1]. If not provided,
            initialized based on irr_status (0.5 if irrigated, 0.0 if not).
        irr_status : NDArray[np.bool_], optional
            Whether each field is irrigated. Used to initialize irrigation
            fractions when they are not provided in spinup data.

        Returns
        -------
        WaterBalanceState
            Initialized state container
        """
        state = cls(n_fields=n_fields)
        state.depl_root = depl_root.copy()
        state.swe = swe.copy()
        state.kr = kr.copy()
        state.ks = ks.copy()
        state.zr = zr.copy()

        if daw3 is not None:
            state.daw3 = daw3.copy()
        if taw3 is not None:
            state.taw3 = taw3.copy()
        if depl_ze is not None:
            state.depl_ze = depl_ze.copy()
        if albedo is not None:
            state.albedo = albedo.copy()
        if s is not None:
            state.s = s.copy()
        if s1 is not None:
            state.s1 = s1.copy()
        if s2 is not None:
            state.s2 = s2.copy()
        if s3 is not None:
            state.s3 = s3.copy()
        if s4 is not None:
            state.s4 = s4.copy()

        # Irrigation fraction tracking
        if irr_frac_root is not None:
            state.irr_frac_root = irr_frac_root.copy()
        elif irr_status is not None:
            # Initialize based on irrigation status: 0.5 for irrigated, 0.0 for non-irrigated
            state.irr_frac_root = np.where(irr_status, 0.5, 0.0).astype(np.float64)

        if irr_frac_l3 is not None:
            state.irr_frac_l3 = irr_frac_l3.copy()
        elif irr_status is not None:
            state.irr_frac_l3 = np.where(irr_status, 0.5, 0.0).astype(np.float64)

        return state

    def copy(self) -> WaterBalanceState:
        """Create a deep copy of the state."""
        return WaterBalanceState(
            n_fields=self.n_fields,
            depl_root=self.depl_root.copy(),
            depl_ze=self.depl_ze.copy(),
            daw3=self.daw3.copy(),
            taw3=self.taw3.copy(),
            swe=self.swe.copy(),
            albedo=self.albedo.copy(),
            zr=self.zr.copy(),
            kr=self.kr.copy(),
            ks=self.ks.copy(),
            irr_continue=self.irr_continue.copy(),
            next_day_irr=self.next_day_irr.copy(),
            s=self.s.copy(),
            s1=self.s1.copy(),
            s2=self.s2.copy(),
            s3=self.s3.copy(),
            s4=self.s4.copy(),
            irr_frac_root=self.irr_frac_root.copy(),
            irr_frac_l3=self.irr_frac_l3.copy(),
        )


@dataclass
class FieldProperties:
    """Static soil and crop properties for each field.

    All arrays have shape (n_fields,). These properties are read from
    the HDF5 input file and do not change during simulation.

    Attributes
    ----------
    n_fields : int
        Number of fields/pixels
    fids : NDArray
        Field identifiers (string or int)
    awc : NDArray[np.float64]
        Available water capacity (mm/m)
    ksat : NDArray[np.float64]
        Saturated hydraulic conductivity (mm/day).

        This is converted to an hourly infiltration capacity (mm/hr) when
        running infiltration-excess runoff (IER) using hourly precipitation.
    rew : NDArray[np.float64]
        Readily evaporable water (mm)
    tew : NDArray[np.float64]
        Total evaporable water (mm)
    cn2 : NDArray[np.float64]
        Curve number for average antecedent moisture (AMC II)
    zr_max : NDArray[np.float64]
        Maximum root depth (m)
    zr_min : NDArray[np.float64]
        Minimum root depth (m)
    mad : NDArray[np.float64]
        Depletion fraction for stress onset
    irr_status : NDArray[np.bool_]
        Whether field is irrigated
    perennial : NDArray[np.bool_]
        Whether crop is perennial (affects root dynamics)
    gw_status : NDArray[np.bool_]
        Whether groundwater subsidy is available
    ke_max : NDArray[np.float64]
        Maximum soil evaporation coefficient, derived from ETf observations
        where NDVI < 0.3 (90th percentile of bare soil ETf)
    kc_max : NDArray[np.float64]
        Maximum crop coefficient, derived from ETf observations
        (90th percentile of all ETf values)
    f_sub : NDArray[np.float64]
        Groundwater subsidy fraction (0-1), derived from ETa/PPT ratio
        where f_sub = (ratio - 1) / ratio when ratio > 1
    """

    n_fields: int
    fids: NDArray = field(default=None)
    awc: NDArray[np.float64] = field(default=None)
    ksat: NDArray[np.float64] = field(default=None)
    rew: NDArray[np.float64] = field(default=None)
    tew: NDArray[np.float64] = field(default=None)
    cn2: NDArray[np.float64] = field(default=None)
    zr_max: NDArray[np.float64] = field(default=None)
    zr_min: NDArray[np.float64] = field(default=None)
    mad: NDArray[np.float64] = field(default=None)
    irr_status: NDArray[np.bool_] = field(default=None)
    perennial: NDArray[np.bool_] = field(default=None)
    gw_status: NDArray[np.bool_] = field(default=None)
    ke_max: NDArray[np.float64] = field(default=None)
    kc_max: NDArray[np.float64] = field(default=None)
    f_sub: NDArray[np.float64] = field(default=None)

    def __post_init__(self):
        """Initialize arrays with reasonable defaults if not provided."""
        n = self.n_fields
        if self.fids is None:
            self.fids = np.arange(n)
        if self.awc is None:
            self.awc = np.full(n, 150.0, dtype=np.float64)  # mm/m
        if self.ksat is None:
            self.ksat = np.full(n, 10.0, dtype=np.float64)  # mm/day
        if self.rew is None:
            self.rew = np.full(n, 9.0, dtype=np.float64)  # mm
        if self.tew is None:
            self.tew = np.full(n, 25.0, dtype=np.float64)  # mm
        if self.cn2 is None:
            self.cn2 = np.full(n, 75.0, dtype=np.float64)
        if self.zr_max is None:
            self.zr_max = np.full(n, 1.0, dtype=np.float64)  # m
        if self.zr_min is None:
            self.zr_min = np.full(n, 0.1, dtype=np.float64)  # m
        if self.mad is None:
            self.mad = np.full(n, 0.5, dtype=np.float64)
        if self.irr_status is None:
            self.irr_status = np.zeros(n, dtype=np.bool_)
        if self.perennial is None:
            self.perennial = np.zeros(n, dtype=np.bool_)
        if self.gw_status is None:
            self.gw_status = np.zeros(n, dtype=np.bool_)
        if self.ke_max is None:
            self.ke_max = np.full(n, 1.0, dtype=np.float64)  # Default: no cap
        if self.kc_max is None:
            self.kc_max = np.full(n, 1.25, dtype=np.float64)  # Default: typical crop max
        if self.f_sub is None:
            self.f_sub = np.zeros(n, dtype=np.float64)  # Default: no GW subsidy

    def compute_taw(self, zr: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute total available water for given root depth.

        TAW = AWC * Zr

        Guards are applied to prevent:
        - Division by zero (TAW >= 0.001)
        - Early stress with shallow roots (TAW >= TEW)

        Parameters
        ----------
        zr : NDArray[np.float64]
            Root depth (m)

        Returns
        -------
        NDArray[np.float64]
            Total available water (mm)
        """
        taw = self.awc * zr
        # Prevent division by zero in stress calculations
        taw = np.maximum(taw, 0.001)
        # TAW should be at least as large as surface layer (TEW)
        # to prevent early stress onset with shallow roots
        taw = np.maximum(taw, self.tew)
        return taw

    def compute_raw(
        self, taw: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute readily available water.

        RAW = p * TAW

        Parameters
        ----------
        taw : NDArray[np.float64]
            Total available water (mm)

        Returns
        -------
        NDArray[np.float64]
            Readily available water (mm)
        """
        return self.mad * taw


@dataclass
class CalibrationParameters:
    """Parameters that can be modified by PEST++ multipliers.

    All arrays have shape (n_fields,). These are the calibration
    parameters that PEST++ adjusts via multiplier files.

    Attributes
    ----------
    n_fields : int
        Number of fields/pixels
    kc_min : NDArray[np.float64]
        Minimum crop coefficient (bare soil)
    ndvi_k : NDArray[np.float64]
        Sigmoid steepness parameter for Kcb
    ndvi_0 : NDArray[np.float64]
        Sigmoid midpoint (inflection) NDVI
    swe_alpha : NDArray[np.float64]
        Snow melt radiation coefficient
    swe_beta : NDArray[np.float64]
        Snow melt degree-day coefficient
    kr_damp : NDArray[np.float64]
        Kr damping factor (0-1)
    ks_damp : NDArray[np.float64]
        Ks damping factor (0-1)
    max_irr_rate : NDArray[np.float64]
        Maximum daily irrigation rate (mm/day)
    """

    n_fields: int
    kc_min: NDArray[np.float64] = field(default=None)
    ndvi_k: NDArray[np.float64] = field(default=None)
    ndvi_0: NDArray[np.float64] = field(default=None)
    swe_alpha: NDArray[np.float64] = field(default=None)
    swe_beta: NDArray[np.float64] = field(default=None)
    kr_damp: NDArray[np.float64] = field(default=None)
    ks_damp: NDArray[np.float64] = field(default=None)
    max_irr_rate: NDArray[np.float64] = field(default=None)

    def __post_init__(self):
        """Initialize arrays with default values if not provided."""
        n = self.n_fields
        if self.kc_min is None:
            self.kc_min = np.full(n, 0.15, dtype=np.float64)
        if self.ndvi_k is None:
            self.ndvi_k = np.full(n, 7.0, dtype=np.float64)
        if self.ndvi_0 is None:
            self.ndvi_0 = np.full(n, 0.4, dtype=np.float64)
        if self.swe_alpha is None:
            self.swe_alpha = np.full(n, 0.5, dtype=np.float64)
        if self.swe_beta is None:
            self.swe_beta = np.full(n, 2.0, dtype=np.float64)
        if self.kr_damp is None:
            self.kr_damp = np.full(n, 0.2, dtype=np.float64)
        if self.ks_damp is None:
            self.ks_damp = np.full(n, 0.2, dtype=np.float64)
        if self.max_irr_rate is None:
            self.max_irr_rate = np.full(n, 25.0, dtype=np.float64)

    @classmethod
    def from_base_with_multipliers(
        cls,
        base: CalibrationParameters,
        multipliers: dict[str, NDArray[np.float64]],
    ) -> CalibrationParameters:
        """Create parameters by applying multipliers to base values.

        Parameters
        ----------
        base : CalibrationParameters
            Base parameter values
        multipliers : dict
            Parameter name -> multiplier array

        Returns
        -------
        CalibrationParameters
            New parameters with multipliers applied
        """
        params = cls(n_fields=base.n_fields)

        # Copy base values
        params.kc_min = base.kc_min.copy()
        params.ndvi_k = base.ndvi_k.copy()
        params.ndvi_0 = base.ndvi_0.copy()
        params.swe_alpha = base.swe_alpha.copy()
        params.swe_beta = base.swe_beta.copy()
        params.kr_damp = base.kr_damp.copy()
        params.ks_damp = base.ks_damp.copy()
        params.max_irr_rate = base.max_irr_rate.copy()

        # Apply multipliers
        for param_name, mult in multipliers.items():
            if hasattr(params, param_name):
                arr = getattr(params, param_name)
                arr *= mult

        return params

    def copy(self) -> CalibrationParameters:
        """Create a deep copy of the parameters."""
        params = CalibrationParameters(n_fields=self.n_fields)
        params.kc_min = self.kc_min.copy()
        params.ndvi_k = self.ndvi_k.copy()
        params.ndvi_0 = self.ndvi_0.copy()
        params.swe_alpha = self.swe_alpha.copy()
        params.swe_beta = self.swe_beta.copy()
        params.kr_damp = self.kr_damp.copy()
        params.ks_damp = self.ks_damp.copy()
        params.max_irr_rate = self.max_irr_rate.copy()
        return params

    @classmethod
    def from_pest_mult(
        cls,
        mult_dir: str,
        fids: list[str],
        base: CalibrationParameters | None = None,
    ) -> CalibrationParameters:
        """Load parameters from PEST++ multiplier CSV files.

        PEST++ writes parameter multiplier files in the format:
            mult/p_{param}_{fid}_0_constant.csv

        The CSV has a header row and the value is in the last column ('1').

        Parameters
        ----------
        mult_dir : str
            Path to the mult/ directory containing CSV files
        fids : list[str]
            Field IDs in simulation order
        base : CalibrationParameters, optional
            Base parameters to start from. If None, uses defaults.

        Returns
        -------
        CalibrationParameters
            Parameters loaded from mult files
        """
        import pandas as pd
        from pathlib import Path

        mult_path = Path(mult_dir)
        n_fields = len(fids)

        # Start from base or defaults
        if base is not None:
            params = base.copy()
        else:
            params = cls(n_fields=n_fields)

        # Map PEST param names to CalibrationParameters attribute names
        # Also track which params go to FieldProperties (returned separately)
        param_map = {
            "ndvi_k": "ndvi_k",
            "ndvi_0": "ndvi_0",
            "swe_alpha": "swe_alpha",
            "swe_beta": "swe_beta",
            "kr_alpha": "kr_damp",  # PEST uses alpha, we use damp
            "ks_alpha": "ks_damp",
        }

        # Read each parameter file
        for pest_name, attr_name in param_map.items():
            for i, fid in enumerate(fids):
                # PEST++ file naming convention
                csv_file = mult_path / f"p_{pest_name}_{fid}_0_constant.csv"
                if csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file, index_col=0, header=0)
                        # Value is in the column named '1' (last column)
                        value = float(df.iloc[0]["1"])
                        arr = getattr(params, attr_name)
                        arr[i] = value
                    except Exception:
                        pass  # Use default if file can't be read

        return params


def load_pest_mult_properties(
    mult_dir: str,
    fids: list[str],
    base_props: FieldProperties,
) -> FieldProperties:
    """Update FieldProperties with values from PEST++ mult files.

    PEST tunes 'aw' (awc) and 'mad' (mad) which are stored
    in FieldProperties, not CalibrationParameters.

    Parameters
    ----------
    mult_dir : str
        Path to the mult/ directory containing CSV files
    fids : list[str]
        Field IDs in simulation order
    base_props : FieldProperties
        Base properties to update

    Returns
    -------
    FieldProperties
        Updated properties with PEST values applied
    """
    import pandas as pd
    from pathlib import Path

    mult_path = Path(mult_dir)
    n_fields = len(fids)

    # Copy base properties
    props = FieldProperties(
        n_fields=n_fields,
        fids=base_props.fids.copy() if base_props.fids is not None else None,
        awc=base_props.awc.copy(),
        ksat=base_props.ksat.copy(),
        rew=base_props.rew.copy(),
        tew=base_props.tew.copy(),
        cn2=base_props.cn2.copy(),
        zr_max=base_props.zr_max.copy(),
        zr_min=base_props.zr_min.copy(),
        mad=base_props.mad.copy(),
        irr_status=base_props.irr_status.copy(),
        perennial=base_props.perennial.copy(),
        gw_status=base_props.gw_status.copy(),
        ke_max=base_props.ke_max.copy(),
        kc_max=base_props.kc_max.copy(),
        f_sub=base_props.f_sub.copy(),
    )

    # Property params from PEST
    property_params = {
        "aw": "awc",
        "mad": "mad",
    }

    for pest_name, attr_name in property_params.items():
        for i, fid in enumerate(fids):
            csv_file = mult_path / f"p_{pest_name}_{fid}_0_constant.csv"
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file, index_col=0, header=0)
                    value = float(df.iloc[0]["1"])
                    arr = getattr(props, attr_name)
                    arr[i] = value
                except Exception:
                    pass  # Use default if file can't be read

    return props
