"""Day loop orchestration for SWIM-RS water balance modeling.

Provides the main simulation loop that steps through each day,
calling physics kernels in the correct sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from swimrs.process.cover_modes import resolve_cover_mode, transpiration_cover_factor
from swimrs.process.kcb_modes import KCB_MODE_LINEAR, resolve_kcb_mode
from swimrs.process.kernels.cover import (
    exposed_soil_fraction,
    fractional_cover,
)
from swimrs.process.kernels.crop_coefficient import kcb_affine, kcb_sigmoid
from swimrs.process.kernels.evaporation import ke_coefficient, kr_damped, kr_reduction
from swimrs.process.kernels.irrigation import groundwater_subsidy, irrigation_demand
from swimrs.process.kernels.irrigation_tracking import (
    redistribute_irrigation_fractions,
    update_irrigation_fraction_l3,
    update_irrigation_fraction_root,
)
from swimrs.process.kernels.root_growth import (
    root_depth_from_kcb,
    root_water_redistribution,
)
from swimrs.process.kernels.runoff import (
    curve_number_adjust,
    scs_runoff,
    scs_runoff_smoothed,
)
from swimrs.process.kernels.snow import (
    albedo_decay,
    degree_day_melt,
    partition_precip,
    snow_water_equivalent,
)
from swimrs.process.kernels.transpiration import ks_damped, ks_stress
from swimrs.process.kernels.water_balance import (
    actual_et,
    deep_percolation,
    layer3_storage,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from swimrs.process.input import SwimInput
    from swimrs.process.state import (
        CalibrationParameters,
        FieldProperties,
        WaterBalanceState,
    )

__all__ = ["run_daily_loop", "DailyOutput", "step_day"]

# Fixed fraction of gross irrigation that forces drainage from the root zone.
# The water balance applies the algebraically equivalent 90% net addition, but
# source tracking first mixes all irrigation and then withdraws this drainage
# proportionally. The drainage is therefore not assumed to be pure irrigation.
IRR_FORCED_DRAINAGE_FRAC = 0.1


def _check_finite_state_arrays(context: str, arrays: dict[str, np.ndarray]) -> None:
    """Raise if any tracked state array contains non-finite values."""
    bad = {
        name: int(np.count_nonzero(~np.isfinite(values)))
        for name, values in arrays.items()
        if values is not None and np.count_nonzero(~np.isfinite(values)) > 0
    }
    if bad:
        details = ", ".join(f"{name}={count}" for name, count in sorted(bad.items()))
        raise ValueError(f"Non-finite state detected at {context}: {details}")


def _check_irrigation_source_balance(
    irrigation_storage_before: np.ndarray,
    irrigation_input: np.ndarray,
    irrigation_et: np.ndarray,
    irrigation_dperc: np.ndarray,
    irrigation_storage_after: np.ndarray,
) -> None:
    """Require irrigation-source mass to close for every field and day."""
    residual = (
        irrigation_storage_before
        + irrigation_input
        - irrigation_et
        - irrigation_dperc
        - irrigation_storage_after
    )
    scale = np.maximum(
        1.0,
        np.abs(irrigation_storage_before)
        + np.abs(irrigation_input)
        + np.abs(irrigation_et)
        + np.abs(irrigation_dperc)
        + np.abs(irrigation_storage_after),
    )
    tolerance = 1e-8 + 1e-10 * scale
    bad = np.abs(residual) > tolerance
    if np.any(bad):
        bad_indices = np.flatnonzero(bad)
        max_idx = int(np.argmax(np.abs(residual)))
        raise ValueError(
            "Irrigation-source balance failed: "
            f"max residual={residual[max_idx]:.12g} mm at field index {max_idx}; "
            f"failing field indices={bad_indices[:10].tolist()}"
        )


def _check_irrigation_fraction_bounds(context: str, fractions: dict[str, np.ndarray]) -> None:
    """Require finite irrigation-source fractions within physical bounds."""
    _check_finite_state_arrays(context, fractions)
    tolerance = 1e-10
    for name, values in fractions.items():
        bad = (values < -tolerance) | (values > 1.0 + tolerance)
        if np.any(bad):
            bad_indices = np.flatnonzero(bad)
            raise ValueError(
                f"Irrigation-source fraction outside [0, 1] at {context}: "
                f"{name} failing field indices={bad_indices[:10].tolist()}"
            )


def _enforce_post_redistribution_invariants(
    *,
    context: str,
    depl_root: np.ndarray,
    taw_root: np.ndarray,
    daw3: np.ndarray | None,
    taw3: np.ndarray | None,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Bound post-redistribution storage and fail fast on non-finite state."""
    depl_root = np.clip(depl_root, 0.0, taw_root)
    if taw3 is not None:
        taw3 = np.maximum(taw3, 0.0)
    if daw3 is not None and taw3 is not None:
        daw3 = np.clip(daw3, 0.0, taw3)

    arrays_to_check = {"depl_root": depl_root, "taw_root": taw_root}
    if daw3 is not None:
        arrays_to_check["daw3"] = daw3
    if taw3 is not None:
        arrays_to_check["taw3"] = taw3
    if extra_arrays:
        arrays_to_check.update(extra_arrays)
    _check_finite_state_arrays(context, arrays_to_check)
    return depl_root, daw3, taw3


@dataclass
class DailyOutput:
    """Container for daily output arrays.

    All arrays have shape (n_days, n_fields).

    Attributes
    ----------
    n_days : int
        Number of simulation days
    n_fields : int
        Number of fields
    eta : NDArray[np.float64]
        Actual ET (mm/day)
    evap : NDArray[np.float64]
        Soil evaporation component of ETa (mm/day); transpiration = eta - evap
    etf : NDArray[np.float64]
        ET fraction (ETa/ETr)
    kcb : NDArray[np.float64]
        Basal crop coefficient
    ke : NDArray[np.float64]
        Evaporation coefficient
    ks : NDArray[np.float64]
        Water stress coefficient
    kr : NDArray[np.float64]
        Evaporation reduction coefficient
    runoff : NDArray[np.float64]
        Surface runoff (mm)
    rain : NDArray[np.float64]
        Liquid precipitation (mm)
    melt : NDArray[np.float64]
        Snowmelt (mm)
    swe : NDArray[np.float64]
        Snow water equivalent (mm)
    depl_root : NDArray[np.float64]
        Root zone depletion (mm)
    dperc : NDArray[np.float64]
        Deep percolation (mm)
    irr_sim : NDArray[np.float64]
        Simulated irrigation (mm)
    gw_sim : NDArray[np.float64]
        Groundwater subsidy (mm)
    et_irr : NDArray[np.float64]
        ET from irrigation water (mm) - consumptive use
    dperc_irr : NDArray[np.float64]
        Deep percolation of irrigation water (mm) - return flow
    irr_frac_root : NDArray[np.float64]
        Irrigation fraction in the root-zone source-tracking reservoir [0, 1]
    irr_frac_l3 : NDArray[np.float64]
        Irrigation fraction in layer 3 [0, 1]
    """

    n_days: int
    n_fields: int
    eta: NDArray[np.float64] = field(default=None)
    evap: NDArray[np.float64] = field(default=None)
    etf: NDArray[np.float64] = field(default=None)
    kcb: NDArray[np.float64] = field(default=None)
    ke: NDArray[np.float64] = field(default=None)
    ks: NDArray[np.float64] = field(default=None)
    kr: NDArray[np.float64] = field(default=None)
    runoff: NDArray[np.float64] = field(default=None)
    rain: NDArray[np.float64] = field(default=None)
    melt: NDArray[np.float64] = field(default=None)
    swe: NDArray[np.float64] = field(default=None)
    depl_root: NDArray[np.float64] = field(default=None)
    dperc: NDArray[np.float64] = field(default=None)
    irr_sim: NDArray[np.float64] = field(default=None)
    gw_sim: NDArray[np.float64] = field(default=None)
    et_irr: NDArray[np.float64] = field(default=None)
    dperc_irr: NDArray[np.float64] = field(default=None)
    irr_frac_root: NDArray[np.float64] = field(default=None)
    irr_frac_l3: NDArray[np.float64] = field(default=None)
    daw3: NDArray[np.float64] = field(default=None)
    zr: NDArray[np.float64] = field(default=None)
    depl_ze: NDArray[np.float64] = field(default=None)

    def __post_init__(self):
        """Initialize output arrays."""
        shape = (self.n_days, self.n_fields)
        if self.eta is None:
            self.eta = np.zeros(shape, dtype=np.float64)
        if self.evap is None:
            self.evap = np.zeros(shape, dtype=np.float64)
        if self.etf is None:
            self.etf = np.zeros(shape, dtype=np.float64)
        if self.kcb is None:
            self.kcb = np.zeros(shape, dtype=np.float64)
        if self.ke is None:
            self.ke = np.zeros(shape, dtype=np.float64)
        if self.ks is None:
            self.ks = np.zeros(shape, dtype=np.float64)
        if self.kr is None:
            self.kr = np.zeros(shape, dtype=np.float64)
        if self.runoff is None:
            self.runoff = np.zeros(shape, dtype=np.float64)
        if self.rain is None:
            self.rain = np.zeros(shape, dtype=np.float64)
        if self.melt is None:
            self.melt = np.zeros(shape, dtype=np.float64)
        if self.swe is None:
            self.swe = np.zeros(shape, dtype=np.float64)
        if self.depl_root is None:
            self.depl_root = np.zeros(shape, dtype=np.float64)
        if self.dperc is None:
            self.dperc = np.zeros(shape, dtype=np.float64)
        if self.irr_sim is None:
            self.irr_sim = np.zeros(shape, dtype=np.float64)
        if self.gw_sim is None:
            self.gw_sim = np.zeros(shape, dtype=np.float64)
        if self.et_irr is None:
            self.et_irr = np.zeros(shape, dtype=np.float64)
        if self.dperc_irr is None:
            self.dperc_irr = np.zeros(shape, dtype=np.float64)
        if self.irr_frac_root is None:
            self.irr_frac_root = np.zeros(shape, dtype=np.float64)
        if self.irr_frac_l3 is None:
            self.irr_frac_l3 = np.zeros(shape, dtype=np.float64)
        if self.daw3 is None:
            self.daw3 = np.zeros(shape, dtype=np.float64)
        if self.zr is None:
            self.zr = np.zeros(shape, dtype=np.float64)
        if self.depl_ze is None:
            self.depl_ze = np.zeros(shape, dtype=np.float64)


def run_daily_loop(
    swim_input: SwimInput,
    parameters: CalibrationParameters | None = None,
    properties: FieldProperties | None = None,
    cover_scaling: bool | None = None,
    stress_depletion_fraction: float | None = None,
    cover_mode: str | int | None = None,
    kcb_mode: str | int | None = None,
) -> tuple[DailyOutput, WaterBalanceState]:
    """Run the daily water balance simulation loop.

    Parameters
    ----------
    swim_input : SwimInput
        Input data container (HDF5-backed)
    parameters : CalibrationParameters, optional
        Calibration parameters. If not provided, uses swim_input.parameters.
    properties : FieldProperties, optional
        Field properties. If not provided, uses swim_input.properties.
        Pass custom properties to use PEST++ calibrated values (awc, mad).

    Returns
    -------
    output : DailyOutput
        Daily output arrays
    final_state : WaterBalanceState
        Final state after simulation
    """
    if cover_mode is None and cover_scaling is None:
        cover_mode = getattr(swim_input, "cover_mode", None)
        cover_scaling = getattr(swim_input, "cover_scaling", None)
    cover_mode_code = resolve_cover_mode(cover_mode, cover_scaling)
    cover_lin_lo, cover_lin_hi = getattr(swim_input, "cover_linear_ndvi_range", (None, None))
    if kcb_mode is None:
        kcb_mode = getattr(swim_input, "kcb_mode", None)
    kcb_mode_code = resolve_kcb_mode(kcb_mode)
    if stress_depletion_fraction is None:
        stress_depletion_fraction = getattr(swim_input, "stress_depletion_fraction", None)

    params = parameters if parameters is not None else swim_input.parameters
    props = properties if properties is not None else swim_input.properties

    n_days = swim_input.n_days
    n_fields = swim_input.n_fields

    # Initialize state from spinup
    state = swim_input.spinup_state.copy()

    # Initialize output
    output = DailyOutput(n_days=n_days, n_fields=n_fields)

    # Pre-load all time series data to avoid per-day HDF5 reads
    # This dramatically improves performance (7 reads instead of 7*n_days)
    all_ndvi = swim_input.get_time_series("ndvi")  # (n_days, n_fields)
    all_ref_et = swim_input.get_time_series("ref_et")
    all_prcp = swim_input.get_time_series("prcp")
    all_tmin = swim_input.get_time_series("tmin")
    all_tmax = swim_input.get_time_series("tmax")
    all_srad = swim_input.get_time_series("srad")
    all_irr_flag = swim_input.get_irr_flag()  # (n_days, n_fields)

    # Track year-specific f_sub for groundwater subsidy
    has_year_gwsub = swim_input.has_year_specific_gwsub()
    current_year = None
    current_f_sub = None

    # Daily loop
    for day_idx in range(n_days):
        # Get year-specific f_sub if available (matches legacy day_data.py behavior)
        if has_year_gwsub:
            current_date = swim_input.get_date(day_idx)
            if current_date.year != current_year:
                current_year = current_date.year
                current_f_sub = swim_input.get_f_sub_for_year(current_year)

        # Get daily inputs from pre-loaded arrays
        ndvi = all_ndvi[day_idx, :]
        etr = all_ref_et[day_idx, :]
        prcp = all_prcp[day_idx, :]
        tmin = all_tmin[day_idx, :]
        tmax = all_tmax[day_idx, :]
        srad = all_srad[day_idx, :]
        irr_flag = all_irr_flag[day_idx, :]

        # Step the simulation
        day_out = step_day(
            state=state,
            props=props,
            params=params,
            ndvi=ndvi,
            etr=etr,
            prcp=prcp,
            tmin=tmin,
            tmax=tmax,
            srad=srad,
            irr_flag=irr_flag,
            f_sub=current_f_sub,
            cover_mode=cover_mode_code,
            cover_linear_ndvi_range=(cover_lin_lo, cover_lin_hi),
            kcb_mode=kcb_mode_code,
            stress_depletion_fraction=stress_depletion_fraction,
        )

        # Store outputs
        output.eta[day_idx, :] = day_out["eta"]
        output.evap[day_idx, :] = day_out["evap"]
        output.etf[day_idx, :] = day_out["etf"]
        output.kcb[day_idx, :] = day_out["kcb"]
        output.ke[day_idx, :] = day_out["ke"]
        output.ks[day_idx, :] = day_out["ks"]
        output.kr[day_idx, :] = day_out["kr"]
        output.runoff[day_idx, :] = day_out["runoff"]
        output.rain[day_idx, :] = day_out["rain"]
        output.melt[day_idx, :] = day_out["melt"]
        output.swe[day_idx, :] = day_out["swe"]
        output.depl_root[day_idx, :] = day_out["depl_root"]
        output.dperc[day_idx, :] = day_out["dperc"]
        output.irr_sim[day_idx, :] = day_out["irr_sim"]
        output.gw_sim[day_idx, :] = day_out["gw_sim"]
        output.et_irr[day_idx, :] = day_out["et_irr"]
        output.dperc_irr[day_idx, :] = day_out["dperc_irr"]
        output.irr_frac_root[day_idx, :] = day_out["irr_frac_root"]
        output.irr_frac_l3[day_idx, :] = day_out["irr_frac_l3"]

    return output, state


def step_day(
    state: WaterBalanceState,
    props: FieldProperties,
    params: CalibrationParameters,
    ndvi: NDArray[np.float64],
    etr: NDArray[np.float64],
    prcp: NDArray[np.float64],
    tmin: NDArray[np.float64],
    tmax: NDArray[np.float64],
    srad: NDArray[np.float64],
    irr_flag: NDArray[np.bool_],
    f_sub: NDArray[np.float64] | None = None,
    cover_mode: str | int | None = None,
    cover_linear_ndvi_range: tuple[float | None, float | None] = (None, None),
    kcb_mode: str | int | None = None,
    stress_depletion_fraction: float | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Execute a single daily time step.

    This function orchestrates all the physics kernels in the correct
    sequence for one day of simulation.

    Parameters
    ----------
    state : WaterBalanceState
        Current state (modified in-place)
    props : FieldProperties
        Static field properties
    params : CalibrationParameters
        Calibration parameters
    ndvi : (n_fields,)
        NDVI values
    etr : (n_fields,)
        Reference ET (mm/day)
    prcp : (n_fields,)
        Precipitation (mm)
    tmin : (n_fields,)
        Minimum temperature (°C)
    tmax : (n_fields,)
        Maximum temperature (°C)
    srad : (n_fields,)
        Solar radiation (W/m²), daily mean downward shortwave radiation
    irr_flag : (n_fields,)
        Irrigation flag for this day
    f_sub : (n_fields,), optional
        Year-specific groundwater subsidy fraction. If provided, overrides
        props.f_sub for this time step.

    Returns
    -------
    dict
        Daily output values
    """
    n = state.n_fields
    temp_avg = (tmin + tmax) / 2.0

    # Save initial state for irrigation fraction tracking
    # These are the values BEFORE today's fluxes
    depl_root_before = state.depl_root.copy()
    daw3_before = state.daw3.copy()
    irr_frac_root_before = state.irr_frac_root.copy()
    irr_frac_l3_before = state.irr_frac_l3.copy()
    taw = props.compute_taw(state.zr)
    source_capacity = props.compute_source_water_capacity(state.zr)
    root_water_before = np.maximum(source_capacity - state.depl_root, 0.0)
    irrigation_storage_before = (
        root_water_before * irr_frac_root_before + daw3_before * irr_frac_l3_before
    )

    # 1. Snow partitioning and melt
    rain, snow = partition_precip(prcp, temp_avg)

    # Update albedo
    state.albedo = albedo_decay(state.albedo, snow)

    # Calculate melt.
    # NOTE: Legacy model updates SWE with today's snowfall BEFORE computing melt
    # (compute_snow.py: foo.swe += sf, then melt = min(foo.swe, melt_potential)).
    swe_before_melt = state.swe + snow
    melt = degree_day_melt(
        swe_before_melt, tmax, temp_avg, srad, state.albedo, params.swe_alpha, params.swe_beta
    )

    # Update SWE
    swe_new = snow_water_equivalent(state.swe, snow, melt)
    state.swe = swe_new

    # Effective precipitation (rain + melt)
    precip_eff = rain + melt

    # 2. Runoff: SCS Curve Number with antecedent moisture adjustment
    cn_adj = curve_number_adjust(props.cn2, state.depl_ze, props.rew, props.tew)

    if np.any(props.irr_status):
        # Irrigated fields: use smoothed runoff over 4-day S history
        runoff_std, s_new = scs_runoff(precip_eff, cn_adj)
        runoff_smooth = scs_runoff_smoothed(precip_eff, state.s1, state.s2, state.s3, state.s4)
        runoff = np.where(props.irr_status, runoff_smooth, runoff_std)

        # Shift S history (newest to oldest: s_new -> s1 -> s2 -> s3 -> s4)
        # Legacy model sets s1 to today's S (runoff.py: foo.s1 = foo.s).
        state.s4 = state.s3.copy()
        state.s3 = state.s2.copy()
        state.s2 = state.s1.copy()
        state.s1 = s_new.copy()
        state.s = s_new
    else:
        # Non-irrigated: standard SCS runoff
        runoff, s_new = scs_runoff(precip_eff, cn_adj)
        state.s = s_new

    # Net infiltration
    infiltration = precip_eff - runoff

    # 3. Crop coefficient calculation (sigmoid, or the linear NDVI-Kcb relation)
    if resolve_kcb_mode(kcb_mode) == KCB_MODE_LINEAR:
        kcb = kcb_affine(ndvi, props.kc_max, params.ndvi_alpha, params.ndvi_beta)
    else:
        kcb = kcb_sigmoid(ndvi, props.kc_max, params.ndvi_k, params.ndvi_0)

    # 4. Fractional cover from Kcb (FAO-56)
    fc = fractional_cover(kcb, params.kc_min, props.kc_max)
    few = exposed_soil_fraction(fc)

    # Transpiration cover weight. `few` above always stays on the Kcb-derived
    # fc; only the transpiration term follows the configured cover mode.
    lin_lo, lin_hi = cover_linear_ndvi_range
    fc_t = transpiration_cover_factor(
        resolve_cover_mode(cover_mode),
        np.ascontiguousarray(ndvi, dtype=np.float64),
        np.ascontiguousarray(fc, dtype=np.float64),
        np.ascontiguousarray(kcb / props.kc_max, dtype=np.float64),
        np.ascontiguousarray(params.ndvi_k, dtype=np.float64),
        np.ascontiguousarray(params.ndvi_0, dtype=np.float64),
        -1.0 if lin_lo is None else float(lin_lo),
        -1.0 if lin_hi is None else float(lin_hi),
    )

    # 5. Calculate new root depth from kcb (but don't apply redistribution yet)
    # Legacy model: root growth is applied at END of daily loop
    zr_prev = state.zr.copy()
    zr_new = root_depth_from_kcb(kcb, params.kc_min, props.kc_max, props.zr_max, props.zr_min)

    # For perennials, keep root depth constant at max
    zr_new = np.where(props.perennial, props.zr_max, zr_new)

    # 6. Compute TAW and RAW using PREVIOUS day's root depth (matches legacy)
    # Legacy model compute_field_et.py line 76-79 uses swb.zr (not updated yet)
    # `taw` was computed from the previous root depth with the source-storage
    # snapshot, so hydrologic and tracer states use the same depletion.
    raw = props.compute_raw(taw)
    # WP-C7 mad-split: when configured, the Ks stress onset uses a fixed
    # FAO-56 p (stress depletion fraction), decoupling it from the mad-driven
    # irrigation trigger / gw-subsidy which continue to use `raw`.
    if stress_depletion_fraction is None:
        raw_stress = raw
    else:
        raw_stress = stress_depletion_fraction * taw

    # 7. Update surface layer (Ze) with water inputs
    # Matches legacy model compute_field_et.py line 64:
    # swb.depl_ze = swb.depl_ze - (swb.melt + swb.rain + swb.irr_sim)
    # Note: uses previous day's irr_sim (stored in state.prev_irr_sim)
    prev_irr = state.prev_irr_sim if hasattr(state, "prev_irr_sim") else np.zeros(n)
    state.depl_ze = state.depl_ze - (melt + rain + prev_irr)
    state.depl_ze = np.maximum(state.depl_ze, 0.0)  # Can't be negative

    # 9. Calculate base Kr and Ks (using current depletion)
    kr_base = kr_reduction(props.tew, state.depl_ze, props.rew)
    ks_base = ks_stress(taw, state.depl_root, raw_stress)

    # Apply damping
    kr_new = kr_damped(kr_base, state.kr, params.kr_damp)
    ks_new = ks_damped(ks_base, state.ks, params.ks_damp)

    state.kr = kr_new
    state.ks = ks_new

    # 10. Calculate evaporation coefficient
    ke = ke_coefficient(kr_new, props.kc_max, kcb, few, props.ke_max)

    # 11. Calculate actual ET and evaporation
    kc_act, eta = actual_et(ks_new, kcb, fc_t, ke, props.kc_max, etr)
    evap = ke * etr  # Soil evaporation component

    # 11a. Constrain ET to available water (prevents phantom ET)
    # Available water = current storage + expected inputs
    # Current storage = TAW - depl_root_before
    # Inputs = infiltration + irrigation + groundwater (estimated)
    # We use a preliminary estimate of irr/gw since they depend on post-ET depletion
    available_for_et = (taw - depl_root_before) + infiltration
    # Add safety margin for expected irrigation/gw on irrigated/gw fields
    # (actual irr/gw calculated later, but we don't want to over-constrain)
    # For non-irrigated/non-gw fields, this constraint ensures ET <= available water
    eta_uncapped = eta
    eta = np.minimum(eta, np.maximum(0.0, available_for_et))
    # The cap reduces E and T proportionally: scale evap to the share actually
    # contained in the capped eta, so the Ze update and the TEW adjustment
    # below never account for evaporation that didn't happen
    evap = evap * (eta / np.maximum(eta_uncapped, 1e-12))

    # 12. Update Ze with evaporation
    # Matches legacy model compute_field_et.py lines 97-107
    depl_ze_prev = state.depl_ze.copy()
    state.depl_ze = depl_ze_prev + evap
    state.depl_ze = np.maximum(state.depl_ze, 0.0)

    # Cap evaporation at TEW and adjust eta accordingly
    if np.any(state.depl_ze > props.tew):
        potential_e = state.depl_ze - depl_ze_prev
        potential_e = np.maximum(potential_e, 1e-4)
        e_factor = 1.0 - (state.depl_ze - props.tew) / potential_e
        e_factor = np.clip(e_factor, 0.0, 1.0)
        evap_before_cap = evap.copy()
        evap = evap * e_factor
        eta = eta - (evap_before_cap - evap)
        eta = np.maximum(eta, 0.0)
        state.depl_ze = np.where(
            state.depl_ze > props.tew, np.maximum(depl_ze_prev, 0.0) + evap, state.depl_ze
        )

    # Calculate ETf with safe division (avoid invalid divide warnings)
    etf = np.zeros_like(eta)
    np.divide(eta, etr, out=etf, where=etr > 0)

    # 13. First update depletion with ET and infiltration ONLY
    # This matches legacy model compute_field_et.py line 114:
    # swb.depl_root += swb.etc_act - swb.ppt_inf
    depl_after_et = state.depl_root + eta - infiltration

    # 14. Irrigation demand (based on UPDATED depletion)
    irr_sim, irr_cont_new, next_irr_new = irrigation_demand(
        depl_after_et,
        raw,
        params.max_irr_rate,
        irr_flag,
        temp_avg,
        state.irr_continue,
        state.next_day_irr,
    )
    state.irr_continue = irr_cont_new
    state.next_day_irr = next_irr_new

    # 15. Groundwater subsidy (based on UPDATED depletion)
    # Matches legacy model compute_field_et.py lines 150-152
    # Use year-specific f_sub if provided, otherwise fall back to props.f_sub
    f_sub_to_use = f_sub if f_sub is not None else props.f_sub
    gw_sim = groundwater_subsidy(depl_after_et, raw, props.gw_status, f_sub_to_use)

    # 16. Apply irrigation and groundwater subsidy. Applying 90% as a net
    # root-zone addition is algebraically equivalent to mixing all irrigation
    # and then withdrawing the fixed 10% irrigation-forced drainage in step 19.
    irr_net_to_root = (1.0 - IRR_FORCED_DRAINAGE_FRAC) * irr_sim
    depl_new = depl_after_et - irr_net_to_root - gw_sim

    # 17. Deep percolation (excess water when depl < 0)
    dperc, depl_after_perc = deep_percolation(depl_new)

    # 18. Cap depletion at TAW (can't deplete more than available)
    # Matches legacy model compute_field_et.py line 163:
    # swb.depl_root = np.where(swb.depl_root > swb.taw, swb.taw, swb.depl_root)
    state.depl_root = np.minimum(depl_after_perc, taw)

    # 19. Layer 3 storage update. The fixed irrigation-forced drainage and any
    # saturation drainage both leave the mixed root-zone reservoir. Source
    # composition is calculated proportionally below.
    irr_forced_drainage = IRR_FORCED_DRAINAGE_FRAC * irr_sim
    root_to_l3 = dperc + irr_forced_drainage

    if state.taw3 is not None and np.any(state.taw3 > 0):
        state.daw3, dperc_out = layer3_storage(state.daw3, state.taw3, root_to_l3)
    else:
        dperc_out = root_to_l3

    # === IRRIGATION FRACTION TRACKING ===
    # Track the fraction of soil water that originated from irrigation.
    # This enables consumptive use accounting for water rights.

    # 19a. Update root zone irrigation fraction
    # Natural infiltration mixes before ET. Gross irrigation then mixes before
    # both irrigation-forced and saturation drainage withdraw proportionally.
    state.irr_frac_root, et_irr, root_to_l3_irr = update_irrigation_fraction_root(
        root_water_before,
        irr_frac_root_before,
        infiltration,
        irr_sim,
        gw_sim,
        eta,
        root_to_l3,
    )

    # 19b. Update layer 3 irrigation fraction
    state.irr_frac_l3, dperc_irr = update_irrigation_fraction_l3(
        daw3_before, irr_frac_l3_before, root_to_l3, root_to_l3_irr, dperc_out
    )

    # 20. Root growth water redistribution (at END of daily loop, matches legacy)
    # Legacy model grow_root.py is called at line 202 of compute_field_et.py,
    # after all ET/irrigation/dperc calculations

    # Save pre-redistribution values for fraction tracking
    depl_root_pre_redist = state.depl_root.copy()
    daw3_pre_redist = state.daw3.copy()
    irr_frac_root_pre_redist = state.irr_frac_root.copy()
    irr_frac_l3_pre_redist = state.irr_frac_l3.copy()

    state.depl_root, state.daw3, state.taw3 = root_water_redistribution(
        zr_new, zr_prev, props.zr_max, props.awc, depl_root_pre_redist, daw3_pre_redist
    )
    state.zr = zr_new
    taw_post = props.compute_taw(state.zr)
    source_capacity_post = props.compute_source_water_capacity(state.zr)
    state.depl_root, state.daw3, state.taw3 = _enforce_post_redistribution_invariants(
        context="step_day/post_redistribution",
        depl_root=state.depl_root,
        taw_root=taw_post,
        daw3=state.daw3,
        taw3=state.taw3,
        extra_arrays={
            "zr": state.zr,
            "kr": state.kr,
            "ks": state.ks,
            "depl_ze": state.depl_ze,
            "irr_frac_root": state.irr_frac_root,
            "irr_frac_l3": state.irr_frac_l3,
        },
    )

    # 20a. Move source mass with the actual layer-3 exchange. The fixed natural
    # shallow-root buffer makes source mixing capacity conservative as the root
    # boundary moves.
    root_water_pre_redist = np.maximum(source_capacity - depl_root_pre_redist, 0.0)
    root_water_after = np.maximum(source_capacity_post - state.depl_root, 0.0)
    state.irr_frac_root, state.irr_frac_l3 = redistribute_irrigation_fractions(
        root_water_pre_redist,
        irr_frac_root_pre_redist,
        daw3_pre_redist,
        irr_frac_l3_pre_redist,
        root_water_after,
        state.daw3,
    )
    _check_irrigation_fraction_bounds(
        "step_day/post_source_redistribution",
        {
            "irr_frac_root": state.irr_frac_root,
            "irr_frac_l3": state.irr_frac_l3,
        },
    )

    irrigation_storage_after = (
        root_water_after * state.irr_frac_root + state.daw3 * state.irr_frac_l3
    )
    _check_irrigation_source_balance(
        irrigation_storage_before,
        irr_sim,
        et_irr,
        dperc_irr,
        irrigation_storage_after,
    )

    # 21. Store irr_sim for next day's Ze update
    state.prev_irr_sim = irr_sim.copy()

    # Return daily outputs
    return {
        "eta": eta,
        "evap": evap,
        "etf": etf,
        "kcb": kcb,
        "ke": ke,
        "ks": ks_new,
        "kr": kr_new,
        "runoff": runoff,
        "rain": rain,
        "melt": melt,
        "swe": state.swe,
        "depl_root": state.depl_root,
        "dperc": dperc_out,
        "irr_sim": irr_sim,
        "gw_sim": gw_sim,
        "et_irr": et_irr,
        "dperc_irr": dperc_irr,
        "irr_frac_root": state.irr_frac_root.copy(),
        "irr_frac_l3": state.irr_frac_l3.copy(),
    }
