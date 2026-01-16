"""Day loop orchestration for SWIM-RS water balance modeling.

Provides the main simulation loop that steps through each day,
calling physics kernels in the correct sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from swimrs.process.kernels.cover import exposed_soil_fraction, fractional_cover
from swimrs.process.kernels.crop_coefficient import kcb_sigmoid
from swimrs.process.kernels.evaporation import ke_coefficient, kr_damped, kr_reduction
from swimrs.process.kernels.irrigation import groundwater_subsidy, irrigation_demand
from swimrs.process.kernels.root_growth import (
    root_depth_from_kcb,
    root_water_redistribution,
)
from swimrs.process.kernels.runoff import (
    curve_number_adjust,
    infiltration_excess,
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
    """

    n_days: int
    n_fields: int
    eta: NDArray[np.float64] = field(default=None)
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

    def __post_init__(self):
        """Initialize output arrays."""
        shape = (self.n_days, self.n_fields)
        if self.eta is None:
            self.eta = np.zeros(shape, dtype=np.float64)
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


def run_daily_loop(
    swim_input: SwimInput,
    params: CalibrationParameters | None = None,
) -> tuple[DailyOutput, WaterBalanceState]:
    """Run the daily water balance simulation loop.

    Parameters
    ----------
    swim_input : SwimInput
        Input data container (HDF5-backed)
    params : CalibrationParameters, optional
        Calibration parameters. If not provided, uses swim_input.parameters.

    Returns
    -------
    output : DailyOutput
        Daily output arrays
    final_state : WaterBalanceState
        Final state after simulation
    """
    if params is None:
        params = swim_input.parameters

    n_days = swim_input.n_days
    n_fields = swim_input.n_fields
    props = swim_input.properties
    runoff_process = getattr(swim_input, "runoff_process", None) or "cn"

    # Check if hourly precip is available for IER mode
    has_hourly_prcp = swim_input.has_hourly_precip()

    # Initialize state from spinup
    state = swim_input.spinup_state.copy()

    # Initialize output
    output = DailyOutput(n_days=n_days, n_fields=n_fields)

    # Daily loop
    for day_idx in range(n_days):
        # Get daily inputs
        ndvi = swim_input.get_time_series("ndvi", day_idx)
        etr = swim_input.get_time_series("etr", day_idx)
        prcp = swim_input.get_time_series("prcp", day_idx)
        tmin = swim_input.get_time_series("tmin", day_idx)
        tmax = swim_input.get_time_series("tmax", day_idx)
        srad = swim_input.get_time_series("srad", day_idx)
        irr_flag = swim_input.get_irr_flag(day_idx)

        # Get hourly precip if IER mode and available
        prcp_hr = None
        if runoff_process == "ier" and has_hourly_prcp:
            prcp_hr = swim_input.get_hourly_precip(day_idx)

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
            runoff_process=runoff_process,
            prcp_hr=prcp_hr,
        )

        # Store outputs
        output.eta[day_idx, :] = day_out["eta"]
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
    runoff_process: str = "cn",
    prcp_hr: NDArray[np.float64] | None = None,
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
        Solar radiation (MJ/m²/day)
    irr_flag : (n_fields,)
        Irrigation flag for this day
    runoff_process : str
        Runoff mode: 'cn' for Curve Number or 'ier' for infiltration-excess
    prcp_hr : (24, n_fields,), optional
        Hourly precipitation (mm/hr), required for IER mode

    Returns
    -------
    dict
        Daily output values
    """
    n = state.n_fields
    temp_avg = (tmin + tmax) / 2.0

    # 1. Snow partitioning and melt
    rain, snow = partition_precip(prcp, temp_avg)

    # Update albedo
    state.albedo = albedo_decay(state.albedo, snow)

    # Calculate melt
    melt = degree_day_melt(
        state.swe, tmax, temp_avg, srad, state.albedo,
        params.swe_alpha, params.swe_beta
    )

    # Update SWE
    swe_new = snow_water_equivalent(state.swe, snow, melt)
    # Actual melt is the minimum of potential melt and available SWE
    actual_melt = np.minimum(melt, state.swe)
    state.swe = swe_new

    # Effective precipitation (rain + melt)
    precip_eff = rain + actual_melt

    # 2. Runoff calculation based on runoff_process
    if runoff_process == "ier" and prcp_hr is not None:
        # Infiltration-excess method (Hortonian runoff)
        # prcp_hr expected shape: (24, n_fields)
        ksat_hourly = props.ksat / 24.0
        runoff = infiltration_excess(prcp_hr, ksat_hourly)
    else:
        # SCS Curve Number with antecedent moisture adjustment
        cn_adj = curve_number_adjust(
            props.cn2, state.depl_ze, props.rew, props.tew
        )

        if np.any(props.irr_status):
            # Irrigated fields: use smoothed runoff over 4-day S history
            runoff_std, s_new = scs_runoff(precip_eff, cn_adj)
            runoff_smooth = scs_runoff_smoothed(
                precip_eff, state.s1, state.s2, state.s3, state.s4
            )
            runoff = np.where(props.irr_status, runoff_smooth, runoff_std)

            # Shift S history (newest to oldest: s -> s1 -> s2 -> s3 -> s4)
            state.s4 = state.s3.copy()
            state.s3 = state.s2.copy()
            state.s2 = state.s1.copy()
            state.s1 = state.s.copy()
            state.s = s_new
        else:
            # Non-irrigated: standard SCS runoff
            runoff, s_new = scs_runoff(precip_eff, cn_adj)
            state.s = s_new

    # Net infiltration
    infiltration = precip_eff - runoff

    # 3. Crop coefficient calculation
    kcb = kcb_sigmoid(ndvi, params.kc_max, params.ndvi_k, params.ndvi_0)

    # 4. Fractional cover
    fc = fractional_cover(kcb, params.kc_min, params.kc_max)
    few = exposed_soil_fraction(fc)

    # 5. Calculate new root depth from kcb (but don't apply redistribution yet)
    # Legacy model: root growth is applied at END of daily loop
    zr_prev = state.zr.copy()
    zr_new = root_depth_from_kcb(
        kcb, params.kc_min, params.kc_max,
        props.zr_max, props.zr_min
    )

    # For perennials, keep root depth constant at max
    zr_new = np.where(props.perennial, props.zr_max, zr_new)

    # 6. Compute TAW and RAW using PREVIOUS day's root depth (matches legacy)
    # Legacy model compute_field_et.py line 76-79 uses swb.zr (not updated yet)
    taw = props.compute_taw(state.zr)  # Use previous zr, not zr_new
    raw = props.compute_raw(taw)

    # 7. Update surface layer (Ze) with water inputs
    # Matches legacy model compute_field_et.py line 64:
    # swb.depl_ze = swb.depl_ze - (swb.melt + swb.rain + swb.irr_sim)
    # Note: uses previous day's irr_sim (stored in state.prev_irr_sim)
    prev_irr = state.prev_irr_sim if hasattr(state, 'prev_irr_sim') else np.zeros(n)
    state.depl_ze = state.depl_ze - (actual_melt + rain + prev_irr)
    state.depl_ze = np.maximum(state.depl_ze, 0.0)  # Can't be negative

    # 9. Calculate base Kr and Ks (using current depletion)
    kr_base = kr_reduction(props.tew, state.depl_ze, props.rew)
    ks_base = ks_stress(taw, state.depl_root, raw)

    # Apply damping
    kr_new = kr_damped(kr_base, state.kr, params.kr_damp)
    ks_new = ks_damped(ks_base, state.ks, params.ks_damp)

    state.kr = kr_new
    state.ks = ks_new

    # 10. Calculate evaporation coefficient
    ke = ke_coefficient(kr_new, params.kc_max, kcb, few, props.ke_max)

    # 11. Calculate actual ET and evaporation
    kc_act, eta = actual_et(ks_new, kcb, fc, ke, params.kc_max, etr)
    evap = ke * etr  # Soil evaporation component

    # 12. Update Ze with evaporation
    # Matches legacy model compute_field_et.py lines 97-107
    depl_ze_prev = state.depl_ze.copy()
    state.depl_ze = depl_ze_prev + evap
    state.depl_ze = np.maximum(state.depl_ze, 0.0)

    # Cap evaporation at TEW and adjust if exceeded
    if np.any(state.depl_ze > props.tew):
        potential_e = state.depl_ze - depl_ze_prev
        potential_e = np.maximum(potential_e, 1e-4)
        e_factor = 1.0 - (state.depl_ze - props.tew) / potential_e
        e_factor = np.clip(e_factor, 0.0, 1.0)
        evap = evap * e_factor
        # Recalculate ET with adjusted evaporation
        state.depl_ze = np.where(
            state.depl_ze > props.tew,
            np.maximum(depl_ze_prev, 0.0) + evap,
            state.depl_ze
        )

    # Calculate ETf
    etf = np.where(etr > 0, eta / etr, 0.0)

    # 13. First update depletion with ET and infiltration ONLY
    # This matches legacy model compute_field_et.py line 114:
    # swb.depl_root += swb.etc_act - swb.ppt_inf
    depl_after_et = state.depl_root + eta - infiltration

    # 14. Irrigation demand (based on UPDATED depletion)
    irr_sim, irr_cont_new, next_irr_new = irrigation_demand(
        depl_after_et, raw, params.max_irr_rate,
        irr_flag, temp_avg,
        state.irr_continue, state.next_day_irr
    )
    state.irr_continue = irr_cont_new
    state.next_day_irr = next_irr_new

    # 15. Groundwater subsidy (based on UPDATED depletion)
    # Matches legacy model compute_field_et.py lines 150-152
    gw_sim = groundwater_subsidy(
        depl_after_et, raw, props.gw_status, props.f_sub
    )

    # 16. Apply irrigation and groundwater subsidy
    # Matches legacy model compute_field_et.py line 156:
    # swb.depl_root -= (swb.irr_sim + swb.gw_sim)
    depl_new = depl_after_et - irr_sim - gw_sim

    # 17. Deep percolation (excess water when depl < 0)
    dperc, depl_after_perc = deep_percolation(depl_new)

    # 18. Cap depletion at TAW (can't deplete more than available)
    # Matches legacy model compute_field_et.py line 163:
    # swb.depl_root = np.where(swb.depl_root > swb.taw, swb.taw, swb.depl_root)
    state.depl_root = np.minimum(depl_after_perc, taw)

    # 19. Layer 3 storage update with gross deep percolation
    # Matches legacy model compute_field_et.py line 165:
    # gross_dperc = swb.dperc + (0.1 * swb.irr_sim)
    gross_dperc = dperc + 0.1 * irr_sim

    if state.taw3 is not None and np.any(state.taw3 > 0):
        state.daw3, dperc_out = layer3_storage(
            state.daw3, state.taw3, gross_dperc
        )
    else:
        dperc_out = gross_dperc

    # 20. Root growth water redistribution (at END of daily loop, matches legacy)
    # Legacy model grow_root.py is called at line 202 of compute_field_et.py,
    # after all ET/irrigation/dperc calculations
    state.depl_root, state.daw3, state.taw3 = root_water_redistribution(
        zr_new, zr_prev, props.zr_max, props.awc,
        state.depl_root, state.daw3
    )
    state.zr = zr_new

    # 21. Store irr_sim for next day's Ze update
    state.prev_irr_sim = irr_sim.copy()

    # Return daily outputs
    return {
        "eta": eta,
        "etf": etf,
        "kcb": kcb,
        "ke": ke,
        "ks": ks_new,
        "kr": kr_new,
        "runoff": runoff,
        "rain": rain,
        "melt": actual_melt,
        "swe": state.swe,
        "depl_root": state.depl_root,
        "dperc": dperc_out,
        "irr_sim": irr_sim,
        "gw_sim": gw_sim,
    }
