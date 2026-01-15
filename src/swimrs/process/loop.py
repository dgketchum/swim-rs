"""Day loop orchestration for SWIM-RS water balance modeling.

Provides the main simulation loop that steps through each day,
calling physics kernels in the correct sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from swimrs.process.kernels.crop_coefficient import kcb_sigmoid
from swimrs.process.kernels.cover import fractional_cover, exposed_soil_fraction
from swimrs.process.kernels.evaporation import kr_reduction, kr_damped, ke_coefficient
from swimrs.process.kernels.transpiration import ks_stress, ks_damped
from swimrs.process.kernels.runoff import scs_runoff
from swimrs.process.kernels.snow import (
    partition_precip,
    albedo_decay,
    degree_day_melt,
    snow_water_equivalent,
)
from swimrs.process.kernels.water_balance import (
    deep_percolation,
    layer3_storage,
    root_zone_depletion,
    actual_et,
)
from swimrs.process.kernels.root_growth import (
    root_depth_from_kcb,
    root_water_redistribution,
)
from swimrs.process.kernels.irrigation import irrigation_demand, groundwater_subsidy

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from swimrs.process.input import SwimInput
    from swimrs.process.state import (
        WaterBalanceState,
        FieldProperties,
        CalibrationParameters,
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

        temp_avg = (tmin + tmax) / 2.0

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

    # 2. Runoff calculation (SCS Curve Number)
    # Note: scs_runoff_smoothed requires historical S values
    # For simplicity, use standard SCS runoff for all fields
    runoff, s = scs_runoff(precip_eff, props.cn2)

    # Net infiltration
    infiltration = precip_eff - runoff

    # 3. Crop coefficient calculation
    kcb = kcb_sigmoid(ndvi, params.kc_max, params.ndvi_k, params.ndvi_0)

    # 4. Fractional cover
    fc = fractional_cover(kcb, params.kc_min, params.kc_max)
    few = exposed_soil_fraction(fc)

    # 5. Root growth
    zr_prev = state.zr.copy()
    zr_new = root_depth_from_kcb(
        kcb, params.kc_min, params.kc_max,
        props.zr_max, props.zr_min
    )

    # For perennials, keep root depth constant at max
    zr_new = np.where(props.perennial, props.zr_max, zr_new)

    # 6. Compute TAW and RAW
    taw = props.compute_taw(zr_new)
    raw = props.compute_raw(taw)

    # 7. Water redistribution for root growth
    state.depl_root, state.daw3, state.taw3 = root_water_redistribution(
        zr_new, zr_prev, props.zr_max, props.awc,
        state.depl_root, state.daw3
    )
    state.zr = zr_new

    # 8. Irrigation demand
    irr_sim, irr_cont_new, next_irr_new = irrigation_demand(
        state.depl_root, raw, params.max_irr_rate,
        irr_flag, temp_avg,
        state.irr_continue, state.next_day_irr
    )
    state.irr_continue = irr_cont_new
    state.next_day_irr = next_irr_new

    # 9. Groundwater subsidy
    gw_sim = groundwater_subsidy(
        state.depl_root, raw, props.gw_status, params.f_sub
    )

    # 10. Calculate base Kr and Ks
    kr_base = kr_reduction(props.tew, state.depl_ze, props.rew)
    ks_base = ks_stress(taw, state.depl_root, raw)

    # Apply damping
    kr_new = kr_damped(kr_base, state.kr, params.kr_damp)
    ks_new = ks_damped(ks_base, state.ks, params.ks_damp)

    state.kr = kr_new
    state.ks = ks_new

    # 11. Calculate evaporation coefficient
    # ke_coefficient(kr, kc_max, kcb, few, ke_max)
    ke = ke_coefficient(kr_new, params.kc_max, kcb, few, params.ke_max)

    # 12. Calculate actual ET
    # actual_et(ks, kcb, fc, ke, kc_max, refet)
    kc_act, eta = actual_et(ks_new, kcb, fc, ke, params.kc_max, etr)

    # Calculate ETf
    etf = np.where(etr > 0, eta / etr, 0.0)

    # 13. Update root zone depletion
    # root_zone_depletion(depl_root_prev, etc_act, ppt_inf, irr_sim, gw_sim)
    depl_new = root_zone_depletion(
        state.depl_root, eta, infiltration, irr_sim, gw_sim
    )

    # 14. Deep percolation (excess water)
    dperc, depl_after_perc = deep_percolation(depl_new)
    state.depl_root = depl_after_perc

    # 15. Layer 3 storage update
    if state.taw3 is not None and np.any(state.taw3 > 0):
        state.daw3, dperc_out = layer3_storage(
            state.daw3, state.taw3, dperc
        )
    else:
        dperc_out = dperc

    # 16. Update surface layer depletion (simplified - track with root zone)
    # In FAO-56, Ze is a separate layer, but we simplify here
    state.depl_ze = np.minimum(state.depl_root, props.tew)

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
