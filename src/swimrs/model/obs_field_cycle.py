import numpy as np
import pandas as pd

from swimrs.model import TRACKER_PARAMS, compute_field_et, obs_kcb_daily
from swimrs.model.day_data import DayData
from swimrs.model.tracker import TUNABLE_PARAMS, SampleTracker

OUTPUT_FMT = [
    "aw",
    "et_act",
    "etref",
    "kc_act",
    "kc_bas",
    "ks",
    "ke",
    "melt",
    "rain",
    "depl_root",
    "depl_ze",
    "dperc",
    "runoff",
    "delta_soil_water",
    "wbal",
    "ppt",
    "snow_fall",
    "taw",
    "taw3",
    "daw3",
    "delta_daw3",
    "swe",
    "tavg",
    "tmax",
    "irrigation",
    "gw_sim",
    "fc",
    "t",
    "e",
    "few",
    "zr",
    "p_rz",
    "p_eft",
    "soil_water",
    "niwr",
    "irr_day",
]


class WaterBalanceError(Exception):
    """Raised when daily water balance residuals exceed tolerance."""


def field_day_loop(
    config,
    plots,
    debug_flag=False,
    params=None,
    state_in=None,
    capture_state=False,
    single_fid_idx=None,
):
    """Run the daily model loop over the configured date range and fields.

    Orchestrates tracker initialization (initial conditions, parameters, soils,
    root depth, Ke/Kc caps), then iterates days building DayData, selecting
    the appropriate NDVI/refET by irrigation status, computing Kcb, and advancing
    the water balance. Returns either detailed per-field dataframes (debug) or
    ETf/SWE arrays for calibration/forward runs.

    Parameters
    - config: ProjectConfig with paths, dates, model options.
    - plots: SamplePlots containing the prepped input JSON.
    - debug_flag: if True, returns dict of per-field DataFrames with many terms.
    - params: optional dict of calibrated parameter overrides.

    Returns
    - If debug_flag: dict[fid -> pd.DataFrame]. Else: (etf_arr, swe_arr) ndarrays.
    """
    etf, swe = None, None

    # Reconcile plots data with parameters in forecast/calibrate mode
    if config.forecast or config.calibrate:
        common_fields, dropped_plots, dropped_params = plots.reconcile_with_parameters(config)
        if not common_fields:
            raise ValueError("No fields found in both plots data and parameter set")

    size = len(plots.input["order"])

    tracker = SampleTracker(config, plots, size)
    tracker.apply_initial_conditions()
    tracker.apply_parameters(params=params)
    tracker.load_root_depth()
    tracker.load_soils()

    # only set kc/ke max if they were not calibrated
    tracker.set_ke_max()
    # tracker.__setattr__('ke_max', np.array([[0.45]]))

    if debug_flag:
        tunable_state = {k: tracker.__getattribute__(k) for k in TUNABLE_PARAMS}
        if size == 1:
            tunable_state = {k: f"{v[0, 0]:.2f}" for k, v in tunable_state.items()}
        else:
            tunable_state = {
                k: [f"{vv:.2f}" for vv in v.flatten()] for k, v in tunable_state.items()
            }
        # pprint(dict(sorted(tunable_state.items())))

    targets = plots.input["order"]

    if len(pd.date_range(config.start_dt, config.end_dt, freq="D")) == len(
        plots.input["time_series"]
    ):
        valid_data = plots.input["time_series"].copy()
    else:
        valid_data = {
            dt: val
            for dt, val in plots.input["time_series"].items()
            if (config.start_dt <= pd.to_datetime(dt) <= config.end_dt)
        }

    if debug_flag:
        tracker.setup_dataframe(targets)
    else:
        time_range = pd.date_range(config.start_dt, config.end_dt, freq="D")
        empty = np.zeros((len(time_range), len(targets))) * np.nan
        etf, swe = empty.copy(), empty.copy()

    day_data = DayData()

    # Apply provided restart state. If single_fid_idx is None, broadcast across all fields.
    if state_in is not None:
        if single_fid_idx is not None:
            # single-field restart
            for k in TRACKER_PARAMS:
                if k in state_in:
                    arr = tracker.__getattribute__(k)
                    arr[0, single_fid_idx] = state_in[k]
                    tracker.__setattr__(k, arr)
        else:
            # broadcast restart to all fields (scalar or 1xN array accepted)
            for k in TRACKER_PARAMS:
                if k in state_in:
                    v = state_in[k]
                    arr = tracker.__getattribute__(k)
                    if np.isscalar(v):
                        arr[:] = v
                    else:
                        vv = np.array(v).reshape(1, -1)
                        if vv.shape[1] != arr.shape[1]:
                            raise ValueError(f"state_in length mismatch for {k}")
                        arr[:] = vv
                    tracker.__setattr__(k, arr)

    states_out = [] if capture_state else None

    for j, (step_dt, vals) in enumerate(valid_data.items()):
        day_data.update_day(step_dt, size, vals["doy"])

        if day_data.doy == 1 or day_data.irr_status is None:
            day_data.update_annual_irrigation(plots)
            day_data.update_annual_groundwater_subsidy(plots)

        day_data.update_daily_irrigation(plots, vals, config)

        day_data.update_daily_inputs(vals, size)

        if capture_state and single_fid_idx is not None:
            snap = {}
            for k in TRACKER_PARAMS:
                v = tracker.__getattribute__(k)
                snap[k] = float(v[0, single_fid_idx])
            states_out.append(snap)

        obs_kcb_daily.kcb_daily(tracker, day_data)

        compute_field_et.compute_field_et(tracker, day_data)

        if debug_flag:
            tracker.update_dataframe(targets, day_data, step_dt)

        else:
            if np.any(np.isnan(tracker.kc_act)):
                raise ValueError("NaN in Kc_act")

            if np.any(np.isnan(tracker.swe)):
                raise ValueError("NaN in SWE")

            etf[j, :] = tracker.kc_act
            swe[j, :] = tracker.swe

    if debug_flag:
        # pass final dataframe to calling script
        tracker.crop_df = {
            fid: pd.DataFrame().from_dict(tracker.crop_df[fid], orient="index")[OUTPUT_FMT]
            for fid in targets
        }

        for fid in tracker.crop_df:
            tracker.crop_df[fid].index = pd.to_datetime(tracker.crop_df[fid].index)

        return tracker.crop_df

    else:
        # if not debug, just return the actual ET and SWE results as ndarray
        if capture_state:
            return etf, swe, states_out
        return etf, swe


if __name__ == "__main__":
    pass
