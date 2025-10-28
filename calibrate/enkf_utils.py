import os
import os
import json
import copy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import pandas as pd

from swim.config import ProjectConfig
from swim.sampleplots import SamplePlots
from model.obs_field_cycle import field_day_loop
from model.tracker import TUNABLE_PARAMS
from model import TRACKER_PARAMS




def _pred_at_dt_from_state(config, plots, start_dt, end_dt, params, state_in):
    cfg = copy.copy(config)
    cfg.start_dt = start_dt
    cfg.end_dt = end_dt
    kc, swe = field_day_loop(cfg, plots, debug_flag=False, params=params, state_in=state_in,
                             capture_state=False, single_fid_idx=0)
    val = float(kc[-1, 0])
    return val


def _make_ensemble_plots_from_single(single_plots: SamplePlots, fid: str, ne: int) -> SamplePlots:
    """Replicate a single-fid SamplePlots into ne synthetic fields (ensemble).

    The returned SamplePlots has order = [f"{fid.lower()}__e{i}"] and all
    per-day arrays replicated across the ensemble dimension.
    """
    sp = SamplePlots()
    base = single_plots.input
    efids = [f"{fid.lower()}__e{i}" for i in range(ne)]

    ts_sub = {}
    for dt, vals in base['time_series'].items():
        d = {}
        for k, v in vals.items():
            if k == 'doy':
                d[k] = v
            else:
                # single-fid inputs hold lists of length 1; replicate to ne
                vv = v[0] if isinstance(v, (list, tuple, np.ndarray)) else v
                d[k] = [vv] * ne
        ts_sub[dt] = d

    props_sub = {}
    if 'props' in base and fid in base['props']:
        for ef in efids:
            props_sub[ef] = base['props'][fid]

    irr_sub = {}
    if 'irr_data' in base and fid in base['irr_data']:
        for ef in efids:
            irr_sub[ef] = base['irr_data'][fid]

    gwsub_sub = {}
    if 'gwsub_data' in base and fid in base['gwsub_data']:
        for ef in efids:
            gwsub_sub[ef] = base['gwsub_data'][fid]

    kc_max_sub = {}
    if 'kc_max' in base and fid in base['kc_max']:
        for ef in efids:
            kc_max_sub[ef] = base['kc_max'][fid]

    ke_max_sub = {}
    if 'ke_max' in base and fid in base['ke_max']:
        for ef in efids:
            ke_max_sub[ef] = base['ke_max'][fid]

    sp.input = {
        'order': efids,
        'time_series': ts_sub,
        'props': props_sub,
        'irr_data': irr_sub,
        'gwsub_data': gwsub_sub,
        'kc_max': kc_max_sub,
        'ke_max': ke_max_sub,
    }
    return sp


def _preds_ensemble_at_dt_from_state_func(cfg, single_plots, fid, start_dt, end_dt,
                                          theta_mat, enkf_params, base_params_by_fid, state0):
    """Vectorized ensemble propagation for a single field using replicated inputs.

    Builds ne synthetic fields (one per ensemble member), sets per-member
    forecast parameters, broadcasts the initial state, and returns the last-day
    Kc for all ensemble members as a 1D array of length ne.
    """
    ne = int(theta_mat.shape[0])
    cfg2 = copy.copy(cfg)
    cfg2.start_dt = start_dt
    cfg2.end_dt = end_dt
    cfg2.calibrate = False
    cfg2.forecast = True

    plots_e = _make_ensemble_plots_from_single(single_plots, fid, ne)

    # Baseline per-parameter value for this fid when not in enkf_params
    fid_l = fid.lower()
    base_by_group = {}
    for p in TUNABLE_PARAMS:
        key = f'{p}_{fid_l}'
        if base_params_by_fid is not None and key in base_params_by_fid:
            base_by_group[p] = float(base_params_by_fid[key])
        else:
            # fallback to zero if not provided (tracker defaults or caps apply)
            base_by_group[p] = 0.0

    efids = [f"{fid_l}__e{i}" for i in range(ne)]
    fp = {}
    # Fill per-ensemble parameters
    for i, ef in enumerate(efids):
        for p in TUNABLE_PARAMS:
            if p in enkf_params:
                j = enkf_params.index(p)
                fp[f'{p}_{ef}'] = float(theta_mat[i, j])
            else:
                fp[f'{p}_{ef}'] = base_by_group[p]

    # Install the ensemble forecast parameters
    cfg2.forecast_parameters = pd.Series(fp)

    kc, swe = field_day_loop(cfg2, plots_e, debug_flag=False, params=None,
                              state_in=state0, capture_state=False, single_fid_idx=None)
    return kc[-1, :]


class _RingStates:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buf = [None] * self.capacity
        self.base = None  # absolute index of buf[0]
        self.count = 0

    def set(self, abs_idx, state):
        if self.base is None:
            self.base = abs_idx
        if abs_idx < self.base:
            return
        while abs_idx - self.base >= self.capacity:
            self.base += 1
            if self.count > 0:
                self.count -= 1
        pos = (abs_idx - self.base) % self.capacity
        self.buf[pos] = state
        if self.count < self.capacity:
            self.count += 1

    def get(self, abs_idx):
        if self.base is None:
            return None
        if abs_idx < self.base or abs_idx >= self.base + self.count:
            return None
        pos = (abs_idx - self.base) % self.capacity
        return self.buf[pos]


def _assim_fid_step(cfg, plots, start_dt, end_dt, theta_mat, enkf_params, base_params_by_fid,
                    bounds, q_field, q_global, r_fid, y_obs, fid, stochastic_obs, state0):
    ne = theta_mat.shape[0]

    def _build_params(theta_vec):
        params = dict(base_params_by_fid)
        for j, p in enumerate(enkf_params):
            key = f'{p}_{fid}'
            if key in params:
                params[key] = float(theta_vec[j])
        return params

    def _apply_bounds_local(vec):
        arr = vec.copy()
        for j, p in enumerate(enkf_params):
            if p in bounds:
                lo, hi = bounds[p]
                arr[j] = np.minimum(np.maximum(arr[j], lo), hi)
        return arr

    # Vectorized ensemble propagation: replicate the single-fid inputs to ne synthetic fields.
    preds = _preds_ensemble_at_dt_from_state_func(cfg, plots, fid, start_dt, end_dt,
                                                  theta_mat, enkf_params, base_params_by_fid, state0)

    y_bar = float(np.mean(preds))
    theta_bar = np.mean(theta_mat, axis=0)
    y_prime = preds - y_bar
    theta_prime = theta_mat - theta_bar

    cov_ty = (theta_prime.T @ y_prime) / max(ne - 1, 1)
    var_y = float((y_prime @ y_prime) / max(ne - 1, 1))

    k_gain = cov_ty / (var_y + r_fid)

    q_vec = np.array([q_field.get(p, q_global[p]) for p in enkf_params])
    noise = np.random.normal(0.0, np.sqrt(q_vec), size=theta_mat.shape)
    if stochastic_obs:
        obs_noise = np.random.normal(0.0, np.sqrt(max(r_fid, 0.0)), size=ne)
        y_eff = y_obs + obs_noise
    else:
        y_eff = np.ones(ne) * y_obs
    innov = y_eff - preds
    theta_upd = theta_mat + innov[:, None] * k_gain[None, :] + noise
    for m in range(ne):
        theta_upd[m, :] = _apply_bounds_local(theta_upd[m, :])

    theta_mean = np.mean(theta_upd, axis=0)
    # Build a single-field forecast parameter set using theta_mean
    fid_l = fid.lower()
    base_by_group = {}
    for p in TUNABLE_PARAMS:
        key = f'{p}_{fid_l}'
        base_by_group[p] = float(base_params_by_fid.get(key, 0.0))
    fp_single = {}
    for j, p in enumerate(enkf_params):
        fp_single[f'{p}_{fid_l}'] = float(theta_mean[j])
    # include non-updated parameters from baseline
    for p in TUNABLE_PARAMS:
        if p not in enkf_params:
            fp_single[f'{p}_{fid_l}'] = base_by_group[p]
    cfg3 = copy.copy(cfg)
    cfg3.forecast_parameters = pd.Series(fp_single)
    cfg3.calibrate = False
    cfg3.forecast = True
    kc_win, states_win = field_day_loop(cfg3, plots, debug_flag=False, params=None,
                                        state_in=state0, capture_state=True, single_fid_idx=0)
    kc_vec = kc_win[:, 0]
    res = {
        'theta': theta_upd,
        'theta_mean': theta_mean,
        'kc_win': kc_vec,
        'states_win': states_win,
    }
    return res


def _field_worker(single_cfg: ProjectConfig,
                  single_plots: SamplePlots,
                  dates,
                  obs_values,
                  enkf_params,
                  theta_init,
                  bounds,
                  q_field,
                  q_global,
                  r_fid,
                  fid,
                  stochastic_obs,
                  lag_days,
                  smooth_within_window,
                  base_params_by_fid,
                  spinup_state,
                  baseline_kc_col):
    """Run full assimilation for one field across all dates in a separate process.

    Returns a dict with the final kc time series, updated theta, and param trace.
    """
    ne = int(theta_init.shape[0])
    npar = int(theta_init.shape[1])
    theta = theta_init.copy()
    n_days = len(dates)
    param_trace = np.full((n_days, npar), np.nan, dtype=float)
    final_kc_col = baseline_kc_col.copy()

    ring = _RingStates(lag_days + 1)
    if spinup_state is not None:
        st0 = {k: spinup_state[k] for k in TRACKER_PARAMS if k in spinup_state}
        ring.set(0, st0)

    fid_l = fid.lower()

    def _apply_bounds_local(vec):
        arr = vec.copy()
        for j, p in enumerate(enkf_params):
            if p in bounds:
                lo, hi = bounds[p]
                arr[j] = np.minimum(np.maximum(arr[j], lo), hi)
        return arr

    def _params_series_from_theta(theta_vec):
        base_by_group = {}
        for p in TUNABLE_PARAMS:
            key = f'{p}_{fid_l}'
            base_by_group[p] = float(base_params_by_fid.get(key, 0.0))
        fps = {}
        for j, p in enumerate(enkf_params):
            fps[f'{p}_{fid_l}'] = float(theta_vec[j])
        for p in TUNABLE_PARAMS:
            if p not in enkf_params:
                fps[f'{p}_{fid_l}'] = base_by_group[p]
        return pd.Series(fps)

    def _ensure_state_local(target_ix, theta_mean_vec):
        st_at_target = ring.get(target_ix)
        if st_at_target is not None:
            return st_at_target

        base = ring.base
        count = ring.count if hasattr(ring, 'count') else 0
        if base is None or count == 0:
            if spinup_state is None:
                return None
            seed_ix = 0
            seed_state = {k: spinup_state[k] for k in TRACKER_PARAMS if k in spinup_state}
        else:
            last_known_ix = base + count - 1
            seed_ix = None
            seed_state = None
            start_scan = min(target_ix, last_known_ix)
            for ix in range(start_scan, base - 1, -1):
                st = ring.get(ix)
                if st is not None:
                    seed_ix = ix
                    seed_state = st
                    break
            if seed_ix is None:
                if spinup_state is None:
                    return None
                seed_ix = 0
                seed_state = {k: spinup_state[k] for k in TRACKER_PARAMS if k in spinup_state}

        if seed_ix == target_ix:
            return seed_state

        cfg2 = copy.copy(single_cfg)
        cfg2.start_dt = dates[seed_ix]
        cfg2.end_dt = dates[target_ix]
        cfg2.calibrate = False
        cfg2.forecast = True
        cfg2.forecast_parameters = _params_series_from_theta(theta_mean_vec)
        kc_seg, swe_seg, states_seg = field_day_loop(cfg2, single_plots, debug_flag=False, params=None,
                                                     state_in=seed_state, capture_state=True, single_fid_idx=0)
        for k, st in enumerate(states_seg):
            ring.set(seed_ix + k, st)
        return ring.get(target_ix)

    # Iterate dates once for this field
    for t_idx, dt in enumerate(dates):
        y_obs = obs_values[t_idx]
        if pd.isna(y_obs):
            continue

        start_ix = max(0, t_idx - lag_days + 1)
        theta_bar = np.mean(theta, axis=0)
        state0 = _ensure_state_local(start_ix, theta_bar)
        if state0 is None:
            continue

        preds = _preds_ensemble_at_dt_from_state_func(single_cfg, single_plots, fid,
                                                       dates[start_ix], dt, theta,
                                                       enkf_params, base_params_by_fid, state0)

        y_bar = float(np.mean(preds))
        theta_bar = np.mean(theta, axis=0)
        y_prime = preds - y_bar
        theta_prime = theta - theta_bar

        cov_ty = (theta_prime.T @ y_prime) / max(ne - 1, 1)
        var_y = float((y_prime @ y_prime) / max(ne - 1, 1))

        r = r_fid
        k_gain = cov_ty / (var_y + r)

        q_vec = np.array([q_field.get(p, q_global[p]) for p in enkf_params])
        noise = np.random.normal(0.0, np.sqrt(q_vec), size=theta.shape)
        if stochastic_obs:
            obs_noise = np.random.normal(0.0, np.sqrt(max(r, 0.0)), size=ne)
            y_eff = y_obs + obs_noise
        else:
            y_eff = np.ones(ne) * y_obs
        innov = y_eff - preds
        theta = theta + innov[:, None] * k_gain[None, :] + noise
        for m in range(ne):
            theta[m, :] = _apply_bounds_local(theta[m, :])

        theta_mean = np.mean(theta, axis=0)
        param_trace[t_idx, :] = theta_mean

        if smooth_within_window:
            cfg3 = copy.copy(single_cfg)
            cfg3.start_dt = dates[start_ix]
            cfg3.end_dt = dates[t_idx]
            cfg3.calibrate = False
            cfg3.forecast = True
            cfg3.forecast_parameters = _params_series_from_theta(theta_mean)
            kc_win, swe_win, states_win = field_day_loop(cfg3, single_plots, debug_flag=False, params=None,
                                                         state_in=state0, capture_state=True, single_fid_idx=0)
            final_kc_col[start_ix:t_idx + 1] = kc_win[:, 0]
            for k, st in enumerate(states_win):
                ring.set(start_ix + k, st)

    return {
        'fid': fid,
        'theta': theta,
        'param_trace': param_trace,
        'final_kc_col': final_kc_col,
    }

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
