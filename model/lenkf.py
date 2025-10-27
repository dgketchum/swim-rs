import os
import copy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import pandas as pd

from swim.config import ProjectConfig
from swim.sampleplots import SamplePlots
from model.obs_field_cycle import field_day_loop
from model.tracker import TUNABLE_PARAMS


def _propagate_pred_at_dt(config, plots, end_dt, params, fid, fid_idx):
    cfg = copy.copy(config)
    cfg.end_dt = end_dt
    kc, swe = field_day_loop(cfg, plots, debug_flag=False, params=params)
    val = float(kc[-1, fid_idx])
    return val


def _pred_at_dt_from_state(config, plots, start_dt, end_dt, params, state_in):
    cfg = copy.copy(config)
    cfg.start_dt = start_dt
    cfg.end_dt = end_dt
    kc, swe = field_day_loop(cfg, plots, debug_flag=False, params=params, state_in=state_in, capture_state=False, single_fid_idx=0)
    val = float(kc[-1, 0])
    return val


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

    preds = np.zeros(ne)
    for m in range(ne):
        params_m = _build_params(theta_mat[m, :])
        preds[m] = _pred_at_dt_from_state(cfg, plots, start_dt, end_dt, params_m, state0)

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
    params_mean = _build_params(theta_mean)
    kc_win, states_win = field_day_loop(cfg, plots, debug_flag=False, params=params_mean,
                                        state_in=state0, capture_state=True, single_fid_idx=0)
    kc_vec = kc_win[:, 0]
    res = {
        'theta': theta_upd,
        'theta_mean': theta_mean,
        'kc_win': kc_vec,
        'states_win': states_win,
    }
    return res


class LaggedEnKF:
    def __init__(self, config: ProjectConfig, plots: SamplePlots, obs_df: pd.DataFrame,
                 enkf_params=None, lag_days=7, ensemble_size=50, bounds=None,
                 fields_subset=None, num_workers=None, use_processes=False,
                 stochastic_obs=True, par_fields=True, field_workers=None,
                 smooth_within_window=True):
        self.config = config
        self.plots = plots
        self.obs_df = obs_df
        self.fields = self.plots.input['order']
        self.active_fields = [f for f in (fields_subset if fields_subset is not None else self.fields) if f in self.fields]
        self.field_idx = {f: i for i, f in enumerate(self.fields)}
        self.enkf_params = enkf_params if enkf_params is not None else ['ndvi_k', 'ndvi_0', 'ks_alpha', 'kr_alpha']
        self.lag_days = int(lag_days)
        self.ne = int(ensemble_size)
        self.bounds = bounds if bounds is not None else {}
        self.num_workers = int(num_workers) if num_workers is not None else None
        self.use_processes = bool(use_processes)
        self.stochastic_obs = bool(stochastic_obs)
        self.par_fields = bool(par_fields)
        self.field_workers = int(field_workers) if field_workers is not None else None
        self.smooth_within_window = bool(smooth_within_window)

        if self.config.forecast_parameters is None:
            self.config.read_forecast_parameters()

        self.param_keys = list(self.config.forecast_parameters.index)
        self.base_params = {k: float(v) for k, v in self.config.forecast_parameters.items()}

        self.param_cols = None
        self.par_df = None
        self._load_param_ensemble()

        self.theta = {fid: np.zeros((self.ne, len(self.enkf_params))) for fid in self.fields}
        self._init_theta_from_posterior()

        self.R = {fid: 0.01 for fid in self.fields}
        self.Q = {p: 1e-4 for p in self.enkf_params}

        dr = pd.date_range(self.config.start_dt, self.config.end_dt, freq='D')
        self.dates = list(dr)
        self.final_kc = np.zeros((len(self.dates), len(self.fields))) * np.nan
        self.baseline_kc = None
        self.param_trace = {fid: pd.DataFrame(index=self.dates, columns=self.enkf_params, dtype=float) for fid in self.fields}
        self._single_cfg = {}
        self._single_plots = {}
        self._prepare_singleton_views()
        self.state_snapshots = {fid: None for fid in self.fields}
        self.Q_per_field = {fid: {p: 1e-4 for p in self.enkf_params} for fid in self.fields}

    def _load_param_ensemble(self):
        csv_path = self.config.forecast_param_csv if self.config.forecast_param_csv else self.config.forecast_parameters_csv
        if csv_path is None:
            return
        df = pd.read_csv(csv_path, index_col=0)
        self.par_df = df

    def _find_param_col(self, param, fid):
        cols = [c for c in self.par_df.columns if (param in c and fid.lower() in c)]
        if len(cols) == 0:
            return None
        # if multiple, take the first match
        col = cols[0]
        return col

    def _init_theta_from_posterior(self):
        for fid in self.fields:
            for j, p in enumerate(self.enkf_params):
                key = f'{p}_{fid}'
                if self.par_df is not None:
                    col = self._find_param_col(p, fid)
                    if col is not None:
                        vals = self.par_df[col].values
                        if len(vals) >= self.ne:
                            self.theta[fid][:, j] = vals[:self.ne]
                        else:
                            reps = int(np.ceil(self.ne / len(vals)))
                            tiled = np.tile(vals, reps)[:self.ne]
                            self.theta[fid][:, j] = tiled
                        continue
                v = self.base_params[key] if key in self.base_params else 0.0  # likely error if key missing
                self.theta[fid][:, j] = v

    def estimate_Q_R(self, q_scale=0.05, r_scale=1.0):
        if self.par_df is not None:
            for p in self.enkf_params:
                col_vars = []
                for fid in self.active_fields:
                    col = self._find_param_col(p, fid)
                    if col is not None:
                        v = float(np.nanvar(self.par_df[col].values, ddof=1))
                        self.Q_per_field[fid][p] = v * q_scale
                        col_vars.append(v)
                if len(col_vars) > 0:
                    self.Q[p] = float(np.nanmean(col_vars) * q_scale)

        if self.baseline_kc is None:
            self.run_baseline()
        r = {}
        for fid in self.active_fields:
            if fid in self.obs_df.columns:
                obs_series = self.obs_df[fid].reindex(self.dates)
                sim_series = pd.Series(self.baseline_kc[:, self.field_idx[fid]], index=self.dates)
                mask = (~obs_series.isna()) & (~sim_series.isna())
                if mask.any():
                    resid = (obs_series[mask] - sim_series[mask]).values
                    r[fid] = float(np.nanvar(resid, ddof=1) * r_scale)
        for fid in self.active_fields:
            if fid in r:
                self.R[fid] = r[fid]

    def run_baseline(self):
        start_dt = self.config.start_dt
        end_dt = self.config.end_dt
        kc, swe = field_day_loop(self.config, self.plots, debug_flag=False, params=None)
        self.baseline_kc = kc
        self.final_kc = self.baseline_kc.copy()
        self.config.start_dt = start_dt
        self.config.end_dt = end_dt

    def _prepare_singleton_views(self):
        for fid in self.active_fields:
            cfg = copy.deepcopy(self.config)
            if cfg.forecast_parameters is not None:
                idx = [i for i in cfg.forecast_parameters.index if i.endswith(f'_{fid.lower()}')]
                if len(idx) > 0:
                    cfg.forecast_parameters = cfg.forecast_parameters.loc[idx]
            self._single_cfg[fid] = cfg
            self._single_plots[fid] = self._make_single_plots(fid)

    def _make_single_plots(self, fid):
        sp = SamplePlots()
        base = self.plots.input
        idx = self.field_idx[fid]
        order = [fid]
        ts_sub = {}
        for dt, vals in base['time_series'].items():
            d = {}
            for k, v in vals.items():
                if k == 'doy':
                    d[k] = v
                else:
                    d[k] = [v[idx]]
            ts_sub[dt] = d
        sp.input = {'order': order, 'time_series': ts_sub}
        return sp

    def _build_params_for_field(self, fid, theta_vec):
        params = dict(self.base_params)
        for j, p in enumerate(self.enkf_params):
            key = f'{p}_{fid}'
            if key in params:
                params[key] = float(theta_vec[j])
        return params

    def _propagate(self, end_dt, params):
        cfg = copy.copy(self.config)
        cfg.end_dt = end_dt
        kc, swe = field_day_loop(cfg, self.plots, debug_flag=False, params=params)
        return kc

    def _run_range(self, fid, start_ix, end_ix, params, state0, capture_state=False):
        start_dt = self.dates[start_ix]
        end_dt = self.dates[end_ix]
        cfg = copy.copy(self._single_cfg[fid])
        cfg.start_dt = start_dt
        cfg.end_dt = end_dt
        if capture_state:
            kc, swe, states = field_day_loop(cfg, self._single_plots[fid], debug_flag=False, params=params,
                                             state_in=state0, capture_state=True, single_fid_idx=0)
            return kc[:, 0], states
        else:
            kc, swe = field_day_loop(cfg, self._single_plots[fid], debug_flag=False, params=params,
                                     state_in=state0, capture_state=False, single_fid_idx=0)
            return kc[:, 0], None

    def _ensure_state(self, fid, target_ix, params):
        state0 = self.state_snapshots[fid].get(target_ix)
        if state0 is not None:
            return state0
        # find nearest available index <= target_ix
        base = self.state_snapshots[fid].base
        if base is None:
            return None
        known_ix = max(base, target_ix - self.lag_days)
        st = self.state_snapshots[fid].get(known_ix)
        if st is None:
            return None
        kc_seg, states_seg = self._run_range(fid, known_ix, target_ix, params, st, capture_state=True)
        for k, st_ in enumerate(states_seg):
            self.state_snapshots[fid].set(known_ix + k, st_)
        return self.state_snapshots[fid].get(target_ix)

    def _apply_bounds(self, fid, theta_vec):
        arr = theta_vec.copy()
        for j, p in enumerate(self.enkf_params):
            if p in self.bounds:
                lo, hi = self.bounds[p]
                arr[j] = np.minimum(np.maximum(arr[j], lo), hi)
        return arr

    def run(self):
        if self.baseline_kc is None:
            self.run_baseline()

        self.estimate_Q_R()

        # initialize state snapshots for restartable single-field runs (only day 0)
        init_iter = tqdm(self.active_fields, desc='init', leave=False)
        for fid in init_iter:
            cfg = copy.copy(self._single_cfg[fid])
            cfg.start_dt = self.dates[0]
            cfg.end_dt = self.dates[0]
            base_params = self._build_params_for_field(fid, np.array([self.base_params.get(f'{p}_{fid}', 0.0) for p in self.enkf_params]))
            kc, swe, states = field_day_loop(cfg, self._single_plots[fid], debug_flag=False, params=base_params,
                                             state_in=None, capture_state=True, single_fid_idx=0)
            self.state_snapshots[fid] = _RingStates(self.lag_days + 1)
            self.state_snapshots[fid].set(0, states[0])

        date_iter = tqdm(self.dates, desc='dates')
        for t_idx, dt in enumerate(date_iter):
            if dt not in self.obs_df.index:
                continue

            if self.par_fields and self.field_workers and self.field_workers > 1:
                with ProcessPoolExecutor(max_workers=self.field_workers) as fpool:
                    futures = {}
                    bar_fields = tqdm(total=len(self.active_fields), leave=False, desc='fields')
                    for fid in self.active_fields:
                        if fid not in self.obs_df.columns:
                            bar_fields.update(1)
                            continue
                        y_obs = self.obs_df.at[dt, fid]
                        if pd.isna(y_obs):
                            bar_fields.update(1)
                            continue
                        start_ix = max(0, t_idx - self.lag_days + 1)
                        theta_mat = self.theta[fid].copy()
                        theta_bar = np.mean(theta_mat, axis=0)
                        params_bar = self._build_params_for_field(fid, theta_bar)
                        state0 = self._ensure_state(fid, start_ix, params_bar)
                        if state0 is None:
                            bar_fields.update(1)
                            continue
                        fut = fpool.submit(_assim_fid_step,
                                           self._single_cfg[fid],
                                           self._single_plots[fid],
                                           self.dates[start_ix], dt,
                                           theta_mat,
                                           self.enkf_params,
                                           {k: v for k, v in self.base_params.items() if k.endswith(f'_{fid}')},
                                           self.bounds,
                                           self.Q_per_field.get(fid, {}),
                                           self.Q,
                                           self.R[fid],
                                           y_obs,
                                           fid,
                                           self.stochastic_obs,
                                           state0)
                        futures[fut] = fid
                    for fut in as_completed(futures):
                        fid = futures[fut]
                        res = fut.result()
                        self.theta[fid] = res['theta']
                        self.param_trace[fid].loc[dt, :] = res['theta_mean']
                        kc_win = res['kc_win']
                        states_win = res['states_win']
                        self.final_kc[start_ix:t_idx + 1, self.field_idx[fid]] = kc_win
                        for k, st in enumerate(states_win):
                            self.state_snapshots[fid].set(start_ix + k, st)
                        bar_fields.update(1)
                    bar_fields.close()
            else:
                fields_iter = tqdm(self.active_fields, leave=False, desc='fields')
                for fid in fields_iter:
                    if fid not in self.obs_df.columns:
                        continue
                    y_obs = self.obs_df.at[dt, fid]
                    if pd.isna(y_obs):
                        continue

                    start_ix = max(0, t_idx - self.lag_days + 1)
                    theta_bar = np.mean(self.theta[fid], axis=0)
                    params_bar = self._build_params_for_field(fid, theta_bar)
                    state0 = self._ensure_state(fid, start_ix, params_bar)
                    if state0 is None:
                        continue

                    preds = np.zeros(self.ne)
                    if self.num_workers and self.num_workers > 1 and not self.par_fields:
                        if self.use_processes:
                            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                                futures = {}
                                for m in range(self.ne):
                                    params_m = self._build_params_for_field(fid, self.theta[fid][m, :])
                                    fut = ex.submit(_pred_at_dt_from_state,
                                                    self._single_cfg[fid],
                                                    self._single_plots[fid],
                                                    self.dates[start_ix], dt,
                                                    params_m, state0)
                                    futures[fut] = m
                                for fut in as_completed(futures):
                                    m = futures[fut]
                                    preds[m] = fut.result()
                        else:
                            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                                futures = {}
                                for m in range(self.ne):
                                    fut = ex.submit(_pred_at_dt_from_state,
                                                    self._single_cfg[fid],
                                                    self._single_plots[fid],
                                                    self.dates[start_ix], dt,
                                                    self._build_params_for_field(fid, self.theta[fid][m, :]),
                                                    state0)
                                    futures[fut] = m
                                for fut in as_completed(futures):
                                    m = futures[fut]
                                    preds[m] = fut.result()
                    else:
                        for m in range(self.ne):
                            params_m = self._build_params_for_field(fid, self.theta[fid][m, :])
                            preds[m] = _pred_at_dt_from_state(self._single_cfg[fid], self._single_plots[fid],
                                                              self.dates[start_ix], dt, params_m, state0)

                    y_bar = float(np.mean(preds))
                    theta_bar = np.mean(self.theta[fid], axis=0)

                    y_prime = preds - y_bar
                    theta_prime = self.theta[fid] - theta_bar

                    cov_ty = (theta_prime.T @ y_prime) / max(self.ne - 1, 1)
                    var_y = float((y_prime @ y_prime) / max(self.ne - 1, 1))

                    r = self.R[fid]
                    k_gain = cov_ty / (var_y + r)

                    q_vec = np.array([self.Q_per_field.get(fid, {}).get(p, self.Q[p]) for p in self.enkf_params])
                    noise = np.random.normal(0.0, np.sqrt(q_vec), size=self.theta[fid].shape)
                    if self.stochastic_obs:
                        obs_noise = np.random.normal(0.0, np.sqrt(max(r, 0.0)), size=self.ne)
                        y_eff = y_obs + obs_noise
                    else:
                        y_eff = np.ones(self.ne) * y_obs
                    innov = y_eff - preds
                    self.theta[fid] = self.theta[fid] + innov[:, None] * k_gain[None, :] + noise
                    for m in range(self.ne):
                        self.theta[fid][m, :] = self._apply_bounds(fid, self.theta[fid][m, :])

                    theta_mean = np.mean(self.theta[fid], axis=0)
                    self.param_trace[fid].loc[dt, :] = theta_mean

                    if self.smooth_within_window:
                        params_mean = self._build_params_for_field(fid, theta_mean)
                        kc_win, states_win = self._run_range(fid, start_ix, t_idx, params_mean, state0, capture_state=True)
                        self.final_kc[start_ix:t_idx + 1, self.field_idx[fid]] = kc_win
                        for k, st in enumerate(states_win):
                            self.state_snapshots[fid].set(start_ix + k, st)

    def _pred_at_dt(self, fid, dt, m):
        start_ix = 0
        state0 = self.state_snapshots[fid].get(start_ix)
        params_m = self._build_params_for_field(fid, self.theta[fid][m, :])
        val = _pred_at_dt_from_state(self._single_cfg[fid], self._single_plots[fid], self.dates[start_ix], dt, params_m, state0)
        return val

    def get_final_kc(self):
        df = pd.DataFrame(self.final_kc, index=self.dates, columns=self.fields)
        return df

    def get_param_trace(self):
        dct = {fid: self.param_trace[fid].copy() for fid in self.fields}
        return dct


if __name__ == '__main__':
    home = os.path.expanduser('~')
    project = '5_Flux_Ensemble'

    root = os.path.join(home, 'code', 'swim-rs')
    project_ws_ = os.path.join(root, 'tutorials', project)
    config_file = os.path.join(project_ws_, '5_Flux_Ensemble.toml')

    config_ = ProjectConfig()
    config_.read_config(config_file)

    target_dir = os.path.join(config_.project_ws, 'diy_ensemble')
    config_.forecast_parameters_csv = os.path.join(target_dir, f'{project}.3.par.csv')
    config_.spinup = os.path.join(target_dir, f'spinup.json')
    station_prepped_input = os.path.join(target_dir, f'prepped_input.json')
    config_.input_data = station_prepped_input

    plots_ = SamplePlots()
    plots_.initialize_plot_data(config_)

    config_.calibrate = False
    config_.forecast = True
    config_.read_forecast_parameters()

    fields = plots_.input['order']
    dates = pd.date_range(config_.start_dt, config_.end_dt, freq='D')
    obs_df = pd.DataFrame(index=dates, columns=None, dtype=float)

    model_map = {'ssebop': 'ssebop_etf_irr', 'sims': 'sims_etf_irr', 'ptjpl': 'ptjpl_etf_irr'}
    obs_col = model_map.get(config_.etf_target_model, 'ssebop_etf_irr')

    subset_fids = ['ALARC2_Smith6', 'S2', 'US-FPe']

    for fid in fields:
        if subset_fids is not None and fid not in subset_fids:
            continue
        fcsv = os.path.join(target_dir, f'{fid}.csv')
        if os.path.exists(fcsv):
            df_ = pd.read_csv(fcsv, index_col=0, parse_dates=True)
            s = df_[obs_col].reindex(dates)
            obs_df[fid] = s

    enkf = LaggedEnKF(config_, plots_, obs_df, enkf_params=['ndvi_k', 'ndvi_0', 'ks_alpha', 'kr_alpha'],
                      lag_days=16, ensemble_size=50, fields_subset=subset_fids, use_processes=1,
                      smooth_within_window=False, stochastic_obs=True)
    enkf.run()

    kc_df = enkf.get_final_kc()
    kc_out = os.path.join(target_dir, 'lenkf_kc.csv')
    kc_df.to_csv(kc_out)

    ptr = enkf.get_param_trace()
    params_df = pd.concat(ptr, axis=1)
    params_df.to_csv(os.path.join(target_dir, 'lenkf_params.csv'))
# ========================= EOF ====================================================================
