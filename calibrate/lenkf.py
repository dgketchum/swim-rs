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

from calibrate.enkf_utils import _field_worker, _RingStates, _make_ensemble_plots_from_single, _pred_at_dt_from_state


class LaggedEnKF:
    def __init__(self, config: ProjectConfig, plots: SamplePlots,
                 enkf_params=None, lag_days=7, ensemble_size=50, bounds=None,
                 fields_subset=None, num_workers=None, use_processes=False,
                 stochastic_obs=True, par_fields=True, field_workers=None,
                 smooth_within_window=True):
        self.config = config
        self.plots = plots
        self.fields = self.plots.input['order']
        self.active_fields = [f for f in (fields_subset if fields_subset is not None else self.fields) if
                              f in self.fields]
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
        self.param_trace = {fid: pd.DataFrame(index=self.dates, columns=self.enkf_params, dtype=float) for fid in
                            self.fields}
        self._single_cfg = {}
        self._single_plots = {}
        self._prepare_singleton_views()
        self.state_snapshots = {fid: None for fid in self.fields}
        self.Q_per_field = {fid: {p: 1e-4 for p in self.enkf_params} for fid in self.fields}
        self._spinup_states = None
        self.obs_kc = None

    def build_obs_from_plots(self, model=None, irr_threshold=None):
        mdl = model if model is not None else self.config.etf_target_model or 'ssebop'
        thr = float(irr_threshold) if irr_threshold is not None else (self.config.irr_threshold if self.config.irr_threshold is not None else 0.2)
        ts = self.plots.input['time_series']
        irr_info = self.plots.input.get('irr_data', {})
        cols = {fid: [] for fid in self.active_fields}
        idx = []

        for dt in self.dates:
            dts = dt.strftime('%Y-%m-%d')
            if dts not in ts:
                idx.append(dt)
                for fid in self.active_fields:
                    cols[fid].append(np.nan)
                continue
            vals = ts[dts]
            irr_key = f'{mdl}_etf_irr'
            inv_key = f'{mdl}_etf_inv_irr'
            base_key = f'{mdl}_etf'
            idx.append(dt)
            for fid in self.active_fields:
                j = self.field_idx[fid]  # likely error if fid missing
                yr = str(dt.year)
                f_irr = None
                irrigated = None
                if fid in irr_info and yr in irr_info[fid]:
                    rec = irr_info[fid][yr]
                    f_irr = rec.get('f_irr')
                    irrigated = rec.get('irrigated')
                v = np.nan
                has_irr = irr_key in vals
                has_inv = inv_key in vals
                has_base = base_key in vals
                if has_irr and has_inv and f_irr is not None and 0.0 < f_irr < 1.0:
                    vi = vals[irr_key][j]
                    vu = vals[inv_key][j]
                    v = f_irr * vi + (1.0 - f_irr) * vu
                elif has_irr and (irrigated == 1 or (f_irr is not None and f_irr >= thr)):
                    v = vals[irr_key][j]
                elif has_inv and (irrigated == 0 or (f_irr is not None and f_irr < thr)):
                    v = vals[inv_key][j]
                elif has_base:
                    v = vals[base_key][j]
                elif has_irr:
                    v = vals[irr_key][j]
                elif has_inv:
                    v = vals[inv_key][j]
                cols[fid].append(v)
        df = pd.DataFrame(index=idx, data=cols)
        return df

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
            if self.obs_kc is not None and fid in self.obs_kc.columns:
                obs_series = self.obs_kc[fid].reindex(self.dates)
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
        used_csv = False
        base_dir = None
        if self.config.spinup and os.path.exists(self.config.spinup):
            base_dir = os.path.dirname(self.config.spinup)
            req = [os.path.join(base_dir, f'{fid}.csv') for fid in self.active_fields]
            if all(os.path.exists(p) for p in req):
                kc_arr = np.zeros((len(self.dates), len(self.fields))) * np.nan
                for fid in self.active_fields:
                    fcsv = os.path.join(base_dir, f'{fid}.csv')
                    df_ = pd.read_csv(fcsv, index_col=0, parse_dates=True)
                    if 'kc_act' in df_.columns:
                        s_kc = df_['kc_act'].reindex(self.dates)
                    else:
                        s_kc = df_['kc']  # likely error if kc column missing
                        s_kc = s_kc.reindex(self.dates)
                    kc_arr[:, self.field_idx[fid]] = s_kc.values
                self.baseline_kc = kc_arr
                self.final_kc = self.baseline_kc.copy()
                used_csv = True

        if not used_csv:
            kc, swe = field_day_loop(self.config, self.plots, debug_flag=False, params=None)
            self.baseline_kc = kc
            self.final_kc = self.baseline_kc.copy()
        self.config.start_dt = start_dt
        self.config.end_dt = end_dt
        if self.config.spinup and os.path.exists(self.config.spinup):
            with open(self.config.spinup, 'r') as fp:
                sdct = json.load(fp)
            self._spinup_states = sdct
            for fid in self.active_fields:
                var_dct = sdct[fid]
                st0 = {k: var_dct[k] for k in TRACKER_PARAMS if k in var_dct}
                self.state_snapshots[fid] = _RingStates(self.lag_days + 1)
                self.state_snapshots[fid].set(0, st0)

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
        props_sub = {fid: base['props'][fid]} if 'props' in base and fid in base['props'] else {}
        irr_sub = {fid: base['irr_data'][fid]} if 'irr_data' in base and fid in base['irr_data'] else {}
        gwsub_sub = {fid: base['gwsub_data'][fid]} if 'gwsub_data' in base and fid in base['gwsub_data'] else {}
        kc_max_sub = {fid: base['kc_max'][fid]} if 'kc_max' in base and fid in base['kc_max'] else {}
        ke_max_sub = {fid: base['ke_max'][fid]} if 'ke_max' in base and fid in base['ke_max'] else {}
        sp.input = {'order': order,
                    'time_series': ts_sub,
                    'props': props_sub,
                    'irr_data': irr_sub,
                    'gwsub_data': gwsub_sub,
                    'kc_max': kc_max_sub,
                    'ke_max': ke_max_sub}
        return sp

    def _make_ensemble_plots_for_fid(self, fid, ne):
        return _make_ensemble_plots_from_single(self._single_plots[fid], fid, ne)

    def _preds_ensemble_at_dt_from_state(self, fid, start_dt, end_dt, theta_mat, state0):
        ne = int(theta_mat.shape[0])
        cfg2 = copy.copy(self._single_cfg[fid])
        cfg2.start_dt = start_dt
        cfg2.end_dt = end_dt
        cfg2.calibrate = False
        cfg2.forecast = True

        plots_e = self._make_ensemble_plots_for_fid(fid, ne)

        fid_l = fid.lower()
        base_by_group = {}
        for p in TUNABLE_PARAMS:
            key = f'{p}_{fid_l}'
            base_by_group[p] = float(self.base_params.get(key, 0.0))

        efids = [f"{fid_l}__e{i}" for i in range(ne)]
        fp = {}
        for i, ef in enumerate(efids):
            for p in TUNABLE_PARAMS:
                if p in self.enkf_params:
                    j = self.enkf_params.index(p)
                    fp[f'{p}_{ef}'] = float(theta_mat[i, j])
                else:
                    fp[f'{p}_{ef}'] = base_by_group[p]

        cfg2.forecast_parameters = pd.Series(fp)

        kc, swe = field_day_loop(cfg2, plots_e, debug_flag=False, params=None,
                                  state_in=state0, capture_state=False, single_fid_idx=None)
        return kc[-1, :]

    def _build_params_for_field(self, fid, theta_vec):
        params = dict(self.base_params)
        for j, p in enumerate(self.enkf_params):
            key = f'{p}_{fid.lower()}'
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
        """Return state at target_ix, propagating forward from last known if needed.

        This is lag-agnostic and does not assume a state exists at
        (t_idx - lag_days + 1). It finds the most recent available snapshot
        at or before target_ix and runs forward to fill the gap, storing
        states along the way so subsequent calls can reuse them.
        """
        ring = self.state_snapshots[fid]
        st_at_target = ring.get(target_ix)
        if st_at_target is not None:
            return st_at_target

        base = ring.base
        count = ring.count if hasattr(ring, 'count') else 0
        if base is None or count == 0:
            # Fall back to spinup if available
            if self._spinup_states is not None and fid in self._spinup_states:
                st0 = {k: self._spinup_states[fid][k] for k in TRACKER_PARAMS if k in self._spinup_states[fid]}
                seed_ix = 0
                seed_state = st0
            else:
                return None
        else:
            last_known_ix = base + count - 1
            # scan backwards to find nearest stored snapshot <= target_ix
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
                # fallback to spinup if present
                if self._spinup_states is not None and fid in self._spinup_states:
                    seed_ix = 0
                    seed_state = {k: self._spinup_states[fid][k] for k in TRACKER_PARAMS if k in self._spinup_states[fid]}
                else:
                    return None

        if seed_ix == target_ix:
            return seed_state

        kc_seg, states_seg = self._run_range(fid, seed_ix, target_ix, params, seed_state, capture_state=True)
        for k, st_ in enumerate(states_seg):
            ring.set(seed_ix + k, st_)
        return ring.get(target_ix)

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

        if self.obs_kc is None or self.obs_kc.shape[1] == 0:
            self.obs_kc = self.build_obs_from_plots()

        self.estimate_Q_R()

        init_iter = tqdm(self.active_fields, desc='init', leave=False)
        for fid in init_iter:
            if self.state_snapshots[fid] is None:
                self.state_snapshots[fid] = _RingStates(self.lag_days + 1)
                st0 = {k: self._spinup_states[fid][k] for k in TRACKER_PARAMS if k in self._spinup_states[fid]}
                self.state_snapshots[fid].set(0, st0)

        date_iter = tqdm(self.dates, desc='dates')
        if self.par_fields and self.field_workers and self.field_workers > 1:
            # Per-field multiprocessing: each worker handles all dates for one field
            with ProcessPoolExecutor(max_workers=self.field_workers) as fpool:
                futures = {}
                for fid in self.active_fields:
                    if self.obs_kc is not None and fid in self.obs_kc.columns:
                        obs_series = self.obs_kc[fid].reindex(self.dates)
                        obs_vals = obs_series.values.astype(float)
                    else:
                        obs_vals = np.full((len(self.dates),), np.nan, dtype=float)

                    theta_init = self.theta[fid].copy()
                    baseline_col = self.baseline_kc[:, self.field_idx[fid]].copy()
                    base_params_by_fid = {k: v for k, v in self.base_params.items() if k.endswith(f'_{fid.lower()}')}
                    spinup_state = None
                    if self._spinup_states is not None and fid in self._spinup_states:
                        spinup_state = self._spinup_states[fid]

                    fut = fpool.submit(_field_worker,
                                       self._single_cfg[fid],
                                       self._single_plots[fid],
                                       self.dates,
                                       obs_vals,
                                       self.enkf_params,
                                       theta_init,
                                       self.bounds,
                                       self.Q_per_field.get(fid, {}),
                                       self.Q,
                                       self.R[fid],
                                       fid,
                                       self.stochastic_obs,
                                       self.lag_days,
                                       self.smooth_within_window,
                                       base_params_by_fid,
                                       spinup_state,
                                       baseline_col)
                    futures[fut] = fid
                for fut in as_completed(futures):
                    fid = futures[fut]
                    res = fut.result()
                    self.theta[fid] = res['theta']
                    # update param trace for all dates
                    arr = res['param_trace']
                    self.param_trace[fid].iloc[:, :] = arr
                    # update full kc time series for this field
                    self.final_kc[:, self.field_idx[fid]] = res['final_kc_col']

        else:
            for t_idx, dt in enumerate(date_iter):

                if dt not in self.obs_kc.index:
                    continue

                for fid in self.active_fields:
                    if fid not in self.obs_kc.columns:
                        continue
                    y_obs = self.obs_kc.at[dt, fid]
                    if pd.isna(y_obs):
                        continue

                    start_ix = max(0, t_idx - self.lag_days + 1)
                    theta_bar = np.mean(self.theta[fid], axis=0)
                    params_bar = self._build_params_for_field(fid, theta_bar)
                    state0 = self._ensure_state(fid, start_ix, params_bar)
                    if state0 is None:
                        continue

                    # Vectorized ensemble propagation across ne synthetic fields
                    preds = self._preds_ensemble_at_dt_from_state(fid, self.dates[start_ix], dt,
                                                                  self.theta[fid], state0)

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
                        kc_win, states_win = self._run_range(fid, start_ix, t_idx, params_mean, state0,
                                                             capture_state=True)
                        self.final_kc[start_ix:t_idx + 1, self.field_idx[fid]] = kc_win
                        for k, st in enumerate(states_win):
                            self.state_snapshots[fid].set(start_ix + k, st)

    def _pred_at_dt(self, fid, dt, m):
        start_ix = 0
        state0 = self.state_snapshots[fid].get(start_ix)
        params_m = self._build_params_for_field(fid, self.theta[fid][m, :])
        val = _pred_at_dt_from_state(self._single_cfg[fid], self._single_plots[fid],
                                     self.dates[start_ix], dt, params_m, state0)
        return val

    def get_final_kc(self):
        df = pd.DataFrame(self.final_kc, index=self.dates, columns=self.fields)
        return df

    def get_param_trace(self):
        dct = {fid: self.param_trace[fid].copy() for fid in self.fields}
        return dct

    def write_full_model_output(self, target_dir, suffix='_lenkf', file_fmt='csv'):
        """Write full daily state/output for each field using final posterior params.

        For each active field, builds a single-field config with forecast parameters
        set to the final mean of the parameter trace (last non-NaN, forward-filled),
        runs a debug pass to collect the full daily outputs, and writes them to
        `target_dir/{fid}{suffix}.csv` by default.

        Returns a dict[fid -> DataFrame] with the daily outputs.
        """
        os.makedirs(target_dir, exist_ok=True)
        results = {}
        for fid in self.active_fields:
            # Determine final mean parameter vector for this field
            tr = self.param_trace[fid]
            if tr is not None and len(tr) > 0:
                s = tr.ffill().iloc[-1]
            else:
                s = pd.Series(index=self.enkf_params, dtype=float)

            # Fallback if all NaN: use current ensemble mean
            if s.isna().all():
                theta_mean = np.mean(self.theta[fid], axis=0)
                s = pd.Series(theta_mean, index=self.enkf_params)

            fid_l = fid.lower()
            # Build single-field forecast parameters
            fp_single = {}
            for p in TUNABLE_PARAMS:
                if p in self.enkf_params and p in s and pd.notna(s[p]):
                    fp_single[f'{p}_{fid_l}'] = float(s[p])
                else:
                    base_val = float(self.base_params.get(f'{p}_{fid_l}', 0.0))
                    fp_single[f'{p}_{fid_l}'] = base_val

            cfg = copy.copy(self._single_cfg[fid])
            cfg.start_dt = self.config.start_dt
            cfg.end_dt = self.config.end_dt
            cfg.calibrate = False
            cfg.forecast = True
            cfg.forecast_parameters = pd.Series(fp_single)

            # Seed state from spinup if available
            state0 = None
            if self._spinup_states is not None and fid in self._spinup_states:
                var_dct = self._spinup_states[fid]
                state0 = {k: var_dct[k] for k in TRACKER_PARAMS if k in var_dct}

            # Run in debug mode to get full daily outputs
            df_dict = field_day_loop(cfg, self._single_plots[fid], debug_flag=True, params=None,
                                     state_in=state0, capture_state=False, single_fid_idx=0)
            # df_dict is keyed by the field id used in plots order (which is fid)
            df = df_dict.get(fid) if isinstance(df_dict, dict) else None
            if df is None:
                # Fallback: pass through whatever structure came back
                df = pd.DataFrame()
            results[fid] = df

            # Write to file
            out_path = os.path.join(target_dir, f'{fid}{suffix}.csv') if file_fmt == 'csv' else os.path.join(target_dir, f'{fid}{suffix}.parquet')
            if file_fmt == 'csv':
                df.to_csv(out_path)
            elif file_fmt == 'parquet':
                df.to_parquet(out_path, index=True)
            else:
                raise ValueError('Unsupported file_fmt; use csv or parquet')

        return results


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
    subset_fids = ['ALARC2_Smith6', 'S2']

    enkf = LaggedEnKF(config_, plots_, enkf_params=['ndvi_k', 'ndvi_0', 'ks_alpha', 'kr_alpha'],
                      lag_days=16, ensemble_size=50, fields_subset=subset_fids, use_processes=12,
                      smooth_within_window=True, stochastic_obs=True, par_fields=True, field_workers=True)
    enkf.run()

    kc_df = enkf.get_final_kc()
    kc_out = os.path.join(target_dir, 'lenkf_kc.csv')
    kc_df.to_csv(kc_out)

    ptr = enkf.get_param_trace()
    params_df = pd.concat(ptr, axis=1)
    params_df.to_csv(os.path.join(target_dir, 'lenkf_params.csv'))
# ========================= EOF ====================================================================
