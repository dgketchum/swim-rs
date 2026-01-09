"""Standalone OpenET ensemble evaluation module.

Compare SWIM and OpenET ensemble models against flux tower observations.
Includes all 6 OpenET models (GEESEBAL, PTJPL, SSEBop, SIMS, EEMETRIC, DISALEXI)
plus the ensemble mean.
"""
from typing import Optional, Tuple, Dict, Any, Union, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


# Column renaming for OpenET data files
OPENET_COLUMN_RENAME = {
    'GEESEBAL_3x3': 'geesebal',
    'PTJPL_3x3': 'ptjpl',
    'SSEBOP_3x3': 'ssebop',
    'SIMS_3x3': 'sims',
    'EEMETRIC_3x3': 'eemetric',
    'DISALEXI_3x3': 'disalexi',
    'ensemble_mean_3x3': 'openet'
}


def evaluate_openet_site(
    model_output: Union[str, pd.DataFrame],
    flux_data_path: str,
    openet_daily_path: Optional[str],
    openet_monthly_path: Optional[str],
    irr: Dict[str, Any],
    gap_tolerance: int = 5,
    ensemble_members: Optional[List[str]] = None
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """Compare SWIM and OpenET ensemble models against flux observations.

    Args:
        model_output: Path to SWIM output CSV or DataFrame with model results.
            Must contain columns: openet_etf_irr, openet_etf_inv_irr, kc_act,
            et_act, eto, eto_corr
        flux_data_path: Path to flux tower data CSV with columns:
            date (index), ET, ET_corr, ET_fill, ET_gap
        openet_daily_path: Path to OpenET daily data CSV (optional).
        openet_monthly_path: Path to OpenET monthly data CSV (optional).
        irr: Irrigation metadata dict with yearly f_irr values.
            Format: {'2020': {'f_irr': 0.5}, '2021': {'f_irr': 0.1}, ...}
        gap_tolerance: Maximum allowed gap-filled days per month (default 5).
        ensemble_members: Optional list of specific OpenET models to include.
            Defaults to all models: ['geesebal', 'ptjpl', 'ssebop', 'sims',
            'eemetric', 'disalexi', 'openet']

    Returns:
        Tuple of (daily_metrics, overpass_metrics, monthly_metrics) dicts.
        Each dict contains rmse_{model}, r2_{model}, mean_{model}, n_samples.
        Returns (None, None, None) if data loading fails.
    """
    # Load flux data
    flux_data = pd.read_csv(flux_data_path, index_col='date', parse_dates=True)

    # Load model output
    if isinstance(model_output, pd.DataFrame):
        output = model_output.copy()
    else:
        try:
            output = pd.read_csv(model_output, index_col=0)
        except FileNotFoundError:
            print('Model output file not found')
            return None, None, None

    output.index = pd.to_datetime(output.index)

    # Select ETf based on irrigation status
    irr_threshold = 0.3
    irr_years = [int(k) for k, v in irr.items() if k != 'fallow_years'
                 and v['f_irr'] >= irr_threshold]
    irr_index = [i for i in output.index if i.year in irr_years]

    # Use irrigated or non-irrigated ETf based on year
    output['etf'] = output['openet_etf_inv_irr']
    output.loc[irr_index, 'etf'] = output.loc[irr_index, 'openet_etf_irr']
    output['capture'] = ~np.isnan(output['openet_etf_inv_irr'].values)
    output.loc[irr_index, 'capture'] = ~np.isnan(output.loc[irr_index, 'openet_etf_irr'].values)

    # Select ETo source based on irrigation
    if len(irr_years) > 0:
        eto_source = 'eto_corr'
    else:
        eto_source = 'eto'

    # Build comparison dataframe
    df = pd.DataFrame({
        'kc_act': output['kc_act'],
        'ET_corr': flux_data['ET_corr'],
        'etf': output['etf'],
        'capture': output['capture'],
        'eto': output[eto_source]
    })

    # Add flux observations
    df['flux'] = flux_data['ET']
    df['flux_fill'] = flux_data['ET_fill']
    df.loc[np.isnan(df['flux']), 'flux'] = df.loc[np.isnan(df['flux']), 'flux_fill']
    df['flux_gapfill'] = flux_data['ET_gap'].astype(int)

    # Add SWIM ET
    df['swim'] = output['et_act']

    # Define OpenET models to include
    openet_models = list(OPENET_COLUMN_RENAME.values())
    if ensemble_members:
        # Filter to specified members plus always include ensemble mean
        openet_models = [m for m in openet_models if m in ensemble_members or m == 'openet']

    # Load OpenET daily data
    df, openet_daily_loaded = _load_openet_daily(df, openet_daily_path, openet_models)

    # Load OpenET monthly data
    openet_monthly = _load_openet_monthly(openet_monthly_path, openet_models)

    # Models to evaluate
    all_models = ['swim'] + openet_models

    # Save a copy for monthly aggregation
    df_monthly = df.copy()

    # Compute daily metrics
    df_daily = df.dropna(subset=['flux'])
    results_daily = _compute_metrics(df_daily, 'flux', all_models)

    # Compute overpass (capture day) metrics
    df_overpass = df_daily[df_daily['capture'] == 1].copy()
    results_overpass = _compute_metrics(df_overpass, 'flux', all_models, min_samples=2)

    # Compute monthly metrics
    results_monthly = _compute_monthly_metrics(
        df_monthly, all_models, gap_tolerance, openet_monthly, openet_models
    )

    return results_daily, results_overpass, results_monthly


def _load_openet_daily(
    df: pd.DataFrame,
    openet_daily_path: Optional[str],
    openet_models: List[str]
) -> Tuple[pd.DataFrame, bool]:
    """Load and merge OpenET daily data."""
    if openet_daily_path:
        try:
            openet_daily = pd.read_csv(openet_daily_path, index_col='DATE', parse_dates=True)
            openet_daily = openet_daily.rename(columns=OPENET_COLUMN_RENAME)

            present_models = [m for m in openet_daily.columns if m in openet_models]
            openet_daily = openet_daily[present_models]

            for model in openet_models:
                if model in openet_daily.columns:
                    df[model] = openet_daily[model]
                else:
                    df[model] = np.nan
            return df, True

        except FileNotFoundError:
            print(f"Warning: OpenET daily data file not found: {openet_daily_path}")
        except KeyError:
            print('KeyError on OpenET model selection')
            return df, False

    # Fill all models with NaN if no data
    for model in openet_models:
        df[model] = np.nan
    return df, False


def _load_openet_monthly(
    openet_monthly_path: Optional[str],
    openet_models: List[str]
) -> Optional[pd.DataFrame]:
    """Load OpenET monthly data."""
    if openet_monthly_path:
        try:
            openet_monthly = pd.read_csv(openet_monthly_path, index_col='DATE', parse_dates=True)
            openet_monthly = openet_monthly.rename(columns=OPENET_COLUMN_RENAME)

            present_models = [m for m in openet_monthly.columns if m in openet_models]
            openet_monthly = openet_monthly[present_models]

            # Adjust dates to end-of-month
            idx = pd.to_datetime([d.replace(day=pd.Timestamp(d).days_in_month) for d in openet_monthly.index])
            openet_monthly.index = idx
            return openet_monthly

        except FileNotFoundError:
            print(f"Warning: OpenET monthly data file not found: {openet_monthly_path}")
        except KeyError:
            print('KeyError on OpenET model selection')

    return None


def _compute_metrics(
    df: pd.DataFrame,
    flux_col: str,
    model_cols: List[str],
    min_samples: int = 1
) -> Dict[str, float]:
    """Compute RMSE and R2 metrics for multiple models vs flux."""
    results = {}
    for model in model_cols:
        if model not in df.columns:
            continue
        df_temp = df.dropna(subset=[model])
        if df_temp.shape[0] < min_samples:
            continue
        try:
            results[f'rmse_{model}'] = float(np.sqrt(mean_squared_error(df_temp[flux_col], df_temp[model])))
            results[f'r2_{model}'] = r2_score(df_temp[flux_col], df_temp[model])
            if model != 'swim':
                results[f'{model}_mean'] = float(df_temp[model].mean())
            results['n_samples'] = df_temp.shape[0]
        except ValueError:
            continue
    return results


def _compute_monthly_metrics(
    df: pd.DataFrame,
    all_models: List[str],
    gap_tolerance: int,
    openet_monthly: Optional[pd.DataFrame],
    openet_models: List[str]
) -> Dict[str, float]:
    """Compute monthly aggregated metrics."""
    df_monthly = df.copy()
    df_monthly['days_in_month'] = 1
    ct = df_monthly[['days_in_month']].resample('ME').sum()
    df_monthly = df_monthly.rename(columns={'days_in_month': 'daily_obs'})

    start_date = ct.index.min() - pd.tseries.offsets.Day(31)
    end_date = ct.index.max() + pd.tseries.offsets.Day(31)
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='ME')
    ct = ct.reindex(full_date_range).fillna(0.0)
    ct = ct.resample('d').bfill()

    df_monthly.drop(columns=['etf'], inplace=True)
    df_monthly = pd.concat([df_monthly, ct], axis=1)
    df_monthly = df_monthly.dropna()

    month_check = df_monthly['flux_gapfill'].resample('ME').agg('sum')
    full_months = (month_check <= gap_tolerance).index
    full_months = [(i.year, i.month) for i in full_months]

    idx = [i for i in df_monthly.index if (i.year, i.month) in full_months]
    df_monthly = df_monthly.loc[idx].drop(columns=['daily_obs', 'days_in_month'])

    df_monthly = df_monthly.resample('ME').sum()

    # Merge OpenET monthly data
    if openet_monthly is not None:
        for model in openet_models:
            if model in openet_monthly.columns:
                df_monthly[model] = openet_monthly[model]

    # Clean monthly data
    df_monthly.drop(columns=['capture'], inplace=True)
    df_monthly = df_monthly.dropna(axis=0, how='any')
    df_monthly = df_monthly.replace(0.0, np.nan)
    df_monthly = df_monthly.replace([float('inf'), float('-inf')], np.nan)
    df_monthly = df_monthly.dropna(axis=1, how='any')

    results_monthly = {}
    if df_monthly.shape[0] >= 2:
        missing = []
        results_monthly['mean_flux'] = df_monthly['flux'].mean().item()
        for model in all_models:
            try:
                results_monthly[f'rmse_{model}'] = float(
                    np.sqrt(mean_squared_error(df_monthly['flux'], df_monthly[model])))
                results_monthly[f'r2_{model}'] = r2_score(df_monthly['flux'], df_monthly[model])
                results_monthly[f'mean_{model}'] = df_monthly[model].mean().item()
                results_monthly['n_samples'] = df_monthly.shape[0]
            except (ValueError, KeyError):
                missing.append(model)
        if missing:
            print(f'missing results: {missing}')

    return results_monthly


if __name__ == '__main__':
    pass
