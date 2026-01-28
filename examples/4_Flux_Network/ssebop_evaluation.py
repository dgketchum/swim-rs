"""Standalone SSEBop NHM evaluation module.

Compare SWIM and interpolated SSEBop ET against flux tower observations.
No OpenET data needed - only compares swim vs ssebop vs flux.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_ssebop_site(
    model_output: str | pd.DataFrame,
    flux_data_path: str,
    irr: dict[str, Any],
    gap_tolerance: int = 5,
    ssebop_eto_source: str = "eto_corr",
) -> tuple[dict | None, dict | None, dict | None]:
    """Compare SWIM and interpolated SSEBop ET against flux observations.

    Args:
        model_output: Path to SWIM output CSV or DataFrame with model results.
            Must contain columns: ssebop_etf_irr, ssebop_etf_inv_irr, kc_act,
            et_act, eto, eto_corr
        flux_data_path: Path to flux tower data CSV with columns:
            date (index), ET, ET_corr, ET_fill, ET_gap
        irr: Irrigation metadata dict with yearly f_irr values.
            Format: {'2020': {'f_irr': 0.5}, '2021': {'f_irr': 0.1}, ...}
        gap_tolerance: Maximum allowed gap-filled days per month (default 5).
        ssebop_eto_source: ETo column for SSEBop ET calculation (default 'eto_corr').

    Returns:
        Tuple of (daily_metrics, overpass_metrics, monthly_metrics) dicts.
        Each dict contains rmse_{model}, r2_{model}, mean_{model}, n_samples.
        Returns (None, None, None) if data loading fails.
    """
    # Load flux data
    flux_data = pd.read_csv(flux_data_path, index_col="date", parse_dates=True)

    # Load model output
    if isinstance(model_output, pd.DataFrame):
        output = model_output.copy()
    else:
        try:
            output = pd.read_csv(model_output, index_col=0)
        except FileNotFoundError:
            print("Model output file not found")
            return None, None, None

    output.index = pd.to_datetime(output.index)

    # Select ETf based on irrigation status
    irr_threshold = 0.3
    irr_years = [
        int(k) for k, v in irr.items() if k != "fallow_years" and v["f_irr"] >= irr_threshold
    ]
    irr_index = [i for i in output.index if i.year in irr_years]

    # Use irrigated or non-irrigated ETf based on year
    output["etf"] = output["ssebop_etf_inv_irr"]
    output.loc[irr_index, "etf"] = output.loc[irr_index, "ssebop_etf_irr"]
    output["capture"] = ~np.isnan(output["ssebop_etf_inv_irr"].values)
    output.loc[irr_index, "capture"] = ~np.isnan(output.loc[irr_index, "ssebop_etf_irr"].values)

    # Select ETo source based on irrigation
    if len(irr_years) > 0:
        eto_source = "eto_corr"
    else:
        eto_source = "eto"

    # Build comparison dataframe
    df = pd.DataFrame(
        {
            "kc_act": output["kc_act"],
            "ET_corr": flux_data["ET_corr"],
            "etf": output["etf"],
            "capture": output["capture"],
            "eto": output[eto_source],
        }
    )

    # Interpolate SSEBop ETf and calculate ET
    output["etf"] = output["etf"].interpolate()
    output["etf"] = output["etf"].bfill().ffill()
    df["ssebop"] = output["etf"] * output[ssebop_eto_source]

    # Add flux observations
    df["flux"] = flux_data["ET"]
    df["flux_fill"] = flux_data["ET_fill"]
    df.loc[np.isnan(df["flux"]), "flux"] = df.loc[np.isnan(df["flux"]), "flux_fill"]
    df["flux_gapfill"] = flux_data["ET_gap"].astype(int)

    # Add SWIM ET
    df["swim"] = output["et_act"]

    # Models to evaluate
    all_models = ["swim", "ssebop"]

    # Compute daily metrics
    df_daily = df.dropna(subset=["flux"])
    results_daily = _compute_metrics(df_daily, "flux", all_models)

    # Compute overpass (capture day) metrics
    df_overpass = df_daily[df_daily["capture"] == 1].copy()
    results_overpass = _compute_metrics(df_overpass, "flux", all_models, min_samples=2)

    # Compute monthly metrics
    results_monthly = _compute_monthly_metrics(df.copy(), all_models, gap_tolerance)

    return results_daily, results_overpass, results_monthly


def _compute_metrics(
    df: pd.DataFrame, flux_col: str, model_cols: list, min_samples: int = 1
) -> dict[str, float]:
    """Compute RMSE and R2 metrics for multiple models vs flux."""
    results = {}
    for model in model_cols:
        if model not in df.columns:
            continue
        df_temp = df.dropna(subset=[model])
        if df_temp.shape[0] < min_samples:
            continue
        try:
            results[f"rmse_{model}"] = float(
                np.sqrt(mean_squared_error(df_temp[flux_col], df_temp[model]))
            )
            results[f"r2_{model}"] = r2_score(df_temp[flux_col], df_temp[model])
            if model != "swim":
                results[f"{model}_mean"] = float(df_temp[model].mean())
            results["n_samples"] = df_temp.shape[0]
        except ValueError:
            continue
    return results


def _compute_monthly_metrics(
    df: pd.DataFrame, all_models: list, gap_tolerance: int
) -> dict[str, float]:
    """Compute monthly aggregated metrics."""
    df_monthly = df.copy()
    df_monthly["days_in_month"] = 1
    ct = df_monthly[["days_in_month"]].resample("ME").sum()
    df_monthly = df_monthly.rename(columns={"days_in_month": "daily_obs"})

    start_date = ct.index.min() - pd.tseries.offsets.Day(31)
    end_date = ct.index.max() + pd.tseries.offsets.Day(31)
    full_date_range = pd.date_range(start=start_date, end=end_date, freq="ME")
    ct = ct.reindex(full_date_range).fillna(0.0)
    ct = ct.resample("d").bfill()

    df_monthly.drop(columns=["etf"], inplace=True)
    df_monthly = pd.concat([df_monthly, ct], axis=1)
    df_monthly = df_monthly.dropna()

    month_check = df_monthly["flux_gapfill"].resample("ME").agg("sum")
    full_months = (month_check <= gap_tolerance).index
    full_months = [(i.year, i.month) for i in full_months]

    idx = [i for i in df_monthly.index if (i.year, i.month) in full_months]
    df_monthly = df_monthly.loc[idx].drop(columns=["daily_obs", "days_in_month"])

    df_monthly = df_monthly.resample("ME").sum()

    # Clean monthly data
    df_monthly.drop(columns=["capture"], inplace=True)
    df_monthly = df_monthly.dropna(axis=0, how="any")
    df_monthly = df_monthly.replace(0.0, np.nan)
    df_monthly = df_monthly.replace([float("inf"), float("-inf")], np.nan)
    df_monthly = df_monthly.dropna(axis=1, how="any")

    results_monthly = {}
    if df_monthly.shape[0] >= 2:
        missing = []
        results_monthly["mean_flux"] = df_monthly["flux"].mean().item()
        for model in all_models:
            try:
                results_monthly[f"rmse_{model}"] = float(
                    np.sqrt(mean_squared_error(df_monthly["flux"], df_monthly[model]))
                )
                results_monthly[f"r2_{model}"] = r2_score(df_monthly["flux"], df_monthly[model])
                results_monthly[f"mean_{model}"] = df_monthly[model].mean().item()
                results_monthly["n_samples"] = df_monthly.shape[0]
            except (ValueError, KeyError):
                missing.append(model)
        if missing:
            print(f"missing results: {missing}")

    return results_monthly


if __name__ == "__main__":
    pass
