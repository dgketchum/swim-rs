"""
Quick visualization for Fort Peck model results.

Plots daily flux tower ET alongside model ET (and PT-JPL if available)
using the output CSV produced by `swim evaluate` or the notebooks.

Usage:
    python viz.py --results /tmp/swim_fp --site US-FPe
    python viz.py --results . --site US-FPe --save /tmp/plots  # save to files

The script searches for output in multiple locations:
    - <results>/combined_output_<SITE>_calibrated.csv (from notebooks)
    - <results>/<SITE>.csv (from swim evaluate with --out-dir)
    - <project_dir>/<SITE>.csv (from swim evaluate default)
    - <project_dir>/data/<SITE>_daily_data.csv (flux data)

Outputs (when --save is used):
    - <save_dir>/<SITE>_timeseries.png (daily ET comparison for 2006)
    - <save_dir>/<SITE>_scatter.png (2x2 scatter plot with metrics)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_flux(flux_dir: Path, site: str, project_dir: Path = None) -> pd.Series | None:
    """Load flux tower ET for a site if available."""
    # Check multiple possible locations
    candidates = [
        flux_dir / f"{site}_daily_data.csv",
        flux_dir / f"{site}.csv",
    ]
    if project_dir:
        candidates.extend([
            project_dir / "data" / f"{site}_daily_data.csv",
            project_dir / "data" / f"{site}_daily_data.zip",
        ])

    flux_file = next((p for p in candidates if p.exists() and p.suffix == '.csv'), None)
    if flux_file is None:
        # Try zip file
        zip_file = next((p for p in candidates if p.exists() and p.suffix == '.zip'), None)
        if zip_file:
            import zipfile
            with zipfile.ZipFile(zip_file) as zf:
                csv_name = f"{site}_daily_data.csv"
                if csv_name in zf.namelist():
                    import io
                    with zf.open(csv_name) as f:
                        df = pd.read_csv(io.BytesIO(f.read()), parse_dates=["date"])
                        df.set_index("date", inplace=True)
                        for col in ("ET", "ET_mm", "et"):
                            if col in df.columns:
                                return df[col]
        return None

    df = pd.read_csv(flux_file, parse_dates=["date"])
    df.set_index("date", inplace=True)
    for col in ("ET", "ET_mm", "et"):
        if col in df.columns:
            return df[col]
    return None


def calc_metrics(y_true, y_pred):
    """Calculate R², Pearson r, RMSE, and bias."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    r, _ = stats.pearsonr(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    bias = np.mean(y_pred - y_true)
    return r2, r, rmse, bias


def plot_timeseries(df: pd.DataFrame, flux_et: pd.Series, site: str, save_path: Path = None):
    """Plot 2006 time series comparison using Plotly dark theme."""
    # Filter to 2006
    df_2006 = df.loc['2006-01-01':'2006-12-31'].copy()

    if flux_et is not None:
        flux_2006 = flux_et.loc['2006-01-01':'2006-12-31']
    else:
        flux_2006 = None

    # Build PT-JPL ET from ETf * ETref if available
    ptjpl_et = None
    for etf_col in ['etf_inv_irr', 'etf_irr']:
        if etf_col in df_2006.columns and 'etref' in df_2006.columns:
            ptjpl_et = df_2006[etf_col] * df_2006['etref']
            break

    fig = go.Figure()

    # SWIM ET
    fig.add_trace(go.Scatter(
        x=df_2006.index,
        y=df_2006['et_act'],
        name='SWIM ET',
        line=dict(color='#3498db', width=1.5),
    ))

    # Flux tower ET
    if flux_2006 is not None and len(flux_2006.dropna()) > 0:
        fig.add_trace(go.Scatter(
            x=flux_2006.index,
            y=flux_2006,
            name='Flux Tower ET',
            line=dict(color='#2ecc71', width=1.5),
        ))

    # PT-JPL ET (interpolated)
    if ptjpl_et is not None:
        ptjpl_interp = ptjpl_et.interpolate(method='linear')
        fig.add_trace(go.Scatter(
            x=df_2006.index,
            y=ptjpl_interp,
            name='PT-JPL ET (interp)',
            line=dict(color='#e74c3c', width=1, dash='dot'),
            opacity=0.7,
        ))

    fig.update_layout(
        title=f'{site} Daily ET - 2006',
        xaxis_title='Date',
        yaxis_title='ET (mm/day)',
        template='plotly_dark',
        height=500,
        width=1200,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )

    if save_path:
        fig.write_image(str(save_path), scale=2)
        print(f"Saved: {save_path}")
    else:
        fig.show()


def plot_scatter_comparison(df: pd.DataFrame, flux_et: pd.Series, site: str, save_path: Path = None):
    """Create 1x2 scatter comparison plot matching notebook style."""
    # Build comparison DataFrames
    model_et = df['et_act']

    # PT-JPL ET from ETf * ETref
    ptjpl_et_sparse = None
    for etf_col in ['etf_inv_irr', 'etf_irr']:
        if etf_col in df.columns and 'etref' in df.columns:
            ptjpl_et_sparse = df[etf_col] * df['etref']
            break

    if ptjpl_et_sparse is None:
        print("Warning: Could not compute PT-JPL ET (missing etf or etref columns)")
        return

    ptjpl_et_interp = ptjpl_et_sparse.interpolate(method='linear')
    n_ptjpl_obs = ptjpl_et_sparse.notna().sum()

    # FULL TIME SERIES (PT-JPL interpolated)
    full_df = pd.DataFrame({
        'swim_et': model_et,
        'ptjpl_et': ptjpl_et_interp,
        'flux_et': flux_et
    }).dropna()

    # Calculate metrics
    r2_swim, r_swim, rmse_swim, _ = calc_metrics(
        full_df['flux_et'].values, full_df['swim_et'].values)
    r2_ptjpl, r_ptjpl, rmse_ptjpl, _ = calc_metrics(
        full_df['flux_et'].values, full_df['ptjpl_et'].values)

    # Axis limits
    max_et = max(
        full_df['flux_et'].max(),
        full_df['swim_et'].max(),
        full_df['ptjpl_et'].max()
    ) * 1.1

    # Create 1x2 subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"SWIM vs Flux (n={len(full_df)})<br>"
            f"R² = {r2_swim:.3f}, r = {r_swim:.3f}, RMSE = {rmse_swim:.2f} mm",
            f"PT-JPL vs Flux, interpolated (n={len(full_df)})<br>"
            f"R² = {r2_ptjpl:.3f}, r = {r_ptjpl:.3f}, RMSE = {rmse_ptjpl:.2f} mm",
        ],
        horizontal_spacing=0.12,
    )

    # Colors
    scatter_color = '#3498db'
    line_color = '#e74c3c'

    # Left: SWIM vs Flux
    fig.add_trace(go.Scatter(
        x=full_df['flux_et'],
        y=full_df['swim_et'],
        mode='markers',
        marker=dict(color=scatter_color, size=5, opacity=0.4),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, max_et],
        y=[0, max_et],
        mode='lines',
        line=dict(color=line_color, dash='dash'),
        name='1:1 line',
        showlegend=False,
    ), row=1, col=1)

    # Right: PT-JPL vs Flux
    fig.add_trace(go.Scatter(
        x=full_df['flux_et'],
        y=full_df['ptjpl_et'],
        mode='markers',
        marker=dict(color=scatter_color, size=5, opacity=0.4),
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[0, max_et],
        y=[0, max_et],
        mode='lines',
        line=dict(color=line_color, dash='dash'),
        showlegend=False,
    ), row=1, col=2)

    # Update axes
    for col in [1, 2]:
        fig.update_xaxes(
            title_text='Flux ET (mm/day)',
            range=[0, max_et],
            showgrid=False,
            row=1, col=col
        )
        fig.update_yaxes(
            title_text='SWIM ET (mm/day)' if col == 1 else 'PT-JPL ET (mm/day)',
            range=[0, max_et],
            showgrid=False,
            row=1, col=col
        )

    fig.update_layout(
        title=dict(text=f'{site}: ET Comparison - SWIM vs PT-JPL', x=0.5, xanchor='center'),
        template='plotly_dark',
        height=500,
        width=1000,
        showlegend=False,
    )

    # Print metrics summary
    print(f"\n{'='*70}")
    print(f"FULL TIME SERIES ({len(full_df)} days, PT-JPL interpolated from {n_ptjpl_obs} obs)")
    print(f"{'-'*70}")
    print(f"{'Metric':<12} {'SWIM ET':>12} {'PT-JPL ET':>12}")
    print(f"{'-'*38}")
    print(f"{'R²':<12} {r2_swim:>12.3f} {r2_ptjpl:>12.3f}")
    print(f"{'Pearson r':<12} {r_swim:>12.3f} {r_ptjpl:>12.3f}")
    print(f"{'RMSE (mm)':<12} {rmse_swim:>12.3f} {rmse_ptjpl:>12.3f}")
    print(f"{'='*70}\n")

    if save_path:
        fig.write_image(str(save_path), scale=2)
        print(f"Saved: {save_path}")
    else:
        fig.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize model ET vs flux/PT-JPL")
    parser.add_argument("--results", required=True, help="Directory containing model output CSV")
    parser.add_argument("--site", default="US-FPe", help="Site ID (default: US-FPe)")
    parser.add_argument("--flux-dir", default=None, help="Flux directory (default: data/daily_flux_files)")
    parser.add_argument("--save", default=None, help="Directory to save plots (default: show interactively)")
    args = parser.parse_args()

    results_dir = Path(args.results)
    site = args.site
    project_dir = Path(__file__).resolve().parent

    # Prefer calibrated scatter-friendly CSV if present, else fall back to model run CSV
    # Check multiple locations: --results dir, project root, data subdir
    candidates = [
        results_dir / f"combined_output_{site}_calibrated.csv",
        results_dir / f"{site}.csv",
        results_dir / "results" / f"{site}.csv",
        project_dir / f"{site}.csv",  # evaluate default output location
        project_dir / "data" / f"{site}.csv",
    ]
    calibrated_path = next((p for p in candidates if p.exists()), None)
    if calibrated_path is None:
        raise FileNotFoundError(
            f"Output CSV not found for site {site}. Looked in:\n  " +
            "\n  ".join(str(p) for p in candidates)
        )

    flux_dir = Path(args.flux_dir) if args.flux_dir else project_dir / "data" / "daily_flux_files"
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model output from: {calibrated_path}")

    # Load data
    df = pd.read_csv(calibrated_path, index_col=0, parse_dates=True)
    if "date" in df.columns:
        df.set_index("date", inplace=True)

    flux_et = load_flux(flux_dir, site, project_dir)

    print(f"Model ET: {len(df)} days")
    if flux_et is not None:
        print(f"Flux ET: {len(flux_et)} days")
    else:
        print("Flux data not found")

    # Plot time series (2006 only)
    ts_path = save_dir / f"{site}_timeseries.png" if save_dir else None
    plot_timeseries(df, flux_et, site, ts_path)

    # Plot scatter comparison
    if flux_et is not None:
        scatter_path = save_dir / f"{site}_scatter.png" if save_dir else None
        plot_scatter_comparison(df, flux_et, site, scatter_path)


if __name__ == "__main__":
    main()
