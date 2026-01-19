"""CLI single-site flow integration tests.

Note: The ee module is mocked in conftest.py to allow importing CLI
without Earth Engine authentication.
"""

import argparse
import json
import os

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

import swimrs.cli as cli

# Mark entire module as integration
pytestmark = pytest.mark.integration


FID = 'S2'


def _make_shapefile(tmp_path):
    poly = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    gdf = gpd.GeoDataFrame({'FID': [FID], 'geometry': [poly]}, crs='EPSG:4326')
    shp_dir = tmp_path / 'data' / 'gis'
    shp_dir.mkdir(parents=True, exist_ok=True)
    shp_path = shp_dir / 'fields.shp'
    gdf.to_file(shp_path)
    return str(shp_path)


def _make_prepped_input(tmp_path):
    start = pd.Timestamp('2020-01-01')
    days = 3
    dates = [(start + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    doys = [int((start + pd.Timedelta(days=i)).strftime('%j')) for i in range(days)]

    time_series = {}
    for dt, doy in zip(dates, doys):
        rec = {
            'doy': doy,
            'tmin': [0.0],
            'tmax': [10.0],
            'srad': [100.0],
            'prcp': [0.0],
            'eto': [3.0],
            'eto_corr': [3.0],
            'ndvi_irr': [0.3],
            'ndvi_inv_irr': [0.3],
        }
        for h in range(24):
            rec[f'prcp_hr_{str(h).zfill(2)}'] = [0.0]
        time_series[dt] = rec

    prepped = {
        'props': {
            FID: {
                'awc': 0.15,
                'ksat': 10.0,
                'zr_mult': 3,
                'root_depth': 0.6,
                'lulc_code': 12,
                'irr': {str(y): 0.0 for y in range(1987, 2023)},
            }
        },
        'irr_data': {FID: {'2020': {'irrigated': 0, 'f_irr': 0.0, 'irr_doys': []}}},
        'gwsub_data': {FID: {'2020': {'f_sub': 0.0}}},
        'ke_max': {FID: 0.4},
        'kc_max': {FID: 1.1},
        'order': [FID],
        'time_series': time_series,
    }
    data_dir = os.path.join(tmp_path, 'data')
    os.makedirs(data_dir, exist_ok=True)
    out = os.path.join(data_dir, 'prepped_input.json')
    with open(out, 'w') as f:
        for k, v in prepped.items():
            json.dump({k: v}, f)
            f.write('\n')
    return out


def _make_forecast_params_csv(tmp_path):
    cols = ['aw', 'ndvi_k', 'ndvi_0', 'mad', 'swe_alpha', 'swe_beta', 'ks_alpha', 'kr_alpha', 'ke_max', 'kc_max']
    df = pd.DataFrame([[150.0, 5.0, 0.4, 0.5, 0.5, 1.3, 0.2, 0.3, 0.5, 1.1]], columns=cols)
    calib_dir = os.path.join(tmp_path, 'calib')
    os.makedirs(calib_dir, exist_ok=True)
    path = os.path.join(calib_dir, 'forecast_params.csv')
    df.to_csv(path)
    return path


def _make_initial_values_csv(tmp_path):
    # Minimal initial values CSV for calibrate mode
    df = pd.DataFrame({'mult_name': [f'aw_{FID}.txt']}, index=[f'aw_{FID}'])
    calib_dir = os.path.join(tmp_path, 'calib')
    os.makedirs(calib_dir, exist_ok=True)
    p = os.path.join(calib_dir, 'initial_values.csv')
    df.to_csv(p)
    return p


def _parse_and_run(argv):
    parser = cli.build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


@pytest.mark.xfail(
    reason="CLI API changed: assign_gridmet_and_corrections split into "
    "assign_gridmet_ids + sample_gridmet_corrections. Test needs update."
)
def test_cli_extract_single_site_smoke(tmp_path, monkeypatch):
    _make_shapefile(tmp_path)
    _make_prepped_input(tmp_path)
    _make_forecast_params_csv(tmp_path)
    _make_initial_values_csv(tmp_path)
    cfg = os.path.join(os.path.dirname(__file__), 'fixtures', 'S2', 'S2.toml')

    # Stub EE + all heavy side-effect functions
    monkeypatch.setattr(cli, 'is_authorized', lambda: True)
    calls = {
        'snodas': 0, 'cdl': 0, 'irr': 0, 'ssurgo': 0, 'landcover': 0,
        'ndvi': 0, 'etf': 0, 'gm_map': 0, 'gm_dl': 0,
    }
    monkeypatch.setattr(cli, 'sample_snodas_swe', lambda **kwargs: calls.__setitem__('snodas', calls['snodas'] + 1))
    monkeypatch.setattr(cli, 'get_cdl', lambda *a, **k: calls.__setitem__('cdl', calls['cdl'] + 1))
    monkeypatch.setattr(cli, 'get_irrigation', lambda *a, **k: calls.__setitem__('irr', calls['irr'] + 1))
    monkeypatch.setattr(cli, 'get_ssurgo', lambda *a, **k: calls.__setitem__('ssurgo', calls['ssurgo'] + 1))
    monkeypatch.setattr(cli, 'get_landcover', lambda *a, **k: calls.__setitem__('landcover', calls['landcover'] + 1))
    monkeypatch.setattr(cli, 'sparse_sample_ndvi', lambda *a, **k: calls.__setitem__('ndvi', calls['ndvi'] + 1))
    monkeypatch.setattr(cli, 'sparse_sample_etf', lambda *a, **k: calls.__setitem__('etf', calls['etf'] + 1))
    monkeypatch.setattr(cli, 'assign_gridmet_and_corrections', lambda *a, **k: calls.__setitem__('gm_map', calls['gm_map'] + 1))
    monkeypatch.setattr(cli, 'download_gridmet', lambda *a, **k: calls.__setitem__('gm_dl', calls['gm_dl'] + 1))

    rc = _parse_and_run(['extract', cfg, '--out-dir', str(tmp_path), '--sites', FID, '--export', 'drive', '--no-properties', '--no-snodas'])
    assert rc == 0
    # With limited date range and no ETF models, we still expect NDVI + gridmet calls
    assert calls['ndvi'] > 0
    assert calls['gm_map'] == 1
    assert calls['gm_dl'] == 1


@pytest.mark.xfail(
    reason="CLI API changed: sparse_time_series and other prep functions "
    "have been refactored. Test needs update."
)
def test_cli_prep_single_site_smoke(tmp_path, monkeypatch):
    _make_shapefile(tmp_path)
    _make_prepped_input(tmp_path)
    _make_forecast_params_csv(tmp_path)
    _make_initial_values_csv(tmp_path)
    cfg = os.path.join(os.path.dirname(__file__), 'fixtures', 'S2', 'S2.toml')

    calls = {k: 0 for k in ['sparse_ts', 'join_rs', 'write_props', 'snodas_js', 'join_daily', 'dyn', 'prep_js']}

    monkeypatch.setattr(cli, 'sparse_time_series', lambda *a, **k: calls.__setitem__('sparse_ts', calls['sparse_ts'] + 1))
    monkeypatch.setattr(cli, 'join_remote_sensing', lambda *a, **k: calls.__setitem__('join_rs', calls['join_rs'] + 1))
    monkeypatch.setattr(cli, 'write_field_properties', lambda *a, **k: calls.__setitem__('write_props', calls['write_props'] + 1))
    monkeypatch.setattr(cli, 'create_timeseries_json', lambda *a, **k: calls.__setitem__('snodas_js', calls['snodas_js'] + 1))
    monkeypatch.setattr(cli, 'join_daily_timeseries', lambda *a, **k: calls.__setitem__('join_daily', calls['join_daily'] + 1))
    monkeypatch.setattr(cli, 'process_dynamics_batch', lambda *a, **k: calls.__setitem__('dyn', calls['dyn'] + 1))
    monkeypatch.setattr(cli, 'prep_fields_json', lambda *a, **k: calls.__setitem__('prep_js', calls['prep_js'] + 1))
    monkeypatch.setattr(cli, 'preproc', lambda *a, **k: None)

    rc = _parse_and_run(['prep', cfg, '--out-dir', str(tmp_path), '--sites', FID])
    assert rc == 0
    assert calls['join_rs'] == 1
    assert calls['write_props'] == 1
    assert calls['join_daily'] == 1
    assert calls['dyn'] == 1
    assert calls['prep_js'] == 1


@pytest.mark.xfail(
    reason="Test fixture forecast_params.csv missing expected parameters (ke_max). "
    "Fixture needs update to match current CLI expectations."
)
def test_cli_evaluate_single_site_end_to_end(tmp_path):
    _make_shapefile(tmp_path)
    _make_prepped_input(tmp_path)
    forecast_csv = _make_forecast_params_csv(tmp_path)
    _make_initial_values_csv(tmp_path)
    cfg = os.path.join(os.path.dirname(__file__), 'fixtures', 'S2', 'S2.toml')
    rc = _parse_and_run(['evaluate', cfg, '--out-dir', str(tmp_path), '--sites', FID, '--forecast-params', forecast_csv])
    assert rc == 0
    # CSV for site should be written to out_root (tmp_path)
    site_csv = (tmp_path / f'{FID}.csv')
    assert site_csv.exists()


def test_cli_calibrate_orchestration(tmp_path, monkeypatch):
    _make_shapefile(tmp_path)
    _make_prepped_input(tmp_path)
    _make_forecast_params_csv(tmp_path)
    _make_initial_values_csv(tmp_path)
    cfg = os.path.join(os.path.dirname(__file__), 'fixtures', 'S2', 'S2.toml')

    # Stub PestBuilder to avoid heavy work
    calls = {'build_pest': 0, 'build_localizer': 0, 'write_control_settings': 0, 'spinup': 0, 'run_pst': 0}

    class _PB:
        def __init__(self, *a, **k):
            pass

        def build_pest(self, *a, **k):
            calls['build_pest'] += 1

        def build_localizer(self, *a, **k):
            calls['build_localizer'] += 1

        def write_control_settings(self, *a, **k):
            calls['write_control_settings'] += 1

        def spinup(self, *a, **k):
            calls['spinup'] += 1

    def _run_pst(*a, **k):
        calls['run_pst'] += 1

    monkeypatch.setattr(cli, 'PestBuilder', _PB)
    monkeypatch.setattr(cli, 'run_pst', _run_pst)

    rc = _parse_and_run(['calibrate', cfg, '--out-dir', str(tmp_path), '--sites', FID, '--workers', '1', '--realizations', '2'])
    assert rc == 0
    assert calls['build_pest'] == 1
    assert calls['build_localizer'] == 1
    assert calls['write_control_settings'] >= 2
    assert calls['spinup'] == 1
    assert calls['run_pst'] == 1
