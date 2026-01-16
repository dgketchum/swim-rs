import json
import os
from datetime import datetime, timedelta

import numpy as np

from swimrs.workflow import run_field_workflow


def _minimal_prepped_input(tmpdir, fid='TEST1'):
    start = datetime(2020, 1, 1)
    days = 3
    dates = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    doys = [int((start + timedelta(days=i)).strftime('%j')) for i in range(days)]

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
            fid: {
                'awc': 0.15,
                'ksat': 10.0,
                'zr_mult': 3,
                'root_depth': 0.6,
                'lulc_code': 12,
            }
        },
        'irr_data': {fid: {'2020': {'irrigated': 0, 'f_irr': 0.0, 'irr_doys': []}}},
        'gwsub_data': {fid: {'2020': {'f_sub': 0.0}}},
        'ke_max': {fid: 0.4},
        'kc_max': {fid: 1.1},
        'order': [fid],
        'time_series': time_series,
    }
    out = os.path.join(tmpdir, 'prepped_input.json')
    with open(out, 'w') as f:
        for k, v in prepped.items():
            json.dump({k: v}, f)
            f.write('\n')
    return out


def _minimal_config(tmpdir, prepped_path, fields_shp):
    # Create a minimal TOML in-line
    cfg = f'''
project = "test"
root = "{tmpdir}"

[paths]
project_workspace = "{tmpdir}"
data = "{tmpdir}"
landsat = "{tmpdir}/landsat"
landsat_tables = "{tmpdir}/landsat/tables"
sentinel = "{tmpdir}/sentinel"
sentinel_tables = "{tmpdir}/sentinel/tables"
met = "{tmpdir}/met"
gis = "{tmpdir}/gis"
fields_shapefile = "{fields_shp}"
properties = "{tmpdir}/properties"
irr = "{tmpdir}/properties/irr.csv"
ssurgo = "{tmpdir}/properties/ssurgo.csv"
lulc = "{tmpdir}/properties/lulc.csv"
properties_json = "{tmpdir}/properties/properties.json"
snodas_in = "{tmpdir}/snodas/extracts"
snodas_out = "{tmpdir}/snodas/snodas.json"
remote_sensing_tables = "{tmpdir}/rs_tables"
joined_timeseries = "{tmpdir}/plot_timeseries"
dynamics_data = "{tmpdir}/dynamics.json"
prepped_input = "{prepped_path}"

[ids]
feature_id = "FID"

[misc]
irrigation_threshold = 0.3
elev_units = "m"
refet_type = "eto"
runoff_process = "cn"

[date_range]
start_date = "2020-01-01"
end_date = "2020-01-03"

[crop_coefficient]
kc_proxy = "etf"
cover_proxy = "ndvi"
'''
    cfg_path = os.path.join(tmpdir, 'config.toml')
    with open(cfg_path, 'w') as f:
        f.write(cfg)
    return cfg_path


def test_run_field_workflow_minimal(tmp_path):
    # Create dummy shapefile (GeoJSON with one square polygon converted via geopandas)
    import geopandas as gpd
    from shapely.geometry import Polygon

    poly = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    gdf = gpd.GeoDataFrame({'FID': ['TEST1'], 'geometry': [poly]}, crs='EPSG:4326')
    shp_dir = tmp_path / 'gis'
    shp_dir.mkdir(parents=True, exist_ok=True)
    shp_path = shp_dir / 'fields.shp'
    gdf.to_file(shp_path)

    prepped = _minimal_prepped_input(str(tmp_path))
    cfg = _minimal_config(str(tmp_path), prepped, str(shp_path))

    out = run_field_workflow(cfg, field_id='TEST1', debug=False)
    # Returns (etf, swe) arrays
    assert isinstance(out, tuple) and len(out) == 2
    etf, swe = out
    assert etf.shape[0] > 0 and swe.shape[0] > 0
