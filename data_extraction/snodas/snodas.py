import os
import json

import geopandas as gpd

from rasterstats import zonal_stats


def snodas_zonal_stats(in_shp, raster_dir, out_js, targets=None):
    df = gpd.read_file(in_shp)
    df.index = [i for i in df['FID']]

    if targets:
        df = df.loc[targets]

    geo, fids = list(df['geometry']), list(df['FID'])

    l = sorted([os.path.join(raster_dir, x) for x in os.listdir(raster_dir) if x.endswith('.tif')])

    dct = {}

    for r in l:
        dts = os.path.basename(r).replace('.tif', '').split('_')[-1]
        dct[dts] = {}
        stats = zonal_stats(geo, r, stats=['mean'])
        for fid, s in zip(fids, stats):
            if s['mean']:
                dct[dts][fid] = float(s['mean'])
            else:
                dct[dts][fid] = 0.0
        print(os.path.basename(r), dct[dts][1801])

    with open(out_js, 'w') as fp:
        json.dump(dct, fp, indent=4)

    print('wrote', out_js)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
