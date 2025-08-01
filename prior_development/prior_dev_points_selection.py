import os
import geopandas as gpd
import pandas as pd
import os
from collections import Counter
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


def analyze_crop_consistency(stations_path, points_base_path, cdl_base_path, output_dir, output_filename,
                             buffer_distance, target_crs, target_n=50, target_class=None, states=None):
    stations_gdf = gpd.read_file(stations_path)

    if stations_gdf.crs.to_string() != target_crs:
        stations_gdf = stations_gdf.to_crs(target_crs)

    if target_class:
        stations_gdf = stations_gdf[stations_gdf['classifica'] == target_class]

    all_top_fields_gdfs = []

    for state_abbr, stations_in_state in stations_gdf.groupby('state'):

        if states is not None and state_abbr not in states:
            continue

        print(f'Processing {state_abbr}')

        points_shp_path = os.path.join(points_base_path, f"{state_abbr}.shp")
        cdl_csv_path = os.path.join(cdl_base_path, f"{state_abbr}.csv")

        if not os.path.exists(points_shp_path) or not os.path.exists(cdl_csv_path):
            continue

        state_points_gdf = gpd.read_file(points_shp_path)

        state_cdl_df = pd.read_csv(cdl_csv_path)
        state_cdl_df.rename(columns={state_cdl_df.columns[0]: 'OPENET_ID'}, inplace=True)

        for _, station in stations_in_state.iterrows():
            station_id = station.get('STATION_ID', f"station_{_}")

            buffer_geom = station.geometry.buffer(buffer_distance)
            points_in_buffer = state_points_gdf[state_points_gdf.intersects(buffer_geom)]

            if points_in_buffer.empty:
                continue

            openet_ids_in_buffer = points_in_buffer['OPENET_ID'].tolist()
            fields_cdl_data = state_cdl_df[state_cdl_df['OPENET_ID'].isin(openet_ids_in_buffer)]

            if fields_cdl_data.empty:
                continue

            crop_cols = [col for col in fields_cdl_data.columns if col.startswith('CROP_')]

            def calculate_consistency(row):
                crops = [c for c in row.values if pd.notna(c) and c != 0]
                if not crops:
                    return 0, None

                counts = Counter(crops)
                most_common = counts.most_common(1)[0]
                return most_common[1], most_common[0]

            consistency_results = fields_cdl_data[crop_cols].apply(calculate_consistency, axis=1, result_type='expand')
            fields_cdl_data[['consistency', 'major_crop']] = consistency_results

            top_50_fields_cdl = fields_cdl_data.nlargest(target_n, 'consistency')

            if top_50_fields_cdl.empty:
                continue

            top_50_ids = top_50_fields_cdl['OPENET_ID'].tolist()
            top_fields_for_station_gdf = points_in_buffer[points_in_buffer['OPENET_ID'].isin(top_50_ids)].copy()

            top_fields_for_station_gdf['station_id'] = station_id
            top_fields_for_station_gdf = top_fields_for_station_gdf.merge(
                top_50_fields_cdl[['OPENET_ID', 'consistency', 'major_crop']],
                on='OPENET_ID'
            )

            all_top_fields_gdfs.append(top_fields_for_station_gdf)
            print(f'Added {len(top_fields_for_station_gdf)} points from {station_id} in {state_abbr}')

    if all_top_fields_gdfs:
        final_gdf = pd.concat(all_top_fields_gdfs, ignore_index=True)

        final_gdf.rename(columns={
            'consistency': 'consistncy',
            'major_crop': 'major_crop',
            'station_id': 'stn_id'
        }, inplace=True)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        final_gdf.to_file(output_path, driver='ESRI Shapefile', crs=target_crs)


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS'
    if not os.path.exists(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')


    stations_path_ = os.path.join(d, 'climate', 'flux_ET_dataset', 'station_metadata_simple_5071.shp')
    points_base_path_ = os.path.join(d, 'openET', 'OpenET_GeoDatabase_centroids_5071')
    cdl_base_path_ = os.path.join(d, 'openET', 'OpenET_GeoDatabase_cdl')
    output_dir_ = os.path.join(d, 'swim', 'prior_dev')
    buffer_distance_meters = 5000
    target_crs_ = 'EPSG:5071'

    num_targets = 50
    output_filename_ = 'prior_targets_flux_crops_50.shp'

    analyze_crop_consistency(stations_path_,
                             points_base_path_,
                             cdl_base_path_,
                             output_dir_,
                             output_filename_,
                             buffer_distance_meters,
                             target_crs_,
                             target_class='Croplands',
                             states=None)

# ========================= EOF ====================================================================
