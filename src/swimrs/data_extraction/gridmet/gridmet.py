import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import zonal_stats
from tqdm import tqdm

from swimrs.data_extraction.gridmet.thredds import GridMet

CLIMATE_COLS = {
    "etr": {
        "nc": "agg_met_etr_1979_CurrentYear_CONUS",
        "var": "daily_mean_reference_evapotranspiration_alfalfa",
        "col": "etr",
    },
    "pet": {
        "nc": "agg_met_pet_1979_CurrentYear_CONUS",
        "var": "daily_mean_reference_evapotranspiration_grass",
        "col": "eto",
    },
    "pr": {"nc": "agg_met_pr_1979_CurrentYear_CONUS", "var": "precipitation_amount", "col": "prcp"},
    "srad": {
        "nc": "agg_met_srad_1979_CurrentYear_CONUS",
        "var": "daily_mean_shortwave_radiation_at_surface",
        "col": "srad",
    },
    "tmmx": {
        "nc": "agg_met_tmmx_1979_CurrentYear_CONUS",
        "var": "daily_maximum_temperature",
        "col": "tmax",
    },
    "tmmn": {
        "nc": "agg_met_tmmn_1979_CurrentYear_CONUS",
        "var": "daily_minimum_temperature",
        "col": "tmin",
    },
    "vs": {
        "nc": "agg_met_tmmn_1979_CurrentYear_CONUS",
        "var": "daily_minimum_temperature",
        "col": "u2",
    },
    "sph": {
        "nc": "agg_met_tmmn_1979_CurrentYear_CONUS",
        "var": "daily_minimum_temperature",
        "col": "q",
    },
}


def _build_raster_list(gridmet_ras):
    """Return list of monthly correction raster paths for ETo/ETr.

    Parameters
    - gridmet_ras: directory containing `gridmet_corrected_<var>_<month>.tif`.

    Returns
    - list[str] of absolute paths for 12 months and both variables.
    """
    rasters = []
    for v in ["eto", "etr"]:
        [
            rasters.append(os.path.join(gridmet_ras, f"gridmet_corrected_{v}_{m}.tif"))
            for m in range(1, 13)
        ]
    return rasters


def _compute_lat_lon_from_centroids(gdf_5071):
    """Compute centroid latitude/longitude from a 5071-projected GeoDataFrame.

    Returns two numpy arrays of latitude and longitude in EPSG:4326.
    """
    centroids = gdf_5071.geometry.centroid
    wgs84 = centroids.to_crs("EPSG:4326")
    return wgs84.y.values, wgs84.x.values


def assign_gridmet_ids(
    fields,
    fields_join,
    gridmet_points=None,
    field_select=None,
    feature_id="FID",
    gridmet_id_col="GFID",
):
    """Map fields to GridMET IDs (optionally via provided centroids) and write join shapefile."""
    print("Assign field -> GridMET IDs")

    fields = gpd.read_file(fields, engine="fiona")
    if fields.crs is None:
        fields.set_crs("EPSG:5071", inplace=True)

    fields_cent = fields.copy()
    fields_cent["geometry"] = fields_cent.geometry.centroid
    lat_vals, lon_vals = _compute_lat_lon_from_centroids(fields_cent)
    fields["LAT"] = lat_vals
    fields["LON"] = lon_vals

    if field_select is not None:
        mask = fields[feature_id].astype(str).isin(set(field_select))
        fields = fields.loc[mask].copy()
        fields_cent = fields_cent.loc[mask].copy()

    if gridmet_points is not None:
        pts = gpd.read_file(gridmet_points, engine="fiona")

        keep_cols = [c for c in [gridmet_id_col, "lat", "lon", "geometry"] if c in pts.columns]
        pts = pts[keep_cols]

        # Grid-snap approach: round field centroids to nearest GridMET 1/24-deg cell
        pts_4326 = pts.to_crs("EPSG:4326") if pts.crs != "EPSG:4326" else pts
        grid_res = 1.0 / 24.0
        pts_4326["_rlat"] = (pts_4326.geometry.y / grid_res).round() * grid_res
        pts_4326["_rlon"] = (pts_4326.geometry.x / grid_res).round() * grid_res
        lookup = pts_4326.drop_duplicates(subset=["_rlat", "_rlon"]).set_index(["_rlat", "_rlon"])

        fields_4326 = fields.to_crs("EPSG:4326") if fields.crs != "EPSG:4326" else fields
        centroids = fields_4326.geometry.centroid
        fields["_rlat"] = (centroids.y / grid_res).round() * grid_res
        fields["_rlon"] = (centroids.x / grid_res).round() * grid_res

        merged = fields[["_rlat", "_rlon"]].merge(
            lookup[[gridmet_id_col, "lat", "lon"]],
            left_on=["_rlat", "_rlon"],
            right_index=True,
            how="left",
        )
        fields[gridmet_id_col] = merged[gridmet_id_col].values
        fields["STATION_ID"] = fields[gridmet_id_col]
        fields["LAT"] = merged["lat"].values
        fields["LON"] = merged["lon"].values
        fields.drop(columns=["_rlat", "_rlon"], inplace=True)

        n_unique = fields[gridmet_id_col].nunique()
        print(f"Mapped {len(fields)} fields to {n_unique} unique GridMET cells")
        if n_unique == 0:
            raise ValueError(
                f"No fields mapped to any GridMET cell; {gridmet_points} "
                "likely does not cover the field extent. Refusing to write "
                "an empty mapping shapefile."
            )
    else:
        fields[gridmet_id_col] = range(len(fields))
        fields["STATION_ID"] = fields[gridmet_id_col]

    # Fetch elevation once per unique lat/lon pair
    unique_locs = fields.drop_duplicates(subset=["LAT", "LON"])[["LAT", "LON"]]
    elev_cache = {}
    for _, loc in tqdm(unique_locs.iterrows(), desc="Fetching elevations", total=len(unique_locs)):
        key = (loc["LAT"], loc["LON"])
        g = GridMet("elev", lat=key[0], lon=key[1])
        elev_cache[key] = g.get_point_elevation()
    fields["ELEV"] = fields.apply(lambda r: elev_cache.get((r["LAT"], r["LON"])), axis=1)

    oshape = fields.shape[0]
    fields = fields[~pd.isna(fields[gridmet_id_col])]
    print(f"Writing {fields.shape[0]} of {oshape} input features")
    fields[gridmet_id_col] = fields[gridmet_id_col].fillna(-1).astype(int)
    fields.to_file(fields_join, crs=fields.crs or "EPSG:5071", engine="fiona")
    return fields


def sample_gridmet_corrections(fields_join, gridmet_ras, factors_js, gridmet_id_col="GFID"):
    """Sample correction rasters and write factors JSON keyed by GFID."""
    fields = gpd.read_file(fields_join, engine="fiona")
    if fields.crs is None:
        fields.set_crs("EPSG:5071", inplace=True)

    rasters = _build_raster_list(gridmet_ras)
    gridmet_targets = {}

    # Handle case-insensitive lat/lon column names
    lat_col = "LAT" if "LAT" in fields.columns else "lat"
    lon_col = "LON" if "LON" in fields.columns else "lon"

    for i, field in tqdm(
        fields.iterrows(), desc="Sampling correction rasters", total=fields.shape[0]
    ):
        gfid_int = int(fields.at[i, gridmet_id_col])
        geom = fields.at[i, "geometry"]
        gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=fields.crs)
        plat, plon = fields.at[i, lat_col], fields.at[i, lon_col]

        if gfid_int not in gridmet_targets:
            gridmet_targets[gfid_int] = {str(m): {} for m in range(1, 13)}
            gridmet_targets[gfid_int]["lat"] = plat
            gridmet_targets[gfid_int]["lon"] = plon

        for r in rasters:
            splt = r.split("_")
            _var, month = splt[-2], splt[-1].replace(".tif", "")
            stats = zonal_stats(gdf, r, stats=["mean"], nodata=np.nan)[0]["mean"]
            gridmet_targets[gfid_int][month].update({_var: stats})

    with open(factors_js, "w") as fp:
        json.dump(gridmet_targets, fp, indent=4)
    print(f"wrote {factors_js}")


def download_gridmet(
    fields,
    gridmet_factors,
    gridmet_csv_dir,
    start=None,
    end=None,
    overwrite=False,
    append=False,
    target_fields=None,
    feature_id="FID",
    return_df=False,
    gridmet_id_col="GFID",
):
    """Download GridMET time series.

    Downloads one parquet file per unique GFID (GridMET cell). Each file contains
    simple column names (e.g., 'tmin', 'tmax', 'eto', 'eto_corr') without field-specific
    information. The UID-to-GFID mapping is handled during ingestion.

    Output format:
        - Files named: {GFID}.parquet
        - Index: DatetimeIndex (daily dates)
        - Columns: Simple names like 'tmin', 'tmax', 'eto', 'eto_corr', 'prcp', etc.

    Args:
        fields: Path to shapefile with GFID column (from assign_gridmet_ids)
        gridmet_factors: Path to JSON with correction factors (from sample_gridmet_corrections)
        gridmet_csv_dir: Output directory for parquet files
        start: Start date (default: 1987-01-01)
        end: End date (default: 2021-12-31)
        overwrite: If True, overwrite existing files
        append: If True, append new dates to existing files
        target_fields: Optional list of field UIDs to filter GFIDs
        feature_id: Column name for field UID
        return_df: If True, return DataFrame after first download
    """
    if not start:
        start = "1987-01-01"
    if not end:
        end = "2021-12-31"

    fields = gpd.read_file(fields, engine="fiona")
    fields.index = fields[feature_id]

    gridmet_factors_dict = {}
    if gridmet_factors and os.path.exists(gridmet_factors):
        with open(gridmet_factors) as f:
            gridmet_factors_dict = json.load(f)

    # Get unique GFIDs to download
    if target_fields is not None:
        # Filter to GFIDs for the target fields
        target_fields_set = set(str(f) for f in target_fields)
        mask = fields.index.astype(str).isin(target_fields_set)
        filtered_fields = fields[mask]
        unique_gfids = filtered_fields[gridmet_id_col].dropna().unique()
    else:
        unique_gfids = fields[gridmet_id_col].dropna().unique()

    unique_gfids = [str(int(g)) for g in unique_gfids]
    print(f"Downloading {len(unique_gfids)} unique GridMET cells")

    downloaded, skipped_exists = [], []

    for g_fid in tqdm(unique_gfids, desc="Downloading GridMET"):
        _file = os.path.join(gridmet_csv_dir, f"{g_fid}.parquet")

        try:
            # Check if file exists
            if os.path.exists(_file) and not overwrite and not append:
                skipped_exists.append(_file)
                continue

            # Handle append mode
            dl_start, dl_end = start, end
            existing = None
            if os.path.exists(_file) and append:
                existing = pd.read_parquet(_file)
                target_dates = pd.date_range(start, end, freq="D")
                missing_dates = [i for i in target_dates if i not in existing.index]

                if len(missing_dates) == 0:
                    if return_df:
                        return existing
                    continue
                else:
                    dl_start = missing_dates[0].strftime("%Y-%m-%d")
                    dl_end = missing_dates[-1].strftime("%Y-%m-%d")

            # Get lat/lon
            if g_fid in gridmet_factors_dict:
                r = gridmet_factors_dict[g_fid]
                lat, lon = r["lat"], r["lon"]
            else:
                lat = fields.loc[fields[gridmet_id_col] == int(g_fid), "LAT"].values[0]
                lon = fields.loc[fields[gridmet_id_col] == int(g_fid), "LON"].values[0]

            # Download data from THREDDS
            df = pd.DataFrame()
            first = True

            for thredds_var, cols in CLIMATE_COLS.items():
                variable = cols["col"]

                if not thredds_var:
                    continue

                try:
                    g = GridMet(thredds_var, start=dl_start, end=dl_end, lat=lat, lon=lon)
                    s = g.get_point_timeseries()
                except OSError as e:
                    print(f"Error downloading {thredds_var} for GFID {g_fid}: {e}")
                    continue

                df[variable] = s[thredds_var]

                if first:
                    g = GridMet("elev", lat=lat, lon=lon)
                    elev = g.get_point_elevation()
                    df["elev"] = elev
                    first = False

            if df.empty:
                print(f"No data downloaded for GFID {g_fid}")
                continue

            # Calculate vapor pressure from specific humidity
            p_air = air_pressure(df["elev"])
            ea_kpa = actual_vapor_pressure(df["q"], p_air)
            df["ea"] = ea_kpa.copy()

            # Apply bias corrections for ETo and ETr
            if g_fid in gridmet_factors_dict:
                for variable in ["etr", "eto"]:
                    for month in range(1, 13):
                        corr_factor = gridmet_factors_dict[g_fid][str(month)].get(variable)
                        # Use factor of 1.0 (no correction) if factor is missing or None
                        if corr_factor is None:
                            corr_factor = 1.0
                        idx = [i for i in df.index if i.month == month]
                        df.loc[idx, f"{variable}_corr"] = df.loc[idx, variable] * corr_factor

            # Convert temperatures from Kelvin to Celsius
            df["tmax"] = df["tmax"] - 273.15
            df["tmin"] = df["tmin"] - 273.15

            # Drop intermediate columns not needed for output
            df = df.drop(columns=["q"], errors="ignore")

            # Select output columns (simple names, no MultiIndex)
            out_cols = [
                "tmin",
                "tmax",
                "eto",
                "etr",
                "eto_corr",
                "etr_corr",
                "prcp",
                "srad",
                "u2",
                "ea",
                "elev",
            ]

            # Keep only columns that exist
            out_cols = [c for c in out_cols if c in df.columns]
            df = df[out_cols]

            # Append to existing if needed
            if existing is not None and append:
                df = pd.concat([existing, df], axis=0)
                df = df.sort_index()
                # Remove duplicates keeping last
                df = df[~df.index.duplicated(keep="last")]

            df.to_parquet(_file)
            print(f"wrote {_file}")
            downloaded.append(g_fid)

            if return_df:
                return df

        except Exception as exc:
            print(f"Error on GFID {g_fid}: {exc}")
            continue

    print(f"Downloaded {len(downloaded)} files")
    print(f"Skipped {len(skipped_exists)} existing files")


# from CGMorton's RefET (github.com/WSWUP/RefET)
def air_pressure(elev, method="asce"):
    """Mean atmospheric pressure at station elevation (Eqs. 3 & 34)

    Parameters
    ----------
    elev : scalar or array_like of shape(M, )
        Elevation [m].
    method : {'asce' (default), 'refet'}, optional
        Calculation method:
        * 'asce' -- Calculations will follow ASCE-EWRI 2005 [1] equations.
        * 'refet' -- Calculations will follow RefET software.

    Returns
    -------
    ndarray
        Air pressure [kPa].

    Notes
    -----
    The current calculation in Ref-ET:
        101.3 * (((293 - 0.0065 * elev) / 293) ** (9.8 / (0.0065 * 286.9)))
    Equation 3 in ASCE-EWRI 2005:
        101.3 * (((293 - 0.0065 * elev) / 293) ** 5.26)
    Per Dr. Allen, the calculation with full precision:
        101.3 * (((293.15 - 0.0065 * elev) / 293.15) ** (9.80665 / (0.0065 * 286.9)))

    """
    pair = np.array(elev, copy=True, ndmin=1).astype(np.float64)
    pair *= -0.0065
    if method == "asce":
        pair += 293
        pair /= 293
        np.power(pair, 5.26, out=pair)
    elif method == "refet":
        pair += 293
        pair /= 293
        np.power(pair, 9.8 / (0.0065 * 286.9), out=pair)
    # np.power(pair, 5.26, out=pair)
    pair *= 101.3

    return pair


# from CGMorton's RefET (github.com/WSWUP/RefET)
def actual_vapor_pressure(q, pair):
    """ "Actual vapor pressure from specific humidity

    Parameters
    ----------
    q : scalar or array_like of shape(M, )
        Specific humidity [kg/kg].
    pair : scalar or array_like of shape(M, )
        Air pressure [kPa].

    Returns
    -------
    ndarray
        Actual vapor pressure [kPa].

    Notes
    -----
    ea = q * pair / (0.622 + 0.378 * q)

    """
    ea = np.array(q, copy=True, ndmin=1).astype(np.float64)
    ea *= 0.378
    ea += 0.622
    np.reciprocal(ea, out=ea)
    ea *= pair
    ea *= q

    return ea


# from CGMorton's RefET (github.com/WSWUP/RefET)
def wind_height_adjust(uz, zw):
    """Wind speed at 2 m height based on full logarithmic profile (Eq. 33)

    Parameters
    ----------
    uz : scalar or array_like of shape(M, )
        Wind speed at measurement height [m s-1].
    zw : scalar or array_like of shape(M, )
        Wind measurement height [m].

    Returns
    -------
    ndarray
        Wind speed at 2 m height [m s-1].

    """
    return uz * 4.87 / np.log(67.8 * zw - 5.42)


def gridmet_elevation(shp_in, shp_out):
    """Append elevation to point shapefile using GridMET point elevation service.

    Parameters
    - shp_in: input shapefile path with `lat`/`lon` fields.
    - shp_out: output shapefile path with new `ELEV_M` column.
    """
    df = gpd.read_file(shp_in, engine="fiona")
    l = []
    for i, r in df.iterrows():
        lat, lon = r["lat"], r["lon"]
        g = GridMet("elev", lat=lat, lon=lon)
        elev = g.get_point_elevation()
        l.append((i, elev))

    df["ELEV_M"] = [i[1] for i in l]
    df.to_file(shp_out)


if __name__ == "__main__":
    pass
# ========================= EOF ====================================================================
