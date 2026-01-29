import json
import os
from datetime import timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import pytz
from rasterstats import zonal_stats
from tqdm import tqdm

from swimrs.utils.optional_deps import missing_optional_dependency

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

    fields = gpd.read_file(fields)
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
        pts = gpd.read_file(gridmet_points)
        if pts.crs != fields_cent.crs:
            pts = pts.to_crs(fields_cent.crs)

        keep_cols = [c for c in [gridmet_id_col, "lat", "lon", "geometry"] if c in pts.columns]
        pts = pts[keep_cols]

        joined = gpd.sjoin_nearest(
            fields_cent[[feature_id, "geometry"]], pts, how="left", distance_col="dist"
        )

        fields[gridmet_id_col] = joined[gridmet_id_col].values
        fields["STATION_ID"] = fields[gridmet_id_col]

        pts_indexed = pts.set_index(gridmet_id_col)
        # Fill lat/lon from centroids if provided on pts
        for gfid, row in pts_indexed.iterrows():
            if "lat" in row and "lon" in row:
                fields.loc[fields[gridmet_id_col] == gfid, "LAT"] = float(row["lat"])
                fields.loc[fields[gridmet_id_col] == gfid, "LON"] = float(row["lon"])
    else:
        fields[gridmet_id_col] = range(len(fields))
        fields["STATION_ID"] = fields[gridmet_id_col]

    for i, field in tqdm(fields.iterrows(), desc="Fetching elevations", total=fields.shape[0]):
        g = GridMet("elev", lat=fields.at[i, "LAT"], lon=fields.at[i, "LON"])
        elev = g.get_point_elevation()
        fields.at[i, "ELEV"] = elev

    oshape = fields.shape[0]
    fields = fields[~pd.isna(fields[gridmet_id_col])]
    print(f"Writing {fields.shape[0]} of {oshape} input features")
    fields[gridmet_id_col] = fields[gridmet_id_col].fillna(-1).astype(int)
    fields.to_file(fields_join, crs=fields.crs or "EPSG:5071", engine="fiona")
    return fields


def sample_gridmet_corrections(fields_join, gridmet_ras, factors_js, gridmet_id_col="GFID"):
    """Sample correction rasters and write factors JSON keyed by GFID."""
    fields = gpd.read_file(fields_join)
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
    use_nldas=False,
    gridmet_id_col="GFID",
):
    """Download GridMET time series and optionally NLDAS-2 hourly precipitation.

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
        use_nldas: If True, include hourly precipitation from NLDAS-2
    """
    if not start:
        start = "1987-01-01"
    if not end:
        end = "2021-12-31"

    fields = gpd.read_file(fields)
    fields.index = fields[feature_id]

    gridmet_factors_dict = {}
    if gridmet_factors and os.path.exists(gridmet_factors):
        with open(gridmet_factors) as f:
            gridmet_factors_dict = json.load(f)

    hr_cols = ["prcp_hr_{}".format(str(i).rjust(2, "0")) for i in range(0, 24)]

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
                lat = fields.at[fields[gridmet_id_col] == int(g_fid), "LAT"].values[0]
                lon = fields.at[fields[gridmet_id_col] == int(g_fid), "LON"].values[0]

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

                # Download NLDAS hourly precip if requested
                if thredds_var == "pr" and use_nldas:
                    try:
                        import pynldas2 as nld
                    except ImportError as exc:
                        raise missing_optional_dependency(
                            extra="nldas",
                            purpose="NLDAS-2 hourly precipitation (runoff_process='ier')",
                            import_name="pynldas2",
                        ) from exc

                    s_nldas = pd.to_datetime(dl_start) - timedelta(days=1)
                    e_nldas = pd.to_datetime(dl_end) + timedelta(days=2)
                    nldas = nld.get_bycoords(
                        (lon, lat),
                        start_date=s_nldas.strftime("%Y-%m-%d"),
                        end_date=e_nldas.strftime("%Y-%m-%d"),
                        variables=["prcp"],
                    )
                    if nldas.size == 0:
                        raise ValueError(f"Failed to download NLDAS-2 for GFID {g_fid}")

                    central = pytz.timezone("US/Central")
                    nldas = nldas.tz_convert(central)
                    hourly_ppt = nldas.pivot_table(
                        columns=nldas.index.hour, index=nldas.index.date, values="prcp"
                    )
                    df[hr_cols] = hourly_ppt.loc[df.index]

                    nan_ct = np.sum(np.isnan(df[hr_cols].values), axis=0)
                    if sum(nan_ct) > 100:
                        raise ValueError("Too many NaN in NLDAS data")
                    if np.any(nan_ct):
                        df[hr_cols] = df[hr_cols].fillna(0.0)

                    df["nld_ppt_d"] = df[hr_cols].sum(axis=1)

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
            if use_nldas:
                out_cols.extend(["nld_ppt_d"] + hr_cols)

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
    df = gpd.read_file(shp_in)
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
