import geopandas as gpd
import pandas as pd


def create_filtered_shapefiles():
    """Filter flux sites and create point and buffer shapefiles.

    Filters: n_days > 60, end > 2021-01-01, (glc10_lc == 10.0 OR modis_lc == 12.0)
    Outputs: points shapefile and 150m buffer shapefile
    """
    # Read the shapefile
    gdf = gpd.read_file("/data/ssd2/swim/6_Flux_International/data/gis/flux_intl_28DEC2025.shp")

    # Convert end to datetime if it's not already
    gdf["end"] = pd.to_datetime(gdf["end"])

    # Filter: n_days > 60, end > 2021-01-01, and (glc10_lc == 10.0 OR modis_lc == 12.0)
    filtered = gdf[
        (gdf["n_days"] > 60)
        & (gdf["end"] > pd.Timestamp("2021-01-01"))
        & ((gdf["glc10_lc"] == 10.0) | (gdf["modis_lc"] == 12.0))
    ]

    print(f"Original records: {len(gdf)}")
    print(f"Filtered records: {len(filtered)}")
    print("\nFiltered sites:")
    print(filtered[["sid", "n_days", "end", "glc10_lc", "modis_lc", "lat", "lon"]])

    # Create points from lat/lon
    points = filtered.copy()
    points["geometry"] = gpd.points_from_xy(points["lon"], points["lat"])
    points = points.set_crs(epsg=4326)
    points = points.drop(columns=["file"])

    # Save points shapefile
    out_dir = "/data/ssd2/swim/6_Flux_International/data/gis"
    points.to_file(f"{out_dir}/flux_intl_points_06JAN2026.shp")
    print(f"\nSaved points to {out_dir}/flux_intl_points_06JAN2026.shp")

    # Create 150m buffers - need to project to a metric CRS first
    # Use Web Mercator for global coverage
    points_projected = points.to_crs(epsg=3857)
    buffers = points_projected.copy()
    buffers["geometry"] = buffers.geometry.buffer(150)

    # Convert back to WGS84
    buffers = buffers.to_crs(epsg=4326)

    # Save buffers shapefile
    buffers.to_file(f"{out_dir}/flux_intl_buffers_150m_06JAN2026.shp")
    print(f"Saved 150m buffers to {out_dir}/flux_intl_buffers_150m_06JAN2026.shp")


if __name__ == "__main__":
    create_filtered_shapefiles()
