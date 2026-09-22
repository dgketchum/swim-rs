"""Build the georeferenced metered-well applied-water deliverables for Division 3
(Rio Grande / San Luis Valley), Colorado.

Inputs (produced by pull_structures.py, pull_divrec.py, build_parcels.py):
  div3_wells.parquet                 WELL structures (coords, has_meter, attrs)
  div3_well_divrecyear.parquet       annual pumping records (divrecyear, ACFT)
  co_slv_parcel_well_links.parquet   parcel<->gw_well links (acres, n wells)
  co_slv_irrigated_parcels.fgb       crop-typed parcels (for irr_class join)

Outputs (EPSG:4326):
  co_slv_well_pumping.fgb            one POINT feature per (well, year) w/ pumped_af
  co_slv_well_pumping.parquet        same, tabular
  co_slv_well_year_applied_depth.parquet
        per (well, year): served irrigated acres (from parcels listing the well as
        a GW supply) and applied depth = pumped_af / served_acres.  Two acreage
        estimates: *_sum (all parcels listing the well) and *_alloc (parcel acres
        split equally among the wells on that parcel).  irr-class breakdown of the
        served acreage is included so groundwater-only (G) fields can be isolated.
"""

import os

import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    wells = pd.read_parquet(os.path.join(HERE, "div3_wells.parquet"))
    pump = pd.read_parquet(os.path.join(HERE, "div3_well_divrecyear.parquet"))

    wells["wdid"] = wells["wdid"].astype(str)
    pump["wdid"] = pump["wdid"].astype(str)
    # Each well carries a structured water class AND a "<wdid> Total (Diversion)"
    # rollup class with the well's annual total. Keep only the Total to get one
    # clean per-well annual value (and to correctly total multi-class wells).
    pump = pump[pump["wcIdentifier"].str.endswith("Total (Diversion)")].copy()
    pump["year"] = pump["dataMeasDate"].astype(int)
    pump = pump.rename(
        columns={
            "dataValue": "pumped_af",
            "obsCode": "obs_code",
            "approvalStatus": "approval_status",
        }
    )
    pump = pump.drop_duplicates(["wdid", "year"])

    wattr = wells[
        [
            "wdid",
            "structureName",
            "waterDistrict",
            "subdistrict",
            "county",
            "designatedBasinName",
            "managementDistrictName",
            "latdecdeg",
            "longdecdeg",
            "has_meter",
        ]
    ].rename(
        columns={
            "structureName": "structure_name",
            "waterDistrict": "water_district",
            "designatedBasinName": "designated_basin",
            "managementDistrictName": "mgmt_district",
            "latdecdeg": "lat",
            "longdecdeg": "lon",
        }
    )

    df = pump.merge(wattr, on="wdid", how="left")
    df = df[df["lat"].notna() & df["lon"].notna()].copy()
    keep = [
        "wdid",
        "structure_name",
        "water_district",
        "subdistrict",
        "county",
        "designated_basin",
        "mgmt_district",
        "year",
        "pumped_af",
        "measUnits",
        "obs_code",
        "approval_status",
        "has_meter",
        "lat",
        "lon",
    ]
    df = df[keep].rename(columns={"measUnits": "units"})

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    gdf.to_file(os.path.join(HERE, "co_slv_well_pumping.fgb"), driver="FlatGeobuf")
    df.to_parquet(os.path.join(HERE, "co_slv_well_pumping.parquet"), index=False)
    print(f"co_slv_well_pumping.fgb: {len(gdf)} well-year point features")
    print(f"  distinct wells: {df['wdid'].nunique()}  years: {df['year'].min()}-{df['year'].max()}")
    print(f"  units: {df['units'].value_counts().to_dict()}")

    # --- applied depth: join pumping to served irrigated acreage ---
    links_fp = os.path.join(HERE, "co_slv_parcel_well_links.parquet")
    parcels_fp = os.path.join(HERE, "co_slv_irrigated_parcels.fgb")
    if not (os.path.exists(links_fp) and os.path.exists(parcels_fp)):
        print("SKIP applied-depth join (parcels not built yet)")
        return

    links = pd.read_parquet(links_fp)
    links["gw_wdid"] = links["gw_wdid"].astype(str)
    # irr_class per parcel-year from the parcels layer (attrs only)
    pattr = gpd.read_file(parcels_fp, engine="fiona", ignore_geometry=True)[
        ["cal_year", "parcel_id", "irr_class"]
    ].drop_duplicates()
    pattr["parcel_id"] = pattr["parcel_id"].astype(str)
    links["parcel_id"] = links["parcel_id"].astype(str)
    links = links.merge(pattr, on=["cal_year", "parcel_id"], how="left")
    links["acres_alloc"] = links["parcel_acres"] / links["n_gw_wells_on_parcel"].clip(lower=1)

    def agg(g):
        gonly = g[g["irr_class"] == "G"]
        return pd.Series(
            {
                "served_acres_sum": g["parcel_acres"].sum(),
                "served_acres_alloc": g["acres_alloc"].sum(),
                "n_parcels": g["parcel_id"].nunique(),
                "acres_alloc_G": gonly["acres_alloc"].sum(),
                "frac_acres_G": (gonly["acres_alloc"].sum() / g["acres_alloc"].sum())
                if g["acres_alloc"].sum()
                else 0.0,
            }
        )

    served = (
        links.groupby(["cal_year", "gw_wdid"], group_keys=False)
        .apply(agg, include_groups=False)
        .reset_index()
    )
    served = served.rename(columns={"cal_year": "year", "gw_wdid": "wdid"})

    dep = df.merge(served, on=["wdid", "year"], how="left")
    dep["applied_depth_ft_alloc"] = dep["pumped_af"] / dep["served_acres_alloc"]
    dep["applied_depth_ft_sum"] = dep["pumped_af"] / dep["served_acres_sum"]
    dep["applied_depth_mm_alloc"] = dep["applied_depth_ft_alloc"] * 304.8
    dep.to_parquet(os.path.join(HERE, "co_slv_well_year_applied_depth.parquet"), index=False)

    matched = dep[dep["served_acres_alloc"] > 0]
    clean = matched[(matched["frac_acres_G"] >= 0.9) & (matched["pumped_af"] > 0)]
    print(f"\nco_slv_well_year_applied_depth.parquet: {len(dep)} well-years")
    print(f"  with matched irrigated acreage: {len(matched)}")
    print(f"  clean groundwater-only (>=90% G-class acres, pumped>0): {len(clean)}")
    if len(clean):
        print(
            f"  clean applied depth (ft) median {clean['applied_depth_ft_alloc'].median():.2f} "
            f"IQR {clean['applied_depth_ft_alloc'].quantile(0.25):.2f}-"
            f"{clean['applied_depth_ft_alloc'].quantile(0.75):.2f}"
        )


if __name__ == "__main__":
    main()
