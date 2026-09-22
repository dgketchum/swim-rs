"""Build crop-typed irrigated-field polygons for Division 3 (Rio Grande / San Luis
Valley) from the RGDSS (Rio Grande Decision Support System) annual irrigated-lands
shapefiles, and a long parcel<->well link table.

Source zip (CDSS FTP, no credentials):
  https://dnrftp.state.co.us/CDSS/GIS/RGDSS_CDSS_Shapefiles.zip
Contains Div3_Irrig_<YEAR>.shp for YEAR in {1936,1998,2002,2005,2009..2024},
EPSG:26913 (UTM 13N NAD83). Each parcel carries CROP_TYPE, IRRIG_TYPE, ACRES,
IRR class (S=surface, G=groundwater, B=both), up to 9 SW_WDID* surface structures
and up to 20 GW_ID* groundwater-well WDIDs (GW_TYPE*=='WDID').

Outputs (EPSG:4326):
  co_slv_irrigated_parcels.fgb        one polygon feature per parcel-year
  co_slv_parcel_well_links.parquet    one row per (cal_year, parcel_id, gw_wdid)
"""

import glob
import os
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP_URL = "https://dnrftp.state.co.us/CDSS/GIS/RGDSS_CDSS_Shapefiles.zip"
RAW = os.path.join(HERE, "raw")
SHPDIR = os.path.join(RAW, "RGDSS_SHAPEFILES")
# Only build the Landsat-era years that overlap the metered well-pumping record
# (divrecyear pumping starts 2009). Source also has 1936, 1998, 2002, 2005.
MIN_YEAR = 2009


def ensure_raw():
    if glob.glob(os.path.join(SHPDIR, "Div3_Irrig_*.shp")):
        return
    os.makedirs(RAW, exist_ok=True)
    zp = os.path.join(RAW, "RGDSS_CDSS_Shapefiles.zip")
    if not os.path.exists(zp):
        print("downloading", ZIP_URL)
        r = requests.get(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=600)
        r.raise_for_status()
        open(zp, "wb").write(r.content)
    with zipfile.ZipFile(zp) as z:
        z.extractall(RAW)


def joinids(row, cols):
    vals = []
    for c in cols:
        v = row.get(c)
        if pd.isna(v):
            continue
        v = str(v).strip()
        if v and v not in ("0", "None", "nan"):
            vals.append(v)
    return ",".join(dict.fromkeys(vals))  # dedupe, preserve order


def main():
    ensure_raw()
    metered = set()
    mfile = os.path.join(HERE, "metered_well_wdids.json")
    if os.path.exists(mfile):
        import json

        metered = set(json.load(open(mfile)))

    gw_id_cols = [f"GW_ID{i}" for i in range(1, 21)]
    gw_type_cols = [f"GW_TYPE{i}" for i in range(1, 21)]
    sw_cols = [f"SW_WDID{i}" for i in range(1, 10)]

    keep_gdf, links = [], []
    for fp in sorted(glob.glob(os.path.join(SHPDIR, "Div3_Irrig_*.shp"))):
        yr = int(os.path.basename(fp).split("_")[-1].split(".")[0])
        if yr < MIN_YEAR:
            continue
        g = gpd.read_file(fp, engine="fiona")
        if g.crs is None:
            g = g.set_crs("EPSG:26913")
        g = g.to_crs("EPSG:4326")
        # groundwater WDIDs: only where GW_TYPE*=='WDID'
        gw_wdids_list = []
        for _, row in g.iterrows():
            ids = []
            for tc, ic in zip(gw_type_cols, gw_id_cols):
                t = row.get(tc)
                v = row.get(ic)
                if pd.notna(t) and str(t).strip() == "WDID" and pd.notna(v):
                    vv = str(v).strip()
                    if vv and vv not in ("0", "None", "nan"):
                        ids.append(vv)
            ids = list(dict.fromkeys(ids))
            gw_wdids_list.append(ids)

        out = pd.DataFrame(
            {
                "cal_year": g["CAL_YEAR"].astype("Int64"),
                "division": g["DIV"].astype("Int64"),
                "district": g["DISTRICT"].astype("Int64"),
                "parcel_id": g["PARCEL_ID"].astype(str),
                "master_id": g.get("MASTER_ID"),
                "crop_type": g["CROP_TYPE"],
                "crop_src": g["CROP_SRC"],
                "irrig_type": g["IRRIG_TYPE"],
                "acres": g["ACRES"].astype(float),
                "irr_class": g["IRR"],
                "sw_wdids": g.apply(lambda r: joinids(r, sw_cols), axis=1),
            }
        )
        out["gw_wdids"] = [",".join(x) for x in gw_wdids_list]
        out["n_gw_wells"] = [len(x) for x in gw_wdids_list]
        out["n_gw_metered"] = [sum(1 for w in x if w in metered) for x in gw_wdids_list]
        out["farm_unit"] = g.get("FARM_UNIT")
        out = gpd.GeoDataFrame(out, geometry=g.geometry.values, crs="EPSG:4326")
        keep_gdf.append(out)

        for yr, pid, ids, ac, ngw in zip(
            out["cal_year"], out["parcel_id"], gw_wdids_list, out["acres"], out["n_gw_wells"]
        ):
            for w in ids:
                links.append((int(yr), pid, w, float(ac), int(ngw)))
        print(f"  {os.path.basename(fp)}: {len(out)} parcels")

    parcels = gpd.GeoDataFrame(pd.concat(keep_gdf, ignore_index=True), crs="EPSG:4326")
    out_fgb = os.path.join(HERE, "co_slv_irrigated_parcels.fgb")
    parcels.to_file(out_fgb, driver="FlatGeobuf")
    print(f"\nwrote {out_fgb}: {len(parcels)} parcel-year polygons, crs {parcels.crs}")
    print("bbox:", np.round(parcels.total_bounds, 4).tolist())

    lk = pd.DataFrame(
        links, columns=["cal_year", "parcel_id", "gw_wdid", "parcel_acres", "n_gw_wells_on_parcel"]
    )
    lk.to_parquet(os.path.join(HERE, "co_slv_parcel_well_links.parquet"), index=False)
    print(f"wrote co_slv_parcel_well_links.parquet: {len(lk)} parcel-well links")


if __name__ == "__main__":
    main()
