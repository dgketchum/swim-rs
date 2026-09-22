"""Build idwr_wmis_applied_water.fgb: long-format (WMIS POD x year) metered diversion
volumes joined to water-right place-of-use acreage. EPSG:4326.

Sources:
  wmis_raw.json  -- IDWR WMIS FeatureServer (annual diverted acre-feet 2010-2025, per-POD)
  pou_lookup.parquet -- IDWR WaterRightPous (TotalAcres / AcreLimit / WaterUse per RightID)
Volumes are GROSS DIVERSION (applied water), NOT consumptive use / ET.
"""

import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd

YEARS = list(range(2010, 2026))

# ---- load WMIS ----
d = json.load(open("wmis_raw.json"))
recs = []
for f in d["features"]:
    a = f["attributes"]
    g = f.get("geometry") or {}
    a["_x"] = g.get("x")
    a["_y"] = g.get("y")
    recs.append(a)
wm = pd.DataFrame(recs)


def clean(s):
    if s is None:
        return None
    s = str(s).strip()
    return s if s not in ("", "nan", "None", "NaN") else None


for c in wm.columns:
    if wm[c].dtype == object:
        wm[c] = wm[c].map(clean)

# ---- POU acreage/water-use lookup keyed on RightID ----
pou = pd.read_parquet("pou_lookup.parquet")
irr = pou[pou["WaterUse"] == "IRRIGATION"].copy()
# per-RightID irrigation acres (max across any irrigation rows for that RightID)
rid_irr_acres = irr.groupby("RightID")["TotalAcres"].max().to_dict()
rid_acre_limit = irr.groupby("RightID")["AcreLimit"].max().to_dict()
rid_uses = (
    pou.groupby("RightID")["WaterUse"].apply(lambda s: sorted(set(x for x in s if x))).to_dict()
)
rid_wrnum = (
    pou.groupby("RightID")["WaterRightNumber"]
    .apply(lambda s: sorted(set(x for x in s if x)))
    .to_dict()
)


def parse_rids(s):
    if not s:
        return []
    return [int(t) for t in re.split(r"[,\s]+", str(s).strip()) if t.isdigit()]


# ---- per-POD aggregates ----
def pod_attrs(rid_str):
    rids = parse_rids(rid_str)
    uses = set()
    wrnums = set()
    irr_acres, acre_lim = [], []
    n_irr = 0
    for r in rids:
        for u in rid_uses.get(r, []):
            uses.add(u)
        for w in rid_wrnum.get(r, []):
            wrnums.add(w)
        if r in rid_irr_acres and rid_irr_acres[r] and rid_irr_acres[r] > 0:
            irr_acres.append(rid_irr_acres[r])
            n_irr += 1
        if r in rid_acre_limit and rid_acre_limit[r] and rid_acre_limit[r] > 0:
            acre_lim.append(rid_acre_limit[r])
    return {
        "n_rights": len(rids),
        "n_irr_rights": n_irr,
        "has_irrigation": "IRRIGATION" in uses,
        "irr_only": bool(uses) and uses == {"IRRIGATION"},
        "water_uses": ";".join(sorted(uses)) if uses else None,
        "wr_numbers": ";".join(sorted(wrnums)) if wrnums else None,
        "irr_acres_max": max(irr_acres) if irr_acres else None,
        "irr_acres_sum": float(np.sum(irr_acres)) if irr_acres else None,
        "acre_limit_max": max(acre_lim) if acre_lim else None,
    }


agg = wm["RightIDs"].map(pod_attrs).apply(pd.Series)
wm = pd.concat([wm, agg], axis=1)

# ---- melt to long: one row per (POD, year) with reported volume ----
long_rows = []
for _, row in wm.iterrows():
    for y in YEARS:
        v = row.get(f"Volume{y}")
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == 0:
            continue
        long_rows.append(
            {
                "wmis_number": row["WMISNumber"],
                "wm_metal_tag": row.get("WmMetalTag"),
                "sd_metal_tag": row.get("SdMetalTag"),
                "rprt_dstrct": row.get("RprtDstrct"),
                "wtr_dist_num": row.get("WtrDistNum"),
                "option_type": row.get("OptionType"),
                "year": y,
                "volume_af": float(v),
                "method": row.get(f"Option{y}"),
                "qual": row.get(f"Qual{y}"),
                "right_ids": row.get("RightIDs"),
                "wr_numbers": row.get("wr_numbers"),
                "water_uses": row.get("water_uses"),
                "has_irrigation": bool(row.get("has_irrigation")),
                "irr_only": bool(row.get("irr_only")),
                "n_rights": row.get("n_rights"),
                "n_irr_rights": row.get("n_irr_rights"),
                "irr_acres_max": row.get("irr_acres_max"),
                "irr_acres_sum": row.get("irr_acres_sum"),
                "acre_limit_max": row.get("acre_limit_max"),
                "longitude": row["_x"],
                "latitude": row["_y"],
            }
        )
lg = pd.DataFrame(long_rows)

# applied depth (ft, then mm) only where POD is irrigation-only and acreage known
mask = lg["irr_only"] & lg["irr_acres_max"].notna() & (lg["irr_acres_max"] > 0)
lg["applied_depth_ft"] = np.where(mask, lg["volume_af"] / lg["irr_acres_max"], np.nan)
lg["applied_depth_mm"] = lg["applied_depth_ft"] * 304.8

print("long records (POD x year, volume>0):", len(lg))
print("  distinct WMIS PODs:", lg["wmis_number"].nunique())
print("  years:", lg["year"].min(), "-", lg["year"].max())
print("  method dist:\n", lg["method"].value_counts())
print("  irrigation-bearing records:", int(lg["has_irrigation"].sum()))
print("  irr_only w/ acreage (applied depth):", int(mask.sum()))
dd = lg.loc[mask, "applied_depth_ft"]
print(
    f"  applied depth ft: median={dd.median():.2f} IQR={dd.quantile(0.25):.2f}-{dd.quantile(0.75):.2f}"
)

# geometry check: no null coords expected
assert lg["longitude"].notna().all() and lg["latitude"].notna().all(), "null coords!"

gdf = gpd.GeoDataFrame(
    lg, geometry=gpd.points_from_xy(lg["longitude"], lg["latitude"]), crs="EPSG:4326"
)
gdf.to_file("idwr_wmis_applied_water.fgb", driver="FlatGeobuf")
print("\nwrote idwr_wmis_applied_water.fgb:", len(gdf), "features | crs", gdf.crs)
print("bbox:", [round(x, 4) for x in gdf.total_bounds.tolist()])

# also keep the wide raw table (all PODs incl. no-volume) as parquet
wm_out = wm.drop(columns=["_x", "_y"]).copy()
wm_out.to_parquet("wmis_sites_wide.parquet", index=False)
print("wrote wmis_sites_wide.parquet:", len(wm_out), "PODs")
