"""Query WaterRightPous for all RightIDs referenced by WMIS -> acreage/water-use lookup."""

import json
import re
import time

import pandas as pd
import requests

URL = "https://gis.idwr.idaho.gov/hosting/rest/services/Allocation/WaterRightPous/FeatureServer/0/query"
H = {"User-Agent": "Mozilla/5.0 Chrome/120 Safari/537.36"}

d = json.load(open("wmis_raw.json"))
rids = set()
for f in d["features"]:
    r = f["attributes"].get("RightIDs")
    if r:
        for tok in re.split(r"[,\s]+", str(r).strip()):
            if tok.isdigit():
                rids.add(int(tok))
rids = sorted(rids)
print("unique RightIDs to query:", len(rids))

sess = requests.Session()
rows = []
CH = 400
for i in range(0, len(rids), CH):
    chunk = rids[i : i + CH]
    where = "RightID IN (" + ",".join(str(x) for x in chunk) + ")"
    for attempt in range(4):
        try:
            rr = sess.get(
                URL,
                params={
                    "where": where,
                    "outFields": "RightID,WaterRightNumber,BasinNumber,WaterUse,WaterUseCode,TotalAcres,AcreLimit,Owner,Status,PriorityDate,Source",
                    "returnGeometry": "false",
                    "f": "json",
                },
                headers=H,
                timeout=120,
            )
            j = rr.json()
            if "features" in j:
                break
            raise ValueError(j.get("error", j))
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    for f in j["features"]:
        rows.append(f["attributes"])
    print(
        f"  batch {i // CH + 1}/{(len(rids) + CH - 1) // CH}: +{len(j['features'])} (cum {len(rows)})"
    )

pou = pd.DataFrame(rows)
print("\nPOU rows:", len(pou))
print("distinct RightID returned:", pou["RightID"].nunique())
print("WaterUse distribution:\n", pou["WaterUse"].value_counts().head(15))
# rows per RightID cardinality
print("rows-per-RightID max:", pou.groupby("RightID").size().max())
pou.to_parquet("pou_lookup.parquet", index=False)
print("wrote pou_lookup.parquet")
