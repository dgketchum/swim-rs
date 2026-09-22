"""Download all WMIS POD features (metered annual diversion volumes) from IDWR FeatureServer."""

import json
import time

import requests

URL = "https://gis.idwr.idaho.gov/hosting/rest/services/compliance/wmis/FeatureServer/0/query"
HEAD = {"User-Agent": "Mozilla/5.0 Chrome/120 Safari/537.36"}

sess = requests.Session()
# get all objectids to paginate deterministically
r = sess.get(
    URL, params={"where": "1=1", "returnIdsOnly": "true", "f": "json"}, headers=HEAD, timeout=90
)
oids = r.json()["objectIds"]
print(f"total OIDs: {len(oids)}")

feats = []
CH = 500
for i in range(0, len(oids), CH):
    chunk = oids[i : i + CH]
    where = f"OBJECTID>={chunk[0]} AND OBJECTID<={chunk[-1]}"
    for attempt in range(4):
        try:
            rr = sess.get(
                URL,
                params={
                    "where": where,
                    "outFields": "*",
                    "returnGeometry": "true",
                    "outSR": "4326",
                    "f": "json",
                },
                headers=HEAD,
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
    feats.extend(j["features"])
    print(f"  page {i // CH + 1}: +{len(j['features'])} (cum {len(feats)})")

json.dump({"features": feats}, open("wmis_raw.json", "w"))
print(f"wrote wmis_raw.json with {len(feats)} features")
