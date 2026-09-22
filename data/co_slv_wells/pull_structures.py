"""Pull all Division 3 (Rio Grande / San Luis Valley) WELL structures from the
Colorado DWR / CDSS HydroBase REST API and save georeferenced well points.

CDSS REST base: https://dwr.state.co.us/Rest/GET/api/v2
Structures endpoint returns WDID, coordinates (latdecdeg/longdecdeg, NAD83),
structureType, associatedMeters, water district, designated basin, county, etc.

Wells WITH associatedMeters are the ones that carry annual pumping records
(divrecyear); wells without meters return zero divrec rows.
"""

import json
import time

import pandas as pd
import requests

BASE = "https://dwr.state.co.us/Rest/GET/api/v2"
H = {"User-Agent": "Mozilla/5.0 swim-rs research"}
DIVISION = 3


def get_json(url, params, tries=5):
    for a in range(tries):
        try:
            r = requests.get(url, params=params, headers=H, timeout=180)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                return r.json()
            if r.status_code == 404:  # zero records
                return {"ResultList": []}
        except Exception:
            pass
        time.sleep(2 * (a + 1))
    raise RuntimeError(f"failed: {url} {params}")


def main():
    allres, pi = [], 1
    while True:
        j = get_json(
            BASE + "/structures",
            {"format": "json", "pageSize": 50000, "pageIndex": pi, "division": DIVISION},
        )
        res = j.get("ResultList") or []
        allres += res
        print(f"  page {pi}: +{len(res)} (cum {len(allres)})")
        if len(res) < 50000:
            break
        pi += 1

    df = pd.DataFrame(allres)
    df.to_parquet("div3_structures_all.parquet", index=False)
    wells = df[df["structureType"] == "WELL"].copy()
    wells["has_meter"] = wells["associatedMeters"].notna() & (wells["associatedMeters"] != "")
    print(f"total structures: {len(df)}")
    print(
        f"WELL: {len(wells)} | with coords: {wells['latdecdeg'].notna().sum()} "
        f"| with meter: {wells['has_meter'].sum()}"
    )
    wells.to_parquet("div3_wells.parquet", index=False)
    metered = wells[wells["has_meter"]]["wdid"].tolist()
    json.dump(metered, open("metered_well_wdids.json", "w"))
    print(f"wrote metered_well_wdids.json ({len(metered)} wdids)")


if __name__ == "__main__":
    main()
