"""Pull annual diversion (pumping) records for metered Division 3 wells.

For SLV/Rio Grande wells, HydroBase divrecyear holds the administered ANNUAL
pumped volume (measUnits=ACFT). Rio Grande measurement rules require metering,
so these annual values reflect metered pumping (year span ~2009-present).

Batched by comma-separated WDIDs (10 per request is reliable).
"""

import json
import time

import pandas as pd
import requests

BASE = "https://dwr.state.co.us/Rest/GET/api/v2"
H = {"User-Agent": "Mozilla/5.0 swim-rs research"}
BATCH = 10


def get_json(url, params, tries=6):
    for a in range(tries):
        try:
            r = requests.get(url, params=params, headers=H, timeout=180)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                return r.json()
            if r.status_code == 404:
                return {"ResultList": []}
        except Exception:
            pass
        time.sleep(2 * (a + 1))
    raise RuntimeError(f"failed: {params}")


def main():
    wdids = json.load(open("metered_well_wdids.json"))
    print(f"metered wells to query: {len(wdids)}")
    rows = []
    n_batches = (len(wdids) + BATCH - 1) // BATCH
    for i in range(0, len(wdids), BATCH):
        chunk = wdids[i : i + BATCH]
        j = get_json(
            BASE + "/structures/divrec/divrecyear",
            {"format": "json", "pageSize": 50000, "wdid": ",".join(chunk)},
        )
        res = j.get("ResultList") or []
        rows += res
        if (i // BATCH) % 25 == 0:
            print(f"  batch {i // BATCH + 1}/{n_batches}: cum rows {len(rows)}")
    df = pd.DataFrame(rows)
    df.to_parquet("div3_well_divrecyear.parquet", index=False)
    print(f"\ntotal annual pumping records: {len(df)}")
    if len(df):
        print("distinct wells with records:", df["wdid"].nunique())
        print("year span:", df["dataMeasDate"].min(), "-", df["dataMeasDate"].max())
        print("units:", df["measUnits"].value_counts().to_dict())


if __name__ == "__main__":
    main()
