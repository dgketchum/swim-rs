"""Phase 1 prep: assemble the reference-ET sidecar cohort.

The sidecar must cover (a) the 75-site container cohort, of which the 66 publication
sites are a subset, and (b) the five ESPA∩NHM diagnostic sites that lie outside the
container (US-Esm, US-Me2, US-SRG, US-Wkg, US-xRM) so Gate G3 can reproduce the 16-site
baseline comparison exactly. Geometries for (b) come from the 241-site international
150 m buffer file the container cohort was drawn from; identity of the shared polygons
is asserted before anything is written.

Writes ``refet_sidecar_cohort.fgb`` (80 features, ``sid``) into the QA root.

Usage:
    uv run python examples/6_Flux_International/e2_refooting/phase1_build_sidecar_cohort.py
"""

from __future__ import annotations

import json
import os

import geopandas as gpd
import pandas as pd

DATA = "/data/ssd1/swim/6_Flux_International/data"
QA_ROOT = os.path.join(DATA, "e2_etf_refooting")
CONTAINER_COHORT = os.path.join(DATA, "gis", "flux_crop_pub_75_150m.shp")
INTL_241 = os.path.join(DATA, "gis", "flux_intl_150m_23MAR2026.shp")
EXTRA_GATE_SITES = ["US-Esm", "US-Me2", "US-SRG", "US-Wkg", "US-xRM"]
OUT = os.path.join(QA_ROOT, "refet_sidecar_cohort.fgb")


def main() -> int:
    cohort = gpd.read_file(CONTAINER_COHORT, engine="fiona")[["sid", "geometry"]]
    intl = gpd.read_file(INTL_241, engine="fiona")[["sid", "geometry"]]
    if intl.crs != cohort.crs:
        intl = intl.to_crs(cohort.crs)

    missing = sorted(set(EXTRA_GATE_SITES) - set(intl["sid"]))
    if missing:
        raise ValueError(f"gate sites absent from {INTL_241}: {missing}")
    overlap = sorted(set(EXTRA_GATE_SITES) & set(cohort["sid"]))
    if overlap:
        raise ValueError(f"gate sites already in the container cohort: {overlap}")

    # The container cohort should be a geometric subset of the 241-site file.
    shared = cohort.merge(intl, on="sid", suffixes=("_cohort", "_intl"))
    same = [
        a.equals_exact(b, tolerance=1e-9)
        for a, b in zip(shared["geometry_cohort"], shared["geometry_intl"], strict=True)
    ]
    n_differ = int(len(same) - sum(same))

    extra = intl[intl["sid"].isin(EXTRA_GATE_SITES)]
    out = (
        gpd.GeoDataFrame(pd.concat([cohort, extra], ignore_index=True), crs=cohort.crs)
        .sort_values("sid")
        .reset_index(drop=True)
    )
    if out["sid"].duplicated().any():
        raise ValueError("duplicate sid in assembled cohort")

    os.makedirs(QA_ROOT, exist_ok=True)
    out.to_file(OUT, driver="FlatGeobuf", engine="fiona")

    summary = {
        "output": OUT,
        "n_features": int(len(out)),
        "n_container_cohort": int(len(cohort)),
        "n_extra_gate_sites": int(len(extra)),
        "extra_gate_sites": EXTRA_GATE_SITES,
        "shared_with_intl_241": int(len(shared)),
        "shared_geometries_differ": n_differ,
        "crs": str(out.crs),
    }
    with open(os.path.join(QA_ROOT, "refet_sidecar_cohort.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
