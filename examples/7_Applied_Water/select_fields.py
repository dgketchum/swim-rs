"""Select the Example 7 applied-water validation cohort.

50 SLV + 50 ESPA metered irrigated fields, plus rainfed negative controls, chosen
to be relatively large, "normal" (single common crop, center-pivot/sprinkler,
regular shape) with high-confidence geometry. Writes a project shapefile SWIM
extracts on, and a *withheld* metered-truth table used only at scoring.

Applied water is validation-only and never a model input.

Outputs (project gis dir + a versioned copy in this example dir):
    applied_water_fields.shp   feature id `site_id`, `basin`, `crop`, `acres`, `state`
    metered_truth.csv          site_id, year, metered_depth_mm, metered_volume_af,
                               acres, method, source   (ground truth, withheld)
    selection_qc.md            counts + size/crop/depth distributions + QC checks

    uv run python examples/7_Applied_Water/select_fields.py
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path("/home/dgketchum/code/swim-rs")
SLV_DIR = REPO / "data" / "co_slv_wells"
WMIS_DIR = REPO / "data" / "idwr_wmis"
ESPA_FIELDS = Path(
    "/nas/irrmapper/raw_field_polygons/ID/ESPA/"
    "2015_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer/"
    "2015_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer.shp"
)
POU_POLY = WMIS_DIR / "pou_polygons.fgb"
# IrrMapper (2000-2024) irrigated-fraction cache for the ESPA control candidate
# pool, keyed by `fid2015`. Produced by espa_control_irrmapper.py. Controls are
# gated STRICTLY never-irrigated (max annual irr fraction == 0) against this.
ESPA_CONTROL_IRR = WMIS_DIR / "espa_control_irrmapper.csv"

PROJECT_GIS = Path("/data/ssd1/swim/7_Applied_Water/data/gis")
EXAMPLE_DIR = REPO / "examples" / "7_Applied_Water"

AEA = "EPSG:5070"  # CONUS Albers equal area for area/compactness

# "normal" crops (pivot row/forage), single dominant crop
SLV_CROPS = {"ALFALFA", "NEW_ALFALFA", "POTATOES", "BARLEY", "SMALL_GRAINS", "WHEAT_SPRING"}
DEPTH_MIN, DEPTH_MAX = 200.0, 1200.0  # mm, plausible arid-basin applied band
ACRE_MIN = 40.0
# Controls must be field-sized, comparable to the metered irrigated cohort
# (which tops out near 213 ac) — NOT whole rangeland tracts. A tract-scale
# polygon both breaks the point of a per-field negative control and, at 30 m
# over a multi-band annual stack, exceeds Earth Engine's reduceRegion pixel cap.
CONTROL_ACRE_MAX = 220.0
MIN_YEARS = 5
AREA_ACRES_TOL = 0.15  # polygon area within +-15% of reported acres (coarse prefilter)
# The metered depth is volume/service-area; SWIM integrates over the polygon. For the
# comparison to be apples-to-apples the implied service acreage (volume/depth) must match
# the polygon area. Select only fields with high service-to-polygon agreement.
SVC_POLY_TOL = 0.05  # |volume/depth acreage - polygon acreage| / polygon acreage
AREA_CV_MAX = 0.15  # inter-year geometry stability (high confidence)
PP_MIN = 0.60  # Polsby-Popper compactness floor
N_PER_BASIN = 50
N_CONTROLS = 10


def _aea_area_acres(geom_series: gpd.GeoSeries) -> pd.Series:
    """Equal-area polygon area in acres."""
    return geom_series.to_crs(AEA).area / 4046.8564224


def _polsby_popper(geom_series: gpd.GeoSeries) -> pd.Series:
    g = geom_series.to_crs(AEA)
    return (4 * np.pi * g.area) / (g.length**2)


def _pick_near_median(cand, size_col, label):
    """Select the validation cohort for a basin: keep only fields whose metered
    service acreage agrees with the polygon (``svc_agree`` <= SVC_POLY_TOL), then
    take the N_PER_BASIN closest to the eligible pool's median size (tie-break:
    more metered years). Prefers typical-size fields with clean attribution over
    the largest/longest-record fields the prior sort favored.
    """
    elig = cand[cand.svc_agree <= SVC_POLY_TOL]
    if "n_years" in elig:
        elig = elig[elig.n_years >= MIN_YEARS]
    if "area_cv" in elig:
        elig = elig[elig.area_cv <= AREA_CV_MAX]
    target = float(elig[size_col].median())
    by = ["_d", "n_years"] if "n_years" in elig else ["_d"]
    asc = [True, False] if "n_years" in elig else [True]
    kept = elig.assign(_d=(elig[size_col] - target).abs()).sort_values(by, ascending=asc)
    kept = kept.head(N_PER_BASIN).drop(columns="_d")
    print(
        f"  {label}: {len(elig)}/{len(cand)} fields pass svc_agree<={SVC_POLY_TOL:.2f}; "
        f"target {target:.0f} ac, kept {len(kept)} "
        f"({kept[size_col].min():.0f}-{kept[size_col].max():.0f} ac, "
        f"svc_agree med {kept.svc_agree.median():.3f})"
    )
    return kept


# --------------------------------------------------------------------------- SLV
def select_slv() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    links = pd.read_parquet(SLV_DIR / "co_slv_parcel_well_links.parquet")
    depth = pd.read_parquet(SLV_DIR / "co_slv_well_year_applied_depth.parquet")
    parcels = gpd.read_file(SLV_DIR / "co_slv_irrigated_parcels.fgb", engine="fiona")

    # 1:1 well<->parcel in a year: parcel fed by exactly one well, and that well
    # feeds exactly one (single-well) parcel that year -> volume fully attributable.
    one_well_parcel = links[links.n_gw_wells_on_parcel == 1].copy()
    per_well = one_well_parcel.groupby(["cal_year", "gw_wdid"])["parcel_id"].nunique()
    clean_wells = per_well[per_well == 1].reset_index()[["cal_year", "gw_wdid"]]
    lk = one_well_parcel.merge(clean_wells, on=["cal_year", "gw_wdid"])

    # metered groundwater volume + attributable depth
    d = depth[
        (depth.year >= 2011)
        & (depth.has_meter)
        & (depth.pumped_af > 0)
        & (depth.frac_acres_G >= 0.90)
        & (depth.applied_depth_mm_alloc.between(DEPTH_MIN, DEPTH_MAX))
    ][["year", "wdid", "pumped_af", "applied_depth_mm_alloc"]]
    lk = lk.merge(d, left_on=["cal_year", "gw_wdid"], right_on=["year", "wdid"])

    # parcel attrs + geometry; normal pivot crop, single-crop parcel
    pa = parcels[
        (parcels.irrig_type == "SPRINKLER")
        & (parcels.crop_type.isin(SLV_CROPS))
        & (parcels.acres >= ACRE_MIN)
    ][["cal_year", "parcel_id", "master_id", "crop_type", "acres", "geometry"]]
    fy = pa.merge(lk, on=["cal_year", "parcel_id"], how="inner")
    fy = gpd.GeoDataFrame(fy, geometry="geometry", crs=parcels.crs)

    # one field = master_id; one parcel per (master_id, year)
    fy = fy.drop_duplicates(subset=["master_id", "cal_year"])
    fy["poly_acres"] = _aea_area_acres(fy.geometry).values
    fy["pp"] = _polsby_popper(fy.geometry).values
    fy = fy[(fy.pp >= PP_MIN) & ((fy.poly_acres - fy.acres).abs() / fy.acres <= AREA_ACRES_TOL)]
    # service acreage implied by the metered volume and allocated depth
    fy["alloc_acres"] = fy.pumped_af / (fy.applied_depth_mm_alloc / 304.8)

    # field-level: >=MIN_YEARS qualifying years and stable geometry
    grp = fy.groupby("master_id")
    stats = grp.agg(
        n_years=("cal_year", "nunique"),
        acres=("poly_acres", "median"),
        alloc_acres=("alloc_acres", "median"),
        area_cv=("poly_acres", lambda s: s.std() / s.mean() if len(s) > 1 else 0.0),
        crop=("crop_type", lambda s: s.mode().iat[0]),
    )
    stats["svc_agree"] = (stats.alloc_acres - stats.acres).abs() / stats.acres
    keep = _pick_near_median(stats, "acres", "SLV")

    fields, truth = [], []
    for i, (mid, row) in enumerate(keep.iterrows()):
        sub = fy[fy.master_id == mid].sort_values("cal_year")
        rep = sub.iloc[len(sub) // 2]  # representative (median-year) geometry
        sid = f"SLV_{i:03d}"
        fields.append(
            {
                "site_id": sid,
                "basin": "SLV",
                "crop": row.crop,
                "acres": round(float(row.acres), 1),
                "state": "CO",
                "src_id": str(int(mid)),
                "geometry": rep.geometry,
            }
        )
        for _, r in sub.iterrows():
            truth.append(
                {
                    "site_id": sid,
                    "year": int(r.cal_year),
                    "metered_depth_mm": round(float(r.applied_depth_mm_alloc), 1),
                    "metered_volume_af": round(float(r.pumped_af), 1),
                    "acres": round(float(r.poly_acres), 1),
                    "method": "metered_gw",
                    "source": "CO_SLV_RGDSS",
                }
            )
    gdf = gpd.GeoDataFrame(fields, geometry="geometry", crs=parcels.crs)
    return gdf, pd.DataFrame(truth)


# -------------------------------------------------------------------------- ESPA
def select_espa() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    if not POU_POLY.exists():
        raise SystemExit(f"{POU_POLY} missing — run data/idwr_wmis/download_pou_geom.py first.")
    pods = gpd.read_file(WMIS_DIR / "idwr_wmis_applied_water.fgb", engine="fiona")
    pou = gpd.read_file(POU_POLY, engine="fiona").to_crs("EPSG:4326")
    fields2015 = gpd.read_file(ESPA_FIELDS, engine="fiona").to_crs("EPSG:4326")
    irr_fields = fields2015[fields2015.Status_201 == "irrigated"].copy()
    irr_fields["geometry"] = irr_fields.geometry.make_valid()
    irr_fields = irr_fields[irr_fields.geom_type.isin(["Polygon", "MultiPolygon"])]
    irr_fields["fid2015"] = irr_fields.index
    irr_fields["f_acres"] = _aea_area_acres(irr_fields.geometry).values
    irr_fields["f_pp"] = _polsby_popper(irr_fields.geometry).values
    sindex = irr_fields.sindex

    # FM-metered, irrigation-only PODs with a plausible attributable depth
    pods = pods[
        (pods.method == "FM")
        & (pods.irr_only)
        & (pods.irr_acres_max >= ACRE_MIN)
        & (pods.applied_depth_mm.between(DEPTH_MIN, DEPTH_MAX))
    ].copy()
    # service acreage implied by the metered volume and reported depth
    pods["alloc_acres"] = pods.volume_af / (pods.applied_depth_mm / 304.8)

    # field identity = wmis_number across FM years; require >=MIN_YEARS
    yc = pods.groupby("wmis_number")["year"].nunique()
    keep_pods = yc[yc >= MIN_YEARS].index
    pods = pods[pods.wmis_number.isin(keep_pods)]

    # link each POD -> its rights -> POU polygon(s) -> covered digitized field
    pod_geom = pods.drop_duplicates("wmis_number")[
        ["wmis_number", "right_ids", "irr_acres_max", "geometry"]
    ].copy()

    def _rids(s):
        return [int(t) for t in str(s).replace(";", ",").split(",") if t.strip().isdigit()]

    rows = []
    pou_by_rid = pou.set_index("RightID")
    for _, pr in pod_geom.iterrows():
        rids = [r for r in _rids(pr.right_ids) if r in pou_by_rid.index]
        if not rids:
            continue
        pous = pou.loc[pou.RightID.isin(rids)]
        if pous.empty:
            continue
        place = pous.geometry.make_valid().union_all()
        # digitized irrigated fields overlapping the POU (spatial-index prefilter)
        cand = irr_fields.iloc[sindex.query(place, predicate="intersects")].copy()
        if cand.empty:
            continue
        # area of overlap with the POU, keep the dominant single field
        cand["ov"] = cand.geometry.intersection(place).to_crs(AEA).area / 4046.8564224
        cand = cand.sort_values("ov", ascending=False)
        top = cand.iloc[0]
        # high confidence: dominant field covers most of the POU, one clear field,
        # compact, and digitized acres agree with the WMIS right acreage
        pou_acres = (
            float(gpd.GeoSeries([place], crs="EPSG:4326").to_crs(AEA).area.iloc[0]) / 4046.8564224
        )
        dom_frac = top.ov / pou_acres if pou_acres > 0 else 0.0
        if (
            top.f_pp >= PP_MIN
            and dom_frac >= 0.60
            and abs(top.f_acres - pr.irr_acres_max) / pr.irr_acres_max <= AREA_ACRES_TOL
        ):
            rows.append(
                {
                    "wmis_number": pr.wmis_number,
                    "fid2015": top.fid2015,
                    "f_acres": top.f_acres,
                    "geometry": top.geometry,
                }
            )

    matched = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    matched = matched.drop_duplicates("fid2015")  # one geometry per digitized field
    ycount = pods.groupby("wmis_number")["year"].nunique()
    alloc = pods.groupby("wmis_number")["alloc_acres"].median()
    matched["n_years"] = matched.wmis_number.map(ycount)
    matched["alloc_acres"] = matched.wmis_number.map(alloc)
    matched["svc_agree"] = (matched.alloc_acres - matched.f_acres).abs() / matched.f_acres
    matched = _pick_near_median(matched, "f_acres", "ESPA")

    fields, truth = [], []
    for i, (_, m) in enumerate(matched.iterrows()):
        sid = f"ESPA_{i:03d}"
        fields.append(
            {
                "site_id": sid,
                "basin": "ESPA",
                "crop": "UNKNOWN",  # digitized set carries no crop; NDVI drives Kcb
                "acres": round(float(m.f_acres), 1),
                "state": "ID",
                "src_id": str(int(m.wmis_number)),
                "geometry": m.geometry,
            }
        )
        pod_years = pods[pods.wmis_number == m.wmis_number]
        for _, r in pod_years.iterrows():
            truth.append(
                {
                    "site_id": sid,
                    "year": int(r.year),
                    "metered_depth_mm": round(float(r.applied_depth_mm), 1),
                    "metered_volume_af": round(float(r.volume_af), 1),
                    "acres": round(float(m.f_acres), 1),
                    "method": "FM",
                    "source": "ID_WMIS",
                }
            )
    return gpd.GeoDataFrame(fields, geometry="geometry", crs="EPSG:4326"), pd.DataFrame(truth)


def espa_control_candidates() -> gpd.GeoDataFrame:
    """Field-sized, compact ESPA 2015 non-irrigated polygons — the pool from which
    rainfed controls are drawn. `fid2015` (the source shapefile index) is the stable
    key both the IrrMapper cache (espa_control_irrmapper.py) and select_espa_controls
    join on, so the candidate set must be built identically in both places.

    A control must be a single non-irrigated *field* comparable to the metered
    irrigated cohort — not a rangeland tract (the largest "non-irrigated" polygons
    here are 3.5k-52k ac) — so it is bounded to ACRE_MIN..CONTROL_ACRE_MAX and PP_MIN.
    """
    fields2015 = gpd.read_file(ESPA_FIELDS, engine="fiona").to_crs("EPSG:4326")
    nonirr = fields2015[fields2015.Status_201 == "non-irrigated"].copy()
    nonirr["fid2015"] = nonirr.index
    nonirr["f_acres"] = _aea_area_acres(nonirr.geometry).values
    nonirr["f_pp"] = _polsby_popper(nonirr.geometry).values
    return nonirr[
        (nonirr.f_acres >= ACRE_MIN)
        & (nonirr.f_acres <= CONTROL_ACRE_MAX)
        & (nonirr.f_pp >= PP_MIN)
    ]


def select_espa_controls(target_acres: float) -> gpd.GeoDataFrame:
    """Ground-truth rainfed negative controls, gated on IrrMapper.

    The 2015 non-irrigated flag alone is a single-year snapshot; a genuine rainfed
    control must be rainfed *every* year SWIM simulates. IrrMapper (2000-2024, cached
    by espa_control_irrmapper.py) is the authority: only fields it never classified
    irrigated in any year (`max_irr == 0`) are eligible. Among those, the N_CONTROLS
    nearest the irrigated median (``target_acres``) are kept so controls sit at
    typical field size. Pumping truth exists only for irrigated fields, so controls
    carry no metered volume — IrrMapper is their sole rainfed provenance.
    """
    if not ESPA_CONTROL_IRR.exists():
        raise SystemExit(
            f"{ESPA_CONTROL_IRR} missing — run "
            "examples/7_Applied_Water/espa_control_irrmapper.py first "
            "to build the IrrMapper rainfed cache."
        )
    cand = espa_control_candidates()
    irr = pd.read_csv(ESPA_CONTROL_IRR)
    cand = cand.merge(irr[["fid2015", "mean_irr", "max_irr"]], on="fid2015", how="inner")
    rainfed = cand[cand.max_irr == 0.0]  # strictly never irrigated, 2000-2024
    print(
        f"  ESPA controls: {len(rainfed)}/{len(cand)} candidates strictly never "
        f"irrigated (IrrMapper 2000-2024); target {target_acres:.0f} ac"
    )
    if len(rainfed) < N_CONTROLS:
        raise SystemExit(
            f"only {len(rainfed)} strictly-rainfed controls available, need {N_CONTROLS}"
        )
    rainfed = rainfed.reindex((rainfed.f_acres - target_acres).abs().sort_values().index).head(
        N_CONTROLS
    )
    rows = []
    for i, (_, m) in enumerate(rainfed.iterrows()):
        rows.append(
            {
                "site_id": f"ESPActl_{i:03d}",
                "basin": "ESPA",
                "crop": "RAINFED_CONTROL",
                "acres": round(float(m.f_acres), 1),
                "state": "ID",
                "src_id": str(int(m.fid2015)),
                "geometry": m.geometry,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _write_qc(gdf: gpd.GeoDataFrame, truth: pd.DataFrame, path: Path) -> None:
    lines = ["# Example 7 field-selection QC\n"]
    lines.append(f"Total fields: {len(gdf)}  (by basin: {gdf.basin.value_counts().to_dict()})\n")
    lines.append(f"By crop: {gdf.crop.value_counts().to_dict()}\n")
    lines.append(
        f"Acres — median {gdf.acres.median():.0f}, "
        f"IQR {gdf.acres.quantile(0.25):.0f}–{gdf.acres.quantile(0.75):.0f}, "
        f"min {gdf.acres.min():.0f}\n"
    )
    if not truth.empty:
        t = truth[truth.metered_depth_mm > 0]
        lines.append(f"Metered field-years: {len(t)} across {t.site_id.nunique()} fields\n")
        lines.append(
            f"Metered depth (mm) — median {t.metered_depth_mm.median():.0f}, "
            f"IQR {t.metered_depth_mm.quantile(0.25):.0f}–{t.metered_depth_mm.quantile(0.75):.0f}\n"
        )
        yrs = truth.groupby("site_id").year.nunique()
        lines.append(f"Years per field — median {yrs.median():.0f}, min {yrs.min()}\n")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basins", default="slv,espa", help="comma list: slv,espa")
    args = ap.parse_args()
    basins = [b.strip() for b in args.basins.split(",")]

    gdfs, truths = [], []
    if "slv" in basins:
        g, t = select_slv()
        print(f"SLV: {len(g)} fields, {len(t)} field-years")
        gdfs.append(g)
        truths.append(t)
    if "espa" in basins:
        g, t = select_espa()
        print(f"ESPA: {len(g)} fields, {len(t)} field-years")
        gdfs.append(g)
        truths.append(t)
        target_acres = float(pd.concat([x.acres for x in gdfs]).median())
        ctl = select_espa_controls(target_acres)
        print(
            f"ESPA rainfed controls: {len(ctl)} "
            f"(target {target_acres:.0f} ac, range {ctl.acres.min():.0f}-{ctl.acres.max():.0f})"
        )
        gdfs.append(ctl)
        truths.append(
            pd.DataFrame(
                [
                    {
                        "site_id": s,
                        "year": 0,
                        "metered_depth_mm": 0.0,
                        "metered_volume_af": 0.0,
                        "acres": a,
                        "method": "none",
                        "source": "ESPA_rainfed_control",
                    }
                    for s, a in zip(ctl.site_id, ctl.acres)
                ]
            )
        )

    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
    truth = pd.concat(truths, ignore_index=True)

    PROJECT_GIS.mkdir(parents=True, exist_ok=True)
    shp = PROJECT_GIS / "applied_water_fields.shp"
    gdf.to_file(shp, engine="fiona")
    # FlatGeobuf of the exact geometries SWIM extracts over, for QGIS inspection.
    fgb = PROJECT_GIS / "applied_water_fields.fgb"
    gdf.to_file(fgb, driver="FlatGeobuf", engine="fiona")
    (EXAMPLE_DIR / "data").mkdir(parents=True, exist_ok=True)
    truth.to_csv(EXAMPLE_DIR / "data" / "metered_truth.csv", index=False)
    truth.to_csv(PROJECT_GIS.parent / "metered_truth.csv", index=False)
    _write_qc(gdf, truth, EXAMPLE_DIR / "notes" / "selection_qc.md")
    print(f"\nwrote {shp}")
    print(f"wrote {fgb}")
    print(f"wrote {EXAMPLE_DIR / 'data' / 'metered_truth.csv'}")
    print(f"wrote {EXAMPLE_DIR / 'notes' / 'selection_qc.md'}")


if __name__ == "__main__":
    main()
