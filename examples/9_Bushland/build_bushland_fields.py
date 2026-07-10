"""Build the Example 9 Bushland weighing-lysimeter cohort (WP-B2).

Turns the four USDA-ARS Bushland (TX) weighing-lysimeter fields into a SWIM
fields shapefile plus the observed diagnostic tables the prescribed-irrigation
ET check consumes. Each lysimeter (NE/SE/NW/SW) is a 3x3 m precision monolith at
the centre of a ~4.4 ha uniformly managed square field; the field polygon (from
data/bushland/bushland_lysimeter_fields.fgb) is Landsat-resolvable (~49 30 m
pixels), so getInfo NDVI/ETf/refET over the field interior is meaningful. The
instrumented footprint itself is sub-pixel -- flag this scale mismatch when
reading the results.

Products written here:
  1. bushland_fields.shp/.fgb  -- 4 physical fields, feature id ``site_id``
     (NE/SE/NW/SW). Station coords are carried as ``site_lat``/``site_lon`` (NOT
     lowercase lat/lon): assign_gridmet_ids adds uppercase LAT/LON/ELEV and the
     DBF driver is case-insensitive, so lowercase lat/lon would collide.
  2. gridmet_centroids.shp -- copied from the shared CONUS grid so the example is
     self-contained (assign_gridmet_ids keys the met store to these GFIDs).
  3. bushland_sites.csv -- site_id, lysimeter, lat, lon, state, area_ha.
  4. bushland_prescribed_irr.parquet -- date x {NE,SE,NW,SW} daily metered
     irrigation (mm/day) for the OpenET-era crop-years only (maize 2016/2018,
     soybean 2019); every day inside a prescribed crop-year is filled (0 where no
     event) so the internal scheduler is fully OFF for that field-year, and days
     outside the crop-years are absent (-> NaN = keep the scheduler). This is a
     diagnostic physics-bypass input for prescribed_et.py ONLY -- never a
     production or calibration input.
  5. bushland_lysimeter_et.parquet -- date x {NE,SE,NW,SW} daily gold-standard
     weighing-lysimeter ET (mm/day), the WP-B2 validation target. Validation
     only; never a model input.
  6. bushland_field_years.csv -- per (site_id, year) crop + season irr/ET totals.

    uv run python examples/9_Bushland/build_bushland_fields.py
"""

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path("/home/dgketchum/code/swim-rs")
EXAMPLE_DIR = REPO / "examples" / "9_Bushland"
FIELDS_FGB = REPO / "data" / "bushland" / "bushland_lysimeter_fields.fgb"
DAILY_PARQUET = Path("/data/ssd1/swim/soil_moisture/bushland/bushland_daily_irrigation_et.parquet")
# Shared CONUS GridMET centroid grid (EPSG:5071); copied in so we do not depend
# on another example's gis dir at run time.
CONUS_GRIDMET = Path("/data/ssd1/swim/conus_demo/data/gis/gridmet_centroids.shp")
PROJECT_GIS = Path("/data/ssd1/swim/9_Bushland/data/gis")

EQUAL_AREA = "EPSG:5071"  # NAD83 CONUS Albers -- matches Example 5/8 fields

# OpenET-era Bushland crop-years for the prescribed-irrigation ET check.
CROP_YEARS = {
    ("maize", 2016),
    ("maize", 2018),
    ("soybean", 2019),
}
CROP_YEAR_SET = {y for _, y in CROP_YEARS}
LYSIMETERS = ["NE", "SE", "NW", "SW"]


def build_fields() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Dedup the 54-row field-year fgb to the 4 physical lysimeter fields."""
    g = gpd.read_file(FIELDS_FGB, engine="fiona")
    # one unique polygon per lysimeter (verified: 1 geometry each)
    first = g.sort_values("year").drop_duplicates("lysimeter", keep="first").copy()
    first = first[first["lysimeter"].isin(LYSIMETERS)].sort_values("lysimeter")

    proj = first.to_crs(EQUAL_AREA)
    area_ha = (proj.geometry.area / 1e4).round(2).values

    fields = gpd.GeoDataFrame(
        {
            "site_id": first["lysimeter"].values,
            "lysimeter": first["lysimeter"].values,
            "lc_class": "Croplands",
            "state": "TX",
            "source": "Bushland_ARS",
            # coords under non-colliding names (see module docstring)
            "site_lat": first["lat"].values,
            "site_lon": first["lon"].values,
            "area_ha": area_ha,
            "geometry": proj.geometry.values,
        },
        crs=EQUAL_AREA,
    )

    sites = pd.DataFrame(
        {
            "site_id": fields["site_id"].values,
            "lysimeter": fields["lysimeter"].values,
            "lat": fields["site_lat"].values,
            "lon": fields["site_lon"].values,
            "state": "TX",
            "area_ha": area_ha,
        }
    )
    return fields, sites


def build_observed_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prescribed-irrigation, lysimeter-ET, and per-field-year summary tables."""
    df = pd.read_parquet(DAILY_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["lysimeter"].isin(LYSIMETERS)].copy()

    # OpenET-era crop-years only.
    era = df[df.apply(lambda r: (r["crop"], int(r["year"])) in CROP_YEARS, axis=1)].copy()

    # --- prescribed irrigation: fill every day of each prescribed crop-year ---
    # (0 where no metered event) so the scheduler is fully OFF for that field-year.
    # Days outside any prescribed crop-year stay NaN = keep the scheduler.
    prescribed_index = pd.DatetimeIndex(
        sorted({d for y in CROP_YEAR_SET for d in pd.date_range(f"{y}-01-01", f"{y}-12-31")})
    )
    prescribed = pd.DataFrame(index=prescribed_index, columns=LYSIMETERS, dtype=float)
    for (lys, year), grp in era.groupby(["lysimeter", "year"]):
        full = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        s = grp.set_index("date")["irrigation_mm"].reindex(full).fillna(0.0)
        prescribed.loc[s.index, lys] = s.values
    prescribed = prescribed[LYSIMETERS]
    prescribed.index.name = "date"

    # --- lysimeter ET target: only observed days (no fill) ---
    et_wide = era.pivot_table(index="date", columns="lysimeter", values="et_mm", aggfunc="first")
    et_wide = et_wide.reindex(columns=LYSIMETERS)
    et_wide.index.name = "date"

    # --- per-field-year summary (metered irr, lysimeter ET, precip) ---
    summ = (
        era.groupby(["lysimeter", "year", "crop"])
        .agg(
            n_days=("date", "size"),
            irr_mm=("irrigation_mm", "sum"),
            et_mm=("et_mm", "sum"),
            precip_mm=("precip_mm", "sum"),
            date_start=("date", "min"),
            date_end=("date", "max"),
        )
        .reset_index()
        .rename(columns={"lysimeter": "site_id"})
        .sort_values(["site_id", "year"])
    )
    return prescribed, et_wide, summ


def _write_qc(fields, sites, summ, path: Path) -> None:
    lines = [
        "# Example 9 Bushland lysimeter cohort (WP-B2)\n",
        f"Cohort: {len(fields)} USDA-ARS weighing-lysimeter fields "
        f"(NE/SE/NW/SW), Bushland TX; field polygons {sites.area_ha.min():.1f}-"
        f"{sites.area_ha.max():.1f} ha (Landsat-resolvable).\n",
        "The instrumented lysimeter is a 3x3 m monolith at the field centre "
        "(sub-pixel) -- getInfo averages over the ~4.4 ha managed field, so ETf/"
        "NDVI are field-scale while the ET/irrigation truth is the point monolith. "
        "Flag this scale mismatch when interpreting results.\n",
        f"OpenET-era crop-years validated: {sorted(CROP_YEARS)}.\n\n",
        "Metered irrigation and weighing-lysimeter ET are used ONLY at the "
        "prescribed-irrigation comparison (prescribed_et.py): irrigation as a "
        "diagnostic physics-bypass override, ET as the validation target. Neither "
        "is ever a model parameter or a calibration input.\n\n",
        "## Per field-year (season totals, mm)\n",
        summ.to_string(index=False),
        "\n",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    fields, sites = build_fields()
    prescribed, et_wide, summ = build_observed_tables()

    PROJECT_GIS.mkdir(parents=True, exist_ok=True)
    (EXAMPLE_DIR / "data").mkdir(parents=True, exist_ok=True)
    (EXAMPLE_DIR / "notes").mkdir(parents=True, exist_ok=True)

    shp = PROJECT_GIS / "bushland_fields.shp"
    fgb = PROJECT_GIS / "bushland_fields.fgb"
    fields.to_file(shp, engine="fiona")
    fields.to_file(fgb, driver="FlatGeobuf", engine="fiona")

    # copy the shared CONUS GridMET centroid grid (all sidecars) into the project
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        src = CONUS_GRIDMET.with_suffix(ext)
        if src.exists():
            shutil.copy(src, PROJECT_GIS / f"gridmet_centroids{ext}")

    data_dir = EXAMPLE_DIR / "data"
    sites.to_csv(data_dir / "bushland_sites.csv", index=False)
    prescribed.to_parquet(data_dir / "bushland_prescribed_irr.parquet")
    et_wide.to_parquet(data_dir / "bushland_lysimeter_et.parquet")
    summ.to_csv(data_dir / "bushland_field_years.csv", index=False)
    _write_qc(fields, sites, summ, EXAMPLE_DIR / "notes" / "selection_qc.md")

    print(f"wrote {shp}  ({len(fields)} fields, EPSG:5071)")
    print(f"wrote {fgb}")
    print(f"copied CONUS gridmet_centroids -> {PROJECT_GIS / 'gridmet_centroids.shp'}")
    print(f"wrote {data_dir / 'bushland_sites.csv'}")
    print(
        f"wrote {data_dir / 'bushland_prescribed_irr.parquet'}  "
        f"({prescribed.notna().any(axis=1).sum()} prescribed days)"
    )
    print(
        f"wrote {data_dir / 'bushland_lysimeter_et.parquet'}  "
        f"({et_wide.notna().any(axis=1).sum()} ET days)"
    )
    print(f"wrote {data_dir / 'bushland_field_years.csv'}")
    print("\n=== per field-year season totals ===")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
