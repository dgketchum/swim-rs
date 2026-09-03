import os

import ee
import pandas as pd

from swimrs.data_extraction.ee.ee_utils import as_ee_feature_collection, get_lanid

IRR = "projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp"

# See https://websoilsurvey.nrcs.usda.gov/app/WebSoilSurvey.aspx
# to check soil parameters


STATES = ["AZ", "CA", "CO", "ID", "MT", "NM", "NV", "OR", "UT", "WA", "WY"]

WEST_STATES = "users/dgketchum/boundaries/western_11_union"
EAST_STATES = "users/dgketchum/boundaries/eastern_38_dissolved"


def fc_to_dataframe(result: dict, selectors: list[str]) -> pd.DataFrame:
    """Convert a FeatureCollection getInfo() result to a DataFrame.

    Returns one row per feature with columns ordered per `selectors`;
    selectors absent from feature properties become NaN columns (matching
    the behavior of EE table exports with the same selectors).
    """
    df = pd.DataFrame([f["properties"] for f in result["features"]])
    return df.reindex(columns=selectors)


def _write_local(fc: ee.FeatureCollection, selectors: list[str], out_dir: str, desc: str) -> str:
    """Fetch a reduced FeatureCollection via getInfo() and write {out_dir}/{desc}.csv."""
    if not out_dir:
        raise ValueError('dest="local" requires out_dir')
    os.makedirs(out_dir, exist_ok=True)
    df = fc_to_dataframe(fc.getInfo(), selectors)
    out_path = os.path.join(out_dir, f"{desc}.csv")
    df.to_csv(out_path, index=False)
    print(f"{desc}: wrote {out_path}")
    return out_path


def get_cdl(
    fields: str | ee.FeatureCollection,
    desc: str,
    selector: str = "FID",
    dest: str = "drive",
    bucket: str | None = None,
    drive_folder: str = "swim",
    file_prefix: str = "swim",
    drive_categorize: bool = False,
    out_dir: str | None = None,
    task_desc: str | None = None,
) -> None:
    """Export per-feature CDL crop class mode by year to GCS.

    Parameters
    - fields: ee.FeatureCollection asset path or object.
    - desc: filename prefix for the export.
    - selector: property to include as ID in selectors (default 'FID').
    - task_desc: EE task description (defaults to desc if not provided).

    Side Effects
    - Starts ee.batch table export of yearly modes (2008 through the latest
      published CDL year) to `wudr`.
    """
    task_desc = task_desc or desc
    plots = as_ee_feature_collection(fields, feature_id=selector)
    crops, first = None, True
    # Read available years from the collection (full-CONUS CDL starts 2008)
    cdl_years = sorted(
        int(y)
        for y in ee.ImageCollection("USDA/NASS/CDL").aggregate_array("system:index").getInfo()
        if str(y).isdigit() and int(y) >= 2008
    )

    _selectors = [selector]

    for y in cdl_years:
        image = ee.Image(f"USDA/NASS/CDL/{y}")
        crop = image.select("cropland")
        _name = f"crop_{y}"
        _selectors.append(_name)
        if first:
            crops = crop.rename(_name)
            first = False
        else:
            crops = crops.addBands(crop.rename(_name))

    modes = crops.reduceRegions(collection=plots, reducer=ee.Reducer.mode(), scale=30)

    if dest == "local":
        _write_local(modes, _selectors, out_dir, desc)
        return
    if dest == "bucket":
        if not bucket:
            raise ValueError('CDL export dest="bucket" requires a bucket name/url')
        task = ee.batch.Export.table.toCloudStorage(
            modes,
            description=task_desc,
            bucket=bucket,
            fileNamePrefix=f"{file_prefix}/properties/{desc}",
            fileFormat="CSV",
            selectors=_selectors,
        )
    elif dest == "drive":
        drive_folder_name = f"{drive_folder}_properties" if drive_categorize else drive_folder
        task = ee.batch.Export.table.toDrive(
            collection=modes,
            description=task_desc,
            folder=drive_folder_name,
            fileNamePrefix=f"properties/{desc}",
            fileFormat="CSV",
            selectors=_selectors,
        )
    else:
        raise ValueError('dest must be one of {"drive","bucket"}')

    task.start()
    print(task_desc)


def get_irrigation(
    fields: str | ee.FeatureCollection,
    desc: str,
    debug: bool = False,
    selector: str = "FID",
    select: list[str] | None = None,
    lanid: bool = False,
    dest: str = "drive",
    bucket: str | None = None,
    drive_folder: str = "swim",
    file_prefix: str = "swim",
    drive_categorize: bool = False,
    out_dir: str | None = None,
    start_year: int = 1987,
    end_year: int = 2025,
    task_desc: str | None = None,
) -> None:
    """Export annual irrigation fraction per feature using IrrMapper (and LANID).

    Parameters
    - fields: ee.FeatureCollection asset path or object.
    - desc: filename prefix for the export.
    - debug: bool; if True, prints a sample feature.
    - selector: feature ID property to include.
    - select: optional list[str] of selector values to include.
    - lanid: bool; if True, mosaics LANID east of WEST/EAST boundary for years.
    - start_year, end_year: inclusive year range (IrrMapper covers 1985+;
      LANID coverage is narrower, keep the defaults when lanid=True).
    - task_desc: EE task description (defaults to desc if not provided).

    Side Effects
    - Starts ee.batch table export to `wudr` with mean of yearly `irr_<year>`.
    """
    task_desc = task_desc or desc
    east, west = None, None
    plots = as_ee_feature_collection(fields, feature_id=selector)

    # Optionally filter to a subset of features by ID
    if select is not None:
        plots = plots.filter(ee.Filter.inList(selector, select))

    irr_coll = ee.ImageCollection(IRR)
    if lanid:
        lanid = get_lanid()
        west = ee.FeatureCollection(WEST_STATES)
        east = ee.FeatureCollection(EAST_STATES)

    _selectors = [selector, "LAT", "LON"]
    first = True

    area, irr_img = ee.Image.pixelArea(), None

    for year in range(start_year, end_year + 1):
        irr = (
            irr_coll.filterDate(f"{year}-01-01", f"{year}-12-31").select("classification").mosaic()
        )
        irr = irr.lt(1).rename(f"irr_{year}").int()

        if lanid:
            irr = irr.clip(west)
            lanid_yr = lanid.select(f"irr_{year}").clip(east)
            irr = ee.ImageCollection([irr, lanid_yr]).mosaic()

        _name = f"irr_{year}"
        _selectors.append(_name)

        if first:
            irr_img = irr.rename(_name)
            first = False
        else:
            irr_img = irr_img.addBands(irr.rename(_name))

    means = irr_img.reduceRegions(collection=plots, reducer=ee.Reducer.mean(), scale=30)

    if debug:
        debug = means.filterMetadata("FID", "equals", "US-FPe").getInfo()

    if dest == "local":
        _write_local(means, _selectors, out_dir, desc)
        return
    if dest == "bucket":
        if not bucket:
            raise ValueError('Irrigation export dest="bucket" requires a bucket name/url')
        task = ee.batch.Export.table.toCloudStorage(
            means,
            description=task_desc,
            bucket=bucket,
            fileNamePrefix=f"{file_prefix}/properties/{desc}",
            fileFormat="CSV",
            selectors=_selectors,
        )
    elif dest == "drive":
        drive_folder_name = f"{drive_folder}_properties" if drive_categorize else drive_folder
        task = ee.batch.Export.table.toDrive(
            collection=means,
            description=task_desc,
            folder=drive_folder_name,
            fileNamePrefix=f"properties/{desc}",
            fileFormat="CSV",
            selectors=_selectors,
        )
    else:
        raise ValueError('dest must be one of {"drive","bucket"}')

    task.start()
    print(task_desc)


def get_ssurgo(
    fields: str | ee.FeatureCollection,
    desc: str,
    debug: bool = False,
    selector: str = "FID",
    select: list[str] | None = None,
    dest: str = "drive",
    bucket: str | None = None,
    drive_folder: str = "swim",
    file_prefix: str = "swim",
    drive_categorize: bool = False,
    out_dir: str | None = None,
    task_desc: str | None = None,
) -> None:
    """Export SSURGO-derived soil attributes averaged per feature.

    Parameters
    - fields: ee.FeatureCollection asset path or object.
    - desc: filename prefix for the export.
    - debug: bool; if True, prints a sample feature.
    - selector: feature ID property to include.
    - select: optional list[str] of selector values to include.
    - task_desc: EE task description (defaults to desc if not provided).

    Side Effects
    - Starts ee.batch table export (columns: awc, ksat, clay, sand) to `wudr`.
    """
    task_desc = task_desc or desc
    # OpenET AWC is in cm/cm
    awc_asset = "projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite"
    # OpenET KSAT is in micrometers/sec
    ksat_asset = "projects/openet/soil/ssurgo_Ksat_WTA_0to152cm_composite"
    clay_asset = "projects/openet/soil/ssurgo_Clay_WTA_0to152cm_composite"
    sand_asset = "projects/openet/soil/ssurgo_Sand_WTA_0to152cm_composite"

    plots = as_ee_feature_collection(fields, feature_id=selector)

    # Optionally filter to a subset of features by ID
    if select is not None:
        plots = plots.filter(ee.Filter.inList(selector, select))

    ksat_ = ee.Image(ksat_asset).select("b1").rename("ksat")
    awc_ = ee.Image(awc_asset).select("b1").rename("awc")
    clay_ = ee.Image(clay_asset).select("b1").rename("clay")
    sand_ = ee.Image(sand_asset).select("b1").rename("sand")

    img = ksat_.addBands([awc_, clay_, sand_])

    _selectors = [selector] + ["awc", "ksat", "clay", "sand"]

    means = img.reduceRegions(collection=plots, reducer=ee.Reducer.mean(), scale=30)

    if debug:
        debug = means.filterMetadata("FID", "equals", 1789).getInfo()

    if dest == "local":
        _write_local(means, _selectors, out_dir, desc)
        return
    if dest == "bucket":
        if not bucket:
            raise ValueError('SSURGO export dest="bucket" requires a bucket name/url')
        task = ee.batch.Export.table.toCloudStorage(
            means,
            description=task_desc,
            bucket=bucket,
            fileNamePrefix=f"{file_prefix}/properties/{desc}",
            fileFormat="CSV",
            selectors=_selectors,
        )
    elif dest == "drive":
        drive_folder_name = f"{drive_folder}_properties" if drive_categorize else drive_folder
        task = ee.batch.Export.table.toDrive(
            collection=means,
            description=task_desc,
            folder=drive_folder_name,
            fileNamePrefix=f"properties/{desc}",
            fileFormat="CSV",
            selectors=_selectors,
        )
    else:
        raise ValueError('dest must be one of {"drive","bucket"}')

    task.start()
    print(task_desc)


def get_hwsd(
    fields: str | ee.FeatureCollection,
    desc: str,
    debug: bool = False,
    selector: str = "FID",
    out_fmt: str = "CSV",
    local_file: str | None = None,
    dest: str = "drive",
    bucket: str | None = None,
    drive_folder: str = "swim",
    file_prefix: str = "swim",
    drive_categorize: bool = False,
) -> None:
    """Export or save HWSD v2 soil property (AWC) per feature.

    Parameters
    - fields: ee.FeatureCollection asset path or object.
    - desc: export description/prefix.
    - debug: bool; if True, prints a sample feature.
    - selector: feature ID property to include.
    - out_fmt: 'CSV' or other formats supported by EE table export.
    - local_file: if provided, writes a local CSV instead of GCS export.
    """
    plots = as_ee_feature_collection(fields, feature_id=selector)

    stype = ee.Image("projects/sat-io/open-datasets/FAO/HWSD_V2_SMU").select("AWC").rename("awc")

    modes = stype.reduceRegions(collection=plots, reducer=ee.Reducer.mode(), scale=30)

    # single value reduction results in stat name: 'mode' instead of image name
    _selectors = [selector, "mode"]

    if debug:
        debug = modes.filterMetadata("FID", "equals", "US-CRT").getInfo()

    if local_file:
        modes = modes.getInfo()
        df = pd.DataFrame([v["properties"] for v in modes["features"]]).rename(
            columns={"mode": "awc"}
        )
        df.to_csv(local_file)

    else:
        if dest == "bucket":
            if not bucket:
                raise ValueError('HWSD export dest="bucket" requires a bucket name/url')
            export_kwargs = dict(
                description=desc,
                bucket=bucket,
                fileNamePrefix=f"{file_prefix}/properties/{desc}",
                fileFormat=out_fmt,
            )
            if out_fmt == "CSV":
                export_kwargs.update({"selectors": _selectors})
            task = ee.batch.Export.table.toCloudStorage(modes, **export_kwargs)
        elif dest == "drive":
            drive_folder_name = f"{drive_folder}_properties" if drive_categorize else drive_folder
            export_kwargs = dict(
                description=desc,
                folder=drive_folder_name,
                fileNamePrefix=f"properties/{desc}",
                fileFormat=out_fmt,
            )
            if out_fmt == "CSV":
                export_kwargs.update({"selectors": _selectors})
            task = ee.batch.Export.table.toDrive(collection=modes, **export_kwargs)
        else:
            raise ValueError('dest must be one of {"drive","bucket"}')
        task.start()
        print(desc)


def get_landcover(
    fields: str | ee.FeatureCollection,
    desc: str,
    debug: bool = False,
    selector: str = "FID",
    select: list[str] | None = None,
    out_fmt: str = "CSV",
    local_file: str | None = None,
    dest: str = "drive",
    bucket: str | None = None,
    drive_folder: str = "swim",
    file_prefix: str = "swim",
    drive_categorize: bool = False,
    out_dir: str | None = None,
    task_desc: str | None = None,
) -> None:
    """Export dominant landcover from MODIS and FROM-GLC10 per feature.

    ``modis_lc`` is the modal MCD12Q1 LC_Type1 class over the full annual
    record (2001 onward), reduced again by spatial mode within each field.
    ``glc10_lc`` is FROM-GLC10, a single 2017 epoch with no time dimension.

    Parameters
    - fields: ee.FeatureCollection asset path or object.
    - desc: filename prefix for the export.
    - debug: bool; if True, prints a sample feature.
    - selector: feature ID property to include.
    - select: optional list[str] of selector values to include.
    - out_fmt: 'CSV' or other formats supported by EE table export.
    - local_file: if provided, writes a local CSV instead of GCS export.
    - task_desc: EE task description (defaults to desc if not provided).
    """
    task_desc = task_desc or desc
    plots = as_ee_feature_collection(fields, feature_id=selector)

    # Optionally filter to a subset of features by ID
    if select is not None:
        plots = plots.filter(ee.Filter.inList(selector, select))

    # Modal class over the whole MCD12Q1 record. This was .first(), which
    # pinned every export to the 2001 image -- one year, and the oldest one, of
    # a 24-year annual series. The container stores a single integer per field
    # at properties/land_cover/modis_lc, so the series has to collapse to one
    # value: per-pixel temporal mode here, then spatial mode over the field in
    # the reduceRegions below.
    vtype = (
        ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1").mode().rename("modis_lc")
    )
    vtype = vtype.addBands(
        [ee.ImageCollection("projects/sat-io/open-datasets/FROM-GLC10").mosaic().rename("glc10_lc")]
    )

    modes = vtype.reduceRegions(collection=plots, reducer=ee.Reducer.mode(), scale=30)
    _selectors = [selector, "modis_lc", "glc10_lc"]

    if debug:
        debug = modes.filterMetadata("FID", "equals", "US-CRT").getInfo()

    if dest == "local":
        _write_local(modes, _selectors, out_dir, desc)
        return

    if local_file:
        modes = modes.getInfo()
        df = pd.DataFrame([v["properties"] for v in modes["features"]])[_selectors]
        df.to_csv(local_file)

    else:
        if dest == "bucket":
            if not bucket:
                raise ValueError('Landcover export dest="bucket" requires a bucket name/url')
            export_kwargs = dict(
                description=task_desc,
                bucket=bucket,
                fileNamePrefix=f"{file_prefix}/properties/{desc}",
                fileFormat=out_fmt,
            )
            if out_fmt == "CSV":
                export_kwargs.update({"selectors": _selectors})
            task = ee.batch.Export.table.toCloudStorage(modes, **export_kwargs)
        elif dest == "drive":
            drive_folder_name = f"{drive_folder}_properties" if drive_categorize else drive_folder
            export_kwargs = dict(
                description=task_desc,
                folder=drive_folder_name,
                fileNamePrefix=f"properties/{desc}",
                fileFormat=out_fmt,
            )
            if out_fmt == "CSV":
                export_kwargs.update({"selectors": _selectors})
            task = ee.batch.Export.table.toDrive(collection=modes, **export_kwargs)
        else:
            raise ValueError('dest must be one of {"drive","bucket"}')
        task.start()
        print(task_desc)


if __name__ == "__main__":
    pass
# ========================= EOF ====================================================================
