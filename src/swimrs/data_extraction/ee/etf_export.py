"""
Unified ETf (ET fraction) export module using OpenET FOSS packages.

This module provides a single interface for exporting ET fraction zonal statistics
using the open-source OpenET Python packages (openet-ptjpl, openet-ssebop, etc.).

Functions
---------
export_etf : Unified dispatcher for ETf extraction with sparse/clustered modes.
"""

import os
import warnings

import ee
from tqdm import tqdm

from swimrs.utils.optional_deps import missing_optional_dependency

from .common import (
    GRIDMET_BAND,
    GRIDMET_FACTOR,
    GRIDMET_SOURCE,
    LANDSAT_COLLECTIONS,
    WEST_STATES,
    build_feature_collection,
    export_table,
    get_irrigation_mask,
    get_pathrows_from_scenes,
    load_shapefile,
    parse_scene_name,
    setup_irrigation_masks,
    shapefile_to_feature_collection,
)

# Lazy imports for optional OpenET packages
_OPENET_MODULES = {}


def _get_openet_module(model: str):
    """Lazy import of OpenET model module."""
    if model in _OPENET_MODULES:
        return _OPENET_MODULES[model]

    try:
        if model == "ptjpl":
            import openet.ptjpl as mod
        elif model == "ssebop":
            import openet.ssebop as mod
        elif model == "sims":
            import openet.sims as mod
        elif model == "geesebal":
            import openet.geesebal as mod
        else:
            raise ValueError(f"Unknown model: {model}")

        _OPENET_MODULES[model] = mod
        return mod

    except ImportError:
        raise missing_optional_dependency(
            extra="openet",
            purpose=f"{model.upper()} ETf Earth Engine export",
            import_name=f"openet-{model}",
        )


# Reference ET configuration
ET_REF_SOURCE = GRIDMET_SOURCE
ET_REF_BAND = GRIDMET_BAND
ET_REF_FACTOR = GRIDMET_FACTOR
ET_REF_RESAMPLE = "bilinear"

# Models that need ET computed then divided by refET (vs direct et_fraction)
MODELS_NEED_ET_DIVISION = {"ptjpl", "geesebal"}


def _compute_etf_image(model: str, img_id: str) -> ee.Image:
    """
    Compute ET fraction image for a given model and scene.

    Parameters
    ----------
    model : str
        OpenET model name.
    img_id : str
        Landsat scene ID.

    Returns
    -------
    ee.Image
        ET fraction image.
    """
    mod = _get_openet_module(model)

    if model == "ptjpl":
        img = mod.Image.from_landsat_c2_sr(
            img_id,
            et_reference_source=ET_REF_SOURCE,
            et_reference_band=ET_REF_BAND,
            et_reference_factor=ET_REF_FACTOR,
            et_reference_resample=ET_REF_RESAMPLE,
        )
        return img.et_fraction

    elif model == "ssebop":
        img = mod.Image.from_landsat_c2_sr(img_id)
        return img.et_fraction

    elif model == "sims":
        img = mod.Image.from_landsat_c2_sr(
            img_id,
            et_reference_source=ET_REF_SOURCE,
            et_reference_band=ET_REF_BAND,
            et_reference_factor=ET_REF_FACTOR,
            et_reference_resample=ET_REF_RESAMPLE,
        )
        return img.et_fraction

    elif model == "geesebal":
        img = mod.Image.from_landsat_c2_sr(
            img_id,
            et_reference_source=ET_REF_SOURCE,
            et_reference_band=ET_REF_BAND,
            et_reference_factor=ET_REF_FACTOR,
            et_reference_resample=ET_REF_RESAMPLE,
        )
        return img.et_fraction

    else:
        raise ValueError(f"Unknown model: {model}")


def _get_scenes_for_year(
    model: str,
    year: int,
    geometry: ee.Geometry,
) -> list[str]:
    """Get sorted list of Landsat scene IDs for a year and geometry."""
    mod = _get_openet_module(model)

    coll = mod.Collection(
        LANDSAT_COLLECTIONS,
        start_date=f"{year}-01-01",
        end_date=f"{year}-12-31",
        geometry=geometry,
        cloud_cover_max=70,
    )
    scenes = coll.get_image_ids()
    scenes = list(set(scenes))
    scenes = sorted(scenes, key=lambda item: item.split("_")[-1])
    return scenes


def _check_pathrow_warning(
    scenes: list[str],
    clustered: bool,
    pathrow_threshold: int,
    context: str = "",
) -> None:
    """Warn if clustered mode but data spans many path/rows."""
    if not clustered:
        return

    pathrows = get_pathrows_from_scenes(scenes)
    if len(pathrows) > pathrow_threshold:
        pathrow_sample = sorted(pathrows)[:5]
        more = f"... ({len(pathrows)} total)" if len(pathrows) > 5 else ""
        warnings.warn(
            f"Clustered mode detected {len(pathrows)} Landsat path/rows{context}. "
            f"Your data may be spatially dispersed. Consider using --sparse. "
            f"Path/rows: {pathrow_sample}{more}",
            UserWarning,
        )


def export_etf(
    shapefile: str,
    model: str,
    feature_id: str = "FID",
    select: list[str] | None = None,
    start_yr: int = 2000,
    end_yr: int = 2024,
    mask_type: str = "no_mask",
    check_dir: str | None = None,
    state_col: str = "state",
    buffer: float | None = None,
    batch_size: int = 30,
    dest: str = "drive",
    bucket: str | None = None,
    drive_folder: str = "swim",
    file_prefix: str = "swim",
    clustered: bool = True,
    pathrow_threshold: int = 4,
) -> None:
    """
    Export ETf zonal statistics using OpenET FOSS packages.

    Parameters
    ----------
    shapefile : str
        Path to polygon shapefile with feature IDs.
    model : str
        OpenET model: 'ptjpl', 'ssebop', 'sims', or 'geesebal'.
    feature_id : str
        Field name for feature identifier.
    select : list, optional
        List of feature IDs to process.
    start_yr : int
        Inclusive start year.
    end_yr : int
        Inclusive end year.
    mask_type : str
        Irrigation masking: 'no_mask', 'irr', or 'inv_irr'.
    check_dir : str, optional
        Skip exports if CSV exists at check_dir/<desc>.csv.
    state_col : str
        Column with state abbreviation for mask source selection.
    buffer : float, optional
        Buffer distance in meters to apply to geometries.
    batch_size : int
        Number of scenes per export batch.
    dest : str
        Export destination: 'drive' or 'bucket'.
    bucket : str, optional
        GCS bucket name (required if dest='bucket').
    drive_folder : str
        Google Drive folder name.
    file_prefix : str
        Bucket path prefix.
    clustered : bool
        If True, export one task per year (efficient for clustered fields).
        If False, export one task per field-year (required for dispersed fields).
    pathrow_threshold : int
        Warn if clustered=True but more than this many path/rows detected.

    Raises
    ------
    ImportError
        If required openet-* package is not installed.
    ValueError
        If model is unknown or bucket required but not provided.
    """
    if model not in {"ptjpl", "ssebop", "sims", "geesebal"}:
        raise ValueError(f"Unknown model: {model}. Available: ptjpl, ssebop, sims, geesebal")

    if dest == "bucket" and not bucket:
        raise ValueError("bucket is required when dest='bucket'")

    # Validate openet package is available
    _get_openet_module(model)

    if clustered:
        _export_etf_clustered(
            shapefile=shapefile,
            model=model,
            feature_id=feature_id,
            select=select,
            start_yr=start_yr,
            end_yr=end_yr,
            mask_type=mask_type,
            check_dir=check_dir,
            state_col=state_col,
            batch_size=batch_size,
            dest=dest,
            bucket=bucket,
            drive_folder=drive_folder,
            file_prefix=file_prefix,
            pathrow_threshold=pathrow_threshold,
        )
    else:
        _export_etf_sparse(
            shapefile=shapefile,
            model=model,
            feature_id=feature_id,
            select=select,
            start_yr=start_yr,
            end_yr=end_yr,
            mask_type=mask_type,
            check_dir=check_dir,
            state_col=state_col,
            buffer=buffer,
            batch_size=batch_size,
            dest=dest,
            bucket=bucket,
            drive_folder=drive_folder,
            file_prefix=file_prefix,
        )


def _export_etf_sparse(
    shapefile: str,
    model: str,
    feature_id: str,
    select: list[str] | None,
    start_yr: int,
    end_yr: int,
    mask_type: str,
    check_dir: str | None,
    state_col: str,
    buffer: float | None,
    batch_size: int,
    dest: str,
    bucket: str | None,
    drive_folder: str,
    file_prefix: str,
) -> None:
    """Export ETf with one task per field-year (sparse mode)."""
    df = load_shapefile(shapefile, feature_id, buffer=buffer)

    if select is not None:
        df = df[df.index.isin(select)]

    irr_coll, irr_min_yr_mask, lanid, east_fc = setup_irrigation_masks()

    for fid, row in tqdm(df.iterrows(), desc=f"Export {model} ETf (sparse)", total=len(df)):
        if row["geometry"].geom_type not in ("Polygon", "MultiPolygon"):
            continue

        polygon = ee.Geometry(row.geometry.__geo_interface__)

        for year in range(start_yr, end_yr + 1):
            # Get irrigation mask if needed
            if mask_type in ("irr", "inv_irr"):
                state = row.get(state_col, None) if state_col in row else None
                irr, irr_mask = get_irrigation_mask(
                    year, state, irr_coll, irr_min_yr_mask, lanid, east_fc
                )
            else:
                irr, irr_mask = None, None

            scenes = _get_scenes_for_year(model, year, polygon)
            if not scenes:
                continue

            # Process in batches
            n_batches = (len(scenes) + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(scenes))
                batch_scenes = scenes[batch_start:batch_end]

                if n_batches > 1:
                    desc = f"{model}_etf_{fid}_{mask_type}_{year}_b{batch_idx:02d}"
                else:
                    desc = f"{model}_etf_{fid}_{mask_type}_{year}"

                if check_dir:
                    check_path = os.path.join(check_dir, f"{desc}.csv")
                    if os.path.exists(check_path):
                        continue

                fn_prefix = (
                    f"{file_prefix}/remote_sensing/landsat/extracts/{model}_etf/{mask_type}/{desc}"
                )

                first, bands = True, None
                selectors = [feature_id]

                for img_id in batch_scenes:
                    _name = parse_scene_name(img_id)
                    selectors.append(_name)

                    try:
                        etf_img = _compute_etf_image(model, img_id).rename(_name)
                    except ee.ee_exception.EEException as e:
                        print(f"{_name} error: {e}")
                        continue

                    # Apply masking
                    if mask_type == "irr":
                        etf_img = etf_img.clip(polygon).mask(irr_mask)
                    elif mask_type == "inv_irr":
                        etf_img = etf_img.clip(polygon).mask(irr.gt(0))
                    else:
                        etf_img = etf_img.clip(polygon)

                    if first:
                        bands = etf_img
                        first = False
                    else:
                        bands = bands.addBands([etf_img])

                if bands is None:
                    continue

                fc = build_feature_collection(polygon, fid, feature_id)
                data = bands.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)

                export_table(
                    data=data,
                    desc=desc,
                    selectors=selectors,
                    dest=dest,
                    bucket=bucket,
                    fn_prefix=fn_prefix,
                    drive_folder=drive_folder,
                )


def _export_etf_clustered(
    shapefile: str,
    model: str,
    feature_id: str,
    select: list[str] | None,
    start_yr: int,
    end_yr: int,
    mask_type: str,
    check_dir: str | None,
    state_col: str,
    batch_size: int,
    dest: str,
    bucket: str | None,
    drive_folder: str,
    file_prefix: str,
    pathrow_threshold: int,
) -> None:
    """Export ETf with one task per year (clustered mode)."""
    # Build FeatureCollection from shapefile
    feature_coll = shapefile_to_feature_collection(
        shapefile, feature_id, select=select, keep_props=[state_col]
    )

    # Determine if all features are in west or east states
    df = load_shapefile(shapefile, feature_id)
    if select is not None:
        df = df[df.index.isin(select)]

    if state_col not in df.columns:
        raise ValueError(f"state_col '{state_col}' not found in shapefile")

    states_in_data = set(df[state_col].dropna().unique())
    west_set = set(WEST_STATES)
    all_west = states_in_data.issubset(west_set)
    all_east = states_in_data.isdisjoint(west_set)

    if not (all_west or all_east):
        raise ValueError(
            "Clustered mode requires all features be in western states or all in eastern states. "
            f"Found mixed: {states_in_data}"
        )
    use_west = all_west

    irr_coll, irr_min_yr_mask, lanid, east_fc = setup_irrigation_masks()

    # Get geometry for scene queries
    fc_geometry = feature_coll.geometry()

    pathrow_warned = False

    for year in tqdm(range(start_yr, end_yr + 1), desc=f"Export {model} ETf (clustered)"):
        # Get irrigation masks for this year
        if mask_type in ("irr", "inv_irr"):
            if use_west:
                irr = (
                    irr_coll.filterDate(f"{year}-01-01", f"{year}-12-31")
                    .select("classification")
                    .mosaic()
                )
                irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))
            else:
                irr_mask = lanid.select(f"irr_{year}").clip(east_fc)
                irr = ee.Image(1).subtract(irr_mask)
        else:
            irr, irr_mask = None, None

        scenes = _get_scenes_for_year(model, year, fc_geometry)
        if not scenes:
            continue

        # Check pathrow warning once per export
        if not pathrow_warned:
            _check_pathrow_warning(
                scenes,
                clustered=True,
                pathrow_threshold=pathrow_threshold,
                context=f" for year {year}",
            )
            pathrow_warned = True

        # Process in batches
        n_batches = (len(scenes) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(scenes))
            batch_scenes = scenes[batch_start:batch_end]

            if n_batches > 1:
                desc = f"{model}_etf_{mask_type}_{year}_b{batch_idx:02d}"
            else:
                desc = f"{model}_etf_{mask_type}_{year}"

            if check_dir:
                check_path = os.path.join(check_dir, f"{desc}.csv")
                if os.path.exists(check_path):
                    continue

            fn_prefix = (
                f"{file_prefix}/remote_sensing/landsat/extracts/{model}_etf/{mask_type}/{desc}"
            )

            first, bands = True, None
            selectors = [feature_id]

            for img_id in batch_scenes:
                _name = parse_scene_name(img_id)
                selectors.append(_name)

                try:
                    etf_img = _compute_etf_image(model, img_id).rename(_name)
                except ee.ee_exception.EEException as e:
                    print(f"{_name} error: {e}")
                    continue

                # Apply masking
                if mask_type == "irr":
                    etf_img = etf_img.clip(fc_geometry).mask(irr_mask)
                elif mask_type == "inv_irr":
                    etf_img = etf_img.clip(fc_geometry).mask(irr.gt(0))
                else:
                    etf_img = etf_img.clip(fc_geometry)

                if first:
                    bands = etf_img
                    first = False
                else:
                    bands = bands.addBands([etf_img])

            if bands is None:
                continue

            data = bands.reduceRegions(collection=feature_coll, reducer=ee.Reducer.mean(), scale=30)

            export_table(
                data=data,
                desc=desc,
                selectors=selectors,
                dest=dest,
                bucket=bucket,
                fn_prefix=fn_prefix,
                drive_folder=drive_folder,
            )
