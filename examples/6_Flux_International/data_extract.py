import os
import sys
from pathlib import Path

# Add this directory to path so etf package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from swimrs.data_extraction.ee.ee_utils import is_authorized
from swimrs.swim.config import ProjectConfig


def _load_config() -> ProjectConfig:
    project_dir = Path(__file__).resolve().parent
    conf_path = project_dir / "6_Flux_International.toml"
    config = ProjectConfig()

    # Prefer the configured root (e.g., /data/ssd2/swim) when available; otherwise run in-repo.
    if os.path.isdir("/data/ssd2/swim"):
        config.read_config(str(conf_path))
    else:
        config.read_config(str(conf_path), project_root_override=str(project_dir.parent))

    return config


def extract_era5land(conf: ProjectConfig, overwrite: bool = False) -> None:
    """Exports monthly CSVs to Cloud Storage (ERA5-Land is large; bucket export only)."""
    from swimrs.data_extraction.ee.ee_era5 import sample_era5_land_variables_daily

    start_yr = conf.start_dt.year
    end_yr = conf.end_dt.year

    is_authorized()
    sample_era5_land_variables_daily(
        shapefile=conf.fields_shapefile,
        bucket=conf.ee_bucket,
        debug=False,
        check_dir=conf.met_dir,
        overwrite=overwrite,
        start_yr=start_yr,
        end_yr=end_yr,
        feature_id_col=conf.feature_id_col,
        file_prefix=conf.project_name,
    )


def extract_properties(conf: ProjectConfig) -> None:
    """Exports landcover + HWSD (AWC) to bucket."""
    from swimrs.data_extraction.ee.ee_props import get_hwsd, get_landcover

    is_authorized()
    project = conf.project_name or "swim"
    get_landcover(
        conf.fields_shapefile,
        f"{project}_landcover",
        debug=False,
        selector=conf.feature_id_col,
        dest="bucket",
        bucket=conf.ee_bucket,
        file_prefix=project,
    )
    get_hwsd(
        conf.fields_shapefile,
        f"{project}_hwsd",
        debug=False,
        selector=conf.feature_id_col,
        dest="bucket",
        bucket=conf.ee_bucket,
        file_prefix=project,
    )


def extract_ndvi(conf: ProjectConfig, overwrite=False) -> None:
    """Exports Landsat + Sentinel-2 NDVI with mask_type='no_mask' (international workflow)."""
    from swimrs.data_extraction.ee.ndvi_export import sparse_sample_ndvi

    start_yr = conf.start_dt.year
    end_yr = conf.end_dt.year

    is_authorized()
    mask = "no_mask"

    if not overwrite:
        landsat_check = os.path.join(conf.landsat_dir or "", "extracts", "ndvi", mask)
    else:
        landsat_check = None

    sparse_sample_ndvi(
        conf.fields_shapefile,
        bucket=conf.ee_bucket,
        dest="bucket",
        debug=False,
        satellite="landsat",
        mask_type=mask,
        check_dir=landsat_check,
        start_yr=start_yr,
        end_yr=end_yr,
        feature_id=conf.feature_id_col,
        file_prefix=conf.project_name,
    )

    sentinel_check = os.path.join(conf.sentinel_dir or "", "extracts", "ndvi", mask)
    sparse_sample_ndvi(
        conf.fields_shapefile,
        bucket=conf.ee_bucket,
        dest="bucket",
        debug=False,
        satellite="sentinel",
        mask_type=mask,
        check_dir=sentinel_check,
        start_yr=max(2017, start_yr),
        end_yr=end_yr,
        feature_id=conf.feature_id_col,
        file_prefix=conf.project_name,
    )


def extract_etf(conf: ProjectConfig, sites=None, overwrite: bool = False) -> None:
    """
    Extract PT-JPL ETf zonal statistics using ERA5LAND meteorology.

    This function uses the etf/ package module to export per-scene ET fraction
    zonal means for international flux sites where US-specific datasets
    (GRIDMET, IrrMapper) are not available.

    Parameters
    ----------
    conf : ProjectConfig
        Project configuration object.
    sites : list, optional
        List of site IDs to process. If None, processes all sites.
    overwrite : bool, optional
        If True, skip check_dir and re-export all data. Default is False.
    """
    import ee

    ee.Initialize()

    from etf import export_ptjpl_zonal_stats

    if overwrite:
        chk_dir = None
    else:
        chk_dir = os.path.join(conf.landsat_dir, "extracts", "ptjpl_etf")

    print(f"\n{'=' * 60}")
    print("Extracting PT-JPL ETf zonal statistics (ERA5LAND)")
    print(f"{'=' * 60}")

    export_ptjpl_zonal_stats(
        shapefile=conf.fields_shapefile,
        bucket=conf.ee_bucket,
        feature_id=conf.feature_id_col,
        select=sites,
        start_yr=conf.start_dt.year,
        end_yr=conf.end_dt.year,
        check_dir=chk_dir,
        buffer=None,
        batch_size=60,
        file_prefix=conf.project_name,
    )


if __name__ == "__main__":
    cfg = _load_config()

    extract_era5land(cfg, overwrite=True)
    # extract_properties(cfg)
    # extract_ndvi(cfg, overwrite=False)

    # PT-JPL ETf extraction using ERA5LAND meteorology
    # extract_etf(cfg, overwrite=True)

# ========================= EOF ====================================================================
