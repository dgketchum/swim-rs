"""Earth Engine / GridMET extraction for Example 10, reusing the Example 5 pipeline.

Thin wrapper over Example 5 ``data_extract`` (the Example 7 pattern) with three
deliberate departures:

- ETf is pulled for the four LST-based OpenET v2.1 members only (SSEBop,
  eeMETRIC, geeSEBAL, DisALEXI), from each member's first available year.
- The properties step pulls CDL, SSURGO and land cover but **not** IrrMapper or
  LANID; irrigation status is decided later by the water-balance classifier.
- GridMET is a single cell (680670) for every field; the mapping is prebuilt by
  ``prepare_fields.py``, so the Example 5 step only downloads.

All outputs land under the project data dir on the data volume
(``/data/ssd1/swim/or_irr_test/data``), including the ETf and reference-ET
tables that Example 5 keeps beside its scripts.

    uv run python examples/10_OR_Irrigation_Test/data_extract.py [--steps ...] [--sites ...]
"""

import argparse
import os
import sys

import geopandas as gpd
from ex10_paths import E5, LST_MODELS, load_config

if str(E5) not in sys.path:
    sys.path.insert(0, str(E5))

import data_extract as e5  # noqa: E402  (Example 5 pipeline)

import swimrs.data_extraction.gridmet.gridmet as _gm  # noqa: E402

# Example 5's extract_gridmet passes use_nldas, which download_gridmet no longer
# accepts (same shim as Example 7).
_orig_download_gridmet = _gm.download_gridmet


def _download_gridmet_compat(*args, **kwargs):
    kwargs.pop("use_nldas", None)
    return _orig_download_gridmet(*args, **kwargs)


_gm.download_gridmet = _download_gridmet_compat


def extract_properties_no_irrmapper(cfg) -> None:
    """CDL, SSURGO and land cover via .getInfo(); no IrrMapper/LANID table."""
    e5.is_authorized()
    from swimrs.data_extraction.ee.ee_props import get_cdl, get_landcover, get_ssurgo

    project = cfg.project_name
    out_dir = os.path.join(cfg.data_dir, "properties", "getinfo")
    get_cdl(
        cfg.fields_shapefile,
        f"{project}_cdl",
        selector=cfg.feature_id_col,
        dest="local",
        out_dir=out_dir,
    )
    get_ssurgo(
        cfg.fields_shapefile,
        f"{project}_ssurgo",
        selector=cfg.feature_id_col,
        dest="local",
        out_dir=out_dir,
    )
    get_landcover(
        cfg.fields_shapefile,
        f"{project}_landcover",
        selector=cfg.feature_id_col,
        dest="local",
        out_dir=out_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--steps", type=str, default="all", help=f"Comma-separated: {', '.join(e5.ALL_STEPS)}"
    )
    parser.add_argument(
        "--models", type=str, default=",".join(LST_MODELS), help="OpenET v2.1 members for etf_v21"
    )
    parser.add_argument(
        "--sites", type=str, default=None, help="Comma-separated OPENET_IDs (default: all)"
    )
    args = parser.parse_args()

    steps = e5.ALL_STEPS if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    unknown = [s for s in steps if s not in e5.ALL_STEPS]
    if unknown:
        raise SystemExit(f"Unknown steps: {unknown}. Choose from: {e5.ALL_STEPS}")
    etf_models = [m.strip() for m in args.models.split(",")]

    cfg = load_config()
    e5._load_config = load_config
    # extract_etf_v21 / extract_openet_refet write to Path(__file__).parents[1]/data;
    # point that at the project workspace so they land on the data volume.
    e5.__file__ = os.path.join(cfg.project_ws, "container_build", "data_extract.py")

    if not os.path.exists(cfg.gridmet_mapping_shp):
        raise SystemExit(
            f"GridMET mapping missing: {cfg.gridmet_mapping_shp}. Run prepare_fields.py first."
        )

    gdf = gpd.read_file(cfg.fields_shapefile, engine="fiona")
    all_sites = gdf[cfg.feature_id_col].astype(str).to_list()
    if args.sites:
        select_sites = [s.strip() for s in args.sites.split(",")]
        missing = [s for s in select_sites if s not in all_sites]
        if missing:
            raise SystemExit(f"Sites not in shapefile: {missing}")
    else:
        select_sites = all_sites

    if "snodas" in steps:
        e5.extract_snodas(cfg, select_sites)
    if "properties" in steps:
        extract_properties_no_irrmapper(cfg)
    if "ndvi" in steps:
        e5.extract_ndvi(cfg, select_sites, get_sentinel=True)
    if "gridmet" in steps:
        os.makedirs(cfg.met_dir, exist_ok=True)
        e5.extract_gridmet(cfg, select_sites)
    if "etf_v21" in steps:
        e5.extract_etf_v21(cfg, select_sites, models=etf_models)
    if "refet" in steps:
        e5.extract_openet_refet(cfg, select_sites)


if __name__ == "__main__":
    main()
