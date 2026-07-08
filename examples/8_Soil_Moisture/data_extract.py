"""Earth Engine extraction for Example 8 (SCAN soil moisture), reusing Example 5.

Thin wrapper: imports the Example 5 ``data_extract`` module and repoints its
config loader at ``8_Soil_Moisture.toml``. All extraction logic is identical to
Example 5 — SCAN stations are CONUS, so the same inputs and Run 22 methodology
apply: no_mask 150 m footprints, synchronous .getInfo() calls, OpenET
bias-corrected refET (ETo), 6-model OpenET v2.1 member ETf, and Landsat +
Sentinel-2 NDVI. E5's driver lives inline under ``__main__`` (no callable
``main()``), so we replicate the step dispatch here against E5's public
``extract_*`` functions.

    uv run python examples/8_Soil_Moisture/data_extract.py [--steps ...] [--sites ...]
"""

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).resolve().parent
E5 = HERE.parent / "5_Flux_Ensemble"
if str(E5) not in sys.path:
    sys.path.insert(0, str(E5))

import data_extract as e2  # noqa: E402  (Example 5 pipeline)

from swimrs.swim.config import ProjectConfig  # noqa: E402


def _load_config() -> ProjectConfig:
    conf = HERE / "8_Soil_Moisture.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(HERE.parent))
    return cfg


# Repoint the E5 pipeline's config loader at this project.
e2._load_config = _load_config

# extract_etf_v21 / extract_openet_refet derive their output dir from
# ``Path(__file__).parent`` (the E5 module's location). Repoint E5's module
# ``__file__`` at this project so those CSVs land in examples/8_Soil_Moisture/
# data/{etf_v21_openet_eto,openet_refet}/ instead of the Example 5 dir. Every
# other step writes to cfg.data_dir, which the _load_config override already
# points here, so this only affects the two __file__-relative outputs.
e2.__file__ = str(HERE / "data_extract.py")

# The current swimrs download_gridmet() signature dropped the `use_nldas` kwarg
# (NLDAS hourly-precip support was removed), but E5's extract_gridmet still passes
# it. E8 uses runoff_process="cn" (NLDAS not needed), so strip the stale kwarg.
# extract_gridmet does a function-local `from ...gridmet import download_gridmet`,
# which resolves the patched module attribute at call time.
import swimrs.data_extraction.gridmet.gridmet as _gm  # noqa: E402

_orig_download_gridmet = _gm.download_gridmet


def _download_gridmet_compat(*args, **kwargs):
    kwargs.pop("use_nldas", None)
    return _orig_download_gridmet(*args, **kwargs)


_gm.download_gridmet = _download_gridmet_compat


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=str, default="all")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--sites", type=str, default=None)
    args = parser.parse_args()

    valid_steps = e2.ALL_STEPS + e2.BUCKET_STEPS
    steps = e2.ALL_STEPS if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    unknown = [s for s in steps if s not in valid_steps]
    if unknown:
        raise SystemExit(f"Unknown steps: {unknown}. Choose from: {valid_steps}")

    etf_models = [m.strip() for m in args.models.split(",")] if args.models else None

    config = _load_config()
    gdf = gpd.read_file(config.fields_shapefile, engine="fiona")
    if config.feature_id_col not in gdf.columns:
        raise ValueError(
            f"Feature ID column {config.feature_id_col!r} not found in {config.fields_shapefile}"
        )
    all_sites = gdf[config.feature_id_col].astype(str).to_list()

    if args.sites:
        select_sites = [s.strip() for s in args.sites.split(",")]
        missing = [s for s in select_sites if s not in all_sites]
        if missing:
            raise SystemExit(f"Sites not in shapefile: {missing}")
    else:
        select_sites = all_sites

    if "snodas" in steps:
        e2.extract_snodas(config, select_sites)
    if "properties" in steps:
        e2.extract_properties(config)
    if "ndvi" in steps:
        e2.extract_ndvi(config, select_sites, get_sentinel=True)
    if "gridmet" in steps:
        # download_gridmet() does not create its output dir (unlike the ETf/refET
        # extractors); create the met store before the first write.
        os.makedirs(config.met_dir, exist_ok=True)
        e2.extract_gridmet(config, select_sites)
    if "etf_v21" in steps:
        e2.extract_etf_v21(config, select_sites, models=etf_models)
    if "refet" in steps:
        e2.extract_openet_refet(config, select_sites)

    if "snodas_bucket" in steps:
        e2.extract_snodas_bucket(config)
    if "properties_bucket" in steps:
        e2.extract_properties_bucket(config)
    if "ndvi_bucket" in steps:
        e2.extract_ndvi_bucket(config, select_sites, get_sentinel=True)


if __name__ == "__main__":
    main()
