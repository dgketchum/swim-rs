"""Base container for Example 10, reusing the Example 5 pipeline.

Thin wrapper over Example 5 ``container_prep`` (the Example 7 pattern):

- the fields shapefile and GridMET mapping are prebuilt by ``prepare_fields.py``;
- properties ingest passes no IrrMapper/LANID table (CDL is ingested as well);
- dynamics are deferred to ``build_container.py``, where the water-balance
  classifier runs on the four-member ETf ensemble.

    uv run python examples/10_OR_Irrigation_Test/container_prep.py --overwrite --getinfo
"""

import argparse
import os
import sys

from ex10_paths import E5, load_config

if str(E5) not in sys.path:
    sys.path.insert(0, str(E5))

import container_prep as e5  # noqa: E402  (Example 5 pipeline)


def _guard_shapefile(cfg, overwrite=False, exclude_sites=None):
    if not os.path.exists(cfg.fields_shapefile):
        raise SystemExit(
            f"Fields shapefile not found: {cfg.fields_shapefile}. Run prepare_fields.py first."
        )
    print(f"Using prebuilt fields shapefile: {cfg.fields_shapefile}")


def _guard_gridmet_mapping(cfg, overwrite=False):
    if not os.path.exists(cfg.gridmet_mapping_shp):
        raise SystemExit(
            f"GridMET mapping not found: {cfg.gridmet_mapping_shp}. Run prepare_fields.py first."
        )
    print(f"Using prebuilt GridMET mapping: {cfg.gridmet_mapping_shp}")


def _ingest_properties(container, cfg, overwrite=False, getinfo=False):
    """SSURGO + land cover + CDL from the getInfo tables; no irrigation table."""
    print("\n=== Ingesting Properties (no IrrMapper/LANID) ===")
    prop_dir = os.path.join(cfg.data_dir, "properties", "getinfo")
    container.ingest.properties(
        soils_csv=os.path.join(prop_dir, os.path.basename(cfg.ssurgo_csv)),
        lulc_csv=os.path.join(prop_dir, os.path.basename(cfg.lulc_csv)),
        cdl_csv=os.path.join(prop_dir, f"{cfg.project_name}_cdl.csv"),
        irr_csv=None,
        uid_column=cfg.feature_id_col,
        overwrite=overwrite,
    )


def _defer_dynamics(container, cfg, overwrite=False):
    print("\n=== Dynamics deferred to build_container.py (water-balance classifier on ETf) ===")


e5._load_config = load_config
e5.build_shapefile = _guard_shapefile
e5.build_gridmet_mapping = _guard_gridmet_mapping
e5.ingest_properties = _ingest_properties
e5.compute_dynamics = _defer_dynamics


def main() -> None:
    p = argparse.ArgumentParser(description="Base container for 10_OR_Irrigation_Test")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sites", type=str, default=None)
    p.add_argument("--skip-sentinel", action="store_true")
    p.add_argument("--getinfo", action="store_true", default=True, help="(always on here)")
    args = p.parse_args()

    sites = [s.strip() for s in args.sites.split(",")] if args.sites else None
    cfg = load_config()

    container = e5.create_project_container(cfg, overwrite=args.overwrite)
    e5.prep_all(
        container,
        cfg,
        sites=sites,
        overwrite=args.overwrite,
        add_sentinel=not args.skip_sentinel,
        getinfo=True,
    )
    container.close()
    print(f"\nContainer saved to: {container.path}")


if __name__ == "__main__":
    main()
