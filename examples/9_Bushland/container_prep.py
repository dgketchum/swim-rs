"""Container prep for Example 9 (Bushland lysimeter), reusing Example 5.

Thin wrapper over Example 5 ``container_prep`` (mirrors Example 8): repoints the
config loader at ``9_Bushland.toml`` and replaces ``build_shapefile`` with a
guard, because the fields shapefile is prebuilt by ``build_bushland_fields.py``.
Everything else -- GridMET mapping, container creation, ingest of met/RS/snow/
properties, fused NDVI, dynamics -- is identical to Example 5.

    uv run python examples/9_Bushland/container_prep.py --overwrite --getinfo
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E5 = HERE.parent / "5_Flux_Ensemble"
if str(E5) not in sys.path:
    sys.path.insert(0, str(E5))

import container_prep as e2  # noqa: E402  (Example 5 pipeline)

from swimrs.swim.config import ProjectConfig  # noqa: E402


def _load_config() -> ProjectConfig:
    conf = HERE / "9_Bushland.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(HERE.parent))
    return cfg


def _guard_shapefile(cfg, overwrite=False, exclude_sites=None):
    """The shapefile is prebuilt by build_bushland_fields.py; do not rebuild it."""
    if not os.path.exists(cfg.fields_shapefile):
        raise SystemExit(
            f"Fields shapefile not found: {cfg.fields_shapefile}\n"
            "Run examples/9_Bushland/build_bushland_fields.py first."
        )
    print(f"Using prebuilt fields shapefile: {cfg.fields_shapefile}")


# Repoint config + neutralize the flux-cohort shapefile builder.
e2._load_config = _load_config
e2.build_shapefile = _guard_shapefile


def main() -> None:
    p = argparse.ArgumentParser(description="Container prep for 9_Bushland")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sites", type=str, default=None)
    p.add_argument("--skip-sentinel", action="store_true")
    p.add_argument("--openet-source", choices=["diy", "ee"], default="diy")
    p.add_argument("--getinfo", action="store_true")
    args = p.parse_args()

    sites = [s.strip() for s in args.sites.split(",")] if args.sites else None
    cfg = _load_config()

    e2.build_gridmet_mapping(cfg, overwrite=args.overwrite)
    container = e2.create_project_container(cfg, overwrite=args.overwrite)
    e2.prep_all(
        container,
        cfg,
        sites=sites,
        overwrite=args.overwrite,
        add_sentinel=not args.skip_sentinel,
        openet_source=args.openet_source,
        getinfo=args.getinfo,
    )
    container.close()
    print(f"\nContainer saved to: {container.path}")


if __name__ == "__main__":
    main()
