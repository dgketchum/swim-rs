"""Build the Example 10 run container from the base container.

Thin wrapper over Example 5 ``build_container`` (the Example 7 pattern) with
the departures this test needs:

- four LST-based members (SSEBop, eeMETRIC, geeSEBAL, DisALEXI), simple
  nanmean ensemble, ETf kept from each member's first year (1999 / 2001)
  rather than cut to 2016;
- OpenET bias-corrected ETo/ETr written as ``{var}_corr`` with pre-1999 spinup
  backfilled from raw GridMET (unchanged from Example 5);
- dynamics in the Example 6 water-balance classifier mode (``use_lulc=True``,
  ``annual_2yr`` with the retain / demand-season Stage 2) on the four-member
  ensemble. The cropland gate reads the CDL cultivated flag rather than
  FROM-GLC10, which labels 16 of these 24 wheat-fallow / alfalfa fields as
  grassland and would exclude them from the test. IrrMapper never enters.
- groundwater subsidy is zeroed by default (``--keep-gwsub`` retains it): the
  subsidy test reads the same ET > P excess the irrigation test does, and a
  subsidy switched on at a dryland field would refill the root zone and erase
  the stress signal the rainfed / irrigated arms are meant to expose.

    uv run python examples/10_OR_Irrigation_Test/build_container.py --run e10cal
    uv run python examples/10_OR_Irrigation_Test/build_container.py --run e10cal --dynamics-only
"""

import argparse
import json
import os
import shutil
import sys

from ex10_paths import E5, LST_MODELS, etf_dir, load_config, refet_dir

if str(E5) not in sys.path:
    sys.path.insert(0, str(E5))

import build_container as b5  # noqa: E402  (Example 5 pipeline)
import numpy as np  # noqa: E402

from swimrs.container import open_container  # noqa: E402

b5.MODELS = LST_MODELS
b5.ETF_START = "1999-01-01"
b5.ETF_END = "2025-12-31"


def compute_dynamics(container_path, cfg, gwsub_irr_fallback=True, keep_gwsub=False):
    """Water-balance irrigation classifier (Example 6 mode) on the LST ensemble."""
    c = open_container(container_path, mode="r+")
    c.compute.dynamics(
        etf_model="ensemble",
        masks=b5.MASKS,
        instruments=("landsat", "sentinel"),
        use_mask=False,
        use_lulc=True,
        irr_threshold=cfg.irrigation_threshold or 0.3,
        met_source=cfg.met_source,
        overwrite=True,
        gwsub_irr_fallback=gwsub_irr_fallback,
        lulc_irr_method="annual_2yr",
        annual_subsidy_ratio=1.3,
        annual_retain_ratio=1.1,
        season_storage_mm=150.0,
        cropland_source="cdl",
    )
    if not keep_gwsub:
        arr = c._root["derived/dynamics/gwsub_data"]
        n_years = sum(
            sum(1 for v in json.loads(s).values() if isinstance(v, dict) and v.get("f_sub", 0) > 0)
            for s in arr[:]
        )
        arr[:] = np.array(["{}"] * arr.shape[0], dtype=arr.dtype)
        print(
            f"  gwsub_data zeroed ({n_years} subsidized field-years removed); --keep-gwsub retains them"
        )
    c.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Example 10 run container")
    parser.add_argument("--run", required=True, help="Run tag (e.g. e10cal)")
    parser.add_argument(
        "--source", default=None, help="Source container (default: base from config)"
    )
    parser.add_argument(
        "--no-gwsub-irr-fallback",
        action="store_true",
        help="Do not let the groundwater-subsidy test flip a non-irrigated year on (Example 6 default lets it)",
    )
    parser.add_argument(
        "--keep-gwsub",
        action="store_true",
        help="Retain the groundwater-subsidy years the ET/P test flags (default: zero them, see module doc)",
    )
    parser.add_argument(
        "--dynamics-only",
        action="store_true",
        help="Recompute dynamics on the existing run container (no copy / ETf / ETo ingest)",
    )
    args = parser.parse_args()

    cfg = load_config()
    source = args.source or os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")
    dest = os.path.join(cfg.data_dir, f"{cfg.project_name}_{args.run}.swim")
    etf_tables = etf_dir(cfg)
    refet_tables = refet_dir(cfg)

    if args.dynamics_only:
        if not os.path.exists(dest):
            raise FileNotFoundError(f"Run container not found: {dest}")
        print(f"Recomputing dynamics only on {dest}")
        compute_dynamics(
            dest, cfg, gwsub_irr_fallback=not args.no_gwsub_irr_fallback, keep_gwsub=args.keep_gwsub
        )
        b5.validate(dest)
        return

    for path, what in (
        (source, "Source container"),
        (etf_tables, "ETf tables"),
        (refet_tables, "RefET tables"),
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{what} not found: {path}")
    if os.path.exists(dest):
        raise FileExistsError(f"Destination already exists (non-clobbering): {dest}")

    print(f"Building {args.run}\nCopying {source}\n    -> {dest}")
    shutil.copytree(source, dest)

    print(
        f"\n=== Step 1: Ingest ETf ({', '.join(b5.MODELS)}; {b5.ETF_START}..{b5.ETF_END}; MAX dedup) ==="
    )
    b5.ingest_new_etf(dest, etf_tables)

    print("\n=== Step 2: Ingest OpenET reference ETo/ETr ===")
    b5.ingest_openet_eto(dest, refet_tables)

    print("\n=== Step 3: Simple mean ensemble of the LST members ===")
    b5.compute_mean_ensemble(dest)

    print("\n=== Step 4: Dynamics (water-balance classifier, no IrrMapper) ===")
    compute_dynamics(
        dest, cfg, gwsub_irr_fallback=not args.no_gwsub_irr_fallback, keep_gwsub=args.keep_gwsub
    )

    print("\n=== Validation Summary ===")
    b5.validate(dest)
    print(f"\n{args.run} container ready: {dest}")


if __name__ == "__main__":
    main()
