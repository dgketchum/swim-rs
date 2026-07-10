"""Build an Example 9 forward-run container from the getInfo base container.

Thin wrapper over Example 5 ``build_container`` (mirrors Example 8): reuses E5's
container-path-based functions unchanged and supplies its own ``main()`` so the
destination is named ``9_Bushland_{run}.swim``. Steps are identical to E5 / Run
22 methodology: copy the base container, re-ingest the 6-model OpenET v2.1 ETf
(ETo denominator, MAX dedup), ingest OpenET reference ETo/ETr, compute the simple
mean ensemble, recompute dynamics, validate.

This example runs FORWARD only (WP-B2 prescribed-irrigation ET check); the
ensemble ETf assembled here is the crop coefficient the forward run reads, not a
calibration target. ETf/refET CSVs come from this project's repo data dir
(``examples/9_Bushland/data/{etf_v21_openet_eto,openet_refet}/``).

    uv run python examples/9_Bushland/build_container.py --run e9for
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E5 = HERE.parent / "5_Flux_Ensemble"
if str(E5) not in sys.path:
    sys.path.insert(0, str(E5))

import build_container as b5  # noqa: E402  (Example 5 pipeline)

from swimrs.swim.config import ProjectConfig  # noqa: E402


def _load_config() -> ProjectConfig:
    conf = HERE / "9_Bushland.toml"
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf))
    else:
        cfg.read_config(str(conf), project_root_override=str(HERE.parent))
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Example 9 forward-run container")
    parser.add_argument("--run", required=True, help="Run name (e.g. e9for)")
    parser.add_argument(
        "--source", default=None, help="Source container (default: base container from config)"
    )
    parser.add_argument(
        "--mad",
        action="store_true",
        help="Use MAD-filtered ensemble (Volk methodology) instead of simple mean",
    )
    args = parser.parse_args()

    cfg = _load_config()

    source = args.source or os.path.join(cfg.data_dir, f"{cfg.project_name}.swim")
    dest = os.path.join(cfg.data_dir, f"{cfg.project_name}_{args.run}.swim")
    etf_dir = str(HERE / "data" / "etf_v21_openet_eto")
    refet_dir = str(HERE / "data" / "openet_refet")

    if not os.path.exists(source):
        raise FileNotFoundError(f"Source container not found: {source}")

    if os.path.exists(dest):
        raise FileExistsError(
            f"Destination already exists (non-clobbering): {dest}\n"
            f"Remove manually if you want to rebuild."
        )

    print(f"Building {args.run}")
    print(f"Copying {source}\n    -> {dest}")
    shutil.copytree(source, dest)

    print("\n=== Step 1: Ingest ETf (6 models, MAX dedup) ===")
    b5.ingest_new_etf(dest, etf_dir)

    print("\n=== Step 2: Ingest OpenET reference ETo/ETr ===")
    b5.ingest_openet_eto(dest, refet_dir)

    if args.mad:
        print("\n=== Step 3: Compute MAD-filtered ensemble ===")
        b5.compute_mad_ensemble(dest)
    else:
        print("\n=== Step 3: Compute simple mean ensemble ===")
        b5.compute_mean_ensemble(dest)

    print("\n=== Step 4: Recompute dynamics ===")
    b5.compute_dynamics(dest, cfg)

    print("\n=== Validation Summary ===")
    b5.validate(dest)

    print(f"\n{args.run} container ready: {dest}")


if __name__ == "__main__":
    main()
