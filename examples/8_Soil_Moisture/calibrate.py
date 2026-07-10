"""PEST++ IES calibration for Example 8 (SCAN soil moisture), reusing Example 5.

Thin wrapper over Example 5 ``calibrate``: reuses ``run_pest_sequence`` and its
RUN_POLICY archiver (``_archive_pest_outputs``) unchanged, repointing the config
loader at ``8_Soil_Moisture.toml``. Because ``run_pest_sequence`` chdir's to
``Path(__file__).parent`` of the E5 module during the build, we also repoint the
E5 module's ``__file__`` at this project so PstFrom's scratch log and the master
process land here, not in the Example 5 dir.

Single-run (not batch) calibration, matching Example 5: 28 sites is far under the
~200-field single-run ceiling in notes/calibration_guidance.md. 8 params/site,
200 realizations, 3 IES iterations, ensemble ETf target (6 OpenET v2.1 members),
spread weighting. In-situ soil moisture is never used here — calibration targets
only the satellite ETf record, exactly as in Examples 1-3 and 5. Observed theta is
compared to modeled theta_avail only at evaluation (evaluate.py).

    uv run python examples/8_Soil_Moisture/calibrate.py \
        --container /data/ssd1/swim/8_Soil_Moisture/data/8_Soil_Moisture_e8cal.swim \
        --etf-target-model ensemble --results-tag e8cal --keep-pestrun
"""

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
E5 = HERE.parent / "5_Flux_Ensemble"
if str(E5) not in sys.path:
    sys.path.insert(0, str(E5))

import calibrate as c5  # noqa: E402  (Example 5 pipeline)

from swimrs.swim.config import ProjectConfig  # noqa: E402

# Config TOML to load. Defaults to the canonical 8_Soil_Moisture.toml (e8cal); set
# via --config to a variant such as 8_Soil_Moisture_c1.toml (WP-C1 scheduler recal,
# which repoints pest_run_dir to pestrun_c1 and activates the scheduler knobs).
CONFIG_FILE = HERE / "8_Soil_Moisture.toml"


def _load_config() -> ProjectConfig:
    conf = CONFIG_FILE
    cfg = ProjectConfig()
    if os.path.isdir("/data/ssd2/swim"):
        cfg.read_config(str(conf), calibrate=True)
    else:
        cfg.read_config(str(conf), project_root_override=str(HERE.parent), calibrate=True)
    return cfg


# Repoint config + the module __file__ (run_pest_sequence chdir's to it) at E8.
c5._load_config = _load_config
c5.__file__ = str(HERE / "calibrate.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Example 8 calibration")
    parser.add_argument("--container", type=str, default=None, help="Override container path")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config TOML (default 8_Soil_Moisture.toml; use 8_Soil_Moisture_c1.toml for WP-C1)",
    )
    parser.add_argument("--results-tag", default="e8cal", help="Results subdirectory name")
    parser.add_argument(
        "--etf-target-model",
        type=str,
        default=None,
        help="Override etf_target_model from TOML (e.g. 'ensemble')",
    )
    parser.add_argument(
        "--ensemble-source",
        type=str,
        default=None,
        help="Override ensemble_source from TOML (e.g. 'computed', 'openet')",
    )
    parser.add_argument(
        "--debug-fields",
        type=str,
        default=None,
        help="Comma-separated site IDs for debug subset (drops to 20 reals)",
    )
    parser.add_argument(
        "--keep-pestrun",
        action="store_true",
        help="Preserve pest/master/workers dirs after archiving (RUN_POLICY)",
    )
    args = parser.parse_args()

    if args.config is not None:
        global CONFIG_FILE
        CONFIG_FILE = Path(args.config).resolve()

    cfg = _load_config()
    if args.etf_target_model is not None:
        cfg.etf_target_model = args.etf_target_model
    if args.ensemble_source is not None:
        cfg.ensemble_source = args.ensemble_source

    debug_fields = None
    if args.debug_fields:
        debug_fields = [s.strip() for s in args.debug_fields.split(",")]

    results = os.path.join(cfg.project_ws, "results", args.results_tag)

    print(f"ETf target: {cfg.etf_target_model}")
    print(f"Ensemble source: {cfg.ensemble_source}")
    print(f"Results dir: {results}")

    t0 = time.time()
    c5.run_pest_sequence(
        cfg,
        results,
        pdc_remove=False,
        debug_fields=debug_fields,
        container_path=args.container,
        keep_pestrun=args.keep_pestrun,
    )
    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
