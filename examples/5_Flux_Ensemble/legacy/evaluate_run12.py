"""Evaluate Run 12 (ETo-corrected ETf) against v2.1 Volk flux data.

Thin wrapper around evaluate.py that writes all output CSVs to the
run12_eto_corrected results subdirectory instead of the top-level results/.

Usage:
    python evaluate_run12.py
"""

import os
import sys
from pathlib import Path

from evaluate import (
    evaluate,
    evaluate_etf,
    evaluate_monthly,
    find_par_csv,
    load_config,
    resolve_flux_dir,
)
from swimrs.container import SwimContainer

RESULTS_TAG = "run12_eto_corrected"
CONTAINER_NAME = "5_Flux_Ensemble_corrected.swim"


def main():
    cfg = load_config()
    flux_dir = resolve_flux_dir(cfg)

    results_dir = os.path.join(cfg.project_ws, "results", RESULTS_TAG)
    container_path = os.path.join(cfg.data_dir, CONTAINER_NAME)
    par_csv = find_par_csv(results_dir, cfg.project_name)

    if par_csv is None:
        sys.exit(f"No .par.csv found in {results_dir}")

    print(f"Parameters: {par_csv}")
    print(f"Container:  {container_path}")
    print(f"Output dir: {results_dir}")

    container = SwimContainer.open(container_path, mode="r")
    fids = container.field_uids

    try:
        # Daily ET vs Volk 3x3
        print("\n" + "#" * 80)
        print("  DAILY ET vs FLUX (Volk 3x3 ensemble)")
        print("#" * 80)
        daily_metrics = evaluate(cfg, container, par_csv, fids, flux_dir, openet_source="volk")
        daily_metrics.to_csv(os.path.join(results_dir, "evaluation_metrics.csv"))
        print(f"Saved: {results_dir}/evaluation_metrics.csv")

        # Monthly ET vs Volk 3x3
        print("\n" + "#" * 80)
        print("  MONTHLY ET vs FLUX (Volk 3x3 ensemble)")
        print("#" * 80)
        monthly_metrics = evaluate_monthly(cfg, container, par_csv, fids, flux_dir)
        monthly_metrics.to_csv(os.path.join(results_dir, "evaluation_monthly_metrics.csv"))
        print(f"Saved: {results_dir}/evaluation_monthly_metrics.csv")

        # ETf at capture dates
        print("\n" + "#" * 80)
        print("  ETf vs OpenET (at Landsat capture dates)")
        print("#" * 80)
        etf_metrics = evaluate_etf(cfg, container, par_csv, fids)
        etf_metrics.to_csv(os.path.join(results_dir, "evaluation_etf_metrics.csv"))
        print(f"Saved: {results_dir}/evaluation_etf_metrics.csv")

    finally:
        container.close()


if __name__ == "__main__":
    main()
