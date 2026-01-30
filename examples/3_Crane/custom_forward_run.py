"""Placeholder forward run script for PEST++ workers.

PestBuilder auto-generates the actual custom_forward_run.py in the pest/
directory during calibration. This file exists only as a fallback template.
"""

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    """Forward runner for PEST++ workers using the process package."""
    from swimrs.process.input import SwimInput
    from swimrs.process.loop_fast import run_daily_loop_fast
    from swimrs.process.state import CalibrationParameters, load_pest_mult_properties

    cwd = os.getcwd()
    h5_path = os.path.join(cwd, "swim_input.h5")
    mult_dir = os.path.join(cwd, "mult")
    pred_dir = os.path.join(cwd, "pred")
    os.makedirs(pred_dir, exist_ok=True)

    swim_input = SwimInput(h5_path=h5_path)

    params = CalibrationParameters.from_pest_mult(
        mult_dir=mult_dir, fids=swim_input.fids, base=swim_input.parameters
    )
    props = load_pest_mult_properties(
        mult_dir=mult_dir, fids=swim_input.fids, base_props=swim_input.properties
    )

    output, _ = run_daily_loop_fast(swim_input=swim_input, parameters=params, properties=props)

    # Write prediction files for PEST++
    import numpy as np

    for i, fid in enumerate(swim_input.fids):
        pred_file = os.path.join(pred_dir, f"{fid}.csv")
        np.savetxt(pred_file, output.etf[:, i], fmt="%.6f")


if __name__ == "__main__":
    run()
