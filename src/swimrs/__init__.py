"""
SWIM-RS: Soil Water Inverse Modeling with Remote Sensing.

A Python framework for inverse modeling of soil water parameters using
remote sensing observations. Integrates with OpenET algorithms for
evapotranspiration estimation and uses PEST/pyemu for parameter estimation.

Subpackages:
    swim: Core configuration and data containers for model runs.
    container: Unified Zarr-based data storage with component API.
    calibrate: PEST-based parameter estimation and calibration.
    data_extraction: Tools for retrieving data from Earth Engine and GridMET.
    process: Field-scale water balance simulation kernels.
    viz: Visualization utilities for model outputs.

Example:
    >>> from swimrs.container import SwimContainer
    >>> from swimrs.process.input import build_swim_input
    >>> from swimrs.process.loop import run_daily_loop
    >>>
    >>> # Load project data from container
    >>> container = SwimContainer.open("project.swim")
    >>> swim_input = build_swim_input(container, "swim_input.h5")
    >>>
    >>> # Run simulation
    >>> output, final_state = run_daily_loop(swim_input)
"""

__version__ = "0.1.0"

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
