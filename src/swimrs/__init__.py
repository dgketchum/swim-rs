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
    >>> from swimrs.swim import ProjectConfig, SamplePlots
    >>> from swimrs.calibrate import PestBuilder, run_pst
    >>>
    >>> # Load project configuration
    >>> config = ProjectConfig.from_toml("project.toml")
    >>> plots = SamplePlots.from_json("prepped_input.json")
    >>>
    >>> # Build and run calibration
    >>> builder = PestBuilder(config, plots)
    >>> pst = builder.build()
    >>> run_pst(pst, num_workers=4)
"""

__version__ = "0.1.0"

if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
