"""
PEST-based calibration module for SWIM-RS.

Provides tools for building PEST control files, running parameter estimation,
and processing calibration results using pyemu.

Key Classes:
    PestBuilder: Constructs PEST control files from SWIM-RS configuration.
    PestResults: Parses and analyzes calibration output files.

Key Functions:
    run_pst: Execute PEST calibration with parallel workers.

Example:
    >>> from swimrs.calibrate import PestBuilder, PestResults, run_pst
    >>>
    >>> # Build PEST control file
    >>> builder = PestBuilder(config, plots)
    >>> pst = builder.build()
    >>>
    >>> # Run calibration
    >>> run_pst(pst, num_workers=4)
    >>>
    >>> # Analyze results
    >>> results = PestResults("pest_output/")
    >>> best_params = results.get_best_parameters()
"""

from swimrs.calibrate.pest_builder import PestBuilder
from swimrs.calibrate.pest_cleanup import PestResults
from swimrs.calibrate.run_pest import run_pst

__all__ = ["PestBuilder", "PestResults", "run_pst"]
# ========================= EOF ====================================================================
