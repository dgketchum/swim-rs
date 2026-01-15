"""
SWIM-RS Process Package

Modernized soil water balance modeling with:
- Pure physics kernels (numba JIT)
- Typed state containers
- Portable HDF5 input format for PEST++ workers
- Structured logging
"""

from swimrs.process import kernels
from swimrs.process.state import (
    WaterBalanceState,
    FieldProperties,
    CalibrationParameters,
)
from swimrs.process.input import SwimInput, build_swim_input
from swimrs.process.loop import run_daily_loop, DailyOutput, step_day

__all__ = [
    "kernels",
    "WaterBalanceState",
    "FieldProperties",
    "CalibrationParameters",
    "SwimInput",
    "build_swim_input",
    "run_daily_loop",
    "DailyOutput",
    "step_day",
]
