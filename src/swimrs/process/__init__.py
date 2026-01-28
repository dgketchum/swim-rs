"""
SWIM-RS Process Package

Modernized soil water balance modeling with:
- Pure physics kernels (numba JIT)
- Typed state containers
- Portable HDF5 input format for PEST++ workers
- Structured logging
"""

from swimrs.process import kernels
from swimrs.process.input import SwimInput, build_swim_input
from swimrs.process.loop import DailyOutput, run_daily_loop, step_day
from swimrs.process.loop_fast import run_daily_loop_fast
from swimrs.process.state import (
    CalibrationParameters,
    FieldProperties,
    WaterBalanceState,
)

__all__ = [
    "kernels",
    "WaterBalanceState",
    "FieldProperties",
    "CalibrationParameters",
    "SwimInput",
    "build_swim_input",
    "run_daily_loop",
    "run_daily_loop_fast",
    "DailyOutput",
    "step_day",
]
