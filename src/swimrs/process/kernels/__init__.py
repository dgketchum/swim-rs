"""
Physics kernels for SWIM-RS soil water balance modeling.

Design Rules:
1. Functions take numpy arrays or scalars as input
2. Functions return numpy arrays or scalars as output
3. No file I/O, no `self`, no state mutation
4. All physical constraints documented in docstrings
5. Numba JIT compiled with cache=True for performance
"""

from swimrs.process.kernels import (
    cover,
    crop_coefficient,
    evaporation,
    irrigation,
    root_growth,
    runoff,
    snow,
    transpiration,
    water_balance,
)

__all__ = [
    "crop_coefficient",
    "cover",
    "evaporation",
    "transpiration",
    "runoff",
    "snow",
    "water_balance",
    "root_growth",
    "irrigation",
]
