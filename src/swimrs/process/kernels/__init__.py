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
    crop_coefficient,
    cover,
    evaporation,
    transpiration,
    runoff,
    snow,
    water_balance,
    root_growth,
    irrigation,
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
