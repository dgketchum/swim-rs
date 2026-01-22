# ET-Demands vs SWIM-RS: Comprehensive Comparison

This document compares SWIM-RS with [ET-Demands](https://github.com/WSWUP/et-demands),
the legacy codebase from which SWIM-RS evolved. While SWIM-RS has been completely
rewritten, understanding the differences helps contextualize design decisions.

## Software Paradigmatic Differences

| Aspect | ET-Demands | SWIM-RS |
|--------|-----------|---------|
| **Execution Model** | Sequential per-crop, per-cell loops with optional multiprocessing | Vectorized NumPy operations with Numba JIT compilation |
| **Data Flow** | Pandas DataFrames, scalar computations inside `iterrows()` | NumPy arrays with shape `(n_fields,)` or `(n_days, n_fields)` |
| **Configuration** | Single INI file parsed at runtime | TOML config + HDF5 binary container |
| **State Management** | Mutable class attributes (`InitializeCropCycle`) | Immutable-style dataclasses with explicit `.copy()` |
| **Modularity** | Functions in `.py` files, no clear kernel separation | Explicit kernel layer (`kernels/*.py`) with pure physics functions |
| **Type Safety** | No type hints, relies on runtime duck typing | Full type annotations, `NDArray[np.float64]` throughout |
| **Dependencies** | Pandas-heavy, optional xlrd/shapefile support | NumPy/Numba core, HDF5 for I/O, minimal pandas |
| **Parallelism** | `multiprocessing.Pool` by cell or crop | Numba `prange` within kernels (thread-level) |

## Overall Software Approach

### ET-Demands: Crop-Centric Lookup Model

- **Philosophy**: Pre-defined crop coefficient curves (35-point tables) interpolated based on phenological progress
- **Kcb source**: Static crop parameter tables, selected by crop type ID
- **Calibration**: Manual parameter adjustment via INI/crop files
- **Target use**: Planning and regulatory compliance for specific crop types
- **Data paradigm**: Text files (CSV), one file per cell/crop combination

### SWIM-RS: Observation-Driven Inverse Model

- **Philosophy**: Remote sensing observations (NDVI→Kcb) drive coefficients; parameters calibrated via PEST++
- **Kcb source**: Daily NDVI time series, sigmoid transform with calibrated parameters
- **Calibration**: Automated parameter estimation using pyemu/PEST++
- **Target use**: Spatially distributed ET estimation without crop-type assumptions
- **Data paradigm**: HDF5 containers with all fields/days in single binary file

## Algorithmic Differences

### Crop Coefficient (Kcb)

| ET-Demands | SWIM-RS |
|-----------|---------|
| 35-point lookup table per crop type | Sigmoid function: `Kcb = Kc_max / (1 + exp(-k*(NDVI - ndvi_0)))` |
| Interpolated by GDD, %PL-EC, or DOY | Computed directly from daily NDVI |
| Season detection via T30/GDD thresholds | No explicit season detection (NDVI handles it) |
| Crop-specific curves (alfalfa, corn, etc.) | Generic sigmoid, calibrated per field |

### Evaporation Model

| ET-Demands | SWIM-RS |
|-----------|---------|
| Two-area model: irrigation-wetted vs precip-wetted | Single exposed area: `few = 1 - fc` |
| Three-stage drying (REW→TEW→TEW3) | Two-stage drying (REW→TEW) |
| Separate `ke_irr` and `ke_ppt` | Single `Ke` coefficient |
| `Kr2` parameter for cracking soils | No stage-3 drying |

### Stress Coefficient (Ks)

| ET-Demands | SWIM-RS |
|-----------|---------|
| Binary: `Ks = max((TAW - Dr) / (TAW - RAW), 0)` | Same formula with damping |
| `invoke_stress` flag can disable | Always active |
| Unrecoverable stress flag (perennials) | No unrecoverable stress |
| No damping (instantaneous response) | `ks_damp` parameter smooths response |

### Runoff

| ET-Demands | SWIM-RS |
|-----------|---------|
| SCS Curve Number only | CN method (default) or Infiltration-Excess (IER) |
| ANT moisture: dry (I), avg (II), wet (III) | CN adjusted by surface layer depletion |
| 4-day S averaging for irrigated fields | Same 4-day averaging for irrigated |
| No hourly precipitation support | Hourly precip for IER mode |

### Irrigation Logic

| ET-Demands | SWIM-RS |
|-----------|---------|
| Trigger: `depl > MAD * TAW` | Same |
| Requires `Kcb > 0.22` | No Kcb threshold |
| Multiple types: auto, real, manual, special | Single simulated irrigation |
| `days_after_planting_irrigation` delay | Irrigation flag schedule from observations |
| Irrigation fraction (`fw`) for partial wetting | No partial wetting model |

### Deep Percolation

| ET-Demands | SWIM-RS |
|-----------|---------|
| Two-phase: Ze layer + root zone | Single-phase from root zone |
| 20mm storage above FC when recently wet | Immediate percolation when Dr < 0 |
| Conditional on 2-day dry window | No dry window requirement |
| Feeds lower layer (`aw3`) | Same (`daw3`) |
| 10% irrigation bypass to aw3 | Same 10% bypass |

### Snow Dynamics

| ET-Demands | SWIM-RS |
|-----------|---------|
| Albedo-based Kc_mult reduction | Degree-day + radiation melt model |
| Simple: `Kc_mult = f(albedo, DOY)` | Full SWE tracking with decay |
| No SWE state variable | `swe`, `albedo` state variables |
| No melt calculation | Melt computed and added to infiltration |

### Root Growth

| ET-Demands | SWIM-RS |
|-----------|---------|
| Sigmoidal function of days-after-planting | Linear function of Kcb |
| Borg & Grimes (1986) formula | `zr = zr_min + (zr_max - zr_min) * f(Kcb)` |
| GDD-based progress option | Kcb-based progress |
| Perennial: constant at max | Perennial: constant at max (same) |

### Features Unique to ET-Demands

- **Cutting cycles** for alfalfa/forage crops
- **Frost-kill termination** for winter crops
- **Aridity temperature adjustments**
- **CO2 correction factors**
- **Historical phenology anchoring** (±40 day constraint on season start)

### Features Unique to SWIM-RS

- **Irrigation fraction tracking** (`et_irr`, `dperc_irr`) for water rights accounting
- **Damped stress coefficients** (`kr_damp`, `ks_damp`) for smooth temporal response
- **Year-specific groundwater subsidy** (`f_sub` by year)
- **PEST++ integration** for automated parameter estimation
- **Numba-compiled kernels** for performance

## Testing Comparison

### ET-Demands

Virtually no automated tests. The only test file contains a placeholder:

```python
def test():
    assert True
```

Testing relies on:
- Integration testing via example runs
- Visual inspection of output files
- Comparison with legacy VB6 code output

### SWIM-RS

Comprehensive test suite:
- 60+ kernel unit tests
- State initialization and copy tests
- Conservation law verification
- Property-based bounds testing
- pytest framework with CI integration

