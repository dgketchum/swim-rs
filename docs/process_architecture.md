# Process Package Architecture

The `swimrs.process` package provides a restructured, high-performance implementation of the SWIM-RS water balance model using typed dataclasses and Numba-compiled kernels.

## Package Structure

```
src/swimrs/process/
├── __init__.py
├── input.py          # SwimInput: HDF5-backed data container
├── state.py          # Dataclasses: WaterBalanceState, FieldProperties, CalibrationParameters
├── loop.py           # Daily simulation loop orchestration
└── kernels/          # Numba-compiled physics functions
    ├── cover.py
    ├── crop_coefficient.py
    ├── evaporation.py
    ├── irrigation.py
    ├── root_growth.py
    ├── runoff.py
    ├── snow.py
    ├── transpiration.py
    └── water_balance.py
```

---

## Core Dataclasses

The process package uses five main dataclasses to organize simulation data.
All array attributes have shape `(n_fields,)` unless otherwise noted.

### SwimInput

The top-level container that packages everything needed for a simulation run.
It wraps an HDF5 file and provides lazy access to time series data. We build
it once from `prepped_input.json` and distribute the resulting `.h5` file to
PEST++ workers. It holds references to the three dataclasses below plus
methods to retrieve daily forcing data by index.

### FieldProperties

Static soil and crop properties that do not change during simulation. These
come from soil surveys (AWC, Ksat), land cover databases (root depth, crop
type), and observation-derived values (ke_max from bare-soil ETf, f_sub from
ET/precipitation ratios). Examples: available water capacity, curve number,
maximum root depth, irrigation status, perennial flag.

### CalibrationParameters

Parameters that PEST++ adjusts during inverse modeling. These control the
NDVI-to-Kcb relationship (ndvi_k, ndvi_0), snow melt rates (swe_alpha,
swe_beta), stress response damping (ks_damp, kr_damp), and irrigation
behavior (max_irr_rate). The `from_base_with_multipliers()` method applies
PEST++ multiplier files to base values.

### WaterBalanceState

Mutable state that evolves each day. The simulation loop reads and writes
these arrays in place. Key variables: root zone depletion (depl_root),
surface layer depletion (depl_ze), snow water equivalent (swe), current
root depth (zr), and the damped stress coefficients (ks, kr). Initialized
from spinup values at simulation start.

### DailyOutput

Accumulator for simulation results. Shape is `(n_days, n_fields)` for all
arrays. Stores actual ET, crop coefficients, runoff, irrigation, and other
diagnostics. Created by `run_daily_loop()` and populated incrementally as
the simulation advances through each day.

---

## Data Flow Diagram

Shows how data moves from input files through the simulation loop to outputs.

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        JSON["prepped_input.json"]
        HDF5["project.h5"]
        JSON --> |build_swim_input| HDF5
    end

    subgraph Container["SwimInput Container"]
        HDF5 --> Props["FieldProperties<br/>(static soil/crop)"]
        HDF5 --> Params["CalibrationParameters<br/>(PEST++ tunable)"]
        HDF5 --> Spinup["WaterBalanceState<br/>(initial conditions)"]
        HDF5 --> TS["Time Series<br/>(ndvi, prcp, tmin, tmax, srad, etr)"]
    end

    subgraph Loop["Daily Loop (loop.py)"]
        direction TB
        State["WaterBalanceState<br/>(mutable)"]
        StepDay["step_day()"]
        Output["DailyOutput<br/>(eta, etf, kcb, ke, ks, kr, ...)"]
    end

    subgraph Kernels["Numba Kernels"]
        direction LR
        Snow["snow.py<br/>partition_precip<br/>albedo_decay<br/>degree_day_melt<br/>snow_water_equivalent"]
        Runoff["runoff.py<br/>scs_runoff<br/>infiltration_excess"]
        Crop["crop_coefficient.py<br/>kcb_sigmoid"]
        Cover["cover.py<br/>fractional_cover<br/>exposed_soil_fraction"]
        Root["root_growth.py<br/>root_depth_from_kcb<br/>root_water_redistribution"]
        Evap["evaporation.py<br/>kr_reduction<br/>kr_damped<br/>ke_coefficient"]
        Trans["transpiration.py<br/>ks_stress<br/>ks_damped"]
        Irr["irrigation.py<br/>irrigation_demand<br/>groundwater_subsidy"]
        WB["water_balance.py<br/>deep_percolation<br/>layer3_storage<br/>actual_et"]
    end

    Props --> StepDay
    Params --> StepDay
    Spinup --> State
    TS --> |"day_idx"| StepDay
    State <--> StepDay
    StepDay --> Output
    StepDay --> Kernels
```

---

## Class/Function Relationships

Shows the dataclass structure and how modules interact.

```mermaid
classDiagram
    class SwimInput {
        +Path h5_path
        +datetime start_date, end_date
        +int n_days, n_fields
        +list fids
        +str runoff_process, refet_type
        +FieldProperties properties
        +CalibrationParameters parameters
        +WaterBalanceState spinup_state
        +get_time_series(variable, day_idx)
        +get_irr_flag(day_idx)
        +close()
    }

    class FieldProperties {
        +int n_fields
        +NDArray fids, awc, ksat
        +NDArray rew, tew, cn2
        +NDArray zr_max, zr_min, p_depletion
        +NDArray~bool~ irr_status, perennial, gw_status
        +NDArray ke_max, f_sub
        +compute_taw(zr) NDArray
        +compute_raw(taw) NDArray
    }

    class CalibrationParameters {
        +int n_fields
        +NDArray kc_max, kc_min
        +NDArray ndvi_k, ndvi_0
        +NDArray swe_alpha, swe_beta
        +NDArray kr_damp, ks_damp
        +NDArray max_irr_rate
        +from_base_with_multipliers(base, multipliers)$ CalibrationParameters
        +copy() CalibrationParameters
    }

    class WaterBalanceState {
        +int n_fields
        +NDArray depl_root, depl_ze
        +NDArray daw3, taw3
        +NDArray swe, albedo
        +NDArray zr, kr, ks
        +NDArray irr_continue, next_day_irr
        +from_spinup(...)$ WaterBalanceState
        +copy() WaterBalanceState
    }

    class DailyOutput {
        +int n_days, n_fields
        +NDArray eta, etf, kcb, ke
        +NDArray ks, kr, runoff
        +NDArray rain, melt, swe
        +NDArray depl_root, dperc
        +NDArray irr_sim, gw_sim
    }

    SwimInput *-- FieldProperties
    SwimInput *-- CalibrationParameters
    SwimInput *-- WaterBalanceState

    class loop_py {
        <<module>>
        +run_daily_loop(swim_input, params) tuple
        +step_day(state, props, params, ...) dict
    }

    loop_py ..> SwimInput : reads
    loop_py ..> WaterBalanceState : mutates
    loop_py ..> DailyOutput : creates
    loop_py ..> FieldProperties : uses
    loop_py ..> CalibrationParameters : uses
```

---

## Kernel Call Sequence

Shows the order of kernel function calls within a single `step_day()` execution.

```mermaid
sequenceDiagram
    participant L as loop.step_day
    participant S as snow
    participant R as runoff
    participant C as crop_coefficient
    participant CV as cover
    participant RG as root_growth
    participant E as evaporation
    participant T as transpiration
    participant I as irrigation
    participant W as water_balance

    L->>S: partition_precip(prcp, temp)
    L->>S: albedo_decay(albedo, snow)
    L->>S: degree_day_melt(swe, tmax, ...)
    L->>S: snow_water_equivalent(swe, snow, melt)
    L->>R: scs_runoff(precip_eff, cn2)
    L->>C: kcb_sigmoid(ndvi, kc_max, ...)
    L->>CV: fractional_cover(kcb, kc_min, kc_max)
    L->>CV: exposed_soil_fraction(fc)
    L->>RG: root_depth_from_kcb(kcb, ...)
    L->>RG: root_water_redistribution(zr_new, zr_prev, ...)
    L->>E: kr_reduction(tew, depl_ze, rew)
    L->>T: ks_stress(taw, depl_root, raw)
    L->>E: kr_damped(kr_base, kr_prev, damp)
    L->>T: ks_damped(ks_base, ks_prev, damp)
    L->>E: ke_coefficient(kr, kc_max, kcb, few, ke_max)
    L->>W: actual_et(ks, kcb, fc, ke, kc_max, etr)
    L->>I: irrigation_demand(depl, raw, max_rate, ...)
    L->>I: groundwater_subsidy(depl, raw, gw_status, f_sub)
    L->>W: deep_percolation(depl_new)
    L->>W: layer3_storage(daw3, taw3, gross_dperc)
```

---

## Key Design Decisions

### Typed Dataclasses
All state and parameter containers use Python dataclasses with explicit NumPy array types. This provides:
- Clear documentation of array shapes and meanings
- IDE autocompletion and type checking
- Easy serialization to/from HDF5

### HDF5 Data Container
`SwimInput` wraps an HDF5 file for:
- Portable distribution to PEST++ workers
- Lazy loading of large time series arrays
- Single-file packaging of all simulation inputs

### Numba Kernels
All physics computations are in separate kernel modules using `@njit` decorators:
- Parallel execution across fields with `parallel=True`
- Cache compilation with `cache=True`
- Pure functions with no side effects (except state mutation in loop.py)

### Separation of Concerns
- **input.py**: Data I/O and HDF5 management
- **state.py**: Data structure definitions only
- **loop.py**: Orchestration logic
- **kernels/**: Physics implementations

This separation allows kernels to be tested independently and potentially reused in other contexts.
