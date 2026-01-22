# SWIM-RS Algorithm Description

## Overview

SWIM-RS (**S**oil **W**ater **I**nverse **M**odeling with **R**emote
**S**ensing) implements a FAO-56 dual crop coefficient soil water balance
model driven by remote sensing observations. The model simulates daily
evapotranspiration, soil moisture dynamics, irrigation, and groundwater
interactions for agricultural fields. We use NDVI time series from satellite
imagery to estimate crop coefficients dynamically, enabling spatially
distributed ET estimation without crop-specific parameterization.

A complex architecture surrounds the model: we use special extraction and
data storage objects to help organize the data and relieve the user of dealing
with the many files that are required to run the model, see [container architecture](container_architecture.md) 
for a description of the SwimContainer.

We use special water balance state, properties, calibrated parameter, and input 
data objects to organize information required to run the algorithm. This modular approach 
makes all components testable, enabling new features. See [process architecture](process_architecture.md)
for a description of SwimInput, FieldProperties, CalibrationParameters, and WaterBalanceState classes.

We've made a considerable investment to design the software in such a way that it is testable and usable
by non-developers. The architecture, while complex, will allow easier integration of expected 
future features, such as specialized irrigation scheduling and simulation, different snow, runoff,
and soil water models, and new proxies for ET and Kcb.

While this software has been completely rewritten, it was originally based on a fork
of [et-demands](https://github.com/WSWUP/et-demands); shoutout to Dr. Richard Allen, Chris Pearson, Charles Morton, Blake Minor,
Thomas Ott, Dr. Justin Huntington, and others who've contributed to that project over the years.
For a detailed comparison of the two codebases, see [ET-Demands Comparison](et-demands-compare.md).

## Model Inputs

### Meteorological Forcing

- **Reference ET (ETr or ETo)**: Alfalfa or grass reference
  evapotranspiration (mm/day)
- **Precipitation (prcp)**: Daily total precipitation (mm)
- **Temperature (tmin, tmax)**: Daily minimum and maximum air
  temperature (°C)
- **Solar radiation (srad)**: Incoming shortwave radiation (MJ/m²/day)

### Remote Sensing Observations

- **NDVI time series**: Normalized Difference Vegetation Index,
  interpolated to daily values

### Soil Properties

- **AWC**: Available water capacity (mm/m)
- **CN2**: Curve number for average antecedent moisture condition
- **Ksat**: Saturated hydraulic conductivity (mm/hr), used for
  infiltration-excess runoff
- **REW**: Readily evaporable water in surface layer (mm)
- **TEW**: Total evaporable water in surface layer (mm)
- **zr_max**: Maximum root depth (m)

### Management and Field Properties

- **Irrigation schedule flags**: Boolean flags indicating irrigation days
- **Irrigation status**: Whether field is irrigated
- **Perennial status**: Whether crop has persistent root system
- **Groundwater status**: Whether groundwater subsidy is available

## Daily Iteration Structure

We iterate through each day in the simulation period. For each day, we
perform the following calculations in sequence, updating the water balance
state.

## Algorithm Steps

### Step 1: Basal Crop Coefficient from NDVI

We compute the basal crop coefficient (Kcb) from NDVI using a sigmoid
relationship:

```
Kcb = Kc_max / (1 + exp(-ndvi_k × (NDVI - ndvi_0)))
```

- Higher NDVI indicates greater vegetation vigor and higher transpiration
  potential
- The inflection point `ndvi_0` controls where the transition from bare to
  vegetated occurs
- The steepness `ndvi_k` controls how sharp that transition is
- At `NDVI = ndvi_0`, `Kcb = Kc_max / 2`

**Parameters**: `ndvi_k` (slope), `ndvi_0` (inflection point),
`kc_max` (ceiling)

### Step 2: Snow Dynamics

We partition precipitation into rain or snow and compute snowmelt:

**Partitioning**:
- If T_avg < 1°C: precipitation falls as snow
- If T_avg ≥ 1°C: precipitation falls as rain

**Albedo evolution**:
- Fresh snowfall (> 3 mm) resets albedo to 0.98
- Albedo decays exponentially between snowfall events
- Minimum albedo for old snow: 0.45

**Snowmelt** (combined degree-day and radiation approach):

```
melt = (1 - albedo) × srad × swe_alpha + max(T_avg - 1.8, 0) × swe_beta
```

- Melt is bounded by available SWE
- No melt occurs when T_max ≤ 0°C

**Parameters**: `swe_alpha` (radiation melt coefficient),
`swe_beta` (degree-day factor)

### Step 3: Runoff Calculation

Two methods are available, selected via `runoff_process`:

**Curve Number (CN) Method** (`runoff_process = "cn"`):

We adjust CN for antecedent moisture based on surface layer depletion:
- Wet conditions (low depletion): CN → CN_III (higher runoff)
- Dry conditions (high depletion): CN → CN_I (lower runoff)

Runoff is calculated using the SCS equation:

```
S = 250 × (100/CN - 1)
Ia = 0.2 × S
Q = (P - Ia)² / (P - Ia + S)  when P > Ia
```

For irrigated fields, we smooth runoff using a 4-day average of S values
to reduce irrigation-induced variability.

**Infiltration-Excess Method** (`runoff_process = "ier"`):

Uses hourly precipitation vs. infiltration capacity:

```
Q = Σ max(P_hr - Ksat_hr, 0)
```

**Infiltrating precipitation** = rain + snowmelt - runoff

### Step 4: Fractional Cover and Exposed Soil

We compute fractional vegetation cover from Kcb:

```
fc = (Kcb - Kc_min) / (Kc_max - Kc_min)
few = 1 - fc
```

- `fc` is bounded [0, 0.99] to ensure exposed soil fraction remains
  positive
- `few` represents the fraction of soil available for evaporation

### Step 5: Root Growth

For **annual crops**, root depth varies with crop vigor:

```
zr = zr_min + (zr_max - zr_min) × (Kcb - Kc_min) / (Kc_max - Kc_min)
```

For **perennial crops**, root depth remains constant at `zr_max`.

When roots grow or shrink, we redistribute water between the root zone
and layer 3 to conserve mass.

### Step 6: Surface Evaporation (Kr and Ke)

**Evaporation reduction coefficient (Kr)**:

```
Kr = (TEW - De) / (TEW - REW)  when De > REW
Kr = 1                          when De ≤ REW
```

We apply damping to smooth day-to-day transitions:

```
Kr_new = Kr_prev + kr_damp × (Kr_current - Kr_prev)
```

**Evaporation coefficient (Ke)**:

```
Ke = min(Kr × (Kc_max - Kcb), few × Kc_max)
Ke = min(Ke, Ke_max)
```

The dual constraint ensures evaporation doesn't exceed available energy
or exposed soil area.

**Parameters**: `kr_damp` (damping factor), `ke_max` (maximum soil
evaporation)

### Step 7: Root Zone Stress Coefficient (Ks)

When root zone depletion exceeds RAW, transpiration stress begins:

```
Ks = (TAW - Dr) / (TAW - RAW)  when Dr > RAW
Ks = 1                          when Dr ≤ RAW
```

Where:
- `TAW = AWC × zr` (total available water)
- `RAW = MAD × TAW` (readily available water)
- `MAD` is the management allowed depletion fraction

We apply damping for smooth response:

```
Ks_new = Ks_prev + ks_damp × (Ks_current - Ks_prev)
```

**Parameters**: `ks_damp` (damping factor), `p_depletion` (MAD value)

### Step 8: Actual Evapotranspiration

We calculate actual ET from the stress and evaporation coefficients:

```
Kc_act = min(Ks × Kcb × fc + Ke, Kc_max)
ETc_act = Kc_act × ETref
```

Where:
- Transpiration component: `T = Ks × Kcb × fc × ETref`
- Evaporation component: `E = Ke × ETref`

We update root zone depletion:

```
depl_root = depl_root + ETc_act - infiltrating_precip
```

### Step 9: Irrigation Logic

Irrigation is triggered when ALL conditions are met:

1. It's an irrigation day (from schedule)
2. Root zone depletion > RAW
3. Average temperature ≥ 5°C

We apply irrigation:

```
irr_sim = min(max_irr_rate, depl_root × 1.1)
```

If depletion exceeds daily capacity, irrigation continues on subsequent
days until the deficit is satisfied.

**Parameters**: `max_irr_rate` (maximum daily irrigation, mm/day)

### Step 10: Groundwater Subsidy

For fields with shallow water tables (`gw_status = True` and
`f_sub > 0.2`):

When `depl_root > RAW`:

```
gw_sim = depl_root - RAW
```

This represents capillary rise filling the root zone back to the RAW
threshold.

**Parameters**: `f_sub` (subsidy fraction, derived from observations)

### Step 11: Deep Percolation

If `depl_root < 0` after all inputs (excess water):

```
dperc = -depl_root
depl_root = 0
```

The excess water drains to layer 3 (below-root storage):

```
gross_dperc = dperc + 0.1 × irr_sim
```

Layer 3 stores water up to its capacity (`taw3 = AWC × (zr_max - zr)`).
When full, excess drains as deep percolation leaving the system.

## Three-Layer Soil Model

The model tracks water in three conceptual layers:

| Layer      | Symbol  | Description                      | Depth          |
|------------|---------|----------------------------------|----------------|
| Surface    | Ze      | Evaporation source layer         | ~0.1-0.15 m    |
| Root zone  | zr      | Transpiration source, dynamic    | 0.1 to zr_max  |
| Below-root | Layer 3 | Storage reservoir below roots    | zr_max - zr    |

- **Surface layer**: Controls evaporation availability via `depl_ze`
- **Root zone**: Primary transpiration source, depth varies with crop
  growth
- **Below-root zone**: Water reservoir that roots can access as they
  grow deeper

## Key State Variables

| Variable    | Description                              | Units |
|-------------|------------------------------------------|-------|
| `depl_root` | Root zone depletion (0 = field capacity) | mm    |
| `depl_ze`   | Surface layer depletion                  | mm    |
| `daw3`      | Available water in layer 3               | mm    |
| `swe`       | Snow water equivalent                    | mm    |
| `zr`        | Current root depth                       | m     |
| `ks`        | Water stress coefficient (damped)        | -     |
| `kr`        | Evaporation reduction coefficient        | -     |
| `albedo`    | Snow albedo                              | -     |

## Tunable Coefficients

### Calibration Parameters (adjustable via PEST++)

| Parameter      | Description                      | Typical Range |
|----------------|----------------------------------|---------------|
| `ndvi_k`       | Sigmoid steepness for NDVI→Kcb   | 4-10          |
| `ndvi_0`       | Sigmoid inflection point NDVI    | 0.1-0.7       |
| `swe_alpha`    | Radiation melt coefficient       | -0.5-1.0      |
| `swe_beta`     | Degree-day melt factor           | 0.5-2.5       |
| `kr_damp`      | Kr damping factor                | 0.1-0.5       |
| `ks_damp`      | Ks damping factor                | 0.1-0.5       |
| `max_irr_rate` | Maximum irrigation rate          | 15-40 mm/day  |

### Field Properties (derived from data)

| Property            | Description                  | Source                      |
|---------------------|------------------------------|-----------------------------|
| `awc`               | Available water capacity     | Soil surveys (gSSURGO)      |
| `p_depletion` (MAD) | Management allowed depletion | Calibration/literature      |
| `ke_max`            | Maximum Ke                   | 90th %ile ETf, NDVI < 0.3   |
| `f_sub`             | Groundwater subsidy fraction | ETa/PPT ratio analysis      |
| `zr_max`            | Maximum root depth           | Land cover type             |

### MAD Parameter: Dual Role

The MAD (management allowed depletion, stored as `p_depletion`) serves
two purposes:

1. **Irrigation trigger threshold**: Irrigation is applied when
   `depl_root > RAW` (= MAD × TAW)
2. **Stress onset threshold**: Ks begins declining when
   `depl_root > RAW`

**Implications**:
- Higher MAD → irrigation triggered later → less frequent irrigation
- Higher MAD → stress begins later → less stress-induced ET reduction
- For rainfed fields, MAD controls only stress sensitivity

This coupling allows MAD to act as a proxy for both management practices
and crop stress tolerance.

## Model Outputs

### Daily Arrays (shape: n_days × n_fields)

| Output      | Description                      | Units   |
|-------------|----------------------------------|---------|
| `eta`       | Actual evapotranspiration        | mm/day  |
| `etf`       | ET fraction (ETa/ETref)          | -       |
| `kcb`       | Basal crop coefficient           | -       |
| `ke`        | Evaporation coefficient          | -       |
| `ks`        | Water stress coefficient         | -       |
| `kr`        | Evaporation reduction coefficient| -       |
| `runoff`    | Surface runoff                   | mm      |
| `rain`      | Liquid precipitation             | mm      |
| `melt`      | Snowmelt                         | mm      |
| `swe`       | Snow water equivalent            | mm      |
| `depl_root` | Root zone depletion              | mm      |
| `dperc`     | Deep percolation                 | mm      |
| `irr_sim`   | Simulated irrigation             | mm      |
| `gw_sim`    | Groundwater subsidy              | mm      |

## Source Code Reference

| File                        | Section Coverage                        |
|-----------------------------|-----------------------------------------|
| `loop.py`                   | Daily iteration, kernel orchestration   |
| `kernels/crop_coefficient.py` | Step 1 (NDVI → Kcb sigmoid)           |
| `kernels/snow.py`           | Step 2 (Snow partitioning, albedo, melt)|
| `kernels/runoff.py`         | Step 3 (CN and infiltration-excess)     |
| `kernels/cover.py`          | Step 4 (Fractional cover, few)          |
| `kernels/evaporation.py`    | Step 6 (Kr, Ke coefficients)            |
| `kernels/transpiration.py`  | Step 7 (Ks stress coefficient)          |
| `kernels/water_balance.py`  | Step 8 (Actual ET, deep percolation)    |
| `kernels/irrigation.py`     | Steps 9-10 (Irrigation, groundwater)    |
| `kernels/root_growth.py`    | Step 5, 11 (Root depth, redistribution) |
| `state.py`                  | State variable containers               |
| `input.py`                  | Input data structures (HDF5 container)  |

## References

- Allen, R.G., et al. (1998). FAO Irrigation and Drainage Paper 56:
  Crop Evapotranspiration.
- USDA-SCS (1972). National Engineering Handbook, Section 4: Hydrology.
- US Army Corps of Engineers (1956). Snow Hydrology.
