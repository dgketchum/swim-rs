# API Reference

This is the complete API reference for SWIM-RS (Soil Water Inverse Modeling with Remote Sensing).

## Quick Links

- [Container Package](#container) - Zarr-based data management
- [Process Package](#process) - Physics engine and daily loop
- [Swim Package](#swim) - Configuration and data containers
- [Calibrate Package](#calibrate) - PEST++ integration
- [Data Extraction](#data-extraction) - Earth Engine and GridMET

---

## Container

The container package provides unified data management using Zarr storage.

### Core Functions

::: swimrs.container.open_container
    options:
      show_source: false

::: swimrs.container.create_container
    options:
      show_source: false

### SwimContainer

::: swimrs.container.SwimContainer
    options:
      show_source: false
      members:
        - create
        - open
        - ingest
        - compute
        - export
        - query
        - close

### Components

::: swimrs.container.Ingestor
    options:
      show_source: false

::: swimrs.container.Calculator
    options:
      show_source: false

::: swimrs.container.Exporter
    options:
      show_source: false

::: swimrs.container.Query
    options:
      show_source: false

### Schema and Enums

::: swimrs.container.SwimSchema
    options:
      show_source: false

::: swimrs.container.Instrument
    options:
      show_source: false

::: swimrs.container.MaskType
    options:
      show_source: false

::: swimrs.container.ETModel
    options:
      show_source: false

---

## Process

The process package provides the physics engine for soil water balance modeling.

### Daily Loop

::: swimrs.process.run_daily_loop
    options:
      show_source: false

::: swimrs.process.step_day
    options:
      show_source: false

::: swimrs.process.DailyOutput
    options:
      show_source: false

### State Containers

::: swimrs.process.WaterBalanceState
    options:
      show_source: false

::: swimrs.process.FieldProperties
    options:
      show_source: false

::: swimrs.process.CalibrationParameters
    options:
      show_source: false

### Input Management

::: swimrs.process.SwimInput
    options:
      show_source: false

::: swimrs.process.build_swim_input
    options:
      show_source: false

---

## Swim

Configuration and data containers for SWIM-RS projects.

::: swimrs.swim.ProjectConfig
    options:
      show_source: false

::: swimrs.swim.SamplePlots
    options:
      show_source: false

::: swimrs.swim.ContainerPlots
    options:
      show_source: false

---

## Calibrate

PEST++ integration for parameter estimation and inverse modeling.

::: swimrs.calibrate.PestBuilder
    options:
      show_source: false

::: swimrs.calibrate.PestResults
    options:
      show_source: false

::: swimrs.calibrate.run_pst
    options:
      show_source: false

---

## Data Extraction

### Earth Engine

Functions for exporting ET fraction data from Google Earth Engine.

::: swimrs.data_extraction.ee.export_ptjpl_zonal_stats
    options:
      show_source: false

::: swimrs.data_extraction.ee.export_ssebop_zonal_stats
    options:
      show_source: false

::: swimrs.data_extraction.ee.export_sims_zonal_stats
    options:
      show_source: false

::: swimrs.data_extraction.ee.export_geesebal_zonal_stats
    options:
      show_source: false

### GridMET

::: swimrs.data_extraction.gridmet.GridMet
    options:
      show_source: false

---

## Deprecated Modules

!!! warning "Deprecated"
    These modules are deprecated and will be removed in a future version.
    See migration guides in each module's docstring.

### prep (DEPRECATED)

Use `swimrs.container` instead. See [Container Package](#container).

### model (LEGACY)

Superseded by `swimrs.process`. See [Process Package](#process).

### analysis (DEPRECATED)

Functionality moved to external analysis scripts.
