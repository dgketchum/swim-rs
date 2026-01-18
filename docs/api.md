# API Reference

This is the complete API reference for SWIM-RS (Soil Water Inverse Modeling with Remote Sensing).

## Quick Links

- [Container Package](#container) - Zarr-based data management
- [Process Package](#process) - Physics engine and daily loop
- [Swim Package](#swim) - Configuration and data containers
- [TOML Requirements](#toml-requirements) - Minimal/optional project config template
- [Calibrate Package](#calibrate) - PEST++ integration
- [Data Extraction](#data-extraction) - Earth Engine and GridMET

---

## Container

The container package provides unified data management using Zarr storage.

### Core Functions

::: swimrs.container.open_container
    options:
      show_source: false
      show_signature_annotations: true

::: swimrs.container.create_container
    options:
      show_source: false
      show_signature_annotations: true

### SwimContainer

::: swimrs.container.SwimContainer
    options:
      show_source: false
      show_signature_annotations: true
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
      members: true
      show_signature_annotations: true

::: swimrs.container.Calculator
    options:
      show_source: false
      members: true
      show_signature_annotations: true

::: swimrs.container.Exporter
    options:
      show_source: false
      members: true
      show_signature_annotations: true

::: swimrs.container.Query
    options:
      show_source: false
      members: true
      show_signature_annotations: true

### Schema and Enums

::: swimrs.container.SwimSchema
    options:
      show_source: false
      members: true

::: swimrs.container.Instrument
    options:
      show_source: false
      members: true

::: swimrs.container.MaskType
    options:
      show_source: false
      members: true

::: swimrs.container.ETModel
    options:
      show_source: false
      members: true

---

## Process

The process package provides the physics engine for soil water balance modeling.

### Daily Loop

::: swimrs.process.run_daily_loop
    options:
      show_source: false
      show_signature_annotations: true

::: swimrs.process.step_day
    options:
      show_source: false
      show_signature_annotations: true

::: swimrs.process.DailyOutput
    options:
      show_source: false
      members: true
      show_signature_annotations: true

### State Containers

::: swimrs.process.WaterBalanceState
    options:
      show_source: false
      members: true
      show_signature_annotations: true
      merge_init_into_class: true

::: swimrs.process.FieldProperties
    options:
      show_source: false
      members: true
      show_signature_annotations: true
      merge_init_into_class: true

::: swimrs.process.CalibrationParameters
    options:
      show_source: false
      members: true
      show_signature_annotations: true
      merge_init_into_class: true

### Input Management

::: swimrs.process.SwimInput
    options:
      show_source: false
      members: true
      show_signature_annotations: true
      merge_init_into_class: true

::: swimrs.process.build_swim_input
    options:
      show_source: false
      show_signature_annotations: true

---

## Swim

Configuration and data containers for SWIM-RS projects.

::: swimrs.swim.ProjectConfig
    options:
      show_source: false
      members: true
      show_signature_annotations: true

::: swimrs.swim.SamplePlots
    options:
      show_source: false
      members: true
      show_signature_annotations: true

::: swimrs.swim.ContainerPlots
    options:
      show_source: false
      members: true
      show_signature_annotations: true

---

## TOML Requirements

See the minimal and optional project configuration template (Fort Peck example) in [docs/toml_requirements.md](toml_requirements.md). It outlines required keys and optional settings such as GridMET/ERA5 met sources, snow source selection, GridMET mapping/corrections, and legacy `prepped_input` export.

---

## Calibrate

PEST++ integration for parameter estimation and inverse modeling.

::: swimrs.calibrate.PestBuilder
    options:
      show_source: false
      members: true
      show_signature_annotations: true

::: swimrs.calibrate.PestResults
    options:
      show_source: false
      members: true
      show_signature_annotations: true

::: swimrs.calibrate.run_pst
    options:
      show_source: false
      show_signature_annotations: true

---

## Data Extraction

### Earth Engine

Functions for exporting ET fraction data from Google Earth Engine.

::: swimrs.data_extraction.ee.export_ptjpl_zonal_stats
    options:
      show_source: false
      show_signature_annotations: true

::: swimrs.data_extraction.ee.export_ssebop_zonal_stats
    options:
      show_source: false
      show_signature_annotations: true

::: swimrs.data_extraction.ee.export_sims_zonal_stats
    options:
      show_source: false
      show_signature_annotations: true

::: swimrs.data_extraction.ee.export_geesebal_zonal_stats
    options:
      show_source: false
      show_signature_annotations: true

### GridMET

::: swimrs.data_extraction.gridmet.GridMet
    options:
      show_source: false
      members: true
      show_signature_annotations: true

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
