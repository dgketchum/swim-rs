# Container Package (`swimrs.container`)

Zarr-backed `.swim` storage with ingest/compute/export/query components, schema definitions, and storage providers.

## Core entry points

::: swimrs.container.open_container
    options:
      show_source: false
      show_signature_annotations: true

::: swimrs.container.create_container
    options:
      show_source: false
      show_signature_annotations: true

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

## Components

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

## Schema and enums

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

::: swimrs.container.storage.StorageProvider
    options:
      show_source: false
      members: true
      show_signature_annotations: true
