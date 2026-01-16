# SwimContainer Architecture

The `swimrs.container` package provides a Zarr-based data container that unifies all SWIM-RS project data in a single `.swim` file. It replaces scattered CSV/JSON/Parquet files with a self-describing archive featuring full provenance tracking.

## Package Structure

```
src/swimrs/container/
├── __init__.py           # Public API exports
├── container.py          # SwimContainer main class
├── state.py              # ContainerState: centralized state + xarray interface
├── schema.py             # Data schema definitions and enums
├── provenance.py         # Audit trail for all operations
├── inventory.py          # Coverage tracking and validation
├── metrics.py            # Operation performance metrics
├── logging.py            # Structured JSON logging
├── components/           # Functional component classes
│   ├── base.py           # Component base class
│   ├── ingestor.py       # Data ingestion
│   ├── calculator.py     # Derived computations
│   ├── exporter.py       # Export to model formats
│   └── query.py          # Data access and status
├── storage/              # Pluggable storage backends
│   ├── base.py           # StorageProvider interface
│   ├── local.py          # ZipStore, DirectoryStore, MemoryStore
│   ├── cloud.py          # S3, GCS providers
│   └── factory.py        # URI-based provider selection
└── workflow/             # Multi-step workflow orchestration
    ├── config.py         # YAML configuration parsing
    ├── steps.py          # Workflow step definitions
    └── engine.py         # Workflow execution engine
```

---

## Core Classes

The container package uses a main container class with four functional
components that share a common state object.

### SwimContainer

The top-level class that represents a `.swim` file. It provides factory
methods (`create()`, `open()`) and delegates work to its component objects.
Opening a container in read mode (`'r'`) prevents modifications; write mode
(`'r+'`) allows updates. The container manages the storage backend lifecycle
and coordinates saves.

### ContainerState

Centralized state shared by all components. It holds the zarr root group,
field UIDs, time index, provenance log, and inventory tracker. Provides an
xarray Dataset view of the data for vectorized operations. Components read
and write through this object to maintain consistency.

### Ingestor (`container.ingest`)

Handles data ingestion from external sources. Methods like `ndvi()`,
`gridmet()`, `snodas()`, and `properties()` parse source files (CSVs,
NetCDFs) and write them to the appropriate zarr paths. Each ingest operation
records its source, parameters, and timestamp in the provenance log.

### Calculator (`container.compute`)

Performs derived computations on ingested data. Examples: merging NDVI from
multiple instruments (`merged_ndvi()`), fusing NDVI time series
(`fused_ndvi()`), and computing irrigation windows and crop dynamics
(`dynamics()`). Results are written to `derived/` paths in the container.

### Exporter (`container.export`)

Exports container data to formats needed by downstream tools. Key method
`prepped_input_json()` produces the JSON file consumed by the process
package. Also supports CSV exports, shapefiles, and direct conversion to
xarray Datasets or pandas DataFrames.

### Query (`container.query`)

Read-only data access and validation. The `status()` method reports data
coverage; `validate()` checks for missing or invalid data. Provides
`xarray()`, `dataframe()`, and `geodataframe()` methods for extracting
subsets. Used to inspect container contents without modification.

---

## Data Flow Diagram

Shows how data flows from external sources through ingestion, computation,
and export.

```mermaid
flowchart TB
    subgraph External["External Data Sources"]
        EE["Earth Engine CSVs<br/>(NDVI, ETf)"]
        GridMET["GridMET<br/>(meteorology)"]
        ERA5["ERA5<br/>(meteorology)"]
        SNODAS["SNODAS<br/>(snow)"]
        Shapefiles["Shapefiles<br/>(field boundaries)"]
        Soils["Soils/LULC CSVs<br/>(properties)"]
    end

    subgraph Container["SwimContainer (.swim file)"]
        direction TB
        subgraph Storage["Zarr Storage"]
            RS["remote_sensing/<br/>ndvi/, etf/"]
            Met["meteorology/<br/>gridmet/, era5/"]
            Snow["snow/<br/>snodas/"]
            Props["properties/<br/>soils, lulc, irrigation"]
            Derived["derived/<br/>dynamics, fused_ndvi"]
            Geom["geometry/<br/>centroids, bounds"]
        end

        subgraph State["ContainerState"]
            Root["zarr.Group root"]
            UIDs["field_uids[]"]
            TimeIdx["time_index"]
            Prov["ProvenanceLog"]
            Inv["Inventory"]
        end
    end

    subgraph Components["Component API"]
        Ingest["container.ingest"]
        Compute["container.compute"]
        Export["container.export"]
        Query["container.query"]
    end

    subgraph Outputs["Model Outputs"]
        JSON["prepped_input.json"]
        HDF5["project.h5<br/>(process package)"]
        CSV["CSV exports"]
        XR["xarray.Dataset"]
    end

    EE --> Ingest
    GridMET --> Ingest
    ERA5 --> Ingest
    SNODAS --> Ingest
    Shapefiles --> Ingest
    Soils --> Ingest

    Ingest --> RS
    Ingest --> Met
    Ingest --> Snow
    Ingest --> Props
    Ingest --> Geom

    RS --> Compute
    Met --> Compute
    Props --> Compute
    Compute --> Derived

    Storage --> State
    State --> Query
    State --> Export

    Export --> JSON
    Export --> HDF5
    Export --> CSV
    Query --> XR
```

---

## Class Relationships

Shows the main classes and their relationships.

```mermaid
classDiagram
    class SwimContainer {
        +Path path
        +str mode
        +ContainerState state
        +Ingestor ingest
        +Calculator compute
        +Exporter export
        +Query query
        +create(path, shapefile, ...)$ SwimContainer
        +open(uri, mode)$ SwimContainer
        +save()
        +close()
    }

    class ContainerState {
        +zarr.Group root
        +List~str~ field_uids
        +Dict uid_to_index
        +DatetimeIndex time_index
        +ProvenanceLog provenance
        +Inventory inventory
        +str mode
        +bool modified
        +int n_fields
        +int n_days
        +xr.Dataset dataset
        +get_subset(variables, fields)
        +mark_modified()
        +refresh()
    }

    class Component {
        <<abstract>>
        #ContainerState _state
        #ContainerLogger _log
        #_ensure_writable()
        #_track_operation()
    }

    class Ingestor {
        +ndvi(source_dir, instrument, mask)
        +etf(source_dir, model, mask)
        +gridmet(source_dir)
        +era5(source_dir)
        +snodas(source_dir)
        +properties(lulc_csv, soils_csv)
        +dynamics(dynamics_csv)
    }

    class Calculator {
        +merged_ndvi(instruments)
        +fused_ndvi()
        +dynamics(etf_model, ...)
        +irrigation_windows()
    }

    class Exporter {
        +prepped_input_json(output_path)
        +shapefile(output_path)
        +csv(output_path, variables)
        +model_inputs(output_dir)
        +to_xarray(variables)
        +to_dataframe(variables)
        +observations(output_path)
    }

    class Query {
        +status(detailed) str
        +validate() ValidationResult
        +validate_fields(fields)
        +valid_fields() List~str~
        +xarray(path) xr.DataArray
        +dataset() xr.Dataset
        +dataframe(variables) DataFrame
        +geodataframe() GeoDataFrame
        +field_timeseries(uid, variables)
        +dynamics(uid) Dict
        +inventory() Inventory
    }

    SwimContainer *-- ContainerState
    SwimContainer *-- Ingestor
    SwimContainer *-- Calculator
    SwimContainer *-- Exporter
    SwimContainer *-- Query

    Component <|-- Ingestor
    Component <|-- Calculator
    Component <|-- Exporter
    Component <|-- Query

    Component --> ContainerState : uses
```

---

## Storage Backend Architecture

Shows the pluggable storage provider system.

```mermaid
classDiagram
    class StorageProvider {
        <<abstract>>
        +str location
        +str mode
        +str uri
        +zarr.Group root
        +open() zarr.Group
        +close()
        +exists() bool
        +save()
    }

    class ZipStoreProvider {
        +Path path
        Portable .swim files
    }

    class DirectoryStoreProvider {
        +Path path
        Fast local development
    }

    class MemoryStoreProvider {
        In-memory for testing
    }

    class S3StoreProvider {
        +str bucket
        +str key
        Amazon S3 / S3-compatible
    }

    class GCSStoreProvider {
        +str bucket
        +str blob
        Google Cloud Storage
    }

    class StorageProviderFactory {
        +from_uri(uri, mode)$ StorageProvider
        +detect_type(path)$ str
    }

    StorageProvider <|-- ZipStoreProvider
    StorageProvider <|-- DirectoryStoreProvider
    StorageProvider <|-- MemoryStoreProvider
    StorageProvider <|-- S3StoreProvider
    StorageProvider <|-- GCSStoreProvider

    StorageProviderFactory ..> StorageProvider : creates
```

---

## Workflow Engine

Shows the YAML-driven workflow orchestration system.

```mermaid
flowchart LR
    subgraph Config["WorkflowConfig (YAML)"]
        Project["project:<br/>name, shapefile, dates"]
        Sources["sources:<br/>ndvi, etf, met, props"]
        Compute["workflow:<br/>fused_ndvi, dynamics"]
        Export["export:<br/>format, output"]
    end

    subgraph Engine["WorkflowEngine"]
        Parse["Parse YAML"]
        Plan["Build step graph"]
        Execute["Execute steps"]
        Resume["Resume from checkpoint"]
    end

    subgraph Steps["WorkflowSteps"]
        S1["IngestNDVIStep"]
        S2["IngestETFStep"]
        S3["IngestMeteorologyStep"]
        S4["IngestPropertiesStep"]
        S5["ComputeFusedNDVIStep"]
        S6["ComputeDynamicsStep"]
        S7["ExportStep"]
    end

    subgraph Progress["WorkflowProgress"]
        Status["step status"]
        Errors["error messages"]
        Timing["execution times"]
    end

    Config --> Parse
    Parse --> Plan
    Plan --> Execute
    Execute --> Steps
    Steps --> Progress
    Progress --> Resume
    Resume --> Execute
```

---

## Zarr Data Schema

Shows the hierarchical data organization within the container.

```mermaid
flowchart TB
    subgraph Root["/ (zarr root)"]
        subgraph RS["remote_sensing/"]
            NDVI["ndvi/<br/>├─ landsat/<br/>│  ├─ irr (time, field)<br/>│  └─ inv_irr<br/>├─ sentinel/<br/>└─ combined/"]
            ETF["etf/<br/>├─ landsat/<br/>│  ├─ ssebop/<br/>│  │  ├─ irr<br/>│  │  └─ inv_irr<br/>│  └─ ptjpl/<br/>└─ ecostress/"]
        end

        subgraph Met["meteorology/"]
            GM["gridmet/<br/>├─ eto (time, field)<br/>├─ etr<br/>├─ prcp<br/>├─ tmin<br/>├─ tmax<br/>├─ srad<br/>└─ vpd"]
            E5["era5/<br/>└─ (same variables)"]
        end

        subgraph Sn["snow/"]
            SN["snodas/<br/>└─ swe (time, field)"]
        end

        subgraph Pr["properties/"]
            Soils["soils/<br/>├─ awc (field,)<br/>├─ ksat<br/>├─ clay<br/>└─ sand"]
            LULC["lulc/<br/>├─ modis_lc<br/>└─ cdl"]
            Irr["irrigation/<br/>├─ lanid<br/>└─ irrmapper"]
        end

        subgraph Dv["derived/"]
            Dyn["dynamics/<br/>├─ irr_doy_start<br/>├─ irr_doy_end<br/>├─ perennial<br/>└─ zr_max"]
            Fused["fused_ndvi/<br/>└─ irr (time, field)"]
        end

        subgraph Gm["geometry/"]
            Geo["├─ centroids (field, 2)<br/>├─ bounds (field, 4)<br/>└─ areas (field,)"]
        end

        subgraph Meta["attrs (metadata)"]
            Attrs["├─ schema_version<br/>├─ created_at<br/>├─ project_name<br/>├─ start_date<br/>└─ end_date"]
        end
    end
```

---

## Component Operation Sequence

Shows a typical workflow sequence from container creation to export.

```mermaid
sequenceDiagram
    participant U as User
    participant C as SwimContainer
    participant I as Ingestor
    participant Ca as Calculator
    participant E as Exporter
    participant Q as Query
    participant S as ContainerState
    participant P as ProvenanceLog

    U->>C: create(shapefile, dates)
    C->>S: Initialize state
    C->>P: Record creation

    U->>I: ndvi(source_dir, "landsat", "irr")
    I->>S: Parse CSVs to xarray
    I->>S: Write to zarr
    I->>P: Record ingest event

    U->>I: gridmet(met_dir)
    I->>S: Parse met data
    I->>S: Write to zarr
    I->>P: Record ingest event

    U->>Q: status()
    Q->>S: Check data coverage
    Q-->>U: Coverage report

    U->>Ca: dynamics(etf_model="ssebop")
    Ca->>S: Read NDVI, ETf, met
    Ca->>S: Compute irrigation windows
    Ca->>S: Write derived/dynamics
    Ca->>P: Record compute event

    U->>E: prepped_input_json(output_path)
    E->>S: Read all required data
    E->>E: Build JSON structure
    E->>P: Record export event
    E-->>U: prepped_input.json

    U->>C: save()
    C->>S: Flush to storage
```

---

## Key Design Decisions

### Zarr Backend
- **Single-file packaging**: ZipStore creates portable `.swim` files
- **Lazy loading**: Time series loaded on-demand, not all at once
- **Cloud-ready**: Same API works with S3/GCS backends

### Component Architecture
- **Separation of concerns**: Ingest, Compute, Export, Query are independent
- **Shared state**: All components access `ContainerState` for consistency
- **IDE-friendly**: `container.ingest.ndvi()` provides autocomplete

### Provenance Tracking
- **Full audit trail**: Every operation recorded with timestamps
- **Reproducibility**: Parameters and sources captured
- **Debugging**: Can trace data lineage through operations

### xarray Integration
- **Vectorized operations**: Efficient computation over fields × time
- **Familiar API**: Scientists know xarray/pandas
- **Memory efficient**: Dask integration for large datasets

### Schema Enums
- **Type safety**: `Instrument.LANDSAT` vs string typos
- **Discoverability**: IDE shows valid options
- **Validation**: Invalid combinations caught early
