"""
Container components providing a clean, namespace-organized API.

Components provide focused functionality as attributes of SwimContainer:
- container.ingest: Data ingestion operations
- container.compute: Derived data computation
- container.export: Data export operations
- container.query: Data access and status queries

This design provides immediate IDE autocomplete and clear organization
compared to the flat mixin-based API.

Example:
    # Clean component API
    container.ingest.ndvi(source_dir, instrument="landsat", mask="irr")
    container.compute.dynamics(etf_model="ssebop")
    container.export.observations("obs/", etf_model="ssebop")
    print(container.query.status())
"""

from .base import Component
from .calculator import Calculator
from .exporter import Exporter
from .ingestor import Ingestor
from .query import Query

__all__ = [
    "Component",
    "Ingestor",
    "Calculator",
    "Exporter",
    "Query",
]
