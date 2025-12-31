"""
SWIM Container Mixins - domain-specific functionality for SwimContainer.

Each mixin provides a focused set of methods:
- IngestionMixin: Data ingestion from various sources
- ComputeMixin: Derived data computation
- ExportMixin: Data export in various formats
- QueryMixin: Data access and status queries
"""

from swimrs.container.mixins.ingest import IngestionMixin
from swimrs.container.mixins.compute import ComputeMixin
from swimrs.container.mixins.export import ExportMixin
from swimrs.container.mixins.query import QueryMixin

__all__ = [
    "IngestionMixin",
    "ComputeMixin",
    "ExportMixin",
    "QueryMixin",
]
