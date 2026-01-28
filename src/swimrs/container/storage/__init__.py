"""
Storage providers for SwimContainer.

This module provides pluggable storage backends that decouple
container logic from the underlying storage implementation.

Available Providers:
    StorageProvider: Abstract base class defining the interface
    ZipStoreProvider: Local zip file storage (.swim files)
    DirectoryStoreProvider: Local directory storage (faster for development)
    MemoryStoreProvider: In-memory storage (for testing)
    S3StoreProvider: Amazon S3 / S3-compatible cloud storage
    GCSStoreProvider: Google Cloud Storage

Factory:
    StorageProviderFactory: Create providers from URIs
    open_storage: Convenience function to open storage from URI

Examples:
    # Open local container
    from swimrs.container.storage import open_storage

    with open_storage("project.swim", mode="r+") as storage:
        root = storage.root
        # ... work with zarr groups

    # Create provider explicitly
    from swimrs.container.storage import ZipStoreProvider

    provider = ZipStoreProvider("project.swim", mode="r+")
    root = provider.open()
    provider.close()

    # Use factory for URI-based selection
    from swimrs.container.storage import StorageProviderFactory

    # Automatically selects ZipStoreProvider
    provider = StorageProviderFactory.from_uri("project.swim")

    # Automatically selects S3StoreProvider
    provider = StorageProviderFactory.from_uri("s3://bucket/project.zarr")
"""

from pathlib import Path
from typing import Union

from .base import StorageProvider
from .local import DirectoryStoreProvider, MemoryStoreProvider, ZipStoreProvider
from .factory import StorageProviderFactory, open_storage


def detect_storage_type(path: Union[str, Path]) -> str:
    """
    Detect storage type from path.

    Auto-detection logic following scientific computing norms:
    - If path is an existing file → "zip"
    - If path is an existing directory → "directory"
    - If path doesn't exist → "directory" (default for new containers)

    Args:
        path: Path to check

    Returns:
        Storage type string: "zip" or "directory"

    Examples:
        >>> detect_storage_type("existing.swim")  # existing file
        'zip'
        >>> detect_storage_type("existing_dir.swim")  # existing directory
        'directory'
        >>> detect_storage_type("new_project.swim")  # doesn't exist
        'directory'
    """
    path = Path(path)

    if path.is_file():
        return "zip"
    elif path.is_dir():
        return "directory"
    else:
        # New path - default to directory for development
        return "directory"


# Cloud providers imported lazily to avoid hard dependency
__all__ = [
    # Base
    "StorageProvider",
    # Local providers
    "ZipStoreProvider",
    "DirectoryStoreProvider",
    "MemoryStoreProvider",
    # Factory
    "StorageProviderFactory",
    "open_storage",
    # Helpers
    "detect_storage_type",
]


def __getattr__(name: str):
    """Lazy loading for cloud providers."""
    if name == "S3StoreProvider":
        from .cloud import S3StoreProvider
        return S3StoreProvider
    elif name == "GCSStoreProvider":
        from .cloud import GCSStoreProvider
        return GCSStoreProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
