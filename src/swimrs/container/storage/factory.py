"""
Storage provider factory.

Provides automatic provider selection based on URI patterns,
allowing transparent use of different storage backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .base import StorageProvider
from .local import DirectoryStoreProvider, MemoryStoreProvider, ZipStoreProvider


class StorageProviderFactory:
    """
    Factory for creating storage providers from URIs or paths.

    Automatically selects the appropriate provider based on:
    - URI scheme (file://, s3://, gs://, memory://)
    - File extension (.swim, .zip, .zarr)
    - Path type (file vs directory)

    Examples:
        # Local zip file (default for .swim files)
        provider = StorageProviderFactory.from_uri("project.swim")
        provider = StorageProviderFactory.from_uri("file:///path/to/project.swim")

        # Local directory
        provider = StorageProviderFactory.from_uri("project.zarr/")
        provider = StorageProviderFactory.from_uri("/path/to/project/")

        # S3 storage
        provider = StorageProviderFactory.from_uri(
            "s3://bucket/path/to/project.zarr",
            aws_access_key_id="...",
            aws_secret_access_key="...",
        )

        # Google Cloud Storage
        provider = StorageProviderFactory.from_uri(
            "gs://bucket/path/to/project.zarr",
            project="my-gcp-project",
        )

        # In-memory (for testing)
        provider = StorageProviderFactory.from_uri("memory://test")
    """

    # Map schemes to provider classes
    _scheme_handlers: dict[str, type] = {}

    @classmethod
    def register_scheme(cls, scheme: str, provider_class: type) -> None:
        """
        Register a custom scheme handler.

        Args:
            scheme: URI scheme (e.g., "s3", "gs", "custom")
            provider_class: StorageProvider subclass to handle this scheme
        """
        cls._scheme_handlers[scheme.lower()] = provider_class

    @classmethod
    def from_uri(
        cls,
        uri: str | Path,
        mode: str = "r",
        storage: str = "auto",
        **kwargs: Any,
    ) -> StorageProvider:
        """
        Create appropriate storage provider from URI or path.

        Args:
            uri: Storage location as URI or path:
                - Local path: "project.swim", "/path/to/project.zarr"
                - File URI: "file:///path/to/project.swim"
                - S3 URI: "s3://bucket/key"
                - GCS URI: "gs://bucket/key"
                - Memory: "memory://name"
            mode: Access mode ('r', 'r+', 'a', 'w')
            storage: Storage type for local paths ("auto", "directory", "zip").
                - "auto" (default): auto-detect from path, defaults to directory for new
                - "directory": explicit directory store
                - "zip": explicit zip store
            **kwargs: Additional arguments passed to the provider

        Returns:
            StorageProvider: Appropriate provider for the given URI

        Raises:
            ValueError: If URI scheme is not supported
        """
        # Handle Path objects
        if isinstance(uri, Path):
            return cls._from_local_path(uri, mode, storage=storage, **kwargs)

        # Parse URI
        uri_str = str(uri)

        # Handle memory:// scheme
        if uri_str.startswith("memory://"):
            name = uri_str[9:] or "default"
            return MemoryStoreProvider(name=name, mode=mode)

        # Parse as URL
        parsed = urlparse(uri_str)

        # Handle explicit schemes
        scheme = parsed.scheme.lower()

        if scheme == "":
            # No scheme - treat as local path
            return cls._from_local_path(Path(uri_str), mode, storage=storage, **kwargs)

        elif scheme == "file":
            # File URI
            path = Path(parsed.path)
            return cls._from_local_path(path, mode, storage=storage, **kwargs)

        elif scheme == "s3":
            # S3 URI
            from .cloud import S3StoreProvider

            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            return S3StoreProvider(bucket=bucket, key=key, mode=mode, **kwargs)

        elif scheme == "gs":
            # Google Cloud Storage URI
            from .cloud import GCSStoreProvider

            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            return GCSStoreProvider(bucket=bucket, key=key, mode=mode, **kwargs)

        elif scheme in cls._scheme_handlers:
            # Custom registered handler
            handler = cls._scheme_handlers[scheme]
            return handler(uri_str, mode=mode, **kwargs)

        else:
            raise ValueError(
                f"Unsupported URI scheme: {scheme}. Supported schemes: file, s3, gs, memory"
            )

    @classmethod
    def _from_local_path(
        cls,
        path: Path,
        mode: str,
        storage: str = "auto",
        force_zip: bool = False,
        force_directory: bool = False,
        **kwargs: Any,
    ) -> StorageProvider:
        """
        Create local storage provider from path.

        Selection logic:
        1. If storage="zip" or force_zip=True: ZipStoreProvider
        2. If storage="directory" or force_directory=True: DirectoryStoreProvider
        3. If storage="auto" (default):
           a. If path exists as file: ZipStoreProvider
           b. If path exists as directory: DirectoryStoreProvider
           c. If path doesn't exist: DirectoryStoreProvider (for development)

        Args:
            path: Local filesystem path
            mode: Access mode
            storage: Storage type ("auto", "directory", "zip")
            force_zip: Force zip storage (deprecated, use storage="zip")
            force_directory: Force directory storage (deprecated, use storage="directory")
            **kwargs: Additional arguments (ignored for local)

        Returns:
            StorageProvider: ZipStoreProvider or DirectoryStoreProvider
        """
        # Handle deprecated force_* parameters
        if force_zip:
            storage = "zip"
        if force_directory:
            storage = "directory"

        if storage not in ("auto", "directory", "zip"):
            raise ValueError(
                f"Invalid storage type: {storage}. Must be 'auto', 'directory', or 'zip'."
            )

        # Explicit storage selection
        if storage == "zip":
            return ZipStoreProvider(path, mode=mode)
        if storage == "directory":
            return DirectoryStoreProvider(path, mode=mode)

        # Auto-detection: check if path exists
        if path.exists():
            if path.is_file():
                return ZipStoreProvider(path, mode=mode)
            elif path.is_dir():
                return DirectoryStoreProvider(path, mode=mode)

        # New path: default to DirectoryStore for development
        # DirectoryStore is more resilient and easier to debug
        return DirectoryStoreProvider(path, mode=mode)

    @classmethod
    def for_testing(cls, name: str = "test") -> StorageProvider:
        """
        Create in-memory provider for testing.

        Convenience method that creates a writable memory provider.

        Args:
            name: Identifier for the memory store

        Returns:
            MemoryStoreProvider in append mode
        """
        return MemoryStoreProvider(name=name, mode="a")


# Convenience function at module level
def open_storage(
    uri: str | Path,
    mode: str = "r",
    **kwargs: Any,
) -> StorageProvider:
    """
    Open storage from URI or path.

    Convenience function that wraps StorageProviderFactory.from_uri.

    Args:
        uri: Storage location
        mode: Access mode ('r', 'r+', 'a', 'w')
        **kwargs: Additional provider arguments

    Returns:
        StorageProvider: Opened storage provider

    Example:
        with open_storage("project.swim", mode="r+") as storage:
            root = storage.root
            # ... work with zarr groups
    """
    provider = StorageProviderFactory.from_uri(uri, mode=mode, **kwargs)
    provider.open()
    return provider
