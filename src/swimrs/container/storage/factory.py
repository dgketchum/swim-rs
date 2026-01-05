"""
Storage provider factory.

Provides automatic provider selection based on URI patterns,
allowing transparent use of different storage backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
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
    _scheme_handlers: Dict[str, type] = {}

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
        uri: Union[str, Path],
        mode: str = "r",
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
            **kwargs: Additional arguments passed to the provider

        Returns:
            StorageProvider: Appropriate provider for the given URI

        Raises:
            ValueError: If URI scheme is not supported
        """
        # Handle Path objects
        if isinstance(uri, Path):
            return cls._from_local_path(uri, mode, **kwargs)

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
            return cls._from_local_path(Path(uri_str), mode, **kwargs)

        elif scheme == "file":
            # File URI
            path = Path(parsed.path)
            return cls._from_local_path(path, mode, **kwargs)

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
                f"Unsupported URI scheme: {scheme}. "
                f"Supported schemes: file, s3, gs, memory"
            )

    @classmethod
    def _from_local_path(
        cls,
        path: Path,
        mode: str,
        force_zip: bool = False,
        force_directory: bool = False,
        **kwargs: Any,
    ) -> StorageProvider:
        """
        Create local storage provider from path.

        Selection logic:
        1. If force_zip=True: ZipStoreProvider
        2. If force_directory=True: DirectoryStoreProvider
        3. If path ends with .swim or .zip: ZipStoreProvider
        4. If path ends with / or .zarr: DirectoryStoreProvider
        5. If path exists and is a directory: DirectoryStoreProvider
        6. If path exists and is a file: ZipStoreProvider
        7. Default for creation: infer from extension or use ZipStoreProvider

        Args:
            path: Local filesystem path
            mode: Access mode
            force_zip: Force zip storage regardless of path
            force_directory: Force directory storage regardless of path
            **kwargs: Additional arguments (ignored for local)

        Returns:
            StorageProvider: ZipStoreProvider or DirectoryStoreProvider
        """
        if force_zip and force_directory:
            raise ValueError("Cannot specify both force_zip and force_directory")

        if force_zip:
            return ZipStoreProvider(path, mode=mode)

        if force_directory:
            return DirectoryStoreProvider(path, mode=mode)

        # Infer from extension
        suffix = path.suffix.lower()
        name = path.name.lower()

        if suffix in (".swim", ".zip"):
            return ZipStoreProvider(path, mode=mode)

        if suffix == ".zarr" or name.endswith("/"):
            return DirectoryStoreProvider(path, mode=mode)

        # If path exists, check if it's a file or directory
        if path.exists():
            if path.is_dir():
                return DirectoryStoreProvider(path, mode=mode)
            else:
                return ZipStoreProvider(path, mode=mode)

        # For new containers, default to zip storage with .swim extension
        # unless the path already suggests directory storage
        return ZipStoreProvider(path, mode=mode)

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
    uri: Union[str, Path],
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
