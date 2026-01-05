"""
Cloud storage providers.

Provides storage backends for cloud object stores:
- S3StoreProvider: Amazon S3 / S3-compatible storage
- GCSStoreProvider: Google Cloud Storage

These providers require additional dependencies:
- S3: s3fs, aiobotocore
- GCS: gcsfs

Install with: pip install swimrs[cloud] or pip install s3fs gcsfs
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import zarr

from .base import StorageProvider


class S3StoreProvider(StorageProvider):
    """
    Amazon S3 storage provider.

    Stores the container in an S3 bucket using fsspec/s3fs.
    Enables cloud-native workflows and sharing across teams.

    Features:
        - Scalable cloud storage
        - Works with any S3-compatible storage (AWS, MinIO, etc.)
        - Supports IAM credentials, profiles, or explicit keys

    Requirements:
        pip install s3fs aiobotocore

    Example:
        # Using IAM credentials (recommended)
        provider = S3StoreProvider(
            bucket="my-bucket",
            key="projects/flux_network.zarr",
        )

        # Using explicit credentials
        provider = S3StoreProvider(
            bucket="my-bucket",
            key="projects/flux_network.zarr",
            aws_access_key_id="AKIA...",
            aws_secret_access_key="...",
        )

        root = provider.open()
        # ... work with zarr groups
        provider.close()
    """

    def __init__(
        self,
        bucket: str,
        key: str,
        mode: str = "r",
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        profile: Optional[str] = None,
        **s3_kwargs: Any,
    ):
        """
        Initialize S3 storage provider.

        Args:
            bucket: S3 bucket name
            key: Object key (path within bucket)
            mode: Access mode ('r', 'r+', 'a', 'w')
            endpoint_url: Custom endpoint for S3-compatible storage
            region_name: AWS region name
            aws_access_key_id: AWS access key (optional if using IAM/profile)
            aws_secret_access_key: AWS secret key
            profile: AWS credentials profile name
            **s3_kwargs: Additional arguments passed to s3fs.S3FileSystem
        """
        super().__init__(mode)
        self._bucket = bucket
        self._key = key.strip("/")
        self._endpoint_url = endpoint_url
        self._region_name = region_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._profile = profile
        self._s3_kwargs = s3_kwargs
        self._fs = None  # s3fs filesystem object

    @property
    def uri(self) -> str:
        """S3 URI for this storage."""
        return f"s3://{self._bucket}/{self._key}"

    @property
    def location(self) -> str:
        """S3 path."""
        return f"{self._bucket}/{self._key}"

    def exists(self) -> bool:
        """Check if the S3 path exists."""
        try:
            import s3fs
        except ImportError:
            raise ImportError(
                "s3fs is required for S3 storage. "
                "Install with: pip install s3fs aiobotocore"
            )

        fs = self._get_filesystem()
        # Check for .zgroup or .zattrs to confirm it's a zarr store
        zgroup_path = f"{self._bucket}/{self._key}/.zgroup"
        zattrs_path = f"{self._bucket}/{self._key}/.zattrs"
        return fs.exists(zgroup_path) or fs.exists(zattrs_path)

    def delete(self) -> None:
        """Delete the S3 path (recursively)."""
        if self.is_open:
            raise RuntimeError("Cannot delete open storage. Close first.")
        fs = self._get_filesystem()
        path = f"{self._bucket}/{self._key}"
        if fs.exists(path):
            fs.rm(path, recursive=True)

    def _get_filesystem(self):
        """Get or create the s3fs filesystem."""
        if self._fs is not None:
            return self._fs

        try:
            import s3fs
        except ImportError:
            raise ImportError(
                "s3fs is required for S3 storage. "
                "Install with: pip install s3fs aiobotocore"
            )

        fs_kwargs: Dict[str, Any] = {**self._s3_kwargs}

        if self._endpoint_url:
            fs_kwargs["client_kwargs"] = fs_kwargs.get("client_kwargs", {})
            fs_kwargs["client_kwargs"]["endpoint_url"] = self._endpoint_url

        if self._region_name:
            fs_kwargs["client_kwargs"] = fs_kwargs.get("client_kwargs", {})
            fs_kwargs["client_kwargs"]["region_name"] = self._region_name

        if self._aws_access_key_id and self._aws_secret_access_key:
            fs_kwargs["key"] = self._aws_access_key_id
            fs_kwargs["secret"] = self._aws_secret_access_key
        elif self._profile:
            fs_kwargs["profile"] = self._profile

        self._fs = s3fs.S3FileSystem(**fs_kwargs)
        return self._fs

    def open(self) -> zarr.Group:
        """
        Open the S3 store and return root group.

        Uses S3Map for zarr-compatible access.
        """
        if self.is_open:
            return self._root

        try:
            import s3fs
        except ImportError:
            raise ImportError(
                "s3fs is required for S3 storage. "
                "Install with: pip install s3fs aiobotocore"
            )

        # Validate existence for read modes
        if self._mode in ("r", "r+") and not self.exists():
            raise FileNotFoundError(f"Container not found: {self.uri}")

        fs = self._get_filesystem()
        path = f"{self._bucket}/{self._key}"

        # Use S3Map as zarr store
        self._store = s3fs.S3Map(root=path, s3=fs, check=False)

        # Determine zarr mode
        if self._mode == "r":
            zarr_mode = "r"
        elif self._mode in ("r+", "a"):
            zarr_mode = "a"
        else:
            zarr_mode = self._mode

        self._root = zarr.open_group(self._store, mode=zarr_mode)

        return self._root

    def close(self) -> None:
        """Close the store."""
        self._store = None
        self._root = None
        self._fs = None


class GCSStoreProvider(StorageProvider):
    """
    Google Cloud Storage provider.

    Stores the container in a GCS bucket using gcsfs.

    Features:
        - Google Cloud native storage
        - Supports service account and default credentials

    Requirements:
        pip install gcsfs

    Example:
        # Using default credentials
        provider = GCSStoreProvider(
            bucket="my-bucket",
            key="projects/flux_network.zarr",
        )

        # Using service account
        provider = GCSStoreProvider(
            bucket="my-bucket",
            key="projects/flux_network.zarr",
            token="/path/to/service-account.json",
        )

        root = provider.open()
        # ... work with zarr groups
        provider.close()
    """

    def __init__(
        self,
        bucket: str,
        key: str,
        mode: str = "r",
        project: Optional[str] = None,
        token: Optional[str] = None,
        **gcs_kwargs: Any,
    ):
        """
        Initialize GCS storage provider.

        Args:
            bucket: GCS bucket name
            key: Object path within bucket
            mode: Access mode ('r', 'r+', 'a', 'w')
            project: GCP project ID
            token: Path to service account JSON or "cloud" for default credentials
            **gcs_kwargs: Additional arguments passed to gcsfs.GCSFileSystem
        """
        super().__init__(mode)
        self._bucket = bucket
        self._key = key.strip("/")
        self._project = project
        self._token = token
        self._gcs_kwargs = gcs_kwargs
        self._fs = None

    @property
    def uri(self) -> str:
        """GCS URI for this storage."""
        return f"gs://{self._bucket}/{self._key}"

    @property
    def location(self) -> str:
        """GCS path."""
        return f"{self._bucket}/{self._key}"

    def exists(self) -> bool:
        """Check if the GCS path exists."""
        fs = self._get_filesystem()
        zgroup_path = f"{self._bucket}/{self._key}/.zgroup"
        zattrs_path = f"{self._bucket}/{self._key}/.zattrs"
        return fs.exists(zgroup_path) or fs.exists(zattrs_path)

    def delete(self) -> None:
        """Delete the GCS path (recursively)."""
        if self.is_open:
            raise RuntimeError("Cannot delete open storage. Close first.")
        fs = self._get_filesystem()
        path = f"{self._bucket}/{self._key}"
        if fs.exists(path):
            fs.rm(path, recursive=True)

    def _get_filesystem(self):
        """Get or create the gcsfs filesystem."""
        if self._fs is not None:
            return self._fs

        try:
            import gcsfs
        except ImportError:
            raise ImportError(
                "gcsfs is required for GCS storage. "
                "Install with: pip install gcsfs"
            )

        fs_kwargs: Dict[str, Any] = {**self._gcs_kwargs}

        if self._project:
            fs_kwargs["project"] = self._project
        if self._token:
            fs_kwargs["token"] = self._token

        self._fs = gcsfs.GCSFileSystem(**fs_kwargs)
        return self._fs

    def open(self) -> zarr.Group:
        """
        Open the GCS store and return root group.

        Uses GCSMap for zarr-compatible access.
        """
        if self.is_open:
            return self._root

        try:
            import gcsfs
        except ImportError:
            raise ImportError(
                "gcsfs is required for GCS storage. "
                "Install with: pip install gcsfs"
            )

        # Validate existence for read modes
        if self._mode in ("r", "r+") and not self.exists():
            raise FileNotFoundError(f"Container not found: {self.uri}")

        fs = self._get_filesystem()
        path = f"{self._bucket}/{self._key}"

        # Use GCSMap as zarr store
        self._store = gcsfs.GCSMap(root=path, gcs=fs, check=False)

        # Determine zarr mode
        if self._mode == "r":
            zarr_mode = "r"
        elif self._mode in ("r+", "a"):
            zarr_mode = "a"
        else:
            zarr_mode = self._mode

        self._root = zarr.open_group(self._store, mode=zarr_mode)

        return self._root

    def close(self) -> None:
        """Close the store."""
        self._store = None
        self._root = None
        self._fs = None
