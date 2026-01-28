"""
Local filesystem storage providers.

Provides storage backends for:
- ZipStoreProvider: Single .swim (zip) file storage (default, good for sharing)
- DirectoryStoreProvider: Uncompressed directory storage (faster for development)
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Optional, Union

import zarr
import zarr.storage
from filelock import FileLock, Timeout

from .base import StorageProvider


class ZipStoreProvider(StorageProvider):
    """
    Local zip file storage provider.

    Stores the container as a single compressed file (.swim extension).
    This is the default storage format, optimized for sharing and archival.

    Features:
        - Single-file storage for easy sharing
        - Automatic file locking for concurrent access safety
        - Compression reduces storage size

    Trade-offs:
        - Slower than directory storage for incremental writes
        - Requires closing to finalize changes

    Example:
        provider = ZipStoreProvider(Path("project.swim"), mode="r+")
        root = provider.open()
        # ... work with zarr groups
        provider.close()
    """

    def __init__(self, path: Union[str, Path], mode: str = "r"):
        """
        Initialize zip storage provider.

        Args:
            path: Path to the .swim/.zip file
            mode: Access mode ('r', 'r+', 'a', 'w')
        """
        super().__init__(mode)
        self._path = Path(path)
        self._lock: Optional[FileLock] = None

    @property
    def uri(self) -> str:
        """File URI for this storage."""
        return f"file://{self._path.resolve()}"

    @property
    def location(self) -> Path:
        """Path to the storage file."""
        return self._path

    def exists(self) -> bool:
        """Check if the zip file exists."""
        return self._path.exists()

    def delete(self) -> None:
        """Delete the zip file and any associated lock file."""
        if self.is_open:
            raise RuntimeError("Cannot delete open storage. Close first.")
        if self._path.exists():
            self._path.unlink()
        # Also remove lock file if present
        lock_path = Path(str(self._path) + ".lock")
        if lock_path.exists():
            lock_path.unlink()

    def open(self) -> zarr.Group:
        """
        Open the zip store and return root group.

        Acquires file lock for write modes to prevent concurrent modification.
        """
        if self.is_open:
            return self._root

        # Validate existence for read modes
        if self._mode in ("r", "r+") and not self.exists():
            raise FileNotFoundError(f"Container not found: {self._path}")

        # Acquire file lock for write modes
        if self._mode in ("r+", "a", "w"):
            lock_path = str(self._path) + ".lock"
            self._lock = FileLock(lock_path, timeout=3)
            try:
                self._lock.acquire()
            except Timeout:
                raise RuntimeError(
                    f"Could not acquire lock for container: {self._path}\n"
                    f"Another process may have it open, or a previous process crashed.\n"
                    f"If you're sure no other process is using this container, delete the lock file:\n"
                    f"  {lock_path}"
                ) from None

            # Suppress ZipStore duplicate name warnings during writes.
            # Zarr's ZipStore creates duplicate zip entries when writing to the
            # same chunk multiple times, which is expected during ingestion.
            warnings.filterwarnings(
                "ignore",
                message="Duplicate name:",
                category=UserWarning,
                module="zipfile",
            )

        # Map modes: ZipStore only accepts 'r', 'w', 'x', 'a'
        # 'r+' for read-write on existing -> use 'a' (append)
        if self._mode == "r":
            zarr_mode = "r"
        elif self._mode in ("r+", "a"):
            zarr_mode = "a"
        else:
            zarr_mode = self._mode

        self._store = zarr.storage.ZipStore(str(self._path), mode=zarr_mode)
        self._root = zarr.open_group(
            self._store, mode="a" if zarr_mode != "r" else "r"
        )

        return self._root

    def close(self) -> None:
        """Close the store and release file lock."""
        if self._store is not None:
            self._store.close()
            self._store = None
            self._root = None

        if self._lock is not None:
            try:
                self._lock.release()
            except Exception:
                pass  # Lock may already be released
            self._lock = None


class DirectoryStoreProvider(StorageProvider):
    """
    Local directory storage provider.

    Stores the container as an uncompressed directory of files.
    Faster than zip storage for development and incremental operations.

    Features:
        - Fast incremental writes (no zip overhead)
        - Easy inspection of individual chunks
        - Good for development and debugging

    Trade-offs:
        - Takes more disk space (no compression)
        - Many small files (harder to share)

    Example:
        provider = DirectoryStoreProvider(Path("project.zarr"), mode="r+")
        root = provider.open()
        # ... work with zarr groups
        provider.close()
    """

    def __init__(self, path: Union[str, Path], mode: str = "r"):
        """
        Initialize directory storage provider.

        Args:
            path: Path to the directory
            mode: Access mode ('r', 'r+', 'a', 'w')
        """
        super().__init__(mode)
        self._path = Path(path)
        self._lock: Optional[FileLock] = None

    @property
    def uri(self) -> str:
        """File URI for this storage."""
        return f"file://{self._path.resolve()}"

    @property
    def location(self) -> Path:
        """Path to the storage directory."""
        return self._path

    def exists(self) -> bool:
        """Check if the directory exists and is a zarr store."""
        if not self._path.exists():
            return False
        # Check for zarr files to confirm it's a zarr store
        # zarr 2.x uses .zgroup/.zattrs, zarr 3.x uses zarr.json
        return (
            (self._path / ".zgroup").exists()
            or (self._path / ".zattrs").exists()
            or (self._path / "zarr.json").exists()
        )

    def delete(self) -> None:
        """Delete the entire directory and any associated lock file."""
        if self.is_open:
            raise RuntimeError("Cannot delete open storage. Close first.")
        if self._path.exists():
            shutil.rmtree(self._path)
        # Also remove lock file if present
        lock_path = Path(str(self._path) + ".lock")
        if lock_path.exists():
            lock_path.unlink()

    def open(self) -> zarr.Group:
        """
        Open the directory store and return root group.

        Uses file lock for write modes to prevent concurrent modification.
        """
        if self.is_open:
            return self._root

        # Validate existence for read modes
        if self._mode in ("r", "r+") and not self.exists():
            raise FileNotFoundError(f"Container not found: {self._path}")

        # Acquire file lock for write modes
        if self._mode in ("r+", "a", "w"):
            lock_path = str(self._path) + ".lock"
            self._lock = FileLock(lock_path, timeout=3)
            try:
                self._lock.acquire()
            except Timeout:
                raise RuntimeError(
                    f"Could not acquire lock for container: {self._path}\n"
                    f"Another process may have it open, or a previous process crashed.\n"
                    f"If you're sure no other process is using this container, delete the lock file:\n"
                    f"  {lock_path}"
                ) from None

        # DirectoryStore maps modes directly
        if self._mode == "r":
            zarr_mode = "r"
        elif self._mode in ("r+", "a"):
            zarr_mode = "a"
        else:
            zarr_mode = self._mode

        # Use LocalStore (zarr 3.x) for uncompressed directory access
        self._store = zarr.storage.LocalStore(str(self._path))
        self._root = zarr.open_group(self._store, mode=zarr_mode)

        return self._root

    def close(self) -> None:
        """Close the store and release file lock."""
        # DirectoryStore doesn't need explicit close, but we clean up references
        self._store = None
        self._root = None

        if self._lock is not None:
            try:
                self._lock.release()
            except Exception:
                pass
            self._lock = None


class MemoryStoreProvider(StorageProvider):
    """
    In-memory storage provider.

    Stores data entirely in memory. Useful for:
    - Unit testing (fast, no disk I/O)
    - Temporary processing
    - Building containers before persisting

    Note:
        All data is lost when the provider is closed or program exits.

    Example:
        provider = MemoryStoreProvider(name="test_container")
        root = provider.open()
        # ... work with zarr groups
        provider.close()  # Data is lost
    """

    def __init__(self, name: str = "memory", mode: str = "a"):
        """
        Initialize memory storage provider.

        Args:
            name: Identifier for this memory store (for URI)
            mode: Access mode (typically 'a' since memory is always writable)
        """
        super().__init__(mode)
        self._name = name

    @property
    def uri(self) -> str:
        """Memory URI for this storage."""
        return f"memory://{self._name}"

    @property
    def location(self) -> str:
        """Return the memory store name."""
        return self._name

    def exists(self) -> bool:
        """Memory stores always 'exist' when open."""
        return self.is_open

    def delete(self) -> None:
        """Clear the memory store."""
        if self.is_open:
            self._store.clear()

    def open(self) -> zarr.Group:
        """
        Open the memory store and return root group.

        Creates a fresh memory store on each open.
        """
        if self.is_open:
            return self._root

        self._store = zarr.storage.MemoryStore()
        self._root = zarr.open_group(self._store, mode="a")

        return self._root

    def close(self) -> None:
        """Close the store (data is lost)."""
        self._store = None
        self._root = None
