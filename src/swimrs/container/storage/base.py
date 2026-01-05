"""
Abstract base class for storage providers.

Defines the interface that all storage backends must implement,
allowing SwimContainer to work with different storage types
(local zip, local directory, cloud storage, etc.) without modification.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import zarr


class StorageProvider(ABC):
    """
    Abstract interface for container storage backends.

    Implementations handle the specifics of opening, closing, and managing
    zarr groups for different storage types while presenting a unified interface.

    Attributes:
        _mode: Current access mode ('r', 'r+', 'a', 'w')
        _root: The opened zarr root group (None when closed)
    """

    def __init__(self, mode: str = "r"):
        """
        Initialize the storage provider.

        Args:
            mode: Access mode:
                - 'r': Read-only (default)
                - 'r+': Read-write on existing
                - 'a': Append (create if doesn't exist)
                - 'w': Write (overwrite if exists)
        """
        self._mode = mode
        self._root: Optional[zarr.Group] = None
        self._store = None  # Backend-specific store object

    @property
    def mode(self) -> str:
        """Current access mode."""
        return self._mode

    @property
    def is_open(self) -> bool:
        """Whether the storage is currently open."""
        return self._root is not None

    @property
    def is_writable(self) -> bool:
        """Whether the storage is open in a writable mode."""
        return self._mode in ("r+", "a", "w")

    @property
    def root(self) -> Optional[zarr.Group]:
        """The zarr root group, or None if not open."""
        return self._root

    @abstractmethod
    def open(self) -> zarr.Group:
        """
        Open the storage and return the root zarr group.

        Returns:
            zarr.Group: The root group of the container

        Raises:
            FileNotFoundError: If storage doesn't exist and mode is 'r' or 'r+'
            FileExistsError: If storage exists and mode is 'x'
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the storage connection.

        Should safely flush any pending writes and release resources.
        """
        pass

    @abstractmethod
    def exists(self) -> bool:
        """
        Check if the storage location exists.

        Returns:
            bool: True if storage exists and is accessible
        """
        pass

    @abstractmethod
    def delete(self) -> None:
        """
        Delete the storage location.

        Raises:
            PermissionError: If deletion is not permitted
        """
        pass

    @property
    @abstractmethod
    def uri(self) -> str:
        """
        Return a URI identifying this storage location.

        Examples:
            - "file:///path/to/container.swim"
            - "s3://bucket/key/container.zarr"
            - "memory://container"
        """
        pass

    @property
    @abstractmethod
    def location(self) -> Union[str, Path]:
        """
        Return the location in the most appropriate native format.

        For local storage, returns Path. For cloud storage, returns string URL.
        """
        pass

    def __enter__(self) -> "StorageProvider":
        """Context manager entry - opens the storage."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - closes the storage."""
        self.close()
        return False

    def __repr__(self) -> str:
        status = "open" if self.is_open else "closed"
        return f"{self.__class__.__name__}({self.uri!r}, mode={self._mode!r}, {status})"
