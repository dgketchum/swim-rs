"""
Tests for DirectoryStore as default storage backend.

Tests the changes from DIRECTORY_STORE_PLAN.md:
1. DirectoryStore is default for new containers
2. Auto-detection works for existing containers
3. storage parameter controls storage type explicitly
4. pack() and unpack() methods work correctly
"""

import shutil
from pathlib import Path

import pytest

from swimrs.container import SwimContainer, create_container, open_container
from swimrs.container.storage import (
    detect_storage_type,
    StorageProviderFactory,
    DirectoryStoreProvider,
    ZipStoreProvider,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_shapefile(tmp_path):
    """Create a simple test shapefile."""
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"uid": ["site1", "site2"], "name": ["Site 1", "Site 2"]},
        geometry=[Point(-105.0, 40.0), Point(-105.1, 40.1)],
        crs="EPSG:4326",
    )
    shp_path = tmp_path / "test_fields.shp"
    gdf.to_file(shp_path)
    return shp_path


# =============================================================================
# detect_storage_type() Tests
# =============================================================================

class TestDetectStorageType:
    """Tests for detect_storage_type helper function."""

    def test_detects_existing_file_as_zip(self, tmp_path):
        """Existing file is detected as zip storage."""
        file_path = tmp_path / "test.swim"
        file_path.write_text("dummy")
        assert detect_storage_type(file_path) == "zip"

    def test_detects_existing_directory_as_directory(self, tmp_path):
        """Existing directory is detected as directory storage."""
        dir_path = tmp_path / "test.swim"
        dir_path.mkdir()
        assert detect_storage_type(dir_path) == "directory"

    def test_new_path_defaults_to_directory(self, tmp_path):
        """New (non-existent) path defaults to directory storage."""
        new_path = tmp_path / "new_container.swim"
        assert not new_path.exists()
        assert detect_storage_type(new_path) == "directory"


# =============================================================================
# StorageProviderFactory Tests
# =============================================================================

class TestStorageProviderFactory:
    """Tests for factory storage selection."""

    def test_new_path_creates_directory_provider(self, tmp_path):
        """New path creates DirectoryStoreProvider by default."""
        new_path = tmp_path / "new.swim"
        provider = StorageProviderFactory.from_uri(new_path, mode="w")
        assert isinstance(provider, DirectoryStoreProvider)

    def test_existing_file_creates_zip_provider(self, tmp_path):
        """Existing file creates ZipStoreProvider."""
        file_path = tmp_path / "test.swim"
        # Create a minimal zip file
        import zipfile
        with zipfile.ZipFile(file_path, "w") as zf:
            zf.writestr(".zgroup", '{"zarr_format": 3}')

        provider = StorageProviderFactory.from_uri(file_path, mode="r")
        assert isinstance(provider, ZipStoreProvider)

    def test_existing_directory_creates_directory_provider(self, tmp_path):
        """Existing directory creates DirectoryStoreProvider."""
        dir_path = tmp_path / "test.swim"
        dir_path.mkdir()
        (dir_path / ".zgroup").write_text('{"zarr_format": 3}')

        provider = StorageProviderFactory.from_uri(dir_path, mode="r")
        assert isinstance(provider, DirectoryStoreProvider)

    def test_explicit_zip_storage(self, tmp_path):
        """storage='zip' forces ZipStoreProvider."""
        new_path = tmp_path / "new.swim"
        provider = StorageProviderFactory.from_uri(new_path, mode="w", storage="zip")
        assert isinstance(provider, ZipStoreProvider)

    def test_explicit_directory_storage(self, tmp_path):
        """storage='directory' forces DirectoryStoreProvider."""
        new_path = tmp_path / "new.swim"
        provider = StorageProviderFactory.from_uri(
            new_path, mode="w", storage="directory"
        )
        assert isinstance(provider, DirectoryStoreProvider)

    def test_invalid_storage_raises(self, tmp_path):
        """Invalid storage type raises ValueError."""
        new_path = tmp_path / "new.swim"
        with pytest.raises(ValueError, match="Invalid storage type"):
            StorageProviderFactory.from_uri(new_path, mode="w", storage="invalid")


# =============================================================================
# SwimContainer.create() Tests
# =============================================================================

class TestContainerCreate:
    """Tests for container creation with storage parameter."""

    def test_create_defaults_to_directory(self, simple_shapefile, tmp_path):
        """Container creation defaults to DirectoryStore."""
        container_path = tmp_path / "test.swim"
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
        container.close()

        # Verify it's a directory
        assert container_path.is_dir()
        # zarr 3.x uses zarr.json, zarr 2.x uses .zgroup
        assert (container_path / "zarr.json").exists() or (container_path / ".zgroup").exists()

    def test_create_with_storage_zip(self, simple_shapefile, tmp_path):
        """Container creation with storage='zip' creates zip file."""
        container_path = tmp_path / "test.swim"
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
            storage="zip",
        )
        container.close()

        # Verify it's a file (zip)
        assert container_path.is_file()

    def test_create_with_storage_directory(self, simple_shapefile, tmp_path):
        """Container creation with storage='directory' creates directory."""
        container_path = tmp_path / "test.swim"
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
            storage="directory",
        )
        container.close()

        # Verify it's a directory
        assert container_path.is_dir()


# =============================================================================
# open_container() Auto-detection Tests
# =============================================================================

class TestOpenContainerAutoDetection:
    """Tests for auto-detection when opening containers."""

    def test_opens_directory_container(self, simple_shapefile, tmp_path):
        """Opens directory container correctly."""
        container_path = tmp_path / "test.swim"

        # Create as directory
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
            storage="directory",
        )
        container.close()

        # Reopen - should auto-detect as directory
        container = open_container(str(container_path), mode="r")
        assert isinstance(container._provider, DirectoryStoreProvider)
        assert container.n_fields == 2
        container.close()

    def test_opens_zip_container(self, simple_shapefile, tmp_path):
        """Opens zip container correctly."""
        container_path = tmp_path / "test.swim"

        # Create as zip
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
            storage="zip",
        )
        container.close()

        # Reopen - should auto-detect as zip
        container = open_container(str(container_path), mode="r")
        assert isinstance(container._provider, ZipStoreProvider)
        assert container.n_fields == 2
        container.close()


# =============================================================================
# pack() and unpack() Tests
# =============================================================================

class TestPackUnpack:
    """Tests for pack() and unpack() methods."""

    def test_pack_directory_to_zip(self, simple_shapefile, tmp_path):
        """pack() creates zip from directory container."""
        container_path = tmp_path / "test.swim"
        packed_path = tmp_path / "packed.swim"

        # Create directory container
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
            storage="directory",
        )

        # Pack to zip
        result_path = container.pack(str(packed_path))
        container.close()

        # Verify zip was created
        assert result_path == packed_path
        assert packed_path.is_file()

        # Verify original directory still exists
        assert container_path.is_dir()

        # Verify packed container is valid
        packed_container = open_container(str(packed_path), mode="r")
        assert packed_container.n_fields == 2
        packed_container.close()

    def test_unpack_zip_to_directory(self, simple_shapefile, tmp_path):
        """unpack() creates directory from zip container."""
        container_path = tmp_path / "test.swim"
        unpacked_path = tmp_path / "unpacked.swim"

        # Create zip container
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
            storage="zip",
        )

        # Unpack to directory
        unpacked_container = container.unpack(str(unpacked_path))
        container.close()

        # Verify directory was created
        assert unpacked_path.is_dir()

        # Verify original zip still exists
        assert container_path.is_file()

        # Verify unpacked container is valid
        assert unpacked_container.n_fields == 2
        unpacked_container.close()

    def test_pack_existing_zip_copies(self, simple_shapefile, tmp_path):
        """pack() on zip container just copies the file."""
        container_path = tmp_path / "test.swim"
        packed_path = tmp_path / "packed.swim"

        # Create zip container
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
            storage="zip",
        )

        # Pack (should just copy)
        result_path = container.pack(str(packed_path))
        container.close()

        # Verify both files exist
        assert container_path.is_file()
        assert packed_path.is_file()

    def test_unpack_existing_directory_copies(self, simple_shapefile, tmp_path):
        """unpack() on directory container just copies the directory."""
        container_path = tmp_path / "test.swim"
        unpacked_path = tmp_path / "unpacked.swim"

        # Create directory container
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
            storage="directory",
        )

        # Unpack (should just copy)
        unpacked_container = container.unpack(str(unpacked_path))
        container.close()

        # Verify both directories exist
        assert container_path.is_dir()
        assert unpacked_path.is_dir()
        unpacked_container.close()

    def test_pack_fails_if_output_exists(self, simple_shapefile, tmp_path):
        """pack() fails if output path already exists."""
        container_path = tmp_path / "test.swim"
        packed_path = tmp_path / "packed.swim"
        packed_path.write_text("existing file")

        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )

        with pytest.raises(FileExistsError):
            container.pack(str(packed_path))
        container.close()

    def test_unpack_fails_if_output_exists(self, simple_shapefile, tmp_path):
        """unpack() fails if output path already exists."""
        container_path = tmp_path / "test.swim"
        unpacked_path = tmp_path / "unpacked.swim"
        unpacked_path.mkdir()

        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
            storage="zip",
        )

        with pytest.raises(FileExistsError):
            container.unpack(str(unpacked_path))
        container.close()


# =============================================================================
# Behavior Matrix Tests (from plan)
# =============================================================================

class TestBehaviorMatrix:
    """Tests for the behavior matrix from the plan."""

    def test_create_new_path_makes_directory(self, simple_shapefile, tmp_path):
        """create() on new path creates DirectoryStore."""
        container_path = tmp_path / "test.swim"
        assert not container_path.exists()

        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
        container.close()

        assert container_path.is_dir()

    def test_open_nonexistent_raises(self, tmp_path):
        """open() on non-existent path raises FileNotFoundError."""
        container_path = tmp_path / "nonexistent.swim"

        with pytest.raises(FileNotFoundError):
            open_container(str(container_path))

    def test_create_existing_file_without_overwrite_raises(
        self, simple_shapefile, tmp_path
    ):
        """create() on existing file without overwrite raises FileExistsError."""
        container_path = tmp_path / "test.swim"

        # Create first container
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
        container.close()

        # Try to create again without overwrite
        with pytest.raises(FileExistsError):
            SwimContainer.create(
                uri=str(container_path),
                fields_shapefile=str(simple_shapefile),
                uid_column="uid",
                start_date="2020-01-01",
                end_date="2020-12-31",
            )

    def test_create_existing_with_overwrite_works(self, simple_shapefile, tmp_path):
        """create() on existing path with overwrite=True replaces it."""
        container_path = tmp_path / "test.swim"

        # Create first container
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
        )
        container.close()

        # Create again with overwrite
        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2021-01-01",  # Different date
            end_date="2021-12-31",
            overwrite=True,
        )

        assert container.start_date.strftime("%Y-%m-%d") == "2021-01-01"
        container.close()

    def test_explicit_zip_storage_creates_file(self, simple_shapefile, tmp_path):
        """create() with storage='zip' creates zip file."""
        container_path = tmp_path / "test.swim"

        container = SwimContainer.create(
            uri=str(container_path),
            fields_shapefile=str(simple_shapefile),
            uid_column="uid",
            start_date="2020-01-01",
            end_date="2020-12-31",
            storage="zip",
        )
        container.close()

        assert container_path.is_file()
