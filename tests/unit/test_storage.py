"""
Fast unit tests for container storage helpers.

These tests intentionally avoid creating full Zarr containers (which is slow and
covered by higher-level workflows) and instead validate the parts that are easy
to regress: storage type detection, provider selection, and pack/unpack behavior
for local paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swimrs.container import SwimContainer
from swimrs.container.storage import (
    DirectoryStoreProvider,
    StorageProviderFactory,
    ZipStoreProvider,
    detect_storage_type,
)


def _dummy_container(*, provider, modified: bool = False) -> SwimContainer:
    container = SwimContainer.__new__(SwimContainer)
    container._provider = provider
    container._modified = modified
    return container


class TestDetectStorageType:
    def test_detects_existing_file_as_zip(self, tmp_path):
        file_path = tmp_path / "test.swim"
        file_path.write_text("dummy")
        assert detect_storage_type(file_path) == "zip"

    def test_detects_existing_directory_as_directory(self, tmp_path):
        dir_path = tmp_path / "test.swim"
        dir_path.mkdir()
        assert detect_storage_type(dir_path) == "directory"

    def test_new_path_defaults_to_directory(self, tmp_path):
        new_path = tmp_path / "new_container.swim"
        assert not new_path.exists()
        assert detect_storage_type(new_path) == "directory"


class TestStorageProviderFactory:
    def test_new_path_creates_directory_provider(self, tmp_path):
        new_path = tmp_path / "new.swim"
        provider = StorageProviderFactory.from_uri(new_path, mode="w")
        assert isinstance(provider, DirectoryStoreProvider)

    def test_existing_file_creates_zip_provider(self, tmp_path):
        file_path = tmp_path / "test.swim"
        file_path.write_text("dummy")
        provider = StorageProviderFactory.from_uri(file_path, mode="r")
        assert isinstance(provider, ZipStoreProvider)

    def test_existing_directory_creates_directory_provider(self, tmp_path):
        dir_path = tmp_path / "test.swim"
        dir_path.mkdir()
        provider = StorageProviderFactory.from_uri(dir_path, mode="r")
        assert isinstance(provider, DirectoryStoreProvider)

    def test_explicit_zip_storage(self, tmp_path):
        new_path = tmp_path / "new.swim"
        provider = StorageProviderFactory.from_uri(new_path, mode="w", storage="zip")
        assert isinstance(provider, ZipStoreProvider)

    def test_explicit_directory_storage(self, tmp_path):
        new_path = tmp_path / "new.swim"
        provider = StorageProviderFactory.from_uri(new_path, mode="w", storage="directory")
        assert isinstance(provider, DirectoryStoreProvider)

    def test_invalid_storage_raises(self, tmp_path):
        new_path = tmp_path / "new.swim"
        with pytest.raises(ValueError, match="Invalid storage type"):
            StorageProviderFactory.from_uri(new_path, mode="w", storage="invalid")


class TestPackUnpack:
    def test_pack_directory_to_zip_creates_archive(self, tmp_path):
        source_dir = tmp_path / "source.swim"
        source_dir.mkdir()
        (source_dir / "file.txt").write_text("hello")

        provider = DirectoryStoreProvider(source_dir, mode="r")
        container = _dummy_container(provider=provider)

        out = container.pack(tmp_path / "packed.swim")
        assert out == tmp_path / "packed.swim"
        assert out.is_file()

    def test_pack_zip_copies(self, tmp_path):
        src = tmp_path / "src.swim"
        src.write_text("hello")

        provider = ZipStoreProvider(src, mode="r")
        container = _dummy_container(provider=provider)

        out = container.pack(tmp_path / "copy.swim")
        assert out.is_file()
        assert out.read_text() == "hello"

    def test_pack_fails_if_output_exists(self, tmp_path):
        src_dir = tmp_path / "source.swim"
        src_dir.mkdir()
        provider = DirectoryStoreProvider(src_dir, mode="r")
        container = _dummy_container(provider=provider)

        out = tmp_path / "packed.swim"
        out.write_text("existing")

        with pytest.raises(FileExistsError):
            container.pack(out)

    def test_unpack_zip_extracts_and_opens(self, tmp_path, monkeypatch):
        import zipfile

        src = tmp_path / "src.swim"
        with zipfile.ZipFile(src, "w") as zf:
            zf.writestr("file.txt", "hello")

        provider = ZipStoreProvider(src, mode="r")
        container = _dummy_container(provider=provider)

        opened: list[Path] = []

        def fake_open(path, mode="r+", **kwargs):
            opened.append(Path(path))
            return "OPENED"

        monkeypatch.setattr(SwimContainer, "open", staticmethod(fake_open))

        out_dir = tmp_path / "unpacked.swim"
        result = container.unpack(out_dir)

        assert result == "OPENED"
        assert out_dir.is_dir()
        assert (out_dir / "file.txt").read_text() == "hello"
        assert opened == [out_dir]

    def test_unpack_directory_copies_and_opens(self, tmp_path, monkeypatch):
        src_dir = tmp_path / "src.swim"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("hello")

        provider = DirectoryStoreProvider(src_dir, mode="r")
        container = _dummy_container(provider=provider)

        opened: list[Path] = []

        def fake_open(path, mode="r+", **kwargs):
            opened.append(Path(path))
            return "OPENED"

        monkeypatch.setattr(SwimContainer, "open", staticmethod(fake_open))

        out_dir = tmp_path / "copied.swim"
        result = container.unpack(out_dir)

        assert result == "OPENED"
        assert out_dir.is_dir()
        assert (out_dir / "file.txt").read_text() == "hello"
        assert opened == [out_dir]

    def test_unpack_fails_if_output_exists(self, tmp_path):
        import zipfile

        src = tmp_path / "src.swim"
        with zipfile.ZipFile(src, "w") as zf:
            zf.writestr("file.txt", "hello")

        provider = ZipStoreProvider(src, mode="r")
        container = _dummy_container(provider=provider)

        out_dir = tmp_path / "unpacked.swim"
        out_dir.mkdir()

        with pytest.raises(FileExistsError):
            container.unpack(out_dir)


class TestCreateValidation:
    def test_create_raises_if_uid_column_missing(self, tmp_path, monkeypatch):
        import geopandas as gpd
        import pandas as pd

        monkeypatch.setattr(gpd, "read_file", lambda *args, **kwargs: pd.DataFrame({"x": [1]}))

        out = tmp_path / "test.swim"
        with pytest.raises(ValueError, match="UID column"):
            SwimContainer.create(
                uri=str(out),
                fields_shapefile="dummy.shp",
                uid_column="uid",
                start_date="2020-01-01",
                end_date="2020-01-03",
            )

        assert not out.exists()

    def test_create_raises_on_duplicate_uids(self, tmp_path, monkeypatch):
        import geopandas as gpd
        import pandas as pd

        monkeypatch.setattr(
            gpd,
            "read_file",
            lambda *args, **kwargs: pd.DataFrame({"uid": ["a", "a"]}),
        )

        out = tmp_path / "test.swim"
        with pytest.raises(ValueError, match="duplicate"):
            SwimContainer.create(
                uri=str(out),
                fields_shapefile="dummy.shp",
                uid_column="uid",
                start_date="2020-01-01",
                end_date="2020-01-03",
            )

        assert not out.exists()
