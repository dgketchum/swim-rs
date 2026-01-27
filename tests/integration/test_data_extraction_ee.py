"""Tests for swimrs.data_extraction.ee pure logic functions.

Note: The ee module is mocked in conftest.py to allow testing pure logic
functions without Earth Engine authentication.
"""

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from swimrs.data_extraction.ee.common import parse_scene_name, load_shapefile
from swimrs.data_extraction.ee.etf_export import get_utm_epsg


class TestParseSceneName:
    """Tests for parse_scene_name function."""

    def test_landsat8_scene(self):
        """Parse standard Landsat 8 scene ID."""
        img_id = "LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716"
        result = parse_scene_name(img_id)
        assert result == "LC08_044033_20170716"

    def test_landsat5_scene(self):
        """Parse Landsat 5 scene ID."""
        img_id = "LANDSAT/LT05/C02/T1_L2/LT05_035032_19990815"
        result = parse_scene_name(img_id)
        assert result == "LT05_035032_19990815"

    def test_landsat7_scene(self):
        """Parse Landsat 7 scene ID."""
        img_id = "LANDSAT/LE07/C02/T1_L2/LE07_038029_20100601"
        result = parse_scene_name(img_id)
        assert result == "LE07_038029_20100601"

    def test_landsat9_scene(self):
        """Parse Landsat 9 scene ID."""
        img_id = "LANDSAT/LC09/C02/T1_L2/LC09_044033_20220115"
        result = parse_scene_name(img_id)
        assert result == "LC09_044033_20220115"

    def test_extracts_last_three_parts(self):
        """Function extracts last 3 underscore-separated parts."""
        # Even with unusual path, should get last 3 parts
        img_id = "some/path/LC08_044033_20170716"
        result = parse_scene_name(img_id)
        assert result == "LC08_044033_20170716"

    def test_short_scene_name(self):
        """Handle scene name that already has minimal parts."""
        img_id = "LC08_044033_20170716"
        result = parse_scene_name(img_id)
        assert result == "LC08_044033_20170716"


class TestGetUtmEpsg:
    """Tests for get_utm_epsg function."""

    def test_northern_hemisphere_zone_12(self):
        """Test UTM zone 12N (Montana)."""
        lat, lon = 46.0, -110.0  # Montana
        epsg, zone_str = get_utm_epsg(lat, lon)
        assert epsg == 32612
        assert zone_str == "12N"

    def test_northern_hemisphere_zone_10(self):
        """Test UTM zone 10N (California coast)."""
        lat, lon = 37.0, -122.0  # San Francisco area
        epsg, zone_str = get_utm_epsg(lat, lon)
        assert epsg == 32610
        assert zone_str == "10N"

    def test_southern_hemisphere(self):
        """Test southern hemisphere UTM zone."""
        lat, lon = -33.9, 18.4  # Cape Town, South Africa
        epsg, zone_str = get_utm_epsg(lat, lon)
        assert epsg == 32734  # Zone 34S
        assert zone_str == "34S"

    def test_zone_boundary_west(self):
        """Test near zone boundary in western US."""
        lat, lon = 40.0, -105.0  # Boulder, CO area
        epsg, zone_str = get_utm_epsg(lat, lon)
        assert 32600 < epsg < 32700  # Northern hemisphere
        assert zone_str.endswith("N")

    def test_eastern_us(self):
        """Test eastern US location."""
        lat, lon = 40.7, -74.0  # New York City
        epsg, zone_str = get_utm_epsg(lat, lon)
        assert epsg == 32618  # Zone 18N
        assert zone_str == "18N"

    def test_returns_tuple(self):
        """get_utm_epsg returns a tuple of (int, str)."""
        result = get_utm_epsg(45.0, -110.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], str)

    def test_equator_northern_letter(self):
        """Test location at equator with northern zone letter."""
        lat, lon = 0.5, -80.0  # Ecuador
        epsg, zone_str = get_utm_epsg(lat, lon)
        assert zone_str.endswith("N")

    def test_equator_southern_letter(self):
        """Test location just south of equator."""
        lat, lon = -0.5, -80.0  # Southern Ecuador
        epsg, zone_str = get_utm_epsg(lat, lon)
        # Zone letter < 'N' means southern
        assert zone_str.endswith("S")


class TestLoadShapefile:
    """Tests for load_shapefile function."""

    @pytest.fixture
    def simple_shapefile(self, tmp_path):
        """Create a simple test shapefile."""
        gdf = gpd.GeoDataFrame(
            {
                "FID": ["field_001", "field_002", "field_003"],
                "name": ["A", "B", "C"],
                "geometry": [
                    Polygon([(-110, 45), (-109, 45), (-109, 46), (-110, 46)]),
                    Polygon([(-108, 44), (-107, 44), (-107, 45), (-108, 45)]),
                    Polygon([(-106, 43), (-105, 43), (-105, 44), (-106, 44)]),
                ],
            },
            crs="EPSG:4326",
        )
        shp_path = tmp_path / "test_fields.shp"
        gdf.to_file(shp_path)
        return shp_path

    @pytest.fixture
    def shapefile_5071(self, tmp_path):
        """Create a shapefile in EPSG:5071 (Albers Equal Area)."""
        # Create in 4326 first, then reproject
        gdf = gpd.GeoDataFrame(
            {
                "FID": ["field_001", "field_002"],
                "name": ["A", "B"],
                "geometry": [
                    Polygon([(-110, 45), (-109, 45), (-109, 46), (-110, 46)]),
                    Polygon([(-108, 44), (-107, 44), (-107, 45), (-108, 45)]),
                ],
            },
            crs="EPSG:4326",
        )
        gdf = gdf.to_crs("EPSG:5071")
        shp_path = tmp_path / "test_fields_5071.shp"
        gdf.to_file(shp_path)
        return shp_path

    def test_loads_shapefile(self, simple_shapefile):
        """load_shapefile returns GeoDataFrame."""
        result = load_shapefile(str(simple_shapefile), "FID")
        assert isinstance(result, gpd.GeoDataFrame)

    def test_sets_index_to_feature_id(self, simple_shapefile):
        """load_shapefile sets index to feature_id column."""
        result = load_shapefile(str(simple_shapefile), "FID")
        assert list(result.index) == ["field_001", "field_002", "field_003"]

    def test_converts_to_4326(self, shapefile_5071):
        """load_shapefile converts CRS to EPSG:4326."""
        result = load_shapefile(str(shapefile_5071), "FID")
        assert result.crs.to_epsg() == 4326

    def test_already_4326_unchanged(self, simple_shapefile):
        """load_shapefile preserves 4326 CRS."""
        result = load_shapefile(str(simple_shapefile), "FID")
        assert result.crs.to_epsg() == 4326

    def test_applies_buffer(self, simple_shapefile):
        """load_shapefile applies buffer when specified."""
        no_buffer = load_shapefile(str(simple_shapefile), "FID")
        with_buffer = load_shapefile(str(simple_shapefile), "FID", buffer=0.01)

        # Buffered geometry should be larger
        assert with_buffer.geometry.area.sum() > no_buffer.geometry.area.sum()

    def test_preserves_columns(self, simple_shapefile):
        """load_shapefile preserves original columns."""
        result = load_shapefile(str(simple_shapefile), "FID")
        assert "FID" in result.columns
        assert "name" in result.columns

    def test_returns_geodataframe_type(self, simple_shapefile):
        """load_shapefile returns correct type."""
        result = load_shapefile(str(simple_shapefile), "FID")
        assert isinstance(result, gpd.GeoDataFrame)
        assert hasattr(result, "geometry")


class TestLoadShapefileEdgeCases:
    """Edge case tests for load_shapefile."""

    @pytest.fixture
    def point_shapefile(self, tmp_path):
        """Create a point shapefile."""
        gdf = gpd.GeoDataFrame(
            {
                "FID": ["pt_001", "pt_002"],
                "name": ["P1", "P2"],
                "geometry": [Point(-110, 45), Point(-108, 44)],
            },
            crs="EPSG:4326",
        )
        shp_path = tmp_path / "test_points.shp"
        gdf.to_file(shp_path)
        return shp_path

    def test_point_geometry(self, point_shapefile):
        """load_shapefile handles point geometries."""
        result = load_shapefile(str(point_shapefile), "FID")
        assert len(result) == 2

    def test_point_with_buffer(self, point_shapefile):
        """load_shapefile can buffer points."""
        result = load_shapefile(str(point_shapefile), "FID", buffer=0.1)
        # Points become polygons after buffering
        assert result.geometry.geom_type.iloc[0] == "Polygon"


class TestParseSceneNameEdgeCases:
    """Edge case tests for parse_scene_name."""

    def test_empty_path_prefix(self):
        """Handle scene with no path prefix."""
        img_id = "LC08_044033_20170716"
        result = parse_scene_name(img_id)
        assert result == "LC08_044033_20170716"

    def test_deep_path(self):
        """Handle deeply nested path."""
        img_id = "a/b/c/d/e/f/LC08_044033_20170716"
        result = parse_scene_name(img_id)
        assert result == "LC08_044033_20170716"
