"""Integration tests for data_extraction external services.

Run with:
    pytest -m integration tests/test_data_extraction_integration.py -v

To include Earth Engine tests (requires authentication):
    pytest -m integration --run-ee tests/test_data_extraction_integration.py -v
"""

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestGridMetIntegration:
    """Integration tests for GridMet THREDDS service."""

    pytestmark = [pytest.mark.integration, pytest.mark.requires_network]

    def test_get_point_elevation(self):
        """Fetch elevation for a single point from THREDDS."""
        import numpy as np

        from swimrs.data_extraction.gridmet.thredds import GridMet

        # Fort Peck, Montana (flux tower site)
        lat, lon = 48.3077, -105.1019

        gm = GridMet(variable="elev", lat=lat, lon=lon)
        elev = gm.get_point_elevation()

        assert isinstance(elev, float | np.floating)
        assert 600 < elev < 800  # Fort Peck is ~634m

    def test_get_point_timeseries_july(self):
        """Fetch one month of ETr data from THREDDS."""
        from swimrs.data_extraction.gridmet.thredds import GridMet

        lat, lon = 48.3077, -105.1019

        gm = GridMet(
            variable="etr",
            lat=lat,
            lon=lon,
            start="2020-07-01",
            end="2020-07-31",
        )
        df = gm.get_point_timeseries()

        assert len(df) == 31
        assert "etr" in df.columns
        assert df["etr"].mean() > 0  # ETr should be positive in July

    def test_get_point_timeseries_precip(self):
        """Fetch precipitation data from THREDDS."""
        from swimrs.data_extraction.gridmet.thredds import GridMet

        lat, lon = 48.3077, -105.1019

        gm = GridMet(
            variable="pr",
            lat=lat,
            lon=lon,
            start="2020-07-01",
            end="2020-07-31",
        )
        df = gm.get_point_timeseries()

        assert len(df) == 31
        assert "pr" in df.columns
        assert df["pr"].min() >= 0  # Precip can't be negative


@pytest.mark.requires_ee
class TestEarthEngineIntegration:
    """Integration tests for Earth Engine functions.

    These tests require ee.Initialize() to have been called.
    Run with: pytest --run-ee
    """

    @pytest.fixture(autouse=True)
    def init_ee(self):
        """Initialize Earth Engine, skip if unavailable."""
        try:
            import ee

            ee.Initialize()
        except Exception as e:
            pytest.skip(f"Earth Engine not available: {e}")

    def test_get_lanid_asset(self):
        """Load LANID irrigation dataset from EE."""
        import ee

        from swimrs.data_extraction.ee.common import get_lanid

        lanid = get_lanid()

        try:
            info = lanid.getInfo()
            assert info["type"] == "Image"
        except ee.ee_exception.EEException as e:
            if "not found" in str(e) or "does not have access" in str(e):
                pytest.skip(f"LANID asset not accessible: {e}")
            raise

    def test_load_shapefile_to_fc(self, tmp_path):
        """Load a local shapefile for EE processing."""
        import geopandas as gpd
        from shapely.geometry import Polygon

        from swimrs.data_extraction.ee.common import load_shapefile

        # Create a test shapefile (single field in Montana)
        gdf = gpd.GeoDataFrame(
            {
                "FID": ["test_001"],
                "geometry": [
                    Polygon([(-105.2, 48.3), (-105.1, 48.3), (-105.1, 48.4), (-105.2, 48.4)])
                ],
            },
            crs="EPSG:4326",
        )
        shp_path = tmp_path / "test_field.shp"
        gdf.to_file(shp_path)

        result = load_shapefile(str(shp_path), "FID")

        assert len(result) == 1
        assert result.index[0] == "test_001"
