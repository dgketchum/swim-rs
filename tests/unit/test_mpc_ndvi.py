"""Unit tests for the Planetary Computer NDVI extractor (no network)."""

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box

from swimrs.data_extraction.mpc import export, landsat, sentinel, zonal
from swimrs.data_extraction.mpc.grid import NODATA, GridSpec, block_reduce_mean
from swimrs.data_extraction.mpc.masks import IrrMapperMasks

UTM = CRS.from_epsg(32611)


def make_grid(width=4, height=4, res=30.0, origin=(500000.0, 4400000.0)):
    return GridSpec(UTM, from_origin(origin[0], origin[1], res, res), width, height)


class TestLandsatScience:
    def test_qa_clear_mask_bits(self):
        qa = np.array([0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6], dtype=np.uint16)
        radsat = np.zeros_like(qa)
        clear = landsat.qa_clear_mask(qa, radsat)
        # bits 1-5 masked; bit 6 (clear flag) is not itself tested
        assert clear.tolist() == [True, False, False, False, False, False, True]

    def test_qa_radsat_masks(self):
        qa = np.zeros(3, dtype=np.uint16)
        radsat = np.array([0, 1, 4], dtype=np.uint16)
        assert landsat.qa_clear_mask(qa, radsat).tolist() == [True, False, False]

    def test_sbaf_matches_ee_coefficients(self):
        from swimrs.data_extraction.ee.ee_utils import SBAF_COEFFICIENTS as EE_SBAF

        for sensor in ("TM", "ETM", "OLI"):
            assert landsat.SBAF_COEFFICIENTS[sensor] == EE_SBAF[sensor]
        assert sentinel.MSI_SBAF == EE_SBAF["MSI"]

    def test_harmonize_identity_for_oli(self):
        red = np.array([0.1])
        nir = np.array([0.4])
        red_h, nir_h = landsat.harmonize(red, nir, "landsat-8")
        assert red_h[0] == pytest.approx(0.1)
        assert nir_h[0] == pytest.approx(0.4)

    def test_harmonize_tm_and_etm_shift(self):
        red = np.array([0.1])
        nir = np.array([0.4])
        for platform in ("landsat-5", "landsat-7"):
            red_h, nir_h = landsat.harmonize(red, nir, platform)
            assert red_h[0] == pytest.approx(0.1 * 0.9047 + 0.0061)
            assert nir_h[0] == pytest.approx(0.4 * 0.8462 + 0.0412)


class TestSentinelScience:
    def test_dn_offset_by_baseline(self):
        assert sentinel.dn_offset({"properties": {"s2:processing_baseline": "05.00"}}) == 1000
        assert sentinel.dn_offset({"properties": {"s2:processing_baseline": "04.00"}}) == 1000
        assert sentinel.dn_offset({"properties": {"s2:processing_baseline": "03.01"}}) == 0
        assert sentinel.dn_offset({"properties": {}}) == 0

    def test_scl_clear_mask(self):
        scl = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        clear = sentinel.scl_clear_mask(scl)
        masked_classes = set(scl[~clear].tolist())
        assert masked_classes == {3, 8, 9, 10}

    def test_block_reduce_mean_nodata_aware(self):
        arr = np.array([[10, 20], [0, 30]], dtype=np.uint16)  # 0 = nodata
        out = block_reduce_mean(arr, 2, nodata=0)
        assert out.shape == (1, 1)
        assert out[0, 0] == pytest.approx((10 + 20 + 30) / 3)


class TestIrrMapperMasks:
    @pytest.fixture
    def masks_dir(self, tmp_path):
        """Annual rasters on the same grid as make_grid(): 0=irr, 1/2 other."""
        grid = make_grid()
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "width": grid.width,
            "height": grid.height,
            "crs": grid.crs,
            "transform": grid.transform,
        }
        # 2022: left half irrigated; 2023: only top-left pixel irrigated
        for year, cls in (
            (
                2022,
                np.where(np.arange(4)[None, :] < 2, 0, 1).astype(np.uint8)
                * np.ones((4, 4), dtype=np.uint8),
            ),
            (2023, np.pad(np.zeros((1, 1), np.uint8), ((0, 3), (0, 3)), constant_values=2)),
        ):
            with rasterio.open(tmp_path / f"irrmapper_NV_{year}.tif", "w", **profile) as dst:
                dst.write(cls, 1)
        # min-yr mask: everything qualifies except the bottom row
        min_yr = np.ones((4, 4), dtype=np.uint8)
        min_yr[3, :] = 0
        with rasterio.open(tmp_path / "irr_min_yr_mask.tif", "w", **profile) as dst:
            dst.write(min_yr, 1)
        return tmp_path

    def test_irr_requires_both_year_and_min_yr(self, masks_dir):
        masks = IrrMapperMasks(masks_dir)
        grid = make_grid()
        irr = masks.mask_for("irr", 2022, grid)
        assert irr[:3, :2].all()  # left half, min-yr ok
        assert not irr[:, 2:].any()  # right half not irrigated in 2022
        assert not irr[3, :].any()  # bottom row fails min-yr

    def test_inv_irr_ignores_min_yr(self, masks_dir):
        masks = IrrMapperMasks(masks_dir)
        grid = make_grid()
        inv = masks.mask_for("inv_irr", 2022, grid)
        assert not inv[:, :2].any()
        assert inv[:, 2:].all()  # includes bottom row: min-yr not consulted

    def test_year_capped_at_irr_max_year(self, masks_dir):
        masks = IrrMapperMasks(masks_dir)
        grid = make_grid()
        assert np.array_equal(masks.mask_for("irr", 2025, grid), masks.mask_for("irr", 2023, grid))

    def test_no_mask_is_all_true(self, masks_dir):
        masks = IrrMapperMasks(masks_dir)
        assert IrrMapperMasks(masks_dir).mask_for("no_mask", 2022, make_grid()).all()
        assert masks.mask_for("no_mask", 1999, make_grid()).all()

    def test_min_yr_built_from_annuals_when_absent(self, masks_dir):
        (masks_dir / "irr_min_yr_mask.tif").unlink()
        masks = IrrMapperMasks(masks_dir, min_years=1)
        grid = make_grid()
        built = masks.min_yr_mask(grid)
        # 2022 or 2023 irrigated at least once in the left column
        assert built[0, 0] == 1
        assert built[:, 2:].sum() == 0


class TestZonal:
    def test_coverage_weighted_mean(self):
        import geopandas as gpd

        grid = make_grid(width=2, height=1)
        values = np.array([[0.2, 0.6]], dtype=np.float32)
        # polygon covers all of pixel 1 and half of pixel 2
        x0, _, _, y1 = grid.bounds
        poly = box(x0, y1 - 30, x0 + 45, y1)
        gdf = gpd.GeoDataFrame({"fid": ["a"], "geometry": [poly]}, crs=UTM)
        df = zonal.field_means(values, grid, gdf, "fid")
        expected = (0.2 * 1.0 + 0.6 * 0.5) / 1.5
        assert df["mean"][0] == pytest.approx(expected, abs=1e-6)
        assert df["count"][0] == pytest.approx(1.5, abs=1e-6)

    def test_fully_masked_field_is_nan(self):
        import geopandas as gpd

        grid = make_grid(width=2, height=1)
        values = np.full((1, 2), NODATA, dtype=np.float32)
        x0, _, x1, y1 = grid.bounds
        gdf = gpd.GeoDataFrame({"fid": ["a"], "geometry": [box(x0, y1 - 30, x1, y1)]}, crs=UTM)
        df = zonal.field_means(values, grid, gdf, "fid")
        assert np.isnan(df["mean"][0])
        assert df["count"][0] == 0


class TestPartitioning:
    def test_chunk_rule_matches_etf_layout(self):
        import geopandas as gpd

        rows = [{"FIPS": "32001", "OPENET_ID": f"NV_{i}"} for i in range(1800)]
        rows += [{"FIPS": "32009", "OPENET_ID": f"NV_{9000 + i}"} for i in range(900)]
        gdf = gpd.GeoDataFrame(rows)
        parts = export.partition_fields(gdf, "OPENET_ID")
        labels = [lbl for lbl, _ in parts]
        assert labels == ["32001a", "32001b", "32009"]
        assert len(parts[0][1]) == 900
        assert len(parts[1][1]) == 900

    def test_build_targets_subtracts_existing(self):
        parts = [("32001a", ["NV_1"]), ("32009", ["NV_2"])]
        existing = {("32001a", "irr", 2020)}
        missing, per_year = export.build_targets(parts, ["irr"], [2020, 2021], existing)
        assert ("32001a", "irr", 2020) not in missing
        assert ("32009", "irr", 2020) in missing
        assert per_year[2020]["fields"] == {"NV_2"}
        assert per_year[2021]["fields"] == {"NV_1", "NV_2"}


class TestCsvContract:
    @pytest.fixture
    def records(self, tmp_path):
        year_dir = tmp_path / "landsat" / "2016"
        year_dir.mkdir(parents=True)
        s1 = pd.DataFrame(
            {
                "OPENET_ID": ["NV_1", "NV_2", "NV_2"],
                "_row": [0, 1, 2],
                "mean": [0.31, 0.42, np.nan],
                "count": [10.0, 8.0, 0.0],
                "mask": ["irr"] * 3,
                "scene_id": ["LE07_040030_20160702"] * 3,
            }
        )
        s2 = pd.DataFrame(
            {
                "OPENET_ID": ["NV_1"],
                "_row": [0],
                "mean": [0.55],
                "count": [10.0],
                "mask": ["irr"],
                "scene_id": ["LC08_040030_20160401"] * 1,
            }
        )
        s1.to_parquet(year_dir / "LE07_040030_20160702.parquet")
        s2.to_parquet(year_dir / "LC08_040030_20160401.parquet")
        return tmp_path

    def test_assemble_contract_and_ingest_roundtrip(self, records, tmp_path):
        out = tmp_path / "out" / "ndvi_irr_2016.csv"
        field_rows = [(0, "NV_1"), (1, "NV_2"), (2, "NV_2")]
        export.assemble_csv(records, "landsat", 2016, "irr", field_rows, "OPENET_ID", out)

        df = pd.read_csv(out)
        # chronological columns, duplicate-id rows preserved, empty cell kept
        assert list(df.columns) == ["OPENET_ID", "LC08_040030_20160401", "LE07_040030_20160702"]
        assert df["OPENET_ID"].tolist() == ["NV_1", "NV_2", "NV_2"]
        assert df["LE07_040030_20160702"].tolist()[0] == pytest.approx(0.31)
        assert np.isnan(df.loc[2, "LE07_040030_20160702"])
        assert np.isnan(df.loc[1, "LC08_040030_20160401"])

        from swimrs.container.components.ingestor import _parse_single_csv

        series = _parse_single_csv(out, "OPENET_ID", "landsat", {"NV_1", "NV_2"}, None)
        by_field = {s.name: s for s in series}
        assert set(by_field) == {"NV_1", "NV_2"}
        assert by_field["NV_1"][pd.Timestamp("2016-04-01")] == pytest.approx(0.55)
        assert by_field["NV_1"][pd.Timestamp("2016-07-02")] == pytest.approx(0.31)

    def test_column_sort_key_handles_both_instruments(self):
        cols = [
            "LE07_040030_20160702",
            "20160104T185801_20160104T185757_T10SFH",
            "LC08_040030_20160401",
        ]
        ordered = sorted(cols, key=export._column_date_key)
        assert ordered[0].startswith("20160104")
        assert ordered[1] == "LC08_040030_20160401"

    def test_csv_path_layouts(self, tmp_path):
        p_ls = export.csv_path(tmp_path, "landsat", "32001a", "irr", 2016)
        p_s2 = export.csv_path(tmp_path, "sentinel", "32001a", "inv_irr", 2019)
        assert str(p_ls).endswith("32001a/ndvi/irr/ndvi_irr_2016.csv")
        assert str(p_s2).endswith("32001a/ndvi_s2/inv_irr/ndvi_s2_inv_irr_2019.csv")
