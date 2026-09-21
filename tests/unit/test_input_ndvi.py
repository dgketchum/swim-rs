"""Tests for NDVI consolidation in build_swim_input()."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from zarr.core.dtype import VariableLengthUTF8

from swimrs.container import SwimContainer
from swimrs.process.input import build_swim_input

FIXTURE_SHP = (
    Path(__file__).parent.parent / "fixtures" / "S2" / "data" / "gis" / "flux_footprint_s2.shp"
)


def _build_irrigation_ndvi_container(
    tmp_path,
    *,
    ndvi_irr: list[float],
    ndvi_inv_irr: list[float],
) -> tuple[SwimContainer, Path]:
    container_path = tmp_path / "ndvi_input_test.swim"
    container = SwimContainer.create(
        str(container_path),
        fields_shapefile=str(FIXTURE_SHP),
        uid_column="site_id",
        start_date="2020-01-01",
        end_date="2020-01-05",
    )

    awc = container._create_property_array("properties/soils/awc")
    awc[:] = np.array([0.15], dtype=np.float32)

    ksat = container._create_property_array("properties/soils/ksat")
    ksat[:] = np.array([10.0], dtype=np.float32)

    glc10 = container._create_property_array(
        "properties/land_cover/glc10",
        dtype="int16",
        fill_value=-1,
    )
    glc10[:] = np.array([10], dtype=np.int16)

    modis = container._create_property_array(
        "properties/land_cover/modis_lc",
        dtype="int16",
        fill_value=-1,
    )
    modis[:] = np.array([12], dtype=np.int16)

    ndvi_irr_arr = container._create_timeseries_array("remote_sensing/ndvi/landsat/irr")
    ndvi_irr_arr[:] = np.asarray(ndvi_irr, dtype=np.float32).reshape(-1, 1)

    ndvi_inv_arr = container._create_timeseries_array("remote_sensing/ndvi/landsat/inv_irr")
    ndvi_inv_arr[:] = np.asarray(ndvi_inv_irr, dtype=np.float32).reshape(-1, 1)

    for path, values in {
        "meteorology/gridmet/prcp": [0.0, 0.0, 0.0, 0.0, 0.0],
        "meteorology/gridmet/tmin": [5.0, 5.0, 5.0, 5.0, 5.0],
        "meteorology/gridmet/tmax": [15.0, 15.0, 15.0, 15.0, 15.0],
        "meteorology/gridmet/srad": [20.0, 20.0, 20.0, 20.0, 20.0],
        "meteorology/gridmet/eto": [3.0, 3.0, 3.0, 3.0, 3.0],
    }.items():
        arr = container._create_timeseries_array(path)
        arr[:] = np.asarray(values, dtype=np.float32).reshape(-1, 1)

    dyn = container._root.require_group("derived/dynamics")
    irr_data = dyn.create_array("irr_data", shape=(1,), dtype=VariableLengthUTF8())
    irr_data[:] = [json.dumps({"2020": {"f_irr": 0.5, "irr_doys": []}})]

    container.save()
    return container, container_path


def test_irrigated_year_falls_back_to_inv_irr_when_obs_are_insufficient(tmp_path):
    container, _ = _build_irrigation_ndvi_container(
        tmp_path,
        ndvi_irr=[0.70, np.nan, np.nan, np.nan, 0.90],
        ndvi_inv_irr=[0.20, 0.21, 0.22, 0.23, 0.24],
    )

    try:
        swim_input = build_swim_input(
            container=container,
            output_h5=tmp_path / "fallback.h5",
            mask_mode="irrigation",
            met_source="gridmet",
        )
        try:
            ndvi = swim_input.get_time_series("ndvi")[:, 0]
        finally:
            swim_input.close()

        np.testing.assert_allclose(ndvi, np.array([0.20, 0.21, 0.22, 0.23, 0.24]))
    finally:
        container.close()


def test_irrigated_year_uses_irr_when_obs_threshold_is_met(tmp_path):
    container, _ = _build_irrigation_ndvi_container(
        tmp_path,
        ndvi_irr=[0.70, np.nan, 0.80, np.nan, 0.90],
        ndvi_inv_irr=[0.20, 0.21, 0.22, 0.23, 0.24],
    )

    try:
        swim_input = build_swim_input(
            container=container,
            output_h5=tmp_path / "use_irr.h5",
            mask_mode="irrigation",
            met_source="gridmet",
        )
        try:
            ndvi = swim_input.get_time_series("ndvi")[:, 0]
        finally:
            swim_input.close()

        np.testing.assert_allclose(ndvi, np.array([0.70, 0.75, 0.80, 0.85, 0.90]))
    finally:
        container.close()


def test_build_swim_input_fails_when_merged_ndvi_remains_nonfinite(tmp_path):
    container, _ = _build_irrigation_ndvi_container(
        tmp_path,
        ndvi_irr=[np.nan, np.nan, np.nan, np.nan, np.nan],
        ndvi_inv_irr=[np.nan, np.nan, np.nan, np.nan, np.nan],
    )

    try:
        with pytest.raises(ValueError, match="Consolidated NDVI contains non-finite values"):
            build_swim_input(
                container=container,
                output_h5=tmp_path / "bad_ndvi.h5",
                mask_mode="irrigation",
                met_source="gridmet",
            )
    finally:
        container.close()


def test_build_swim_input_writes_configured_max_irr_rate(tmp_path):
    container, _ = _build_irrigation_ndvi_container(
        tmp_path,
        ndvi_irr=[0.70, np.nan, 0.80, np.nan, 0.90],
        ndvi_inv_irr=[0.20, 0.21, 0.22, 0.23, 0.24],
    )

    try:
        swim_input = build_swim_input(
            container=container,
            output_h5=tmp_path / "max_irr_rate.h5",
            mask_mode="irrigation",
            met_source="gridmet",
            max_irr_rate=25.0,
        )
        try:
            np.testing.assert_allclose(swim_input.parameters.max_irr_rate, np.array([25.0]))
        finally:
            swim_input.close()
    finally:
        container.close()
