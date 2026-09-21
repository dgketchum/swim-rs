"""Tests for explicit AWC units at ingest and the m/m guards downstream.

The container stores `properties/soils/awc` in m/m; the process model and the
PEST prior builder each multiply by 1000 to get mm/m. Sources delivered in
mm/m (HWSD v2) must be declared with `awc_units="mm/m"` at ingest.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zarr.core.dtype import VariableLengthUTF8

from swimrs.container.components.ingestor import Ingestor
from swimrs.container.inventory import Inventory
from swimrs.container.provenance import ProvenanceLog
from swimrs.container.state import ContainerState
from swimrs.container.storage import MemoryStoreProvider
from swimrs.units import assert_awc_m_per_m

MM_PER_M_AWC = [150.0, 214.0, 40.0]


def _make_container_state(n_fields=3):
    """Create an in-memory ContainerState with string UIDs '1'..'N'."""
    provider = MemoryStoreProvider(mode="w")
    root = provider.open()

    uids = [str(i) for i in range(1, n_fields + 1)]
    time_index = pd.date_range("2020-01-01", "2020-12-31", freq="D")

    time_grp = root.create_group("time")
    time_grp.create_array("daily", data=time_index.values.astype("datetime64[ns]"))

    geom_grp = root.create_group("geometry")
    uid_arr = geom_grp.create_array("uid", shape=(n_fields,), dtype=VariableLengthUTF8())
    uid_arr[:] = uids

    for grp in ("properties", "remote_sensing", "meteorology", "snow", "derived"):
        root.create_group(grp)

    state = ContainerState(
        provider=provider,
        field_uids=uids,
        time_index=time_index,
        provenance=ProvenanceLog(),
        inventory=Inventory(root, uids),
        mode="w",
    )
    return state, uids


def _write_soils_csv(tmpdir, awc_values):
    csv_path = Path(tmpdir) / "soils.csv"
    pd.DataFrame(
        {
            "FID": [1, 2, 3],
            "awc": awc_values,
            "ksat": [10.0, 11.0, 12.0],
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def test_ingest_mm_per_m_converted_and_attrs_recorded():
    """Declared mm/m source is divided by 1000 and the units are recorded."""
    state, _ = _make_container_state()
    ingestor = Ingestor(state, container=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _write_soils_csv(tmpdir, MM_PER_M_AWC)
        ingestor.properties(soils_csv=str(csv_path), uid_column="FID", awc_units="mm/m")

    awc = np.asarray(state.root["properties/soils/awc"][:])
    np.testing.assert_allclose(awc, [0.150, 0.214, 0.040], rtol=1e-5)

    attrs = state.root["properties/soils"].attrs
    assert attrs["awc_units_source"] == "mm/m"
    assert attrs["awc_units_stored"] == "m/m"


def test_ingest_m_per_m_source_stored_as_is():
    """An m/m source is stored unchanged and tagged as such."""
    state, _ = _make_container_state()
    ingestor = Ingestor(state, container=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _write_soils_csv(tmpdir, [0.15, 0.214, 0.04])
        ingestor.properties(soils_csv=str(csv_path), uid_column="FID")

    awc = np.asarray(state.root["properties/soils/awc"][:])
    np.testing.assert_allclose(awc, [0.15, 0.214, 0.04], rtol=1e-5)
    assert state.root["properties/soils"].attrs["awc_units_source"] == "m/m"


def test_ingest_mm_per_m_without_declaration_raises():
    """An mm/m CSV ingested with the default m/m declaration fails loudly."""
    state, _ = _make_container_state()
    ingestor = Ingestor(state, container=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _write_soils_csv(tmpdir, MM_PER_M_AWC)
        with pytest.raises(ValueError, match="AWC must be m/m in the container"):
            ingestor.properties(soils_csv=str(csv_path), uid_column="FID")


def test_ingest_rejects_unknown_awc_units():
    state, _ = _make_container_state()
    ingestor = Ingestor(state, container=None)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = _write_soils_csv(tmpdir, [0.15, 0.2, 0.04])
        with pytest.raises(ValueError, match="awc_units must be"):
            ingestor.properties(soils_csv=str(csv_path), uid_column="FID", awc_units="cm/m")


def test_helper_accepts_valid_and_all_nan():
    assert_awc_m_per_m([0.06, 0.42, np.nan], where="unit test")
    assert_awc_m_per_m([np.nan, np.nan], where="unit test")
    assert_awc_m_per_m([], where="unit test")


@pytest.mark.parametrize("bad", [[0.15, 150.0], [0.0, 0.2], [-0.1]])
def test_helper_raises_outside_unit_interval(bad):
    with pytest.raises(ValueError, match="AWC must be m/m in the container"):
        assert_awc_m_per_m(bad, where="unit test")


def test_helper_message_names_ingest_kwarg():
    with pytest.raises(ValueError) as exc:
        assert_awc_m_per_m([150.0], where="unit test")
    assert "awc_units='mm/m'" in str(exc.value)


def test_pest_builder_guard_rejects_mm_per_m():
    """The PEST prior path guards its AWC values before the x1000."""
    from swimrs.calibrate.pest_builder import PestBuilder

    builder = PestBuilder.__new__(PestBuilder)
    builder.plot_order = ["a", "b"]
    builder.plot_properties = {"a": {"awc": 150.0}, "b": {"awc": 214.0}}

    with pytest.raises(ValueError, match="AWC must be m/m in the container"):
        builder.get_pest_builder_args()


FIXTURE_SHP = (
    Path(__file__).parent.parent / "fixtures" / "S2" / "data" / "gis" / "flux_footprint_s2.shp"
)


def _build_minimal_container(tmp_path, awc_value):
    """Minimal container sufficient for build_swim_input (cf. test_runs.py)."""
    from swimrs.container import SwimContainer

    container = SwimContainer.create(
        str(tmp_path / "awc_units_test.swim"),
        fields_shapefile=str(FIXTURE_SHP),
        uid_column="site_id",
        start_date="2020-04-01",
        end_date="2020-04-05",
    )

    awc = container._create_property_array("properties/soils/awc")
    awc[:] = np.array([awc_value], dtype=np.float32)

    ksat = container._create_property_array("properties/soils/ksat")
    ksat[:] = np.array([10.0], dtype=np.float32)

    for path, value in [
        ("properties/land_cover/glc10", 10),
        ("properties/land_cover/modis_lc", 12),
    ]:
        arr = container._create_property_array(path, dtype="int16", fill_value=-1)
        arr[:] = np.array([value], dtype=np.int16)

    ndvi = container._create_timeseries_array("remote_sensing/ndvi/landsat/no_mask")
    ndvi[:] = np.array([[0.30], [0.35], [0.40], [0.45], [0.50]], dtype=np.float32)

    for path, values in {
        "meteorology/gridmet/prcp": [0.0, 2.0, 0.0, 1.0, 0.0],
        "meteorology/gridmet/tmin": [5.0, 6.0, 7.0, 8.0, 9.0],
        "meteorology/gridmet/tmax": [15.0, 16.0, 17.0, 18.0, 19.0],
        "meteorology/gridmet/srad": [18.0, 18.5, 19.0, 19.5, 20.0],
        "meteorology/gridmet/eto": [3.0, 3.2, 3.4, 3.6, 3.8],
    }.items():
        arr = container._create_timeseries_array(path)
        arr[:] = np.asarray(values, dtype=np.float32).reshape(-1, 1)

    container.save()
    return container


@pytest.mark.skipif(not FIXTURE_SHP.exists(), reason="S2 fixture shapefile not available")
def test_build_swim_input_raises_on_mm_per_m_container(tmp_path):
    """A container whose stored AWC is mm/m trips the SwimInput guard."""
    from swimrs.process.input import build_swim_input

    container = _build_minimal_container(tmp_path, 150.0)
    try:
        with pytest.raises(ValueError, match="AWC must be m/m in the container"):
            build_swim_input(container, output_h5=str(tmp_path / "input.h5"), mask_mode="none")
    finally:
        container.close()


@pytest.mark.skipif(not FIXTURE_SHP.exists(), reason="S2 fixture shapefile not available")
def test_build_swim_input_accepts_m_per_m_container(tmp_path):
    """The same container with m/m AWC passes the guard."""
    from swimrs.process.input import build_swim_input

    container = _build_minimal_container(tmp_path, 0.15)
    try:
        swim_input = build_swim_input(
            container, output_h5=str(tmp_path / "input.h5"), mask_mode="none"
        )
        assert swim_input is not None
    finally:
        container.close()
