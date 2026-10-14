"""Tests for shared ETf validity bounds in container ingestion."""

import numpy as np
import pandas as pd
import pytest
from zarr.core.dtype import VariableLengthUTF8

from swimrs.container.components.ingestor import Ingestor
from swimrs.container.inventory import Inventory
from swimrs.container.provenance import ProvenanceLog
from swimrs.container.state import ContainerState
from swimrs.container.storage import MemoryStoreProvider


def _make_container_state():
    """Create an in-memory container state for ETf ingestion tests."""
    provider = MemoryStoreProvider(mode="w")
    root = provider.open()

    uids = ["US-FPe"]
    time_index = pd.date_range("2020-01-01", periods=3, freq="D")

    time_grp = root.create_group("time")
    time_grp.create_array("daily", data=time_index.values.astype("datetime64[ns]"))

    geom_grp = root.create_group("geometry")
    uid_arr = geom_grp.create_array("uid", shape=(1,), dtype=VariableLengthUTF8())
    uid_arr[:] = uids

    root.create_group("properties")
    root.create_group("remote_sensing")
    root.create_group("meteorology")
    root.create_group("snow")
    root.create_group("derived")

    provenance = ProvenanceLog()
    inventory = Inventory(root, uids)

    return ContainerState(
        provider=provider,
        field_uids=uids,
        time_index=time_index,
        provenance=provenance,
        inventory=inventory,
        mode="w",
    )


def test_ingest_etf_masks_values_outside_valid_range(tmp_path):
    """ETf ingest should drop values below min_etf and above the shared ceiling."""
    state = _make_container_state()
    ingestor = Ingestor(state, container=None)

    csv_path = tmp_path / "ptjpl_etf_US-FPe_no_mask_2020.csv"
    pd.DataFrame(
        {
            "sid": ["US-FPe"],
            "LC08_033029_20200101": [0.04],
            "LC08_033029_20200102": [0.80],
            "LC08_033029_20200103": [2.50],
        }
    ).to_csv(csv_path, index=False)

    ingestor.etf(
        source_dir=str(tmp_path),
        uid_column="sid",
        model="ptjpl",
        instrument="landsat",
        mask="no_mask",
    )

    values = np.asarray(state.root["remote_sensing/etf/landsat/ptjpl/no_mask"][:])[:, 0]
    assert np.isnan(values[0])
    assert values[1] == pytest.approx(0.80)
    assert np.isnan(values[2])


def test_ingest_etf_masks_scaled_values_above_shared_ceiling(tmp_path):
    """Scale-factor adjustments must not leave stored ETf above 2.0."""
    state = _make_container_state()
    ingestor = Ingestor(state, container=None)

    csv_path = tmp_path / "ptjpl_etf_US-FPe_no_mask_2020.csv"
    pd.DataFrame(
        {
            "sid": ["US-FPe"],
            "LC08_033029_20200101": [1.50],
            "LC08_033029_20200102": [1.90],
            "LC08_033029_20200103": [0.75],
        }
    ).to_csv(csv_path, index=False)

    ingestor.etf(
        source_dir=str(tmp_path),
        uid_column="sid",
        model="ptjpl",
        instrument="landsat",
        mask="no_mask",
        scale_factor=1.2,
    )

    values = np.asarray(state.root["remote_sensing/etf/landsat/ptjpl/no_mask"][:])[:, 0]
    assert values[0] == pytest.approx(1.8)
    assert np.isnan(values[1])
    assert values[2] == pytest.approx(0.9)
