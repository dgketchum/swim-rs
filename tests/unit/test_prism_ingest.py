"""Tests for PRISM daily precipitation ingestion.

PRISM and GridMET label different 24 h windows::

    GridMET day D   06Z(D)   -> 06Z(D+1)   labelled by the window START
    PRISM   day D   12Z(D-1) -> 12Z(D)     labelled by the window END

so PRISM D overlaps GridMET D-1 by 18 h and GridMET D by only 6 h. The
exported CSVs keep PRISM's own labels; ``Ingestor.prism`` moves them onto the
container's GridMET day axis. The 25% residual overlap means an unshifted
ingest still correlates weakly instead of failing outright (r = 0.280 versus
0.982 over 226 Esmeralda fields, 2018-2022), so the shift is pinned here.
"""

import numpy as np
import pandas as pd
import pytest
from zarr.core.dtype import VariableLengthUTF8

from swimrs.container.components.ingestor import Ingestor
from swimrs.container.inventory import Inventory
from swimrs.container.provenance import ProvenanceLog
from swimrs.container.schema import MetSource
from swimrs.container.state import ContainerState
from swimrs.container.storage import MemoryStoreProvider

PRISM_PATH = "meteorology/prism/prcp"


def _make_container_state(n_fields=3, start="2020-01-01", end="2020-12-31"):
    """In-memory ContainerState with string UIDs '1'..'N' on a daily index."""
    provider = MemoryStoreProvider(mode="w")
    root = provider.open()

    uids = [str(i) for i in range(1, n_fields + 1)]
    time_index = pd.date_range(start, end, freq="D")

    time_grp = root.create_group("time")
    time_grp.create_array("daily", data=time_index.values.astype("datetime64[ns]"))

    geom_grp = root.create_group("geometry")
    uid_arr = geom_grp.create_array("uid", shape=(n_fields,), dtype=VariableLengthUTF8())
    uid_arr[:] = uids

    for group in ("properties", "remote_sensing", "meteorology", "snow", "derived"):
        root.create_group(group)

    state = ContainerState(
        provider=provider,
        field_uids=uids,
        time_index=time_index,
        provenance=ProvenanceLog(),
        inventory=Inventory(root, uids),
        mode="w",
    )
    return state, uids


def _write_prism_csv(tmp_path, dates, values, uids, uid_column="FID", name="ppt_2020.csv"):
    """Write one PRISM export CSV: rows=fields, columns=YYYYMMDD, values=mm."""
    df = pd.DataFrame(values, index=uids, columns=[d.strftime("%Y%m%d") for d in dates])
    df.index.name = uid_column
    path = tmp_path / name
    df.to_csv(path)
    return path


def _read_back(state):
    return np.asarray(state.root[PRISM_PATH][:])


def test_prism_label_lands_one_day_earlier(tmp_path):
    """A spike on PRISM label D is written to container day D-1."""
    state, uids = _make_container_state()
    ingestor = Ingestor(state, container=None)

    dates = pd.date_range("2020-06-10", "2020-06-14", freq="D")
    values = np.zeros((len(uids), len(dates)))
    values[:, 2] = 7.5  # PRISM label 2020-06-12
    _write_prism_csv(tmp_path, dates, values, uids)

    ingestor.prism(tmp_path, uid_column="FID")

    arr = _read_back(state)
    spike = state.time_index.get_loc(pd.Timestamp("2020-06-11"))
    assert arr[spike, 0] == pytest.approx(7.5)
    assert arr[state.time_index.get_loc(pd.Timestamp("2020-06-12")), 0] == pytest.approx(0.0)


def test_prism_unshifted_keeps_true_labels(tmp_path):
    """align_to_gridmet_day=False leaves the value on PRISM's own label."""
    state, uids = _make_container_state()
    ingestor = Ingestor(state, container=None)

    dates = pd.date_range("2020-06-10", "2020-06-14", freq="D")
    values = np.zeros((len(uids), len(dates)))
    values[:, 2] = 7.5
    _write_prism_csv(tmp_path, dates, values, uids)

    ingestor.prism(tmp_path, uid_column="FID", align_to_gridmet_day=False)

    arr = _read_back(state)
    assert arr[state.time_index.get_loc(pd.Timestamp("2020-06-12")), 0] == pytest.approx(7.5)
    assert arr[state.time_index.get_loc(pd.Timestamp("2020-06-11")), 0] == pytest.approx(0.0)


def test_shift_puts_prism_in_phase_with_gridmet(tmp_path):
    """The shift reconstructs a GridMET-convention series exactly; omitting it does not.

    ``truth`` is a series on GridMET's day axis. The PRISM CSV carries each of
    its values one label later, which is how the two products actually relate.
    """
    state, uids = _make_container_state()
    ingestor = Ingestor(state, container=None)

    rng = np.random.default_rng(0)
    gridmet_days = pd.date_range("2020-03-01", "2020-09-30", freq="D")
    truth = rng.gamma(0.3, 4.0, size=len(gridmet_days))

    prism_labels = gridmet_days + pd.Timedelta(days=1)
    values = np.tile(truth, (len(uids), 1))
    _write_prism_csv(tmp_path, prism_labels, values, uids)

    ingestor.prism(tmp_path, uid_column="FID")

    arr = _read_back(state)
    rows = [state.time_index.get_loc(d) for d in gridmet_days]
    np.testing.assert_allclose(arr[rows, 0], truth, rtol=0, atol=1e-6)

    # Same data ingested on PRISM's labels is off by a day, and the residual
    # 6 h overlap is exactly why that is a quiet error rather than a loud one.
    ingestor.prism(tmp_path, uid_column="FID", overwrite=True, align_to_gridmet_day=False)
    shifted = _read_back(state)
    assert not np.allclose(shifted[rows, 0], truth, atol=1e-6)


def test_values_are_millimeters_unchanged(tmp_path):
    """PRISM ppt is already mm -- no conversion on the way in."""
    state, uids = _make_container_state()
    ingestor = Ingestor(state, container=None)

    dates = pd.date_range("2020-04-01", "2020-04-03", freq="D")
    values = np.array([[1.25, 0.0, 33.75]] * len(uids))
    _write_prism_csv(tmp_path, dates, values, uids)

    ingestor.prism(tmp_path, uid_column="FID")

    arr = _read_back(state)
    got = [arr[state.time_index.get_loc(d - pd.Timedelta(days=1)), 0] for d in dates]
    assert got == pytest.approx([1.25, 0.0, 33.75])


def test_year_edges_shift_across_the_file_boundary(tmp_path):
    """The shift moves a year's file back onto Dec 31 of the year before.

    PRISM 20200101 maps to 2019-12-31, outside a 2020-only container, and is
    dropped by the reindex rather than landing on the wrong day. Symmetrically,
    2020-12-31 can only be filled by the 2021 file, so a container's final day
    stays empty unless the following year is exported too.
    """
    state, uids = _make_container_state(start="2020-01-01", end="2020-12-31")
    ingestor = Ingestor(state, container=None)

    dates = pd.date_range("2020-01-01", "2020-01-03", freq="D")
    values = np.array([[9.0, 4.0, 2.0]] * len(uids))
    _write_prism_csv(tmp_path, dates, values, uids)

    ingestor.prism(tmp_path, uid_column="FID")

    arr = _read_back(state)
    # 9.0 (PRISM 20200101) fell off the front of the container.
    assert arr[state.time_index.get_loc(pd.Timestamp("2020-01-01")), 0] == pytest.approx(4.0)
    assert arr[state.time_index.get_loc(pd.Timestamp("2020-01-02")), 0] == pytest.approx(2.0)
    assert np.isnan(arr[state.time_index.get_loc(pd.Timestamp("2020-01-03")), 0])
    assert np.isnan(arr[state.time_index.get_loc(pd.Timestamp("2020-12-31")), 0])


def test_multiple_years_concatenate(tmp_path):
    """One CSV per year, as run_ppt writes them, load into a single series."""
    state, uids = _make_container_state(start="2019-01-01", end="2020-12-31")
    ingestor = Ingestor(state, container=None)

    for year, spike in ((2019, 11.0), (2020, 22.0)):
        dates = pd.date_range(f"{year}-07-10", f"{year}-07-12", freq="D")
        values = np.zeros((len(uids), len(dates)))
        values[:, 1] = spike
        _write_prism_csv(tmp_path, dates, values, uids, name=f"ppt_{year}.csv")

    ingestor.prism(tmp_path, uid_column="FID")

    arr = _read_back(state)
    assert arr[state.time_index.get_loc(pd.Timestamp("2019-07-10")), 0] == pytest.approx(11.0)
    assert arr[state.time_index.get_loc(pd.Timestamp("2020-07-10")), 0] == pytest.approx(22.0)


def test_provenance_records_the_shift(tmp_path):
    """The alignment choice is recorded, so a container can be audited for it."""
    state, uids = _make_container_state()
    ingestor = Ingestor(state, container=None)

    dates = pd.date_range("2020-05-01", "2020-05-03", freq="D")
    _write_prism_csv(tmp_path, dates, np.ones((len(uids), len(dates))), uids)

    event = ingestor.prism(tmp_path, uid_column="FID")

    assert event.target == PRISM_PATH
    assert event.params["align_to_gridmet_day"] is True
    assert event.records_count == len(uids) * len(dates)


def test_prism_is_a_known_met_source():
    """The path Ingestor.prism writes is the one MetSource names."""
    assert PRISM_PATH == f"meteorology/{MetSource.PRISM.value}/prcp"
