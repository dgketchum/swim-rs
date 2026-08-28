"""CDL-cultivated override for the perennial decision (asymmetric switch).

GLC10's 2017 snapshot labels chemfallow/winter-wheat strips as Barren (90),
routing them into perennial mechanics. A cultivated CDL history rescues those
units back to annual-crop flow; the override never pushes a unit INTO
perennial mode.
"""

import pandas as pd
from zarr.core.dtype import VariableLengthUTF8

from swimrs.container.components.ingestor import Ingestor
from swimrs.container.inventory import Inventory
from swimrs.container.provenance import ProvenanceLog
from swimrs.container.schema import is_cdl_cultivated, is_perennial
from swimrs.container.state import ContainerState
from swimrs.container.storage import MemoryStoreProvider


class TestIsCdlCultivated:
    def test_chemfallow_rotation_is_cultivated(self):
        # 61 Fallow/Idle Cropland is the load-bearing class
        assert is_cdl_cultivated([61, 24, 61, 23])

    def test_grass_pasture_history_not_cultivated(self):
        assert not is_cdl_cultivated([176, 176, 152, 131, 176])

    def test_single_year_noise_rejected(self):
        assert not is_cdl_cultivated([176] * 16 + [61])

    def test_fraction_arm_with_fill(self):
        # 2 cultivated of 4 valid years: fails the 3-year arm, passes >= 30%
        assert is_cdl_cultivated([61, 176, 24, -1, 176])

    def test_all_fill_is_false(self):
        assert not is_cdl_cultivated([-1, -1, 0])


class TestPerennialDecision:
    def test_barren_with_cultivated_history_runs_annual(self):
        assert not is_perennial(90, "glc10", cultivated=True)

    def test_grassland_without_cultivated_history_stays_perennial(self):
        assert is_perennial(30, "glc10", cultivated=False)

    def test_glc10_cropland_stays_annual_regardless_of_cdl(self):
        assert not is_perennial(10, "glc10", cultivated=False)
        assert not is_perennial(10, "glc10", cultivated=True)


def _make_container_state(n_fields=3):
    provider = MemoryStoreProvider(mode="w")
    root = provider.open()
    uids = [str(i) for i in range(1, n_fields + 1)]
    time_index = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    time_grp = root.create_group("time")
    time_grp.create_array("daily", data=time_index.values.astype("datetime64[ns]"))
    geom_grp = root.create_group("geometry")
    uid_arr = geom_grp.create_array("uid", shape=(n_fields,), dtype=VariableLengthUTF8())
    uid_arr[:] = uids
    for grp in ["properties", "remote_sensing", "meteorology", "snow", "derived"]:
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


class TestCdlIngest:
    def test_float_modes_round_trip_int16_with_fill(self, tmp_path):
        state, uids = _make_container_state(n_fields=3)
        ingestor = Ingestor(state, container=None)
        # site 1: chemfallow rotation w/ Reducer.mode() float noise
        # site 2: grass/pasture; site 3 absent from the CSV -> -1 fill
        df = pd.DataFrame(
            {
                "FID": ["1", "2"],
                "crop_2008": [61.0000000001, 176.0],
                "crop_2009": [23.9999999998, 176.0],
                "crop_2010": [61.0, 152.0],
                "crop_2011": [24.0, 176.0],
            }
        )
        csv = tmp_path / "cdl.csv"
        df.to_csv(csv, index=False)

        ingestor.properties(cdl_csv=csv, uid_column="FID")

        cdl = state.root["properties/land_cover/cdl"]
        cult = state.root["properties/land_cover/cdl_cultivated"]
        assert cdl.dtype == "int16"
        assert list(cdl.attrs["years"]) == [2008, 2009, 2010, 2011]
        assert list(cdl[0, :]) == [61, 24, 61, 24]
        assert list(cdl[2, :]) == [-1, -1, -1, -1]
        assert cult[0] == 1  # cultivated rotation
        assert cult[1] == 0  # grass/shrub history
        assert cult[2] == -1  # no CDL data
