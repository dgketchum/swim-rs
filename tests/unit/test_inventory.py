import numpy as np

from swimrs.container.inventory import DataStatus, Inventory


class _FakeArray:
    def __init__(self, data, *, attrs=None):
        self._data = np.asarray(data)
        self.attrs = attrs or {}

    @property
    def ndim(self) -> int:
        return self._data.ndim

    def __getitem__(self, item):
        return self._data[item]


class _FakeRoot:
    def __init__(self, mapping: dict[str, _FakeArray]):
        self._mapping = dict(mapping)

    def __getitem__(self, key: str) -> _FakeArray:
        try:
            return self._mapping[key]
        except KeyError as e:
            raise KeyError(key) from e

    def __contains__(self, key: str) -> bool:
        if key in self._mapping:
            return True
        prefix = f"{key}/"
        return any(k.startswith(prefix) for k in self._mapping)


def test_inventory_get_coverage_missing_path_is_not_present():
    root = _FakeRoot({})
    inv = Inventory(root, field_uids=["a", "b"])

    cov = inv.get_coverage("properties/soils/awc")
    assert cov.status == DataStatus.NOT_PRESENT
    assert cov.fields_present == 0
    assert cov.fields_total == 2
    assert cov.fields_missing == ["a", "b"]


def test_inventory_get_coverage_1d_properties_counts_nan_as_missing():
    root = _FakeRoot({"properties/soils/awc": _FakeArray([150.0, np.nan])})
    inv = Inventory(root, field_uids=["a", "b"])

    cov = inv.get_coverage("properties/soils/awc")
    assert cov.status == DataStatus.PARTIAL
    assert cov.fields_present == 1
    assert cov.fields_missing == ["b"]
    assert cov.percent_complete == 50.0


def test_inventory_get_coverage_2d_timeseries_sets_date_range_and_event_ids():
    root = _FakeRoot(
        {
            "time/daily": _FakeArray(
                np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[D]")
            ),
            "meteorology/gridmet/eto": _FakeArray(
                [
                    [1.0, np.nan],
                    [2.0, np.nan],
                    [3.0, np.nan],
                ],
                attrs={"event_ids": ["evt_1", "evt_2"]},
            ),
        }
    )
    inv = Inventory(root, field_uids=["a", "b"])

    cov = inv.get_coverage("meteorology/gridmet/eto")
    assert cov.status == DataStatus.PARTIAL
    assert cov.fields_present == 1
    assert cov.fields_missing == ["b"]
    assert cov.date_range == ("2020-01-01", "2020-01-03")
    assert cov.event_ids == ["evt_1", "evt_2"]


def test_inventory_refresh_clears_cache():
    root = _FakeRoot({"properties/soils/awc": _FakeArray([150.0, np.nan])})
    inv = Inventory(root, field_uids=["a", "b"])

    cov1 = inv.get_coverage("properties/soils/awc")
    cov2 = inv.get_coverage("properties/soils/awc")
    assert cov1 is cov2  # cached object

    inv.refresh()
    cov3 = inv.get_coverage("properties/soils/awc")
    assert cov3 is not cov1


def test_inventory_validate_requirements_tracks_ready_fields_with_partial_and_missing():
    root = _FakeRoot(
        {
            "a": _FakeArray(
                [
                    [1.0, np.nan],
                    [1.0, np.nan],
                ]
            )
        }
    )
    inv = Inventory(root, field_uids=["uid1", "uid2"])

    result = inv._validate_requirements(["a", "b"], operation="test-op")
    assert result.ready is False
    assert result.missing_data == ["b"]
    assert [c.path for c in result.incomplete_data] == ["a"]
    assert result.ready_fields == ["uid1"]
    assert result.not_ready_fields == ["uid2"]
