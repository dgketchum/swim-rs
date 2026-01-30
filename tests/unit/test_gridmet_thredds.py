import numpy as np
import pytest
import xarray as xr


def test_gridmet_build_url_elevation():
    from swimrs.data_extraction.gridmet.thredds import GridMet

    gm = GridMet(variable="elev", lat=48.3, lon=-105.1)
    url = gm._build_url()
    assert url.startswith("http://thredds.northwestknowledge.net:8080/")
    assert url.endswith("/thredds/dodsC/MET/elev/metdata_elevationdata.nc")


def test_gridmet_point_timeseries_uses_nearest_and_date_index(monkeypatch):
    from swimrs.data_extraction.gridmet import thredds
    from swimrs.data_extraction.gridmet.thredds import GridMet

    day = np.array(["2020-07-01", "2020-07-02", "2020-07-03"], dtype="datetime64[D]")
    lat = np.array([48.0, 49.0], dtype=float)
    lon = np.array([-105.0, -104.0], dtype=float)

    # Unique, easy-to-spot values (day_index + lat_index*10 + lon_index*100)
    data = np.zeros((len(day), len(lat), len(lon)), dtype=float)
    for t in range(len(day)):
        for i in range(len(lat)):
            for j in range(len(lon)):
                data[t, i, j] = t + i * 10 + j * 100

    ds = xr.Dataset(
        {"daily_mean_reference_evapotranspiration_alfalfa": (("day", "lat", "lon"), data)},
        coords={"day": day, "lat": lat, "lon": lon},
    )

    opened_urls: list[str] = []

    def fake_open_dataset(url, *args, **kwargs):
        opened_urls.append(url)
        return ds

    monkeypatch.setattr(thredds, "open_dataset", fake_open_dataset)

    gm = GridMet(
        variable="etr",
        lat=48.3,
        lon=-105.1,
        start="2020-07-01",
        end="2020-07-03",
    )
    df = gm.get_point_timeseries()

    assert opened_urls and opened_urls[0].endswith("#fillmismatch")
    assert list(df.columns) == ["etr"]
    assert len(df) == 3

    # Nearest should select lat=48.0 (idx 0) and lon=-105.0 (idx 0).
    assert np.allclose(df["etr"].to_numpy(), np.array([0.0, 1.0, 2.0]))
    assert str(df.index[0])[:10] == "2020-07-01"
    assert str(df.index[-1])[:10] == "2020-07-03"


def test_gridmet_point_elevation(monkeypatch):
    from swimrs.data_extraction.gridmet import thredds
    from swimrs.data_extraction.gridmet.thredds import GridMet

    lat = np.array([48.0, 49.0], dtype=float)
    lon = np.array([-105.0, -104.0], dtype=float)
    # Include a length-1 dim so `.values[0]` works for the current implementation.
    elev = np.array([[[650.0, 700.0], [800.0, 900.0]]], dtype=float)  # (band=1, lat=2, lon=2)

    ds = xr.Dataset(
        {"elevation": (("band", "lat", "lon"), elev)},
        coords={"band": np.array([0]), "lat": lat, "lon": lon},
    )

    monkeypatch.setattr(thredds, "open_dataset", lambda *args, **kwargs: ds)

    gm = GridMet(variable="elev", lat=48.3, lon=-105.1)
    assert gm.get_point_elevation() == pytest.approx(650.0)
