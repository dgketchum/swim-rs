"""Tests for persisted simulation runs stored in SwimContainer."""

from pathlib import Path

import numpy as np

from swimrs.container import SwimContainer
from swimrs.container.runs import CORE_OUTPUTS

FIXTURE_SHP = (
    Path(__file__).parent.parent / "fixtures" / "S2" / "data" / "gis" / "flux_footprint_s2.shp"
)


def _build_runnable_container(tmp_path):
    container_path = tmp_path / "run_test.swim"
    container = SwimContainer.create(
        str(container_path),
        fields_shapefile=str(FIXTURE_SHP),
        uid_column="site_id",
        start_date="2020-04-01",
        end_date="2020-04-05",
    )

    # Minimal properties required by build_swim_input()
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

    # Minimal forcing and NDVI data for a short forward run.
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
    return container, container_path


def test_persisted_run_roundtrip(tmp_path):
    container, container_path = _build_runnable_container(tmp_path)
    field_uid = container.field_uids[0]

    result = container.run(run_id="baseline", profile="core", mask_mode="none")
    assert result.persisted is True
    container.save()
    container.close()

    reopened = SwimContainer.open(container_path, mode="r+")
    try:
        assert reopened.list_runs() == ["baseline"]

        meta = reopened.runs.metadata("baseline")
        assert meta["profile"] == "core"
        assert meta["persisted_outputs"] == list(CORE_OUTPUTS)

        ds = reopened.open_run_dataset("baseline")
        assert sorted(ds.data_vars) == sorted(CORE_OUTPUTS)
        assert ds.sizes["time"] == 5
        assert ds.sizes["site"] == 1

        df = reopened.run_dataframe("baseline", field_uid, variables=["eta", "runoff"])
        assert list(df.columns) == ["eta", "runoff"]
        assert len(df) == 5

        final_state = reopened.runs.final_state("baseline", fields=[field_uid])
        assert final_state["depl_root"].shape == (1,)

        event = reopened.provenance.get_latest_event_for_target("simulation/runs/baseline")
        assert event is not None
        assert event.operation == "simulate"
        assert reopened._root.attrs["last_run"]["run_id"] == "baseline"
    finally:
        reopened.close()


def test_restart_from_persisted_run_state_only(tmp_path):
    container, _ = _build_runnable_container(tmp_path)
    field_uid = container.field_uids[0]

    container.run(run_id="baseline", profile="core", mask_mode="none")
    baseline_final = container.runs.final_state("baseline", fields=[field_uid])

    container.run(
        run_id="restart_only",
        profile="state_only",
        mask_mode="none",
        restart_from="baseline",
        start_date="2020-04-03",
        end_date="2020-04-05",
    )

    meta = container.runs.metadata("restart_only")
    assert meta["profile"] == "state_only"
    assert meta["restart_from"] == "baseline"
    assert "simulation/runs/restart_only/outputs" not in container._root

    restarted_initial = container.runs.initial_state("restart_only", fields=[field_uid])
    np.testing.assert_allclose(restarted_initial["depl_root"], baseline_final["depl_root"])

    empty_ds = container.open_run_dataset("restart_only")
    assert len(empty_ds.data_vars) == 0

    container.close()


def test_default_restart_is_used_when_restart_not_explicit(tmp_path):
    container, _ = _build_runnable_container(tmp_path)
    field_uid = container.field_uids[0]

    container.run(run_id="baseline", profile="core", mask_mode="none")
    baseline_final = container.runs.final_state("baseline", fields=[field_uid])
    container.runs.set_default_restart("baseline")

    container.run(
        run_id="auto_restart",
        profile="state_only",
        mask_mode="none",
        start_date="2020-04-03",
        end_date="2020-04-05",
    )

    meta = container.runs.metadata("auto_restart")
    assert meta["restart_from"] == "baseline"

    auto_initial = container.runs.initial_state("auto_restart", fields=[field_uid])
    np.testing.assert_allclose(auto_initial["depl_root"], baseline_final["depl_root"])

    container.close()


def test_status_surfaces_default_restart_and_runs(tmp_path):
    container, _ = _build_runnable_container(tmp_path)
    container.run(run_id="baseline", profile="core", mask_mode="none")
    container.runs.set_default_restart("baseline")

    status = container.query.status()

    assert "Default restart: baseline" in status
    assert "SIMULATION RUNS:" in status
    assert "baseline: role=core, profile=core" in status

    container.close()


def test_run_report_renders_artifacts(tmp_path):
    container, _ = _build_runnable_container(tmp_path)
    container.run(run_id="baseline", profile="core", mask_mode="none")

    report_dir = tmp_path / "run_report"
    report = container.run_report("baseline", output_dir=report_dir)

    assert report.run_id == "baseline"
    assert report.run_metadata["profile"] == "core"
    assert report.n_variables == len(CORE_OUTPUTS)
    assert report.all_outputs_finite is True
    assert report.provenance_event is not None
    assert report.provenance_event["operation"] == "simulate"

    json_path = report_dir / "run.json"
    html_path = report_dir / "run.html"
    png_path = report_dir / "run.png"

    assert json_path.exists()
    assert html_path.exists()
    assert png_path.exists()

    json_text = json_path.read_text()
    html_text = html_path.read_text()
    assert '"run_id": "baseline"' in json_text
    assert "SWIM Run Report" in html_text
    assert "baseline" in html_text

    container.close()
