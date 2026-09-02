"""Persisted simulation runs stored inside SwimContainer."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from zarr.core.dtype import VariableLengthUTF8

from swimrs.process.input import build_swim_input
from swimrs.process.loop import DailyOutput, run_daily_loop
from swimrs.process.loop_fast import run_daily_loop_fast
from swimrs.process.state import WaterBalanceState

if TYPE_CHECKING:
    import xarray as xr

    from swimrs.container.container import SwimContainer

RUNS_ROOT = "simulation/runs"

CORE_OUTPUTS = (
    "eta",
    "etf",
    "runoff",
    "dperc",
    "irr_sim",
    "gw_sim",
    "swe",
    "depl_root",
)

FULL_OUTPUTS = (
    "eta",
    "etf",
    "kcb",
    "ke",
    "ks",
    "kr",
    "runoff",
    "rain",
    "melt",
    "swe",
    "depl_root",
    "dperc",
    "irr_sim",
    "gw_sim",
    "et_irr",
    "dperc_irr",
    "irr_frac_root",
    "irr_frac_l3",
)

STATE_FIELDS = (
    "depl_root",
    "depl_ze",
    "daw3",
    "taw3",
    "swe",
    "albedo",
    "zr",
    "kr",
    "ks",
    "irr_continue",
    "next_day_irr",
    "s",
    "s1",
    "s2",
    "s3",
    "s4",
    "irr_frac_root",
    "irr_frac_l3",
)


@dataclass
class SimulationRunResult:
    """In-memory result of a model execution."""

    run_id: str
    profile: str
    engine: str
    field_uids: list[str]
    dates: pd.DatetimeIndex
    output: DailyOutput
    initial_state: WaterBalanceState
    final_state: WaterBalanceState
    ref_et: np.ndarray
    prcp: np.ndarray
    tmin: np.ndarray
    tmax: np.ndarray
    persisted: bool = False


class RunManager:
    """Read, write, and execute persisted simulation runs."""

    def __init__(self, container: SwimContainer):
        self._container = container

    def list(self) -> list[str]:
        """List available run ids."""
        root = self._container._root
        if RUNS_ROOT not in root:
            return []
        return sorted(root[RUNS_ROOT].keys())

    def metadata(self, run_id: str) -> dict[str, Any]:
        """Get run metadata attrs."""
        return dict(self._get_run_group(run_id).attrs)

    def default_restart_run_id(self) -> str | None:
        """Return the configured default restart run id, if valid."""
        run_id = self._container._root.attrs.get("default_restart_run_id")
        if not run_id:
            return None
        normalized = _normalize_run_id(str(run_id))
        if f"{RUNS_ROOT}/{normalized}" not in self._container._root:
            return None
        return normalized

    def set_default_restart(self, run_id: str | None) -> None:
        """Set or clear the container default restart run id."""
        if self._container._mode == "r":
            raise ValueError("Cannot modify default restart: container opened in read-only mode")
        if run_id is None:
            self._container._root.attrs.pop("default_restart_run_id", None)
        else:
            normalized = _normalize_run_id(run_id)
            self._get_run_group(normalized)
            self._container._root.attrs["default_restart_run_id"] = normalized
        self._container._mark_modified()
        self._container.state.refresh()

    def final_state(
        self,
        run_id: str,
        fields: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Load final state arrays for a persisted run."""
        return self._load_state_dict(run_id, "final", fields=fields)

    def initial_state(
        self,
        run_id: str,
        fields: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Load initial state arrays for a persisted run."""
        return self._load_state_dict(run_id, "initial", fields=fields)

    def open_dataset(
        self,
        run_id: str,
        variables: list[str] | None = None,
        fields: list[str] | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> xr.Dataset:
        """Open persisted run outputs as an xarray Dataset."""
        import xarray as xr

        run_group = self._get_run_group(run_id)
        time_index = pd.DatetimeIndex(run_group["time/daily"][:])
        field_uids = _decode_uids(run_group["fields/uid"][:])

        time_slice = _get_time_slice(time_index, start_date, end_date)
        selected_time = time_index[time_slice]

        if fields is None:
            field_indices = slice(None)
            selected_fields = field_uids
        else:
            field_lookup = {uid: idx for idx, uid in enumerate(field_uids)}
            missing = [uid for uid in fields if uid not in field_lookup]
            if missing:
                raise KeyError(
                    f"Field(s) not found in persisted run {run_id!r}: {', '.join(missing)}"
                )
            field_indices = [field_lookup[uid] for uid in fields]
            selected_fields = list(fields)

        if "outputs" not in run_group:
            return xr.Dataset(coords={"time": selected_time, "site": selected_fields})

        output_group = run_group["outputs"]
        if variables is None:
            variables = sorted(output_group.keys())

        data_vars = {}
        for variable in variables:
            if variable not in output_group:
                raise KeyError(f"Output variable {variable!r} not found in run {run_id!r}")
            arr = output_group[variable]
            data_vars[variable] = xr.DataArray(
                arr[time_slice, field_indices],
                dims=["time", "site"],
                coords={"time": selected_time, "site": selected_fields},
                name=variable,
            )

        return xr.Dataset(data_vars, coords={"time": selected_time, "site": selected_fields})

    def to_dataframe(
        self,
        run_id: str,
        field_uid: str,
        variables: list[str] | None = None,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Get persisted outputs for one field as a DataFrame."""
        ds = self.open_dataset(
            run_id,
            variables=variables,
            fields=[field_uid],
            start_date=start_date,
            end_date=end_date,
        )
        if "site" in ds.dims:
            ds = ds.sel(site=field_uid)
        df = ds.to_dataframe()
        if "site" in df.columns:
            df = df.drop(columns=["site"])
        return df

    def run(
        self,
        *,
        run_id: str | None = None,
        profile: str = "core",
        overwrite: bool = False,
        persist: bool = True,
        engine: str = "python",
        restart_from: str | None = None,
        spinup_state: dict[str, np.ndarray] | None = None,
        spinup_json_path: str | Path | None = None,
        calibrated_params_path: str | Path | None = None,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
        refet_type: str = "eto",
        etf_model: str = "ssebop",
        met_source: str = "gridmet",
        fields: list[str] | None = None,
        empirical_kc_max: bool = False,
        kc_max_overrides: dict[str, float] | None = None,
        mask_mode: str = "irrigation",
        ndvi_mode: str = "observed",
        max_irr_rate: float = 100.0,
        command: str | None = None,
        run_attrs: dict[str, Any] | None = None,
        use_default_restart: bool = True,
    ) -> SimulationRunResult:
        """Execute a model run and optionally persist it in the container."""
        if profile not in {"core", "full", "state_only"}:
            raise ValueError("profile must be one of {'core', 'full', 'state_only'}")
        if engine not in {"python", "fast"}:
            raise ValueError("engine must be one of {'python', 'fast'}")
        if persist and engine != "python":
            raise ValueError("Persisted runs require engine='python' for restart-safe state output")
        if persist and self._container._mode == "r":
            raise ValueError("Cannot persist runs: container opened in read-only mode")
        if restart_from and (spinup_state is not None or spinup_json_path is not None):
            raise ValueError(
                "restart_from cannot be combined with spinup_state or spinup_json_path"
            )

        resolved_run_id = _normalize_run_id(run_id or _default_run_id())
        resolved_fields = list(fields) if fields else None
        resolved_spinup_state = spinup_state
        resolved_restart_from = restart_from
        if (
            resolved_restart_from is None
            and resolved_spinup_state is None
            and spinup_json_path is None
            and use_default_restart
        ):
            default_restart = self.default_restart_run_id()
            if default_restart is not None and default_restart != resolved_run_id:
                resolved_restart_from = default_restart
        if resolved_restart_from is not None:
            target_fields = resolved_fields or self._container.field_uids
            resolved_spinup_state = self._load_state_dict(
                resolved_restart_from,
                "final",
                fields=target_fields,
            )

        fd, temp_h5_path = tempfile.mkstemp(suffix=".h5", prefix="swim_run_")
        os.close(fd)
        Path(temp_h5_path).unlink(missing_ok=True)

        try:
            swim_input = build_swim_input(
                self._container,
                output_h5=temp_h5_path,
                spinup_state=resolved_spinup_state,
                spinup_json_path=spinup_json_path,
                calibrated_params_path=calibrated_params_path,
                start_date=start_date,
                end_date=end_date,
                refet_type=refet_type,
                etf_model=etf_model,
                met_source=met_source,
                fields=resolved_fields,
                empirical_kc_max=empirical_kc_max,
                kc_max_overrides=kc_max_overrides,
                mask_mode=mask_mode,
                ndvi_mode=ndvi_mode,
                max_irr_rate=max_irr_rate,
            )

            try:
                initial_state = swim_input.spinup_state.copy()
                run_func = run_daily_loop if engine == "python" else run_daily_loop_fast
                output, final_state = run_func(swim_input)
                dates = pd.date_range(swim_input.start_date, periods=swim_input.n_days, freq="D")
                result = SimulationRunResult(
                    run_id=resolved_run_id,
                    profile=profile,
                    engine=engine,
                    field_uids=list(swim_input.fids),
                    dates=dates,
                    output=output,
                    initial_state=initial_state,
                    final_state=final_state,
                    ref_et=swim_input.get_time_series("ref_et"),
                    prcp=swim_input.get_time_series("prcp"),
                    tmin=swim_input.get_time_series("tmin"),
                    tmax=swim_input.get_time_series("tmax"),
                    persisted=False,
                )

                if persist:
                    self._write_run(
                        result=result,
                        overwrite=overwrite,
                        refet_type=refet_type,
                        etf_model=etf_model,
                        met_source=met_source,
                        mask_mode=mask_mode,
                        ndvi_mode=ndvi_mode,
                        restart_from=resolved_restart_from,
                        spinup_json_path=spinup_json_path,
                        calibrated_params_path=calibrated_params_path,
                        command=command,
                        run_attrs=run_attrs,
                    )
                    result.persisted = True

                return result
            finally:
                swim_input.close()
        finally:
            Path(temp_h5_path).unlink(missing_ok=True)

    def persist_result(
        self,
        result: SimulationRunResult,
        *,
        overwrite: bool = False,
        refet_type: str = "eto",
        etf_model: str = "ssebop",
        met_source: str = "gridmet",
        mask_mode: str = "irrigation",
        ndvi_mode: str = "observed",
        restart_from: str | None = None,
        spinup_json_path: str | Path | None = None,
        calibrated_params_path: str | Path | None = None,
        command: str | None = None,
        run_attrs: dict[str, Any] | None = None,
    ) -> None:
        """Persist an in-memory simulation result into the container."""
        if self._container._mode == "r":
            raise ValueError("Cannot persist runs: container opened in read-only mode")
        self._write_run(
            result=result,
            overwrite=overwrite,
            refet_type=refet_type,
            etf_model=etf_model,
            met_source=met_source,
            mask_mode=mask_mode,
            ndvi_mode=ndvi_mode,
            restart_from=restart_from,
            spinup_json_path=spinup_json_path,
            calibrated_params_path=calibrated_params_path,
            command=command,
            run_attrs=run_attrs,
        )
        result.persisted = True

    def _write_run(
        self,
        *,
        result: SimulationRunResult,
        overwrite: bool,
        refet_type: str,
        etf_model: str,
        met_source: str,
        mask_mode: str,
        ndvi_mode: str,
        restart_from: str | None,
        spinup_json_path: str | Path | None,
        calibrated_params_path: str | Path | None,
        command: str | None,
        run_attrs: dict[str, Any] | None,
    ) -> None:
        run_path = f"{RUNS_ROOT}/{result.run_id}"
        root = self._container._root
        runs_group = self._container._ensure_group(RUNS_ROOT)

        if result.run_id in runs_group:
            if not overwrite:
                raise ValueError(
                    f"Run {result.run_id!r} already exists in container; use overwrite=True"
                )
            del runs_group[result.run_id]

        run_group = runs_group.create_group(result.run_id)

        time_group = run_group.create_group("time")
        time_group.create_array("daily", data=result.dates.values.astype("datetime64[ns]"))

        fields_group = run_group.create_group("fields")
        uid_arr = fields_group.create_array(
            "uid",
            shape=(len(result.field_uids),),
            dtype=VariableLengthUTF8(),
        )
        uid_arr[:] = list(result.field_uids)

        output_names = _resolve_output_names(result.profile)
        if output_names:
            output_group = run_group.create_group("outputs")
            for name in output_names:
                output_group.create_array(
                    name,
                    data=np.asarray(getattr(result.output, name), dtype=np.float32),
                    chunks=(365, min(100, len(result.field_uids))),
                )

        state_group = run_group.create_group("state")
        self._write_state_group(state_group, "initial", result.initial_state)
        self._write_state_group(state_group, "final", result.final_state)

        created_at = datetime.now(UTC).isoformat()
        run_group.attrs.update(
            {
                "run_id": result.run_id,
                "created_at": created_at,
                "status": "completed",
                "profile": result.profile,
                "engine": result.engine,
                "field_count": len(result.field_uids),
                "n_days": len(result.dates),
                "start_date": str(result.dates[0].date()),
                "end_date": str(result.dates[-1].date()),
                "refet_type": (refet_type or "eto").lower().strip(),
                "etf_model": etf_model,
                "met_source": met_source,
                "mask_mode": mask_mode,
                "ndvi_mode": ndvi_mode,
                "persisted_outputs": list(output_names),
                "restart_from": restart_from,
                "spinup_json_path": str(spinup_json_path) if spinup_json_path is not None else None,
                "calibrated_params_path": (
                    str(calibrated_params_path) if calibrated_params_path is not None else None
                ),
                "container_uri": self._container.uri,
                "command": command,
            }
        )
        if run_attrs:
            run_group.attrs.update(dict(run_attrs))

        self._container.provenance.record(
            "simulate",
            target=run_path,
            source=self._container.uri,
            params={
                "run_id": result.run_id,
                "profile": result.profile,
                "engine": result.engine,
                "refet_type": (refet_type or "eto").lower().strip(),
                "etf_model": etf_model,
                "met_source": met_source,
                "mask_mode": mask_mode,
                "ndvi_mode": ndvi_mode,
                "restart_from": restart_from,
                "persisted_outputs": list(output_names),
            },
            fields_affected=list(result.field_uids),
            date_range=[str(result.dates[0].date()), str(result.dates[-1].date())],
            records_count=int(len(result.dates) * len(result.field_uids)),
        )

        root.attrs["last_run"] = {
            "run_id": result.run_id,
            "created_at": created_at,
            "path": run_path,
            "profile": result.profile,
            "field_count": len(result.field_uids),
            "n_days": len(result.dates),
        }

        self._container._mark_modified()
        self._container.state.refresh()

    def copy_from(
        self,
        source_container: SwimContainer,
        run_id: str,
        *,
        target_run_id: str | None = None,
        overwrite: bool = False,
        set_as_default: bool = False,
    ) -> str:
        """Copy a persisted run from another container into this container."""
        if self._container._mode == "r":
            raise ValueError("Cannot copy runs: container opened in read-only mode")

        source_run_id = _normalize_run_id(run_id)
        destination_run_id = _normalize_run_id(target_run_id or source_run_id)
        src_path = f"{RUNS_ROOT}/{source_run_id}"
        if src_path not in source_container._root:
            raise KeyError(f"Persisted run not found in source container: {run_id!r}")

        runs_group = self._container._ensure_group(RUNS_ROOT)
        if destination_run_id in runs_group:
            if not overwrite:
                raise ValueError(
                    f"Run {destination_run_id!r} already exists in container; use overwrite=True"
                )
            del runs_group[destination_run_id]

        type(self._container)._copy_zarr_group(
            source_container._root[src_path],
            runs_group,
            destination_run_id,
        )

        copied_group = runs_group[destination_run_id]
        copied_group.attrs["copied_from_container"] = source_container.uri
        copied_group.attrs["copied_from_run_id"] = source_run_id

        if set_as_default:
            self._container._root.attrs["default_restart_run_id"] = destination_run_id

        copied_fields = _decode_uids(copied_group["fields/uid"][:])
        copied_meta = dict(copied_group.attrs)
        self._container.provenance.record(
            "copy_restart_state",
            target=f"{RUNS_ROOT}/{destination_run_id}",
            source=source_container.uri,
            params={
                "source_run_id": source_run_id,
                "target_run_id": destination_run_id,
                "set_as_default": set_as_default,
                "profile": copied_meta.get("profile"),
                "start_date": copied_meta.get("start_date"),
                "end_date": copied_meta.get("end_date"),
            },
            fields_affected=copied_fields,
            date_range=[
                copied_meta.get("start_date"),
                copied_meta.get("end_date"),
            ],
            records_count=int(copied_meta.get("n_days", 0))
            * int(copied_meta.get("field_count", 0)),
        )
        self._container._mark_modified()
        self._container.state.refresh()
        return destination_run_id

    @staticmethod
    def _write_state_group(parent, name: str, state: WaterBalanceState) -> None:
        group = parent.create_group(name)
        for field_name in STATE_FIELDS:
            group.create_array(
                field_name,
                data=np.asarray(getattr(state, field_name), dtype=np.float64),
            )

    def _get_run_group(self, run_id: str):
        run_path = f"{RUNS_ROOT}/{_normalize_run_id(run_id)}"
        if run_path not in self._container._root:
            raise KeyError(f"Persisted run not found: {run_id!r}")
        return self._container._root[run_path]

    def _load_state_dict(
        self,
        run_id: str,
        which: str,
        fields: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        run_group = self._get_run_group(run_id)
        state_path = f"state/{which}"
        if state_path not in run_group:
            raise KeyError(f"State group not found for run {run_id!r}: {which}")

        run_field_uids = _decode_uids(run_group["fields/uid"][:])
        run_lookup = {uid: idx for idx, uid in enumerate(run_field_uids)}

        if fields is None:
            field_indices = slice(None)
        else:
            missing = [uid for uid in fields if uid not in run_lookup]
            if missing:
                raise KeyError(
                    f"Field(s) not found in persisted run {run_id!r}: {', '.join(missing)}"
                )
            field_indices = [run_lookup[uid] for uid in fields]

        state_group = run_group[state_path]
        return {
            name: np.asarray(state_group[name][field_indices], dtype=np.float64)
            for name in STATE_FIELDS
            if name in state_group
        }


def _resolve_output_names(profile: str) -> tuple[str, ...]:
    if profile == "core":
        return CORE_OUTPUTS
    if profile == "full":
        return FULL_OUTPUTS
    if profile == "state_only":
        return ()
    raise ValueError(f"Unsupported profile: {profile!r}")


def _default_run_id() -> str:
    return f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"


def _normalize_run_id(run_id: str) -> str:
    value = str(run_id).strip()
    if not value:
        raise ValueError("run_id cannot be empty")
    if "/" in value:
        raise ValueError("run_id cannot contain '/'")
    return value


def _decode_uids(values: np.ndarray) -> list[str]:
    decoded = list(values)
    if decoded and isinstance(decoded[0], bytes):
        decoded = [value.decode("utf-8") for value in decoded]
    return [str(value) for value in decoded]


def _get_time_slice(
    time_index: pd.DatetimeIndex,
    start_date: str | pd.Timestamp | None,
    end_date: str | pd.Timestamp | None,
) -> slice:
    if start_date is None:
        start_idx = 0
    else:
        start_loc = time_index.get_loc(pd.Timestamp(start_date))
        start_idx = start_loc.start if isinstance(start_loc, slice) else start_loc

    if end_date is None:
        end_idx = len(time_index)
    else:
        end_loc = time_index.get_loc(pd.Timestamp(end_date))
        end_idx = end_loc.stop if isinstance(end_loc, slice) else end_loc + 1

    return slice(start_idx, end_idx)
