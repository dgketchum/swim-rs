"""
Calculator component for derived data computation.

Provides a clean API for computing derived products from ingested data.
Usage: container.compute.dynamics(...)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from swimrs.container import schema

from .base import Component

if TYPE_CHECKING:
    import xarray as xr

    from swimrs.container.provenance import ProvenanceEvent
    from swimrs.container.state import ContainerState


class Calculator(Component):
    """
    Component for computing derived data products.

    Provides methods for computing dynamics (irrigation detection,
    groundwater subsidy, K-parameters) and merging multi-sensor NDVI.

    Example:
        container.compute.dynamics(etf_model="ssebop")
        container.compute.merged_ndvi()
    """

    def __init__(self, state: ContainerState, container=None):
        """
        Initialize the Calculator.

        Args:
            state: ContainerState instance
            container: Optional reference to parent SwimContainer
        """
        super().__init__(state, container)

    def merged_ndvi(
        self,
        masks: tuple[str, ...] = ("irr", "inv_irr"),
        instruments: tuple[str, ...] = ("landsat", "sentinel"),
        preference_order: tuple[str, ...] = ("landsat", "sentinel"),
        overwrite: bool = False,
    ) -> ProvenanceEvent:
        """
        Merge harmonized sensor time series into a unified NDVI dataset.

        Systematic spectral differences between sensors are handled during
        GEE extraction via Spectral Bandpass Adjustment Factors (SBAF).
        This method performs a simple chronological merge, combining
        observations from multiple sensors into a denser time series.

        References:
            Roy, D.P., et al. (2016). Characterization of Landsat-7 to Landsat-8
            reflectance and NDVI differences. Remote Sensing of Environment.

            Claverie, M., et al. (2018). The Harmonized Landsat and Sentinel-2
            (HLS) product. Remote Sensing of Environment.

        Args:
            masks: Mask types to process (e.g., "irr", "inv_irr")
            instruments: Instruments to merge (e.g., "landsat", "sentinel")
            preference_order: When multiple sensors have data for the same date,
                prefer sensors in this order (default: Landsat > Sentinel)
            overwrite: If True, replace existing results

        Returns:
            ProvenanceEvent recording the operation
        """
        self._ensure_writable()

        with self._track_operation(
            "compute_merged_ndvi",
            target="derived/merged_ndvi",
            instruments=instruments,
        ) as ctx:
            total_records = 0
            fields_processed = set()

            for mask in masks:
                path = f"derived/merged_ndvi/{mask}"

                if path in self._state.root and not overwrite:
                    self._log.debug("skipping_existing", path=path)
                    continue
                if path in self._state.root:
                    self._safe_delete_path(path)

                # Collect NDVI from all available instruments
                sensor_data = []
                for instrument in instruments:
                    inst_path = f"remote_sensing/ndvi/{instrument}/{mask}"
                    if inst_path in self._state.root:
                        da = self._state.get_xarray(inst_path)
                        sensor_data.append((instrument, da))

                if not sensor_data:
                    self._log.warning("no_ndvi_sources", mask=mask)
                    continue

                # Merge sensors chronologically
                if len(sensor_data) == 1:
                    merged = sensor_data[0][1]
                else:
                    merged = self._merge_sensors(sensor_data, preference_order)

                # Write result
                arr = self._state.create_timeseries_array(path)
                arr[:, :] = merged.values

                total_records += int(np.count_nonzero(~np.isnan(merged.values)))
                fields_processed.update(merged.coords["site"].values)

            ctx["records_processed"] = total_records
            ctx["fields_processed"] = len(fields_processed)

            event = self._state.provenance.record(
                "compute",
                target="derived/merged_ndvi",
                params={
                    "masks": list(masks),
                    "instruments": list(instruments),
                    "preference_order": list(preference_order),
                },
                fields_affected=list(fields_processed),
                records_count=total_records,
            )

            self._state.mark_modified()
            self._state.refresh()

            return event

    # Keep fused_ndvi as alias for backward compatibility
    def fused_ndvi(self, *args, **kwargs) -> ProvenanceEvent:
        """Deprecated: Use merged_ndvi() instead."""
        return self.merged_ndvi(*args, **kwargs)

    def ndvi_climatology(
        self,
        source_container=None,
        masks: tuple[str, ...] = ("irr", "inv_irr"),
        overwrite: bool = False,
    ) -> ProvenanceEvent:
        """Compute per-DOY median NDVI climatology from merged NDVI.

        Groups the merged NDVI time series by day-of-year (1-366) and
        computes the median across years for each field. The result is a
        (366, n_fields) array stored at ``derived/ndvi_climatology/{mask}``.

        This climatology is used as the NDVI input for forecast containers
        where no observed NDVI is available.

        Args:
            source_container: Optional SwimContainer to read merged NDVI
                from. If None, reads from the current container.
            masks: Mask types to compute climatology for.
            overwrite: If True, replace existing climatology arrays.

        Returns:
            ProvenanceEvent recording the operation.
        """
        self._ensure_writable()

        src = source_container if source_container is not None else self._container

        with self._track_operation(
            "compute_ndvi_climatology",
            target="derived/ndvi_climatology",
        ) as ctx:
            total_records = 0

            for mask in masks:
                out_path = f"derived/ndvi_climatology/{mask}"
                if out_path in self._state.root and not overwrite:
                    self._log.debug("skipping_existing", path=out_path)
                    continue
                if out_path in self._state.root:
                    self._safe_delete_path(out_path)

                # Read merged NDVI from source
                merged_path = f"derived/merged_ndvi/{mask}"
                raw_path = f"remote_sensing/ndvi/landsat/{mask}"
                src_root = src._root if src is not None else self._state.root
                if merged_path in src_root:
                    da = src.to_xarray(merged_path)
                elif raw_path in src_root:
                    da = src.to_xarray(raw_path)
                else:
                    self._log.warning("no_ndvi_for_climatology", mask=mask)
                    continue

                # Group by DOY, compute median
                doy = da.coords["time"].dt.dayofyear
                clim = da.groupby(doy).median(dim="time")  # shape (366, n_fields)

                # Ensure all 366 DOYs are present
                all_doys = np.arange(1, 367)
                clim = clim.reindex(dayofyear=all_doys)

                # Interpolate any missing DOYs
                clim = clim.interpolate_na(dim="dayofyear", method="linear")
                clim = clim.bfill(dim="dayofyear").ffill(dim="dayofyear")

                clim_values = clim.values.astype(np.float32)

                # Write (366, n_fields) array
                parent_path = "/".join(out_path.split("/")[:-1])
                name = out_path.split("/")[-1]
                parent = self._state.root
                for part in parent_path.split("/"):
                    if part not in parent:
                        parent = parent.create_group(part)
                    else:
                        parent = parent[part]

                parent.create_array(
                    name,
                    data=clim_values,
                    chunks=(366, min(100, clim_values.shape[1])),
                    dtype="float32",
                )

                total_records += int(np.count_nonzero(~np.isnan(clim_values)))

            ctx["records_processed"] = total_records
            ctx["fields_processed"] = self._state.n_fields

            event = self._state.provenance.record(
                "compute",
                target="derived/ndvi_climatology",
                params={"masks": list(masks)},
                records_count=total_records,
            )

            self._state.mark_modified()
            self._state.refresh()

            return event

    def dynamics(
        self,
        etf_model: str = "ssebop",
        irr_threshold: float = 0.1,
        masks: tuple[str, ...] = ("irr", "inv_irr"),
        instruments: tuple[str, ...] = ("landsat", "sentinel"),
        use_mask: bool = False,
        use_lulc: bool = True,
        lookback: int = 10,
        ndvi_threshold: float = 0.3,
        ndvi_min_start: float = 0.25,
        min_pos_days: int = 10,
        met_source: str = "gridmet",
        fields: list[str] | None = None,
        overwrite: bool = False,
        gwsub_irr_fallback: bool = True,
        lulc_irr_method: str = "annual_2yr",
        annual_subsidy_ratio: float = 1.3,
        annual_retain_ratio: float = 1.1,
        season_storage_mm: float = 150.0,
        etf_gap_fallback_model: str | None = None,
    ) -> ProvenanceEvent:
        """
        Compute field dynamics: irrigation detection, groundwater subsidy, K-parameters.

        This is the main computation method that:
        1. Detects irrigation events per year using ETf/NDVI patterns
        2. Calculates groundwater subsidy (ET/PPT ratio)
        3. Extracts Ke (evaporation) and Kc (crop) parameters

        Two modes for determining irrigation status:

        use_mask=True (CONUS - Examples 4 & 5):
            - Reads per-year irrigation fraction from properties/irrigation/irr_yearly
              (IrrMapper/LANID table ingested from the irrigation properties CSV)
            - Year is irrigated if f_irr > irr_threshold
            - Independent of ``masks``: works with masks=("no_mask",) as well as
              masked groups — ``masks`` only selects which NDVI/ETf container
              groups are read (see _load_dynamics_dataset)

        use_lulc=True (International - Example 6):
            - Computes irrigation from a water-balance subsidy test, gated on
              the field being cropped (annual rooting habit, per LULC)
            - The water balance runs on the nanmean ETf ensemble (mean of all
              available ETf members per date — the primary model plus any
              ``etf_gap_fallback_model``), matching the calibration target and
              averaging out single-model bias.
            - ``lulc_irr_method`` selects the subsidy test:
                "annual_2yr" (default): a two-stage rule. STAGE 1 (site-level)
                    classifies a field as irrigation-equipped only if MORE THAN
                    ONE-THIRD of its rolling 2-year sum(ET)/sum(PPT) windows
                    exceed annual_subsidy_ratio. The 2-year window spans a
                    crop+fallow cycle, so dryland systems that bank soil moisture
                    across a fallow year (e.g. wheat-fallow) do not read as
                    subsidized; the one-third floor makes the decision robust to
                    a lone drought or boundary year while admitting genuine
                    intermittent-irrigation fields. STAGE 2 then flags, within an
                    equipped field, each year whose single-year ET/PPT exceeds
                    annual_retain_ratio (a lower keep-on threshold than the
                    Stage 1 equip threshold — Schmitt trigger), OR whose
                    demand-season balance ET_season/(PPT_season +
                    season_storage_mm) exceeds annual_subsidy_ratio. The season
                    test rescues wet-winter years (Mediterranean climates) whose
                    calendar-year ratio is diluted below any usable threshold by
                    off-season rain even though the crop transpires through a
                    nearly rain-free summer. Non-equipped (dryland) fields are
                    never flagged.
                "annual": same two-stage rule with single-year windows in Stage 1.
                "monthly": legacy test — irrigated if >= 3 months have
                    ET/(PPT+1) > 1.3. Over-flags moisture-banking dryland
                    fields; retained for back-compat.
            - Works with masks=("no_mask",)

        Args:
            etf_model: ET model to use ("ssebop", "ptjpl", etc.)
            irr_threshold: Fraction threshold for classifying irrigated years
            masks: Mask types to process
            instruments: Instruments for NDVI data
            use_mask: If True, use irrigation mask properties (CONUS mode)
            use_lulc: If True, use water balance + LULC (International mode)
            lookback: Days of lookback for irrigation window extension
            ndvi_threshold: NDVI threshold for forward extension decay floor
            ndvi_min_start: Minimum smoothed NDVI to initiate a window
            min_pos_days: Minimum consecutive positive slope days
            met_source: Meteorology source ("gridmet", "era5")
            fields: Optional list of field UIDs to process
            overwrite: If True, replace existing results
            gwsub_irr_fallback: If True (default), override non-irrigated status
                when gwsub detects ET > PPT at cropland sites. Set False to rely
                solely on use_mask/use_lulc for irrigation classification.
            lulc_irr_method: Subsidy test for use_lulc mode — "annual_2yr"
                (default), "annual", or "monthly" (legacy). See above.
            annual_subsidy_ratio: ET/PPT ratio above which a balance window is
                treated as subsidized — used by the Stage 1 site one-third
                test and the Stage 2 demand-season test (default 1.3).
            annual_retain_ratio: Stage 2 keep-on threshold for the single-year
                ratio at an equipped field (default 1.1). Lower than the
                Stage 1 equip threshold so a marginal year at an established
                irrigated field does not flip to fallow.
            season_storage_mm: Soil-storage allowance (mm) added to demand-season
                precipitation in the Stage 2 season test (default 150) — the
                carryover water a non-irrigated field could plausibly draw, so
                senesced/fallow years do not pass on stored winter moisture.
            etf_gap_fallback_model: Optional second ETf model (e.g. "ptjpl")
                averaged with the primary into the nanmean ensemble that drives
                the annual/annual_2yr water balance. Fills the primary's capture
                gaps and reduces single-model bias; not a per-year override, so
                no per-year provenance tags are written.

        Returns:
            ProvenanceEvent recording the operation

        Raises:
            ValueError: If neither use_mask nor use_lulc is True
        """
        self._ensure_writable()

        # Validate mode selection
        if not (use_mask or use_lulc):
            raise ValueError("Must use either use_mask or use_lulc for irrigation analysis")

        with self._track_operation(
            "compute_dynamics",
            target="derived/dynamics",
            etf_model=etf_model,
            met_source=met_source,
        ) as ctx:
            # Get fields to process
            target_fields = fields if fields else self._state.field_uids

            # Require merged NDVI for reliable irrigation window detection
            fused_path = f"derived/merged_ndvi/{masks[0]}"
            if fused_path not in self._state.root:
                raise ValueError(
                    f"Merged NDVI not found at '{fused_path}'. "
                    "Run container.compute.merged_ndvi() before dynamics() to ensure "
                    "reliable irrigation window detection from multi-sensor NDVI."
                )

            # Load required data
            ds = self._load_dynamics_dataset(
                target_fields,
                etf_model,
                masks,
                instruments,
                met_source,
                etf_gap_fallback_model=etf_gap_fallback_model,
            )

            if ds is None:
                self._log.warning("insufficient_data")
                return self._state.provenance.record(
                    "compute",
                    target="derived/dynamics",
                    params={"etf_model": etf_model},
                    records_count=0,
                    success=True,
                )

            # Compute K parameters
            ke_max, kc_max = self._compute_k_parameters(ds)

            # Compute groundwater subsidy
            gwsub_data = self._compute_groundwater_subsidy(ds, irr_threshold)

            # Load per-year irrigation properties if using mask mode
            irr_props = None
            if use_mask:
                irr_props = self._get_yearly_irrigation_properties()

            # Compute irrigation windows
            irr_data = self._compute_irrigation_data(
                ds,
                irr_threshold,
                lookback,
                ndvi_threshold,
                ndvi_min_start,
                min_pos_days,
                use_mask,
                use_lulc,
                irr_props,
                gwsub_data if gwsub_irr_fallback else None,
                lulc_irr_method=lulc_irr_method,
                annual_subsidy_ratio=annual_subsidy_ratio,
                annual_retain_ratio=annual_retain_ratio,
                season_storage_mm=season_storage_mm,
            )

            # Write results
            self._write_dynamics_results(
                ke_max, kc_max, irr_data, gwsub_data, target_fields, overwrite
            )

            ctx["records_processed"] = len(target_fields)
            ctx["fields_processed"] = len(target_fields)

            event = self._state.provenance.record(
                "compute",
                target="derived/dynamics",
                params={
                    "etf_model": etf_model,
                    "irr_threshold": irr_threshold,
                    "masks": list(masks),
                    "instruments": list(instruments),
                    "use_mask": use_mask,
                    "use_lulc": use_lulc,
                    "gwsub_irr_fallback": gwsub_irr_fallback,
                    "lulc_irr_method": lulc_irr_method,
                    "annual_subsidy_ratio": annual_subsidy_ratio,
                    "annual_retain_ratio": annual_retain_ratio,
                    "season_storage_mm": season_storage_mm,
                    "etf_gap_fallback_model": etf_gap_fallback_model,
                    "lookback": lookback,
                    "ndvi_min_start": ndvi_min_start,
                    "met_source": met_source,
                },
                fields_affected=target_fields,
                records_count=len(target_fields),
            )

            self._state.mark_modified()
            self._state.refresh()

            return event

    def irrigation_windows(
        self,
        etf_model: str = "ssebop",
        fields: list[str] | None = None,
    ) -> dict[str, dict]:
        """
        Get irrigation windows for specific fields.

        Returns per-year irrigation start/end dates and classifications.

        Args:
            etf_model: ET model to use
            fields: Fields to analyze

        Returns:
            Dict mapping field UID to yearly irrigation data

        Note:
            This requires dynamics to have been computed first.
        """
        irr_path = "derived/dynamics/irr_data"
        if irr_path not in self._state.root:
            raise ValueError("Dynamics not computed. Run compute.dynamics() first.")

        arr = self._state.root[irr_path]
        target_fields = fields if fields else self._state.field_uids

        results = {}
        for field_uid in target_fields:
            if field_uid not in self._state._uid_to_index:
                continue
            idx = self._state._uid_to_index[field_uid]
            data = arr[idx]
            if data is not None and data != "":
                try:
                    results[field_uid] = json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    continue

        return results

    def compute_irr_data(
        self,
        irr_threshold: float = 0.1,
        masks: tuple[str, ...] = ("irr", "inv_irr"),
        use_mask: bool = True,
        lookback: int = 10,
        ndvi_threshold: float = 0.3,
        ndvi_min_start: float = 0.25,
        min_pos_days: int = 10,
        overwrite: bool = False,
    ) -> ProvenanceEvent:
        """Compute irrigation windows (irr_data) from NDVI only.

        Unlike ``dynamics()``, this does NOT require ETf data. It uses
        NDVI slope analysis and per-year irrigation properties to detect
        irrigation periods. Intended for hindcast/forecast containers
        where ETf observations are unavailable.

        ke_max, kc_max, and gwsub_data are expected to already exist
        (copied from the calibration container).

        Args:
            irr_threshold: f_irr threshold for classifying irrigated years.
            masks: Mask types to process.
            use_mask: If True, use irrigation mask properties (CONUS mode).
            lookback: Days of lookback for window extension.
            ndvi_threshold: NDVI threshold for window extension.
            min_pos_days: Minimum consecutive positive slope days.
            overwrite: If True, replace existing irr_data.
        """
        import xarray as xr

        self._ensure_writable()

        fused_path = f"derived/merged_ndvi/{masks[0]}"
        if fused_path not in self._state.root:
            raise ValueError(
                f"Merged NDVI not found at '{fused_path}'. "
                "Ingest or tile NDVI before computing irrigation windows."
            )

        with self._track_operation(
            "compute_irr_data",
            target="derived/dynamics/irr_data",
        ) as ctx:
            fields = self._state.field_uids

            # Load NDVI from all masks and combine
            ndvi_data = None
            for mask in masks:
                path = f"derived/merged_ndvi/{mask}"
                if path not in self._state.root:
                    continue
                mask_ndvi = self._state.get_xarray(path, fields=fields)
                if ndvi_data is None:
                    ndvi_data = mask_ndvi
                else:
                    ndvi_data = ndvi_data.fillna(mask_ndvi)

            if ndvi_data is None:
                self._log.warning("no_ndvi_data")
                return self._state.provenance.record(
                    "compute",
                    target="derived/dynamics/irr_data",
                    params={},
                    records_count=0,
                    success=True,
                )

            # Build a minimal dataset with dummy met so
            # _compute_irrigation_data can be reused (it only accesses
            # etf/eto/prcp in the use_lulc branch, not use_mask).
            dummy = xr.full_like(ndvi_data, np.nan)
            ds = xr.Dataset(
                {
                    "ndvi": ndvi_data,
                    "etf": dummy,
                    "eto": dummy,
                    "prcp": dummy,
                }
            )

            irr_props = self._get_yearly_irrigation_properties() if use_mask else None

            # Project f_irr into years not covered by irr_yearly
            # (e.g. forecast 2026-2100 or hindcast years beyond the
            # irrigation product). Uses median of last 5 available years.
            if irr_props:
                target_years = sorted(pd.DatetimeIndex(ds.coords["time"].values).year.unique())
                irr_props = _extend_irr_props(irr_props, target_years)

            irr_data = self._compute_irrigation_data(
                ds,
                irr_threshold,
                lookback,
                ndvi_threshold,
                ndvi_min_start,
                min_pos_days,
                use_mask=use_mask,
                use_lulc=False,
                irr_props=irr_props,
            )

            # Write irr_data only
            self._write_irr_data(irr_data, fields, overwrite)

            ctx["records_processed"] = len(fields)
            ctx["fields_processed"] = len(fields)

            event = self._state.provenance.record(
                "compute",
                target="derived/dynamics/irr_data",
                params={
                    "irr_threshold": irr_threshold,
                    "masks": list(masks),
                    "use_mask": use_mask,
                },
                fields_affected=fields,
                records_count=len(fields),
            )

            self._state.mark_modified()
            self._state.refresh()

            return event

    def _write_irr_data(
        self,
        irr_data: dict[str, dict],
        fields: list[str],
        overwrite: bool,
    ) -> None:
        """Write only irr_data to derived/dynamics/irr_data."""
        from zarr.core.dtype import VariableLengthUTF8

        irr_path = "derived/dynamics/irr_data"
        if irr_path in self._state.root and not overwrite:
            return
        if irr_path in self._state.root:
            self._safe_delete_path(irr_path)

        parent = self._state.ensure_group("derived/dynamics")
        arr = parent.create_array(
            "irr_data",
            shape=(self._state.n_fields,),
            dtype=VariableLengthUTF8(),
        )
        values = [""] * self._state.n_fields
        for field_uid in fields:
            if field_uid in self._state._uid_to_index and field_uid in irr_data:
                idx = self._state._uid_to_index[field_uid]
                values[idx] = json.dumps(irr_data[field_uid])
        arr[:] = values

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _merge_sensors(
        self,
        sensor_data: list[tuple[str, xr.DataArray]],
        preference_order: tuple[str, ...],
    ) -> xr.DataArray:
        """
        Merge multiple sensor DataArrays chronologically.

        Since sensors are already harmonized via SBAF during extraction,
        this performs a simple merge. When multiple sensors have data
        for the same date, the preferred sensor's value is used.

        Args:
            sensor_data: List of (instrument_name, DataArray) tuples
            preference_order: Instruments in order of preference

        Returns:
            Merged DataArray with values from all sensors
        """
        # Sort sensors by preference order
        pref_rank = {inst: i for i, inst in enumerate(preference_order)}
        sorted_data = sorted(sensor_data, key=lambda x: pref_rank.get(x[0], len(pref_rank)))

        # Start with highest preference sensor
        result = sorted_data[0][1].copy()

        # Fill NaN values from lower preference sensors (vectorized)
        for inst_name, da in sorted_data[1:]:
            # Use np.where for fast vectorized fill: keep result where valid, else use da
            result_vals = result.values
            da_vals = da.values
            result.values = np.where(np.isnan(result_vals), da_vals, result_vals)

        return result

    def _load_dynamics_dataset(
        self,
        fields: list[str],
        etf_model: str,
        masks: tuple[str, ...],
        instruments: tuple[str, ...],
        met_source: str,
        etf_gap_fallback_model: str | None = None,
    ) -> xr.Dataset | None:
        """
        Load all data needed for dynamics computation.

        Combines data from multiple masks - uses primary mask data where available,
        falls back to secondary mask for fields with no data in primary mask.
        This handles cases where some fields have only irr mask data and others
        have only inv_irr mask data.
        """
        import xarray as xr

        # Meteorology — prefer bias-corrected ETo when available
        eto_corr_path = f"meteorology/{met_source}/eto_corr"
        eto_raw_path = f"meteorology/{met_source}/eto"
        eto_path = eto_corr_path if eto_corr_path in self._state.root else eto_raw_path
        prcp_path = f"meteorology/{met_source}/prcp"
        if eto_path not in self._state.root or prcp_path not in self._state.root:
            self._log.warning("no_met_data", source=met_source)
            return None

        # Load NDVI from all masks and combine
        ndvi_data = None
        for mask in masks:
            # Try fused NDVI first, then raw
            fused_path = f"derived/merged_ndvi/{mask}"
            ndvi_path = f"remote_sensing/ndvi/{instruments[0]}/{mask}"

            path = fused_path if fused_path in self._state.root else ndvi_path
            if path not in self._state.root:
                continue

            mask_ndvi = self._state.get_xarray(path, fields=fields)
            if ndvi_data is None:
                ndvi_data = mask_ndvi
            else:
                # Fill NaN values from secondary mask
                ndvi_data = ndvi_data.fillna(mask_ndvi)

        if ndvi_data is None:
            self._log.warning("no_ndvi_data")
            return None

        # Load ETf from all masks and combine
        etf_data = None
        if etf_model == "ensemble":
            # Average across all available ETf models
            etf_data = self._load_ensemble_etf(fields, masks, instruments)
        else:
            for mask in masks:
                etf_path = f"remote_sensing/etf/{instruments[0]}/{etf_model}/{mask}"
                if etf_path not in self._state.root:
                    continue

                mask_etf = self._state.get_xarray(etf_path, fields=fields)
                if etf_data is None:
                    etf_data = mask_etf
                else:
                    # Fill NaN values from secondary mask
                    etf_data = etf_data.fillna(mask_etf)

        if etf_data is None:
            self._log.warning("no_etf_data", model=etf_model)
            return None

        # Load meteorology
        eto_data = self._state.get_xarray(eto_path, fields=fields)
        prcp_data = self._state.get_xarray(prcp_path, fields=fields)

        # Combine into dataset
        ds = xr.Dataset(
            {
                "ndvi": ndvi_data,
                "etf": etf_data,
                "eto": eto_data,
                "prcp": prcp_data,
            }
        )

        if etf_gap_fallback_model:
            fb_data = None
            for mask in masks:
                fb_path = f"remote_sensing/etf/{instruments[0]}/{etf_gap_fallback_model}/{mask}"
                if fb_path not in self._state.root:
                    continue
                mask_fb = self._state.get_xarray(fb_path, fields=fields)
                if fb_data is None:
                    fb_data = mask_fb
                else:
                    fb_data = fb_data.fillna(mask_fb)
            if fb_data is None:
                self._log.warning("no_etf_fallback_data", model=etf_gap_fallback_model)
            else:
                ds["etf_fallback"] = fb_data

        return ds

    def _load_ensemble_etf(
        self,
        fields: list[str],
        masks: tuple[str, ...],
        instruments: tuple[str, ...],
    ) -> xr.DataArray | None:
        """Compute the mean ETf across all available models in the container.

        Discovers every ``remote_sensing/etf/{instrument}/{model}/{mask}``
        path, loads each model's data (combining masks via fill), then
        returns the per-date, per-field mean.

        Returns:
            DataArray of ensemble-mean ETf, or ``None`` if no models found.
        """
        import xarray as xr

        instrument = instruments[0]
        etf_prefix = f"remote_sensing/etf/{instrument}"

        # Discover available models by scanning the zarr group
        if etf_prefix not in self._state.root:
            return None

        model_arrays = []
        model_group = self._state.root[etf_prefix]
        for model_name in sorted(model_group.keys()):
            model_etf = None
            for mask in masks:
                etf_path = f"{etf_prefix}/{model_name}/{mask}"
                if etf_path not in self._state.root:
                    continue
                mask_etf = self._state.get_xarray(etf_path, fields=fields)
                if model_etf is None:
                    model_etf = mask_etf
                else:
                    model_etf = model_etf.fillna(mask_etf)
            if model_etf is not None:
                model_arrays.append(model_etf)
                self._log.debug("ensemble_member_loaded", model=model_name)

        if not model_arrays:
            return None

        self._log.info("ensemble_etf_mean", n_models=len(model_arrays))
        stacked = xr.concat(model_arrays, dim="model")
        return stacked.mean(dim="model")

    def _compute_k_parameters(self, ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Compute ke_max and kc_max from ETf and NDVI data.

        ke_max: 90th percentile of ETf where NDVI < 0.3
        kc_max: 90th percentile of all ETf values

        Uses only raw observations where both ETf and NDVI are present.
        """
        import xarray as xr

        etf = ds["etf"]
        ndvi = ds["ndvi"]
        time_index = pd.DatetimeIndex(ds.coords["time"].values)

        results_ke = {}
        results_kc = {}

        for site in ds.coords["site"].values:
            site_str = str(site)

            # Extract site data as pandas Series
            site_etf = etf.sel(site=site).to_pandas()
            site_ndvi = ndvi.sel(site=site).to_pandas()

            # Use only raw observations where both ETf and NDVI are present
            valid_mask = site_etf.notna() & site_ndvi.notna()
            etf_valid = site_etf[valid_mask].values
            ndvi_valid = site_ndvi[valid_mask].values

            if len(etf_valid) == 0:
                results_ke[site_str] = 1.0
                results_kc[site_str] = 1.25
                continue

            # Create combined DataFrame (no interpolation for k-parameters)
            adf = pd.DataFrame({"etf": etf_valid, "ndvi": ndvi_valid})

            # Extract values and remove any remaining NaNs
            all_etf = adf["etf"].values.flatten()
            all_ndvi = adf["ndvi"].values.flatten()

            nan_mask = np.isnan(all_etf)
            all_etf = all_etf[~nan_mask]
            sub_ndvi = all_ndvi[~nan_mask]

            # ke_max: 90th percentile of ETf where NDVI < 0.3
            # (uses only observations where both ETf and NDVI are present)
            ke_max_mask = sub_ndvi < 0.3
            if np.any(ke_max_mask):
                ke_max = float(np.nanpercentile(all_etf[ke_max_mask], 90))
            else:
                ke_max = 1.0
                self._log.debug("no_low_ndvi", site=site_str)

            # kc_max: 90th percentile of ALL ETf observations, with floor of 1.35
            # (does NOT require NDVI to be present)
            all_etf_obs = site_etf.dropna().values
            if len(all_etf_obs) > 0:
                kc_max = max(float(np.percentile(all_etf_obs, 90)), 1.35)
            else:
                kc_max = 1.35

            results_ke[site_str] = ke_max
            results_kc[site_str] = kc_max

        # Convert back to xarray DataArrays
        sites = ds.coords["site"].values
        ke_values = [results_ke.get(str(s), 1.0) for s in sites]
        kc_values = [results_kc.get(str(s), 1.25) for s in sites]

        ke_da = xr.DataArray(ke_values, coords={"site": sites}, dims=["site"])
        kc_da = xr.DataArray(kc_values, coords={"site": sites}, dims=["site"])

        return ke_da, kc_da

    def _compute_groundwater_subsidy(self, ds: xr.Dataset, irr_threshold: float) -> dict[str, dict]:
        """
        Compute groundwater subsidy for each field and year.

        Subsidy is detected when interpolated RS-derived ET > precipitation (ratio > 1).
        f_sub = (ratio - 1) / ratio

        ETf is interpolated to daily before annual summation so that the
        ET/PPT ratio reflects the full year, not just Landsat capture dates.
        """
        etf = ds["etf"]
        eto = ds["eto"]
        ppt = ds["prcp"]

        results = {}
        all_years = sorted(pd.DatetimeIndex(ds.coords["time"].values).year.unique())

        for site in ds.coords["site"].values:
            site_str = str(site)
            site_data = {}

            # Extract site data once as pandas Series (fast)
            site_etf_s = etf.sel(site=site).to_pandas()
            site_eto_s = eto.sel(site=site).to_pandas()
            site_ppt_s = ppt.sel(site=site).to_pandas()

            # Interpolate ETf to daily before computing ET (matches irrigation path)
            etf_interp = site_etf_s.interpolate()
            etf_interp = etf_interp.bfill().ffill()
            site_eta_s = etf_interp * site_eto_s

            # Raw ETf sum to detect years with valid captures
            raw_etf_yearly = site_etf_s.groupby(site_etf_s.index.year).sum()

            # Pre-compute yearly sums using pandas groupby (vectorized)
            years_idx = site_eta_s.index.year
            eta_yearly = site_eta_s.groupby(years_idx).sum()
            ppt_yearly = site_ppt_s.groupby(years_idx).sum()

            # Pre-compute monthly sums for subsidy month detection
            monthly_idx = site_eta_s.index.to_period("M")
            eta_monthly = site_eta_s.groupby(monthly_idx).sum()
            ppt_monthly = site_ppt_s.groupby(monthly_idx).sum()

            # Track years with valid ETf data
            etf_years = []
            gw_count = 0

            for yr in all_years:
                # Check if this year has valid ETf captures (raw, not interpolated)
                etf_yr_sum = raw_etf_yearly.get(yr, 0)
                if etf_yr_sum <= 0:
                    continue

                etf_years.append(int(yr))

                eta_yr = eta_yearly.get(yr, 0)
                ppt_yr = ppt_yearly.get(yr, 0)

                if ppt_yr <= 0:
                    continue

                ratio = eta_yr / (ppt_yr + 1.0)

                if ratio > 1:
                    subsidized = 1
                    f_sub = (ratio - 1) / ratio
                else:
                    subsidized = 0
                    f_sub = 0.0

                # Find months where eta > ppt for this year
                months = []
                for month in range(1, 13):
                    period = pd.Period(year=yr, month=month, freq="M")
                    if period in eta_monthly.index and period in ppt_monthly.index:
                        e = eta_monthly[period]
                        p = ppt_monthly[period]
                        if not np.isnan(e) and not np.isnan(p) and e > p:
                            months.append(month)

                site_data[int(yr)] = {
                    "subsidized": subsidized,
                    "f_sub": float(f_sub),
                    "ratio": float(ratio),
                    "months": months,
                    "ppt": float(ppt_yr),
                    "eta": float(eta_yr),
                }

                if f_sub > 0.1:
                    gw_count += 1

            # Impute missing years if >50% of years with data are subsidized
            missing_years = [y for y in all_years if y not in etf_years]
            if len(etf_years) > 0 and gw_count / len(etf_years) > 0.5 and missing_years:
                site_data = self._impute_missing_gwsub(site_data, etf_years, missing_years)

            results[site_str] = site_data

        return results

    def _impute_missing_gwsub(
        self,
        site_data: dict,
        etf_years: list[int],
        missing_years: list[int],
    ) -> dict:
        """
        Impute groundwater subsidy for years without ETf data.

        If >50% of years with data show subsidy (f_sub > 0.1),
        fills missing years with mean values from subsidized years.
        Matches original SamplePlotDynamics behavior.
        """
        # Calculate means from years with data
        mean_sub = np.mean(
            [site_data[y]["f_sub"] for y in etf_years if y in site_data and "f_sub" in site_data[y]]
        )

        # Collect all subsidy months across years
        all_months = []
        for y in etf_years:
            if y in site_data and "months" in site_data[y]:
                all_months.extend(site_data[y]["months"])
        unique_months = list(set(all_months))

        mean_ppt = np.mean(
            [site_data[y]["ppt"] for y in etf_years if y in site_data and "ppt" in site_data[y]]
        )
        mean_eta = np.mean(
            [site_data[y]["eta"] for y in etf_years if y in site_data and "eta" in site_data[y]]
        )

        for yr in missing_years:
            site_data[int(yr)] = {
                "subsidized": 1,
                "f_sub": float(mean_sub),
                "f_irr": 0.0,  # Not irrigated (no ETf data)
                "ratio": float(mean_eta / mean_ppt) if mean_ppt > 0 else 0.0,
                "months": unique_months,
                "ppt": float(mean_ppt),
                "eta": float(mean_eta),
            }

        return site_data

    def _compute_irrigation_data(
        self,
        ds: xr.Dataset,
        irr_threshold: float,
        lookback: int,
        ndvi_threshold: float,
        ndvi_min_start: float,
        min_pos_days: int,
        use_mask: bool,
        use_lulc: bool,
        irr_props: dict[str, dict[str, float]] | None = None,
        gwsub_data: dict[str, dict] | None = None,
        lulc_irr_method: str = "annual_2yr",
        annual_subsidy_ratio: float = 1.3,
        annual_retain_ratio: float = 1.1,
        season_storage_mm: float = 150.0,
    ) -> dict[str, dict]:
        """
        Compute irrigation windows for each field and year.

        Two modes for determining irrigation status:

        use_mask=True (CONUS):
            Reads f_irr from irr_props (per-year irrigation fraction from properties).
            Year is irrigated if f_irr > irr_threshold.
            Fallback: if not irrigated by mask but gwsub_data shows ET > PPT
            and LULC is cropland, override to irrigated (mask data missed it).

        use_lulc=True (International):
            Computes irrigation from a water-balance subsidy test on the nanmean
            ETf ensemble, gated on the field being cropped (annual rooting
            habit). ``lulc_irr_method`` selects the test — "annual_2yr" (default)
            is a two-stage rule: Stage 1 marks a field irrigation-equipped only
            if MORE THAN ONE-THIRD of its rolling 2-year sum(ET)/sum(PPT)
            windows exceed annual_subsidy_ratio (robust to a lone drought/
            boundary year and to dryland moisture banking); Stage 2 then flags,
            within an equipped field, each year whose single-year ET/PPT exceeds
            annual_retain_ratio, or whose demand-season balance
            ET/(PPT + season_storage_mm) exceeds annual_subsidy_ratio (rescues
            wet-winter years the calendar-year ratio dilutes).
            "annual" is the same rule with single-year Stage 1 windows;
            "monthly" is the legacy >=3-months-of-ET/(PPT+1)>1.3 test.

        Uses NDVI slope analysis to detect irrigation periods.
        """
        ndvi = ds["ndvi"]
        etf = ds["etf"]
        eto = ds["eto"]
        ppt = ds["prcp"]

        results = {}
        years = sorted(pd.DatetimeIndex(ds.coords["time"].values).year.unique())
        time_index = pd.DatetimeIndex(ds.coords["time"].values)

        # LULC needed for both paths: use_lulc (original) and use_mask (fallback)
        lulc_by_site = self._get_lulc_by_site(ds.coords["site"].values)

        # Track years needing backfill (irrigated but no windows detected)
        backfill_tracker = {}

        for site in ds.coords["site"].values:
            site_str = str(site)
            site_data = {}
            fallow_years = []
            years_needing_backfill = []

            # Check if field is cropped
            lulc_info = lulc_by_site.get(site_str)
            cropped = schema.is_cropland(*lulc_info) if lulc_info else False
            if not use_lulc and lulc_info is None:
                # No LULC in the container: keep the legacy CONUS assumption
                # that a gwsub-subsidized field is cropped so the fallback can
                # rescue mask-missed irrigation. With LULC present, natural
                # vegetation whose ET exceeds precipitation (phreatophyte
                # shrubland, wet meadow) must stay non-irrigated — the
                # subsidy is groundwater, not irrigation.
                cropped = True

            # Get per-year irrigation properties for this site if using mask mode
            site_irr_props = irr_props.get(site_str, {}) if irr_props else {}

            # Extract site data once as pandas Series (fast) - avoid repeated .sel() in year loop
            site_etf_s = etf.sel(site=site).to_pandas()
            site_eto_s = eto.sel(site=site).to_pandas()
            site_ppt_s = ppt.sel(site=site).to_pandas()
            site_ndvi_s = ndvi.sel(site=site).to_pandas()

            # Pre-compute eta and ppt sums for use_lulc mode (vectorized).
            # Interpolate ETf to daily (matches legacy _build_et_frame behavior)
            # so the water balance reflects the full year, not capture dates.
            if use_lulc:
                # Irrigation water balance runs on the nanmean ETf ensemble:
                # the mean of all available ETf members per date (the primary
                # model plus any configured second member). This matches the
                # ensemble used as the calibration target and averages out
                # single-model bias (e.g. PT-JPL running high over cropland).
                # With no second member this is just the primary model.
                if "etf_fallback" in ds:
                    site_fb_s = ds["etf_fallback"].sel(site=site).to_pandas()
                    ens_etf = pd.concat([site_etf_s, site_fb_s], axis=1).mean(axis=1, skipna=True)
                    cap_daily = site_etf_s.notna() | site_fb_s.notna()
                else:
                    ens_etf = site_etf_s
                    cap_daily = site_etf_s.notna()
                etf_interp = ens_etf.interpolate().bfill().ffill()
                site_eta_s = etf_interp * site_eto_s
                monthly_idx = site_eta_s.index.to_period("M")
                eta_monthly_all = site_eta_s.groupby(monthly_idx).sum()
                ppt_monthly_all = site_ppt_s.groupby(monthly_idx).sum()
                eto_monthly_all = site_eto_s.groupby(monthly_idx).sum()
                # Demand season: months whose climatological ETo exceeds PPT —
                # the part of the year where sustained ET cannot come from
                # concurrent rain (used by the Stage 2 season test)
                clim_eto = eto_monthly_all.groupby(eto_monthly_all.index.month).mean()
                clim_ppt = ppt_monthly_all.groupby(ppt_monthly_all.index.month).mean()
                season_months = [
                    m for m in range(1, 13) if clim_eto.get(m, 0.0) > clim_ppt.get(m, 0.0)
                ]
                year_idx = site_eta_s.index.year
                eta_yearly_all = site_eta_s.groupby(year_idx).sum()
                ppt_yearly_all = site_ppt_s.groupby(year_idx).sum()
                cap_yearly = cap_daily.groupby(cap_daily.index.year).sum()

                # Stage 1 (annual/annual_2yr): site-level "equipped for
                # irrigation" decision from the fraction of subsidized rolling
                # win-year balance windows. A field is irrigation-equipped only
                # if MORE THAN ONE-THIRD of its windows are subsidized (ET/PPT >
                # annual_subsidy_ratio). The one-third floor keeps the site
                # classification robust to a single drought or boundary year (a
                # lone anomalous year touches ~2 windows, well under the floor,
                # so it cannot flip a dryland field), while still admitting
                # genuine intermittent-irrigation sites (e.g. US-MH2 at ~6/13).
                equipped = False
                if cropped and lulc_irr_method in ("annual", "annual_2yr"):
                    win = 2 if lulc_irr_method == "annual_2yr" else 1
                    n_sub = n_win = 0
                    for y in years:
                        wy = [yy for yy in range(y - win + 1, y + 1) if yy in eta_yearly_all.index]
                        if not wy:  # boundary year: look forward instead
                            wy = [yy for yy in range(y, y + win) if yy in eta_yearly_all.index]
                        if sum(int(cap_yearly.get(yy, 0)) for yy in wy) == 0:
                            continue
                        ppt_w = sum(float(ppt_yearly_all.get(yy, 0.0)) for yy in wy)
                        if ppt_w <= 0:
                            continue
                        eta_w = sum(float(eta_yearly_all.get(yy, 0.0)) for yy in wy)
                        n_win += 1
                        if eta_w / (ppt_w + 1.0) > annual_subsidy_ratio:
                            n_sub += 1
                    equipped = n_win > 0 and (n_sub * 3 > n_win)

            for yr in years:
                yr_str = str(yr)

                # Determine irrigation status based on mode
                if use_mask:
                    # CONUS mode: read f_irr from properties
                    f_irr = site_irr_props.get(yr_str, np.nan)
                    if pd.isna(f_irr):
                        f_irr = 0.0
                    irrigated = f_irr > irr_threshold

                    # Fallback: mask missed irrigation at a cropland site where
                    # RS-derived ET exceeds precipitation (gwsub detected subsidy).
                    # For cropland the water source is irrigation, not groundwater.
                    if not irrigated and gwsub_data:
                        yr_gw = gwsub_data.get(site_str, {}).get(int(yr), {})
                        if yr_gw.get("subsidized", 0) and cropped:
                            irrigated = True
                            f_irr = 1.0

                elif use_lulc:
                    # International mode: water-balance subsidy test, gated on
                    # the field being cropped (annual rooting habit).
                    if lulc_irr_method == "monthly":
                        # Legacy: count months where in-season RS ET exceeds
                        # 1.3x rain. Over-flags moisture-banking dryland systems
                        # (e.g. wheat-fallow) whose biennial balance closes.
                        subsidy = 0
                        for month in range(1, 13):
                            period = pd.Period(year=yr, month=month, freq="M")
                            if period in eta_monthly_all.index and period in ppt_monthly_all.index:
                                e = eta_monthly_all[period]
                                p = ppt_monthly_all[period]
                                if not np.isnan(e) and not np.isnan(p) and p > 0:
                                    if e / (p + 1.0) > 1.3:
                                        subsidy += 1
                        irrigated = subsidy >= 3 and cropped
                    else:
                        # Stage 2: within an irrigation-equipped field (the
                        # Stage 1 mode test, precomputed above), a year stays
                        # irrigated if its single-year balance exceeds the
                        # retain threshold (lower than the Stage 1 equip
                        # threshold: an established irrigated field should not
                        # flip to fallow on a marginal year), OR if its
                        # demand-season balance is subsidized. Wet winters in
                        # Mediterranean climates dilute the calendar-year ratio
                        # below any workable threshold even though the crop
                        # transpires through a nearly rain-free summer; the
                        # season test scopes the balance to months where ET
                        # cannot come from concurrent rain, with a soil-storage
                        # allowance so senesced/fallow years drawing stored
                        # winter moisture do not pass. Dryland fields are never
                        # equipped, so neither test can create spurious
                        # irrigation there.
                        if equipped:
                            ppt_y = float(ppt_yearly_all.get(yr, 0.0))
                            eta_y = float(eta_yearly_all.get(yr, 0.0))
                            ratio = eta_y / (ppt_y + 1.0) if ppt_y > 0 else 0.0
                            irrigated = ratio > annual_retain_ratio
                            if not irrigated and len(season_months) >= 3:
                                # season test only on years whose full demand
                                # season lies within the record (a truncated
                                # boundary year would understate season sums)
                                season_start = pd.Timestamp(year=yr, month=season_months[0], day=1)
                                season_end = pd.Period(
                                    year=yr, month=season_months[-1], freq="M"
                                ).end_time
                                if (
                                    season_start >= time_index.min()
                                    and season_end <= time_index.max()
                                ):
                                    periods = [
                                        pd.Period(year=yr, month=m, freq="M") for m in season_months
                                    ]
                                    sea_eta = sum(
                                        float(eta_monthly_all.get(p, 0.0)) for p in periods
                                    )
                                    sea_ppt = sum(
                                        float(ppt_monthly_all.get(p, 0.0)) for p in periods
                                    )
                                    irrigated = (
                                        sea_eta / (sea_ppt + season_storage_mm)
                                        > annual_subsidy_ratio
                                    )
                        else:
                            irrigated = False

                    f_irr = 1.0 if irrigated else 0.0
                else:
                    raise ValueError("Must use either use_mask or use_lulc for irrigation analysis")

                if not irrigated:
                    fallow_years.append(int(yr))
                    site_data[int(yr)] = {
                        "irr_doys": [],
                        "irrigated": 0,
                        "f_irr": float(f_irr),
                    }
                    continue

                # Get NDVI with extended-year context for boundary handling (using pre-extracted pandas)
                ndvi_series = self._get_extended_year_ndvi_fast(site_ndvi_s, yr, years)

                # Detect irrigation windows from NDVI patterns
                irr_doys = self._detect_irrigation_windows(
                    ndvi_series,
                    lookback,
                    ndvi_threshold,
                    min_pos_days,
                    yr,
                    ndvi_min_start=ndvi_min_start,
                )

                # Track for backfill if irrigated but no windows detected
                if len(irr_doys) == 0:
                    years_needing_backfill.append(int(yr))

                site_data[int(yr)] = {
                    "irr_doys": irr_doys,
                    "irrigated": int(irrigated),
                    "f_irr": float(f_irr),
                }

            site_data["fallow_years"] = fallow_years
            backfill_tracker[site_str] = years_needing_backfill
            results[site_str] = site_data

        # Backfill irrigation windows from nearest year with data
        results = self._backfill_irrigation_windows(results, backfill_tracker)

        return results

    def _get_lulc_by_site(self, sites: np.ndarray) -> dict[str, tuple[int, str]]:
        """Get LULC code and source for each site from container properties.

        Returns dict mapping site_str -> (code, source) where source is
        "glc10" or "modis". GLC10 is preferred; MODIS is the fallback.
        """
        glc10_path = "properties/land_cover/glc10"
        modis_path = "properties/land_cover/modis_lc"

        has_glc10 = glc10_path in self._state.root
        has_modis = modis_path in self._state.root
        if not has_glc10 and not has_modis:
            return {}

        glc10_arr = self._state.root[glc10_path] if has_glc10 else None
        modis_arr = self._state.root[modis_path] if has_modis else None

        result = {}
        for site in sites:
            site_str = str(site)
            if site_str not in self._state._uid_to_index:
                continue
            idx = self._state._uid_to_index[site_str]

            # Try GLC10 first
            if glc10_arr is not None:
                val = glc10_arr[idx]
                if not np.isnan(val) and int(val) != -1:
                    result[site_str] = (int(val), "glc10")
                    continue

            # Fall back to MODIS
            if modis_arr is not None:
                val = modis_arr[idx]
                if not np.isnan(val) and int(val) != -1:
                    result[site_str] = (int(val), "modis")

        return result

    def _get_yearly_irrigation_properties(self) -> dict[str, dict[str, float]]:
        """
        Get per-year irrigation fraction from container properties.

        Returns:
            Dict mapping site_id -> {year_str: f_irr, ...}
            e.g., {"US-FPe": {"2020": 0.0, "2021": 0.0}, "ALARC2_Smith6": {"2020": 1.0}}
        """
        import json

        yearly_path = "properties/irrigation/irr_yearly"
        if yearly_path not in self._state.root:
            return {}

        arr = self._state.root[yearly_path]
        result = {}

        for site_str in self._state.field_uids:
            if site_str in self._state._uid_to_index:
                idx = self._state._uid_to_index[site_str]
                json_str = arr[idx]
                # Handle zarr v3 ndarray returns
                if hasattr(json_str, "item"):
                    json_str = json_str.item()
                if json_str:
                    try:
                        result[site_str] = json.loads(json_str)
                    except json.JSONDecodeError:
                        result[site_str] = {}
        return result

    def _get_extended_year_ndvi(
        self,
        ndvi: xr.DataArray,
        site: str,
        year: int,
        years: list[int],
        time_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Get NDVI with extended year context for boundary handling.

        Uses ±1 year of data for smoother detection at year boundaries,
        matching the original SamplePlotDynamics._compose_ndvi() behavior.
        """
        # Determine extended years based on position in range
        if year == years[0]:
            extended_years = [year, year + 1] if len(years) > 1 else [year]
        elif year == years[-1]:
            extended_years = [year - 1, year]
        else:
            extended_years = [year - 1, year, year + 1]

        ext_mask = time_index.year.isin(extended_years)
        return ndvi.sel(site=site).isel(time=ext_mask).to_pandas()

    def _get_extended_year_ndvi_fast(
        self,
        site_ndvi_s: pd.Series,
        year: int,
        years: list[int],
    ) -> pd.Series:
        """
        Get NDVI with extended year context for boundary handling (pandas version).

        Uses ±1 year of data for smoother detection at year boundaries.
        This is the fast pandas-based version of _get_extended_year_ndvi.
        """
        # Determine extended years based on position in range
        if year == years[0]:
            extended_years = [year, year + 1] if len(years) > 1 else [year]
        elif year == years[-1]:
            extended_years = [year - 1, year]
        else:
            extended_years = [year - 1, year, year + 1]

        ext_mask = site_ndvi_s.index.year.isin(extended_years)
        return site_ndvi_s[ext_mask]

    def _backfill_irrigation_windows(
        self,
        irr_data: dict[str, dict],
        backfill_tracker: dict[str, list[int]],
    ) -> dict[str, dict]:
        """
        Backfill irrigation DOYs from nearest year with data.

        For irrigated years that had no detected irrigation windows,
        copies the windows from the nearest year that has data.
        Matches original SamplePlotDynamics._backfill_irrigation_days().
        """
        for site_str, years_needing_backfill in backfill_tracker.items():
            if not years_needing_backfill or site_str not in irr_data:
                continue

            site_data = irr_data[site_str]

            # Find candidate years with actual irrigation windows
            candidates = [
                int(y)
                for y, v in site_data.items()
                if isinstance(v, dict)
                and v.get("f_irr", 0) > 0
                and v.get("irr_doys")
                and y != "fallow_years"
            ]

            if not candidates:
                continue

            for yr in years_needing_backfill:
                if yr in site_data and not site_data[yr].get("irr_doys"):
                    # Find nearest year with data
                    nearest = min(candidates, key=lambda x: abs(x - yr))
                    site_data[yr]["irr_doys"] = site_data[nearest]["irr_doys"].copy()

        return irr_data

    def _detect_irrigation_windows(
        self,
        ndvi_series: pd.Series,
        lookback: int,
        ndvi_threshold: float,
        min_pos_days: int,
        year: int,
        ndvi_min_start: float = 0.25,
    ) -> list[int]:
        """
        Detect irrigation windows from NDVI time series.

        Algorithm:
        1. Apply 32-day rolling mean to smooth NDVI
        2. Compute slope (diff)
        3. Find consecutive positive slope periods >= min_pos_days
        4. Skip groups where smoothed NDVI never reaches ndvi_min_start
           (filters dormant-season noise)
        5. Extend windows by lookback and until NDVI drops below a
           decay threshold (max of ndvi_threshold and 50% of peak NDVI)
        """
        ydf = pd.DataFrame({"ndvi": ndvi_series})

        # Interpolate and smooth
        ydf["ndvi"] = ydf["ndvi"].interpolate()
        ydf["ndvi"] = ydf["ndvi"].bfill().ffill()
        ydf["ndvi"] = ydf["ndvi"].rolling(window=32, center=True).mean()
        ydf["ndvi"] = ydf["ndvi"].bfill().ffill()

        # Compute slope
        ydf["diff"] = ydf["ndvi"].diff()

        # Check data quality - if all NaN after processing, skip
        if np.count_nonzero(np.isnan(ydf["ndvi"].values)) > 200:
            return []

        # Find local minima (where slope goes from negative to positive)
        local_min_indices = ydf[(ydf["diff"] > 0) & (ydf["diff"].shift(1) < 0)].index

        # Group consecutive positive/negative slope days
        positive_slope = ydf["diff"] > 0
        groups = (positive_slope != positive_slope.shift()).cumsum()
        ydf["groups"] = groups

        # Find groups with >= min_pos_days of positive slope
        group_counts = positive_slope.groupby(groups).sum()
        long_positive_slope_groups = group_counts[group_counts >= min_pos_days].index

        irr_doys = []

        for group in long_positive_slope_groups:
            group_indices = positive_slope[groups == group].index

            if len(group_indices) == 0:
                continue

            start_index = group_indices[0]
            end_index = group_indices[-1]

            # Skip groups where NDVI never reaches ndvi_min_start —
            # these are dormant-season noise (e.g. NDVI 0.19 → 0.22)
            group_ndvi = ydf.loc[group_indices, "ndvi"]
            if group_ndvi.max() < ndvi_min_start:
                continue

            # If the group starts below ndvi_min_start, clip to where
            # NDVI first crosses above it
            if ydf.loc[start_index, "ndvi"] < ndvi_min_start:
                above = group_indices[group_ndvi.values >= ndvi_min_start]
                if len(above) == 0:
                    continue
                start_index = above[0]

            # Extend start by lookback if at local minimum
            if start_index in local_min_indices:
                start_day = start_index - pd.Timedelta(days=lookback)
            else:
                start_day = start_index

            # Forward extension: stop when NDVI drops below a decay
            # threshold tied to the peak within this window, so the
            # window doesn't run deep into senescence
            peak_ndvi = ydf.loc[start_index:end_index, "ndvi"].max()
            decay_threshold = max(ndvi_threshold, peak_ndvi * 0.5)

            end_day = end_index + pd.Timedelta(days=2)
            try:
                ndvi_val = ydf.loc[end_day - pd.Timedelta(days=1), "ndvi"]
                while ndvi_val > decay_threshold and end_day in ydf.index:
                    end_day += pd.Timedelta(days=1)
                    ndvi_val = ydf.loc[end_day - pd.Timedelta(days=1), "ndvi"]
            except (KeyError, TypeError):
                pass

            end_day = end_day + pd.Timedelta(days=1)

            # Extract DOYs for the target year only
            try:
                date_range = pd.date_range(start_day, end_day)
                doys = [d.dayofyear for d in date_range if d.year == year]
                irr_doys.extend(doys)
            except Exception:
                continue

        return sorted(list(set(irr_doys)))

    def _write_dynamics_results(
        self,
        ke_max: xr.DataArray,
        kc_max: xr.DataArray,
        irr_data: dict[str, dict],
        gwsub_data: dict[str, dict],
        fields: list[str],
        overwrite: bool,
    ) -> None:
        """Write computed dynamics results to container."""
        from zarr.core.dtype import VariableLengthUTF8

        # Write ke_max
        ke_path = "derived/dynamics/ke_max"
        if ke_path in self._state.root and not overwrite:
            pass
        else:
            if ke_path in self._state.root:
                self._safe_delete_path(ke_path)
            arr = self._state.create_property_array(ke_path)
            for site in ke_max.coords["site"].values:
                if str(site) in self._state._uid_to_index:
                    idx = self._state._uid_to_index[str(site)]
                    arr[idx] = float(ke_max.sel(site=site).values)

        # Write kc_max
        kc_path = "derived/dynamics/kc_max"
        if kc_path in self._state.root and not overwrite:
            pass
        else:
            if kc_path in self._state.root:
                self._safe_delete_path(kc_path)
            arr = self._state.create_property_array(kc_path)
            for site in kc_max.coords["site"].values:
                if str(site) in self._state._uid_to_index:
                    idx = self._state._uid_to_index[str(site)]
                    arr[idx] = float(kc_max.sel(site=site).values)

        # Write irr_data as JSON strings
        irr_path = "derived/dynamics/irr_data"
        if irr_path in self._state.root and not overwrite:
            pass
        else:
            if irr_path in self._state.root:
                self._safe_delete_path(irr_path)
            parent = self._state.ensure_group("derived/dynamics")
            arr = parent.create_array(
                "irr_data",
                shape=(self._state.n_fields,),
                dtype=VariableLengthUTF8(),
            )
            # Build values list then assign at once
            values = [""] * self._state.n_fields
            for field_uid in fields:
                if field_uid in self._state._uid_to_index and field_uid in irr_data:
                    idx = self._state._uid_to_index[field_uid]
                    values[idx] = json.dumps(irr_data[field_uid])
            arr[:] = values

        # Write gwsub_data as JSON strings
        gwsub_path = "derived/dynamics/gwsub_data"
        if gwsub_path in self._state.root and not overwrite:
            pass
        else:
            if gwsub_path in self._state.root:
                self._safe_delete_path(gwsub_path)
            parent = self._state.ensure_group("derived/dynamics")
            arr = parent.create_array(
                "gwsub_data",
                shape=(self._state.n_fields,),
                dtype=VariableLengthUTF8(),
            )
            # Build values list then assign at once
            values = [""] * self._state.n_fields
            for field_uid in fields:
                if field_uid in self._state._uid_to_index and field_uid in gwsub_data:
                    idx = self._state._uid_to_index[field_uid]
                    values[idx] = json.dumps(gwsub_data[field_uid])
            arr[:] = values


def _extend_irr_props(
    irr_props: dict[str, dict[str, float]],
    target_years: list[int],
    n_recent: int = 5,
) -> dict[str, dict[str, float]]:
    """Extend per-year irrigation fractions to cover target years.

    For years not present in the irrigation product (e.g. forecast
    2026-2100), uses the median f_irr from the last *n_recent*
    available years as a projection.  This prevents all future years
    from collapsing to non-irrigated.

    Args:
        irr_props: {site_id: {year_str: f_irr, ...}, ...}
        target_years: All years in the target container.
        n_recent: Number of recent years to use for median projection.

    Returns:
        Updated irr_props with projected years filled in.
    """
    target_year_strs = {str(y) for y in target_years}

    for site, yearly in irr_props.items():
        available_years = sorted(yearly.keys())
        if not available_years:
            continue

        missing = target_year_strs - set(available_years)
        if not missing:
            continue

        # Median f_irr from last n_recent years
        recent = available_years[-n_recent:]
        recent_vals = [yearly[y] for y in recent]
        projected = float(np.median(recent_vals))

        for yr_str in missing:
            yearly[yr_str] = projected

    return irr_props
