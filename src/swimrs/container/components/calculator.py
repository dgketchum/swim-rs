"""
Calculator component for derived data computation.

Provides a clean API for computing derived products from ingested data.
Usage: container.compute.dynamics(...) instead of container.compute_dynamics(...)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import Component

if TYPE_CHECKING:
    import xarray as xr
    from swimrs.container.state import ContainerState
    from swimrs.container.provenance import ProvenanceEvent


class Calculator(Component):
    """
    Component for computing derived data products.

    Provides methods for computing dynamics (irrigation detection,
    groundwater subsidy, K-parameters) and fusing multi-sensor NDVI.

    Example:
        container.compute.dynamics(etf_model="ssebop")
        container.compute.fused_ndvi()
    """

    def __init__(self, state: "ContainerState", container=None):
        """
        Initialize the Calculator.

        Args:
            state: ContainerState instance
            container: Optional reference to parent SwimContainer
        """
        super().__init__(state, container)

    def fused_ndvi(
        self,
        masks: Tuple[str, ...] = ("irr", "inv_irr", "no_mask"),
        instrument1: str = "landsat",
        instrument2: str = "sentinel",
        min_pairs: int = 20,
        window_days: int = 5,
        overwrite: bool = False,
    ) -> "ProvenanceEvent":
        """
        Compute fused NDVI by combining multiple sensors.

        Uses quantile mapping to harmonize NDVI from different sensors
        (typically Landsat and Sentinel) into a combined time series
        with better temporal coverage.

        Args:
            masks: Mask types to process
            instrument1: Primary instrument (base for fusion)
            instrument2: Secondary instrument to fuse
            min_pairs: Minimum paired observations for quantile mapping
            window_days: Time window for finding paired observations
            overwrite: If True, replace existing results

        Returns:
            ProvenanceEvent recording the operation
        """
        self._ensure_writable()

        with self._track_operation(
            "compute_fused_ndvi",
            target="derived/combined_ndvi",
            instrument1=instrument1,
            instrument2=instrument2,
        ) as ctx:
            total_records = 0
            fields_processed = set()

            for mask in masks:
                path = f"derived/combined_ndvi/{mask}"

                if path in self._state.root and not overwrite:
                    self._log.debug("skipping_existing", path=path)
                    continue
                if path in self._state.root:
                    self._safe_delete_path(path)

                # Load NDVI from both instruments
                path1 = f"remote_sensing/ndvi/{instrument1}/{mask}"
                path2 = f"remote_sensing/ndvi/{instrument2}/{mask}"

                if path1 not in self._state.root:
                    self._log.warning("missing_source", path=path1)
                    continue

                # Get data as xarray
                da1 = self._state.get_xarray(path1)

                # Check if second instrument exists
                if path2 in self._state.root:
                    da2 = self._state.get_xarray(path2)
                    # Apply quantile mapping to harmonize
                    fused = self._fuse_sensors(da1, da2, min_pairs, window_days)
                else:
                    # Just use first instrument
                    fused = da1

                # Write result
                arr = self._state.create_timeseries_array(path)
                arr[:, :] = fused.values

                total_records += int(np.count_nonzero(~np.isnan(fused.values)))
                fields_processed.update(fused.coords["site"].values)

            ctx["records_processed"] = total_records
            ctx["fields_processed"] = len(fields_processed)

            event = self._state.provenance.record(
                "compute",
                target="derived/combined_ndvi",
                params={
                    "masks": list(masks),
                    "instrument1": instrument1,
                    "instrument2": instrument2,
                    "min_pairs": min_pairs,
                    "window_days": window_days,
                },
                fields_affected=list(fields_processed),
                records_count=total_records,
            )

            self._state.mark_modified()
            self._state.refresh()

            return event

    def dynamics(
        self,
        etf_model: str = "ssebop",
        irr_threshold: float = 0.1,
        masks: Tuple[str, ...] = ("irr", "inv_irr"),
        instruments: Tuple[str, ...] = ("landsat", "sentinel"),
        use_mask: bool = False,
        use_lulc: bool = True,
        lookback: int = 10,
        ndvi_threshold: float = 0.3,
        min_pos_days: int = 10,
        met_source: str = "gridmet",
        fields: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> "ProvenanceEvent":
        """
        Compute field dynamics: irrigation detection, groundwater subsidy, K-parameters.

        This is the main computation method that:
        1. Detects irrigation events per year using ETf/NDVI patterns
        2. Calculates groundwater subsidy (ET/PPT ratio)
        3. Extracts Ke (evaporation) and Kc (crop) parameters

        Two modes for determining irrigation status:

        use_mask=True (CONUS - Examples 4 & 5):
            - Reads per-year irrigation fraction from properties/irrigation/irr_yearly
            - Year is irrigated if f_irr > irr_threshold
            - Requires masks=("irr", "inv_irr")

        use_lulc=True (International - Example 6):
            - Computes irrigation from water balance (ET/PPT ratio)
            - Year is irrigated if subsidy_months >= 3 AND field is cropped (LULC)
            - Works with masks=("no_mask",)

        Args:
            etf_model: ET model to use ("ssebop", "ptjpl", etc.)
            irr_threshold: Fraction threshold for classifying irrigated years
            masks: Mask types to process
            instruments: Instruments for NDVI data
            use_mask: If True, use irrigation mask properties (CONUS mode)
            use_lulc: If True, use water balance + LULC (International mode)
            lookback: Days of lookback for irrigation window extension
            ndvi_threshold: NDVI threshold for window extension
            min_pos_days: Minimum consecutive positive slope days
            met_source: Meteorology source ("gridmet", "era5")
            fields: Optional list of field UIDs to process
            overwrite: If True, replace existing results

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

            # Check for fused NDVI and warn if missing
            fused_path = f"derived/combined_ndvi/{masks[0]}"
            if fused_path not in self._state.root:
                self._log.warning(
                    "fused_ndvi_missing",
                    message="Fused NDVI not found. Consider running compute.fused_ndvi() first for better results.",
                )

            # Load required data
            ds = self._load_dynamics_dataset(
                target_fields, etf_model, masks, instruments, met_source
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
                ds, irr_threshold, lookback, ndvi_threshold, min_pos_days,
                use_mask, use_lulc, irr_props
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
                    "lookback": lookback,
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
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
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

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _fuse_sensors(
        self,
        da1: "xr.DataArray",
        da2: "xr.DataArray",
        min_pairs: int,
        window_days: int,
    ) -> "xr.DataArray":
        """
        Fuse two sensor DataArrays using quantile mapping.

        For each site, finds paired observations within window_days
        and uses them to build a quantile mapping from sensor2 to sensor1.
        """
        import xarray as xr
        from scipy.stats import percentileofscore

        # Result array starts as copy of primary sensor
        result = da1.copy()

        for site in da1.coords["site"].values:
            s1 = da1.sel(site=site).to_pandas().dropna()
            s2 = da2.sel(site=site).to_pandas().dropna()

            if s1.empty or s2.empty:
                continue

            # Find paired observations within tolerance
            s1_df = s1.rename("val1").rename_axis("time").reset_index()
            s2_df = s2.rename("val2").rename_axis("time").reset_index()

            paired = pd.merge_asof(
                s1_df.sort_values("time"),
                s2_df.sort_values("time"),
                on="time",
                direction="nearest",
                tolerance=pd.Timedelta(days=window_days),
            )
            paired = paired.dropna(subset=["val1", "val2"])

            if len(paired) < min_pairs:
                # Not enough pairs, just fill with sensor2 values directly
                for t in s2.index:
                    if pd.isna(result.sel(site=site, time=t).values):
                        result.loc[dict(site=site, time=t)] = s2[t]
                continue

            # Build quantile mapping
            train_vals1 = paired["val1"].values
            train_vals2 = paired["val2"].values

            # Adjust sensor2 values and fill gaps in sensor1
            for t in s2.index:
                if pd.isna(result.sel(site=site, time=t).values):
                    # Apply quantile mapping
                    val2 = s2[t]
                    pctl = percentileofscore(train_vals2, val2, kind="weak")
                    pctl = np.clip(pctl, 0, 100)
                    adjusted = np.percentile(train_vals1, pctl)
                    result.loc[dict(site=site, time=t)] = adjusted

        return result

    def _load_dynamics_dataset(
        self,
        fields: List[str],
        etf_model: str,
        masks: Tuple[str, ...],
        instruments: Tuple[str, ...],
        met_source: str,
    ) -> Optional["xr.Dataset"]:
        """
        Load all data needed for dynamics computation.

        Combines data from multiple masks - uses primary mask data where available,
        falls back to secondary mask for fields with no data in primary mask.
        This handles cases where some fields have only irr mask data and others
        have only inv_irr mask data.
        """
        import xarray as xr

        # Meteorology
        eto_path = f"meteorology/{met_source}/eto"
        prcp_path = f"meteorology/{met_source}/prcp"
        if eto_path not in self._state.root or prcp_path not in self._state.root:
            self._log.warning("no_met_data", source=met_source)
            return None

        # Load NDVI from all masks and combine
        ndvi_data = None
        for mask in masks:
            # Try fused NDVI first, then raw
            fused_path = f"derived/combined_ndvi/{mask}"
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
        ds = xr.Dataset({
            "ndvi": ndvi_data,
            "etf": etf_data,
            "eto": eto_data,
            "prcp": prcp_data,
        })

        return ds

    def _compute_k_parameters(
        self, ds: "xr.Dataset"
    ) -> Tuple["xr.DataArray", "xr.DataArray"]:
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

            # kc_max: 90th percentile of ALL ETf observations
            # (does NOT require NDVI to be present)
            all_etf_obs = site_etf.dropna().values
            if len(all_etf_obs) > 0:
                kc_max = float(np.percentile(all_etf_obs, 90))
            else:
                kc_max = 1.25

            results_ke[site_str] = ke_max
            results_kc[site_str] = kc_max

        # Convert back to xarray DataArrays
        sites = ds.coords["site"].values
        ke_values = [results_ke.get(str(s), 1.0) for s in sites]
        kc_values = [results_kc.get(str(s), 1.25) for s in sites]

        ke_da = xr.DataArray(ke_values, coords={"site": sites}, dims=["site"])
        kc_da = xr.DataArray(kc_values, coords={"site": sites}, dims=["site"])

        return ke_da, kc_da

    def _compute_groundwater_subsidy(
        self, ds: "xr.Dataset", irr_threshold: float
    ) -> Dict[str, Dict]:
        """
        Compute groundwater subsidy for each field and year.

        Subsidy is detected when ET > precipitation (ratio > 1).
        f_sub = (ratio - 1) / ratio

        Matches original SamplePlotDynamics._analyze_field_groundwater_subsidy().
        """
        eta = ds["etf"] * ds["eto"]
        ppt = ds["prcp"]
        etf = ds["etf"]

        results = {}
        all_years = sorted(pd.DatetimeIndex(ds.coords["time"].values).year.unique())
        time_index = pd.DatetimeIndex(ds.coords["time"].values)

        for site in ds.coords["site"].values:
            site_str = str(site)
            site_data = {}
            site_eta = eta.sel(site=site)
            site_ppt = ppt.sel(site=site)
            site_etf = etf.sel(site=site)

            # Track years with valid ETf data
            etf_years = []
            gw_count = 0

            for yr in all_years:
                yr_mask = time_index.year == yr

                # Check if this year has valid ETf data (sum > 0)
                etf_yr_sum = float(site_etf.isel(time=yr_mask).sum(skipna=True))
                if etf_yr_sum <= 0:
                    continue

                etf_years.append(int(yr))

                eta_yr = float(site_eta.isel(time=yr_mask).sum(skipna=True))
                ppt_yr = float(site_ppt.isel(time=yr_mask).sum(skipna=True))

                if ppt_yr <= 0:
                    continue

                ratio = eta_yr / (ppt_yr + 1.0)

                if ratio > 1:
                    subsidized = 1
                    f_sub = (ratio - 1) / ratio
                else:
                    subsidized = 0
                    f_sub = 0.0

                # Find months where eta > ppt (monthly aggregation)
                eta_monthly = site_eta.isel(time=yr_mask).resample(time="ME").sum()
                ppt_monthly = site_ppt.isel(time=yr_mask).resample(time="ME").sum()
                months = []
                for i, (e, p) in enumerate(
                    zip(eta_monthly.values, ppt_monthly.values)
                ):
                    if not np.isnan(e) and not np.isnan(p) and e > p:
                        months.append(i + 1)

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
                site_data = self._impute_missing_gwsub(
                    site_data, etf_years, missing_years
                )

            results[site_str] = site_data

        return results

    def _impute_missing_gwsub(
        self,
        site_data: Dict,
        etf_years: List[int],
        missing_years: List[int],
    ) -> Dict:
        """
        Impute groundwater subsidy for years without ETf data.

        If >50% of years with data show subsidy (f_sub > 0.1),
        fills missing years with mean values from subsidized years.
        Matches original SamplePlotDynamics behavior.
        """
        # Calculate means from years with data
        mean_sub = np.mean([
            site_data[y]["f_sub"] for y in etf_years
            if y in site_data and "f_sub" in site_data[y]
        ])

        # Collect all subsidy months across years
        all_months = []
        for y in etf_years:
            if y in site_data and "months" in site_data[y]:
                all_months.extend(site_data[y]["months"])
        unique_months = list(set(all_months))

        mean_ppt = np.mean([
            site_data[y]["ppt"] for y in etf_years
            if y in site_data and "ppt" in site_data[y]
        ])
        mean_eta = np.mean([
            site_data[y]["eta"] for y in etf_years
            if y in site_data and "eta" in site_data[y]
        ])

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
        ds: "xr.Dataset",
        irr_threshold: float,
        lookback: int,
        ndvi_threshold: float,
        min_pos_days: int,
        use_mask: bool,
        use_lulc: bool,
        irr_props: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Dict]:
        """
        Compute irrigation windows for each field and year.

        Two modes for determining irrigation status:

        use_mask=True (CONUS):
            Reads f_irr from irr_props (per-year irrigation fraction from properties).
            Year is irrigated if f_irr > irr_threshold.

        use_lulc=True (International):
            Computes irrigation from water balance (ET/PPT ratio).
            Year is irrigated if subsidy_months >= 3 AND field is cropped.

        Uses NDVI slope analysis to detect irrigation periods.
        Matches original SamplePlotDynamics behavior for exact parity.
        """
        ndvi = ds["ndvi"]
        etf = ds["etf"]
        eto = ds["eto"]
        ppt = ds["prcp"]

        results = {}
        years = sorted(pd.DatetimeIndex(ds.coords["time"].values).year.unique())
        time_index = pd.DatetimeIndex(ds.coords["time"].values)

        # Get LULC data for cropped classification if use_lulc is True
        lulc_by_site = self._get_lulc_by_site(ds.coords["site"].values) if use_lulc else {}

        # Track years needing backfill (irrigated but no windows detected)
        backfill_tracker = {}

        for site in ds.coords["site"].values:
            site_str = str(site)
            site_data = {}
            fallow_years = []
            years_needing_backfill = []

            # Check if field is cropped (LULC 12, 13, 14)
            cropped = lulc_by_site.get(site_str, 12) in [12, 13, 14] if use_lulc else True

            # Get per-year irrigation properties for this site if using mask mode
            site_irr_props = irr_props.get(site_str, {}) if irr_props else {}

            for yr in years:
                yr_mask = time_index.year == yr
                yr_str = str(yr)

                site_etf = etf.sel(site=site).isel(time=yr_mask)
                site_eto = eto.sel(site=site).isel(time=yr_mask)
                site_ppt = ppt.sel(site=site).isel(time=yr_mask)

                # Determine irrigation status based on mode
                if use_mask:
                    # CONUS mode: read f_irr from properties
                    f_irr = site_irr_props.get(yr_str, np.nan)
                    if pd.isna(f_irr):
                        f_irr = 0.0
                    irrigated = f_irr > irr_threshold

                elif use_lulc:
                    # International mode: compute from water balance
                    eta = site_etf * site_eto
                    eta_monthly = eta.resample(time="ME").sum()
                    ppt_monthly = site_ppt.resample(time="ME").sum()

                    subsidy_months = 0
                    for e, p in zip(eta_monthly.values, ppt_monthly.values):
                        if not np.isnan(e) and not np.isnan(p) and p > 0:
                            if e / (p + 1.0) > 1.3:
                                subsidy_months += 1

                    if subsidy_months >= 3 and cropped:
                        irrigated = True
                        f_irr = 1.0
                    else:
                        irrigated = False
                        f_irr = 0.0
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

                # Get NDVI with extended-year context for boundary handling
                ndvi_series = self._get_extended_year_ndvi(
                    ndvi, site, yr, years, time_index
                )

                # Detect irrigation windows from NDVI patterns
                irr_doys = self._detect_irrigation_windows(
                    ndvi_series, lookback, ndvi_threshold, min_pos_days, yr
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

    def _get_lulc_by_site(self, sites: np.ndarray) -> Dict[str, int]:
        """Get LULC code for each site from container properties."""
        lulc_path = "properties/land_cover/modis_lc"
        if lulc_path not in self._state.root:
            return {}

        lulc_arr = self._state.root[lulc_path]
        result = {}
        for site in sites:
            site_str = str(site)
            if site_str in self._state._uid_to_index:
                idx = self._state._uid_to_index[site_str]
                value = lulc_arr[idx]
                if not np.isnan(value):
                    result[site_str] = int(value)
        return result

    def _get_yearly_irrigation_properties(self) -> Dict[str, Dict[str, float]]:
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
                if hasattr(json_str, 'item'):
                    json_str = json_str.item()
                if json_str:
                    try:
                        result[site_str] = json.loads(json_str)
                    except json.JSONDecodeError:
                        result[site_str] = {}
        return result

    def _get_extended_year_ndvi(
        self,
        ndvi: "xr.DataArray",
        site: str,
        year: int,
        years: List[int],
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

    def _backfill_irrigation_windows(
        self,
        irr_data: Dict[str, Dict],
        backfill_tracker: Dict[str, List[int]],
    ) -> Dict[str, Dict]:
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
                int(y) for y, v in site_data.items()
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
    ) -> List[int]:
        """
        Detect irrigation windows from NDVI time series.

        Algorithm:
        1. Apply 32-day rolling mean to smooth NDVI
        2. Compute slope (diff)
        3. Find consecutive positive slope periods >= min_pos_days
        4. Extend windows by lookback and until NDVI drops below threshold
        """
        ydf = pd.DataFrame({"ndvi": ndvi_series})

        # Interpolate and smooth
        ydf["ndvi"] = ydf["ndvi"].interpolate()
        ydf["ndvi"] = ydf["ndvi"].bfill().ffill()
        ydf["ndvi"] = ydf["ndvi"].rolling(window=32, center=True).mean()
        ydf["ndvi"] = ydf["ndvi"].bfill().ffill()

        # Compute slope
        ydf["diff"] = ydf["ndvi"].diff()

        # Check data quality
        if ydf["ndvi"].isna().sum() > 200:
            return []

        # Find local minima
        local_min_indices = ydf[(ydf["diff"] > 0) & (ydf["diff"].shift(1) < 0)].index

        # Group consecutive positive slope days
        positive_slope = ydf["diff"] > 0
        groups = (positive_slope != positive_slope.shift()).cumsum()
        ydf["groups"] = groups

        # Find groups with >= min_pos_days
        group_counts = positive_slope.groupby(groups).sum()
        long_groups = group_counts[group_counts >= min_pos_days].index

        irr_doys = []
        for group in long_groups:
            group_mask = groups == group
            group_indices = positive_slope[group_mask].index

            if len(group_indices) == 0:
                continue

            start_index = group_indices[0]
            end_index = group_indices[-1]

            # Extend start by lookback if at local minimum
            if start_index in local_min_indices:
                start_day = start_index - pd.Timedelta(days=lookback)
            else:
                start_day = start_index

            # Extend end until NDVI drops below threshold
            end_day = end_index + pd.Timedelta(days=2)
            try:
                while end_day in ydf.index:
                    prev_ndvi = ydf.loc[end_day - pd.Timedelta(days=1), "ndvi"]
                    if prev_ndvi <= ndvi_threshold or pd.isna(prev_ndvi):
                        break
                    end_day += pd.Timedelta(days=1)
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
        ke_max: "xr.DataArray",
        kc_max: "xr.DataArray",
        irr_data: Dict[str, Dict],
        gwsub_data: Dict[str, Dict],
        fields: List[str],
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
