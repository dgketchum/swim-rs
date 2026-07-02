import json
import os
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

# pyemu is not yet compatible with pandas 3.0 Arrow-backed StringDtype;
# disable it so string columns stay as object dtype.
pd.options.future.infer_string = False

# Suppress pyemu's flopy warning - flopy is optional and not needed for SWIM-RS
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Failed to import legacy module")
    from pyemu import Matrix, ObservationEnsemble, Pst
    from pyemu.utils import PstFrom
    from pyemu.utils.os_utils import run_ossystem, run_sp

from swimrs.container.schema import SWE_PATHS, find_swe_path
from swimrs.process.input import build_swim_input


class PestBuilder:
    """Builder for PEST++ IES calibration control files.

    Constructs PEST++ control files, observation files, and parameter templates
    for calibrating SWIM-RS model parameters against ET fraction and SWE observations.

    The builder handles:
    - Parameter setup with prior information from soil and vegetation data
    - Observation file generation from remote sensing ET and SNODAS SWE
    - Localization matrix construction for ensemble methods
    - Forward run script generation

    Attributes:
        config: ProjectConfig instance with calibration settings.
        pest_run_dir: Root directory for PEST++ files.
        pest_dir: Directory containing the .pst control file.
        master_dir: Directory for PEST++ master process.
        pst_file: Path to the generated .pst control file.

    Example:
        >>> from swimrs.swim import ProjectConfig
        >>> from swimrs.calibrate import PestBuilder
        >>>
        >>> config = ProjectConfig()
        >>> config.read_config("project.toml", calibrate=True)
        >>>
        >>> with PestBuilder(config) as builder:
        ...     builder.spinup()
        ...     builder.build_pest(target_etf='ssebop')
        ...     builder.build_localizer()
        ...     builder.write_control_settings(noptmax=4, reals=250)
    """

    def __init__(
        self,
        config,
        container,
        use_existing: bool = False,
        python_script: str | None = None,
        prior_constraint: dict | None = None,
        conflicted_obs: str | None = None,
        verbose: bool = True,
        select_fields: list[str] | None = None,
    ) -> None:
        """Initialize PestBuilder for PEST++ calibration.

        Args:
            config: ProjectConfig instance
            container: SwimContainer instance or path to .swim directory.
                       Required - all data is sourced from the container.
            use_existing: If True, use existing PEST++ setup
            python_script: Path to custom forward run script
            prior_constraint: Prior constraint settings
            conflicted_obs: Path to conflicted observations file
            verbose: If False, suppress pyemu/PstFrom output. Default True.
            select_fields: Optional list of field UIDs to calibrate. If None, all fields
                from the container are used.
        """
        self.config = config
        self.project_ws = config.project_ws
        self.pest_run_dir = config.pest_run_dir

        # Initialize container (required)
        self._container = None
        self._container_path = None
        self._owns_container = False
        self._init_container(container)

        if not os.path.isdir(self.pest_run_dir):
            os.mkdir(self.pest_run_dir)

        # Extract data from container (replaces SamplePlots)
        self._load_data_from_container()

        # Filter to selected fields if requested
        if select_fields is not None:
            valid = [f for f in select_fields if f in self.plot_order]
            if not valid:
                raise ValueError(
                    f"None of select_fields {select_fields} found in container "
                    f"(available: {self.plot_order[:5]}...)"
                )
            self.plot_order = valid

        self.observation_index = {}

        mask_mode = getattr(self.config, "mask_mode", "irrigation")
        if mask_mode == "none":
            self.masks = ["no_mask"]
        else:
            self.masks = ["inv_irr", "irr"]

        self.etf_instrument = getattr(self.config, "etf_target_instrument", "landsat")

        self.pest = None
        self.etf_std = None
        self.etf_capture_indexes = []
        self._weight_audit_rows = []

        self.params_file = os.path.join(self.pest_run_dir, "params.csv")

        self.prior_contstraint = prior_constraint

        self.conflicted_obs = conflicted_obs

        self.pest_dir = os.path.join(config.pest_run_dir, "pest")
        self.master_dir = os.path.join(config.pest_run_dir, "master")

        self.workers_dir = os.path.join(config.pest_run_dir, "workers")
        self.obs_dir = getattr(config, "obs_folder", None) or os.path.join(
            config.pest_run_dir, "obs"
        )

        self.pst_file = os.path.join(self.pest_dir, f"{self.config.project_name}.pst")
        self.obs_idx_file = os.path.join(self.pest_dir, f"{self.config.project_name}.idx.csv")

        self.pest_args = self.get_pest_builder_args()

        self.verbose = verbose

        if python_script is None:
            python_script = os.path.join(os.path.dirname(__file__), "custom_forward_run.py")
            if verbose:
                print(f"Using default Python script at: {python_script}")

        self.python_script = python_script
        self.pest_args.update({"python_script": self.python_script})

        if use_existing:
            self.overwrite_build = False
        else:
            self.overwrite_build = True

    def _init_container(self, container) -> None:
        """Initialize container from instance or path."""
        from swimrs.container import SwimContainer

        if isinstance(container, str | Path):
            self._container_path = Path(container)
            self._container = SwimContainer.open(self._container_path, mode="r")
            self._owns_container = True
        else:
            self._container = container
            self._owns_container = False

    def _load_data_from_container(self) -> None:
        """Load all data from container (replaces SamplePlots).

        Populates:
        - self.plot_order: field UIDs
        - self.plot_properties: field properties dict
        - self.irr: irrigation data dict
        - self.ke_max: bare soil evaporation coefficient dict
        - self.kc_max: max crop coefficient dict
        - self.date_range: (start_date, end_date) tuple
        """
        if self._container is None:
            raise ValueError("Container not initialized")

        # Field order
        self.plot_order = self._container.field_uids

        # Get properties and dynamics from container's export infrastructure
        exporter = self._container.export
        self.plot_properties = exporter._get_properties_dict(self.plot_order)

        # Get dynamics (irr_data, gwsub_data, ke_max, kc_max)
        dynamics = exporter._get_dynamics_dict(self.plot_order)
        self.irr = dynamics.get("irr", {})
        self.ke_max = dynamics.get("ke_max", {})
        self.kc_max = dynamics.get("kc_max", {})
        self.gwsub_data = dynamics.get("gwsub", {})

        # Date range from container
        self.date_range = (self._container.start_date, self._container.end_date)

    def _get_etf_data(self, fid: str, model: str = "ssebop") -> pd.DataFrame:
        """
        Get ETf data for a field from container.

        Returns DataFrame with columns like '{model}_etf_{mask}' for each mask.

        If model='ensemble', computes the mean across all available ETf models.
        """
        if self._container is None:
            raise ValueError("No container available. Pass container to PestBuilder.__init__")

        result = pd.DataFrame(
            index=pd.date_range(self.config.start_dt, self.config.end_dt, freq="D")
        )

        if model == "ensemble":
            ensemble_source = getattr(self.config, "ensemble_source", "computed")
            if ensemble_source == "openet":
                # Use OpenET's pre-computed ensemble directly from the container
                for mask in self.masks:
                    path = f"remote_sensing/etf/{self.etf_instrument}/ensemble/{mask}"
                    if path in self._container.state.root:
                        df = self._container.query.dataframe(path, fields=[fid])
                        if fid in df.columns:
                            result[f"ensemble_etf_{mask}"] = df[fid]
            else:
                # Compute ensemble mean from the frozen config member list.
                # Use etf_ensemble_members from config, not auto-discovery,
                # so the target cannot drift with container contents.
                configured_members = getattr(self.config, "etf_ensemble_members", None)
                if configured_members:
                    available_models = [
                        m
                        for m in configured_members
                        if any(
                            f"remote_sensing/etf/{self.etf_instrument}/{m}/{mask}"
                            in self._container.state.root
                            for mask in self.masks
                        )
                    ]
                else:
                    available_models = self._discover_etf_models()
                if not available_models:
                    return result

                for mask in self.masks:
                    mask_data = []
                    for m in available_models:
                        path = f"remote_sensing/etf/{self.etf_instrument}/{m}/{mask}"
                        if path in self._container.state.root:
                            df = self._container.query.dataframe(path, fields=[fid])
                            if fid in df.columns:
                                mask_data.append(df[fid])

                    if mask_data:
                        combined = pd.concat(mask_data, axis=1)
                        result[f"ensemble_etf_{mask}"] = combined.mean(axis=1)
        else:
            for mask in self.masks:
                path = f"remote_sensing/etf/{self.etf_instrument}/{model}/{mask}"
                if path in self._container.state.root:
                    df = self._container.query.dataframe(path, fields=[fid])
                    if fid in df.columns:
                        result[f"{model}_etf_{mask}"] = df[fid]

        return result

    def _discover_etf_models(self) -> list[str]:
        """Discover available ETf models in the container."""
        known_models = ["ssebop", "ptjpl", "sims", "geesebal", "eemetric", "disalexi"]
        available = []

        for model in known_models:
            # Check if at least one mask exists for this model
            for mask in self.masks:
                path = f"remote_sensing/etf/{self.etf_instrument}/{model}/{mask}"
                if path in self._container.state.root:
                    available.append(model)
                    break

        return available

    def _get_swe_data(self, fid: str) -> pd.DataFrame:
        """Get SWE data for a field from container."""
        if self._container is None:
            raise ValueError("No container available. Pass container to PestBuilder.__init__")

        path = find_swe_path(self._container.state.root)
        if path is None:
            raise ValueError(f"SWE data not found in container (checked {', '.join(SWE_PATHS)})")

        df = self._container.query.dataframe(path, fields=[fid])
        # Slice to config date range to match the exported .np obs files
        start = getattr(self.config, "start_dt", None)
        end = getattr(self.config, "end_dt", None)
        if start is not None and end is not None:
            date_range = pd.date_range(start, end, freq="D")
            df = df.reindex(date_range)
        result = pd.DataFrame(index=df.index)
        result["swe"] = df[fid] if fid in df.columns else np.nan
        return result

    def close(self) -> None:
        """Close container if we own it."""
        if self._owns_container and self._container is not None:
            self._container.close()
            self._container = None

    def __enter__(self) -> "PestBuilder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def get_pest_builder_args(self) -> dict:
        targets = self.plot_order

        # Some projects (international) may not have SSURGO; allow missing AWC
        aw = [self.plot_properties.get(t, {}).get("awc", np.nan) for t in targets]
        ke_max = [self.ke_max.get(t, 1.0) for t in targets]

        et_ins = [f"etf_{fid}.ins" for fid in targets]
        swe_ins = [f"swe_{fid}.ins" for fid in targets]

        init_pars = self.initial_parameter_dict()
        p_list = list(init_pars.keys())
        pars = OrderedDict(
            {f"{k}_{fid}": v.copy() for k, v in init_pars.items() for fid in targets}
        )

        params = []

        # Prior information from pre-processing
        for i, fid in enumerate(targets):
            for p in p_list:
                k = f"{p}_{fid}"

                if "aw_" in k:
                    aw_ = aw[i] * 1000.0
                    if np.isnan(aw_) or aw_ < pars[k]["lower_bound"]:
                        aw_ = 150.0

                    if aw_ > pars[k]["upper_bound"]:
                        aw_ = pars[k]["upper_bound"] * 0.8

                    params.append((k, aw_, f"p_{k}_0_constant.csv"))

                elif "ke_max_" in k:
                    ke_max_ = ke_max[i]
                    params.append((k, ke_max_, f"p_{k}_0_constant.csv"))

                elif "mad_" in k:
                    # Prefer properties-based irrigation fraction when present, otherwise use inferred dynamics.
                    try:
                        irr = np.nanmean(
                            [self.plot_properties[fid]["irr"][str(yr)] for yr in range(1987, 2023)]
                        )
                    except Exception:
                        irr_data = self.irr.get(fid, {})
                        irr_vals = []
                        for yy, vv in irr_data.items():
                            if yy == "fallow_years":
                                continue
                            try:
                                irr_vals.append(float(vv.get("f_irr", np.nan)))
                            except Exception:
                                continue
                        irr = float(np.nanmean(irr_vals)) if irr_vals else 0.0
                    # Irrigation-dependent initial value AND bounds:
                    # - Irrigated: low MAD (0.10), trigger irrigation early, bounds [0.10, 0.3]
                    # - Non-irrigated: high MAD (0.5), tolerate depletion, bounds [0.3, 0.8]
                    if irr > 0.2:
                        params.append((k, 0.10, f"p_{k}_0_constant.csv"))
                        pars[k]["lower_bound"] = 0.10
                        pars[k]["upper_bound"] = 0.3
                    else:
                        params.append((k, 0.5, f"p_{k}_0_constant.csv"))
                        pars[k]["lower_bound"] = 0.3
                        pars[k]["upper_bound"] = 0.8

                elif "ndvi_0_" in k:
                    # Informed prior based on irrigation status (see PARAMETER_SEARCH.md):
                    # With fc in kc_act equation (kc_act = fc*ks*kcb + ke):
                    #   - Grassland/non-irrigated: ndvi_0 ~ 0.20 (transpiration begins at low NDVI)
                    #   - Irrigated crops: ndvi_0 ~ 0.55 (optimal region 0.50-0.60)
                    try:
                        irr = np.nanmean(
                            [self.plot_properties[fid]["irr"][str(yr)] for yr in range(1987, 2023)]
                        )
                    except Exception:
                        irr_data = self.irr.get(fid, {})
                        irr_vals = []
                        for yy, vv in irr_data.items():
                            if yy == "fallow_years":
                                continue
                            try:
                                irr_vals.append(float(vv.get("f_irr", np.nan)))
                            except Exception:
                                continue
                        irr = float(np.nanmean(irr_vals)) if irr_vals else 0.0
                    if irr > 0.2:
                        params.append((k, 0.55, f"p_{k}_0_constant.csv"))
                    else:
                        params.append((k, 0.20, f"p_{k}_0_constant.csv"))

                else:
                    params.append((k, pars[k]["initial_value"], f"p_{k}_0_constant.csv"))

        idx, vals, _names = [x[0] for x in params], [x[1] for x in params], [x[2] for x in params]
        vals = np.array([vals, _names]).T
        df = pd.DataFrame(index=idx, data=vals, columns=["value", "mult_name"])
        df.to_csv(self.params_file)

        for e, (ii, r) in enumerate(df.iterrows()):
            pars[ii]["use_rows"] = e
            if any(prefix in ii for prefix in ["aw_", "ke_max_", "mad_", "ndvi_0_"]):
                val = float(r["value"])
                pars[ii]["initial_value"] = val

                if "ke_max_" in ii:
                    # ke_max is a prior (not calibrated) — collapse bounds
                    if val < pars[ii]["lower_bound"]:
                        pars[ii]["lower_bound"] = val - 0.2
                        pars[ii]["initial_value"] = val - 0.1
                        pars[ii]["upper_bound"] = val
                    if val > pars[ii]["upper_bound"]:
                        pars[ii]["lower_bound"] = val - 0.3
                        pars[ii]["initial_value"] = val - 0.1
                        pars[ii]["upper_bound"] = val

        etf_obs_files = [f"obs/obs_etf_{fid}.np" for fid in targets]
        swe_obs_files = [f"obs/obs_swe_{fid}.np" for fid in targets]

        dct = {
            "targets": targets,
            "etf_obs": {"file": etf_obs_files, "insfile": et_ins},
            "swe_obs": {"file": swe_obs_files, "insfile": swe_ins},
            "pars": pars,
        }

        return dct

    def build_pest(self, target_etf: str = "openet", members: list[str] | None = None) -> None:
        """Build the PEST++ control file and supporting files.

        Creates the .pst control file, observation files, parameter templates,
        and forward run script in the pest directory.

        Uses the process package with portable swim_input.h5 file. Workers are
        fully self-contained and can run without shared storage.

        Args:
            target_etf: ET model to use as calibration target ('ssebop', 'ptjpl', etc.).
            members: Optional list of ensemble member models for uncertainty weighting.
                If provided, observation weights are computed from inter-model spread.

        Raises:
            NotImplementedError: If use_existing=True was set in constructor.
        """
        if self.overwrite_build is False:
            raise NotImplementedError(
                "Use of exising Pest++ project was specified, "
                'running "build_pest" will overwrite it.'
            )

        # Observation files are the source of obsval in the .pst (via PstFrom).
        # Ensure they reflect the requested ETf target model (and SWE).
        self._export_observations(etf_model=target_etf)

        # Create minimal template directory for PstFrom
        # (Avoids copying workers/master/pest dirs which causes recursive copying)
        import shutil

        template_dir = os.path.join(self.pest_run_dir, "_template")
        if os.path.exists(template_dir):
            shutil.rmtree(template_dir)
        os.makedirs(template_dir)

        # Copy only the files PstFrom needs (params.csv, obs files)
        # Update self.params_file to point to template location for PstFrom
        if os.path.exists(self.params_file):
            shutil.copy2(self.params_file, template_dir)
            self.params_file = os.path.join(template_dir, "params.csv")
            # Update parameter dicts to use the new path
            for k, v in self.pest_args["pars"].items():
                if "file" in v:
                    v["file"] = self.params_file

        if os.path.exists(self.obs_dir):
            shutil.copytree(self.obs_dir, os.path.join(template_dir, "obs"))

        self.pest = PstFrom(template_dir, self.pest_dir, remove_existing=True, echo=self.verbose)

        self._write_params()

        i = self._write_etf_obs(target_etf, members)
        count = i + 1
        self._write_swe_obs(count)

        ofiles = [str(x).replace("obs", "pred") for x in self.pest.output_filenames]
        self.pest.output_filenames = ofiles

        os.makedirs(os.path.join(self.pest_dir, "pred"))

        self.pest.py_run_file = "custom_forward_run.py"
        self.pest.mod_command = "python custom_forward_run.py"

        self.pest.build_pst(filename=self.pst_file, version=2)

        # Build portable input file and generate forward run script
        self._build_swim_input()
        self._write_forward_run_script()

        self._finalize_obs()
        if self.verbose:
            self.print_build_diagnostics()

        # Clean up template directory
        if os.path.exists(template_dir):
            shutil.rmtree(template_dir)

        if self.verbose:
            print("Configured PEST++ for {} targets, ".format(len(self.pest_args["targets"])))

    def export_weight_audit(self, output_path: str) -> None:
        """Write per-observation weight audit CSV for ablation diagnostics.

        Must be called after build_pest(). Contains one row per ETf capture
        date per site with obsval, member spread, eligibility, and final weight.
        """
        if not self._weight_audit_rows:
            return
        df = pd.DataFrame(self._weight_audit_rows)
        col_order = [
            "fid",
            "date",
            "obsval",
            "member_count",
            "member_mean",
            "member_std",
            "weight_mode",
            "weight_pre_pdc",
            "weight_final",
            "eligible",
        ]
        df = df[[c for c in col_order if c in df.columns]]
        df.to_csv(output_path, index=False)
        if self.verbose:
            print(f"Weight audit: {len(df)} rows -> {output_path}")

    def _export_observations(self, etf_model: str) -> None:
        """Export ETf/SWE observation arrays for PstFrom from the SwimContainer.

        PstFrom sets each observation's ``obsval`` from the numpy files in ``obs_dir``.
        If these files contain model output (or are missing), calibration will not
        target satellite ETf/SWE as intended.
        """
        if self._container is None:
            raise ValueError("Container not initialized")

        obs_dir = Path(self.obs_dir)
        obs_dir.mkdir(parents=True, exist_ok=True)

        # Prefer config irrigation threshold; fall back to typical 0.3.
        irr_threshold = getattr(self.config, "irr_threshold", None)
        if irr_threshold is None:
            irr_threshold = getattr(self.config, "irrigation_threshold", 0.3)

        # Limit export to the current target set (supports debug_fields slicing).
        fields = list(self.pest_args.get("targets", self.plot_order))

        masks = tuple(self.masks)

        start_date = (
            self.config.start_dt.date().isoformat()
            if getattr(self.config, "start_dt", None)
            else None
        )
        end_date = (
            self.config.end_dt.date().isoformat() if getattr(self.config, "end_dt", None) else None
        )

        ensemble_source = getattr(self.config, "ensemble_source", "computed")
        ensemble_members = getattr(self.config, "etf_ensemble_members", None)

        try:
            self._container.export.observations(
                output_dir=obs_dir,
                etf_model=etf_model,
                etf_instrument=self.etf_instrument,
                masks=masks,
                irr_threshold=float(irr_threshold),
                fields=fields,
                start_date=start_date,
                end_date=end_date,
                ensemble_source=ensemble_source,
                ensemble_members=ensemble_members,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to export observations to {obs_dir}: {e}") from e

    def print_build_diagnostics(self, max_groups: int = 25) -> pd.DataFrame:
        """Print a compact diagnostics table after building the PEST++ project.

        This is meant to make it obvious whether calibration is actually using
        the intended observations/weights (e.g., ETf weights not all zero).

        Returns
        -------
        pd.DataFrame
            Per-observation-group summary table (also printed).
        """
        try:
            pst = Pst(self.pst_file)
            obs = pst.observation_data.copy()
        except Exception as e:
            print(f"[PEST++ diagnostics] Failed to load pst/observation data: {e}")
            return pd.DataFrame()

        if obs is None or obs.empty:
            print("[PEST++ diagnostics] No observations found in pst.")
            return pd.DataFrame()

        w = (
            obs["weight"].fillna(0.0).astype(float)
            if "weight" in obs.columns
            else pd.Series(0.0, index=obs.index)
        )
        y = (
            obs["obsval"].astype(float)
            if "obsval" in obs.columns
            else pd.Series(np.nan, index=obs.index)
        )
        valid_obs = np.isfinite(y.values) & (y.values != -99.0)
        nonzero_w = w.values > 0.0

        # Overall header
        print("\n=== PEST++ Build Diagnostics ===")
        print(f"pst: {self.pst_file}")
        print(
            "observations: "
            f"total={len(obs)}, valid={int(valid_obs.sum())}, "
            f"nonzero_weight={int(nonzero_w.sum())}"
        )

        # Type-specific quick checks
        etf_mask = obs.index.to_series().str.contains("etf", case=False, regex=False)
        swe_mask = obs.index.to_series().str.contains("swe", case=False, regex=False)
        if etf_mask.any():
            etf_nonzero = int((nonzero_w & etf_mask.values).sum())
            etf_valid = int((valid_obs & etf_mask.values).sum())
            print(f"ETf: valid={etf_valid}, nonzero_weight={etf_nonzero}")

            # Per-site ETf weight share
            etf_obs = obs[etf_mask].copy()
            etf_w = w[etf_mask]
            total_etf_w = etf_w.sum()
            if total_etf_w > 0:
                # Extract site ID from obs name: oname:obs_etf_{fid}_otype:...
                site_ids = etf_obs.index.to_series().str.extract(
                    r"oname:obs_etf_(.+?)_otype:", expand=False
                )
                site_w = etf_w.groupby(site_ids).sum()
                site_share = (site_w / total_etf_w * 100).sort_values(ascending=False)
                site_count = (etf_w > 0).groupby(site_ids).sum()
                site_total = site_ids.groupby(site_ids).count()

                print("\nETf per-site weight share (top 10):")
                for sid in site_share.head(10).index:
                    pct = site_share[sid]
                    nz = int(site_count.get(sid, 0))
                    tot = int(site_total.get(sid, 0))
                    print(f"  {sid:30s}  {pct:5.1f}%  ({nz}/{tot} obs)")

                # Warn about sites with heavy pruning
                pruned = site_count[site_count < site_total * 0.3]
                if len(pruned) > 0:
                    print(f"\nWARNING: {len(pruned)} sites have >70% ETf obs zeroed:")
                    for sid in pruned.index:
                        nz = int(pruned[sid])
                        tot = int(site_total[sid])
                        print(f"  {sid}: {nz}/{tot} remaining ({100 * nz / tot:.0f}%)")

        if swe_mask.any():
            swe_nonzero = int((nonzero_w & swe_mask.values).sum())
            swe_valid = int((valid_obs & swe_mask.values).sum())
            print(f"SWE: valid={swe_valid}, nonzero_weight={swe_nonzero}")

        table = self._build_obs_diagnostics_table(obs)
        # Limit printed rows for readability
        show = table.head(max_groups).copy()
        if len(table) > max_groups:
            more = len(table) - max_groups
            print(
                f"\nTop {max_groups} observation groups (of {len(table)}). ({more} more not shown)"
            )
        else:
            print(f"\nObservation groups: {len(table)}")

        # Make output stable/compact
        pd.set_option("display.max_colwidth", 90)
        print(show.to_string(index=False))

        # Parameter quick summary (helps confirm tuned params exist)
        try:
            par = pst.parameter_data.copy()
            if par is not None and not par.empty:
                at_lower = (
                    par["parval1"].astype(float) <= par["parlbnd"].astype(float) + 1e-12
                ).sum()
                at_upper = (
                    par["parval1"].astype(float) >= par["parubnd"].astype(float) - 1e-12
                ).sum()
                print(
                    "\nparameters: "
                    f"n={len(par)}, at_lower={int(at_lower)}, at_upper={int(at_upper)}, "
                    f"groups={par['pargp'].nunique() if 'pargp' in par.columns else 'n/a'}"
                )
        except Exception:
            pass

        return table

    @staticmethod
    def _build_obs_diagnostics_table(obs: pd.DataFrame) -> pd.DataFrame:
        """Build per-observation-group diagnostics for a PEST++ observation table."""
        if obs is None or obs.empty:
            return pd.DataFrame()

        grp = obs["obgnme"] if "obgnme" in obs.columns else pd.Series("obs", index=obs.index)

        rows: list[dict] = []
        for gname, gdf in obs.groupby(grp, dropna=False):
            gw = (
                gdf["weight"].fillna(0.0).astype(float)
                if "weight" in gdf.columns
                else pd.Series(0.0, index=gdf.index)
            )
            gy = (
                gdf["obsval"].astype(float)
                if "obsval" in gdf.columns
                else pd.Series(np.nan, index=gdf.index)
            )
            gsd = (
                gdf["standard_deviation"].astype(float)
                if "standard_deviation" in gdf.columns
                else pd.Series(np.nan, index=gdf.index)
            )
            gvalid = np.isfinite(gy.values) & (gy.values != -99.0)
            gnonzero = gw.values > 0.0
            rows.append(
                {
                    "group": str(gname),
                    "n": int(len(gdf)),
                    "valid": int(gvalid.sum()),
                    "w>0": int(gnonzero.sum()),
                    "w_sum": float(gw.sum()),
                    "w_max": float(gw.max()) if len(gw) else 0.0,
                    "obs_min": float(np.nanmin(gy.values[gvalid])) if gvalid.any() else np.nan,
                    "obs_max": float(np.nanmax(gy.values[gvalid])) if gvalid.any() else np.nan,
                    "sd_nan%": float(np.mean(~np.isfinite(gsd.values)) * 100.0)
                    if len(gsd)
                    else 0.0,
                }
            )

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values(by=["w_sum", "valid", "n"], ascending=False)

    def _write_forward_run_script(self) -> None:
        """Generate custom_forward_run.py with portable relative paths.

        Uses the process package with swim_input.h5 for fully portable workers.
        All paths are relative to the worker directory - no shared storage needed.
        """
        script_path = os.path.join(self.pest_dir, "custom_forward_run.py")

        script_content = '''"""Auto-generated forward run script for PEST++ calibration.

Uses the swimrs.process package with portable swim_input.h5 file.
All paths are relative - workers are fully self-contained.
"""
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    """Forward runner for PEST++ workers."""
    start_time = time.time()

    from swimrs.process.input import SwimInput
    from swimrs.process.loop_fast import run_daily_loop_fast
    from swimrs.process.state import (
        CalibrationParameters,
        load_pest_mult_properties,
    )

    cwd = os.getcwd()

    # All paths relative to worker directory
    h5_path = os.path.join(cwd, "swim_input.h5")
    mult_dir = os.path.join(cwd, "mult")
    pred_dir = os.path.join(cwd, "pred")

    os.makedirs(pred_dir, exist_ok=True)

    # Load portable input data
    swim_input = SwimInput(h5_path=h5_path)

    # Update parameters and properties from PEST++ multiplier files
    params = CalibrationParameters.from_pest_mult(
        mult_dir=mult_dir,
        fids=swim_input.fids,
        base=swim_input.parameters,
    )
    props = load_pest_mult_properties(
        mult_dir=mult_dir,
        fids=swim_input.fids,
        base_props=swim_input.properties,
    )

    # Run the model (uses fast JIT-compiled loop)
    output, _ = run_daily_loop_fast(
        swim_input=swim_input,
        properties=props,
        parameters=params,
    )

    # Write predictions (ETf and SWE)
    # Replace NaN with 0.0 so PEST++ instruction files can parse the output.
    for i, fid in enumerate(swim_input.fids):
        etf_path = os.path.join(pred_dir, f"pred_etf_{fid}.np")
        swe_path = os.path.join(pred_dir, f"pred_swe_{fid}.np")
        np.savetxt(etf_path, np.nan_to_num(output.etf[:, i], nan=0.0))
        np.savetxt(swe_path, np.nan_to_num(output.swe[:, i], nan=0.0))

    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    run()
'''

        with open(script_path, "w") as f:
            f.write(script_content)

    def build_localizer(self) -> None:
        """Build the localization matrix for ensemble Kalman methods.

        Creates a sparse matrix that restricts parameter-observation correlations
        to physically meaningful relationships. ET observations only update
        ET-related parameters, SWE observations only update snow parameters.

        Writes loc.mat and localizer_summary.json to the pest directory.
        """
        et_params = ["aw", "ndvi_k", "ndvi_0", "mad", "kr_alpha", "ks_alpha"]
        snow_params = ["swe_alpha", "swe_beta"]

        par_relation = {"etf": et_params, "swe": snow_params}

        pst = Pst(self.pst_file)

        pdict = {}
        for i, r in pst.parameter_data.iterrows():
            if r["pargp"] not in pdict.keys():
                pdict[r["pargp"]] = [r["parnme"]]
            else:
                pdict[r["pargp"]].append(r["parnme"])

        pnames = pst.parameter_data["parnme"].values

        # Localizer matrix covers non-zero-weight observations only.
        # PI equations are NOT included — PEST++ IES handles prior information
        # separately from the localizer. Including PI rows causes PEST++ to
        # reject the localizer with "rows not found in observation names".
        df = Matrix.from_names(pst.nnz_obs_names, pnames).to_dataframe()

        localizer = df.copy()

        # Parse site IDs from ETf/SWE observation names (skip PI rows)
        # TODO: replace with explicit observation name parsers
        sites = list(
            set(["_".join(i.split("_")[2:-3]) for i in df.index if not i.startswith("pi_")])
        )

        track = {k: [] for k in sites}

        # Date range from config — observations are indexed relative to config start/end,
        # not the full container range. Using container dates here causes all-zero localizer
        # when the config date range is a subset of the container (e.g. 2018-2025 vs 1987-2025).
        dt = list(pd.date_range(self.config.start_dt, self.config.end_dt, freq="D"))
        years = list(range(self.config.start_dt.year, self.config.end_dt.year + 1))

        for s in sites:
            for ob_type, params in par_relation.items():
                if ob_type == "etf":
                    for yr in years:
                        t_idx = [f"_i:{int(i)}_" for i, r in enumerate(dt) if r.year == yr]

                        idx = [i for i in df.index if f"{ob_type}_{s}" in i]
                        idx = [i for i in idx if "_{}_".format(i.split("_")[-2]) in t_idx]
                        cols = list(
                            np.array(
                                [[c for c in df.columns if f"{p}_{s}" in c] for p in et_params]
                            ).flatten()
                        )
                        localizer.loc[idx, cols] = 1.0

                else:
                    idx = [i for i in df.index if f"{ob_type}_{s}" in i]
                    cols = list(
                        np.array(
                            [[c for c in df.columns if f"{p}_{s}" in c] for p in params]
                        ).flatten()
                    )
                    localizer.loc[idx, cols] = 1.0

        vals = localizer.values.copy()
        vals[np.isnan(vals)] = 0.0
        vals[vals < 1.0] = 0.0
        localizer.loc[localizer.index, localizer.columns] = vals
        mat_file = os.path.join(os.path.dirname(self.pst_file), "loc.mat")

        col_sums = {col: int(localizer[col].sum()) for col in localizer.columns}
        summary = {
            "shape": localizer.shape,
            "non_zero_count": int(np.count_nonzero(localizer.values)),
            "sites": sites,
            "tracked_irrigation_years": track,
            "parameter_groups": list(pdict.keys()),
            "column_sums": col_sums,
            "pi_note": "PI equations not in localizer — PEST++ IES handles them via ies_reg_factor",
        }
        summary_file = os.path.join(os.path.dirname(self.pst_file), "localizer_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)

        Matrix.from_dataframe(localizer).to_ascii(mat_file)

        pst.write(self.pst_file, version=2)

    def write_control_settings(
        self, noptmax: int = -2, reals: int = 250, ies_num_threads: int | None = None
    ) -> None:
        """Write PEST++ IES control settings to the .pst file.

        Args:
            noptmax: Maximum optimization iterations. Use -2 for parameter
                estimation mode, positive values for optimization.
            reals: Number of realizations in the ensemble.
            ies_num_threads: Number of threads for PEST++ IES ensemble upgrade.
                If None, PEST++ uses its default (single-threaded upgrade).
        """
        pst = Pst(self.pst_file)
        pst.pestpp_options["ies_localizer"] = "loc.mat"
        pst.pestpp_options["ies_num_reals"] = reals
        pst.pestpp_options["ies_drop_conflicts"] = "true"

        # Enable IES regularization if PI equations are present.
        # ies_reg_factor controls prior-to-measurement balance in the
        # ensemble update — higher values pull parameters toward the
        # initial ensemble (which was seeded from LULC priors).
        if pst.nprior > 0:
            reg_factor = getattr(self.config, "prior_regularization_fraction", 0.2)
            pst.pestpp_options["ies_reg_factor"] = reg_factor

        # Tier 1 performance options
        pst.pestpp_options["num_tpl_ins_threads"] = 10
        pst.pestpp_options["overdue_giveup_fac"] = 3.0
        pst.pestpp_options["ies_verbose_level"] = 0

        if ies_num_threads is not None:
            pst.pestpp_options["ies_num_threads"] = ies_num_threads

        pst.control_data.noptmax = noptmax
        oe = ObservationEnsemble.from_gaussian_draw(pst=pst, num_reals=reals)
        oe.to_csv(self.pst_file.replace(".pst", ".oe.csv"))
        pst.write(self.pst_file, version=2)
        if self.verbose:
            print(f"writing {self.pst_file} with noptmax={noptmax}, {reals} realizations")

    def initial_parameter_dict(self) -> OrderedDict:
        p = OrderedDict(
            {
                # 'aw' and 'zr' are applied by Tracker.load_soils and load_root_depth
                "aw": {
                    "file": self.params_file,
                    "std": 100.0,
                    "initial_value": None,
                    "lower_bound": 100.0,
                    "upper_bound": 400.0,
                    "pargp": "aw",
                    "index_cols": 0,
                    "use_cols": 1,
                    "use_rows": None,
                },
                # Stress coefficients - centered in bounds
                "ks_alpha": {
                    "file": self.params_file,
                    "std": 0.15,
                    "initial_value": 0.5,
                    "lower_bound": 0.01,
                    "upper_bound": 1.0,
                    "pargp": "ks_alpha",
                    "index_cols": 0,
                    "use_cols": 1,
                    "use_rows": None,
                },
                "kr_alpha": {
                    "file": self.params_file,
                    "std": 0.15,
                    "initial_value": 0.5,
                    "lower_bound": 0.01,
                    "upper_bound": 1.0,
                    "pargp": "kr_alpha",
                    "index_cols": 0,
                    "use_cols": 1,
                    "use_rows": None,
                },
                # NDVI-Kcb relationship: sigmoid function
                # kcb = kc_max / (1 + exp(-k * (NDVI - ndvi_0)))
                # kc_act = fc * ks * kcb + ke (FAO-56 dual crop coefficient)
                "ndvi_k": {
                    "file": self.params_file,
                    "std": 1.0,
                    "initial_value": 10.0,
                    "lower_bound": 3.0,
                    "upper_bound": 20.0,
                    "pargp": "ndvi_k",
                    "index_cols": 0,
                    "use_cols": 1,
                    "use_rows": None,
                },
                "ndvi_0": {
                    "file": self.params_file,
                    "std": 0.15,
                    "initial_value": 0.55,
                    "lower_bound": 0.1,
                    "upper_bound": 0.80,
                    "pargp": "ndvi_0",
                    "index_cols": 0,
                    "use_cols": 1,
                    "use_rows": None,
                },
                # Management allowed depletion
                "mad": {
                    "file": self.params_file,
                    "std": 0.15,
                    "initial_value": None,
                    "lower_bound": 0.10,
                    "upper_bound": 0.9,
                    "pargp": "mad",
                    "index_cols": 0,
                    "use_cols": 1,
                    "use_rows": None,
                },
                # Snow melt parameters
                "swe_alpha": {
                    "file": self.params_file,
                    "std": 0.2,
                    "initial_value": 0.3,
                    "lower_bound": -0.5,
                    "upper_bound": 1.0,
                    "pargp": "swe_alpha",
                    "index_cols": 0,
                    "use_cols": 1,
                    "use_rows": None,
                },
                "swe_beta": {
                    "file": self.params_file,
                    "std": 0.3,
                    "initial_value": 1.5,
                    "lower_bound": 0.5,
                    "upper_bound": 2.5,
                    "pargp": "swe_beta",
                    "index_cols": 0,
                    "use_cols": 1,
                    "use_rows": None,
                },
            }
        )

        return p

    def dry_run(self, exe: str = "pestpp-ies") -> None:
        cmd = " ".join([exe, os.path.join(self.pest_dir, self.pst_file)])
        wd = self.pest_dir
        try:
            run_sp(cmd, wd, verbose=False)
        except Exception:
            run_ossystem(cmd, wd, verbose=False)

    def spinup(self, overwrite: bool = False) -> None:
        """Run model spinup to initialize state variables.

        Runs the model with initial parameters and saves the final state
        to the spinup JSON file for warm-starting calibration runs.

        This method also creates the initial swim_input.h5 file (without spinup
        state). After spinup completes, _build_swim_input() rebuilds the h5
        with the spinup state baked in.

        Args:
            overwrite: If True, regenerate spinup even if file exists.
        """
        from swimrs.process.loop_fast import run_daily_loop_fast

        if not os.path.exists(self.config.spinup) or overwrite:
            if self.verbose:
                print("RUNNING SPINUP")

            if overwrite:
                try:
                    os.remove(self.config.spinup)
                except FileNotFoundError:
                    pass

            # Build swim_input.h5 for spinup (no existing spinup state)
            # This file will be rebuilt with spinup state by _build_swim_input()
            os.makedirs(self.pest_dir, exist_ok=True)
            h5_path = os.path.join(self.pest_dir, "swim_input.h5")
            swim_input = build_swim_input(
                container=self._container,
                output_h5=h5_path,
                start_date=self.config.start_dt,
                end_date=self.config.end_dt,
                refet_type=getattr(self.config, "refet_type", "eto") or "eto",
                etf_model=getattr(self.config, "etf_target_model", "ssebop"),
                met_source=getattr(self.config, "met_source", "gridmet"),
                empirical_kc_max=True,
                mask_mode=getattr(self.config, "mask_mode", "irrigation"),
                fields=self.plot_order,
                transpiration_cover_scaling=getattr(
                    self.config, "transpiration_cover_scaling", True
                ),
                # Fresh calibration: a stale calibration/ group (e.g. from a
                # copied container) must not contaminate the PEST base run.
                use_container_calibration=False,
            )

            # Run simulation to generate spinup state (uses fast JIT loop)
            output, final_state = run_daily_loop_fast(swim_input)
            swim_input.close()

            # Save final state as spinup JSON
            spn_dct = {}
            for i, fid in enumerate(swim_input.fids):
                spn_dct[fid] = {
                    "depl_root": float(final_state.depl_root[i]),
                    "swe": float(final_state.swe[i]),
                    "kr": float(final_state.kr[i]),
                    "ks": float(final_state.ks[i]),
                    "zr": float(final_state.zr[i]),
                }
                # Add optional state if available
                if final_state.depl_ze is not None:
                    spn_dct[fid]["depl_ze"] = float(final_state.depl_ze[i])
                if final_state.s is not None:
                    spn_dct[fid]["s"] = float(final_state.s[i])
                    spn_dct[fid]["s1"] = float(final_state.s1[i])
                    spn_dct[fid]["s2"] = float(final_state.s2[i])
                    spn_dct[fid]["s3"] = float(final_state.s3[i])
                    spn_dct[fid]["s4"] = float(final_state.s4[i])

            # Guard: reject spinup if any field has NaN state
            nan_fields = [
                fid
                for fid, vals in spn_dct.items()
                if any(np.isnan(v) for v in vals.values() if isinstance(v, float))
            ]
            if nan_fields:
                raise ValueError(
                    f"Spinup produced NaN state for {len(nan_fields)} field(s): "
                    f"{nan_fields[:5]}... "
                    "This usually means forcing data has gaps at the end of the "
                    "simulation period. Check that met data covers the full "
                    f"date range ({self.config.start_dt} to {self.config.end_dt})."
                )

            with open(self.config.spinup, "w") as f:
                json.dump(spn_dct, f, indent=2)

            if self.verbose:
                print(f"Spinup saved to {self.config.spinup}")

        else:
            if self.verbose:
                print("SPINUP exists, skipping")

    def _build_swim_input(self) -> str:
        """Build portable swim_input.h5 file for workers with spinup state.

        Creates a self-contained HDF5 file with all input data needed
        for model execution, including spinup state if available. This file
        is copied to each PEST++ worker for isolated execution.

        If spinup() was called first, this rebuilds the h5 with spinup state
        baked in. The rebuild is necessary because spinup creates the h5
        without spinup state (since it's generating it).

        Returns:
            str: Path to the created swim_input.h5 file.
        """
        h5_path = os.path.join(self.pest_dir, "swim_input.h5")

        # Get spinup path if available
        spinup_path = None
        if hasattr(self.config, "spinup") and self.config.spinup:
            if os.path.exists(self.config.spinup):
                spinup_path = self.config.spinup

        # If h5 exists but no spinup, keep as-is (no spinup available)
        # If h5 exists and spinup exists, rebuild to bake in spinup state
        if os.path.exists(h5_path) and spinup_path is None:
            if self.verbose:
                print(f"swim_input.h5 exists at {h5_path} (no spinup), skipping")
            return h5_path

        if self.verbose:
            print("Building portable swim_input.h5 with spinup state...")

        # Build the HDF5 file from container
        build_swim_input(
            container=self._container,
            output_h5=h5_path,
            spinup_json_path=spinup_path,
            start_date=self.config.start_dt,
            end_date=self.config.end_dt,
            refet_type=getattr(self.config, "refet_type", "eto") or "eto",
            etf_model=getattr(self.config, "etf_target_model", "ssebop"),
            met_source=getattr(self.config, "met_source", "gridmet"),
            empirical_kc_max=True,
            mask_mode=getattr(self.config, "mask_mode", "irrigation"),
            max_irr_rate=getattr(self.config, "max_irr_rate", 100.0) or 100.0,
            fields=self.plot_order,
            transpiration_cover_scaling=getattr(self.config, "transpiration_cover_scaling", True),
            # Fresh calibration: a stale calibration/ group (e.g. from a
            # copied container) must not contaminate the PEST base run.
            use_container_calibration=False,
        )

        if self.verbose:
            print(f"Created swim_input.h5 at {h5_path}")
        return h5_path

    def _write_params(self) -> None:
        _file = None

        for k, v in self.pest_args["pars"].items():
            # pop out unneeded 'std' keyword
            _ = v.pop("std")
            if "file" in v.keys():
                _file = v.pop("file")
            if v["lower_bound"] <= 0.0:
                transform = "none"
            else:
                transform = "log"
            self.pest.add_parameters(
                _file, "constant", transform=transform, alt_inst_str=f"{k}_", **v
            )

    def _write_swe_obs(self, count: int) -> None:
        obsnme_str = "oname:obs_swe_{}_otype:arr_i:{}_j:0"

        for j, fid in enumerate(self.pest_args["targets"]):
            # Get SWE data from container
            swe_df = self._get_swe_data(fid)

            self.pest.add_observations(
                self.pest_args["swe_obs"]["file"][j],
                insfile=self.pest_args["swe_obs"]["insfile"][j],
            )

            swe_df["obs_id"] = [obsnme_str.format(fid, k) for k in range(swe_df.shape[0])]
            valid = [ix for ix, r in swe_df.iterrows() if np.isfinite(r["swe"]) and r["swe"] > 0.0]
            valid = swe_df["obs_id"].loc[valid]

            d = self.pest.obs_dfs[j + count].copy()
            d["weight"] = 0.0

            # Weight SWE by inverse magnitude, scaled so SWE contributes
            # ~10-20% of total phi alongside magnitude-weighted ETf.
            try:
                swe_obs = d.loc[valid, "obsval"].values.astype(float)
                d.loc[valid, "weight"] = 1.0 / (26.0 * (swe_obs + 10.0))
            except KeyError:
                valid = [v.lower() for v in valid.values]
                swe_obs = d.loc[valid, "obsval"].values.astype(float)
                d.loc[valid, "weight"] = 1.0 / (26.0 * (swe_obs + 10.0))

            d.loc[np.isnan(d["obsval"]), "weight"] = 0.0
            d.loc[np.isnan(d["obsval"]), "obsval"] = -99.0

            d["idx"] = d.index.map(lambda i: int(i.split(":")[3].split("_")[0]))
            d = d.sort_values(by="idx")
            d.drop(columns=["idx"], inplace=True)

            self.pest.obs_dfs[j + count] = d

    def _write_etf_obs(self, target: str, members: list[str] | None) -> int:
        obsnme_str = "oname:obs_etf_{}_otype:arr_i:{}_j:0"

        weighting_mode = getattr(self.config, "etf_weighting_mode", "spread")
        fixed_sd = getattr(self.config, "etf_weighting_fixed_sd", 0.33)
        spread_floor = getattr(self.config, "etf_weighting_spread_floor", 0.1)
        min_members = getattr(self.config, "etf_weighting_min_members", 2)

        if members is not None:
            self.etf_std = {fid: None for fid in self.pest_args["targets"]}

        total_valid_obs = 0
        for i, fid in enumerate(self.pest_args["targets"]):
            # Get ETf data from container
            etf = self._get_etf_data(fid, model=target)

            self.pest.add_observations(
                self.pest_args["etf_obs"]["file"][i],
                insfile=self.pest_args["etf_obs"]["insfile"][i],
            )

            etf["obs_id"] = [obsnme_str.format(fid, j).lower() for j in range(etf.shape[0])]
            etf["obs_id"].to_csv(self.obs_idx_file, mode="a", header=(i == 0), index=False)

            self.observation_index[fid] = pd.DataFrame(
                data=etf["obs_id"].index, index=etf["obs_id"], columns=["obs_idx"]
            ).copy()

            captures_for_this_target = []
            for ix, r in etf.iterrows():
                for mask in self.masks:
                    if f"{target}_etf_{mask}" in r and not np.isnan(r[f"{target}_etf_{mask}"]):
                        captures_for_this_target.append(etf.loc[ix, "obs_id"])

            self.etf_capture_indexes.append(captures_for_this_target)

            if members is not None:
                etf_std = pd.DataFrame()
                irr = self.irr.get(fid, {})
                irr_threshold = 0.3
                irr_years = [
                    int(k)
                    for k, v in irr.items()
                    if k != "fallow_years" and v["f_irr"] >= irr_threshold
                ]
                irr_index = [i for i in etf.index if hasattr(i, "year") and i.year in irr_years]
                # Compute spread from members only (not members + target)
                for member in members:
                    member_etf = self._get_etf_data(fid, model=member)

                    mask_cols = []
                    for mask in self.masks:
                        col = f"{member}_etf_{mask}"
                        if col in member_etf.columns:
                            mask_cols.append(col)

                    if mask_cols:
                        etf_std[member] = member_etf[mask_cols].mean(axis=1)
                        if irr_index:
                            irr_col = f"{member}_etf_irr"
                            if irr_col in member_etf.columns:
                                etf_std.loc[irr_index, member] = member_etf.loc[irr_index, irr_col]
                    else:
                        etf_std[member] = pd.Series(np.nan, index=etf.index)

                valid_members = [m for m in members if m in etf_std.columns]

                multimodel_dt_mean = pd.Series(index=etf_std.index, dtype=float)
                multimodel_dt_std = pd.Series(index=etf_std.index, dtype=float)
                multimodel_dt_count = pd.Series(index=etf_std.index, dtype=int)

                if valid_members:
                    data_subset = etf_std[valid_members]
                    capture_mask = data_subset.notna()
                    multimodel_dt_count = capture_mask.sum(axis=1)
                    masked_data = data_subset.where(capture_mask)
                    multimodel_dt_mean = masked_data.mean(axis=1)
                    multimodel_dt_std = masked_data.std(axis=1)

                etf_std["std"] = multimodel_dt_std
                etf_std["ct"] = multimodel_dt_count
                etf_std["mean"] = multimodel_dt_mean
                self.etf_std[fid] = etf_std.copy()

            total_valid_obs = sum(len(c) for c in self.etf_capture_indexes)

        for i, fid in enumerate(self.pest_args["targets"]):
            d = self.pest.obs_dfs[i].copy()
            d.index = d.index.str.lower()
            captures_for_this_df = d.index.intersection(self.etf_capture_indexes[i])
            capture_dates = (
                self.observation_index[fid].loc[captures_for_this_df, "obs_idx"].to_list()
            )

            d["weight"] = 0.0

            if not captures_for_this_df.empty and total_valid_obs > 0:
                obsvals = d.loc[captures_for_this_df, "obsval"].values

                # Build common eligibility mask from member count so that
                # both spread and fixed_sd modes use the same capture dates.
                if self.etf_std is not None and self.etf_std.get(fid) is not None:
                    ct_vals = self.etf_std[fid].loc[capture_dates, "ct"].values
                    eligible = ct_vals >= min_members
                else:
                    eligible = np.ones(len(obsvals), dtype=bool)

                if (
                    weighting_mode == "spread"
                    and self.etf_std is not None
                    and self.etf_std.get(fid) is not None
                ):
                    std_vals = self.etf_std[fid].loc[capture_dates, "std"].values
                    weights = np.where(
                        eligible,
                        obsvals / (std_vals + spread_floor),
                        0.0,
                    )
                    d.loc[captures_for_this_df, "weight"] = weights
                else:
                    weights = np.where(eligible, obsvals / fixed_sd, 0.0)
                    d.loc[captures_for_this_df, "weight"] = weights

            # Collect weight audit rows for ablation diagnostics.
            # obs_idx values are Timestamps from the ETf DataFrame index.
            if not captures_for_this_df.empty and total_valid_obs > 0:
                date_stamps = self.observation_index[fid].loc[captures_for_this_df, "obs_idx"]
                for j_cap, (obs_id, dt) in enumerate(zip(captures_for_this_df, date_stamps)):
                    weight_val = float(d.loc[obs_id, "weight"])
                    row = {
                        "fid": fid,
                        "date": dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt),
                        "obsval": obsvals[j_cap],
                        "weight_mode": weighting_mode,
                        "weight_pre_pdc": weight_val,
                        "weight_final": weight_val,
                        "eligible": bool(eligible[j_cap]),
                    }
                    if self.etf_std is not None and self.etf_std.get(fid) is not None:
                        std_df = self.etf_std[fid]
                        row["member_count"] = int(std_df.loc[dt, "ct"])
                        row["member_mean"] = float(std_df.loc[dt, "mean"])
                        row["member_std"] = float(std_df.loc[dt, "std"])
                    else:
                        row["member_count"] = 0
                        row["member_mean"] = np.nan
                        row["member_std"] = np.nan
                    self._weight_audit_rows.append(row)

            d.loc[d["obsval"].isna(), "obsval"] = -99.0
            d.loc[d["weight"].isna(), "weight"] = 0.0

            d["idx"] = d.index.map(lambda i: int(i.split(":")[3].split("_")[0]))
            d = d.sort_values(by="idx").drop(columns=["idx"])

            self.pest.obs_dfs[i] = d

            if self.conflicted_obs:
                self._drop_conflicts(i, fid)

        # Hard check: ensure calibration has usable ETf observations
        n_targets = len(self.pest_args["targets"])
        total_nonzero_etf = sum((df["weight"] > 0).sum() for df in self.pest.obs_dfs[:n_targets])
        if total_valid_obs == 0:
            raise RuntimeError(
                "No valid ETf observations found for any field. "
                f"Checked masks={self.masks}. "
                "Verify that the container has ETf data for the requested mask_mode."
            )
        if total_nonzero_etf == 0:
            raise RuntimeError(
                f"All {total_valid_obs} ETf observations have zero weight. "
                "Check etf_std index alignment with capture_dates."
            )

        return i

    def _finalize_obs(self) -> None:
        """Write std to observations dataframes.

        We *should* be able to write std to the observations dataframes in the etf
        and swe writers, but they are lost in the pest build call, so are written here.
        """
        pst = Pst(self.pst_file)
        obs = pst.observation_data

        obs["standard_deviation"] = 0.00
        etf_idx = [i for i in obs.index if "etf" in i]

        if self.etf_std is not None:
            etf_std_vals = []

            [etf_std_vals.extend(self.etf_std[k]["std"].values) for k in self.pest_args["targets"]]

            obs.loc[etf_idx, "standard_deviation"] = np.array(etf_std_vals)

        else:
            fixed_sd = getattr(self.config, "etf_weighting_fixed_sd", 0.33)
            obs.loc[etf_idx, "standard_deviation"] = fixed_sd

        swe_idx = [i for i, r in obs.iterrows() if "swe" in i and r["obsval"] > 0.0]
        obs.loc[swe_idx, "standard_deviation"] = 5.0

        # add time information
        obs["time"] = [float(i.split(":")[3].split("_")[0]) for i in obs.index]

        pst.write(pst.filename, version=2)

    def apply_prior_params(self, prior_params_path: str) -> None:
        """Override .pst parval1 with LULC-specific prior values from a JSON file.

        The JSON should map site IDs to parameter dicts, e.g.:
            {"US-Bi1": {"aw": 361, "ndvi_k": 4.85, "ks_alpha": 0.39, ...}, ...}

        Parameters not present in the JSON for a given site are left unchanged.
        This must be called after build_pest() and before add_regularization().
        """
        with open(prior_params_path) as f:
            prior_params = json.load(f)

        # Map JSON param names to .pst pargp names
        name_map = {"kr_alpha": "kr_alpha", "ks_alpha": "ks_alpha"}

        pst = Pst(self.pst_file)
        par = pst.parameter_data
        n_updated = 0

        for fid, site_params in prior_params.items():
            for param_name, value in site_params.items():
                pargp = name_map.get(param_name, param_name)
                # Find the matching parameter row
                mask = (par["pargp"] == pargp) & par.index.str.contains(
                    f"_{fid.lower()}_", case=False
                )
                if mask.any():
                    idx = par.index[mask]
                    # Clamp to bounds
                    lb = par.loc[idx, "parlbnd"].values[0]
                    ub = par.loc[idx, "parubnd"].values[0]
                    clamped = max(lb, min(ub, float(value)))
                    par.loc[idx, "parval1"] = clamped
                    n_updated += 1

        pst.write(self.pst_file, version=2)
        print(f"  Applied prior params to {n_updated} parameters from {prior_params_path}")

    @staticmethod
    def _etf_information_mass_by_site(pst) -> dict:
        """Compute sum(weight²) for ETf observations per site."""
        obs = pst.observation_data
        etf_mask = obs.index.str.contains("obs_etf_", case=False)
        etf_obs = obs.loc[etf_mask].copy()

        site_mass = {}
        for obs_name, row in etf_obs.iterrows():
            w = float(row.get("weight", 0.0))
            if not np.isfinite(w) or w <= 0:
                continue
            # Extract site ID: oname:obs_etf_{fid}_otype:...
            parts = obs_name.split("obs_etf_")[1].split("_otype:")[0]
            fid = parts
            site_mass[fid] = site_mass.get(fid, 0.0) + w * w
        return site_mass

    @staticmethod
    def _prior_scale_transformed(partrans, parval1, prior_scale_raw):
        """Convert a raw parameter-space scale to the transformed space used by PI equations."""
        if partrans != "log":
            return prior_scale_raw
        # Log-transform: approximate scale via symmetric finite difference
        v = float(parval1)
        delta = float(prior_scale_raw)
        if v <= 0 or delta <= 0:
            return max(delta, 1e-10)
        v_lo = max(v - delta, 1e-10)
        v_hi = v + delta
        scale = abs(np.log10(v_hi) - np.log10(v_lo)) / 2.0
        return max(scale, 1e-10)

    def _regularization_param_groups(self):
        """Return the list of parameter groups to regularize."""
        configured = getattr(self.config, "prior_regularization_params", None)
        if configured:
            return list(configured)
        return ["aw", "ndvi_k", "ndvi_0", "mad", "ks_alpha", "kr_alpha"]

    def add_regularization(self) -> None:
        """Add phi-balanced Tikhonov prior information equations.

        PI weights are scaled so the total prior phi contribution per site
        equals ``prior_regularization_fraction`` of that site's ETf information
        mass (sum of ETf weight²). This ensures the prior pull is proportional
        to the observation signal strength and prevents parameters with large
        raw scales (like ``aw``) from being effectively unregularized.

        Emits ``regularization_audit.csv`` beside the .pst for inspection.
        """
        f_prior = getattr(self.config, "prior_regularization_fraction", 0.2)
        reg_groups = set(self._regularization_param_groups())
        init_pars = self.initial_parameter_dict()

        pst = Pst(self.pst_file)
        par = pst.parameter_data

        # Step 1: site-local ETf information mass
        site_mass = self._etf_information_mass_by_site(pst)

        # Step 2: count regularized parameters per site
        site_param_count = {}
        for pargp in reg_groups:
            if pargp not in init_pars:
                continue
            target_params = par.loc[par["pargp"] == pargp]
            for par_name, row in target_params.iterrows():
                fid = par_name.split(":")[1].replace(f"{row['pname']}_{row['pargp']}_", "")
                fid = fid[:-1]
                site_param_count[fid] = site_param_count.get(fid, 0) + 1

        # Step 3: add PI equations with phi-balanced weights
        audit_rows = []
        n_pi = 0

        for pargp, values in init_pars.items():
            if pargp not in reg_groups:
                continue
            prior_std_raw = values["std"]
            target_params = par.loc[par["pargp"] == pargp].copy()

            for par_name, row in target_params.iterrows():
                fid = par_name.split(":")[1].replace(f"{row['pname']}_{row['pargp']}_", "")
                fid = fid[:-1]

                partrans = row["partrans"]
                prior_val = float(row["parval1"])
                rhs = np.log10(prior_val) if partrans == "log" else prior_val

                # Compute phi-balanced weight
                I_s = site_mass.get(fid, 0.0)
                n_s = site_param_count.get(fid, 1)
                B_s = f_prior * I_s  # site prior budget
                B_sp = B_s / n_s  # per-parameter budget

                delta_trans = self._prior_scale_transformed(partrans, prior_val, prior_std_raw)

                if B_sp > 0 and delta_trans > 0:
                    weight = np.sqrt(B_sp) / delta_trans
                else:
                    weight = 0.0

                if weight > 0:
                    pst.add_pi_equation(
                        par_names=[par_name],
                        pilbl=f"pi_{pargp}_{fid}",
                        rhs=rhs,
                        weight=weight,
                        obs_group=f"pi_{pargp}",
                    )
                    n_pi += 1

                audit_rows.append(
                    {
                        "fid": fid,
                        "par_name": par_name,
                        "pargp": pargp,
                        "partrans": partrans,
                        "prior_value": prior_val,
                        "prior_scale_raw": prior_std_raw,
                        "prior_scale_trans": delta_trans,
                        "etf_info_mass": I_s,
                        "site_prior_budget": B_s,
                        "param_budget": B_sp,
                        "pi_weight": weight,
                    }
                )

        pst.reg_data.phimlim = sum(len(c) for c in self.etf_capture_indexes)
        pst.reg_data.phimaccept = 1.1 * pst.reg_data.phimlim

        pst.write(self.pst_file, version=2)

        # Emit audit CSV
        if audit_rows:
            audit_df = pd.DataFrame(audit_rows)
            audit_path = os.path.join(self.pest_dir, "regularization_audit.csv")
            audit_df.to_csv(audit_path, index=False)

        # Summary
        if site_mass:
            median_mass = np.median(list(site_mass.values()))
            median_budget = f_prior * median_mass
        else:
            median_mass = 0
            median_budget = 0
        weights_by_group = {}
        for r in audit_rows:
            weights_by_group.setdefault(r["pargp"], []).append(r["pi_weight"])
        print(f"  Regularization: {n_pi} PI equations, f_prior={f_prior}")
        print(f"    ETf info mass: median={median_mass:.0f}, sites={len(site_mass)}")
        print(f"    Prior budget: median={median_budget:.0f}")
        for g in sorted(weights_by_group):
            ws = weights_by_group[g]
            print(
                f"    {g:>12}: median_weight={np.median(ws):.2f}, "
                f"min={min(ws):.2f}, max={max(ws):.2f}"
            )

    def _drop_conflicts(self, i: int, fid: str) -> None:
        pdc = pd.read_csv(self.conflicted_obs, index_col=0)

        d = self.pest.obs_dfs[i].copy()
        start_weight = d["weight"].sum()
        prefix = f"oname:obs_etf_{fid.lower()}_otype:"
        idx = [i for i in pdc.index if i.startswith(prefix)]
        d.loc[idx, "weight"] = 0.0
        end_weight = d["weight"].sum()
        removed = start_weight - end_weight
        self.pest.obs_dfs[i] = d
        print(f"Removed {int(removed)} conflicted obs from etf, leaving {int(end_weight)}")


if __name__ == "__main__":
    pass

# ========================= EOF ====================================================================
