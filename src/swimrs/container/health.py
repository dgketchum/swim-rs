"""
Container health checking, policy enforcement, and reporting.

Provides declarative health policies, automated container checks, and
machine-readable + human-readable reports (JSON, HTML, PNG).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np

# ---------------------------------------------------------------------------
# Policy rules
# ---------------------------------------------------------------------------


@dataclass
class PolicyRule:
    """Single policy rule: config choice -> data requirement."""

    config_key: str  # e.g. "mask_mode"
    config_value: str  # e.g. "irrigation"
    required_path: str  # e.g. "properties/irrigation/irr"
    check: str  # "not_all_nan", "not_all_fill", "exists", "not_all_empty"
    threshold: float  # e.g. 0.9 for 90% coverage
    severity: str  # "FAIL" or "WARN"
    message: str  # human-readable explanation


class HealthPolicy:
    """Declarative health contract mapping config choices to data requirements."""

    VERSION = "1.0"
    PROFILE_CALIBRATION = "calibration"
    PROFILE_FORWARD_RUN = "forward_run"
    PROFILE_GENERIC = "generic"

    BASE_RULES = [
        PolicyRule(
            "*",
            "*",
            "properties/soils/awc",
            "not_all_nan",
            0,
            "FAIL",
            "AWC is required for water balance",
        ),
        PolicyRule(
            "*",
            "*",
            "properties/soils/clay",
            "not_all_nan",
            0,
            "WARN",
            "Clay fraction used for soil evaporation",
        ),
        PolicyRule(
            "*",
            "*",
            "properties/land_cover/glc10",
            "not_all_fill",
            -1,
            "WARN",
            "GLC10 land cover missing; falling back to MODIS",
        ),
        PolicyRule(
            "*",
            "*",
            "properties/land_cover/modis_lc",
            "not_all_fill",
            -1,
            "WARN",
            "MODIS land cover missing; GLC10 required",
        ),
    ]

    CONDITIONAL_RULES = [
        PolicyRule(
            "mask_mode",
            "irrigation",
            "properties/irrigation/irr",
            "not_all_nan",
            0,
            "FAIL",
            "mask_mode=irrigation requires irrigation fractions",
        ),
        PolicyRule(
            "mask_mode",
            "irrigation",
            "properties/irrigation/irr_yearly",
            "not_all_empty",
            0,
            "FAIL",
            "mask_mode=irrigation requires per-year irrigation data",
        ),
    ]

    @classmethod
    def for_config(cls, config: dict) -> list[PolicyRule]:
        """Return all applicable rules for the given config."""
        rules = list(cls.BASE_RULES)
        health_profile = str(config.get("health_profile") or cls.PROFILE_GENERIC).strip().lower()

        for rule in cls.CONDITIONAL_RULES:
            cfg_val = config.get(rule.config_key)
            if cfg_val is not None and str(cfg_val) == rule.config_value:
                rules.append(rule)

        # Field-level remote sensing coverage rules based on mask_mode
        mask_mode = config.get("mask_mode")
        if mask_mode == "irrigation":
            # NDVI: every field needs obs in irr OR inv_irr
            rules.append(
                PolicyRule(
                    "mask_mode",
                    "irrigation",
                    "remote_sensing/ndvi/landsat",
                    "field_coverage_union",
                    0,
                    "FAIL",
                    "Every field must have NDVI obs in irr or inv_irr",
                )
            )
        elif mask_mode == "no_mask":
            rules.append(
                PolicyRule(
                    "mask_mode",
                    "no_mask",
                    "remote_sensing/ndvi/landsat/no_mask",
                    "field_coverage",
                    0,
                    "FAIL",
                    "mask_mode=no_mask requires NDVI for all fields",
                )
            )

        # Dynamic rules for etf_target_model
        etf_model = config.get("etf_target_model")
        members = config.get("etf_ensemble_members")
        if etf_model and health_profile != cls.PROFILE_FORWARD_RUN:
            instrument = config.get("etf_target_instrument", "landsat")
            if mask_mode == "irrigation":
                mask = "irr"
            elif mask_mode in ("no_mask", "none"):
                mask = "no_mask"
            else:
                mask = "inv_irr"
            # Skip pre-computed array check when ensemble is computed on-the-fly
            # from configured members
            if not (etf_model == "ensemble" and members):
                path = f"remote_sensing/etf/{instrument}/{etf_model}/{mask}"
                rules.append(
                    PolicyRule(
                        "etf_target_model",
                        etf_model,
                        path,
                        "exists",
                        0,
                        "FAIL",
                        f"etf_target_model={etf_model} requires ETf array at {path}",
                    )
                )
                # Field-level coverage for ETf
                if mask_mode == "irrigation":
                    rules.append(
                        PolicyRule(
                            "etf_target_model",
                            etf_model,
                            f"remote_sensing/etf/{instrument}/{etf_model}",
                            "field_coverage_union",
                            0,
                            "FAIL",
                            f"Every field must have ETf ({etf_model}) obs in irr or inv_irr",
                        )
                    )
                elif mask_mode in ("no_mask", "none"):
                    no_mask_path = f"remote_sensing/etf/{instrument}/{etf_model}/no_mask"
                    rules.append(
                        PolicyRule(
                            "etf_target_model",
                            etf_model,
                            no_mask_path,
                            "field_coverage",
                            0,
                            "FAIL",
                            f"mask_mode=no_mask requires ETf ({etf_model}) for all fields",
                        )
                    )

        # Dynamic rules for ensemble members
        if members and health_profile != cls.PROFILE_FORWARD_RUN:
            if mask_mode == "irrigation":
                mask = "irr"
            elif mask_mode in ("no_mask", "none"):
                mask = "no_mask"
            else:
                mask = "inv_irr"
            for member in members:
                path = f"remote_sensing/etf/landsat/{member}/{mask}"
                rules.append(
                    PolicyRule(
                        "etf_ensemble_members",
                        member,
                        path,
                        "exists",
                        0,
                        "FAIL",
                        f"Ensemble member {member} requires ETf array at {path}",
                    )
                )
                # Field-level coverage for ensemble members
                # SIMS only runs on crop-type fields (CDL), so missing
                # fields are expected — downgrade to WARN.
                severity = "WARN" if member == "sims" else "FAIL"
                if mask_mode == "irrigation":
                    rules.append(
                        PolicyRule(
                            "etf_ensemble_members",
                            member,
                            f"remote_sensing/etf/landsat/{member}",
                            "field_coverage_union",
                            0,
                            severity,
                            f"Every field must have ETf ({member}) obs in irr or inv_irr",
                        )
                    )
                elif mask_mode in ("no_mask", "none"):
                    no_mask_path = f"remote_sensing/etf/landsat/{member}/no_mask"
                    rules.append(
                        PolicyRule(
                            "etf_ensemble_members",
                            member,
                            no_mask_path,
                            "field_coverage",
                            0,
                            severity,
                            f"mask_mode=no_mask requires ETf ({member}) for all fields",
                        )
                    )

        # Dynamic rule for met_source
        met_source = config.get("met_source")
        if met_source:
            rules.append(
                PolicyRule(
                    "met_source",
                    met_source,
                    f"meteorology/{met_source}/eto",
                    "exists",
                    0,
                    "FAIL",
                    f"met_source={met_source} requires meteorology arrays",
                )
            )

        # Dynamic rule for snow_source
        snow_source = config.get("snow_source")
        if snow_source and health_profile != cls.PROFILE_FORWARD_RUN:
            # ERA5 SWE lives under meteorology/era5/swe; SNODAS under snow/snodas/swe
            if snow_source == "era5":
                swe_path = "meteorology/era5/swe"
            else:
                swe_path = f"snow/{snow_source}/swe"
            rules.append(
                PolicyRule(
                    "snow_source",
                    snow_source,
                    swe_path,
                    "exists",
                    0,
                    "FAIL",
                    f"snow_source={snow_source} requires SWE array",
                )
            )

        return rules


# ---------------------------------------------------------------------------
# Check results and health report
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Single health check result."""

    section: str  # "properties", "time_series", "dynamics", "policy"
    path: str  # zarr path or policy rule name
    severity: str  # "FAIL", "WARN", "PASS"
    message: str  # human-readable
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "section": self.section,
            "path": self.path,
            "severity": self.severity,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class HealthReport:
    """Container health report with machine-readable and human-readable output."""

    container_path: str
    n_fields: int
    n_days: int
    date_range: tuple[str, str]
    checks: list[CheckResult]
    policy_version: str
    container_fingerprint: str
    health_profile: str = HealthPolicy.PROFILE_GENERIC
    config_hash: str | None = None

    @property
    def passed(self) -> bool:
        return not any(c.severity == "FAIL" for c in self.checks)

    @property
    def display_profile(self) -> str:
        labels = {
            HealthPolicy.PROFILE_CALIBRATION: "Calibration Run",
            HealthPolicy.PROFILE_FORWARD_RUN: "Forward Run",
            HealthPolicy.PROFILE_GENERIC: "Generic",
        }
        return labels.get(self.health_profile, self.health_profile.replace("_", " ").title())

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "FAIL"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "WARN"]

    def summary(self) -> str:
        """Compact text summary for console output."""
        n_fail = len(self.failures)
        n_warn = len(self.warnings)
        n_pass = sum(1 for c in self.checks if c.severity == "PASS")
        status = "PASS" if self.passed else "FAIL"

        lines = [
            f"Container Health: {status}",
            f"  Path: {self.container_path}",
            f"  Profile: {self.display_profile}",
            f"  Fields: {self.n_fields}  Days: {self.n_days}  "
            f"Range: {self.date_range[0]} to {self.date_range[1]}",
            f"  Checks: {n_pass} pass, {n_warn} warn, {n_fail} fail",
            f"  Fingerprint: {self.container_fingerprint}",
            f"  Policy: v{self.policy_version}",
        ]

        if self.failures:
            lines.append("  FAILURES:")
            for c in self.failures:
                lines.append(f"    [{c.section}] {c.path}: {c.message}")

        if self.warnings:
            lines.append("  WARNINGS:")
            for c in self.warnings:
                lines.append(f"    [{c.section}] {c.path}: {c.message}")

        return "\n".join(lines)

    def to_json(self) -> dict:
        """Machine-readable dict for serialization."""
        return {
            "container_path": self.container_path,
            "n_fields": self.n_fields,
            "n_days": self.n_days,
            "date_range": list(self.date_range),
            "passed": self.passed,
            "health_profile": self.health_profile,
            "policy_version": self.policy_version,
            "container_fingerprint": self.container_fingerprint,
            "config_hash": self.config_hash,
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": [c.to_dict() for c in self.checks],
            "summary": {
                "n_pass": sum(1 for c in self.checks if c.severity == "PASS"),
                "n_warn": len(self.warnings),
                "n_fail": len(self.failures),
            },
        }

    def write_json(self, path: str | Path) -> Path:
        """Serialize report to JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.to_json(), indent=2))
        return path


# ---------------------------------------------------------------------------
# Container health check engine
# ---------------------------------------------------------------------------


class ContainerHealthCheck:
    """Run health checks against a container's zarr root."""

    # Known 1D property paths and their fill values
    PROPERTY_PATHS = {
        "properties/soils/awc": {"fill": float("nan"), "dtype": "float"},
        "properties/soils/clay": {"fill": float("nan"), "dtype": "float"},
        "properties/soils/sand": {"fill": float("nan"), "dtype": "float"},
        "properties/soils/ksat": {"fill": float("nan"), "dtype": "float"},
        "properties/land_cover/glc10": {"fill": -1, "dtype": "int"},
        "properties/land_cover/modis_lc": {"fill": -1, "dtype": "int"},
        "properties/land_cover/cdl_cultivated": {"fill": -1, "dtype": "int"},
        "properties/irrigation/irr": {"fill": float("nan"), "dtype": "float"},
        "properties/location/lat": {"fill": float("nan"), "dtype": "float"},
        "properties/location/lon": {"fill": float("nan"), "dtype": "float"},
        "properties/location/elevation": {"fill": float("nan"), "dtype": "float"},
    }

    def __init__(self, zarr_root, field_uids: list[str], config: dict | None = None):
        self._root = zarr_root
        self._field_uids = field_uids
        self._config = config or {}
        self._checks: list[CheckResult] = []

    def run(self) -> HealthReport:
        """Execute all health checks and return a report."""
        self._checks = []
        self._check_properties()
        self._check_cdl_override()
        self._check_time_series()
        self._check_dynamics()
        self._check_policy()

        # Composite check: at least one LULC source must have valid data
        glc10_ok = any(
            c.path == "properties/land_cover/glc10" and c.severity == "PASS" for c in self._checks
        )
        modis_ok = any(
            c.path == "properties/land_cover/modis_lc" and c.severity == "PASS"
            for c in self._checks
        )
        if not glc10_ok and not modis_ok:
            self._checks.append(
                CheckResult(
                    section="properties",
                    path="properties/land_cover",
                    severity="FAIL",
                    message="No valid land cover data (need at least one of GLC10 or MODIS)",
                )
            )

        fp = fingerprint_container(self._root, self._field_uids)

        # Compute config hash
        config_hash = None
        if self._config:
            config_json = json.dumps(self._config, sort_keys=True, default=str)
            config_hash = f"sha256:{hashlib.sha256(config_json.encode()).hexdigest()}"
        health_profile = (
            str(self._config.get("health_profile") or HealthPolicy.PROFILE_GENERIC).strip().lower()
        )

        # Get date range
        date_range = ("unknown", "unknown")
        try:
            time_arr = self._root["time/daily"][:]
            date_range = (str(time_arr[0])[:10], str(time_arr[-1])[:10])
        except (KeyError, IndexError):
            pass

        # Get n_days
        n_days = 0
        try:
            n_days = len(self._root["time/daily"][:])
        except KeyError:
            pass

        return HealthReport(
            container_path=str(self._root.store) if hasattr(self._root, "store") else "unknown",
            n_fields=len(self._field_uids),
            n_days=n_days,
            date_range=date_range,
            checks=self._checks,
            policy_version=HealthPolicy.VERSION,
            container_fingerprint=fp,
            health_profile=health_profile,
            config_hash=config_hash,
        )

    def _check_properties(self):
        """Check 1D property arrays for NaN/fill coverage."""
        for path, spec in self.PROPERTY_PATHS.items():
            try:
                arr = self._root[path]
            except KeyError:
                # Not present is not necessarily a failure; policy rules handle that
                continue

            data = np.asarray(arr[:])
            total = len(data)

            if spec["dtype"] == "int":
                fill_val = spec["fill"]
                n_fill = int(np.sum(data == fill_val))
                n_valid = total - n_fill
                valid_data = data[data != fill_val]
            else:
                n_fill = int(np.sum(np.isnan(data)))
                n_valid = total - n_fill
                valid_data = data[~np.isnan(data)]

            detail = {
                "total": total,
                "valid": n_valid,
                "fill_or_nan": n_fill,
            }

            if len(valid_data) > 0:
                detail["min"] = float(np.min(valid_data))
                detail["max"] = float(np.max(valid_data))

            if n_valid == 0:
                self._checks.append(
                    CheckResult(
                        "properties",
                        path,
                        "FAIL",
                        f"All {total} values are NaN/fill",
                        detail,
                    )
                )
            elif n_fill / total > 0.1:
                self._checks.append(
                    CheckResult(
                        "properties",
                        path,
                        "WARN",
                        f"{n_fill}/{total} ({100 * n_fill / total:.1f}%) values are NaN/fill",
                        detail,
                    )
                )
            else:
                self._checks.append(
                    CheckResult(
                        "properties",
                        path,
                        "PASS",
                        f"{n_valid}/{total} valid values",
                        detail,
                    )
                )

    def _check_cdl_override(self):
        """Report where the CDL-cultivated override rescues units from perennial mode."""
        from swimrs.container.schema import is_cropland

        try:
            cult = np.asarray(self._root["properties/land_cover/cdl_cultivated"][:])
        except KeyError:
            return

        lulc, source = None, None
        for path, src in (
            ("properties/land_cover/glc10", "glc10"),
            ("properties/land_cover/modis_lc", "modis"),
        ):
            try:
                lulc = np.asarray(self._root[path][:])
                source = src
                break
            except KeyError:
                continue
        if lulc is None:
            return

        crop = np.array([is_cropland(int(c), source) for c in lulc])
        fired = ~crop & (lulc > 0) & (cult == 1)
        reverse = crop & (cult == 0)

        uids = np.asarray(self._field_uids)
        self._checks.append(
            CheckResult(
                "properties",
                "properties/land_cover/cdl_cultivated",
                "PASS",
                f"CDL-cultivated override fired for {int(fired.sum())}/{len(uids)} units",
                {
                    "override_fired": int(fired.sum()),
                    "override_uids": uids[fired].tolist()[:50],
                },
            )
        )
        if reverse.any():
            self._checks.append(
                CheckResult(
                    "properties",
                    "properties/land_cover/cdl_cultivated",
                    "WARN",
                    f"{int(reverse.sum())} {source} cropland units have no cultivated "
                    "CDL history (no action taken)",
                    {
                        "reverse_disagreement": int(reverse.sum()),
                        "reverse_uids": uids[reverse].tolist()[:50],
                    },
                )
            )

    def _check_time_series(self):
        """Check 2D time series arrays for observation coverage."""
        ts_prefixes = ["remote_sensing/", "meteorology/", "snow/"]

        for prefix in ts_prefixes:
            self._walk_and_check_ts(prefix)

    def _walk_and_check_ts(self, prefix: str):
        """Walk zarr groups under prefix and check any 2D arrays found."""
        try:
            parts = prefix.rstrip("/").split("/")
            group = self._root
            for part in parts:
                group = group[part]
        except KeyError:
            return

        self._walk_group_ts(group, prefix.rstrip("/"))

    def _walk_group_ts(self, group, path: str):
        """Recursively walk a zarr group checking 2D arrays."""
        try:
            members = list(group.keys())
        except AttributeError:
            return

        # If this group has both irr and inv_irr 2D children, skip them —
        # the field_coverage_union policy rules handle correctness.
        has_irr_pair = "irr" in members and "inv_irr" in members
        if has_irr_pair:
            try:
                irr_child = group["irr"]
                inv_child = group["inv_irr"]
                irr_is_2d = hasattr(irr_child, "ndim") and irr_child.ndim == 2
                inv_is_2d = hasattr(inv_child, "ndim") and inv_child.ndim == 2
            except KeyError:
                irr_is_2d = inv_is_2d = False
            skip_paired = irr_is_2d and inv_is_2d
        else:
            skip_paired = False

        for name in members:
            child_path = f"{path}/{name}"

            # Skip irr/inv_irr paired arrays (policy rules check these)
            if skip_paired and name in ("irr", "inv_irr"):
                continue

            try:
                child = group[name]
            except KeyError:
                continue

            # Check if it's an array (has ndim) vs a group (has keys)
            if hasattr(child, "ndim"):
                if child.ndim == 2:
                    self._check_single_ts(child, child_path)
            elif hasattr(child, "keys"):
                self._walk_group_ts(child, child_path)

    def _check_single_ts(self, arr, path: str):
        """Check a single 2D time series array."""
        data = np.asarray(arr[:])
        n_time, n_fields = data.shape

        # Count non-NaN obs per field
        valid_per_field = np.sum(~np.isnan(data), axis=0)

        n_all_nan_fields = int(np.sum(valid_per_field == 0))
        total_fields = n_fields

        detail = {
            "shape": list(data.shape),
            "obs_per_field_min": int(np.min(valid_per_field)),
            "obs_per_field_p25": int(np.percentile(valid_per_field, 25)),
            "obs_per_field_median": int(np.median(valid_per_field)),
            "obs_per_field_p75": int(np.percentile(valid_per_field, 75)),
            "obs_per_field_max": int(np.max(valid_per_field)),
            "fields_with_zero_obs": n_all_nan_fields,
        }

        # Abbreviated path for display
        short_path = path.replace("remote_sensing/", "rs/").replace("meteorology/", "met/")

        if np.all(np.isnan(data)):
            self._checks.append(
                CheckResult(
                    "time_series",
                    path,
                    "FAIL",
                    f"Entire array is NaN ({short_path})",
                    detail,
                )
            )
        elif n_all_nan_fields > 0:
            self._checks.append(
                CheckResult(
                    "time_series",
                    path,
                    "WARN",
                    f"{n_all_nan_fields}/{total_fields} fields have zero valid obs ({short_path})",
                    detail,
                )
            )
        else:
            self._checks.append(
                CheckResult(
                    "time_series",
                    path,
                    "PASS",
                    f"All fields have obs, median={detail['obs_per_field_median']} ({short_path})",
                    detail,
                )
            )

    def _check_dynamics(self):
        """Check derived dynamics arrays."""
        for dyn_path in ["derived/dynamics/ke_max", "derived/dynamics/kc_max"]:
            try:
                arr = self._root[dyn_path]
                data = np.asarray(arr[:])
                n_nan = int(np.sum(np.isnan(data)))
                total = data.size
                valid = data[~np.isnan(data)]
                detail = {"total": total, "nan_count": n_nan}
                if len(valid) > 0:
                    detail["min"] = float(np.min(valid))
                    detail["max"] = float(np.max(valid))

                if n_nan == total:
                    self._checks.append(
                        CheckResult(
                            "dynamics",
                            dyn_path,
                            "FAIL",
                            "All NaN",
                            detail,
                        )
                    )
                elif n_nan > 0:
                    self._checks.append(
                        CheckResult(
                            "dynamics",
                            dyn_path,
                            "WARN",
                            f"{n_nan}/{total} NaN values",
                            detail,
                        )
                    )
                else:
                    self._checks.append(
                        CheckResult(
                            "dynamics",
                            dyn_path,
                            "PASS",
                            f"Valid range [{detail.get('min', '?'):.3f}, {detail.get('max', '?'):.3f}]",
                            detail,
                        )
                    )
            except KeyError:
                pass

        # Check irr_data
        try:
            irr_data_arr = self._root["derived/dynamics/irr_data"]
            data = irr_data_arr[:]
            n_irrigated = 0
            for val in data:
                try:
                    s = val if isinstance(val, str) else val.decode("utf-8")
                    parsed = json.loads(s)
                    if any(
                        yr.get("irrigated", 0) > 0 for yr in parsed.values() if isinstance(yr, dict)
                    ):
                        n_irrigated += 1
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass

            detail = {"n_fields": len(data), "n_irrigated": n_irrigated}
            if n_irrigated == 0:
                self._checks.append(
                    CheckResult(
                        "dynamics",
                        "derived/dynamics/irr_data",
                        "WARN",
                        "Zero fields have irrigated years (may indicate upstream property failure)",
                        detail,
                    )
                )
            else:
                self._checks.append(
                    CheckResult(
                        "dynamics",
                        "derived/dynamics/irr_data",
                        "PASS",
                        f"{n_irrigated}/{len(data)} fields have irrigated years",
                        detail,
                    )
                )
        except KeyError:
            pass

    def _check_policy(self):
        """Evaluate HealthPolicy rules against the container."""
        rules = HealthPolicy.for_config(self._config)

        for rule in rules:
            result = self._evaluate_rule(rule)
            self._checks.append(result)

    def _evaluate_rule(self, rule: PolicyRule) -> CheckResult:
        """Evaluate a single policy rule."""
        path = rule.required_path

        # Dispatch checks that operate on groups or need custom reads
        if rule.check == "field_coverage_union":
            return self._check_field_coverage_union(rule)
        if rule.check == "field_coverage":
            return self._check_field_coverage(rule)

        try:
            arr = self._root[path]
        except KeyError:
            return CheckResult(
                "policy",
                path,
                rule.severity,
                f"Missing: {rule.message}",
                {"rule": rule.message, "check": rule.check},
            )

        data = np.asarray(arr[:])

        if rule.check == "not_all_nan":
            if data.dtype.kind == "f" and np.all(np.isnan(data)):
                return CheckResult(
                    "policy",
                    path,
                    rule.severity,
                    f"All NaN: {rule.message}",
                    {"check": rule.check},
                )

        elif rule.check == "not_all_fill":
            fill_val = rule.threshold
            if np.all(data == fill_val):
                return CheckResult(
                    "policy",
                    path,
                    rule.severity,
                    f"All fill value ({fill_val}): {rule.message}",
                    {"check": rule.check, "fill_value": fill_val},
                )

        elif rule.check == "exists":
            # Array exists, that's sufficient
            pass

        elif rule.check == "not_all_empty":
            # For string/JSON arrays, check that not all are empty
            all_empty = True
            for val in data:
                try:
                    s = val if isinstance(val, str) else val.decode("utf-8")
                    if s and s != "{}":
                        all_empty = False
                        break
                except (AttributeError, TypeError):
                    if val:
                        all_empty = False
                        break

            if all_empty:
                return CheckResult(
                    "policy",
                    path,
                    rule.severity,
                    f"All empty: {rule.message}",
                    {"check": rule.check},
                )

        return CheckResult(
            "policy",
            path,
            "PASS",
            f"OK: {rule.message}",
            {"check": rule.check},
        )

    def _check_field_coverage_union(self, rule: PolicyRule) -> CheckResult:
        """Check that every field has obs in at least one of irr or inv_irr."""
        path = rule.required_path
        irr_path = f"{path}/irr"
        inv_irr_path = f"{path}/inv_irr"

        n_fields = len(self._field_uids)
        irr_valid = np.zeros(n_fields, dtype=int)
        inv_irr_valid = np.zeros(n_fields, dtype=int)

        try:
            arr_irr = self._root[irr_path]
            if arr_irr.ndim == 2:
                irr_valid = np.sum(~np.isnan(np.asarray(arr_irr[:])), axis=0)
        except KeyError:
            pass

        try:
            arr_inv = self._root[inv_irr_path]
            if arr_inv.ndim == 2:
                inv_irr_valid = np.sum(~np.isnan(np.asarray(arr_inv[:])), axis=0)
        except KeyError:
            pass

        # A field passes if it has obs in either irr or inv_irr
        union_valid = (irr_valid > 0) | (inv_irr_valid > 0)
        n_uncovered = int(np.sum(~union_valid))

        if n_uncovered > 0:
            return CheckResult(
                "policy",
                path,
                rule.severity,
                f"{n_uncovered}/{n_fields} fields have zero obs in both irr and inv_irr: {rule.message}",
                {
                    "check": rule.check,
                    "uncovered_fields": n_uncovered,
                    "total_fields": n_fields,
                },
            )

        return CheckResult(
            "policy",
            path,
            "PASS",
            f"OK: all fields covered (irr∪inv_irr): {rule.message}",
            {"check": rule.check},
        )

    def _check_field_coverage(self, rule: PolicyRule) -> CheckResult:
        """Check that every field has at least one valid obs."""
        path = rule.required_path
        n_fields = len(self._field_uids)

        try:
            arr = self._root[path]
        except KeyError:
            return CheckResult(
                "policy",
                path,
                rule.severity,
                f"Missing: {rule.message}",
                {"check": rule.check},
            )

        if arr.ndim != 2:
            return CheckResult(
                "policy",
                path,
                "PASS",
                f"OK (non-2D array): {rule.message}",
                {"check": rule.check},
            )

        valid_per_field = np.sum(~np.isnan(np.asarray(arr[:])), axis=0)
        n_uncovered = int(np.sum(valid_per_field == 0))

        if n_uncovered > 0:
            return CheckResult(
                "policy",
                path,
                rule.severity,
                f"{n_uncovered}/{n_fields} fields have zero obs: {rule.message}",
                {
                    "check": rule.check,
                    "uncovered_fields": n_uncovered,
                    "total_fields": n_fields,
                },
            )

        return CheckResult(
            "policy",
            path,
            "PASS",
            f"OK: all fields have obs: {rule.message}",
            {"check": rule.check},
        )


# ---------------------------------------------------------------------------
# Container fingerprint
# ---------------------------------------------------------------------------


def fingerprint_container(zarr_root, field_uids: list[str]) -> str:
    """Sample-based hash: metadata + first/last 10 rows of each array."""
    h = hashlib.sha256()

    # Hash field UIDs
    for uid in field_uids:
        h.update(uid.encode())

    # Walk arrays and hash shapes, dtypes, and sample data
    _hash_group(h, zarr_root)

    return h.hexdigest()[:16]


def _hash_group(h: Any, group):
    """Recursively hash zarr group contents."""
    try:
        members = sorted(group.keys())
    except AttributeError:
        return

    for name in members:
        try:
            child = group[name]
        except KeyError:
            continue

        h.update(name.encode())

        if hasattr(child, "ndim"):
            # It's an array
            h.update(str(child.shape).encode())
            h.update(str(child.dtype).encode())

            try:
                if child.ndim == 1:
                    # Hash full content of 1D arrays (small)
                    h.update(np.asarray(child[:]).tobytes())
                elif child.ndim == 2:
                    # Hash first n and last n rows via partial reads
                    n = min(10, child.shape[0])
                    h.update(np.asarray(child[:n]).tobytes())
                    h.update(np.asarray(child[-n:]).tobytes())
            except Exception:
                pass
        elif hasattr(child, "keys"):
            _hash_group(h, child)


# ---------------------------------------------------------------------------
# HTML report renderer
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Container Health Report - {{ report.display_profile }}</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #333; }
h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
h2 { color: #555; margin-top: 30px; }
.status-pass { color: #22863a; font-weight: bold; }
.status-fail { color: #cb2431; font-weight: bold; }
.status-warn { color: #b08800; font-weight: bold; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
th { background: #f6f8fa; }
tr.fail { background: #ffeef0; }
tr.warn { background: #fffbdd; }
tr.pass { background: #f0fff4; }
.meta { color: #666; font-size: 0.9em; }
.detail { font-size: 0.85em; color: #555; }
</style>
</head>
<body>
<h1>Container Health Report - {{ report.display_profile }}</h1>
<p class="status-{{ 'pass' if report.passed else 'fail' }}">
  Overall: {{ 'PASS' if report.passed else 'FAIL' }}
</p>

<h2>Overview</h2>
<table>
<tr><td>Container</td><td>{{ report.container_path }}</td></tr>
<tr><td>Profile</td><td>{{ report.display_profile }}</td></tr>
<tr><td>Fields</td><td>{{ report.n_fields }}</td></tr>
<tr><td>Days</td><td>{{ report.n_days }}</td></tr>
<tr><td>Date Range</td><td>{{ report.date_range[0] }} to {{ report.date_range[1] }}</td></tr>
<tr><td>Fingerprint</td><td><code>{{ report.container_fingerprint }}</code></td></tr>
<tr><td>Config Hash</td><td><code>{{ report.config_hash or 'N/A' }}</code></td></tr>
<tr><td>Policy Version</td><td>{{ report.policy_version }}</td></tr>
</table>

<h2>Checks ({{ checks | length }})</h2>
<table>
<tr><th>Section</th><th>Path</th><th>Status</th><th>Message</th><th>Detail</th></tr>
{% for c in checks %}
<tr class="{{ c.severity | lower }}">
  <td>{{ c.section }}</td>
  <td><code>{{ c.path }}</code></td>
  <td class="status-{{ c.severity | lower }}">{{ c.severity }}</td>
  <td>{{ c.message }}</td>
  <td class="detail">{{ c.detail }}</td>
</tr>
{% endfor %}
</table>

<p class="meta">Generated: {{ timestamp }} | Policy v{{ report.policy_version }}</p>
</body>
</html>"""


def render_html_report(report: HealthReport, output_path: str | Path) -> Path:
    """Render HealthReport to a self-contained HTML file."""
    from jinja2 import Template

    output_path = Path(output_path)

    # Sort checks: FAIL first, then WARN, then PASS
    severity_order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    sorted_checks = sorted(report.checks, key=lambda c: severity_order.get(c.severity, 3))

    template = Template(_HTML_TEMPLATE)
    html = template.render(
        report=report,
        checks=[c.to_dict() for c in sorted_checks],
        timestamp=datetime.now(UTC).isoformat(),
    )

    output_path.write_text(html)
    return output_path


# ---------------------------------------------------------------------------
# Summary PNG renderer
# ---------------------------------------------------------------------------


def render_summary_png(report: HealthReport, output_path: str | Path) -> Path:
    """Render a compact 1-page summary figure."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        f"Container Health: {'PASS' if report.passed else 'FAIL'}",
        fontsize=14,
        fontweight="bold",
    )

    colors = {"PASS": "#22863a", "WARN": "#b08800", "FAIL": "#cb2431"}

    # Panel 1: Properties coverage
    ax1 = fig.add_subplot(1, 3, 1)
    prop_checks = [c for c in report.checks if c.section == "properties"]
    if prop_checks:
        labels = [c.path.split("/")[-1] for c in prop_checks]
        valid_pcts = []
        bar_colors = []
        for c in prop_checks:
            total = c.detail.get("total", 1)
            valid = c.detail.get("valid", 0)
            valid_pcts.append(100 * valid / total if total > 0 else 0)
            bar_colors.append(colors.get(c.severity, "#999"))
        ax1.barh(labels, valid_pcts, color=bar_colors)
        ax1.set_xlim(0, 105)
        ax1.set_xlabel("% Valid")
    ax1.set_title("Properties Coverage")

    # Panel 2: Time series obs/field
    ax2 = fig.add_subplot(1, 3, 2)
    ts_checks = [c for c in report.checks if c.section == "time_series"]
    if ts_checks:
        labels = []
        medians = []
        for c in ts_checks:
            short = c.path.replace("remote_sensing/", "rs/").replace("meteorology/", "met/")
            labels.append(short.split("/")[-1] if len(short) > 30 else short)
            medians.append(c.detail.get("obs_per_field_median", 0))
        ax2.barh(labels, medians, color="#4a90d9")
        ax2.set_xlabel("Median Obs/Field")
    ax2.set_title("Time Series Coverage")

    # Panel 3: Policy check grid
    ax3 = fig.add_subplot(1, 3, 3)
    policy_checks = [c for c in report.checks if c.section == "policy"]
    if policy_checks:
        labels = [c.path.split("/")[-1] for c in policy_checks]
        x_colors = [colors.get(c.severity, "#999") for c in policy_checks]
        y_pos = range(len(labels))
        ax3.barh(list(y_pos), [1] * len(labels), color=x_colors)
        ax3.set_yticks(list(y_pos))
        ax3.set_yticklabels(labels)
        ax3.set_xticks([])
    ax3.set_title("Policy Checks")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def health_report_output_dir(
    container_path: str | Path,
    *,
    base_dir: str | Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Build a timestamped directory for persisted health-report artifacts."""

    stamp = (timestamp or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")

    if base_dir is None:
        container_root = _container_path_to_local_path(container_path)
        root = Path(f"{container_root}.reports")
    else:
        root = Path(base_dir)

    return root / "health" / stamp


def _container_path_to_local_path(container_path: str | Path) -> Path:
    """Resolve file:// URIs and plain paths to a local filesystem path."""

    path_str = str(container_path)
    parsed = urlparse(path_str)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(path_str)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ContainerHealthError(Exception):
    """Raised when a container fails health checks and raise_on_fail is True."""

    def __init__(self, report: HealthReport):
        self.report = report
        failures = report.failures
        msg = f"Container health check failed: {len(failures)} FAIL(s). " + "; ".join(
            f"[{c.path}] {c.message}" for c in failures[:5]
        )
        super().__init__(msg)
