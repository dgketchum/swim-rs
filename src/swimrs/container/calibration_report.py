"""Calibration report for SWIM containers.

Generates summary statistics, QC flags, and HTML/PNG reports for
calibrated parameters stored in a container's ``calibration/`` group.

Usage::

    from swimrs.container import open_container

    container = open_container("project.swim")
    report = container.calibration_report()
    print(report.summary())
    report.render_html("calibration.html")
    report.render_png("calibration.png")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Parameter bounds from pest_builder — the authoritative PEST++ prior ranges.
PARAM_BOUNDS = {
    "aw": (100.0, 400.0),
    "ks_damp": (0.01, 1.0),
    "kr_damp": (0.01, 1.0),
    "ndvi_k": (3.0, 20.0),
    "ndvi_0": (0.1, 0.80),
    "ndvi_alpha": (-0.7, 1.5),
    "ndvi_beta": (0.5, 1.7),
    "mad": (0.10, 0.9),
    "swe_alpha": (-0.5, 1.0),
    "swe_beta": (0.5, 2.5),
}

PARAM_DESCRIPTIONS = {
    "aw": "Available water capacity (mm)",
    "ks_damp": "Soil stress damping coefficient",
    "kr_damp": "Evaporation stress damping coefficient",
    "ndvi_k": "NDVI-Kcb sigmoid steepness",
    "ndvi_0": "NDVI-Kcb sigmoid midpoint",
    "ndvi_alpha": "NDVI-Kcb linear intercept",
    "ndvi_beta": "NDVI-Kcb linear slope",
    "mad": "Management allowed depletion fraction",
    "swe_alpha": "Snow melt temperature sensitivity",
    "swe_beta": "Snow melt radiation sensitivity",
}


@dataclass
class ParamStats:
    """Summary statistics for one calibrated parameter across all fields."""

    name: str
    values: np.ndarray
    uncertainty: np.ndarray
    lower_bound: float
    upper_bound: float

    @property
    def n_fields(self) -> int:
        return len(self.values)

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(~np.isnan(self.values)))

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.values))

    @property
    def median(self) -> float:
        return float(np.nanmedian(self.values))

    @property
    def std(self) -> float:
        return float(np.nanstd(self.values))

    @property
    def pmin(self) -> float:
        return float(np.nanmin(self.values))

    @property
    def pmax(self) -> float:
        return float(np.nanmax(self.values))

    @property
    def mean_uncertainty(self) -> float:
        return float(np.nanmean(self.uncertainty))

    @property
    def n_at_lower(self) -> int:
        """Fields within 1% of the lower bound."""
        tol = 0.01 * (self.upper_bound - self.lower_bound)
        return int(np.count_nonzero(self.values <= self.lower_bound + tol))

    @property
    def n_at_upper(self) -> int:
        """Fields within 1% of the upper bound."""
        tol = 0.01 * (self.upper_bound - self.lower_bound)
        return int(np.count_nonzero(self.values >= self.upper_bound - tol))

    @property
    def high_uncertainty_threshold(self) -> float:
        """Uncertainty > 25% of the parameter range is flagged."""
        return 0.25 * (self.upper_bound - self.lower_bound)

    @property
    def n_high_uncertainty(self) -> int:
        return int(np.count_nonzero(self.uncertainty > self.high_uncertainty_threshold))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": PARAM_DESCRIPTIONS.get(self.name, ""),
            "n_fields": self.n_fields,
            "n_valid": self.n_valid,
            "mean": round(self.mean, 4),
            "median": round(self.median, 4),
            "std": round(self.std, 4),
            "min": round(self.pmin, 4),
            "max": round(self.pmax, 4),
            "bounds": [self.lower_bound, self.upper_bound],
            "mean_uncertainty": round(self.mean_uncertainty, 4),
            "n_at_lower_bound": self.n_at_lower,
            "n_at_upper_bound": self.n_at_upper,
            "n_high_uncertainty": self.n_high_uncertainty,
        }


@dataclass
class FieldFlag:
    """A QC flag for a specific field."""

    fid: str
    param: str
    flag: str  # "at_lower", "at_upper", "high_uncertainty"
    value: float
    uncertainty: float


@dataclass
class CalibrationReport:
    """Calibration report with summary stats, QC flags, and rendering."""

    container_path: str
    n_fields: int
    n_calibrated: int
    n_batches: int
    params: list[ParamStats]
    field_flags: list[FieldFlag] = field(default_factory=list)
    batch_info: dict = field(default_factory=dict)
    field_uids: list[str] = field(default_factory=list)
    irr_fractions: np.ndarray | None = None
    lulc_codes: np.ndarray | None = None
    timestamp: str = ""

    @property
    def n_flags(self) -> int:
        return len(self.field_flags)

    @property
    def flagged_fids(self) -> set[str]:
        return {f.fid for f in self.field_flags}

    @property
    def n_flagged_fields(self) -> int:
        return len(self.flagged_fids)

    def summary(self) -> str:
        """Compact text summary for console output."""
        lines = [
            "Calibration Report",
            f"  Container: {self.container_path}",
            f"  Fields: {self.n_calibrated}/{self.n_fields} calibrated, {self.n_batches} batches",
            f"  Parameters: {len(self.params)}",
        ]

        lines.append("")
        lines.append(
            f"  {'Param':<12} {'Mean':>8} {'Std':>8} "
            f"{'Min':>8} {'Max':>8} {'Bounds':>14} "
            f"{'AtLo':>5} {'AtHi':>5} {'HiUnc':>5}"
        )
        lines.append("  " + "-" * 80)
        for p in self.params:
            lo, hi = p.lower_bound, p.upper_bound
            lines.append(
                f"  {p.name:<12} {p.mean:>8.3f} {p.std:>8.3f} "
                f"{p.pmin:>8.3f} {p.pmax:>8.3f} "
                f"[{lo:>5.2f},{hi:>5.2f}] "
                f"{p.n_at_lower:>5} {p.n_at_upper:>5} {p.n_high_uncertainty:>5}"
            )

        if self.field_flags:
            lines.append("")
            lines.append(f"  QC Flags: {self.n_flags} flags on {self.n_flagged_fields} fields")
            # Summarize by flag type
            flag_counts = {}
            for f in self.field_flags:
                key = f"{f.param}:{f.flag}"
                flag_counts[key] = flag_counts.get(key, 0) + 1
            for key, count in sorted(flag_counts.items()):
                lines.append(f"    {key}: {count} fields")
        else:
            lines.append("")
            lines.append("  QC Flags: none")

        return "\n".join(lines)

    def to_json(self) -> dict:
        """Machine-readable dict for serialization."""
        return {
            "container_path": str(self.container_path),
            "timestamp": self.timestamp,
            "n_fields": self.n_fields,
            "n_calibrated": self.n_calibrated,
            "n_batches": self.n_batches,
            "parameters": [p.to_dict() for p in self.params],
            "n_flags": self.n_flags,
            "n_flagged_fields": self.n_flagged_fields,
            "flags_by_type": _summarize_flags(self.field_flags),
            "field_flags": [
                {
                    "fid": f.fid,
                    "param": f.param,
                    "flag": f.flag,
                    "value": round(f.value, 4),
                    "uncertainty": round(f.uncertainty, 4),
                }
                for f in self.field_flags
            ],
        }

    def to_dataframe(self):
        """Export calibrated parameters as a DataFrame (one row per field)."""
        import pandas as pd

        data = {"fid": self.field_uids}
        for p in self.params:
            data[p.name] = p.values
            data[f"{p.name}_std"] = p.uncertainty
        if self.irr_fractions is not None:
            data["irr"] = self.irr_fractions
        if self.lulc_codes is not None:
            data["lulc"] = self.lulc_codes
        return pd.DataFrame(data)

    def render_html(self, output_path: str | Path) -> Path:
        """Render to a self-contained HTML file."""
        return render_html_report(self, output_path)

    def render_png(self, output_path: str | Path) -> Path:
        """Render a summary PNG figure."""
        return render_png_report(self, output_path)


def _summarize_flags(flags: list[FieldFlag]) -> dict:
    counts = {}
    for f in flags:
        key = f"{f.param}:{f.flag}"
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_calibration_report(
    root, field_uids: list[str], container_path: str = ""
) -> CalibrationReport:
    """Build a CalibrationReport from a zarr root with a ``calibration/`` group.

    Parameters
    ----------
    root : zarr.Group
        Container root group (must have ``calibration/parameters/`` etc.)
    field_uids : list[str]
        Ordered field UIDs matching array indices.
    container_path : str
        Display path for the container.

    Returns
    -------
    CalibrationReport
    """
    if "calibration" not in root or "calibration/parameters" not in root:
        raise ValueError("Container has no calibration data")

    n_fields = len(field_uids)

    # Read calibrated flag
    calibrated = np.zeros(n_fields, dtype="uint8")
    if "calibration/metadata/calibrated" in root:
        calibrated = np.asarray(root["calibration/metadata/calibrated"][:])
    n_calibrated = int(np.count_nonzero(calibrated))

    # Batch info
    n_batches = 0
    batch_info = {}
    if "calibration" in root:
        batches_str = root["calibration"].attrs.get("batches", "{}")
        if isinstance(batches_str, str):
            batch_info = json.loads(batches_str)
        elif isinstance(batches_str, dict):
            batch_info = batches_str
        n_batches = len(batch_info)

    # Build per-parameter stats
    param_stats = []
    for name in sorted(PARAM_BOUNDS.keys()):
        param_path = f"calibration/parameters/{name}"
        unc_path = f"calibration/uncertainty/{name}"
        if param_path not in root:
            continue

        values = np.asarray(root[param_path][:], dtype="float64")
        if unc_path in root:
            uncertainty = np.asarray(root[unc_path][:], dtype="float64")
        else:
            uncertainty = np.full(n_fields, np.nan)

        lo, hi = PARAM_BOUNDS[name]
        param_stats.append(ParamStats(name, values, uncertainty, lo, hi))

    # Build field-level QC flags
    field_flags = []
    for ps in param_stats:
        tol = 0.01 * (ps.upper_bound - ps.lower_bound)
        for i, fid in enumerate(field_uids):
            val = ps.values[i]
            unc = ps.uncertainty[i]
            if np.isnan(val):
                continue
            if val <= ps.lower_bound + tol:
                field_flags.append(FieldFlag(fid, ps.name, "at_lower", val, unc))
            elif val >= ps.upper_bound - tol:
                field_flags.append(FieldFlag(fid, ps.name, "at_upper", val, unc))
            if not np.isnan(unc) and unc > ps.high_uncertainty_threshold:
                field_flags.append(FieldFlag(fid, ps.name, "high_uncertainty", val, unc))

    # Read optional properties for context
    irr_fractions = None
    lulc_codes = None
    if "properties/irrigation/irr" in root:
        irr_fractions = np.asarray(root["properties/irrigation/irr"][:], dtype="float64")
    if "properties/land_cover/glc10" in root:
        lulc_codes = np.asarray(root["properties/land_cover/glc10"][:])
    elif "properties/land_cover/modis_lc" in root:
        lulc_codes = np.asarray(root["properties/land_cover/modis_lc"][:])

    return CalibrationReport(
        container_path=container_path,
        n_fields=n_fields,
        n_calibrated=n_calibrated,
        n_batches=n_batches,
        params=param_stats,
        field_flags=field_flags,
        batch_info=batch_info,
        field_uids=list(field_uids),
        irr_fractions=irr_fractions,
        lulc_codes=lulc_codes,
        timestamp=datetime.now(UTC).isoformat(),
    )


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SWIM Calibration Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #24292e; }
  h1 { border-bottom: 2px solid #e1e4e8; padding-bottom: 0.3em; }
  h2 { border-bottom: 1px solid #e1e4e8; padding-bottom: 0.2em; margin-top: 1.5em; }
  .summary { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px;
             padding: 1em; margin-bottom: 1.5em; }
  .summary dt { font-weight: 600; display: inline; }
  .summary dd { display: inline; margin-left: 0.3em; margin-right: 1.5em; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
  th, td { border: 1px solid #e1e4e8; padding: 6px 10px; text-align: right; }
  th { background: #f6f8fa; text-align: left; }
  td:first-child, th:first-child { text-align: left; }
  .flag-at_lower { background: #fff3cd; }
  .flag-at_upper { background: #fff3cd; }
  .flag-high_uncertainty { background: #f8d7da; }
  .good { color: #22863a; }
  .warn { color: #b08800; }
  .bad { color: #cb2431; }
  .bar-container { display: inline-block; width: 120px; height: 14px;
                   background: #e1e4e8; border-radius: 3px; vertical-align: middle; }
  .bar-fill { height: 100%; border-radius: 3px; }
  footer { margin-top: 2em; padding-top: 1em; border-top: 1px solid #e1e4e8;
           font-size: 0.85em; color: #6a737d; }
</style>
</head>
<body>
<h1>SWIM Calibration Report</h1>

<div class="summary">
  <dl>
    <dt>Container:</dt><dd>{{ report.container_path }}</dd>
    <dt>Fields:</dt><dd>{{ report.n_calibrated }}/{{ report.n_fields }} calibrated</dd>
    <dt>Batches:</dt><dd>{{ report.n_batches }}</dd>
    <dt>Parameters:</dt><dd>{{ report.params | length }}</dd>
    <dt>QC Flags:</dt><dd>{{ report.n_flags }} on {{ report.n_flagged_fields }} fields</dd>
  </dl>
</div>

<h2>Parameter Summary</h2>
<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
      <th>Mean</th>
      <th>Std</th>
      <th>Min</th>
      <th>Max</th>
      <th>Bounds</th>
      <th>At Lower</th>
      <th>At Upper</th>
      <th>High Unc.</th>
      <th>Range Used</th>
    </tr>
  </thead>
  <tbody>
  {% for p in params %}
    <tr>
      <td><strong>{{ p.name }}</strong></td>
      <td>{{ p.description }}</td>
      <td>{{ "%.3f" | format(p.mean) }}</td>
      <td>{{ "%.3f" | format(p.std) }}</td>
      <td>{{ "%.3f" | format(p.min) }}</td>
      <td>{{ "%.3f" | format(p.max) }}</td>
      <td>[{{ "%.2f" | format(p.bounds[0]) }}, {{ "%.2f" | format(p.bounds[1]) }}]</td>
      <td class="{{ 'warn' if p.n_at_lower_bound > 0 else 'good' }}">{{ p.n_at_lower_bound }}</td>
      <td class="{{ 'warn' if p.n_at_upper_bound > 0 else 'good' }}">{{ p.n_at_upper_bound }}</td>
      <td class="{{ 'bad' if p.n_high_uncertainty > 0 else 'good' }}">{{ p.n_high_uncertainty }}</td>
      <td>
        {% set range_pct = ((p.max - p.min) / (p.bounds[1] - p.bounds[0]) * 100) | round(0) | int %}
        <span class="bar-container">
          <span class="bar-fill" style="width:{{ range_pct }}%; background:{{ '#22863a' if range_pct < 90 else '#b08800' if range_pct < 99 else '#cb2431' }};"></span>
        </span>
        {{ range_pct }}%
      </td>
    </tr>
  {% endfor %}
  </tbody>
</table>

{% if flags_by_type %}
<h2>QC Flags Summary</h2>
<table>
  <thead><tr><th>Flag</th><th>Fields</th></tr></thead>
  <tbody>
  {% for key, count in flags_by_type.items() %}
    <tr><td>{{ key }}</td><td>{{ count }}</td></tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

<footer>
  Generated {{ timestamp }} by swimrs calibration_report
</footer>
</body>
</html>
"""


def render_html_report(report: CalibrationReport, output_path: str | Path) -> Path:
    """Render CalibrationReport to a self-contained HTML file."""
    from jinja2 import Template

    output_path = Path(output_path)
    params = [p.to_dict() for p in report.params]

    template = Template(_HTML_TEMPLATE)
    html = template.render(
        report=report,
        params=params,
        flags_by_type=_summarize_flags(report.field_flags),
        timestamp=datetime.now(UTC).isoformat(),
    )
    output_path.write_text(html)
    return output_path


# ---------------------------------------------------------------------------
# PNG renderer
# ---------------------------------------------------------------------------


def render_png_report(report: CalibrationReport, output_path: str | Path) -> Path:
    """Render a multi-panel calibration summary figure."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    n_params = len(report.params)

    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(18, 4 * n_rows))
    fig.suptitle(
        f"Calibration: {report.n_calibrated}/{report.n_fields} fields, {report.n_batches} batches",
        fontsize=13,
        fontweight="bold",
    )

    axes = [fig.add_subplot(n_rows, n_cols, i + 1) for i in range(n_rows * n_cols)]

    for i, ps in enumerate(report.params):
        if i >= len(axes):
            break
        ax = axes[i]

        valid = ps.values[~np.isnan(ps.values)]
        if len(valid) == 0:
            ax.set_title(ps.name)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.hist(valid, bins=40, color="#4a90d9", edgecolor="white", linewidth=0.3)
        ax.axvline(ps.lower_bound, color="#cb2431", linestyle="--", linewidth=1, label="bound")
        ax.axvline(ps.upper_bound, color="#cb2431", linestyle="--", linewidth=1)
        ax.axvline(ps.median, color="#22863a", linestyle="-", linewidth=1.5, label="median")
        ax.set_title(ps.name, fontsize=10, fontweight="bold")
        ax.set_xlabel(PARAM_DESCRIPTIONS.get(ps.name, ""), fontsize=7)

        # Annotate bound-hitting counts
        flags = []
        if ps.n_at_lower > 0:
            flags.append(f"{ps.n_at_lower} at lo")
        if ps.n_at_upper > 0:
            flags.append(f"{ps.n_at_upper} at hi")
        if ps.n_high_uncertainty > 0:
            flags.append(f"{ps.n_high_uncertainty} hi unc")
        if flags:
            ax.annotate(
                ", ".join(flags),
                xy=(0.98, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=7,
                color="#cb2431",
            )

    # Hide unused axes
    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
