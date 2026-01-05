"""
Workflow configuration parsing and validation.

Provides YAML-based configuration for automated data preparation workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class ProjectConfig:
    """Project-level configuration."""

    name: str
    shapefile: Path
    uid_column: str
    start_date: str
    end_date: str
    output_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectConfig":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "unnamed_project"),
            shapefile=Path(data["shapefile"]),
            uid_column=data.get("uid_column", "FID"),
            start_date=data["date_range"][0] if "date_range" in data else data.get("start_date"),
            end_date=data["date_range"][1] if "date_range" in data else data.get("end_date"),
            output_path=Path(data["output_path"]) if "output_path" in data else None,
        )


@dataclass
class NDVISourceConfig:
    """NDVI source configuration."""

    instrument: str
    mask: str
    path: Path

    @classmethod
    def from_dict(cls, instrument: str, mask: str, path: str) -> "NDVISourceConfig":
        return cls(instrument=instrument, mask=mask, path=Path(path))


@dataclass
class ETFSourceConfig:
    """ETF source configuration."""

    instrument: str
    model: str
    mask: str
    path: Path

    @classmethod
    def from_dict(
        cls, instrument: str, model: str, mask: str, path: str
    ) -> "ETFSourceConfig":
        return cls(instrument=instrument, model=model, mask=mask, path=Path(path))


@dataclass
class MeteorologyConfig:
    """Meteorology source configuration."""

    source: str
    path: Path
    variables: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeteorologyConfig":
        return cls(
            source=data.get("source", "gridmet"),
            path=Path(data["path"]),
            variables=data.get("variables"),
        )


@dataclass
class PropertiesConfig:
    """Properties source configuration."""

    lulc: Optional[Path] = None
    soils: Optional[Path] = None
    irrigation: Optional[Path] = None
    location: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PropertiesConfig":
        return cls(
            lulc=Path(data["lulc"]) if data.get("lulc") else None,
            soils=Path(data["soils"]) if data.get("soils") else None,
            irrigation=Path(data["irrigation"]) if data.get("irrigation") else None,
            location=Path(data["location"]) if data.get("location") else None,
        )


@dataclass
class SourcesConfig:
    """All data sources configuration."""

    ndvi: List[NDVISourceConfig] = field(default_factory=list)
    etf: List[ETFSourceConfig] = field(default_factory=list)
    meteorology: Optional[MeteorologyConfig] = None
    properties: Optional[PropertiesConfig] = None
    snow: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourcesConfig":
        """Parse sources configuration."""
        config = cls()

        # Parse NDVI sources
        ndvi_data = data.get("ndvi", {})
        for instrument, masks in ndvi_data.items():
            if isinstance(masks, dict):
                for mask, path in masks.items():
                    config.ndvi.append(
                        NDVISourceConfig.from_dict(instrument, mask, path)
                    )

        # Parse ETF sources
        etf_data = data.get("etf", {})
        for instrument, models in etf_data.items():
            if isinstance(models, dict):
                for model, masks in models.items():
                    if isinstance(masks, dict):
                        for mask, path in masks.items():
                            config.etf.append(
                                ETFSourceConfig.from_dict(instrument, model, mask, path)
                            )

        # Parse meteorology
        if "meteorology" in data:
            config.meteorology = MeteorologyConfig.from_dict(data["meteorology"])

        # Parse properties
        if "properties" in data:
            config.properties = PropertiesConfig.from_dict(data["properties"])

        # Parse snow
        if "snow" in data:
            config.snow = data["snow"]

        return config


@dataclass
class ComputeConfig:
    """Computation configuration."""

    compute_fused_ndvi: bool = True
    compute_dynamics: bool = True
    dynamics_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComputeConfig":
        return cls(
            compute_fused_ndvi=data.get("compute_fused_ndvi", True),
            compute_dynamics=data.get("compute_dynamics", True) if not isinstance(
                data.get("compute_dynamics"), dict
            ) else True,
            dynamics_params=data.get("compute_dynamics", {}) if isinstance(
                data.get("compute_dynamics"), dict
            ) else {},
        )


@dataclass
class ValidationConfig:
    """Validation configuration."""

    min_area_m2: float = 0
    min_ndvi_obs: int = 10
    min_etf_obs: int = 5
    require_awc: bool = True
    require_lulc: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationConfig":
        return cls(
            min_area_m2=data.get("min_area_m2", 0),
            min_ndvi_obs=data.get("min_ndvi_obs", 10),
            min_etf_obs=data.get("min_etf_obs", 5),
            require_awc=data.get("require_awc", True),
            require_lulc=data.get("require_lulc", True),
        )


@dataclass
class ExportConfig:
    """Export configuration."""

    format: str = "prepped_input_json"
    output: Path = None
    validate: bool = True
    etf_model: str = "ssebop"
    use_fused_ndvi: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExportConfig":
        return cls(
            format=data.get("format", "prepped_input_json"),
            output=Path(data["output"]) if data.get("output") else None,
            validate=data.get("validate", True),
            etf_model=data.get("etf_model", "ssebop"),
            use_fused_ndvi=data.get("use_fused_ndvi", True),
        )


@dataclass
class WorkflowConfig:
    """
    Complete workflow configuration.

    Parsed from YAML configuration file.

    Example YAML:
        project:
          name: "Flux_Network"
          shapefile: "data/sites.shp"
          uid_column: "site_id"
          date_range: ["2017-01-01", "2023-12-31"]

        sources:
          ndvi:
            landsat:
              irr: "data/ndvi/landsat_irr/"
          meteorology:
            source: gridmet
            path: "data/gridmet/"

        workflow:
          compute_fused_ndvi: true
          compute_dynamics:
            etf_model: ssebop

        export:
          format: prepped_input_json
          output: "output/prepped.json"
    """

    project: ProjectConfig
    sources: SourcesConfig
    compute: ComputeConfig
    validation: ValidationConfig
    export: ExportConfig
    config_path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "WorkflowConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            WorkflowConfig instance
        """
        config_path = Path(config_path)

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data, config_path)

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], config_path: Optional[Path] = None
    ) -> "WorkflowConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Configuration dictionary
            config_path: Optional path to config file (for relative path resolution)

        Returns:
            WorkflowConfig instance
        """
        return cls(
            project=ProjectConfig.from_dict(data.get("project", {})),
            sources=SourcesConfig.from_dict(data.get("sources", {})),
            compute=ComputeConfig.from_dict(data.get("workflow", {})),
            validation=ValidationConfig.from_dict(data.get("validation", {})),
            export=ExportConfig.from_dict(data.get("export", {})),
            config_path=config_path,
        )

    def resolve_path(self, path: Union[str, Path]) -> Path:
        """
        Resolve a path relative to the config file location.

        Args:
            path: Path (absolute or relative)

        Returns:
            Resolved absolute path
        """
        path = Path(path)
        if path.is_absolute():
            return path
        if self.config_path is not None:
            return self.config_path.parent / path
        return path.resolve()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "project": {
                "name": self.project.name,
                "shapefile": str(self.project.shapefile),
                "uid_column": self.project.uid_column,
                "date_range": [self.project.start_date, self.project.end_date],
            },
            "sources": {
                "ndvi": {
                    src.instrument: {src.mask: str(src.path)}
                    for src in self.sources.ndvi
                },
                "etf": {
                    src.instrument: {src.model: {src.mask: str(src.path)}}
                    for src in self.sources.etf
                },
            },
            "workflow": {
                "compute_fused_ndvi": self.compute.compute_fused_ndvi,
                "compute_dynamics": self.compute.dynamics_params
                if self.compute.dynamics_params
                else self.compute.compute_dynamics,
            },
            "validation": {
                "min_area_m2": self.validation.min_area_m2,
                "min_ndvi_obs": self.validation.min_ndvi_obs,
            },
            "export": {
                "format": self.export.format,
                "output": str(self.export.output) if self.export.output else None,
            },
        }

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
