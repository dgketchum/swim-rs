# Changelog

All notable changes to SWIM-RS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

### Changed
- Nothing yet

### Fixed
- Nothing yet

## [0.1.0] - 2025-01-28

### Added
- Initial public release
- SwimContainer API for Zarr-based data management with provenance tracking
- CLI commands: `swim extract`, `swim prep`, `swim calibrate`, `swim evaluate`, `swim inspect`
- Numba-accelerated FAO-56 simulation kernels
- PEST++ IES integration for ensemble calibration via pyemu
- Earth Engine data extraction for Landsat/Sentinel NDVI and OpenET ETf
- GridMET and ERA5-Land meteorology support
- SNODAS snow water equivalent integration
- Five complete examples (Boulder, Fort Peck, Crane, Flux Network, Flux Ensemble)
- MkDocs documentation site with API reference
- CI/CD via GitHub Actions with Codecov integration

[Unreleased]: https://github.com/dgketchum/swim-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dgketchum/swim-rs/releases/tag/v0.1.0
