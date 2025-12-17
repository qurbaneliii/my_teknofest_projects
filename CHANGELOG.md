# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-17

### Added
- **AgroScan module**: Full crop health analysis package
  - `compute_ndvi()`, `compute_savi()`, `compute_evi()`, `compute_ndwi()` functions
  - `classify_stress()` with configurable thresholds
  - `preprocessing` module for band loading and normalization
  - CLI with `agroscan ndvi`, `agroscan stress`, `agroscan generate-samples` commands
- **Kalman tracker**: `KalmanTracker` class with Hungarian algorithm for optimal assignment
  - Configurable via `tracker_type: "kalman"` in config
  - Smooth position estimation and velocity tracking
  - Better occlusion handling
- **FastAPI service**: REST API for detection and NDVI endpoints
  - `/api/v1/detect` - Rodent detection
  - `/api/v1/ndvi` - NDVI computation with stress statistics
  - `/api/v1/ndvi/image` - Generate NDVI/stress map images
  - `/api/v1/savi` - SAVI computation
- **Docker support**: Production-ready containerization
  - Multi-stage Dockerfile (production, development)
  - docker-compose with API, SONIC demo, AgroScan demo, docs services
- **AgroScan Streamlit demo**: Interactive crop health analysis app
  - Synthetic data generation for testing
  - NDVI/SAVI computation and visualization
  - Stress classification with configurable thresholds
  - Statistics charts and export functionality
- **Comprehensive test suite**: 60+ tests for new functionality
  - NDVI computation tests
  - Stress classification tests
  - Kalman tracker tests
  - API endpoint tests
- **Enhanced documentation**: Architecture diagrams, API docs, cross-linking

### Changed
- Updated to version 0.3.0
- pyproject.toml now defines both `sonic` and `agroscan` packages
- Added `scipy` dependency for Kalman filter
- Added `fastapi`, `uvicorn`, `python-multipart` as optional API dependencies
- CI workflow now includes coverage upload to Codecov
- Config now supports `tracker_type`, `kalman_iou_threshold`, `kalman_min_hits`

### Fixed
- Improved test coverage to target >80%

## [0.2.0] - 2025-12-17
### Added
- Modular SONIC architecture (core, alerts, visualization, config, CLI)
- Comprehensive tests and fixtures
- Documentation set: architecture, migration, refactoring summary
- Asset reorganization and semantic naming
- Community files: LICENSE, CODE_OF_CONDUCT, CONTRIBUTING
- GitHub Actions CI workflow
- MkDocs documentation site
- SONIC Streamlit demo
- Pydantic config validation

### Changed
- Switched to pyproject-based packaging and entry points
- Added PDF/text extractor module and CLI

### Fixed
- Removed legacy monolithic script and unsafe config side effects
- Removed .venv and .vscode from version control

## [0.1.0] - 2024-xx-xx
- Initial prototype and monolithic detector script
