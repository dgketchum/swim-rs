# Contributing to SWIM-RS

Thank you for your interest in contributing to SWIM-RS. This document provides guidelines for contributing to the project.

## Development Setup

### Environment

SWIM-RS uses conda for environment management:

```bash
conda create -n swim python=3.13 -y
conda activate swim
conda install -c conda-forge pestpp geopandas rasterio -y
pip install -e ".[dev]"
```

### Pre-commit Hooks

Install pre-commit hooks to ensure code quality:

```bash
pre-commit install
```

This runs ruff for linting and formatting on every commit.

## Code Style

- **Linter/Formatter**: ruff (configured in pyproject.toml)
- **Line length**: 100 characters
- **Target Python**: 3.13

Run checks manually:

```bash
ruff check .
ruff format .
```

## Testing

Run tests with pytest:

```bash
# All tests (excluding Earth Engine)
pytest

# With coverage
pytest --cov=swimrs --cov-branch

# Include Earth Engine tests (requires auth)
pytest --run-ee

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m regression
```

### Test Categories

- `unit`: Fast, isolated tests for physics kernels and pure functions
- `integration`: Tests requiring multiple modules or external services
- `regression`: Golden file comparison tests
- `parity`: Legacy compatibility tests
- `conservation`: Water balance verification tests

## Commit Guidelines

- Keep commit messages short and descriptive (one line)
- Use imperative mood ("add feature" not "added feature")
- Reference issues when applicable

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with appropriate tests
3. Ensure all tests pass and code style checks pass
4. Submit a pull request with a clear description of changes

## Reporting Issues

When reporting bugs, please include:

- Python version and operating system
- SWIM-RS version (`pip show swimrs`)
- Minimal code example that reproduces the issue
- Full error traceback

## Questions

For questions about usage or development, open a GitHub issue with the "question" label.
