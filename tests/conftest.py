"""
Shared pytest fixtures and helpers for SwimContainer regression tests.

This module provides:
- Tolerance-based comparison functions for scientific data
- Fixtures for accessing test data directories
- Utilities for creating test containers
"""

# =============================================================================
# Earth Engine Mock Setup (MUST be before any swimrs imports)
# =============================================================================
# Mock the 'ee' module for tests that don't require actual EE authentication.
# This prevents import errors when swimrs modules use ee type hints at module level.
import sys
import types

def _setup_mock_ee():
    """Create a mock ee module that supports type hints and basic operations."""
    mock_ee = types.SimpleNamespace()
    # These must be types (not SimpleNamespace) to support union type hints
    # like ee.Geometry | ee.FeatureCollection
    mock_ee.Image = type('Image', (), {})
    mock_ee.ImageCollection = type('ImageCollection', (), {})
    mock_ee.FeatureCollection = type('FeatureCollection', (), {})
    mock_ee.Feature = type('Feature', (), {})
    mock_ee.Geometry = type('Geometry', (), {'Point': type('Point', (), {})})
    mock_ee.String = type('String', (), {})
    mock_ee.Number = type('Number', (), {})
    mock_ee.List = type('List', (), {})
    mock_ee.Dictionary = type('Dictionary', (), {})
    mock_ee.Date = type('Date', (), {})
    mock_ee.Filter = type('Filter', (), {})
    mock_ee.Algorithms = types.SimpleNamespace()
    mock_ee.Reducer = types.SimpleNamespace()
    mock_ee.Initialize = lambda *args, **kwargs: None
    mock_ee.Authenticate = lambda *args, **kwargs: None
    return mock_ee

# Always set up/update the mock to ensure it has all required attributes
# (previous partial mocks from other test files might be incomplete)
if 'ee' not in sys.modules:
    sys.modules['ee'] = _setup_mock_ee()
else:
    # Update existing mock to ensure it has all required type attributes
    existing_ee = sys.modules['ee']
    # Only update if it looks like our mock (has no real ee methods)
    if not hasattr(existing_ee, 'data') or isinstance(existing_ee, types.SimpleNamespace):
        sys.modules['ee'] = _setup_mock_ee()

# =============================================================================
# Standard imports
# =============================================================================
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytest


# =============================================================================
# Tolerance Settings
# =============================================================================

DEFAULT_RTOL = 0.01  # 1% relative tolerance
DEFAULT_ATOL = 1e-6  # Absolute tolerance for near-zero values


@pytest.fixture
def tolerance() -> Dict[str, float]:
    """Default tolerance settings for floating-point comparisons."""
    return {"rtol": DEFAULT_RTOL, "atol": DEFAULT_ATOL}


# =============================================================================
# Fixture Path Fixtures
# =============================================================================

@pytest.fixture
def fixtures_path() -> Path:
    """Base path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def s2_fixture_path(fixtures_path) -> Path:
    """Path to S2 single-station fixture."""
    return fixtures_path / "S2"


@pytest.fixture
def multi_station_fixture_path(fixtures_path) -> Path:
    """Path to multi-station fixture."""
    return fixtures_path / "multi_station"


# =============================================================================
# Comparison Helpers
# =============================================================================

def compare_scalars_with_tolerance(
    actual: float,
    expected: float,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    name: str = "value",
) -> bool:
    """
    Compare two scalar values with tolerance.

    Args:
        actual: Computed value
        expected: Reference value
        rtol: Relative tolerance (fraction)
        atol: Absolute tolerance
        name: Name for error messages

    Returns:
        True if values match within tolerance

    Raises:
        AssertionError: If values differ beyond tolerance
    """
    if np.isnan(actual) and np.isnan(expected):
        return True

    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        diff = abs(actual - expected)
        rel_diff = diff / abs(expected) if expected != 0 else float('inf')
        raise AssertionError(
            f"{name}: actual={actual}, expected={expected}, "
            f"diff={diff:.6e}, rel_diff={rel_diff:.4%}"
        )
    return True


def compare_arrays_with_tolerance(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    name: str = "array",
) -> bool:
    """
    Compare two numpy arrays with tolerance.

    Args:
        actual: Computed array
        expected: Reference array
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name for error messages

    Returns:
        True if arrays match within tolerance

    Raises:
        AssertionError: If arrays differ beyond tolerance
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch - actual={actual.shape}, expected={expected.shape}"
        )

    # Handle NaN values - both NaN is considered equal
    actual_nan = np.isnan(actual)
    expected_nan = np.isnan(expected)

    if not np.array_equal(actual_nan, expected_nan):
        nan_diff = np.sum(actual_nan != expected_nan)
        raise AssertionError(
            f"{name}: NaN pattern mismatch - {nan_diff} positions differ"
        )

    # Compare non-NaN values
    mask = ~actual_nan
    if not np.any(mask):
        return True  # All NaN

    if not np.allclose(actual[mask], expected[mask], rtol=rtol, atol=atol):
        diffs = np.abs(actual[mask] - expected[mask])
        max_diff_idx = np.argmax(diffs)
        max_diff = diffs[max_diff_idx]
        expected_val = expected[mask].flat[max_diff_idx]
        rel_diff = max_diff / abs(expected_val) if expected_val != 0 else float('inf')

        raise AssertionError(
            f"{name}: max difference at idx {max_diff_idx}, "
            f"diff={max_diff:.6e}, rel_diff={rel_diff:.4%}"
        )

    return True


def compare_json_with_tolerance(
    actual: Any,
    expected: Any,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    path: str = "",
) -> bool:
    """
    Recursively compare JSON-like structures with tolerance for numeric values.

    Handles nested dicts, lists, and numeric values. String keys are compared
    exactly, numeric values use tolerance.

    Args:
        actual: Computed JSON structure
        expected: Reference JSON structure
        rtol: Relative tolerance for numeric comparisons
        atol: Absolute tolerance
        path: Current path for error messages (internal use)

    Returns:
        True if structures match within tolerance

    Raises:
        AssertionError: If structures differ
    """
    if path:
        loc = f" at '{path}'"
    else:
        loc = ""

    # Handle None
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        raise AssertionError(f"None mismatch{loc}: actual={actual}, expected={expected}")

    # Type check
    if type(actual) != type(expected):
        # Allow int/float comparison
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            pass
        else:
            raise AssertionError(
                f"Type mismatch{loc}: actual={type(actual).__name__}, "
                f"expected={type(expected).__name__}"
            )

    # Dict comparison
    if isinstance(expected, dict):
        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())

        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            msg = f"Key mismatch{loc}"
            if missing:
                msg += f", missing: {missing}"
            if extra:
                msg += f", extra: {extra}"
            raise AssertionError(msg)

        for key in expected_keys:
            child_path = f"{path}.{key}" if path else str(key)
            compare_json_with_tolerance(
                actual[key], expected[key], rtol, atol, child_path
            )
        return True

    # List comparison
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(
                f"Length mismatch{loc}: actual={len(actual)}, expected={len(expected)}"
            )

        for i, (a, e) in enumerate(zip(actual, expected)):
            child_path = f"{path}[{i}]"
            compare_json_with_tolerance(a, e, rtol, atol, child_path)
        return True

    # Numeric comparison
    if isinstance(expected, (int, float)):
        compare_scalars_with_tolerance(
            float(actual), float(expected), rtol, atol, path or "value"
        )
        return True

    # String/bool comparison (exact)
    if actual != expected:
        raise AssertionError(f"Value mismatch{loc}: actual={actual}, expected={expected}")

    return True


# =============================================================================
# Golden File Utilities
# =============================================================================

def load_golden_json(golden_dir: Path, filename: str) -> Any:
    """
    Load a golden reference JSON file.

    Args:
        golden_dir: Path to golden files directory
        filename: Name of JSON file (with or without .json extension)

    Returns:
        Parsed JSON content

    Raises:
        FileNotFoundError: If golden file doesn't exist
    """
    if not filename.endswith('.json'):
        filename = f"{filename}.json"

    filepath = golden_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Golden file not found: {filepath}")

    with open(filepath, 'r') as f:
        return json.load(f)


def save_golden_json(golden_dir: Path, filename: str, data: Any) -> Path:
    """
    Save data as a golden reference JSON file.

    Args:
        golden_dir: Path to golden files directory
        filename: Name of JSON file
        data: Data to save

    Returns:
        Path to saved file
    """
    if not filename.endswith('.json'):
        filename = f"{filename}.json"

    golden_dir.mkdir(parents=True, exist_ok=True)
    filepath = golden_dir / filename

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=_json_serializer)

    return filepath


def _json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# =============================================================================
# Container Test Helpers
# =============================================================================

def create_test_container(
    tmp_path: Path,
    shapefile: Path,
    uid_column: str,
    start_date: str,
    end_date: str,
) -> "SwimContainer":
    """
    Create a SwimContainer for testing.

    Args:
        tmp_path: pytest tmp_path fixture for container storage
        shapefile: Path to fields shapefile
        uid_column: Column name for field UIDs
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Initialized SwimContainer
    """
    from swimrs.container import SwimContainer

    container_path = tmp_path / "test_container.swim"

    return SwimContainer.create(
        uri=str(container_path),
        fields_shapefile=str(shapefile),
        uid_column=uid_column,
        start_date=start_date,
        end_date=end_date,
    )


def get_container_dynamics_values(container) -> Dict[str, Any]:
    """
    Extract dynamics values from a container for comparison.

    Args:
        container: SwimContainer with computed dynamics

    Returns:
        Dict with ke_max, kc_max, irr_data, gwsub_data
    """
    result = {}

    # Scalar arrays
    for name, path in [("ke_max", "derived/dynamics/ke_max"),
                       ("kc_max", "derived/dynamics/kc_max")]:
        if path in container._state.root:
            arr = container._state.root[path][:]
            result[name] = arr.tolist()

    # JSON-stored data
    for name, path in [("irr_data", "derived/dynamics/irr_data"),
                       ("gwsub_data", "derived/dynamics/gwsub_data")]:
        if path in container._state.root:
            arr = container._state.root[path]
            # These are string arrays containing JSON
            data = []
            for i in range(arr.shape[0]):
                val = arr[i]
                # Handle zarr v3 ndarray returns
                if hasattr(val, 'item'):
                    val = val.item()
                if val:
                    data.append(json.loads(val))
                else:
                    data.append(None)
            result[name] = data

    return result


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: fast isolated unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "regression: marks regression tests against golden files"
    )
    config.addinivalue_line(
        "markers", "parity: marks parity tests comparing container vs legacy implementations"
    )
    config.addinivalue_line(
        "markers", "conservation: mass balance and water conservation verification"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_ee: marks tests requiring Earth Engine authentication (run with --run-ee)"
    )


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-ee",
        action="store_true",
        default=False,
        help="Run tests that require Earth Engine authentication",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and command line options."""
    if config.getoption("--run-ee"):
        # --run-ee specified: don't skip EE tests
        return

    skip_ee = pytest.mark.skip(reason="need --run-ee option to run")
    for item in items:
        if "requires_ee" in item.keywords:
            item.add_marker(skip_ee)
