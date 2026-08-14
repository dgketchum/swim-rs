"""Unit tests for composing Earth Engine source and irrigation masks."""

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from swimrs.data_extraction.ee.common import clip_and_apply_irrigation_mask


def test_no_mask_only_clips_source_image():
    image = MagicMock()
    clipped = image.clip.return_value
    geometry = object()

    result = clip_and_apply_irrigation_mask(image, geometry, "no_mask")

    assert result is clipped
    image.clip.assert_called_once_with(geometry)
    clipped.updateMask.assert_not_called()
    clipped.mask.assert_not_called()


def test_irr_intersects_existing_mask():
    image = MagicMock()
    clipped = image.clip.return_value
    irr_mask = object()

    result = clip_and_apply_irrigation_mask(
        image,
        object(),
        "irr",
        irr_mask=irr_mask,
    )

    assert result is clipped.updateMask.return_value
    clipped.updateMask.assert_called_once_with(irr_mask)
    clipped.mask.assert_not_called()


def test_inv_irr_intersects_existing_mask():
    image = MagicMock()
    clipped = image.clip.return_value
    irr = MagicMock()
    inv_irr_mask = irr.gt.return_value

    result = clip_and_apply_irrigation_mask(
        image,
        object(),
        "inv_irr",
        irr=irr,
    )

    assert result is clipped.updateMask.return_value
    irr.gt.assert_called_once_with(0)
    clipped.updateMask.assert_called_once_with(inv_irr_mask)
    clipped.mask.assert_not_called()


@pytest.mark.parametrize(
    ("mask_type", "kwargs", "message"),
    [
        ("irr", {}, "irr_mask is required"),
        ("inv_irr", {}, "irr is required"),
        ("typo", {}, "Unknown mask_type"),
    ],
)
def test_invalid_or_incomplete_mask_inputs_fail(mask_type, kwargs, message):
    image = MagicMock()

    with pytest.raises(ValueError, match=message):
        clip_and_apply_irrigation_mask(image, object(), mask_type, **kwargs)


def test_active_exporters_do_not_replace_existing_masks():
    """Guard against reintroducing the deprecated Image.mask setter."""
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "src/swimrs/data_extraction/ee/common.py",
        root / "src/swimrs/data_extraction/ee/ndvi_export.py",
        root / "src/swimrs/data_extraction/ee/etf_export.py",
        root / "src/swimrs/data_extraction/ee/ptjpl_export.py",
        root / "src/swimrs/data_extraction/ee/geesebal_export.py",
        root / "src/swimrs/data_extraction/ee/ssebop_export.py",
        root / "src/swimrs/data_extraction/ee/sims_export.py",
        root / "examples/4_Flux_Network/ssebop_etf.py",
    ]

    offenders = []
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "mask"
                and (node.args or node.keywords)
            ):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")

    assert not offenders, f"Image.mask setter calls found: {offenders}"
