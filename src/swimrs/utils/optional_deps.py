"""Helpers for optional dependency imports.

Keep error messages consistent and actionable when optional integrations
are not installed.
"""

from __future__ import annotations


def missing_optional_dependency(*, extra: str, purpose: str, import_name: str) -> ImportError:
    msg = (
        f"Missing optional dependency '{import_name}' required for {purpose}. "
        f'Install with `pip install "swimrs[{extra}]"`.'
    )
    return ImportError(msg)


def require_optional_dependency(
    module_obj,
    *,
    extra: str,
    purpose: str,
    import_name: str,
) -> None:
    if module_obj is None:  # pragma: no cover
        raise missing_optional_dependency(extra=extra, purpose=purpose, import_name=import_name)
