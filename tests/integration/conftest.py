"""Opt-in gating + shared env helpers for the integration suite.

All tests here are marked ``integration`` (via each module's ``pytestmark``).
Unless ``ATRIUM_INTEGRATION`` is truthy they are **skipped**, so a plain
``pytest`` at the repo root stays hermetic and green. When opted in, individual
tests still skip if the concrete resource they need is unavailable.
"""

from __future__ import annotations

import os

import pytest

_FALSEY = {"", "0", "false", "no", "off"}


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in _FALSEY


def integration_enabled() -> bool:
    """Whether the opt-in flag ``ATRIUM_INTEGRATION`` is set to a truthy value."""
    return _truthy("ATRIUM_INTEGRATION")


def require_env(*names: str) -> dict[str, str]:
    """Return the given env vars, or ``pytest.skip`` naming the missing ones."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"integration test needs env: {', '.join(missing)}")
    return {n: os.environ[n] for n in names}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: opt-in real-hardware test; runs only with ATRIUM_INTEGRATION=1",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip every ``integration``-marked test unless opted in."""
    if integration_enabled():
        return
    skip = pytest.mark.skip(
        reason="opt-in integration test; set ATRIUM_INTEGRATION=1 to run "
        "(see tests/integration/README.md)"
    )
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)
