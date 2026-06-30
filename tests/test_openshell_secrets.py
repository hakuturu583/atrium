"""Tests for OpenShell secret-env forwarding (host env -> sandbox by reference).

The resolver is pure, so it is tested directly without the OpenShell CLI.
"""

from __future__ import annotations

from atrium.sandbox.openshell import _resolve_secret_env


def test_resolves_present_host_vars():
    resolved = _resolve_secret_env(
        {"GH_TOKEN": "GH_TOKEN", "NPM_TOKEN": "MY_NPM_TOKEN"},
        {"GH_TOKEN": "ghp_xyz", "MY_NPM_TOKEN": "npm_abc", "UNRELATED": "x"},
    )
    # Forwarded under the *container* name, sourced from the *host* name.
    assert resolved == {"GH_TOKEN": "ghp_xyz", "NPM_TOKEN": "npm_abc"}


def test_skips_unset_host_vars():
    resolved = _resolve_secret_env(
        {"GH_TOKEN": "GH_TOKEN", "GITHUB_TOKEN": "GITHUB_TOKEN"},
        {"GH_TOKEN": "ghp_xyz"},  # GITHUB_TOKEN absent on host
    )
    assert resolved == {"GH_TOKEN": "ghp_xyz"}


def test_empty_mapping_resolves_to_nothing():
    assert _resolve_secret_env({}, {"GH_TOKEN": "ghp_xyz"}) == {}
