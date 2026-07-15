"""Opt-in, real-hardware integration tests (OpenShell CLI, tabbyAPI, Phoenix).

These are **excluded from the default ``pytest`` run and from CI** — they need
real infrastructure a unit runner does not have. They run only when
``ATRIUM_INTEGRATION=1`` is set, and each still self-skips when its specific
dependency (the OpenShell CLI, a reachable bridge, a running Phoenix) is absent.
See ``tests/integration/README.md``.
"""
