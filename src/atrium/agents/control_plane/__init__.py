"""Control plane — the trusted seam between human I/O and the workboard.

* ``protocol.py`` — the A2A contract an interface agent and the control plane
  share (``workboard.submit`` / ``workboard_submitted`` / ``job_update``).
* ``agent.py`` — the :class:`ControlPlaneAgent`, sole caller of
  :func:`atrium.orchestration.kick.submit_job`.

Fixed infrastructure (trusted tier), unlike the evolvable interface agents that
forward to it. See ``docs/design/interface-agent.md``.

``__version__`` is the single source of truth for the agent version and its image
tag ``local-registry/control_plane:<__version__>``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from atrium.agents.control_plane.agent import ControlPlaneAgent
from atrium.agents.control_plane.protocol import (
    JOB_UPDATE_TYPE,
    KIND_SUBMIT,
    KIND_UPDATE,
    SUBMIT_TYPE,
    SUBMITTED_TYPE,
    SubmitRequest,
    build_job_update,
    build_submit_request,
    build_submitted_reply,
    parse_submit_request,
)

__all__ = [
    "ControlPlaneAgent",
    "SubmitRequest",
    "build_submit_request",
    "parse_submit_request",
    "build_submitted_reply",
    "build_job_update",
    "KIND_SUBMIT",
    "KIND_UPDATE",
    "SUBMIT_TYPE",
    "SUBMITTED_TYPE",
    "JOB_UPDATE_TYPE",
    "__version__",
]
