"""Tests for the Job value object and its execution-DAG builder (Prefect-free).

A job becomes runnable only when the request + generated flow pair is present and
the flow parses; ``build_execution_workboard`` turns a ready job into the
single review-gated node the trusted ``atrium-workboard`` flow runs. No backend.
"""

from __future__ import annotations

import json

import pytest

from atrium.orchestration import (
    DEFAULT_EXECUTOR_AGENT,
    Job,
    JobNotReadyError,
    build_execution_workboard,
    unsupported_requirements,
)

_FLOW = "from prefect import flow\n\n@flow\ndef main():\n    return 1\n"


def _job(**kw) -> Job:
    base = dict(id="job-1", request={"instruction": "do it"}, flow_source=_FLOW, params={"x": 1})
    base.update(kw)
    return Job(**base)


def test_ready_when_pair_complete_and_valid():
    assert _job().is_ready()


def test_not_ready_without_flow_source():
    assert not _job(flow_source="").is_ready()
    assert not _job(flow_source="   \n ").is_ready()


def test_not_ready_without_request():
    assert not _job(request={}).is_ready()


def test_not_ready_on_broken_syntax():
    job = _job(flow_source="def broken(:\n    pass")
    assert not job.is_ready()
    with pytest.raises(JobNotReadyError):
        job.static_check()


def test_static_check_never_executes_source():
    # A body that would raise if executed still passes the parse-only check —
    # proving ast.parse is parse-only, not exec. (main entrypoint present.)
    job = _job(flow_source="def main():\n    raise RuntimeError('should not run')\n")
    job.static_check()  # no RuntimeError
    assert job.is_ready()


def test_not_ready_without_main_entrypoint():
    # Valid Python, but no 'main' entrypoint the runner can invoke.
    assert not _job(flow_source="x = 1\n").is_ready()


def test_ready_with_async_main_entrypoint():
    assert _job(flow_source="async def main():\n    return 1\n").is_ready()


def test_to_from_dict_round_trip():
    job = _job(requirements=["prefect"], plan_reason="because")
    restored = Job.from_dict(job.to_dict())
    assert restored == job


def test_build_execution_workboard_shape():
    job = _job()
    wb = build_execution_workboard(job)
    wb.validate()  # already validated inside, but assert it stays well-formed
    assert wb.id == "job-1"
    assert [n.id for n in wb.nodes] == ["run_flow"]
    node = wb.nodes[0]
    assert node.agent == DEFAULT_EXECUTOR_AGENT
    assert node.reviewable is True
    assert node.payload["files"]["flow.py"] == _FLOW
    assert json.loads(node.payload["files"]["params.json"]) == {"x": 1}
    assert node.payload["commands"] == ["python flow.py"]


def test_build_execution_workboard_single_node_without_reviewer():
    wb = build_execution_workboard(_job())
    assert [n.id for n in wb.nodes] == ["run_flow"]
    assert wb.nodes[0].depends_on == []


def test_build_execution_workboard_prepends_source_review(monkeypatch):
    wb = build_execution_workboard(
        _job(), executor_agent="runner:active", reviewer_agent="flow_reviewer:active"
    )
    ids = [n.id for n in wb.nodes]
    assert ids == ["review_source", "run_flow"]
    review, run = wb.nodes
    # The reviewer node carries the flow source as the deliverable and is itself
    # NOT reviewable (its verdict is the outcome).
    assert review.agent == "flow_reviewer:active"
    assert review.reviewable is False
    assert review.payload["deliverable"] == _FLOW
    # run_flow depends on the review passing → a rejected flow never executes.
    assert run.agent == "runner:active"
    assert run.depends_on == ["review_source"]
    assert run.reviewable is True


def test_build_execution_workboard_rejects_unready_job():
    with pytest.raises(JobNotReadyError):
        build_execution_workboard(_job(flow_source=""))


# --------------------------------------------------------------------------- #
# Requirements allow-list                                                     #
# --------------------------------------------------------------------------- #
def test_unsupported_requirements_flags_libs_outside_allow_list():
    assert unsupported_requirements(["prefect", "requests"], ["prefect"]) == ["requests"]


def test_unsupported_requirements_normalizes_specs():
    # Version pins / extras / markers don't hide a covered package.
    assert unsupported_requirements(["prefect==3.7.8", "prefect[dask]"], ["prefect"]) == []


def test_unsupported_requirements_no_allow_list_flags_nothing():
    # No configured allow-list → don't gate (runtime import is the backstop).
    assert unsupported_requirements(["anything"], None) == []
    assert unsupported_requirements(["anything"], []) == []
