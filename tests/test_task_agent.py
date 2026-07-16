"""Tests for the TaskAgent self-evolution driver and the SlackTaskAgent gateway.

No sandbox or network: the outward A2A seam (``send_a2a_message``) is scripted,
so the author → build → retry loop, the version-decide and the reply envelope are
exercised in-process against :class:`DelegatingTaskAgent`, while the Slack
normalization + forward/format behaviour is exercised against the
:class:`SlackTaskAgent` gateway. One test routes the build request into a *real*
:class:`BuilderAgent` (with its sandbox mocked) to prove the request/result schema
the two agents exchange actually matches, and one drives the full
Slack → gateway → DelegatingTaskAgent → builder chain end to end.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from atrium.agents.builder_agent import BuilderAgent
from atrium.agents.builder_agent.agent import RESULT_TYPE, STATUS_ERROR, STATUS_OK
from atrium.agents.task_agent import (
    BuildFailedError,
    DelegatingTaskAgent,
    GenerationRequest,
    SlackTaskAgent,
)
from atrium.agents.task_agent.agent import TASK_RESULT_TYPE
from atrium.core.errors import AgentError, PolicyViolationError
from atrium.core.types import ExecutionResult, NetworkMode, SandboxConfig
from atrium.protocol import (
    Role,
    data_message,
    data_part,
    get_message_data,
    get_message_text,
    metadata_dict,
    text_message,
)

_DOCKERFILE = "FROM scratch\n"


def _gen(**overrides):
    kwargs = dict(
        target_name="widget_agent",
        files={"Dockerfile": _DOCKERFILE, "app.py": "print(1)\n"},
    )
    kwargs.update(overrides)
    return GenerationRequest(**kwargs)


def _fixed_author(gen):
    async def author(task, attempt, last_outcome):
        return gen

    return author


def _task_agent(author, **kwargs):
    return DelegatingTaskAgent(
        "task-1", "0.1.0", builder="http://builder.local", author=author, **kwargs
    )


def _slack_gateway(downstream, **kwargs):
    return SlackTaskAgent("slack-1", "0.1.0", downstream=downstream, **kwargs)


def _script_builder(agent, replies):
    """Script the builder replies; record each outbound build request payload."""
    requests: list = []
    counter = {"i": 0}

    async def fake_send(target, message):
        requests.append(message)
        i = min(counter["i"], len(replies) - 1)
        counter["i"] += 1
        return replies[i]

    agent.send_a2a_message = fake_send  # type: ignore[assignment]
    return requests


def _builder_ok(image="local-registry/widget_agent:0.1.0", digest="sha256:abc"):
    return data_message(
        {
            "type": RESULT_TYPE,
            "status": STATUS_OK,
            "image": image,
            "digest": digest,
            "image_ref": f"local-registry/widget_agent@{digest}",
        },
        role=Role.ROLE_AGENT,
        metadata={"kind": "build", "status": STATUS_OK},
    )


def _builder_err(reason="kaniko build failed", logs="boom"):
    return data_message(
        {"type": RESULT_TYPE, "status": STATUS_ERROR, "reason": reason, "stdout_tail": logs},
        role=Role.ROLE_AGENT,
        metadata={"kind": "build", "status": STATUS_ERROR},
    )


def _task_result_ok(target="widget_agent", version="0.1.0", digest="sha256:abc"):
    """What a downstream :class:`DelegatingTaskAgent` replies on success."""
    return data_message(
        {
            "type": TASK_RESULT_TYPE,
            "status": STATUS_OK,
            "target_name": target,
            "version": version,
            "image": f"local-registry/{target}:{version}",
            "digest": digest,
            "image_ref": f"local-registry/{target}@{digest}",
        },
        role=Role.ROLE_AGENT,
        metadata={"kind": "task", "status": STATUS_OK},
    )


def _task_result_err(reason="nope"):
    return data_message(
        {"type": TASK_RESULT_TYPE, "status": STATUS_ERROR, "reason": reason},
        role=Role.ROLE_AGENT,
        metadata={"kind": "task", "status": STATUS_ERROR},
    )


def _slack_request(text="<@U01BOT> build a widget"):
    return text_message(
        "", role=Role.ROLE_USER, extra_parts=[data_part({"event": {"text": text, "user": "U9", "channel": "C1"}})]
    )


def _task_request(instruction="build a widget"):
    return data_message({"instruction": instruction}, role=Role.ROLE_USER)


# --------------------------------------------------------------------------- #
# GenerationRequest.build_payload                                              #
# --------------------------------------------------------------------------- #
def test_build_payload_shape_and_version():
    gen = _gen(build_args={"V": "1"})
    payload = gen.build_payload("0.3.0")
    assert payload["target_name"] == "widget_agent"
    assert payload["target_version"] == "0.3.0"
    assert payload["dockerfile"] == "Dockerfile"
    assert payload["build_args"] == {"V": "1"}
    assert payload["files"]["app.py"] == "print(1)\n"


def test_build_payload_base64_wraps_bytes():
    gen = _gen(files={"Dockerfile": _DOCKERFILE, "blob.bin": b"\x00\x01\x02"})
    payload = gen.build_payload("0.1.0")
    blob = payload["files"]["blob.bin"]
    assert blob["encoding"] == "base64"
    assert base64.b64decode(blob["content"]) == b"\x00\x01\x02"


# --------------------------------------------------------------------------- #
# Security envelope                                                            #
# --------------------------------------------------------------------------- #
def test_task_agent_envelope_is_wan_capable_no_socket():
    agent = _task_agent(_fixed_author(_gen()))
    assert agent.sandbox_config.network is NetworkMode.BRIDGE


def test_task_agent_rejects_docker_socket():
    cfg = SandboxConfig(volumes={"/var/run/docker.sock": "/var/run/docker.sock"})
    with pytest.raises(PolicyViolationError):
        _task_agent(_fixed_author(_gen()), sandbox_config=cfg)


# --------------------------------------------------------------------------- #
# Happy path: author -> build -> success reply                                 #
# --------------------------------------------------------------------------- #
def test_handle_task_builds_and_replies_ok():
    agent = _task_agent(_fixed_author(_gen()))
    requests = _script_builder(agent, [_builder_ok()])
    reply = asyncio.run(agent.dispatch(_task_request()))

    meta = metadata_dict(reply)
    assert meta["status"] == STATUS_OK
    payload = get_message_data(reply)[0]
    assert payload["digest"] == "sha256:abc"
    assert payload["target_name"] == "widget_agent"
    assert "widget_agent:0.1.0" in get_message_text(reply)

    # The outbound build request carries BuilderAgent's expected schema.
    sent = get_message_data(requests[0])[0]
    assert sent["target_name"] == "widget_agent"
    assert sent["target_version"] == "0.1.0"
    assert "Dockerfile" in sent["files"]


# --------------------------------------------------------------------------- #
# Retry: first build fails, author fixes, second succeeds                       #
# --------------------------------------------------------------------------- #
def test_retries_and_passes_last_outcome_to_author():
    seen: list = []

    async def author(task, attempt, last_outcome):
        seen.append((attempt, last_outcome))
        return _gen()

    agent = _task_agent(author)
    _script_builder(agent, [_builder_err(logs="syntax error"), _builder_ok()])
    outcome = asyncio.run(agent.build_generation({"instruction": "x"}))

    assert outcome.ok is True
    # Two attempts; the 2nd received the 1st's failed outcome (with logs).
    assert [a for a, _ in seen] == [1, 2]
    assert seen[0][1] is None
    assert seen[1][1] is not None and seen[1][1].logs == "syntax error"


def test_gives_up_after_max_attempts():
    agent = _task_agent(_fixed_author(_gen()), max_build_attempts=2)
    _script_builder(agent, [_builder_err()])
    with pytest.raises(BuildFailedError, match="after 2 attempt"):
        asyncio.run(agent.build_generation({"instruction": "x"}))


def test_handle_task_reports_failure_without_raising():
    agent = _task_agent(_fixed_author(_gen()), max_build_attempts=1)
    _script_builder(agent, [_builder_err(reason="nope")])
    reply = asyncio.run(agent.dispatch(_task_request()))
    assert metadata_dict(reply)["status"] == STATUS_ERROR
    assert "nope" in get_message_data(reply)[0]["reason"]


# --------------------------------------------------------------------------- #
# Version decide                                                               #
# --------------------------------------------------------------------------- #
def test_version_defaults_to_initial_without_registry():
    agent = _task_agent(_fixed_author(_gen()))
    assert agent._decide_version(_gen()) == "0.1.0"


def test_version_pin_is_respected():
    agent = _task_agent(_fixed_author(_gen()))
    assert agent._decide_version(_gen(version="2.5.0")) == "2.5.0"


def test_version_bumps_off_ledger(monkeypatch):
    from atrium.agents.task_agent import agent as agent_mod

    class StubClient:
        def __init__(self, endpoint):
            pass

        def versions(self, slug):
            return ["0.1.0", "0.2.0"]

    monkeypatch.setattr(agent_mod, "RegistryClient", StubClient)
    agent = _task_agent(_fixed_author(_gen()), registry_endpoint="127.0.0.1:5000")
    assert agent._decide_version(_gen(version_bump="minor")) == "0.3.0"
    assert agent._decide_version(_gen(version_bump="patch")) == "0.2.1"


def test_author_required_when_not_overridden():
    agent = _task_agent(author=None)
    with pytest.raises(AgentError, match="no code author"):
        asyncio.run(agent.author_generation({"instruction": "x"}, attempt=1, last_outcome=None))


# --------------------------------------------------------------------------- #
# SlackTaskAgent gateway: Slack normalization (pure)                            #
# --------------------------------------------------------------------------- #
def test_normalize_strips_mention():
    task = SlackTaskAgent.normalize_slack({"event": {"text": "<@U01> do it", "user": "U9"}})
    assert task["instruction"] == "do it"
    assert task["user"] == "U9"
    assert task["source"] == "slack"


def test_normalize_slash_command():
    task = SlackTaskAgent.normalize_slack(
        {"command": "/build", "text": "make a thing", "user_id": "U5", "channel_id": "C2"}
    )
    assert task["instruction"] == "make a thing"
    assert task["user"] == "U5"
    assert task["channel"] == "C2"


def test_normalize_rejects_empty():
    with pytest.raises(AgentError, match="no instruction"):
        SlackTaskAgent.normalize_slack({"event": {"text": "<@U01>   "}})


# --------------------------------------------------------------------------- #
# SlackTaskAgent gateway: envelope + forward + format                          #
# --------------------------------------------------------------------------- #
def test_gateway_envelope_is_wan_capable_no_socket():
    agent = _slack_gateway("http://downstream.local")
    assert agent.sandbox_config.network is NetworkMode.BRIDGE


def test_gateway_rejects_docker_socket():
    cfg = SandboxConfig(volumes={"/var/run/docker.sock": "/var/run/docker.sock"})
    with pytest.raises(PolicyViolationError):
        _slack_gateway("http://downstream.local", sandbox_config=cfg)


def test_gateway_forwards_normalized_task_and_formats_ok():
    agent = _slack_gateway("http://downstream.local")
    captured: list = []

    async def fake_send(target, message):
        captured.append((target, message))
        return _task_result_ok()

    agent.send_a2a_message = fake_send  # type: ignore[assignment]
    reply = asyncio.run(agent.dispatch(_slack_request()))

    # It forwarded to the downstream target with the normalized task.
    target, forwarded = captured[0]
    assert target == "http://downstream.local"
    assert get_message_text(forwarded) == "build a widget"
    fwd_data = get_message_data(forwarded)[0]
    assert fwd_data["instruction"] == "build a widget"
    assert fwd_data["source"] == "slack"
    assert fwd_data["user"] == "U9"

    # The reply is Slack-flavored, and echoes the structured result.
    assert metadata_dict(reply)["status"] == STATUS_OK
    text = get_message_text(reply)
    assert ":white_check_mark:" in text
    assert "widget_agent:0.1.0" in text
    assert get_message_data(reply)[0]["digest"] == "sha256:abc"


def test_gateway_formats_downstream_error():
    agent = _slack_gateway("http://downstream.local")

    async def fake_send(target, message):
        return _task_result_err(reason="kaniko exploded")

    agent.send_a2a_message = fake_send  # type: ignore[assignment]
    reply = asyncio.run(agent.dispatch(_slack_request()))

    assert metadata_dict(reply)["status"] == STATUS_ERROR
    text = get_message_text(reply)
    assert ":x:" in text
    assert "kaniko exploded" in text


def test_gateway_end_to_end_through_delegating_task_agent():
    """Slack → gateway → DelegatingTaskAgent → (scripted) builder, one chain."""
    task_agent = _task_agent(_fixed_author(_gen()))
    _script_builder(task_agent, [_builder_ok()])
    gateway = _slack_gateway(task_agent)

    async def route(target, message):
        # Exactly what an A2A hop would deliver to the downstream's handler.
        return await target.dispatch(message)

    gateway.send_a2a_message = route  # type: ignore[assignment]
    reply = asyncio.run(gateway.dispatch(_slack_request()))

    assert metadata_dict(reply)["status"] == STATUS_OK
    assert "widget_agent:0.1.0" in get_message_text(reply)


# --------------------------------------------------------------------------- #
# End-to-end schema compatibility with a REAL BuilderAgent (sandbox mocked)     #
# --------------------------------------------------------------------------- #
def test_roundtrip_against_real_builder_agent():
    builder = BuilderAgent("builder-1", "0.1.0")

    async def fake_start():
        return None

    async def fake_write(files, dest, *, clean=False):
        return ExecutionResult(command="stage", exit_code=0)

    async def fake_exec(command, *, timeout=None):
        if command.startswith("cat "):
            return ExecutionResult(command=command, exit_code=0, stdout="sha256:feed\n")
        return ExecutionResult(command=command, exit_code=0, stdout="built")

    builder.start_sandbox = fake_start  # type: ignore[assignment]
    builder.write_files_to_sandbox = fake_write  # type: ignore[assignment]
    builder.execute_in_sandbox = fake_exec  # type: ignore[assignment]

    agent = _task_agent(_fixed_author(_gen()))

    async def route_to_builder(target, message):
        # Exactly what an A2A hop would deliver to the builder's handler.
        return await builder.handle_task(message)

    agent.send_a2a_message = route_to_builder  # type: ignore[assignment]

    outcome = asyncio.run(agent.build_generation({"instruction": "x"}))
    assert outcome.ok is True
    assert outcome.digest == "sha256:feed"
    assert outcome.image == "local-registry/widget_agent:0.1.0"
    assert outcome.image_ref == "local-registry/widget_agent@sha256:feed"
