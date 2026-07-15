"""``TabbyLLMAgent`` — host-side inference agent that speaks **only A2A**.

TabbyLLMAgent manages the lifecycle of a GPU sandbox containing tabbyAPI (on the
exllamav3 backend) plus an in-sandbox A2A↔tabby bridge, and drives inference by
sending A2A messages to that bridge. It holds no OpenAI-HTTP / httpx code itself
— the single OpenAI-HTTP translation point is the bridge (``bridge/server.py``),
co-located with tabbyAPI on loopback.

Because the served model may still be quantizing, ``infer`` cooperatively waits
and retries on a ``not_ready`` status, surfacing :class:`ModelNotReadyError` if
it never becomes available.

    BaseAgent → InferenceAgent → TabbyLLMAgent
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any, Optional

from atrium.agents.inference_agent import InferenceAgent, InferenceSettings
from atrium.agents.prompt_memory import PromptLayer, PromptMemory, default_prompt_memory
from atrium.agents.tabby_llm_agent.cache import KVCacheConfig
from atrium.core.errors import ModelNotReadyError, PolicyViolationError
from atrium.core.types import SandboxConfig, VersionTag
from atrium.protocol import (
    Message,
    Role,
    data_part,
    get_message_data,
    get_message_text,
    metadata_dict,
    text_message,
)

logger = logging.getLogger("atrium.agents.tabby")

# --------------------------------------------------------------------------- #
# Shared A2A "kinds"/"statuses" agreed between the agent and the bridge.       #
# --------------------------------------------------------------------------- #
KIND_INFER = "infer"
KIND_READY = "ready?"
KIND_LOAD = "load"
KIND_UNLOAD = "unload"

STATUS_OK = "ok"
STATUS_NOT_READY = "not_ready"
STATUS_TOOL_CALLS = "tool_calls"

__all__ = [
    "TabbyLLMAgent",
    "TabbyAgentConfig",
    "coding_agent_settings",
    "coding_agent_prompt_memory",
]


def coding_agent_settings() -> InferenceSettings:
    """Default :class:`InferenceSettings` tuned to drive Ornith-1.0-35B (EXL3
    4bpw) as a coding agent on a 24 GB RTX 3090.

    Rationale for this hardware/model:

    * **32k context** — the model supports 256k, and its KV cache is unusually
      cheap here (only 10 of 40 layers are full-attention; 2 KV heads), so a
      large coding context costs only a few hundred MB of the ~6 GB left after
      the 4bpw weights. 32k keeps prefill latency sane on a 3090 while holding
      several files of context.
    * **4096 output tokens** — enough to emit a whole file or a multi-function
      patch in one turn.
    * **temperature 0.2 / top_p 0.95** — near-deterministic, the right register
      for code edits and tool calls.
    * **compaction on, keep last 6 turns** — long coding sessions stay within
      the window by summarizing older turns while preserving recent context.
    """
    return InferenceSettings(
        max_output_tokens=4096,
        temperature=0.2,
        top_p=0.95,
        context_window_tokens=32768,
        compaction_enabled=True,
        compaction_trigger_ratio=0.8,
        compaction_keep_last_turns=6,
        compaction_summary_max_tokens=1024,
        compaction_summary_temperature=0.2,
    )


def coding_agent_prompt_memory() -> PromptMemory:
    """Default layered system prompt for the coding-agent preset.

    Built on :func:`~atrium.agents.prompt_memory.default_prompt_memory`'s
    canonical order, with the *stable* layers (identity / tone / tool_guidance /
    rules) filled with coding-agent guidance and the *volatile* layers
    (capabilities / memory / environment / objective / user_instructions) left
    empty for the caller to populate per turn (empties are skipped on compose).
    The ``tools`` layer renders provided tool schemas as a ``<tools>`` block.

    This is to the system prompt what :func:`coding_agent_settings` is to the
    generation knobs: the tuned default a bare ``TabbyLLMAgent`` boots with.
    """
    memory = default_prompt_memory()
    memory.record(
        PromptLayer(
            "identity",
            order=10,
            content=(
                "You are a focused coding agent working inside an isolated, "
                "WAN-cut-off sandbox. Produce correct, minimal, well-targeted "
                "changes."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "tone",
            order=20,
            content=(
                "Be concise and direct. Prefer concrete actions and code over "
                "prose, and match the conventions of the surrounding codebase."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "tool_guidance",
            order=30,
            content=(
                "Prefer the provided tools over ad-hoc shell when a tool fits. "
                "Call one tool at a time and wait for its result before the next. "
                "Read a file before editing it."
            ),
        )
    )
    memory.record(
        PromptLayer(
            "rules",
            order=60,
            content=(
                "Keep changes minimal and reversible; avoid unrelated edits. "
                "Verify your work by running tests when possible. Stay within the "
                "sandbox — never attempt network egress or data exfiltration."
            ),
        )
    )
    return memory


@dataclass(slots=True)
class TabbyAgentConfig:
    """Host-side *connection* configuration for the in-sandbox bridge (A2A).

    Generation/context knobs live in :class:`InferenceSettings` (see
    :func:`coding_agent_settings`); this is deliberately small and HTTP-free —
    just where the bridge is and how patiently to wait for the model.
    """

    bridge_url: Optional[str] = None
    bridge_port: int = 8730
    model_name: Optional[str] = None
    max_retries: int = 60
    retry_backoff_s: float = 2.0

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "TabbyAgentConfig":
        """Build connection config from a mapping (e.g. a YAML ``tabby:`` block)."""
        if not data:
            return cls()
        known = {f.name for f in dc_fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown TabbyAgentConfig field(s): {sorted(unknown)}; "
                f"valid keys are {sorted(known)}"
            )
        return cls(**dict(data))


class TabbyLLMAgent(InferenceAgent):
    """Concrete inference agent backed by tabbyAPI/exllamav3, spoken to via A2A."""

    AGENT_SLUG = "tabby_llm_agent"

    def __init__(
        self,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        config: Optional[TabbyAgentConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        settings: Optional[InferenceSettings] = None,
        prompt_memory: Optional[PromptMemory] = None,
    ) -> None:
        # Lazy imports keep version/sandbox wiring inside the package directory,
        # so the agent's version and its image tag share one source of truth.
        from atrium.agents.tabby_llm_agent import __version__
        from atrium.agents.tabby_llm_agent.sandbox import build_sandbox_config

        version = version or __version__
        sandbox_config = sandbox_config or build_sandbox_config(str(version))

        self.config = config or TabbyAgentConfig()
        # Default to coding-agent-tuned settings *and* the coding-agent layered
        # system prompt for this machine/model; callers can pass their own
        # ``settings=`` / ``prompt_memory=`` or load them from YAML (from_yaml).
        # A client (see ``connect``) shares another agent's backend and never
        # starts a sandbox, so it legitimately needs no GPU of its own.
        super().__init__(
            agent_id,
            version,
            sandbox_config,
            require_gpu=not self.config.bridge_url,
            settings=settings or coding_agent_settings(),
            prompt_memory=prompt_memory or coding_agent_prompt_memory(),
        )
        self._model_ready = False

    @classmethod
    def from_yaml(
        cls,
        path: str,
        agent_id: str,
        version: "str | VersionTag | None" = None,
        *,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> "TabbyLLMAgent":
        """Construct an agent from a YAML file with ``tabby:``, ``inference:`` and
        ``prompt:`` sections. Inference keys override the coding-agent defaults;
        any omitted key keeps its tuned default. A ``prompt:`` block fully defines
        the layered system prompt (see
        :class:`~atrium.agents.prompt_memory.PromptMemory`); when it is omitted the
        coding-agent default (:func:`coding_agent_prompt_memory`) is kept. Any
        section may be absent.

        Example::

            agent = TabbyLLMAgent.from_yaml("tabby_agent.yaml", "coder-1")
        """
        import yaml

        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, Mapping):
            raise ValueError(f"{path}: top-level YAML must be a mapping")
        config = TabbyAgentConfig.from_mapping(doc.get("tabby"))
        settings = coding_agent_settings().merge(doc.get("inference"))
        # Omitted ``prompt:`` -> None -> constructor keeps the coding-agent default.
        prompt_memory = PromptMemory.from_mapping(doc["prompt"]) if "prompt" in doc else None
        return cls(
            agent_id,
            version,
            config=config,
            sandbox_config=sandbox_config,
            settings=settings,
            prompt_memory=prompt_memory,
        )

    @classmethod
    def connect(
        cls,
        agent_id: str,
        bridge_url: str,
        *,
        model_name: Optional[str] = None,
        version: "str | VersionTag | None" = None,
        settings: Optional[InferenceSettings] = None,
        prompt_memory: Optional[PromptMemory] = None,
    ) -> "TabbyLLMAgent":
        """Build a *client-mode* agent that shares an already-loaded backend.

        The returned agent owns no sandbox and loads no model: ``start_sandbox``
        is a no-op and inference is sent over A2A to ``bridge_url`` (another
        agent's :meth:`bridge_url`). This is how several agents — coder,
        reviewer, tester — fan into one GPU-resident model, which tabbyAPI serves
        concurrently via continuous batching. The weights are paid for once; each
        extra client costs only its slice of the shared KV cache.

        Pass ``model_name`` when the backend has more than one model available so
        requests target the right one; otherwise the backend's loaded model is
        used. ``settings`` / ``prompt_memory`` default to the coding-agent preset,
        exactly as for a backend-owning agent.
        """
        config = TabbyAgentConfig(bridge_url=bridge_url, model_name=model_name)
        return cls(
            agent_id,
            version,
            config=config,
            settings=settings,
            prompt_memory=prompt_memory,
        )

    @property
    def is_client(self) -> bool:
        """True when this agent shares another's backend instead of owning one.

        Set implicitly by a configured ``bridge_url`` (see :meth:`connect`). A
        client never starts a sandbox and must not load or unload the shared
        model — those stay the backend owner's responsibility.
        """
        return bool(self.config.bridge_url)

    def bridge_url(self) -> str:
        """The A2A base URL other agents pass to :meth:`connect` to share this
        agent's backend. For a client, this is just the backend it points at."""
        return self._bridge_target()

    # ------------------------------------------------------------------ #
    # Sandbox lifecycle: bring up tabbyAPI + bridge, then resolve the card #
    # ------------------------------------------------------------------ #
    async def start_sandbox(self) -> None:
        # Client-mode agents share a backend they don't own: no sandbox, no GPU,
        # no backend launch — just talk to the shared bridge over A2A.
        if self.is_client:
            logger.debug(
                "%s in client mode; sharing backend at %s (no sandbox started)",
                self.agent_id,
                self.config.bridge_url,
            )
            return
        await super().start_sandbox()
        await self._launch_backend()

    async def _launch_backend(self) -> None:
        """Start tabbyAPI (exllamav3) and the A2A bridge inside the sandbox.

        The container image's entrypoint already launches both; this is a
        best-effort guard for images started with a bare shell. Failures are
        non-fatal (the entrypoint may already own the processes).
        """
        port = self.config.bridge_port
        command = (
            "pgrep -f tabbyAPI/main.py >/dev/null 2>&1 || "
            "nohup python3 /opt/tabbyAPI/main.py --host 127.0.0.1 --port 5000 "
            ">/tmp/tabby.log 2>&1 & "
            f"pgrep -f atrium.agents.tabby_llm_agent.bridge.server >/dev/null 2>&1 || "
            f"BRIDGE_PORT={port} nohup python3 -m atrium.agents.tabby_llm_agent.bridge.server "
            ">/tmp/bridge.log 2>&1 &"
        )
        try:
            await self.execute_in_sandbox(command)
        except Exception:  # noqa: BLE001 - container entrypoint may already run them
            logger.debug("backend launch guard failed (entrypoint may own it)", exc_info=True)

    def _bridge_target(self) -> str:
        """The A2A base URL of the in-sandbox bridge."""
        if self.config.bridge_url:
            return self.config.bridge_url
        # Host-local address of this agent's sandbox bridge.
        return f"http://{self.agent_id}.local:{self.config.bridge_port}"

    # ------------------------------------------------------------------ #
    # Readiness / model control (all over A2A)                           #
    # ------------------------------------------------------------------ #
    async def is_model_ready(self) -> bool:
        """Ask the bridge (over A2A) whether a model is currently servable."""
        reply = await self._control(KIND_READY)
        status = metadata_dict(reply).get("status")
        self._model_ready = status == STATUS_OK
        return self._model_ready

    async def wait_until_ready(self, timeout: Optional[float] = None) -> None:
        """Poll readiness over A2A until a model is loaded (or ``timeout``)."""
        waited = 0.0
        while True:
            if await self.is_model_ready():
                return
            if timeout is not None and waited >= timeout:
                raise ModelNotReadyError(
                    f"model not ready after {timeout}s (still quantizing/loading?)"
                )
            await asyncio.sleep(self.config.retry_backoff_s)
            waited += self.config.retry_backoff_s

    async def load_model(
        self,
        model_name: Optional[str] = None,
        *,
        cache: Optional[KVCacheConfig] = None,
        **options: Any,
    ) -> None:
        """Tell the bridge to load ``model_name`` once quantization is complete.

        ``cache`` tunes the shared KV cache (quantization mode, pool size, batch
        cap) — size it with :class:`~atrium.agents.tabby_llm_agent.cache.KVCacheConfig`
        for the number of client agents you intend to fan in. Explicit
        ``**options`` win over ``cache`` on key clashes. Loading is a
        backend-owner operation; a client-mode agent must not call it.
        """
        self._require_backend_owner("load_model")
        payload: dict[str, Any] = {"model_name": model_name or self.config.model_name}
        if cache is not None:
            payload.update(cache.load_options())
        payload.update(options)
        await self._control(KIND_LOAD, payload)
        if model_name:
            self.config.model_name = model_name

    async def unload_model(self) -> None:
        """Tell the bridge to unload the current model (backend owner only)."""
        self._require_backend_owner("unload_model")
        await self._control(KIND_UNLOAD)
        self._model_ready = False

    def _require_backend_owner(self, op: str) -> None:
        """Refuse a model-mutating op on a client that shares someone's backend.

        A client's ``load``/``unload`` would swap the model out from under every
        other agent fanned into the same backend — an isolation invariant, so it
        raises :class:`PolicyViolationError`. Read-only readiness checks stay open
        to clients (they legitimately wait for the shared model to come up).
        """
        if self.is_client:
            raise PolicyViolationError(
                f"{op} is a backend-owner operation; {self.agent_id} is a client "
                f"sharing the model at {self.config.bridge_url} and must not "
                f"mutate it (loading/unloading affects every fanned-in agent)"
            )

    async def _control(self, kind: str, payload: Optional[dict[str, Any]] = None) -> Message:
        message = text_message(
            "",
            role=Role.ROLE_USER,
            metadata={"kind": kind},
            extra_parts=[data_part({"type": kind, **(payload or {})})],
        )
        return await self.send_a2a_message(self._bridge_target(), message)

    # ------------------------------------------------------------------ #
    # Inference (the InferenceAgent extension point)                     #
    # ------------------------------------------------------------------ #
    async def infer(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Any = None,
        **params: Any,
    ) -> str:
        """Run a single-prompt chat completion via the bridge (over A2A).

        Returns the assistant text. When the model elects to call tools, returns
        a JSON string of the ``tool_calls`` so the caller can execute them and
        continue via :meth:`chat`. Retries on ``not_ready`` (model quantizing).

        When ``system`` is omitted, the agent's layered ``prompt_memory`` is
        composed into the system message (a no-op when the memory is empty).
        """
        system = self.build_system_prompt(system, tools=tools)
        request: dict[str, Any] = {
            "type": KIND_INFER,
            "messages": _to_messages(prompt, system),
            **self.resolve_generation_params(
                max_tokens=max_tokens, temperature=temperature, **params
            ),
        }
        if self.config.model_name:
            request["model"] = self.config.model_name
        if tools:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        reply = await self._infer_with_retry(request)
        status = metadata_dict(reply).get("status")
        if status == STATUS_TOOL_CALLS:
            return json.dumps(get_message_data(reply))
        return get_message_text(reply)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Any = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **params: Any,
    ) -> Message:
        """Multi-turn chat (incl. ``role:"tool"`` results); returns the reply.

        Unlike :meth:`infer`, this returns the full A2A reply ``Message`` so the
        caller can inspect ``tool_calls`` structurally and drive a tool loop.

        Oversized histories are compacted (older turns summarized) before the
        request is sent, per the agent's :class:`InferenceSettings`.

        When the history carries no ``system`` turn, the agent's layered
        ``prompt_memory`` is composed and prepended as one (a no-op when the
        memory is empty); an existing system turn is always left untouched.
        """
        if not any(m.get("role") == "system" for m in messages):
            system = self.build_system_prompt(None, tools=tools)
            if system:
                messages = [{"role": "system", "content": system}, *messages]
        messages = await self.compact_messages(messages)
        request: dict[str, Any] = {
            "type": KIND_INFER,
            "messages": messages,
            **self.resolve_generation_params(
                max_tokens=max_tokens, temperature=temperature, **params
            ),
        }
        if self.config.model_name:
            request["model"] = self.config.model_name
        if tools:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        return await self._infer_with_retry(request)

    async def _infer_with_retry(self, request: dict[str, Any]) -> Message:
        last_status: Optional[str] = None
        for _attempt in range(self.config.max_retries):
            message = text_message(
                "",
                role=Role.ROLE_USER,
                metadata={"kind": KIND_INFER},
                extra_parts=[data_part(request)],
            )
            reply = await self.send_a2a_message(self._bridge_target(), message)
            last_status = metadata_dict(reply).get("status")
            if last_status == STATUS_NOT_READY:
                await asyncio.sleep(self.config.retry_backoff_s)
                continue
            return reply
        raise ModelNotReadyError(
            f"tabby model not ready after {self.config.max_retries} attempts "
            f"(last status: {last_status})"
        )


def _to_messages(prompt: str, system: Optional[str]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages
