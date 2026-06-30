"""Layered system-prompt construction for inference agents.

An inference agent's system prompt is rarely one monolithic string: it is an
*assembly of reusable, ordered sections* — an identity blurb, tone/format rules,
tool-use guidance, the tool definitions themselves, project memory, environment
context, the current objective, user overrides. This module lets an agent
**record** those sections into a small in-memory registry as named *layers* and
**compose** them into the final prompt, with the kept-layers and their order
configurable (incl. from YAML).

Design lineage (verified prior art):

* **Hermes** (`Hermes-Function-Calling`'s ``PromptManager``) — a YAML of ordered
  named sections, each substituted over a context, with empty sections skipped.
  Tool *definitions* live in their own section, separate from tool-use *prose*.
* **Coding agents** (Roo Code) — the system prompt is an ordered list of
  section-generator functions, each returning one string, concatenated.
* **Letta/MemGPT memory blocks** — labelled, individually editable blocks
  recompiled into the prompt each turn; the inspiration for ``read_only`` and the
  "record into memory in layers" framing.

The composed string is returned verbatim and handed to
:meth:`~atrium.agents.inference_agent.InferenceAgent.build_system_prompt`, which
the inference agents pass as the ``system`` message — so this is a pure host-side
addition with no change to the A2A bridge or the wire format.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional

__all__ = [
    "RenderFn",
    "PromptLayer",
    "PromptMemory",
    "render_tools_block",
    "tools_layer",
    "default_prompt_memory",
]

#: A dynamic section body: receives the compose-time context, returns its text.
RenderFn = Callable[[Mapping[str, Any]], str]


@dataclass(slots=True)
class PromptLayer:
    """One named, ordered section of the system prompt (a "memory layer").

    The body is resolved with a fixed precedence — :attr:`render` (dynamic) wins
    over :attr:`template` (``str.format`` over the compose context) which wins
    over the static :attr:`content`. An empty body causes the layer to be
    dropped during composition (Hermes-style conditional inclusion).
    """

    name: str
    """Unique id / section label, e.g. ``"identity"`` or ``"tools"``."""
    content: str = ""
    """Static body, used when neither :attr:`render` nor :attr:`template` is set."""
    order: int = 100
    """Ascending sort key (lower = earlier). Ties keep insertion order. Only
    consulted for layers *not* pinned by :attr:`PromptMemory.order`."""
    enabled: bool = True
    """When False the layer is skipped entirely."""
    title: Optional[str] = None
    """Optional header line emitted above the body."""
    template: Optional[str] = None
    """A ``str.format`` template evaluated over the compose context."""
    render: Optional[RenderFn] = None
    """A callable producing the body from the compose context (highest priority).
    Not serializable to YAML; built in code (see :func:`tools_layer`)."""
    read_only: bool = False
    """Letta-inspired marker for policy/config layers that in-prompt memory tools
    must not edit. Informational here (composition ignores it)."""

    def body(self, ctx: Mapping[str, Any]) -> str:
        """Resolve the layer's text: ``render`` > ``template`` > ``content``."""
        if self.render is not None:
            return self.render(ctx)
        if self.template is not None:
            return self.template.format(**ctx)
        return self.content

    # --- YAML / dict construction --------------------------------------- #
    @classmethod
    def _yaml_fields(cls) -> frozenset[str]:
        # ``render`` is intentionally excluded — closures aren't YAML-expressible.
        return frozenset(f.name for f in fields(cls) if f.name not in ("name", "render"))

    @classmethod
    def from_mapping(cls, name: str, data: Optional[Mapping[str, Any]]) -> "PromptLayer":
        """Build a static/templated layer from a mapping (unknown keys raise)."""
        data = dict(data or {})
        unknown = set(data) - cls._yaml_fields()
        if unknown:
            raise ValueError(
                f"unknown PromptLayer field(s) for {name!r}: {sorted(unknown)}; "
                f"valid keys are {sorted(cls._yaml_fields())}"
            )
        return cls(name=name, **data)


@dataclass(slots=True)
class PromptMemory:
    """A registry of :class:`PromptLayer` plus the order they compose in.

    "Recording into memory" is :meth:`record` (add/replace a layer by name);
    "synthesizing" is :meth:`compose`. :attr:`order` is the configurable
    sequence — names listed there compose first, in that exact order; any
    remaining layers follow, sorted by their numeric :attr:`PromptLayer.order`.
    """

    layers: dict[str, PromptLayer] = field(default_factory=dict)
    order: tuple[str, ...] = ()
    sep: str = "\n\n"

    # --- recording ------------------------------------------------------ #
    def record(self, layer: PromptLayer) -> "PromptMemory":
        """Add ``layer`` (or replace any existing layer of the same name)."""
        self.layers[layer.name] = layer
        return self

    def remove(self, name: str) -> "PromptMemory":
        """Drop the layer called ``name`` if present (no-op otherwise)."""
        self.layers.pop(name, None)
        return self

    # --- ordering / composition ----------------------------------------- #
    def _sequence(self) -> list[PromptLayer]:
        """Layers in render order: ``order``-pinned names first, then the rest
        by ``PromptLayer.order`` (stable, so ties keep insertion order)."""
        pinned = [self.layers[n] for n in self.order if n in self.layers]
        seen = set(self.order)
        rest = [layer for name, layer in self.layers.items() if name not in seen]
        rest.sort(key=lambda layer: layer.order)
        return pinned + rest

    def compose(
        self,
        ctx: Optional[Mapping[str, Any]] = None,
        *,
        include: Optional[set[str]] = None,
        exclude: Optional[set[str]] = None,
    ) -> str:
        """Synthesize the final system prompt.

        Iterates the layers in :meth:`_sequence` order, dropping any that are
        disabled, filtered out by ``include``/``exclude``, or resolve to an empty
        body. Each kept layer contributes its (optionally titled) body, joined by
        :attr:`sep`.
        """
        ctx = ctx or {}
        out: list[str] = []
        for layer in self._sequence():
            if not layer.enabled:
                continue
            if include is not None and layer.name not in include:
                continue
            if exclude and layer.name in exclude:
                continue
            text = layer.body(ctx).strip()
            if not text:
                continue
            out.append(f"{layer.title}\n{text}" if layer.title else text)
        return self.sep.join(out)

    # --- YAML / dict construction --------------------------------------- #
    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "PromptMemory":
        """Build from a mapping (e.g. a YAML ``prompt:`` block).

        Shape::

            order: [identity, tools, rules]    # the kept-and-ordered names
            sep: "\\n\\n"
            layers:
              identity: {order: 10, content: "You are ..."}
              tools:    {order: 40, mode: json_schema}   # -> tools_layer()
              rules:    {order: 60, content: "Read before editing."}

        A layer mapping carrying a ``mode`` key (or ``tools: true``) is built via
        :func:`tools_layer`; all others via :meth:`PromptLayer.from_mapping`.
        Unknown top-level / layer keys raise ``ValueError``.
        """
        if not data:
            return cls()
        known = {"order", "sep", "layers"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown PromptMemory field(s): {sorted(unknown)}; "
                f"valid keys are {sorted(known)}"
            )
        layers: dict[str, PromptLayer] = {}
        for name, spec in (data.get("layers") or {}).items():
            spec = dict(spec or {})
            is_tools = bool(spec.pop("tools", False)) or "mode" in spec
            if is_tools:
                layers[name] = tools_layer(name=name, **spec)
            else:
                layers[name] = PromptLayer.from_mapping(name, spec)
        return cls(
            layers=layers,
            order=tuple(data.get("order") or ()),
            sep=data.get("sep", "\n\n"),
        )


# --------------------------------------------------------------------------- #
# Tool layer — definitions vs. guidance                                       #
# --------------------------------------------------------------------------- #
def render_tools_block(tools: Optional[list[dict[str, Any]]], *, tag: str = "tools") -> str:
    """Render OpenAI-style tool schemas as a Hermes-style ``<tools>`` block.

    Returns ``""`` for an empty/absent tool list so the owning layer is dropped.
    """
    if not tools:
        return ""
    lines = [f"<{tag}>"]
    lines += [json.dumps(t, ensure_ascii=False) for t in tools]
    lines.append(f"</{tag}>")
    return "\n".join(lines)


def tools_layer(
    *,
    name: str = "tools",
    mode: str = "json_schema",
    guidance: str = "",
    order: int = 40,
    enabled: bool = True,
    title: Optional[str] = None,
    read_only: bool = False,
    tag: str = "tools",
) -> PromptLayer:
    """Build the tool section as a dynamic layer reading ``ctx["tools"]``.

    * ``mode="json_schema"`` (default) — emit ``guidance`` followed by the tool
      schemas as a ``<tools>`` block, for backends without native tool calling.
    * ``mode="native"`` — emit only ``guidance``; the schemas travel in the
      request ``tools`` field instead, so they are *not* duplicated in the prompt.

    Either way the layer is dropped when its body is empty (no guidance and, in
    ``json_schema`` mode, no tools).
    """
    if mode not in ("json_schema", "native"):
        raise ValueError(f"tools_layer mode must be 'json_schema' or 'native', got {mode!r}")

    def _render(ctx: Mapping[str, Any]) -> str:
        parts: list[str] = []
        if guidance:
            parts.append(guidance)
        if mode == "json_schema":
            block = render_tools_block(ctx.get("tools"), tag=tag)
            if block:
                parts.append(block)
        return "\n\n".join(parts)

    return PromptLayer(
        name=name,
        order=order,
        enabled=enabled,
        title=title,
        render=_render,
        read_only=read_only,
    )


def default_prompt_memory() -> PromptMemory:
    """A canonical, opt-in layer set + order (Hermes/coding-agent convergence).

    Stable layers (identity…rules) precede volatile ones (memory, environment,
    user_instructions) so a backend prefix cache can reuse the head. Every layer
    but ``tools`` ships *empty* — composition skips empties, so the default is a
    no-op until a caller fills layers via :meth:`PromptMemory.record` or YAML.
    """
    memory = PromptMemory(
        order=(
            "identity",
            "tone",
            "tool_guidance",
            "tools",
            "capabilities",
            "rules",
            "memory",
            "environment",
            "objective",
            "user_instructions",
        )
    )
    memory.record(PromptLayer("identity", order=10))
    memory.record(PromptLayer("tone", order=20))
    memory.record(PromptLayer("tool_guidance", order=30))
    memory.record(tools_layer(order=40))  # json_schema by default
    memory.record(PromptLayer("capabilities", order=50))
    memory.record(PromptLayer("rules", order=60))
    memory.record(PromptLayer("memory", order=70))  # editable (read_only=False)
    memory.record(PromptLayer("environment", order=80))
    memory.record(PromptLayer("objective", order=90))
    memory.record(PromptLayer("user_instructions", order=100))
    return memory
