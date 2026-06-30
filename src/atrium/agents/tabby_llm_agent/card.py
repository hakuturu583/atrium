"""A2A AgentCard definitions for TabbyLLMAgent and its in-sandbox bridge."""

from __future__ import annotations

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

__all__ = ["build_agent_card"]


def build_agent_card(
    version: str,
    *,
    name: str = "tabby_llm_agent",
    supports_tools: bool = True,
) -> AgentCard:
    """Build the A2A card advertising this agent's inference capability.

    The card declares tool-calling support so callers know they can pass
    ``tools``/``tool_choice`` and receive ``tool_calls`` back over A2A.
    """
    tags = ["llm", "inference", "exllamav3", "tabbyapi"]
    if supports_tools:
        tags.append("tool-calling")

    skill = AgentSkill(
        id="inference",
        name="LLM inference",
        description=(
            "WAN-isolated GPU inference served by tabbyAPI on the exllamav3 "
            "backend, including OpenAI-style function/tool calling."
        ),
        tags=tags,
    )

    return AgentCard(
        name=name,
        description=(
            "Atrium TabbyLLMAgent: offline, GPU-only LLM inference over A2A "
            "(tabbyAPI / exllamav3)."
        ),
        version=str(version),
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
    )
