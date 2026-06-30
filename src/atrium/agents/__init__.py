"""Concrete Atrium agents.

The shared abstract :class:`~atrium.agents.inference_agent.InferenceAgent` lives
here directly. Each *concrete* agent is its own self-contained, independently
versioned package (e.g. :mod:`atrium.agents.tabby_llm_agent`) bundling its
host-side A2A code together with its sandbox container definition and policy —
the unit of self-evolution (Morphing / generational swap).
"""

from __future__ import annotations

from atrium.agents.inference_agent import InferenceAgent
from atrium.agents.prompt_memory import (
    PromptLayer,
    PromptMemory,
    default_prompt_memory,
    tools_layer,
)

__all__ = [
    "InferenceAgent",
    "PromptLayer",
    "PromptMemory",
    "default_prompt_memory",
    "tools_layer",
]
