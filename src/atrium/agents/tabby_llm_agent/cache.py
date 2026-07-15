"""Backend-side KV cache tuning for a *shared* tabbyAPI model load.

On a single GPU the model weights are loaded once and can't shrink, so the only
budget left to scale how many agents run concurrently is the KV cache. tabbyAPI
(exllamav3) shares one paged KV pool across every in-flight sequence and can
quantize it (FP16 → Q8/Q6/Q4), which multiplies how many agent sessions fit
beside the weights. These knobs map directly to tabbyAPI ``/v1/model/load``
options; they live here (not in the connection-only :class:`TabbyAgentConfig`)
because only the backend-owning agent applies them.

exllamav3 does *not* offload KV to CPU RAM or disk, so the GPU KV pool is the
hard ceiling: right-size :attr:`KVCacheConfig.cache_size` for the fleet you want
and quantize the cache to stretch it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["KVCacheConfig", "plan_cache_size"]

#: KV cache quantization modes tabbyAPI/exllamav3 accepts, cheapest last.
CACHE_MODES = ("FP16", "Q8", "Q6", "Q4")


def plan_cache_size(num_agents: int, max_seq_len: int) -> int:
    """Total KV token budget to hold ``num_agents`` concurrent sessions of up to
    ``max_seq_len`` tokens each.

    tabbyAPI shares one paged KV pool across all sequences, so the pool must span
    the *sum* of the per-agent contexts. This is the token figure; the VRAM it
    costs then scales with the model's per-token KV footprint and the chosen
    :attr:`KVCacheConfig.cache_mode` (Q4 ≈ ¼ of FP16).
    """
    if num_agents < 1:
        raise ValueError(f"num_agents must be >= 1, got {num_agents}")
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")
    return num_agents * max_seq_len


@dataclass(slots=True)
class KVCacheConfig:
    """Tuning for the shared model's KV cache (a backend-owner concern).

    * ``cache_mode`` — KV quantization: ``FP16`` | ``Q8`` | ``Q6`` | ``Q4``.
      Quantizing trades a little quality for far more concurrent sessions.
    * ``cache_size`` — total KV tokens across *all* sequences (the shared pool).
      Size it with :func:`plan_cache_size` for the fleet you intend to run.
    * ``max_seq_len`` — per-sequence context length (the model's max context).
    * ``max_batch_size`` — cap on prompts processed at once; tabbyAPI otherwise
      derives it from ``cache_size``.
    """

    cache_mode: str = "FP16"
    cache_size: Optional[int] = None
    max_seq_len: Optional[int] = None
    max_batch_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.cache_mode not in CACHE_MODES:
            raise ValueError(
                f"cache_mode must be one of {CACHE_MODES}, got {self.cache_mode!r}"
            )
        for name in ("cache_size", "max_seq_len", "max_batch_size"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 when set, got {value}")

    @classmethod
    def for_agents(
        cls,
        num_agents: int,
        *,
        max_seq_len: int = 32768,
        cache_mode: str = "Q6",
    ) -> "KVCacheConfig":
        """Cache sized to run ``num_agents`` concurrent agents on one shared model.

        Defaults (32k context, Q6 KV) target a coding fleet on a single 24 GB
        3090: the shared pool spans ``num_agents * max_seq_len`` tokens, and
        ``max_batch_size`` matches the fleet so every agent can be in flight at
        once under tabbyAPI's continuous batching.
        """
        return cls(
            cache_mode=cache_mode,
            cache_size=plan_cache_size(num_agents, max_seq_len),
            max_seq_len=max_seq_len,
            max_batch_size=num_agents,
        )

    def load_options(self) -> dict[str, Any]:
        """Render the set fields as tabbyAPI ``/v1/model/load`` options.

        ``cache_mode`` always carries (it has a default); the sizing knobs only
        appear when set, so unset ones fall back to tabbyAPI's own defaults.
        """
        keys = ("cache_mode", "cache_size", "max_seq_len", "max_batch_size")
        return {k: v for k in keys if (v := getattr(self, k)) is not None}
