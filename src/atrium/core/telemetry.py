"""Distributed tracing helpers and W3C context propagation for Atrium.

Every cross-agent hop and in-sandbox command is wrapped in an OpenTelemetry span
annotated with an `openinference <https://github.com/Arize-ai/openinference>`_
span kind (AGENT / LLM / TOOL / CHAIN) so the whole, physically-isolated,
multi-container flow renders as a single timeline in Arize Phoenix.

The module is defensive: if ``opentelemetry`` or ``openinference`` are not
installed it degrades to no-op shims so the rest of the runtime keeps importing
and working without observability.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, MutableMapping, Optional

logger = logging.getLogger("atrium.telemetry")

# --------------------------------------------------------------------------- #
# OpenTelemetry (optional)                                                     #
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised by import environment
    from opentelemetry import trace
    from opentelemetry.propagate import extract, inject
    from opentelemetry.trace import Status, StatusCode

    _OTEL = True
except Exception:  # pragma: no cover
    _OTEL = False
    trace = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# openinference semantic conventions (optional) — fall back to string consts.  #
# --------------------------------------------------------------------------- #
try:  # pragma: no cover
    from openinference.semconv.trace import (
        OpenInferenceSpanKindValues,
        SpanAttributes,
    )

    SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
    INPUT_VALUE = SpanAttributes.INPUT_VALUE
    OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
    LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
    LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
    LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
    LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
    LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
    AGENT = OpenInferenceSpanKindValues.AGENT.value
    LLM = OpenInferenceSpanKindValues.LLM.value
    TOOL = OpenInferenceSpanKindValues.TOOL.value
    CHAIN = OpenInferenceSpanKindValues.CHAIN.value
except Exception:  # pragma: no cover
    SPAN_KIND = "openinference.span.kind"
    INPUT_VALUE = "input.value"
    OUTPUT_VALUE = "output.value"
    LLM_MODEL_NAME = "llm.model_name"
    LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
    LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
    LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
    LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"
    AGENT, LLM, TOOL, CHAIN = "AGENT", "LLM", "TOOL", "CHAIN"

_TRACER_NAME = "atrium"

__all__ = [
    "get_tracer",
    "start_span",
    "inject_traceparent",
    "extract_context",
    "SPAN_KIND",
    "INPUT_VALUE",
    "OUTPUT_VALUE",
    "LLM_MODEL_NAME",
    "LLM_INVOCATION_PARAMETERS",
    "LLM_TOKEN_COUNT_PROMPT",
    "LLM_TOKEN_COUNT_COMPLETION",
    "LLM_TOKEN_COUNT_TOTAL",
    "AGENT",
    "LLM",
    "TOOL",
    "CHAIN",
]


# --------------------------------------------------------------------------- #
# No-op shims (used only when OpenTelemetry is unavailable)                    #
# --------------------------------------------------------------------------- #
class _NoopSpan:
    def set_attribute(self, *_a: Any, **_k: Any) -> None: ...
    def set_attributes(self, *_a: Any, **_k: Any) -> None: ...
    def set_status(self, *_a: Any, **_k: Any) -> None: ...
    def record_exception(self, *_a: Any, **_k: Any) -> None: ...
    def end(self) -> None: ...


def get_tracer(name: str = _TRACER_NAME):
    """Return an OpenTelemetry tracer (or a no-op tracer when OTel is absent)."""
    if _OTEL:
        return trace.get_tracer(name)
    return None


def _coerce(value: Any) -> Any:
    """Coerce an attribute value into something OTel accepts (primitive/str)."""
    if isinstance(value, (str, bool, int, float)):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


@contextmanager
def start_span(
    name: str,
    *,
    kind: Optional[str] = None,
    attributes: Optional[Mapping[str, Any]] = None,
    context: Any = None,
) -> Iterator[Any]:
    """Start an OpenInference-annotated span as the current span.

    Parameters
    ----------
    name:
        Span name.
    kind:
        An openinference span kind (:data:`AGENT`, :data:`LLM`, :data:`TOOL`,
        :data:`CHAIN`).
    attributes:
        Extra span attributes. Non-primitive values are JSON-encoded.
    context:
        An extracted remote :class:`~opentelemetry.context.Context` to parent
        this span under (used to stitch cross-container traces together).
    """
    if not _OTEL:
        yield _NoopSpan()
        return

    tracer = get_tracer()
    with tracer.start_as_current_span(name, context=context) as span:
        try:
            if kind is not None:
                span.set_attribute(SPAN_KIND, kind)
            for key, value in (attributes or {}).items():
                span.set_attribute(key, _coerce(value))
            yield span
        except Exception as exc:  # noqa: BLE001 - re-raised after recording
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def inject_traceparent(carrier: MutableMapping[str, str]) -> MutableMapping[str, str]:
    """Inject the current W3C trace context (``traceparent``) into ``carrier``.

    ``carrier`` may be a plain ``dict``, HTTP headers, or a protobuf ``Struct``
    (A2A ``Message.metadata``) — anything supporting ``carrier[key] = value``.
    """
    if _OTEL:
        try:
            inject(carrier)
        except Exception:  # pragma: no cover - never break the hot path on tracing
            logger.debug("traceparent injection failed", exc_info=True)
    return carrier


def extract_context(carrier: Mapping[str, str]) -> Any:
    """Extract a remote W3C trace context from ``carrier`` (or ``None``)."""
    if _OTEL:
        try:
            return extract(dict(carrier))
        except Exception:  # pragma: no cover
            logger.debug("traceparent extraction failed", exc_info=True)
    return None
