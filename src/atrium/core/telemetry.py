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
import os
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

#: The standard OTLP endpoint env vars the exporter reads natively; shared by
#: opt-in detection and sandbox forwarding below (single source of truth).
_OTLP_ENDPOINT_VARS = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
)
#: Env vars whose presence opts tracing in by default (an endpoint is configured).
_ENDPOINT_ENV_VARS = (*_OTLP_ENDPOINT_VARS, "PHOENIX_COLLECTOR_ENDPOINT")
#: OTLP env vars forwarded from the host into a sandbox (see :func:`apply_sandbox_env`)
#: so the in-container exporter ships to the same backend and its spans stitch into
#: the host trace. ``OTEL_SERVICE_NAME`` is omitted — each container sets its own;
#: ``PHOENIX_COLLECTOR_ENDPOINT`` is omitted — the OTLP exporter does not read it.
_FORWARD_ENV_VARS = (
    *_OTLP_ENDPOINT_VARS,
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_RESOURCE_ATTRIBUTES",
)
#: Default OTLP/HTTP traces endpoint (Phoenix serves UI + OTLP on :6006).
_DEFAULT_OTLP_TRACES_ENDPOINT = "http://localhost:6006/v1/traces"

# Set by configure_tracing(); _CONFIGURED guards against double-initialisation.
_CONFIGURED = False
_PROVIDER: Any = None

__all__ = [
    "get_tracer",
    "start_span",
    "configure_tracing",
    "shutdown_tracing",
    "inject_traceparent",
    "extract_context",
    "apply_sandbox_env",
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


# --------------------------------------------------------------------------- #
# Bootstrap: install a real exporter so spans actually reach Phoenix           #
# --------------------------------------------------------------------------- #
def _tracing_enabled(enabled: Optional[bool]) -> bool:
    """Resolve whether to install an exporter.

    ``enabled`` (when not ``None``) wins. Otherwise tracing is opt-in: it turns
    on only when an OTLP endpoint env var is present. ``ATRIUM_TRACING_DISABLED``
    is an explicit kill switch that overrides everything (so tests/CI stay
    silent and never open an exporter connection).
    """
    if os.environ.get("ATRIUM_TRACING_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    if enabled is not None:
        return enabled
    return any(os.environ.get(var) for var in _ENDPOINT_ENV_VARS)


def configure_tracing(
    service_name: str = _TRACER_NAME,
    *,
    resource_attributes: Optional[Mapping[str, Any]] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Install a ``TracerProvider`` exporting spans over OTLP/HTTP.

    Call this once per process at startup (entrypoints / launchers) so the spans
    created throughout the runtime are actually shipped to Phoenix instead of
    being dropped by OpenTelemetry's default no-op provider.

    Vendor-neutral: the exporter speaks plain OTLP/HTTP and resolves its endpoint
    from the standard ``OTEL_EXPORTER_OTLP*`` env vars (falling back to Phoenix's
    ``http://localhost:6006/v1/traces``), so Phoenix is just one possible backend.

    Idempotent and defensive: a no-op when OTel is unavailable, when tracing is
    disabled (see :func:`_tracing_enabled`), or when already configured. Returns
    ``True`` only when an exporter was installed.
    """
    global _CONFIGURED, _PROVIDER

    if _CONFIGURED:
        return _PROVIDER is not None
    if not _OTEL or not _tracing_enabled(enabled):
        _CONFIGURED = True
        return False

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:  # pragma: no cover - SDK/exporter extras missing
        logger.warning("tracing requested but OTel SDK/exporter unavailable", exc_info=True)
        _CONFIGURED = True
        return False

    # Give the OTLP/HTTP exporter a default endpoint only if the operator did not
    # set one of the standard env vars (the exporter reads those natively).
    if not any(os.environ.get(v) for v in _ENDPOINT_ENV_VARS):
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = _DEFAULT_OTLP_TRACES_ENDPOINT

    resource = Resource.create({SERVICE_NAME: service_name, **(resource_attributes or {})})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    _PROVIDER = provider
    _CONFIGURED = True
    logger.info("tracing configured: service=%s exporter=otlp/http", service_name)
    return True


def shutdown_tracing() -> None:
    """Flush and shut down the configured provider (call at process exit).

    Safe to call unconditionally — a no-op when tracing was never configured.
    Short-lived host processes need this so buffered spans are exported before
    the process dies (``BatchSpanProcessor`` exports asynchronously).
    """
    global _PROVIDER, _CONFIGURED
    if _PROVIDER is not None:
        try:
            _PROVIDER.shutdown()
        except Exception:  # pragma: no cover - best-effort flush
            logger.debug("tracing shutdown failed", exc_info=True)
        finally:
            _PROVIDER = None
    _CONFIGURED = False


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


def apply_sandbox_env(env: MutableMapping[str, str]) -> None:
    """Copy the host's OTLP env vars into ``env`` (non-overriding).

    So an exporter running inside a sandbox ships to the same backend and its
    spans stitch into the host trace. Mutates ``env`` in place: only vars set in
    the host environment and not already present in ``env`` are added, so an
    explicit per-agent value always wins over the inherited host setting.
    """
    for var in _FORWARD_ENV_VARS:
        value = os.environ.get(var)
        if value and var not in env:
            env[var] = value
