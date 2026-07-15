"""Tests for the tracing bootstrap and instrumentation helpers.

These verify the "元栓" (exporter/provider wiring) that ships spans to Phoenix:
opt-in resolution, idempotency, span creation/attributes, the cross-container
``traceparent`` round-trip, and the host→sandbox OTEL_* env passthrough.

A single in-memory TracerProvider is installed once for the module (OTel only
honours the first ``set_tracer_provider``), so ``start_span`` records into it.
``configure_tracing`` is exercised with a dummy OTLP exporter so no network
connection is ever opened.
"""

from __future__ import annotations

import opentelemetry.exporter.otlp.proto.http.trace_exporter as _otlp_mod
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from atrium.core import telemetry as tel

_ENV_VARS = (*tel._ENDPOINT_ENV_VARS, "ATRIUM_TRACING_DISABLED")


@pytest.fixture(scope="module")
def memory_exporter() -> InMemorySpanExporter:
    """Install one in-memory provider for the module and expose its exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)  # honoured only on first call (this one)
    return exporter


@pytest.fixture(autouse=True)
def clean_state(monkeypatch: pytest.MonkeyPatch, memory_exporter: InMemorySpanExporter):
    """Clear OTLP env vars, reset configure_tracing's guard, empty the exporter."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    tel._CONFIGURED = False
    tel._PROVIDER = None
    memory_exporter.clear()
    yield
    tel._CONFIGURED = False
    tel._PROVIDER = None


class _DummyOTLPExporter:
    """Stand-in for OTLPSpanExporter so configure_tracing opens no socket."""

    def __init__(self, *_a, **_k) -> None: ...
    def export(self, _spans):  # pragma: no cover - never flushed in tests
        return None

    def shutdown(self) -> None: ...
    def force_flush(self, *_a, **_k) -> bool:
        return True


@pytest.fixture
def dummy_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_otlp_mod, "OTLPSpanExporter", _DummyOTLPExporter)


# --------------------------------------------------------------------------- #
# configure_tracing opt-in semantics                                          #
# --------------------------------------------------------------------------- #
def test_noop_without_endpoint_env() -> None:
    """No endpoint env + no explicit enable → tracing stays off."""
    assert tel.configure_tracing(enabled=None) is False
    assert tel._PROVIDER is None


def test_auto_enabled_by_endpoint_env(monkeypatch, dummy_exporter) -> None:
    """Presence of a standard OTLP endpoint env var auto-enables tracing."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://phoenix.local:6006/v1/traces")
    assert tel.configure_tracing(enabled=None) is True
    assert tel._PROVIDER is not None


def test_kill_switch_overrides_explicit_enable(monkeypatch) -> None:
    """ATRIUM_TRACING_DISABLED wins even over enabled=True."""
    monkeypatch.setenv("ATRIUM_TRACING_DISABLED", "1")
    assert tel.configure_tracing(enabled=True) is False
    assert tel._PROVIDER is None


def test_idempotent(dummy_exporter) -> None:
    """Re-invocation does not build a second provider."""
    assert tel.configure_tracing(enabled=True) is True
    first = tel._PROVIDER
    assert tel.configure_tracing(enabled=True) is True
    assert tel._PROVIDER is first
    tel.shutdown_tracing()
    assert tel._PROVIDER is None


def test_default_endpoint_filled_when_unset(monkeypatch, dummy_exporter) -> None:
    """A default OTLP traces endpoint is supplied when the operator set none."""
    tel.configure_tracing(enabled=True)
    assert monkeypatch  # fixture present
    import os

    assert os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") == tel._DEFAULT_OTLP_TRACES_ENDPOINT


# --------------------------------------------------------------------------- #
# span creation + cross-container context propagation                         #
# --------------------------------------------------------------------------- #
def test_start_span_records_kind_and_attributes(memory_exporter) -> None:
    with tel.start_span("unit.span", kind=tel.LLM, attributes={"foo": "bar"}):
        pass
    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "unit.span"
    assert span.attributes[tel.SPAN_KIND] == tel.LLM
    assert span.attributes["foo"] == "bar"


def test_traceparent_roundtrip_stitches_trace(memory_exporter) -> None:
    """inject→extract carries the trace id across a simulated container hop."""
    carrier: dict[str, str] = {}
    with tel.start_span("caller", kind=tel.AGENT) as parent:
        tel.inject_traceparent(carrier)
        parent_trace_id = parent.get_span_context().trace_id

    assert "traceparent" in carrier  # context actually propagated into the carrier

    ctx = tel.extract_context(carrier)
    with tel.start_span("callee", kind=tel.AGENT, context=ctx) as child:
        child_trace_id = child.get_span_context().trace_id

    assert child_trace_id == parent_trace_id


# --------------------------------------------------------------------------- #
# host → sandbox OTEL_* env passthrough                                       #
# --------------------------------------------------------------------------- #
def test_env_passthrough_injects_and_does_not_override(monkeypatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://10.0.0.1:6006/v1/traces")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.namespace=atrium")
    env = {"OTEL_RESOURCE_ATTRIBUTES": "service.namespace=override"}

    tel.apply_sandbox_env(env)

    # Missing key is injected from the host environment...
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://10.0.0.1:6006/v1/traces"
    # ...but an explicit per-agent value is preserved.
    assert env["OTEL_RESOURCE_ATTRIBUTES"] == "service.namespace=override"


def test_env_passthrough_skips_absent_host_vars() -> None:
    env: dict[str, str] = {}
    tel.apply_sandbox_env(env)
    assert env == {}
