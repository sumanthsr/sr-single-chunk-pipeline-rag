"""
app/observability/tracer.py

Provider-agnostic observability hooks.
All tracing calls in the application go through this module.
To activate a provider, set observability.enabled: true and
observability.provider: <name> in config.yml — no application
code changes required.

Currently wired providers: none
Ready to wire:  langfuse | langsmith | arize | otel
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
_cfg = get_config().observability


def _init_provider() -> Optional[Any]:
    """
    Initialise the configured observability provider once at import time.
    Returns the provider client or None if disabled.
    """
    if not _cfg.enabled or _cfg.provider == "none":
        return None

    provider = _cfg.provider

    try:
        if provider == "langfuse":
            from langfuse import Langfuse  # type: ignore
            p = _cfg.langfuse
            client = Langfuse(
                public_key=p.public_key,
                secret_key=p.secret_key,
                host=p.host,
            )
            logger.info("observability.langfuse_initialised")
            return client

        if provider == "langsmith":
            import langsmith  # type: ignore
            p = _cfg.langsmith
            client = langsmith.Client(api_key=p.api_key)
            logger.info("observability.langsmith_initialised")
            return client

        if provider == "otel":
            from opentelemetry import trace  # type: ignore
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore
            from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore

            p = _cfg.otel
            provider_inst = TracerProvider()
            exporter = OTLPSpanExporter(endpoint=p.endpoint)
            provider_inst.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider_inst)
            logger.info("observability.otel_initialised", endpoint=p.endpoint)
            return trace.get_tracer(p.service_name)

    except Exception as exc:
        logger.warning(
            "observability.init_failed",
            provider=provider,
            error=str(exc),
        )

    return None


_provider_client = _init_provider()


@contextmanager
def trace_span(
    name: str,
    metadata: Optional[dict] = None,
) -> Generator[dict, None, None]:
    """
    Context manager that wraps any operation in a trace span.
    Usage:
        with trace_span("retrieval", {"query": q}) as span:
            span["chunks"] = results
    Falls back to a no-op dict if observability is disabled.
    """
    span: dict = {"name": name, "metadata": metadata or {}, "start": time.time()}
    try:
        yield span
    finally:
        span["duration_ms"] = round((time.time() - span["start"]) * 1000, 2)
        if _provider_client is not None:
            _emit_span(span)


def _emit_span(span: dict) -> None:
    """Route completed span to the active provider."""
    provider = _cfg.provider
    try:
        if provider == "langfuse":
            _provider_client.trace(
                name=span["name"],
                metadata={**span.get("metadata", {}), "duration_ms": span["duration_ms"]},
            )
        elif provider == "langsmith":
            pass  # langsmith uses decorators or run trees; extend here
        elif provider == "otel":
            with _provider_client.start_as_current_span(span["name"]) as s:
                for k, v in (span.get("metadata") or {}).items():
                    s.set_attribute(str(k), str(v))
    except Exception as exc:
        logger.warning("observability.emit_failed", error=str(exc))


def log_retrieval(query: str, chunks: list, duration_ms: float) -> None:
    """Convenience wrapper for retrieval-specific tracing."""
    if not _cfg.enabled:
        return
    with trace_span("retrieval", {
        "query": query,
        "num_chunks": len(chunks),
        "duration_ms": duration_ms,
        "pdf_sources": list({c.get("pdf_filename", "") for c in chunks}),
    }):
        pass


def log_generation(query: str, context_length: int, duration_ms: float) -> None:
    """Convenience wrapper for LLM generation tracing."""
    if not _cfg.enabled:
        return
    with trace_span("generation", {
        "query": query,
        "context_length": context_length,
        "duration_ms": duration_ms,
    }):
        pass
