"""
app/observability/tracer.py

Observability stubs — no-ops by default.
To add a provider in future, implement log_retrieval and log_generation here.
Supported providers: langfuse | langsmith | arize | otel
"""

from __future__ import annotations


def log_retrieval(query: str, chunks: list, duration_ms: float) -> None:
    pass


def log_generation(query: str, context_length: int, duration_ms: float) -> None:
    pass