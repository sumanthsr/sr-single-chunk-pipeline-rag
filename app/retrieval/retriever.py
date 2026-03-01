"""
app/retrieval/retriever.py
Retrieval layer: embed query, search ChromaDB, return ranked chunks.
"""

from __future__ import annotations

import time

from app.observability.tracer import log_retrieval
from app.pipeline.embedder import embed_query
from app.pipeline.vector_store import retrieve
from app.utils.logger import get_logger

logger = get_logger(__name__)


def search(query: str) -> list[dict]:
    start = time.time()
    query_vector = embed_query(query)
    chunks = retrieve(query_vector)
    duration_ms = round((time.time() - start) * 1000, 2)
    logger.info(f"Search completed in {duration_ms}ms — {len(chunks)} chunks returned")
    log_retrieval(query=query, chunks=chunks, duration_ms=duration_ms)
    return chunks
