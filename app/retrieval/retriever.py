"""
app/retrieval/retriever.py

Retrieval layer: embed query, search ChromaDB, return ranked chunks.
Observability hooks are called here — swap provider in config.yml.
"""

from __future__ import annotations

import time

from app.observability.tracer import log_retrieval
from app.pipeline.embedder import embed_query
from app.pipeline.vector_store import retrieve
from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
cfg = get_config()


def search(query: str) -> list[dict]:
    """
    Embed the query and retrieve the top-k most relevant chunks.
    Returns a list of chunk dicts with metadata and similarity scores.
    Cross-PDF retrieval is implicit — all PDFs share the same ChromaDB collection.
    """
    start = time.time()

    query_vector = embed_query(query)
    chunks = retrieve(query_vector)

    duration_ms = round((time.time() - start) * 1000, 2)

    logger.info(
        "retriever.search_done",
        query=query[:80],
        chunks_returned=len(chunks),
        duration_ms=duration_ms,
    )

    log_retrieval(query=query, chunks=chunks, duration_ms=duration_ms)

    return chunks
