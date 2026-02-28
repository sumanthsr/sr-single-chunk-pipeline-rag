from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.pipeline.chunker import Chunk
from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
cfg = get_config()


def _get_collection() -> chromadb.Collection:
    ccfg = cfg.chromadb
    client = chromadb.PersistentClient(
        path=ccfg.persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=ccfg.collection_name,
        metadata={"hnsw:space": ccfg.distance_metric},
    )
    return collection


def is_populated() -> bool:
    """Return True if the collection already contains chunks."""
    try:
        collection = _get_collection()
        return collection.count() > 0
    except Exception:
        return False


def store_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """
    Upsert chunks and their vectors into ChromaDB.
    Uses chunk_id as the stable document ID — safe to re-run.
    """
    collection = _get_collection()

    ids = [c.chunk_id for c in chunks]
    documents = [c.prefixed_text for c in chunks]
    metadatas = [
        {
            "pdf_id": c.pdf_id,
            "pdf_filename": c.pdf_filename,
            "title": c.title,
            "section_label": c.section_label,
            "chunk_type": c.chunk_type,
            "page": c.page,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count,
            "text": c.text,
        }
        for c in chunks
    ]

    # ChromaDB upsert in batches of 512 (Chroma default limit)
    batch_size = 512
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i: i + batch_size],
            documents=documents[i: i + batch_size],
            embeddings=vectors[i: i + batch_size],
            metadatas=metadatas[i: i + batch_size],
        )

    logger.info("vector_store.stored", count=len(chunks))


def retrieve(
    query_vector: list[float],
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    """
    Query ChromaDB and return a list of result dicts with metadata and scores.
    Results are cross-PDF by default — no pdf_id filter applied.
    """
    rcfg = cfg.retrieval
    k = top_k or rcfg.top_k
    threshold = score_threshold if score_threshold is not None else rcfg.score_threshold

    collection = _get_collection()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["metadatas", "distances", "documents"],
    )

    output: list[dict] = []
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    documents = results["documents"][0]

    for meta, dist, doc in zip(metadatas, distances, documents):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score in [0, 1]
        score = round(1 - (dist / 2), 4)
        if score < threshold:
            continue
        output.append({
            **meta,
            "score": score,
            "prefixed_text": doc,
        })

    logger.info("vector_store.retrieved", k=k, returned=len(output))
    return output