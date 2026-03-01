"""
app/pipeline/embedder.py
Stage 5 — Embed chunks using pritamdeka/S-PubMedBert-MS-MARCO.
Same model instance used for both ingestion and query — no drift possible.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.pipeline.chunker import Chunk
from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
cfg = get_config()


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    ecfg = cfg.embedding
    logger.info(f"Loading embedding model: {ecfg.model_name}")
    model = SentenceTransformer(ecfg.model_name, cache_folder=ecfg.cache_dir)
    logger.info(f"Embedding model ready: {ecfg.model_name}")
    return model


def embed_chunks(chunks: list[Chunk]) -> tuple[list[Chunk], list[list[float]]]:
    model = _get_model()
    ecfg = cfg.embedding
    texts = [c.prefixed_text for c in chunks]
    vectors = model.encode(
        texts,
        batch_size=ecfg.batch_size,
        normalize_embeddings=ecfg.normalize,
        show_progress_bar=True,
    )
    logger.info(f"Embedded {len(chunks)} chunks")
    return chunks, [v.tolist() for v in vectors]


def embed_query(query: str) -> list[float]:
    model = _get_model()
    ecfg = cfg.embedding
    vector = model.encode(query, normalize_embeddings=ecfg.normalize)
    return vector.tolist()
