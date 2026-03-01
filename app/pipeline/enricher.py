"""
app/pipeline/enricher.py
Stage 4 — Quality filter and near-duplicate removal.
"""

from __future__ import annotations

import math
import string

from datasketch import MinHash, MinHashLSH

from app.pipeline.chunker import Chunk
from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
cfg = get_config()

_MIN_QUALITY = 0.3
_MINHASH_THRESHOLD = 0.85
_MINHASH_NUM_PERM = 128
_scores: dict[str, float] = {}


def enrich(chunks: list[Chunk]) -> list[Chunk]:
    scored = [_score(c) for c in chunks]
    passed = [c for c in scored if _scores.get(c.chunk_id, 0.0) >= _MIN_QUALITY]
    logger.info(f"Quality filter: {len(chunks)} -> {len(passed)} chunks")
    deduped = _dedup(passed)
    logger.info(f"Dedup: {len(passed)} -> {len(deduped)} chunks")
    return deduped


def _score(chunk: Chunk) -> Chunk:
    text = chunk.text
    score = 1.0
    if chunk.token_count < cfg.chunking.min_tokens:
        score *= max(0.1, chunk.token_count / cfg.chunking.min_tokens)
    diversity = _entropy(text)
    if diversity < 0.05:
        score *= 0.1
    elif diversity < 0.15:
        score *= 0.5
    num_ratio = sum(c.isdigit() for c in text) / max(1, len(text))
    if num_ratio > 0.6:
        score *= 0.4
    punct_ratio = sum(c in string.punctuation for c in text) / max(1, len(text))
    if punct_ratio > 0.4:
        score *= 0.5
    _scores[chunk.chunk_id] = round(min(1.0, max(0.0, score)), 4)
    return chunk


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    n = len(text)
    h = -sum((v / n) * math.log2(v / n) for v in freq.values())
    max_h = math.log2(len(freq)) if len(freq) > 1 else 1.0
    return h / max_h


def _shingles(text: str, k: int = 5) -> set[str]:
    t = " ".join(text.lower().split())
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def _build_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=_MINHASH_NUM_PERM)
    for sh in _shingles(text):
        m.update(sh.encode("utf8"))
    return m


def _dedup(chunks: list[Chunk]) -> list[Chunk]:
    lsh = MinHashLSH(threshold=_MINHASH_THRESHOLD, num_perm=_MINHASH_NUM_PERM)
    minhashes: dict[str, MinHash] = {}
    kept: list[Chunk] = []

    for chunk in chunks:
        m = _build_minhash(chunk.text)
        minhashes[chunk.chunk_id] = m
        try:
            results = lsh.query(m)
        except Exception:
            results = []

        if results:
            existing_id = results[0]
            existing = next((c for c in kept if c.chunk_id == existing_id), None)
            if existing and _scores.get(chunk.chunk_id, 0) > _scores.get(existing_id, 0):
                kept = [c for c in kept if c.chunk_id != existing_id]
                try:
                    lsh.remove(existing_id)
                except Exception:
                    pass
                lsh.insert(chunk.chunk_id, m)
                kept.append(chunk)
        else:
            lsh.insert(chunk.chunk_id, m)
            kept.append(chunk)

    return kept
