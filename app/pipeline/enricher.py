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


def enrich(chunks: list[Chunk]) -> list[Chunk]:
    """Quality filter then dedup. Returns cleaned list."""
    scored = [_score(c) for c in chunks]

    before_filter = len(scored)
    passed = [c for c in scored if _quality_score(c) >= _MIN_QUALITY]
    logger.info(
        "enricher.quality_filter",
        before=before_filter,
        after=len(passed),
        dropped=before_filter - len(passed),
    )

    deduped = _dedup(passed)
    logger.info(
        "enricher.dedup",
        before=len(passed),
        after=len(deduped),
        dropped=len(passed) - len(deduped),
    )

    return deduped


# ---------------------------------------------------------------------------
# Quality scoring (stored as a module-level dict to avoid mutating Chunk)
# ---------------------------------------------------------------------------

_scores: dict[str, float] = {}


def _score(chunk: Chunk) -> Chunk:
    text = chunk.text
    score = 1.0

    # Penalise short chunks
    if chunk.token_count < cfg.chunking.min_tokens:
        score *= max(0.1, chunk.token_count / cfg.chunking.min_tokens)

    # Penalise low character diversity
    diversity = _entropy(text)
    if diversity < 0.05:
        score *= 0.1
    elif diversity < 0.15:
        score *= 0.5

    # Penalise mostly numeric content
    num_ratio = sum(c.isdigit() for c in text) / max(1, len(text))
    if num_ratio > 0.6:
        score *= 0.4

    # Penalise excessive punctuation
    punct_ratio = sum(c in string.punctuation for c in text) / max(1, len(text))
    if punct_ratio > 0.4:
        score *= 0.5

    _scores[chunk.chunk_id] = round(min(1.0, max(0.0, score)), 4)
    return chunk


def _quality_score(chunk: Chunk) -> float:
    return _scores.get(chunk.chunk_id, 0.0)


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


# ---------------------------------------------------------------------------
# MinHash dedup
# ---------------------------------------------------------------------------

def _shingles(text: str, k: int = 5) -> set[str]:
    t = " ".join(text.lower().split())
    return {t[i: i + k] for i in range(len(t) - k + 1)}


def _build_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=_MINHASH_NUM_PERM)
    for sh in _shingles(text):
        m.update(sh.encode("utf8"))
    return m


def _dedup(chunks: list[Chunk]) -> list[Chunk]:
    lsh = MinHashLSH(threshold=_MINHASH_THRESHOLD, num_perm=_MINHASH_NUM_PERM)
    minhashes: dict[str, MinHash] = {}
    kept: list[Chunk] = []
    duplicate_ids: set[str] = set()

    for chunk in chunks:
        m = _build_minhash(chunk.text)
        minhashes[chunk.chunk_id] = m

        try:
            results = lsh.query(m)
        except Exception:
            results = []

        if results:
            # Keep whichever has the higher quality score
            existing_id = results[0]
            existing = next((c for c in kept if c.chunk_id == existing_id), None)
            if existing and _quality_score(chunk) > _quality_score(existing):
                duplicate_ids.add(existing_id)
                kept = [c for c in kept if c.chunk_id != existing_id]
                try:
                    lsh.remove(existing_id)
                except Exception:
                    pass
                lsh.insert(chunk.chunk_id, m)
                kept.append(chunk)
            else:
                duplicate_ids.add(chunk.chunk_id)
        else:
            lsh.insert(chunk.chunk_id, m)
            kept.append(chunk)

    return kept