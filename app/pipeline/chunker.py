from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.pipeline.parser import ElementType, ParsedDocument, ParsedElement
from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
cfg = get_config()


@dataclass
class Chunk:
    chunk_id: str
    pdf_id: str
    pdf_filename: str
    title: str
    section_label: str
    chunk_type: str
    page: int
    chunk_index: int
    heading_path: list[str]
    text: str
    prefixed_text: str
    token_count: int


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """
    Convert a ParsedDocument into a flat list of Chunk objects.
    """
    skip = set(cfg.ingestion.skip_sections)
    chunks: list[Chunk] = []
    chunk_index = 0

    for el in doc.elements:
        if el.section_label in skip:
            continue

        if el.element_type == ElementType.TABLE:
            chunk = _make_chunk(el.text, el, doc, chunk_index)
            chunks.append(chunk)
            chunk_index += 1
            continue

        for text in _split(el.text):
            tokens = _count_tokens(text)
            if tokens < cfg.chunking.min_tokens:
                continue
            chunk = _make_chunk(text, el, doc, chunk_index)
            chunks.append(chunk)
            chunk_index += 1

    logger.info(
        "chunker.done",
        filename=doc.pdf_filename,
        chunks=len(chunks),
    )
    return chunks


# ---------------------------------------------------------------------------
# Recursive splitter
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Approximate: 1 token ~ 4 chars for English biomedical text."""
    return max(1, len(text) // 4)


def _split(text: str) -> list[str]:
    target = cfg.chunking.target_tokens
    overlap = cfg.chunking.overlap_tokens
    max_tok = cfg.chunking.max_tokens
    separators = list(cfg.chunking.separators)

    raw = _recursive_split(text, separators, target)
    return _merge_with_overlap(raw, target, overlap, max_tok)


def _recursive_split(text: str, separators: list[str], target: int) -> list[str]:
    if not separators:
        return [text]

    sep = separators[0]
    rest = separators[1:]

    splits = text.split(sep) if sep else [
        text[i: i + target * 4] for i in range(0, len(text), target * 4)
    ]

    result: list[str] = []
    for s in splits:
        s = s.strip()
        if not s:
            continue
        if _count_tokens(s) <= target:
            result.append(s)
        else:
            result.extend(_recursive_split(s, rest, target))
    return result


def _merge_with_overlap(
    splits: list[str],
    target: int,
    overlap: int,
    max_tok: int,
) -> list[str]:
    merged: list[str] = []
    current: list[str] = []
    current_tokens = 0
    prev_overlap = ""

    for s in splits:
        tok = _count_tokens(s)
        if current_tokens + tok > max_tok and current:
            text = " ".join(current)
            if prev_overlap:
                text = prev_overlap + " " + text
            merged.append(text.strip())
            prev_overlap = text[-(overlap * 4):].strip()
            current = [s]
            current_tokens = tok
        else:
            current.append(s)
            current_tokens += tok

    if current:
        text = " ".join(current)
        if prev_overlap:
            text = prev_overlap + " " + text
        merged.append(text.strip())

    return [c for c in merged if c]


# ---------------------------------------------------------------------------
# Chunk construction
# ---------------------------------------------------------------------------

def _make_chunk(
    text: str,
    el: ParsedElement,
    doc: ParsedDocument,
    index: int,
) -> Chunk:
    prefixed = _prefix(doc.title, el.section_label, text)
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        pdf_id=doc.pdf_id,
        pdf_filename=doc.pdf_filename,
        title=doc.title,
        section_label=el.section_label,
        chunk_type=el.element_type.value,
        page=el.page,
        chunk_index=index,
        heading_path=el.heading_path,
        text=text,
        prefixed_text=prefixed,
        token_count=_count_tokens(text),
    )


def _prefix(title: str, section: str, text: str) -> str:
    t = title.strip() or "Unknown Document"
    s = section.replace("_", " ").title()
    return f"[{t}] | [{s}]: {text}"