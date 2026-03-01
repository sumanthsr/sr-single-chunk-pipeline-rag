"""
app/pipeline/ingestor.py
Orchestrates all pipeline stages for a directory of PDFs.
"""

from __future__ import annotations

from pathlib import Path

from app.pipeline.chunker import chunk_document
from app.pipeline.detector import detect_structure
from app.pipeline.embedder import embed_chunks
from app.pipeline.enricher import enrich
from app.pipeline.parser import parse_pdf
from app.pipeline.vector_store import store_chunks
from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
cfg = get_config()


def run_ingestion(pdf_dir: str | None = None) -> dict:
    pdf_path = Path(pdf_dir or cfg.ingestion.pdf_dir)
    pdf_files = sorted(pdf_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDFs found in {pdf_path}")
        return {"pdfs": 0, "chunks": 0, "files": []}

    logger.info(f"Starting ingestion of {len(pdf_files)} PDFs")

    all_chunks = []
    all_vectors = []
    summary = {"pdfs": len(pdf_files), "chunks": 0, "files": []}

    for pdf_file in pdf_files:
        logger.info(f"Processing: {pdf_file.name}")

        parsed = parse_pdf(pdf_file)
        parsed = detect_structure(parsed)
        chunks = chunk_document(parsed)
        chunks = enrich(chunks)

        if not chunks:
            logger.warning(f"No chunks produced for {pdf_file.name}")
            continue

        chunks, vectors = embed_chunks(chunks)
        all_chunks.extend(chunks)
        all_vectors.extend(vectors)

        summary["files"].append({
            "filename": pdf_file.name,
            "chunks": len(chunks),
            "pages": parsed.page_count,
        })
        logger.info(f"Done: {pdf_file.name} — {len(chunks)} chunks")

    if all_chunks:
        store_chunks(all_chunks, all_vectors)
        summary["chunks"] = len(all_chunks)

    logger.info(f"Ingestion complete — {summary['pdfs']} PDFs, {summary['chunks']} chunks")
    return summary
