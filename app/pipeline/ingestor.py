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
    """
    Run the full ingestion pipeline over all PDFs in pdf_dir.
    Returns a summary dict with counts per PDF and totals.
    """
    pdf_path = Path(pdf_dir or cfg.ingestion.pdf_dir)
    pdf_files = sorted(pdf_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning("ingestor.no_pdfs_found", directory=str(pdf_path))
        return {"pdfs": 0, "chunks": 0, "files": []}

    logger.info("ingestor.started", pdf_count=len(pdf_files))

    all_chunks = []
    all_vectors = []
    summary = {"pdfs": len(pdf_files), "chunks": 0, "files": []}

    for pdf_file in pdf_files:
        logger.info("ingestor.processing", filename=pdf_file.name)

        # Stage 1: Parse
        parsed = parse_pdf(pdf_file)

        # Stage 2: Detect structure
        parsed = detect_structure(parsed)

        # Stage 3: Chunk
        chunks = chunk_document(parsed)

        # Stage 4: Enrich
        chunks = enrich(chunks)

        if not chunks:
            logger.warning("ingestor.no_chunks", filename=pdf_file.name)
            continue

        # Stage 5: Embed
        chunks, vectors = embed_chunks(chunks)

        all_chunks.extend(chunks)
        all_vectors.extend(vectors)

        summary["files"].append({
            "filename": pdf_file.name,
            "chunks": len(chunks),
            "pages": parsed.page_count,
        })

        logger.info(
            "ingestor.pdf_done",
            filename=pdf_file.name,
            chunks=len(chunks),
        )

    # Stage 6: Store all at once
    if all_chunks:
        store_chunks(all_chunks, all_vectors)
        summary["chunks"] = len(all_chunks)

    logger.info(
        "ingestor.complete",
        total_pdfs=summary["pdfs"],
        total_chunks=summary["chunks"],
    )
    return summary