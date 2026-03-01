"""
app/pipeline/parser.py
Stage 1 — Parse raw PDF bytes into structured elements.
Strategy: pdfplumber -> PyMuPDF -> Tesseract OCR
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import fitz
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes

from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
cfg = get_config()


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"


@dataclass
class ParsedElement:
    text: str
    page: int
    element_type: ElementType = ElementType.TEXT
    section_label: str = "unknown"
    heading_path: list[str] = field(default_factory=list)
    ocr: bool = False


@dataclass
class ParsedDocument:
    pdf_filename: str
    pdf_id: str
    title: str
    elements: list[ParsedElement] = field(default_factory=list)
    page_count: int = 0
    has_scanned_pages: bool = False
    warnings: list[str] = field(default_factory=list)


def parse_pdf(pdf_path: Path) -> ParsedDocument:
    pdf_id = pdf_path.stem
    pdf_bytes = pdf_path.read_bytes()
    doc = ParsedDocument(pdf_filename=pdf_path.name, pdf_id=pdf_id, title="")
    try:
        elements, page_count, has_scanned = _extract(pdf_bytes)
        doc.elements = elements
        doc.page_count = page_count
        doc.has_scanned_pages = has_scanned
        doc.title = _infer_title(elements, pdf_path.stem)
    except Exception as exc:
        logger.error(f"Parse failed for {pdf_path.name}: {exc}")
        doc.warnings.append(str(exc))
    logger.info(f"Parsed {pdf_path.name}: {doc.page_count} pages, {len(doc.elements)} elements")
    return doc


def _extract(pdf_bytes: bytes) -> tuple[list[ParsedElement], int, bool]:
    elements: list[ParsedElement] = []
    has_scanned = False
    min_chars = cfg.ingestion.min_text_chars_per_page

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            elements.extend(_extract_tables(page, page_num))
            raw = (page.extract_text(x_tolerance=2, y_tolerance=2) or "").strip()
            if len(raw) >= min_chars:
                elements.append(ParsedElement(text=_clean(raw), page=page_num))
            else:
                fitz_text = _fitz_page(pdf_bytes, page_num - 1)
                if len(fitz_text) >= min_chars:
                    elements.append(ParsedElement(text=_clean(fitz_text), page=page_num))
                else:
                    ocr_text = _ocr_page(pdf_bytes, page_num)
                    if ocr_text:
                        has_scanned = True
                        elements.append(ParsedElement(text=_clean(ocr_text), page=page_num, ocr=True))

    return elements, page_count, has_scanned


def _extract_tables(page, page_num: int) -> list[ParsedElement]:
    elements = []
    try:
        for table in page.extract_tables():
            if not table:
                continue
            rows = [" | ".join(str(c).strip() if c else "" for c in row) for row in table]
            text = "\n".join(rows).strip()
            if len(text) > 10:
                elements.append(ParsedElement(text=text, page=page_num, element_type=ElementType.TABLE))
    except Exception as exc:
        logger.warning(f"Table extraction failed on page {page_num}: {exc}")
    return elements


def _fitz_page(pdf_bytes: bytes, page_index: int) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return doc.load_page(page_index).get_text("text")
    except Exception as exc:
        logger.warning(f"PyMuPDF failed on page {page_index}: {exc}")
        return ""


def _ocr_page(pdf_bytes: bytes, page_num: int) -> str:
    try:
        images = convert_from_bytes(pdf_bytes, first_page=page_num, last_page=page_num, dpi=300)
        return pytesseract.image_to_string(images[0], lang="eng") if images else ""
    except Exception as exc:
        logger.warning(f"OCR failed on page {page_num}: {exc}")
        return ""


def _clean(text: str) -> str:
    text = text.replace("\x00", "").replace("\x0c", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"^[\-_.=]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _infer_title(elements: list[ParsedElement], fallback: str) -> str:
    for el in elements:
        if el.page == 1 and el.element_type == ElementType.TEXT:
            first_line = el.text.split("\n")[0].strip()
            if 10 < len(first_line) <= 200:
                return first_line
    return fallback
