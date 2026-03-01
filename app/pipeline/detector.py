"""
app/pipeline/detector.py
Stage 2 — Label every ParsedElement with its IMRaD section.
"""

from __future__ import annotations

import re

from app.pipeline.parser import ParsedDocument, ParsedElement
from app.utils.logger import get_logger

logger = get_logger(__name__)

_HEADINGS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\babstract\b", re.I),                                       "abstract"),
    (re.compile(r"\b(introduction|background)\b", re.I),                      "introduction"),
    (re.compile(r"\b(materials?|methods?|methodology|experimental)\b", re.I), "methods"),
    (re.compile(r"\b(results?|findings?|observations?)\b", re.I),             "results"),
    (re.compile(r"\bdiscussion\b", re.I),                                     "discussion"),
    (re.compile(r"\b(conclusion|summary|closing remarks)\b", re.I),           "conclusion"),
    (re.compile(r"\b(references?|bibliography)\b", re.I),                     "references"),
    (re.compile(r"\b(supplementary|appendix)\b", re.I),                       "supplementary"),
]


def detect_structure(doc: ParsedDocument) -> ParsedDocument:
    current_section = "body"
    heading_path: list[str] = []
    for el in doc.elements:
        detected = _match_heading(el.text)
        if detected:
            current_section = detected
            heading_path = [detected.title()]
        el.section_label = current_section
        el.heading_path = list(heading_path)
    counts = {}
    for el in doc.elements:
        counts[el.section_label] = counts.get(el.section_label, 0) + 1
    logger.info(f"Structure detected for {doc.pdf_filename}: {counts}")
    return doc


def _match_heading(text: str) -> str | None:
    first_line = text.strip().split("\n")[0].strip()
    if len(first_line) > 80 or first_line.endswith("."):
        return None
    for pattern, label in _HEADINGS:
        if pattern.search(first_line):
            return label
    return None
