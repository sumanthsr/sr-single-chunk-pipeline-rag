from __future__ import annotations

import re

from app.pipeline.parser import ParsedDocument, ParsedElement
from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)
cfg = get_config()

# IMRaD heading vocabulary — order matters, first match wins
_HEADINGS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\babstract\b", re.I), "abstract"),
    (re.compile(r"\b(introduction|background)\b", re.I), "introduction"),
    (re.compile(r"\b(materials?|methods?|methodology|experimental)\b", re.I), "methods"),
    (re.compile(r"\b(results?|findings?|observations?)\b", re.I), "results"),
    (re.compile(r"\bdiscussion\b", re.I), "discussion"),
    (re.compile(r"\b(conclusion|summary|closing remarks)\b", re.I), "conclusion"),
    (re.compile(r"\b(references?|bibliography)\b", re.I), "references"),
    (re.compile(r"\b(supplementary|appendix)\b", re.I), "supplementary"),
]


def detect_structure(doc: ParsedDocument) -> ParsedDocument:
    """
    Walk elements in order and assign section_label + heading_path.
    Returns the modified document.
    """
    current_section = "body"
    heading_path: list[str] = []

    for el in doc.elements:
        detected = _match_heading(el.text)
        if detected:
            current_section = detected
            heading_path = [detected.title()]

        el.section_label = current_section
        el.heading_path = list(heading_path)

    logger.info(
        "detector.done",
        filename=doc.pdf_filename,
        section_counts=_counts(doc.elements),
    )
    return doc


def _match_heading(text: str) -> str | None:
    """
    Return a section label if the text looks like an IMRaD heading.
    Headings are short, do not end with a period, and match the vocabulary.
    """
    first_line = text.strip().split("\n")[0].strip()
    if len(first_line) > 80 or first_line.endswith("."):
        return None
    for pattern, label in _HEADINGS:
        if pattern.search(first_line):
            return label
    return None


def _counts(elements: list[ParsedElement]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for el in elements:
        counts[el.section_label] = counts.get(el.section_label, 0) + 1
    return counts