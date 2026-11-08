"""Song identification: local PD catalog first, optional remote discovery."""

from .catalog import (
    CatalogEntry,
    format_catalog_listing,
    load_catalog,
    search_catalog,
)
from .discover import (
    DiscoveryAttempt,
    format_discovery_attempts,
    identify_with_discovery,
    load_discovery_index,
)
from .match import IdentificationResult, identify_sequence
from .musicxml_edition import (
    copy_edition_musicxml,
    excerpt_musicxml_by_notes,
    parse_musicxml_fingerprint,
    resolve_clip_note_span,
    write_match_musicxml,
)
from .omr import assess_pdf_quality, find_audiveris, pdf_to_musicxml
from .score import attach_catalog_score

__all__ = [
    "CatalogEntry",
    "DiscoveryAttempt",
    "IdentificationResult",
    "assess_pdf_quality",
    "attach_catalog_score",
    "copy_edition_musicxml",
    "excerpt_musicxml_by_notes",
    "find_audiveris",
    "format_catalog_listing",
    "format_discovery_attempts",
    "identify_sequence",
    "identify_with_discovery",
    "load_catalog",
    "load_discovery_index",
    "parse_musicxml_fingerprint",
    "pdf_to_musicxml",
    "resolve_clip_note_span",
    "search_catalog",
    "write_match_musicxml",
]
