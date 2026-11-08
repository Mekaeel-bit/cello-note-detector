"""Song identification against a local public-domain cello catalog."""

from .catalog import (
    CatalogEntry,
    format_catalog_listing,
    load_catalog,
    search_catalog,
)
from .match import IdentificationResult, identify_sequence
from .musicxml_edition import copy_edition_musicxml, parse_musicxml_fingerprint
from .score import attach_catalog_score

__all__ = [
    "CatalogEntry",
    "IdentificationResult",
    "attach_catalog_score",
    "copy_edition_musicxml",
    "format_catalog_listing",
    "identify_sequence",
    "load_catalog",
    "parse_musicxml_fingerprint",
    "search_catalog",
]
