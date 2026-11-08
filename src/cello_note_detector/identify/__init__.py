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
from .musicxml_edition import copy_edition_musicxml, parse_musicxml_fingerprint
from .score import attach_catalog_score

__all__ = [
    "CatalogEntry",
    "DiscoveryAttempt",
    "IdentificationResult",
    "attach_catalog_score",
    "copy_edition_musicxml",
    "format_catalog_listing",
    "format_discovery_attempts",
    "identify_sequence",
    "identify_with_discovery",
    "load_catalog",
    "load_discovery_index",
    "parse_musicxml_fingerprint",
    "search_catalog",
]
