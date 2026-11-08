"""Song identification against a local public-domain cello catalog."""

from .catalog import CatalogEntry, load_catalog
from .match import IdentificationResult, identify_sequence
from .musicxml_edition import copy_edition_musicxml, parse_musicxml_fingerprint
from .score import attach_catalog_score

__all__ = [
    "CatalogEntry",
    "IdentificationResult",
    "attach_catalog_score",
    "copy_edition_musicxml",
    "identify_sequence",
    "load_catalog",
    "parse_musicxml_fingerprint",
]
