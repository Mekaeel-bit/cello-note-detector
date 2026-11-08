"""Song identification against a local public-domain cello catalog."""

from .catalog import CatalogEntry, load_catalog
from .match import IdentificationResult, identify_sequence
from .score import attach_catalog_score, reference_to_result

__all__ = [
    "CatalogEntry",
    "IdentificationResult",
    "attach_catalog_score",
    "identify_sequence",
    "load_catalog",
    "reference_to_result",
]
