# services/api/app/ner/__init__.py
"""
Named Entity Recognition module for dual biomedical + general classification.
"""
from app.ner.classifier import (
    DualNERClassifier,
    get_classifier,
    classify_entity,
    classify_entities,
    classify_triplet,
)
from app.ner.types import (
    EntityClassification,
    BioEntityType,
    GenEntityType,
    BIO_ENTITY_LABELS,
    GEN_ENTITY_LABELS,
)

__all__ = [
    "DualNERClassifier",
    "get_classifier",
    "classify_entity",
    "classify_entities",
    "classify_triplet",
    "EntityClassification",
    "BioEntityType",
    "GenEntityType",
    "BIO_ENTITY_LABELS",
    "GEN_ENTITY_LABELS",
]
