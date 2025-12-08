# services/api/app/ner/classifier.py
"""
Dual NER Pipeline for Biomedical + General Entity Classification.
Uses SciSpacy for biomedical entities and spaCy for general entities.
"""
from __future__ import annotations
import logging
from typing import List, Optional, Dict, Tuple
from functools import lru_cache

from app.ner.types import (
    EntityClassification,
    BioEntityType,
    GenEntityType,
    SCISPACY_LABEL_MAP,
    SPACY_LABEL_MAP,
)

log = logging.getLogger("ner.classifier")

# Global model instances (lazy-loaded)
_bio_nlp = None
_gen_nlp = None
_models_loaded = False


def _ensure_models_loaded():
    """Lazy-load both NER models on first use"""
    global _bio_nlp, _gen_nlp, _models_loaded

    if _models_loaded:
        return

    try:
        import spacy
        import scispacy  # noqa: F401 - needed to register pipelines

        log.info("Loading biomedical NER model (en_ner_bionlp13cg_md)...")
        try:
            _bio_nlp = spacy.load("en_ner_bionlp13cg_md")
            log.info("✓ Biomedical NER model loaded successfully")
        except OSError:
            log.warning(
                "⚠️  Biomedical NER model not found. Install with:\n"
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz"
            )
            _bio_nlp = None

        log.info("Loading general NER model (en_core_web_lg)...")
        try:
            _gen_nlp = spacy.load("en_core_web_lg")
            log.info("✓ General NER model loaded successfully")
        except OSError:
            log.warning(
                "⚠️  General NER model not found. Install with:\n"
                "python -m spacy download en_core_web_lg"
            )
            _gen_nlp = None

        _models_loaded = True

    except ImportError as e:
        log.error(f"❌ Failed to import required packages: {e}")
        log.error("Install with: pip install spacy scispacy")
        _bio_nlp = None
        _gen_nlp = None
        _models_loaded = True


class DualNERClassifier:
    """
    Classifies entities using both biomedical (SciSpacy) and general (spaCy) NER models.
    
    Usage:
        classifier = DualNERClassifier()
        result = classifier.classify("bevacizumab")
        # result.bio_type = BioEntityType.SIMPLE_CHEMICAL
        # result.gen_type = None
    """

    def __init__(self):
        _ensure_models_loaded()
        self.bio_nlp = _bio_nlp
        self.gen_nlp = _gen_nlp

    def classify(self, text: str) -> EntityClassification:
        """
        Classify a single entity string using both models.
        
        Args:
            text: Entity text to classify (e.g., "bevacizumab", "University of Chile")
            
        Returns:
            EntityClassification with bio_type and/or gen_type populated
        """
        if not text or not text.strip():
            return EntityClassification(text=text)

        text = text.strip()
        bio_type = None
        gen_type = None
        bio_confidence = 0.0
        gen_confidence = 0.0

        # Try biomedical classification
        if self.bio_nlp:
            try:
                doc = self.bio_nlp(text)
                if doc.ents:
                    # Take the first/longest entity
                    ent = max(doc.ents, key=lambda e: len(e.text))
                    bio_label = ent.label_
                    bio_type = SCISPACY_LABEL_MAP.get(bio_label)
                    bio_confidence = 0.8  # SciSpacy doesn't provide confidence scores
            except Exception as e:
                log.debug(f"Biomedical NER failed for '{text}': {e}")

        # Try general classification
        if self.gen_nlp:
            try:
                doc = self.gen_nlp(text)
                if doc.ents:
                    # Take the first/longest entity
                    ent = max(doc.ents, key=lambda e: len(e.text))
                    gen_label = ent.label_
                    gen_type = SPACY_LABEL_MAP.get(gen_label)
                    gen_confidence = 0.8  # spaCy doesn't expose per-entity confidence in standard models
            except Exception as e:
                log.debug(f"General NER failed for '{text}': {e}")

        return EntityClassification(
            text=text,
            bio_type=bio_type,
            gen_type=gen_type,
            bio_confidence=bio_confidence,
            gen_confidence=gen_confidence,
        )

    def classify_batch(self, texts: List[str]) -> List[EntityClassification]:
        """
        Classify multiple entities efficiently using batch processing.
        
        Args:
            texts: List of entity strings to classify
            
        Returns:
            List of EntityClassification objects, one per input text
        """
        if not texts:
            return []

        results = [EntityClassification(text=t) for t in texts]

        # Batch biomedical classification
        if self.bio_nlp:
            try:
                # spaCy's pipe() processes texts in batches efficiently
                docs = list(self.bio_nlp.pipe(texts, disable=["tagger", "parser"]))
                for i, doc in enumerate(docs):
                    if doc.ents:
                        ent = max(doc.ents, key=lambda e: len(e.text))
                        bio_label = ent.label_
                        results[i].bio_type = SCISPACY_LABEL_MAP.get(bio_label)
                        results[i].bio_confidence = 0.8
            except Exception as e:
                log.warning(f"Batch biomedical NER failed: {e}")

        # Batch general classification
        if self.gen_nlp:
            try:
                docs = list(self.gen_nlp.pipe(texts, disable=["tagger", "parser"]))
                for i, doc in enumerate(docs):
                    if doc.ents:
                        ent = max(doc.ents, key=lambda e: len(e.text))
                        gen_label = ent.label_
                        results[i].gen_type = SPACY_LABEL_MAP.get(gen_label)
                        results[i].gen_confidence = 0.8
            except Exception as e:
                log.warning(f"Batch general NER failed: {e}")

        return results

    def classify_with_probabilities(
        self,
        text: str,
        subject_probably_ebio: float = 0.0,
        subject_probably_ngen: float = 0.0,
        object_probably_ebio: float = 0.0,
        object_probably_ngen: float = 0.0,
    ) -> EntityClassification:
        """
        Classify entity using BOTH dual NER and existing probability scores.
        Uses probabilities to guide which model to prioritize.
        
        Args:
            text: Entity text
            subject_probably_ebio: Existing biomedical probability
            subject_probably_ngen: Existing general entity probability
            
        Returns:
            EntityClassification with enhanced confidence scores
        """
        result = self.classify(text)

        # Enhance confidence scores using existing probabilities
        # Average the NER confidence with the existing probability
        if result.bio_type and subject_probably_ebio > 0:
            result.bio_confidence = (result.bio_confidence + subject_probably_ebio) / 2

        if result.gen_type and subject_probably_ngen > 0:
            result.gen_confidence = (result.gen_confidence + subject_probably_ngen) / 2

        return result

    @property
    def is_ready(self) -> bool:
        """Check if at least one model is loaded"""
        return self.bio_nlp is not None or self.gen_nlp is not None

    @property
    def available_models(self) -> Dict[str, bool]:
        """Get status of available models"""
        return {
            "biomedical": self.bio_nlp is not None,
            "general": self.gen_nlp is not None,
        }


# Singleton instance for reuse across requests
@lru_cache(maxsize=1)
def get_classifier() -> DualNERClassifier:
    """Get or create the singleton dual NER classifier"""
    return DualNERClassifier()


def classify_entity(text: str) -> EntityClassification:
    """Convenience function to classify a single entity"""
    classifier = get_classifier()
    return classifier.classify(text)


def classify_entities(texts: List[str]) -> List[EntityClassification]:
    """Convenience function to classify multiple entities"""
    classifier = get_classifier()
    return classifier.classify_batch(texts)


def classify_triplet(
    subject: str,
    obj: str,
    subject_probably_ebio: float = 0.0,
    subject_probably_ngen: float = 0.0,
    object_probably_ebio: float = 0.0,
    object_probably_ngen: float = 0.0,
) -> Tuple[EntityClassification, EntityClassification]:
    """
    Classify both subject and object of a triplet.
    
    Returns:
        Tuple of (subject_classification, object_classification)
    """
    classifier = get_classifier()

    subject_class = classifier.classify_with_probabilities(
        subject,
        subject_probably_ebio=subject_probably_ebio,
        subject_probably_ngen=subject_probably_ngen,
    )

    object_class = classifier.classify_with_probabilities(
        obj,
        object_probably_ebio=object_probably_ebio,
        object_probably_ngen=object_probably_ngen,
    )

    return subject_class, object_class
