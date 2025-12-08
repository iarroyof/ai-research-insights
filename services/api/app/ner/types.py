# services/api/app/ner/types.py
"""
Entity type definitions for dual NER pipeline.
Maps SciSpacy and spaCy entity labels to human-readable categories.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional, Dict, Set


class BioEntityType(str, Enum):
    """Biomedical entity types from SciSpacy models"""
    AMINO_ACID = "amino_acid"
    ANATOMICAL_SYSTEM = "anatomical_system"
    CANCER = "cancer"
    CELL = "cell"
    CELLULAR_COMPONENT = "cellular_component"
    DEVELOPING_ANATOMICAL_STRUCTURE = "developing_anatomical_structure"
    GENE_OR_GENE_PRODUCT = "gene_or_gene_product"
    IMMATERIAL_ANATOMICAL_ENTITY = "immaterial_anatomical_entity"
    MULTI_TISSUE_STRUCTURE = "multi_tissue_structure"
    ORGAN = "organ"
    ORGANISM = "organism"
    ORGANISM_SUBDIVISION = "organism_subdivision"
    ORGANISM_SUBSTANCE = "organism_substance"
    PATHOLOGICAL_FORMATION = "pathological_formation"
    SIMPLE_CHEMICAL = "simple_chemical"
    TISSUE = "tissue"
    # BC5CDR model types
    DISEASE = "disease"
    CHEMICAL = "chemical"
    # JNLPBA model types
    PROTEIN = "protein"
    DNA = "dna"
    RNA = "rna"
    CELL_LINE = "cell_line"
    CELL_TYPE = "cell_type"


class GenEntityType(str, Enum):
    """General entity types from spaCy models"""
    PERSON = "person"
    NORP = "nationality_or_group"  # Nationalities or religious/political groups
    FAC = "facility"  # Buildings, airports, highways, bridges
    ORG = "organization"
    GPE = "geopolitical_entity"  # Countries, cities, states
    LOC = "location"  # Non-GPE locations, mountain ranges, bodies of water
    PRODUCT = "product"
    EVENT = "event"
    WORK_OF_ART = "work_of_art"
    LAW = "law"
    LANGUAGE = "language"
    DATE = "date"
    TIME = "time"
    PERCENT = "percent"
    MONEY = "money"
    QUANTITY = "quantity"
    ORDINAL = "ordinal"
    CARDINAL = "cardinal"


# Mapping from SciSpacy model labels to our enum
SCISPACY_LABEL_MAP: Dict[str, BioEntityType] = {
    # BIONLP13CG model labels
    "AMINO_ACID": BioEntityType.AMINO_ACID,
    "ANATOMICAL_SYSTEM": BioEntityType.ANATOMICAL_SYSTEM,
    "CANCER": BioEntityType.CANCER,
    "CELL": BioEntityType.CELL,
    "CELLULAR_COMPONENT": BioEntityType.CELLULAR_COMPONENT,
    "DEVELOPING_ANATOMICAL_STRUCTURE": BioEntityType.DEVELOPING_ANATOMICAL_STRUCTURE,
    "GENE_OR_GENE_PRODUCT": BioEntityType.GENE_OR_GENE_PRODUCT,
    "IMMATERIAL_ANATOMICAL_ENTITY": BioEntityType.IMMATERIAL_ANATOMICAL_ENTITY,
    "MULTI_TISSUE_STRUCTURE": BioEntityType.MULTI_TISSUE_STRUCTURE,
    "ORGAN": BioEntityType.ORGAN,
    "ORGANISM": BioEntityType.ORGANISM,
    "ORGANISM_SUBDIVISION": BioEntityType.ORGANISM_SUBDIVISION,
    "ORGANISM_SUBSTANCE": BioEntityType.ORGANISM_SUBSTANCE,
    "PATHOLOGICAL_FORMATION": BioEntityType.PATHOLOGICAL_FORMATION,
    "SIMPLE_CHEMICAL": BioEntityType.SIMPLE_CHEMICAL,
    "TISSUE": BioEntityType.TISSUE,
    # BC5CDR model labels
    "DISEASE": BioEntityType.DISEASE,
    "CHEMICAL": BioEntityType.CHEMICAL,
    # JNLPBA model labels
    "PROTEIN": BioEntityType.PROTEIN,
    "DNA": BioEntityType.DNA,
    "RNA": BioEntityType.RNA,
    "CELL_LINE": BioEntityType.CELL_LINE,
    "CELL_TYPE": BioEntityType.CELL_TYPE,
}

# Mapping from spaCy model labels to our enum
SPACY_LABEL_MAP: Dict[str, GenEntityType] = {
    "PERSON": GenEntityType.PERSON,
    "NORP": GenEntityType.NORP,
    "FAC": GenEntityType.FAC,
    "ORG": GenEntityType.ORG,
    "GPE": GenEntityType.GPE,
    "LOC": GenEntityType.LOC,
    "PRODUCT": GenEntityType.PRODUCT,
    "EVENT": GenEntityType.EVENT,
    "WORK_OF_ART": GenEntityType.WORK_OF_ART,
    "LAW": GenEntityType.LAW,
    "LANGUAGE": GenEntityType.LANGUAGE,
    "DATE": GenEntityType.DATE,
    "TIME": GenEntityType.TIME,
    "PERCENT": GenEntityType.PERCENT,
    "MONEY": GenEntityType.MONEY,
    "QUANTITY": GenEntityType.QUANTITY,
    "ORDINAL": GenEntityType.ORDINAL,
    "CARDINAL": GenEntityType.CARDINAL,
}

# Display labels for UI
BIO_ENTITY_LABELS: Dict[BioEntityType, str] = {
    BioEntityType.AMINO_ACID: "Amino Acid",
    BioEntityType.ANATOMICAL_SYSTEM: "Anatomical System",
    BioEntityType.CANCER: "Cancer",
    BioEntityType.CELL: "Cell",
    BioEntityType.CELLULAR_COMPONENT: "Cellular Component",
    BioEntityType.DEVELOPING_ANATOMICAL_STRUCTURE: "Developing Structure",
    BioEntityType.GENE_OR_GENE_PRODUCT: "Gene/Gene Product",
    BioEntityType.IMMATERIAL_ANATOMICAL_ENTITY: "Anatomical Entity",
    BioEntityType.MULTI_TISSUE_STRUCTURE: "Multi-Tissue Structure",
    BioEntityType.ORGAN: "Organ",
    BioEntityType.ORGANISM: "Organism",
    BioEntityType.ORGANISM_SUBDIVISION: "Organism Part",
    BioEntityType.ORGANISM_SUBSTANCE: "Organism Substance",
    BioEntityType.PATHOLOGICAL_FORMATION: "Pathological Formation",
    BioEntityType.SIMPLE_CHEMICAL: "Chemical",
    BioEntityType.TISSUE: "Tissue",
    BioEntityType.DISEASE: "Disease",
    BioEntityType.CHEMICAL: "Chemical",
    BioEntityType.PROTEIN: "Protein",
    BioEntityType.DNA: "DNA",
    BioEntityType.RNA: "RNA",
    BioEntityType.CELL_LINE: "Cell Line",
    BioEntityType.CELL_TYPE: "Cell Type",
}

GEN_ENTITY_LABELS: Dict[GenEntityType, str] = {
    GenEntityType.PERSON: "Person",
    GenEntityType.NORP: "Nationality/Group",
    GenEntityType.FAC: "Facility",
    GenEntityType.ORG: "Organization",
    GenEntityType.GPE: "Country/City/State",
    GenEntityType.LOC: "Location",
    GenEntityType.PRODUCT: "Product",
    GenEntityType.EVENT: "Event",
    GenEntityType.WORK_OF_ART: "Work of Art",
    GenEntityType.LAW: "Law",
    GenEntityType.LANGUAGE: "Language",
    GenEntityType.DATE: "Date",
    GenEntityType.TIME: "Time",
    GenEntityType.PERCENT: "Percentage",
    GenEntityType.MONEY: "Money",
    GenEntityType.QUANTITY: "Quantity",
    GenEntityType.ORDINAL: "Ordinal",
    GenEntityType.CARDINAL: "Number",
}


class EntityClassification:
    """Result of classifying a single entity"""
    def __init__(
        self,
        text: str,
        bio_type: Optional[BioEntityType] = None,
        gen_type: Optional[GenEntityType] = None,
        bio_confidence: float = 0.0,
        gen_confidence: float = 0.0,
    ):
        self.text = text
        self.bio_type = bio_type
        self.gen_type = gen_type
        self.bio_confidence = bio_confidence
        self.gen_confidence = gen_confidence

    @property
    def primary_type(self) -> Optional[str]:
        """Return the most confident type (bio or gen)"""
        if self.bio_confidence > self.gen_confidence and self.bio_type:
            return self.bio_type.value
        elif self.gen_type:
            return self.gen_type.value
        elif self.bio_type:
            return self.bio_type.value
        return None

    @property
    def primary_label(self) -> str:
        """Return human-readable label for primary type"""
        if self.bio_confidence > self.gen_confidence and self.bio_type:
            return BIO_ENTITY_LABELS.get(self.bio_type, "Biomedical Entity")
        elif self.gen_type:
            return GEN_ENTITY_LABELS.get(self.gen_type, "General Entity")
        elif self.bio_type:
            return BIO_ENTITY_LABELS.get(self.bio_type, "Biomedical Entity")
        return "Entity"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "text": self.text,
            "bio_type": self.bio_type.value if self.bio_type else None,
            "gen_type": self.gen_type.value if self.gen_type else None,
            "bio_label": BIO_ENTITY_LABELS.get(self.bio_type) if self.bio_type else None,
            "gen_label": GEN_ENTITY_LABELS.get(self.gen_type) if self.gen_type else None,
            "bio_confidence": round(self.bio_confidence, 3),
            "gen_confidence": round(self.gen_confidence, 3),
            "primary_type": self.primary_type,
            "primary_label": self.primary_label,
        }
