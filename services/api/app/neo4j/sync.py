# services/api/app/neo4j/sync.py
"""
Synchronize triplets from OpenSearch to Neo4j with entity type classification.
Triggered automatically on data ingestion or manually via API endpoint.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional

from app.graph.neo_client import neo_session
from app.search.os_client import os_client
from app.config import settings
from app.ner import classify_triplet, EntityClassification

log = logging.getLogger("neo4j.sync")


def _create_entity_node_query(entity_name: str, classification: EntityClassification) -> str:
    """
    Generate Cypher query to create/update an Entity node with type properties.
    
    Args:
        entity_name: Name of the entity
        classification: EntityClassification result
        
    Returns:
        Cypher query string
    """
    # Build properties dict
    props = {
        "name": entity_name,
    }
    
    if classification.bio_type:
        props["bio_type"] = classification.bio_type.value
        props["bio_label"] = classification.primary_label if classification.bio_confidence > classification.gen_confidence else None
        props["bio_confidence"] = round(classification.bio_confidence, 3)
    
    if classification.gen_type:
        props["gen_type"] = classification.gen_type.value
        props["gen_label"] = classification.primary_label if classification.gen_confidence > classification.bio_confidence else None
        props["gen_confidence"] = round(classification.gen_confidence, 3)
    
    # Add primary type for easy filtering
    props["primary_type"] = classification.primary_type or "entity"
    props["primary_label"] = classification.primary_label
    
    # Build Cypher query
    props_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
    
    return f"""
    MERGE (e:Entity {{name: $name}})
    SET e += {{{props_str}}}
    RETURN e
    """


async def sync_triplet_to_neo4j(
    subject: str,
    relation: str,
    obj: str,
    triple_id: str,
    article_id: str,
    confidence: float,
    sentence_text: str = "",
    subject_probably_ebio: float = 0.0,
    subject_probably_ngen: float = 0.0,
    object_probably_ebio: float = 0.0,
    object_probably_ngen: float = 0.0,
) -> bool:
    """
    Sync a single triplet to Neo4j with entity classification.
    
    Args:
        subject: Subject entity text
        relation: Relation/predicate
        obj: Object entity text  
        triple_id: Unique triplet ID from OpenSearch
        article_id: Source article ID
        confidence: Triplet confidence score
        sentence_text: Source sentence
        subject_probably_ebio: Existing subject biomedical probability
        subject_probably_ngen: Existing subject general probability
        object_probably_ebio: Existing object biomedical probability
        object_probably_ngen: Existing object general probability
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Classify entities using dual NER pipeline
        subj_class, obj_class = classify_triplet(
            subject,
            obj,
            subject_probably_ebio=subject_probably_ebio,
            subject_probably_ngen=subject_probably_ngen,
            object_probably_ebio=object_probably_ebio,
            object_probably_ngen=object_probably_ngen,
        )
        
        with neo_session() as session:
            # Create/update subject node
            session.run(
                _create_entity_node_query(subject, subj_class),
                name=subject,
                bio_type=subj_class.bio_type.value if subj_class.bio_type else None,
                bio_label=subj_class.primary_label if subj_class.bio_confidence > subj_class.gen_confidence else None,
                bio_confidence=round(subj_class.bio_confidence, 3) if subj_class.bio_type else None,
                gen_type=subj_class.gen_type.value if subj_class.gen_type else None,
                gen_label=subj_class.primary_label if subj_class.gen_confidence > subj_class.bio_confidence else None,
                gen_confidence=round(subj_class.gen_confidence, 3) if subj_class.gen_type else None,
                primary_type=subj_class.primary_type or "entity",
                primary_label=subj_class.primary_label,
            )
            
            # Create/update object node
            session.run(
                _create_entity_node_query(obj, obj_class),
                name=obj,
                bio_type=obj_class.bio_type.value if obj_class.bio_type else None,
                bio_label=obj_class.primary_label if obj_class.bio_confidence > obj_class.gen_confidence else None,
                bio_confidence=round(obj_class.bio_confidence, 3) if obj_class.bio_type else None,
                gen_type=obj_class.gen_type.value if obj_class.gen_type else None,
                gen_label=obj_class.primary_label if obj_class.gen_confidence > obj_class.bio_confidence else None,
                gen_confidence=round(obj_class.gen_confidence, 3) if obj_class.gen_type else None,
                primary_type=obj_class.primary_type or "entity",
                primary_label=obj_class.primary_label,
            )
            
            # Create relationship
            session.run(
                """
                MATCH (s:Entity {name: $subject})
                MATCH (o:Entity {name: $object})
                MERGE (s)-[r:RELATED_TO {triple_id: $triple_id}]->(o)
                SET r.relation = $relation,
                    r.article_id = $article_id,
                    r.confidence = $confidence,
                    r.sentence_text = $sentence_text
                RETURN r
                """,
                subject=subject,
                object=obj,
                triple_id=triple_id,
                relation=relation,
                article_id=article_id,
                confidence=confidence,
                sentence_text=sentence_text[:1000],  # Truncate to avoid huge properties
            )
            
        return True
        
    except Exception as e:
        log.error(f"Failed to sync triplet {triple_id} to Neo4j: {e}")
        return False


async def sync_batch_to_neo4j(
    tenant: str,
    index: str,
    batch_size: int = 1000,
    max_triplets: Optional[int] = None,
) -> Dict[str, int]:
    """
    Batch sync triplets from OpenSearch to Neo4j.
    
    Args:
        tenant: Tenant ID
        index: OpenSearch index name
        batch_size: Number of triplets to process at once
        max_triplets: Maximum triplets to sync (None = all)
        
    Returns:
        Dict with sync statistics (synced, failed, skipped)
    """
    client = os_client()
    stats = {"synced": 0, "failed": 0, "skipped": 0}
    
    try:
        # Scroll through all triplets
        query = {"query": {"match_all": {}}, "size": batch_size}
        
        response = client.search(index=index, body=query, scroll="5m")
        scroll_id = response.get("_scroll_id")
        hits = response.get("hits", {}).get("hits", [])
        
        while hits:
            for hit in hits:
                src = hit.get("_source", {})
                triple_id = hit.get("_id")
                
                # Skip if missing required fields
                if not src.get("subject") or not src.get("object"):
                    stats["skipped"] += 1
                    continue
                
                # Calculate confidence from probabilities
                sp_ebio = float(src.get("subject_probably_EBio", 0.0))
                sp_ngen = float(src.get("subject_probably_NGen", 0.0))
                sp_other = float(src.get("subject_probably_otro", 0.0))
                op_ebio = float(src.get("object_probably_EBio", 0.0))
                op_ngen = float(src.get("object_probably_NGen", 0.0))
                op_other = float(src.get("object_probably_otro", 0.0))
                
                s_max = max(sp_ebio, sp_ngen, sp_other)
                o_max = max(op_ebio, op_ngen, op_other)
                confidence = min(s_max, o_max)
                
                # Sync to Neo4j
                success = await sync_triplet_to_neo4j(
                    subject=src.get("subject"),
                    relation=src.get("relation", "related_to"),
                    obj=src.get("object"),
                    triple_id=triple_id,
                    article_id=src.get("article_id", ""),
                    confidence=confidence,
                    sentence_text=src.get("sentence_text", ""),
                    subject_probably_ebio=sp_ebio,
                    subject_probably_ngen=sp_ngen,
                    object_probably_ebio=op_ebio,
                    object_probably_ngen=op_ngen,
                )
                
                if success:
                    stats["synced"] += 1
                else:
                    stats["failed"] += 1
                
                # Check max limit
                if max_triplets and stats["synced"] >= max_triplets:
                    log.info(f"Reached max triplets limit ({max_triplets}), stopping sync")
                    return stats
            
            # Get next batch
            if scroll_id:
                response = client.scroll(scroll_id=scroll_id, scroll="5m")
                hits = response.get("hits", {}).get("hits", [])
                scroll_id = response.get("_scroll_id")
            else:
                break
        
        log.info(f"Batch sync complete: {stats}")
        return stats
        
    except Exception as e:
        log.error(f"Batch sync failed: {e}")
        stats["error"] = str(e)
        return stats


async def ensure_neo4j_constraints():
    """Create Neo4j constraints and indexes for better performance"""
    try:
        with neo_session() as session:
            # Create constraint on Entity.name (ensures uniqueness)
            session.run(
                """
                CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.name IS UNIQUE
                """
            )
            
            # Create index on primary_type for filtering
            session.run(
                """
                CREATE INDEX entity_primary_type IF NOT EXISTS
                FOR (e:Entity) ON (e.primary_type)
                """
            )
            
            # Create index on bio_type
            session.run(
                """
                CREATE INDEX entity_bio_type IF NOT EXISTS
                FOR (e:Entity) ON (e.bio_type)
                """
            )
            
            # Create index on gen_type
            session.run(
                """
                CREATE INDEX entity_gen_type IF NOT EXISTS
                FOR (e:Entity) ON (e.gen_type)
                """
            )
            
            log.info("✓ Neo4j constraints and indexes ensured")
            
    except Exception as e:
        log.warning(f"Failed to create Neo4j constraints/indexes: {e}")
