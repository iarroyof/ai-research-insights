# services/api/app/search/store.py
from __future__ import annotations

from typing import List, Dict, Any, Union
import anyio
import psycopg
from app.config import settings


def _pg_dsn() -> str:
    """Get PostgreSQL connection string for psycopg."""
    from app.config import settings
    dsn_str = str(settings.postgres.dsn)
    return dsn_str.replace("postgresql+psycopg://", "postgresql://")


def _is_uuid(value: str) -> bool:
    """Check if a string is a valid UUID format."""
    import re
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, value.lower()))


def _get_field(obj, field):
    """Safely get field from dict or Pydantic model."""
    if hasattr(obj, field):
        return getattr(obj, field, None)
    elif isinstance(obj, dict):
        return obj.get(field)
    return None


def _has_text_data(item) -> bool:
    """Check if item already contains text and doesn't need lookup."""
    text = _get_field(item, 'text')
    return text is not None and len(str(text).strip()) > 0


def _normalize_item(item) -> dict:
    """Normalize an item that already has text data."""
    paper_id = _get_field(item, 'paper_id')
    sent_id = _get_field(item, 'sent_id')
    text = _get_field(item, 'text')
    
    result = {
        "paper_id": str(paper_id) if paper_id else None,
        "sent_id": str(sent_id) if sent_id else None,
        "text": str(text) if text else "",
        "title": _get_field(item, 'title'),
        "pmid": _get_field(item, 'pmid'),
        "pmcid": _get_field(item, 'pmcid'),
        "page": _get_field(item, 'page'),
    }
    
    for field, target in [("subject", "subject"), ("relation", "predicate"), ("object", "object"), ("confidence", "confidence")]:
        val = _get_field(item, field)
        if val is not None:
            result[target] = val
    
    # Also check for 'predicate' directly (in case it's already normalized)
    if result.get("predicate") is None:
        predicate_val = _get_field(item, "predicate")
        if predicate_val is not None:
            result["predicate"] = predicate_val
    
    return result


def _fetch_sentences_sync(tenant: str, items: List[Union[dict, Any]]) -> List[dict]:
    """Fetch sentence records from PostgreSQL (UUID-based papers only)."""
    pairs = []
    for item in items:
        paper_id = _get_field(item, 'paper_id')
        sent_id = _get_field(item, 'sent_id')
        if paper_id and sent_id is not None:
            pairs.append((paper_id, sent_id))
    
    if not pairs:
        return []
    
    # Filter to UUID + integer pairs only
    pg_pairs = []
    for paper_id, sent_id in pairs:
        is_uuid_paper = _is_uuid(str(paper_id))
        is_int_sent = isinstance(sent_id, int) or (isinstance(sent_id, str) and sent_id.isdigit())
        if is_uuid_paper and is_int_sent:
            pg_pairs.append((str(paper_id), int(sent_id) if isinstance(sent_id, str) else sent_id))
    
    if not pg_pairs:
        return []
    
    results = []
    try:
        with psycopg.connect(_pg_dsn()) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("SELECT set_config('app.tenant_id', %s, true);", (tenant,))
                except Exception:
                    pass
            
            q = """
            SELECT s.id, s.paper_id, s.sent_id, s.text, s.page, s.pmid, s.pmcid, p.title
            FROM sentences s
            JOIN papers p ON p.id = s.paper_id
            WHERE (s.paper_id, s.sent_id) IN (
              SELECT UNNEST(%s::uuid[]), UNNEST(%s::int[])
            )
            """
            paper_ids = [p for p, _ in pg_pairs]
            sent_ids = [sid for _, sid in pg_pairs]
            
            with conn.cursor() as cur:
                cur.execute(q, (paper_ids, sent_ids))
                rows = cur.fetchall()
            
            for row in rows:
                sentence_id, paper_id, sent_id, text, page, pmid, pmcid, title = row
                results.append({
                    "sentence_id": str(sentence_id),
                    "paper_id": str(paper_id),
                    "sent_id": int(sent_id),
                    "text": text,
                    "page": page,
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "title": title,
                })
    except Exception as e:
        pass  # Silently handle PostgreSQL errors
    
    return results


async def get_sentences_by_ids(tenant: str, items: List[Union[dict, Any]]) -> List[dict]:
    """
    Smart fetcher:
    1. Uses existing text data if present (from search results)
    2. Only queries PostgreSQL for UUID-based papers that need lookup
    """
    if not items:
        return []
    
    ready_items = []
    need_lookup = []
    
    for item in items:
        if _has_text_data(item):
            ready_items.append(_normalize_item(item))
        else:
            need_lookup.append(item)
    
    results = ready_items.copy()
    
    if need_lookup:
        db_results = await anyio.to_thread.run_sync(_fetch_sentences_sync, tenant, need_lookup)
        results.extend(db_results)
    
    return results
