# services/api/app/search/store.py
from __future__ import annotations

from typing import List, Dict, Any
import anyio
import psycopg

from app.config import settings


def _pg_dsn() -> str:
    # new layout: settings.db["postgres"]["dsn"]
    # old layout: settings.pg_dsn or settings.postgres["dsn"]
    try:
        return str(settings.db["postgres"]["dsn"])
    except Exception:
        pass
    if hasattr(settings, "pg_dsn"):
        return str(settings.pg_dsn)
    return str(settings.postgres["dsn"])


def _fetch_sentences_sync(tenant: str, items: List[dict]) -> List[dict]:
    """
    Fetch sentence records by (paper_id, sent_id) pairs.
    Also returns title, pmid/pmcid for link-outs.
    """
    pairs = [(it["paper_id"], it.get("sent_id")) for it in items]
    pairs = [p for p in pairs if p[0] and p[1] is not None]
    if not pairs:
        return []

    with psycopg.connect(_pg_dsn()) as conn:
        # Set RLS tenant (if you use RLS with current_setting)
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT set_config('app.tenant_id', %s, true);", (tenant,))
            except Exception:
                pass

        q = """
        SELECT
          s.id as sentence_id, s.paper_id, s.sent_id, s.text, s.page, s.pmid, s.pmcid,
          p.title
        FROM sentences s
        JOIN papers p ON p.id = s.paper_id
        WHERE (s.paper_id, s.sent_id) IN (
          SELECT UNNEST(%s::uuid[]), UNNEST(%s::int[])
        )
        """
        paper_ids = [p for p, _ in pairs]
        sent_ids = [sid for _, sid in pairs]

        with conn.cursor() as cur:
            cur.execute(q, (paper_ids, sent_ids))
            rows = cur.fetchall()

    out: List[Dict[str, Any]] = []
    for row in rows:
        # psycopg rows are tuples in default cursor
        sentence_id, paper_id, sent_id, text, page, pmid, pmcid, title = row
        out.append(
            {
                "sentence_id": str(sentence_id),
                "paper_id": str(paper_id),
                "sent_id": int(sent_id),
                "text": text,
                "page": page,
                "pmid": pmid,
                "pmcid": pmcid,
                "title": title,
            }
        )
    return out


async def get_sentences_by_ids(tenant: str, items: List[dict]) -> List[dict]:
    return await anyio.to_thread.run_sync(_fetch_sentences_sync, tenant, items)

