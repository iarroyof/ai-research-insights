# services/api/app/triplets/csv_ingest.py
import csv
import tempfile
import uuid
import logging
import time
from typing import Tuple
from pathlib import Path
from fastapi import UploadFile
from opensearchpy import helpers
from app.search.os_client import os_client

log = logging.getLogger("triplets.ingest")

CHUNK_SIZE = 1024 * 1024  # Read 1 MB at a time
BATCH_SIZE = 1000         # Bulk index every 1000 rows
PROGRESS_INTERVAL = 10000 # Log progress every 10K rows

async def ingest_csv_triplets(tenant: str, file: UploadFile) -> Tuple[str, dict]:
    """
    Stream a CSV file into OpenSearch with proper handling of multi-line quoted fields.
    Uses temporary file to handle large files without loading into memory.
    """
    job_id = str(uuid.uuid4())
    index = f"triplets_{tenant}"
    client = os_client()

    # Create index if it doesn't exist
    if not client.indices.exists(index=index):
        log.info(f"Creating index: {index}")
        client.indices.create(
            index=index,
            body={
                "settings": {
                    "index": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "refresh_interval": "30s",
                    }
                },
                "mappings": {
                    "properties": {
                        "article_id": {"type": "keyword"},
                        "sentence_text": {"type": "text"},
                        "subject": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "relation": {"type": "keyword"},
                        "object": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                        "subject_probably_EBio": {"type": "float"},
                        "subject_probably_NGen": {"type": "float"},
                        "subject_probably_otro": {"type": "float"},
                        "object_probably_EBio": {"type": "float"},
                        "object_probably_NGen": {"type": "float"},
                        "object_probably_otro": {"type": "float"},
                    }
                },
            },
        )
        log.info(f"✓ Index created: {index}")
    else:
        log.info(f"✓ Index already exists: {index}")

    ingested = 0
    rejected = 0
    start_time = time.time()
    last_progress_time = start_time
    
    log.info(f"[JOB:{job_id}] Starting CSV ingestion")
    
    # Write uploaded file to temporary location
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp:
        tmp_path = Path(tmp.name)
        
        # Stream file to disk with progress
        log.info("⏳ Streaming upload to temporary file...")
        bytes_written = 0
        while chunk := await file.read(CHUNK_SIZE):
            tmp.write(chunk)
            bytes_written += len(chunk)
            if bytes_written % (50 * CHUNK_SIZE) == 0:  # Every 50MB
                log.info(f"   Uploaded {bytes_written / (1024**3):.2f} GB so far...")
        
        log.info(f"✓ Upload complete: {bytes_written / (1024**3):.2f} GB written to {tmp_path}")
    
    try:
        # Now read with proper CSV parser
        log.info(f"⏳ Parsing CSV and indexing to OpenSearch...")
        with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            
            actions = []
            
            def safe_float(value):
                try:
                    return float(value) if value else 0.0
                except (ValueError, TypeError):
                    return 0.0
            
            for row_num, row in enumerate(reader, start=2):
                try:
                    # Validate required fields
                    subject = (row.get("subject") or "").strip()
                    obj = (row.get("object") or "").strip()
                    
                    if not subject or not obj:
                        rejected += 1
                        continue
                    
                    # Clean sentence_text (normalize whitespace)
                    sentence_text = (row.get("sentence_text") or "").strip()
                    sentence_text = " ".join(sentence_text.split())
                    
                    actions.append({
                        "_index": index,
                        "_source": {
                            "article_id": row.get("article_id", "").strip(),
                            "sentence_text": sentence_text,
                            "subject": subject,
                            "relation": row.get("relation", "").strip(),
                            "object": obj,
                            "subject_probably_EBio": safe_float(row.get("subject_probably_EBio")),
                            "subject_probably_NGen": safe_float(row.get("subject_probably_NGen")),
                            "subject_probably_otro": safe_float(row.get("subject_probably_otro")),
                            "object_probably_EBio": safe_float(row.get("object_probably_EBio")),
                            "object_probably_NGen": safe_float(row.get("object_probably_NGen")),
                            "object_probably_otro": safe_float(row.get("object_probably_otro")),
                        },
                    })
                    
                    ingested += 1
                    
                    # Progress logging
                    if ingested % PROGRESS_INTERVAL == 0:
                        elapsed = time.time() - last_progress_time
                        rate = PROGRESS_INTERVAL / elapsed if elapsed > 0 else 0
                        total_elapsed = time.time() - start_time
                        log.info(f"📊 Progress: {ingested:,} docs indexed, {rejected:,} rejected | Rate: {rate:.0f} docs/sec | Time: {total_elapsed/60:.1f} min")
                        last_progress_time = time.time()
                    
                    # Bulk index in batches
                    if len(actions) >= BATCH_SIZE:
                        try:
                            success, failed = helpers.bulk(client, actions, raise_on_error=False, request_timeout=60)
                            if failed:
                                log.warning(f"⚠️  Failed to index {len(failed)} documents")
                                rejected += len(failed)
                            actions.clear()
                        except Exception as e:
                            log.error(f"❌ Bulk indexing error at row {row_num}: {e}")
                            rejected += len(actions)
                            actions.clear()
                            
                except Exception as e:
                    rejected += 1
                    if rejected % 1000 == 0:
                        log.warning(f"⚠️  Total rejected so far: {rejected:,} (last error at row {row_num}: {e})")
            
            # Index remaining actions
            if actions:
                try:
                    success, failed = helpers.bulk(client, actions, raise_on_error=False, request_timeout=60)
                    if failed:
                        log.warning(f"⚠️  Failed to index {len(failed)} documents in final batch")
                        rejected += len(failed)
                    log.info(f"✓ Indexed final batch: {success} documents")
                except Exception as e:
                    log.error(f"❌ Final bulk indexing error: {e}")
                    rejected += len(actions)
        
        # Refresh index
        log.info("⏳ Refreshing index to make documents searchable...")
        client.indices.refresh(index=index)
        
        total_time = time.time() - start_time
        log.info(f"✅ Ingestion complete: {ingested:,} indexed, {rejected:,} rejected in {total_time/60:.1f} minutes")
        
    finally:
        # Clean up temporary file
        tmp_path.unlink(missing_ok=True)
        log.info("🧹 Temporary file cleaned up")
    
    return job_id, {"ingested": ingested, "rejected": rejected, "duration_seconds": int(total_time)}

