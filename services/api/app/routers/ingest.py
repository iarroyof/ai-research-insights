# services/api/app/routers/ingest.py
from fastapi import APIRouter, UploadFile, File, Depends
from app.triplets.csv_ingest import ingest_csv_triplets

router = APIRouter(prefix="/triplets", tags=["triplets"])

@router.post("/ingest_csv")
async def ingest_triplets_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file of triplets to enqueue ingestion.
    The worker will process and index triplets asynchronously.
    """
    job_id, result = await ingest_csv_triplets("default", file)
    return {"job_id": job_id, **result}
