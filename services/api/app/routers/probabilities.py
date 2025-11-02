# services/api/app/routers/probabilities.py
from __future__ import annotations
from typing import List, Dict
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.zero_shot import score_labels

router = APIRouter(prefix="/predict", tags=["predict"])

class ProbRequest(BaseModel):
    texts: List[str]
    labels: List[str]

class ProbResponse(BaseModel):
    labels: List[str]
    probabilities: List[Dict[str, float]]  # one dict per input text

@router.post("/probabilities", response_model=ProbResponse)
async def predict_probabilities(req: ProbRequest) -> ProbResponse:
    probs = score_labels(req.texts, req.labels)
    return ProbResponse(labels=req.labels, probabilities=probs)

