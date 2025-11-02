from __future__ import annotations
import os, time
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from parser import parse_generation_to_tuples

MODEL_ID = os.getenv("REBEL_MODEL_ID", "Babelscape/rebel-large")
DEVICE = os.getenv("REBEL_DEVICE", "auto")  # "auto"|"cpu"|"cuda"
USE_BEAM = os.getenv("REBEL_USE_BEAM", "1") == "1"
NUM_BEAMS = int(os.getenv("REBEL_NUM_BEAMS", "6"))
MAX_NEW_TOKENS = int(os.getenv("REBEL_MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.getenv("REBEL_TEMPERATURE", "1.0"))
TOP_P = float(os.getenv("REBEL_TOP_P", "1.0"))
PROMPT_PREFIX = os.getenv("REBEL_PROMPT_PREFIX", "")  # e.g. "extract relations: "

app = FastAPI(title="REBEL Extractor")

_tok = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model.to(DEVICE)

class ExtractBody(BaseModel):
    sentences: List[str]
    max_triples: int | None = None

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}

@app.post("/extract")
def extract(body: ExtractBody):
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    for s in body.sentences:
        s_in = (PROMPT_PREFIX + s).strip()
        enc = _tok(s_in, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        gen = _model.generate(
            **enc,
            do_sample=not USE_BEAM,
            num_beams=NUM_BEAMS if USE_BEAM else None,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            early_stopping=True,
        )
        decoded = _tok.batch_decode(gen, skip_special_tokens=False)[0]
        tuples = parse_generation_to_tuples(decoded)
        if body.max_triples:
            tuples = tuples[: body.max_triples]
        results.append({"text": s, "tuples": tuples})
    return {"count": len(results), "elapsed_sec": round(time.time() - t0, 3), "results": results}
