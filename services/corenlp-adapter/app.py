from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os, requests

CORENLP_URL = os.getenv("CORENLP_URL","http://corenlp:9008")
app = FastAPI(title="CoreNLP OpenIE Adapter")

class ExtractRequest(BaseModel):
    sentences: List[str]
    num_extractions: int | None = None

@app.get("/health")
def health():
    try:
        r = requests.get(f"{CORENLP_URL}/live", timeout=5)
        return {"status":"ok" if r.status_code==200 else "degraded","code":r.status_code}
    except Exception as e:
        return {"status":"degraded","error":str(e)}

@app.post("/extract")
def extract(req: ExtractRequest):
    if not req.sentences:
        raise HTTPException(400,"No sentences provided")
    results=[]
    for s in req.sentences:
        r = requests.post(
            f"{CORENLP_URL}/?properties={{\"annotators\":\"tokenize,ssplit,pos,lemma,depparse,natlog,openie\",\"outputFormat\":\"json\"}}",
            data=s.encode("utf-8"), timeout=150)
        r.raise_for_status()
        js = r.json()
        tuples=[]
        for sent in js.get("sentences",[]):
            for ex in sent.get("openie",[]):
                tuples.append({"subject":ex.get("subject",""),
                               "predicate":ex.get("relation",""),
                               "object":ex.get("object","")})
        results.append({"text": s, "tuples": tuples})
    return {"count":len(results),"results":results}
