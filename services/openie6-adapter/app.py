from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import subprocess, tempfile, os, shutil, json, time, errno

# Config via env (see docker-compose.yml)
MODELS_DIR = os.getenv("OPENIE6_MODELS_DIR", "/models")
WORK_DIR   = os.getenv("OPENIE6_WORK_DIR", "/workspace")     # MUST contain run.py
GPUS       = os.getenv("OPENIE6_GPUS", "0")                  # "0" for first GPU, "-1" for CPU
TIMEOUT_S  = int(os.getenv("OPENIE6_TIMEOUT_SEC", "600"))    # hard cap per job
NORMALIZE  = os.getenv("OPENIE6_NORMALIZE_INPUT", "true").lower() == "true"

OIE_MODEL   = os.path.join(MODELS_DIR, "oie_model",  "epoch=14_eval_acc=0.551_v0.ckpt")
CONJ_MODEL  = os.path.join(MODELS_DIR, "conj_model", "epoch=28_eval_acc=0.854.ckpt")
RESCORE_DIR = os.path.join(MODELS_DIR, "rescore_model")
RUN_PY      = os.path.join(WORK_DIR, "run.py")               # openie6-docker places code under /workspace

app = FastAPI(title="OpenIE6 Adapter", version="1.0.0")

class ExtractRequest(BaseModel):
    sentences: List[str] = Field(min_length=1)
    num_extractions: int = Field(default=5, ge=1, le=20)
    normalize: bool | None = None  # override NORMALIZE per request

def _exists(p: str) -> bool:
    return os.path.exists(p)

def _check_health() -> Dict[str, Any]:
    issues = []
    if not _exists(WORK_DIR):
        issues.append(f"WORK_DIR not found: {WORK_DIR}")
    if not _exists(RUN_PY):
        issues.append(f"run.py not found at {RUN_PY}. Ensure OPENIE6_WORK_DIR points to the repo root.")
    if not _exists(MODELS_DIR):
        issues.append(f"MODELS_DIR not found: {MODELS_DIR}")
    if not _exists(OIE_MODEL):
        issues.append(f"Missing OIE model: {OIE_MODEL}")
    if not _exists(CONJ_MODEL):
        issues.append(f"Missing conj model: {CONJ_MODEL}")
    if not _exists(RESCORE_DIR):
        issues.append(f"Missing rescore model dir: {RESCORE_DIR}")
    return {
        "status": "ok" if not issues else "degraded",
        "models_dir": MODELS_DIR,
        "work_dir": WORK_DIR,
        "gpus": GPUS,
        "timeout_sec": TIMEOUT_S,
        "issues": issues,
    }

@app.get("/health")
async def health():
    return _check_health()

def _write_input_file(lines: List[str], dirpath: str, normalize: bool) -> str:
    os.makedirs(dirpath, exist_ok=True)
    f = tempfile.NamedTemporaryFile("w", delete=False, dir=dirpath, suffix=".txt", encoding="utf-8")
    try:
        for s in lines:
            line = (s or "").strip()
            if normalize and line and not line.endswith("."):
                line += "."
            f.write(line + "\n")
        return f.name
    finally:
        f.close()

def _parse_output(out_path: str) -> List[Dict[str, Any]]:
    """
    Expected OpenIE6 splitpredict output. We map SENT k -> tuples.
    Uses the same parser as earlier but returns grouped-by-sentence.
    """
    triples_per_sent: Dict[int, List[Dict[str, str]]] = {}
    if not os.path.exists(out_path):
        return []
    with open(out_path, "r", encoding="utf-8") as f:
        sent_id = -1
        for raw in f:
            line = raw.strip()
            if line.startswith("SENT"):
                # e.g., "SENT 0: <text>"
                try:
                    sent_id = int(line.split(":", 1)[0].split()[1])
                except Exception:
                    sent_id = -1
                triples_per_sent.setdefault(sent_id, [])
                continue
            if line.startswith("TUP") and ":" in line:
                tup = line.split(":", 1)[1].strip()
                if tup.startswith("(") and tup.endswith(")"):
                    tup = tup[1:-1]
                parts = [p.strip() for p in tup.split(";")]
                if len(parts) >= 3:
                    triples_per_sent.setdefault(sent_id, []).append(
                        {"subject": parts[0], "predicate": parts[1], "object": parts[2]}
                    )
    # convert to indexed list
    results = []
    for idx in range(0, max(triples_per_sent.keys()) + 1 if triples_per_sent else 0):
        results.append({"sent_id": idx, "tuples": triples_per_sent.get(idx, [])})
    return results

@app.post("/extract")
async def extract(body: ExtractRequest):
    if not body.sentences:
        raise HTTPException(400, "No sentences provided")

    # preflight health
    h = _check_health()
    if h["issues"]:
        raise HTTPException(503, f"OpenIE6 not ready: {h['issues']}")

    normalize = NORMALIZE if body.normalize is None else body.normalize

    inp_path = _write_input_file(body.sentences, WORK_DIR, normalize)
    out_path = inp_path + ".pred"

    cmd = [
        "python", "run.py", "--mode", "splitpredict",
        "--inp", inp_path,
        "--out", out_path,
        "--rescoring", "--task", "oie", "--gpus", str(GPUS),
        "--oie_model", OIE_MODEL,
        "--conj_model", CONJ_MODEL,
        "--rescore_model", RESCORE_DIR,
        "--num_extractions", str(body.num_extractions),
    ]

    start = time.time()
    try:
        cp = subprocess.run(
            cmd, cwd=WORK_DIR, check=True,
            capture_output=True, text=True, timeout=TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        # Best-effort cleanup
        try: os.remove(inp_path)
        except OSError: pass
        raise HTTPException(504, f"OpenIE6 timed out after {TIMEOUT_S}s")
    except subprocess.CalledProcessError as e:
        try: os.remove(inp_path)
        except OSError: pass
        msg = (e.stderr or e.stdout or "")[-800:]
        raise HTTPException(500, f"OpenIE6 failed (exit {e.returncode}): {msg}...")
    except Exception as e:
        try: os.remove(inp_path)
        except OSError: pass
        raise HTTPException(500, f"OpenIE6 invocation error: {e}")

    elapsed = round(time.time() - start, 3)
    triples = _parse_output(out_path)

    # cleanup temp files (ignore errors)
    for p in (inp_path, out_path):
        try: os.remove(p)
        except OSError as oe:
            if oe.errno != errno.ENOENT:
                pass

    return {
        "count": sum(len(x.get("tuples", [])) for x in triples),
        "elapsed_sec": elapsed,
        "results": triples,
    }
