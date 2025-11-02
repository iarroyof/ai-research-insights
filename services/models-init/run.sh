#!/usr/bin/env bash
set -euo pipefail

echo "[models-init] Preparing HuggingFace cache at $HF_HOME"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
mid = os.environ.get("LLM_MODEL_ID","HuggingFaceH4/zephyr-7b-beta")
print(f"[models-init] Downloading LLM model: {mid}")
snapshot_download(repo_id=mid, local_dir=os.environ.get("HF_HOME","/models"), local_dir_use_symlinks=False)
print("[models-init] Done LLM model.")
PY

# Optional OpenIE6 models
need_any=False
for v in OPENIE6_OIE_URL OPENIE6_CONJ_URL OPENIE6_RESCORE_URL; do
  if [ -n "${!v:-}" ]; then need_any=True; fi
done

if [ "$need_any" = True ]; then
  echo "[models-init] Downloading OpenIE6 models to /openie6 ..."
  mkdir -p /openie6/oie_model /openie6/conj_model /openie6/rescore_model

  if [ -n "${OPENIE6_OIE_URL:-}" ]; then
    echo " - oie_model from $OPENIE6_OIE_URL"
    curl -fsSL "$OPENIE6_OIE_URL" -o /openie6/oie_model/epoch=14_eval_acc=0.551_v0.ckpt
  fi
  if [ -n "${OPENIE6_CONJ_URL:-}" ]; then
    echo " - conj_model from $OPENIE6_CONJ_URL"
    curl -fsSL "$OPENIE6_CONJ_URL" -o /openie6/conj_model/epoch=28_eval_acc=0.854.ckpt
  fi
  if [ -n "${OPENIE6_RESCORE_URL:-}" ]; then
    echo " - rescore_model from $OPENIE6_RESCORE_URL"
    mkdir -p /openie6/rescore_model
    # If this is an archive, replace with appropriate extraction.
    curl -fsSL "$OPENIE6_RESCORE_URL" -o /openie6/rescore_model/model.bin || true
  fi
  echo "[models-init] Done OpenIE6 models."
else
  echo "[models-init] Skipping OpenIE6 downloads (no URLs provided)."
fi

echo "[models-init] All done."
