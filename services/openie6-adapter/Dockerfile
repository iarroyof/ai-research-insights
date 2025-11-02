# syntax=docker/dockerfile:1.7
FROM alanpross/openie6:latest

WORKDIR /workspace

# REST shim
COPY services/openie6-adapter/app.py services/openie6-adapter/parser.py /workspace/

# Use the interpreter in the image; upgrade pip first.
# Pin to FastAPI/Pydantic v1-compatible versions to avoid resolver issues on this base image.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install "fastapi==0.103.2" "uvicorn[standard]==0.23.2" "pydantic<2"

ENV OPENIE6_MODELS_DIR=/models \
    OPENIE6_WORK_DIR=/workspace \ 
    OPENIE6_GPUS=0 \
    OPENIE6_TIMEOUT_SEC=600

EXPOSE 9006
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9006"]
