# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC PIP_NO_CACHE_DIR=0 PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates python3 python3-pip python3-venv build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone the official openie6-docker repo (contains run.py)
RUN git clone --depth 1 https://github.com/Qiming-Liu/openie6-docker.git /workspace/openie6-docker

# Install Python deps from the repo if present
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    if [ -f /workspace/openie6-docker/requirements.txt ]; then \
      pip install -r /workspace/openie6-docker/requirements.txt; \
    fi && \
    # lightweight extras for adapter
    pip install fastapi uvicorn[standard] pydantic==2.8.2

ENV OPENIE6_CODE_DIR=/workspace/openie6-docker \
    OPENIE6_MODELS_DIR=/models

CMD ["bash","-lc","ls -la /workspace/openie6-docker && test -f /workspace/openie6-docker/run.py && echo 'run.py OK'"]
