# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ENV TZ=UTC PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=0 HF_HOME=/models
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY services/rebel-extractor/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install --upgrade pip && pip install -r /tmp/requirements.txt
COPY services/rebel-extractor/app.py services/rebel-extractor/parser.py /app/
VOLUME ["/models"]
ENV REBEL_MODEL_ID=Babelscape/rebel-large REBEL_BATCH_SIZE=4 REBEL_MAX_NEW_TOKENS=128
EXPOSE 9007
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","9007"]
